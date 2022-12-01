import argparse

import torch
import torch.nn as nn

from datautils import *
from database import *
from modelutils import *
from quant import *


parser = argparse.ArgumentParser()

parser.add_argument('model', type=str)
parser.add_argument('dataset', type=str)
parser.add_argument('load', type=str)
parser.add_argument('--database', choices=['', 'mixed', '4block', 'unstr', '4block_8w8a'], default='')
parser.add_argument('--prefix', type=str, default='')

parser.add_argument('--datapath', type=str, default='')
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--nsamples', type=int, default=1024)
parser.add_argument('--batchsize', type=int, default=-1)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--nrounds', type=int, default=-1)
parser.add_argument('--noaug', action='store_true')

parser.add_argument('--skip-firstlast', action='store_true')

parser.add_argument('--bnt', action='store_true')
parser.add_argument('--bnt-batches', type=int, default=100)
parser.add_argument('--lintune', action='store_true')
parser.add_argument('--lintune-loss', action='store_true')
parser.add_argument('--lintune-epochs', type=int, default=100)
parser.add_argument('--lintune-lr', type=float, default=1e-4)
parser.add_argument('--gap', action='store_true')
parser.add_argument('--gap-epochs', type=int, default=100)
parser.add_argument('--gap-lr', type=float, default=1e-5)
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--finetune-mse', action='store_true')
parser.add_argument('--finetune-epochs', type=int, default=2)
parser.add_argument('--finetune-lr', type=float, default=1e-5)
parser.add_argument('--statcorr', action='store_true')
parser.add_argument('--statcorr-samples', type=int, default=-1)
parser.add_argument('--save', type=str)


args = parser.parse_args()

dataloader, testloader = get_loaders(
    args.dataset, path=args.datapath,
    batchsize=args.batchsize, workers=args.workers,
    nsamples=args.nsamples, seed=args.seed,
    noaug=args.noaug
)
get_model, test, run = get_functions(args.model)

modelp = get_model()
if args.load.endswith('.pth'):
    tmp = torch.load(args.load)
    if any('scale' in k for k in tmp):
        add_actquant(modelp)
    if args.skip_firstlast:
        for l in firstlast_names(args.model):
            if any('scale' in k for k in tmp):
                tmp[l + '.quantizer.scale'][:] = 0
                l += '.module'
            l += '.weight'
            tmp[l] = modelp.state_dict()[l]
    modelp.load_state_dict(tmp)
modelp = modelp.to(DEV)

if args.database != '':
    if args.database == 'mixed':
        print('Stitching ...')
        db = QuantNMDatabase(args.model, prefix=args.prefix)
    if args.database in ['4block', 'unstr', '4block_8w8a']:
        db = SparsityDatabase(args.database, args.model, prefix=args.prefix, dev='cpu')
    if args.database in ['mixed', '4block_8w8a']:
        add_actquant(modelp)
    modelp = modelp.to('cpu')
    layersp = find_layers(modelp)
    with open(args.load, 'r') as f:
        config = {}
        for l in f.readlines():
            level, name = l.strip().split(' ')
            config[name] = level 
    db.stitch(layersp, config)
    modelp = modelp.to(DEV)
    layersp = find_layers(modelp)
    if args.save:
        torch.save(modelp.state_dict(), args.save)
        exit()


if args.bnt:
    print('Batchnorm tuning ...')

    loss = 0
    for batch in dataloader:
        loss += run(modelp, batch, loss=True)
    print(loss / args.nsamples)

    batchnorms = find_layers(modelp, [nn.BatchNorm2d])
    for bn in batchnorms.values():
        bn.reset_running_stats()
        bn.momentum = .1
    modelp.train()
    with torch.no_grad():
        i = 0
        while i < args.bnt_batches:
            for batch in dataloader:
                if i == args.bnt_batches:
                    break
                print('%03d' % i)
                run(modelp, batch)
                i += 1
    modelp.eval()

    loss = 0
    for batch in dataloader:
        loss += run(modelp, batch, loss=True)
    print(loss / args.nsamples)

if args.lintune:
    print('Linear tuning ...')
    modeld = get_model()
    params = []
    for n, p in modelp.named_parameters():
        if len(p.shape) == 1:
            params.append(p)
        else:
            p.requires_grad = False
    optim = torch.optim.Adam(params, lr=args.lintune_lr)
    criterion = nn.MSELoss()
    for i in range(args.lintune_epochs):
        cumloss = 0
        for batch in dataloader:
            if args.lintune_loss:
                loss = run(modelp, batch, loss=True)
            else:
                with torch.no_grad():
                    y = run(modeld, batch)
                loss = criterion(run(modelp, batch), y)
            cumloss += loss.item()
            loss.backward()
            optim.step()
            optim.zero_grad()
        print('%02d %.4f' % (i, cumloss / len(dataloader)))

if args.gap:
    modeld = get_model()
    layersp = find_layers(modelp) 
    layersd = find_layers(modeld)

    masks = {n: l.weight.data == 0 for n, l in layersp.items()}

    def cache_output(name, outputs):
        def tmp(layer, inp, out):
            outputs[name] = out
        return tmp
    outputsp = {}
    handlesp = []
    for name in layersp:
        handlesp.append(
            layersp[name].register_forward_hook(cache_output(name, outputsp))
        )
    outputsd = {}
    handlesd = [] 
    for name in layersd:
        handlesd.append(
            layersd[name].register_forward_hook(cache_output(name, outputsd))
        )

    criterion = nn.MSELoss(reduction='sum')
    optim = torch.optim.Adam(modelp.parameters(), lr=args.gap_lr)

    for i in range(args.gap_epochs):
        cumloss = 0
        for batch in dataloader:
            with torch.no_grad():
                run(modeld, batch) 
            run(modelp, batch)
            loss = 0
            for name in outputsd:
                norm = torch.norm(outputsd[name].data).item() ** 2
                loss += criterion(outputsp[name], outputsd[name].data) / norm
            cumloss += loss.item()
            loss.backward()
            optim.step()
            optim.zero_grad()
            for name, mask in masks.items():
                layersp[name].weight.data[mask] = 0
        print('%05d: %.6f' % (i, cumloss / len(dataloader)))

    for h in handlesp:
        h.remove()
    for h in handlesd:
        h.remove()

if args.finetune:
    print('Finetuning ...')
    modeld = get_model()
    masks = {n: p == 0 for n, p in modelp.named_parameters()}
    optim = torch.optim.Adam(modelp.parameters(), lr=args.finetune_lr)
    criterion = nn.MSELoss()
    for i in range(args.finetune_epochs):
        cumloss = 0
        for batch in dataloader:
            if args.finetune_mse:
                with torch.no_grad():
                    y = run(modeld, batch)
                loss = criterion(run(modelp, batch), y)
            else:
                loss = run(modelp, batch, loss=True)
            cumloss += loss.item()
            loss.backward()
            optim.step()
            optim.zero_grad()
            for n, p in modelp.named_parameters():
                p.data[masks[n]] = 0
        print('%02d %.4f' % (i, cumloss / len(dataloader)))

if args.statcorr:
    print('Stat correction ...')

    if args.statcorr_samples == -1:
        args.statcorr_samples = args.nsamples
    trainloader, testloader = get_loaders(
        args.dataset, batchsize=args.statcorr_samples, noaug=True
    )
    batch = next(iter(trainloader))

    modeld = get_model()
    layersd = find_layers(modeld, layers=[nn.BatchNorm2d, nn.LayerNorm])
    layersp = find_layers(modelp, layers=[nn.BatchNorm2d, nn.LayerNorm])

    REDUCE = {
        2: [0],
        3: [0, 1],
        4: [0, 2, 3]
    }

    meansd = {}
    stdsd = {}
    def hookd(name):
        def tmp(layer, inp, out):
            red = REDUCE[len(out.shape)]
            meansd[name] = torch.mean(out.data, red, keepdim=True)
            stdsd[name] = torch.std(out.data, red, keepdim=True)
        return tmp
    meansp = {}
    stdsp = {}
    def hookp(name):
        def tmp(layer, inp, out):
            red = REDUCE[len(out.shape)]
            meansp[name] = torch.mean(out.data, red, keepdim=True)
            stdsp[name] = torch.std(out.data, red, keepdim=True)
            out.data -= meansp[name]
            out.data *= stdsd[name] / (stdsp[name] + 1e-9)
            out.data += meansd[name]
        return tmp
    handles = []
    for name in layersd:
        handles.append(layersd[name].register_forward_hook(hookd(name)))
    with torch.no_grad():
        run(modeld, batch)
    for h in handles:
        h.remove()
    handles = []
    for name in layersp:
        handles.append(layersp[name].register_forward_hook(hookp(name)))
    with torch.no_grad():
        run(modelp, batch)
    for h in handles:
        h.remove()

    def hook(name):
        def tmp(layer, inp, out):
            out.data -= meansp[name]
            out.data *= stdsd[name] / (stdsp[name] + 1e-9)
            out.data += meansd[name]
        return tmp
    for name in layersp:
        layersp[name].register_forward_hook(hook(name))

test(modelp, testloader)
