import torch
import torch.nn as nn

from datautils import *
from modelutils import *
from quant import *


def get_flops(layers, model, sample, run):
    flops = {}
    def record_flops(name):
        def tmp(layer, inp, out):
            inp = inp[0]
            if isinstance(layer, nn.Conv2d):
                flops[name] = inp.shape[2] * inp.shape[3]
                flops[name] *= layer.weight.numel()
                stride = list(layer.stride)
                flops[name] //= stride[0] * stride[1] 
            if isinstance(layer, nn.Linear):
                flops[name] = layer.weight.numel()
        return tmp
    handles = []
    for name, layer in layers.items():
        if hasattr(layer, 'module'):
            layer.module.register_forward_hook(record_flops(name))
        else:
            layer.register_forward_hook(record_flops(name))
    with torch.no_grad():
        run(model, sample)
    for h in handles:
        h.remove()
    return flops

def load_errors(sds, path, norm=False):
    errors = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            name = lines[i].strip()
            errors[name] = {}
            i += 1
            for _ in range(len(sds)):
                err, level = lines[i].strip().split(' ')
                errors[name][level] = float(err)
                i += 1
    if norm:
        for name in errors:
            norm = max(errors[name].values())
            if norm > 0:
                for level in errors[name]:
                    errors[name][level] /= norm
    return errors


class SparsityDatabase:

    def __init__(self, sparsetype, model, prefix='', dev=DEV):
        self.sds = {}
        path = os.path.join(prefix, 'models_' + sparsetype)
        for f in os.listdir(path):
            if not (f.startswith(model + '_') and f.endswith('.pth')):
                continue
            sparsity = '0.' + f.split('.')[0].split('_')[1]
            self.sds[sparsity] = torch.load(os.path.join(path, f), map_location=dev)
        self.sparsetype = sparsetype
        self.model = model
        self.prefix = prefix

    def load(self, layers, name, config='', sd=None):
        if not sd:
            sd = self.sds[config]
        if '8w8a' in self.sparsetype:
            layers[name].module.weight.data = sd[name + '.module.weight']
            layers[name].quantizer.maxq.data = sd[name + '.quantizer.maxq']
            layers[name].quantizer.scale.data = sd[name + '.quantizer.scale']
            layers[name].quantizer.zero.data = sd[name + '.quantizer.zero']
        else:
            layers[name].weight.data = sd[name + '.weight']

    def stitch(self, layers, config):
        for name, layer in layers.items():
            self.load(layers, name, config[name])

    def load_errors(self, name):
        path = os.path.join(
            self.prefix, 'scores/%s_%s_%s.txt' % (self.model, self.sparsetype, name)
        )
        return load_errors(self.sds, path, norm=name == 'squared')

    def get_params(self, layers):
        res = {}
        for name in layers:
            res[name] = {}
            for sparsity in self.sds:
                res[name][sparsity] = torch.sum(
                    (self.sds[sparsity][name + '.weight'] != 0).float()
                ).item()
        return res

    def get_flops(self, layers, model, sample, run):
        flops = get_flops(layers, model, sample, run)
        res = {}
        for name in layers:
            res[name] = {}
            for sparsity in self.sds:
                res[name][sparsity] = flops[name] * torch.mean(
                    (self.sds[sparsity][name + '.weight'] != 0).float()
                ).item()
        return res

    def get_timingsq(self):
        timings = {}
        with open('timings/%sq.txt' % self.model, 'r') as f:
            lines = f.readlines()
            baselinetime = float(lines[0])
            i = 1
            while i < len(lines):
                name = lines[i].strip()
                timings[name] = {}
                i += 1
                for _ in range(len(self.sds)):
                    time, level = lines[i].strip().split(' ')
                    timings[name][level] = float(time)
                    i += 1
        return baselinetime, timings


class QuantNMDatabase:

    def __init__(self, model, prefix=''):
        self.sds = {}
        for path in ['models_quant', 'models_nm_quant']:
            for f in os.listdir(os.path.join(prefix, path)):
                if not (f.startswith(model + '_') and f.endswith('.pth')):
                    continue
                config = '_'.join(f.split('.')[0].split('_')[1:])
                self.sds[config] = torch.load(os.path.join(prefix, path, f), map_location=DEV)
        self.model = model
        self.prefix = prefix

    def load(self, layers, name, config='', sd=None):
        if not sd:
            sd = self.sds[config]
        layers[name].module.weight.data = sd[name + '.module.weight']
        layers[name].quantizer.maxq.data = sd[name + '.quantizer.maxq']
        layers[name].quantizer.scale.data = sd[name + '.quantizer.scale']
        layers[name].quantizer.zero.data = sd[name + '.quantizer.zero']

    def stitch(self, layers, config):
        for name, layer in layers.items():
            self.load(layers, name, config[name])

    def load_errors(self, name):
        path = os.path.join(self.prefix, 'scores/%s_mixed_%s.txt' % (self.model, name))
        return load_errors(self.sds, path, norm=name == 'squared')

    def get_bits(self, layers):
        res = {}
        for name, layer in layers.items():
            paramcount = layer.module.weight.numel()
            res[name] = {
                # '24_4w4a': paramcount * 5,
                # '24_8w8a': paramcount * 9, 
                '24_4w4a': paramcount * 4,
                '24_8w8a': paramcount * 8, 
                   '4w4a': paramcount * 4,
                   '8w8a': paramcount * 8
            }
        return res

    def get_bops(self, layers, model, sample, run):
        flops = get_flops(layers, model, sample, run)
        res = {}
        for name, layer in layers.items():
            res[name] = {
                '24_4w4a': flops[name] * 32 // 2 // 8,
                '24_8w8a': flops[name] * 32 // 2 // 4,
                   '4w4a': flops[name] * 32 // 8,
                   '8w8a': flops[name] * 32 // 4 
            }
            if (layers[name].module.weight.numel() // layers[name].module.weight.shape[0]) % 4 != 0:
                res[name]['24_4w4a'] *= 2
                res[name]['24_8w8a'] *= 2
        return res


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str)
    parser.add_argument('dataset', type=str)
    parser.add_argument('database', choices=['mixed', '4block', 'unstr', '4block_8w8a'])
    parser.add_argument('mode', choices=['loss', 'squared', 'spdy', 'stitch', 'eval'])
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--profile', type=str, default='')
    parser.add_argument('--score_path', type=str, default='scores')

    parser.add_argument('--datapath', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=1024)
    parser.add_argument('--batchsize', type=int, default=-1)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--nrounds', type=int, default=-1)

    args = parser.parse_args()

    get_model, test, run = get_functions(args.model)
    dataloader, testloader = get_loaders(
        args.dataset, path=args.datapath,
        batchsize=args.batchsize, workers=args.workers,
        nsamples=args.nsamples, seed=args.seed,
        noaug=args.mode == 'loss'
    )
    if args.nrounds == -1:
        args.nrounds = 1 if 'yolo' in args.model or 'bert' in args.model else 10
        if args.mode == 'loss':
            args.nrounds = 1

    filepath = os.path.join(args.prefix, args.score_path, '%s_%s_%s.txt' % (args.model, args.database, args.mode))

    modelp = get_model()
    if args.database == 'mixed':
        db = QuantNMDatabase(args.model, prefix=args.prefix)
    if args.database in ['4block', 'unstr', '4block_8w8a']:
        db = SparsityDatabase(args.database, args.model, prefix=args.prefix)
    if args.database in ['mixed', '4block_8w8a']:
        add_actquant(modelp)
    layersp = find_layers(modelp)

    for i in range(layersp['fc'].weight.shape[0]):
        print(i)
        W = layersp['fc'].weight.data
        thresh = torch.sort(torch.abs(W[i, :]), descending=True)[0][9]
        W[i, torch.abs(W[i, :]) < thresh] = 0
        print(torch.mean((W[i, :] == 0).float()))
    test(modelp, testloader)
    exit()

    config = {n: '0.0000' for n in layersp}
    config['fc'] = '0.9797' # '0.9900'
    db.stitch(layersp, config)
    with torch.no_grad():
        print(run(modelp, next(iter(dataloader)), loss=True) / args.nsamples)
    test(modelp, testloader)
    exit()

    if args.mode == 'stitch':
        with open(args.profile, 'r') as f:
            config = {}
            for l in f.readlines():
                level, name = l.strip().split(' ')
                config[name] = '24_8w8a' # level
            db.stitch(layersp, config)
            test(modelp, testloader)
        exit()

    if args.mode == 'eval':
        for s in sorted(db.sds):
            db.stitch(layersp, {n: s for n in layersp})
            print(s)
            test(modelp, testloader)
        exit()

    if args.mode == 'spdy':
        layersp = find_layers(modelp)
        tmp = (np.arange(len(db.sds)) / (len(db.sds) - 1)) ** 2
        print(len(db.sds))
        print(len(tmp))
        with open(filepath, 'w') as f:
            for layer in layersp:
                print(layer)
                f.write(layer + '\n')
                for i, name in enumerate(sorted(db.sds)):
                    f.write('%.6f %s\n' % (tmp[i], name))
        exit()

    if args.mode == 'squared':
        modeld = get_model()
        layersd = find_layers(modeld)

        errs = {n: {} for n in layersp}
        def accumerrs(name):
            def tmp(layer, inp, out):
                errs[name]['dense'] = errs[name].get('dense', 0) + torch.sum(out.data ** 2).item()
                for config in sorted(db.sds):
                    db.load(layersp, name, config)
                    errs[name][config] = errs[name].get(config, 0) + torch.sum((layersp[name](inp[0].data) - out.data) ** 2).item()
            return tmp
        for name in layersd:
            layersd[name].register_forward_hook(accumerrs(name))

        with torch.no_grad():
            for _ in range(args.nrounds):
                for i, batch in enumerate(dataloader):
                    print(i)
                    run(modeld, batch)

        with open(filepath, 'w') as f:
            for name in errs:
                f.write(name + '\n') 
                for config in sorted(errs[name]):
                    if config != 'dense':
                        f.write('%.6f %s\n' % (errs[name][config] / errs[name]['dense'], config))
        exit()

    if args.mode == 'loss':
        sd = modelp.state_dict()
        errs = {n: {} for n in layersp}
        baseloss = 0

        for _ in range(args.nrounds):
            for i, batch in enumerate(dataloader):
                print(i)
                with torch.no_grad():
                    baseloss += run(modelp, batch, loss=True)
                    for name in layersp:
                        print(name)
                        for config in sorted(db.sds):
                            db.load(layersp, name, config)
                            errs[name][config] = errs[name].get(config, 0) + run(modelp, batch, loss=True)
                        db.load(layersp, name, sd=sd)
        baseloss /= len(dataloader) * args.nrounds
        for name in errs:
            for config in errs[name]:
                errs[name][config] /= len(dataloader) * args.nrounds

        with open(filepath, 'w') as f:
            for name in errs:
                f.write(name + '\n') 
                for config in sorted(errs[name]):
                    f.write('%+.6f %s\n' % (errs[name][config] - baseloss, config))
        exit()
