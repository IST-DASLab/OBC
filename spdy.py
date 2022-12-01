# Adapted from code provided by the authors of SPDY [9]

import argparse
import math
import os
import random

import numpy as np
import torch

from database import *
from datautils import *
from modelutils import *
from quant import *


parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('dataset')
parser.add_argument('target', type=float)
parser.add_argument('database', choices=['mixed', '4block', 'unstr', '4block_8w8a'])
parser.add_argument('--errors', choices=['', 'squared', 'loss'], default='')
parser.add_argument('--constr', choices=['', 'bits', 'bops', 'flops', 'timingsq'], default='')
parser.add_argument('--nobatchnorm', action='store_true')
parser.add_argument('--statcorr', action='store_true')
parser.add_argument('--dpbuckets', type=int, default=10000)
parser.add_argument('--dp', action='store_true')
parser.add_argument('--score', type=str, default='')

parser.add_argument('--prefix', type=str, default='')
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
    noaug=True
)

modelp = get_model()
if args.database in ['mixed', '4block_8w8a']:
    add_actquant(modelp)
layersp = find_layers(modelp)

batches = []
for batch in dataloader:
    batches.append(run(modelp, batch, retmoved=True))

if args.database == 'mixed':
    db = QuantNMDatabase(args.model, prefix=args.prefix)
if args.database in ['4block', 'unstr', '4block_8w8a']:
    db = SparsityDatabase(args.database, args.model, prefix=args.prefix)

DEFAULT_CONSTR = {
    'mixed': 'bops',
    '4block': 'flops',
    'unstr': 'flops',
    '4block_8w8a': 'timingsq'
}
if not args.constr:
    args.constr = DEFAULT_CONSTR[args.database]
if not args.errors:
    args.errors = 'loss' if args.dp else 'spdy'

errors = db.load_errors(args.errors)
baseline_constr = None
if args.constr == 'bits':
    constr = db.get_bits(layersp)
if args.constr == 'bops':
    constr = db.get_bops(layersp, modelp, batches[0], run)
if args.constr == 'flops':
    constr = db.get_flops(layersp, modelp, batches[0], run)
if args.constr == 'timingsq':
    baseline_constr, constr = db.get_timingsq()


modelp.train()
if args.nobatchnorm or args.statcorr:
    batchnorms = find_layers(modelp, [nn.BatchNorm2d])
    for bn in batchnorms.values():
        bn.eval()
if args.statcorr:
    batch = batches[0] 
    batches = [batch] 
    args.nsamples = args.batchsize

    modeld = get_model()
    layersd = find_layers(modeld, layers=[nn.BatchNorm2d, nn.LayerNorm])
    layersp1 = find_layers(modelp, layers=[nn.BatchNorm2d, nn.LayerNorm])

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
    def hookp(name):
        def tmp(layer, inp, out):
            red = REDUCE[len(out.shape)]
            meanp = torch.mean(out.data, red, keepdim=True)
            stdp = torch.std(out.data, red, keepdim=True)
            out.data -= meanp
            out.data *= stdsd[name] / (stdp + 1e-9)
            out.data += meansd[name]
        return tmp
    for name in layersd:
        layersd[name].register_forward_hook(hookd(name))
    with torch.no_grad():
        run(modeld, batch)
    for name in layersp1:
        layersp1[name].register_forward_hook(hookp(name))


layers = list(layersp.keys())
sparsities = list(errors[layers[0]].keys())
costs = [[errors[l][s] for s in sparsities] for l in layers] 
timings = [[constr[l][s] for s in sparsities] for l in layers]
costs = np.array(costs)

prunabletime = sum(max(c) for c in timings)
if baseline_constr is None:
    baseline_constr = prunabletime
target_constr = baseline_constr / args.target - (baseline_constr - prunabletime)
best = sum(min(c) for c in timings)
print('Max target:', baseline_constr / (best + baseline_constr - prunabletime))
bucketsize = target_constr / args.dpbuckets

for row in timings:
    for i in range(len(row)):
        row[i] = int(round(row[i] / bucketsize))

def dp(costs):
    DP = np.full((len(layers), args.dpbuckets + 1), float('inf'))
    PD = np.full((len(layers), args.dpbuckets + 1), -1)

    for sparsity in range(len(sparsities)):
        if costs[0][sparsity] < DP[0][timings[0][sparsity]]:
            DP[0][timings[0][sparsity]] = costs[0][sparsity]
            PD[0][timings[0][sparsity]] = sparsity
    for layer in range(1, len(DP)):
        for sparsity in range(len(sparsities)):
            timing = timings[layer][sparsity]
            if timing == 0 and layer == len(DP) - 1:
                DP[layer] = DP[layer - 1]
                PD[layer] = 0
                continue
            if timing == 0 and layer == len(DP) - 1:
                DP[layer] = DP[layer - 1]
                PD[layer] = 0
                continue
            if timing < 1 or timing > args.dpbuckets:
                continue
            score = costs[layer][sparsity]
            tmp = DP[layer - 1][:-timing] + score
            better = tmp < DP[layer][timing:]
            if np.sum(better):
                DP[layer][timing:][better] = tmp[better]
                PD[layer][timing:][better] = sparsity

    score = np.min(DP[-1, :])
    timing = np.argmin(DP[-1, :])
    
    solution = []
    for layer in range(len(DP) - 1, -1, -1):
        solution.append(PD[layer][timing])
        timing -= timings[layer][solution[-1]]
    solution.reverse()
    return solution

def gen_costs(coefs):
    return costs * coefs.reshape((-1, 1))

def stitch_model(solution):
    config = {n: sparsities[s] for n, s in zip(layers, solution)}
    db.stitch(layersp, config)
    return modelp

@torch.no_grad()
def get_loss(model):
    loss = 0
    for batch in batches:
        loss += run(modelp, batch, loss=True)
    return loss / args.nsamples 

def get_score(coefs):
    costs = gen_costs(coefs)
    solution = dp(costs)
    model = stitch_model(solution)
    return get_loss(model)


if args.score:
    with open(args.score, 'r') as f:
        solution = []
        for l in f.readlines():
            splits = l.split(' ')
            sparsity = splits[0]
            name = splits[1][:-1]
            i = sparsities.index(sparsity) 
            solution.append(i)
    print(baseline_constr / (baseline_constr - prunabletime + sum(t[s] for s, t in zip(solution, timings)) * bucketsize))
    print(get_loss(stitch_model(solution)))
    exit()

def save_profile(coefs, name=''):
    solution = dp(gen_costs(coefs))
    if name:
        with open(name, 'w') as f:
            for s, n in zip(solution, layers):
                f.write('%s %s\n' % (sparsities[s], n))
    else:
        for s, n in zip(solution, layers):
            print('%s %s' % (sparsities[s], n))

print('Base:', get_loss(modelp))

name = '%s_%s_%dx_spdy' % (args.model, args.database, int(args.target * 100))
name = os.path.join(args.prefix, name)

if args.dp:
    name = name.replace('spdy', 'dp')
    coefs = np.ones(len(layers))
    print(get_score(np.ones(len(layers))))
    save_profile(coefs)
    save_profile(coefs, name + '.txt')
    exit()

evals = 0
print('Finding init ...')
coefs = None
score = float('inf')
for _ in range(100):
    coefs1 = np.random.uniform(0, 1, size=len(layers))
    score1 = get_score(coefs1)
    evals += 1
    print(evals)
    if score1 < score:
        print(score1)
        score = score1
        coefs = coefs1
print('Running local search ...')
for resamplings in range(int(.1 * len(layers)), 0, -1):
    print('Trying %d resamplings ...' % resamplings)
    improved = True
    while improved: 
        improved = False
        for _ in range(100):
            coefs1 = coefs.copy()
            for _ in range(resamplings):
                coefs1[random.randint(0, len(layers) - 1)] = np.random.uniform(0, 1)
            score1 = get_score(coefs1)
            evals += 1
            print(evals)
            if score1 < score:
                print(score1)
                score = score1
                coefs = coefs1
                improved = True
                break
            
print(coefs)
save_profile(coefs)
save_profile(coefs, name + '.txt')
