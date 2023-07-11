import math
import time

import torch
import torch.nn as nn

from quant import *


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEBUG = False 


class TrueOBS:

    def __init__(self, layer, rel_damp=0):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        # Accumulate in double precision
        self.H = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.double)
        self.nsamples = 0
        self.rel_damp = rel_damp

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        self.H += 2 / self.nsamples * (inp.matmul(inp.t())).double()

    def invert(self, H):
        try:
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        except RuntimeError:
            print('Hessian not full rank.')
            tmp = 1 * torch.eye(self.columns, device=self.dev)
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H + tmp))
        return Hinv

    def prepare(self, columnslast=False):
        if columnslast: 
            perm = torch.arange(self.columns, device=self.dev)
            if len(self.layer.weight.shape) == 4:
                perm = perm.reshape(list(self.layer.weight.shape)[1:])
                perm = perm.permute([1, 2, 0])
                perm = perm.flatten()
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        H = self.H.float()
        if self.rel_damp > 0:
            damp = self.rel_damp * torch.diag(H).mean()
            H += damp * torch.eye(H.shape[0], device=self.dev)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        if columnslast:
            H = H[perm, :][:, perm]
            W = W[:, perm]
        Hinv = self.invert(H)
        Losses = torch.zeros([self.rows, self.columns + 1], device=self.dev)
        if columnslast:
            return W, H, Hinv, Losses, perm
        return W, H, Hinv, Losses

    def prepare_iter(self, i1, parallel, W, Hinv1):
        i2 = min(i1 + parallel, self.rows)
        count = i2 - i1
        w = W[i1:i2, :]
        Hinv = Hinv1.unsqueeze(0).repeat((count, 1, 1))
        mask = torch.zeros_like(w).bool()
        rangecount = torch.arange(count, device=self.dev)
        idxcount = rangecount + i1
        return i2, count, w, Hinv, mask, rangecount, idxcount

    def prepare_sparse(self, w, mask, Hinv, H): 
        start = int(torch.min(torch.sum((w == 0).float(), 1)).item()) + 1
        for i in range(w.shape[0]):
            tmp = w[i] == 0
            H1 = H.clone()
            H1[tmp, :] = 0
            H1[:, tmp] = 0
            H1[tmp, tmp] = 1
            Hinv[i] = self.invert(H1)
            mask[i, torch.nonzero(tmp, as_tuple=True)[0][:(start - 1)]] = True
        return start

    def quantize(self, parallel=32):
        W, H, Hinv1, Losses = self.prepare()

        Q = torch.zeros_like(W)
        self.quantizer.find_params(W, weight=True)

        for i1 in range(0, self.rows, parallel):
            i2, count, w, Hinv, mask, rangecount, idxcount = self.prepare_iter(i1, parallel, W, Hinv1)
            start = self.prepare_sparse(w, mask, Hinv, H)

            outlier = .25 * (self.quantizer.scale ** 2)[i1:i2, :]
            scale = self.quantizer.scale[i1:i2, :]
            zero = self.quantizer.zero[i1:i2, :]

            tick = time.time()

            for quant in range(start, self.columns + 1):
                q = quantize(w, scale, zero, self.quantizer.maxq)
                err = (w - q) ** 2
                diag = torch.diagonal(Hinv, dim1=1, dim2=2)
                scores = err / diag
                scores[mask] = float('inf')
                err[mask] = 0
                j = torch.argmin(scores, 1)
                sel = torch.any(err > outlier, 1)
                sel &= w[rangecount, j] != 0
                if torch.any(sel):
                    j[sel] = torch.argmax(err[sel, :], 1)
                Losses[i1:i2, quant] = scores[rangecount, j]
                q1 = q[rangecount, j]
                Q[idxcount, j] = q1
                row = Hinv[rangecount, j, :]
                d = diag[rangecount, j]
                w -= row * ((w[rangecount, j] - q1) / d).unsqueeze(1)
                mask[rangecount, j] = True
                if quant == self.columns:
                    break
                row /= torch.sqrt(d).unsqueeze(1)
                Hinv -= torch.bmm(row.unsqueeze(2), row.unsqueeze(1))
            Losses[i1:i2, :] /= 2

            torch.cuda.synchronize()
            print('%04d %04d time %.2f' % (i1, i2, time.time() - tick))

        print('error', torch.sum(Losses).item())
        self.layer.weight.data = Q.reshape(self.layer.weight.shape)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2) / 128)

    def nmprune(self, n=2, m=4, parallel=32):
        W, H, Hinv1, Losses, perm = self.prepare(columnslast=True)

        for i1 in range(0, self.rows, parallel):
            i2, count, w, Hinv, mask, rangecount, idxcount = self.prepare_iter(i1, parallel, W, Hinv1)

            buckets = torch.zeros((count, self.columns // m, 1), device=self.dev)

            tick = time.time()

            for zeros in range(1, self.columns + 1):
                diag = torch.diagonal(Hinv, dim1=1, dim2=2)
                scores = w ** 2 / diag 
                tmp = (buckets >= n).repeat((1, 1, m)).flatten(1)
                scores[mask | tmp] = float('inf')
                j = torch.argmin(scores, 1)
                Losses[i1:i2, zeros] = scores[rangecount, j]
                row = Hinv[rangecount, j, :]
                d = diag[rangecount, j]
                w -= row * (w[rangecount, j] / d).unsqueeze(1)
                mask[rangecount, j] = True
                buckets[rangecount, torch.div(j, m, rounding_mode='floor'), :] += 1
                if zeros == self.columns * n / m:
                    break
                row /= torch.sqrt(d).unsqueeze(1)
                Hinv -= torch.bmm(row.unsqueeze(2), row.unsqueeze(1))
            Losses[i1:i2, :] /= 2
            w[mask] = 0
            W[i1:i2, :] = w

            torch.cuda.synchronize()
            print('%04d %04d time %.2f' % (i1, i2, time.time() - tick))

        print('error', torch.sum(Losses).item())
        W = W[:, torch.argsort(perm)]
        self.layer.weight.data = W.reshape(self.layer.weight.shape)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2) / 128)

    def prepare_unstr(self, parallel=32):
        W, H, Hinv1, Losses = self.prepare()

        self.Losses = Losses
        self.Traces = []

        for i1 in range(0, self.rows, parallel):
            i2, count, w, Hinv, mask, rangecount, idxcount = self.prepare_iter(i1, parallel, W, Hinv1)
            start = self.prepare_sparse(w, mask, Hinv, H) 

            Trace = torch.zeros((self.columns + 1, count, self.columns), device=self.dev)
            Trace[0, :, :] = w
            Trace[:start, :, :] = w

            tick = time.time()

            for zeros in range(start, self.columns + 1):
                diag = torch.diagonal(Hinv, dim1=1, dim2=2)
                scores = (w ** 2) / diag
                scores[mask] = float('inf')
                j = torch.argmin(scores, 1)
                self.Losses[i1:i2, zeros] = scores[rangecount, j]
                row = Hinv[rangecount, j, :]
                d = diag[rangecount, j]
                w -= row * (w[rangecount, j] / d).unsqueeze(1)
                mask[rangecount, j] = True
                w[mask] = 0
                Trace[zeros, :, :] = w
                if zeros == self.columns:
                    break
                row /= torch.sqrt(d).unsqueeze(1)
                Hinv -= torch.bmm(row.unsqueeze(2), row.unsqueeze(1))
            self.Losses[i1:i2, :] /= 2
            self.Traces.append(Trace.cpu())

            torch.cuda.synchronize()
            print('%04d %04d time %.2f' % (i1, i2, time.time() - tick))

    def prune_unstr(self, sparsities):
        return self.prune_blocked(sparsities)

    def prepare_blocked(self, size=4, parallel=32):
        W, H, Hinv1, Losses, perm = self.prepare(columnslast=True)

        self.Traces = []
        blockcount = self.columns // size
        self.Losses = torch.zeros((self.rows, blockcount + 1), device=self.dev)
        rangeblockcount = torch.arange(blockcount, device=self.dev)
        rangecolumns = torch.arange(self.columns, device=self.dev)

        for i1 in range(0, self.rows, parallel):
            i2, count, w, Hinv, _, rangecount, _ = self.prepare_iter(i1, parallel, W, Hinv1)

            mask = torch.zeros((count, blockcount), device=self.dev).bool()
            mask1 = torch.zeros((count, blockcount, size), device=self.dev).bool()
            Trace = torch.zeros((blockcount + 1, count, self.columns), device=self.dev)
            Trace[0, :, :] = w
            rangeblockunroll = torch.arange(count * blockcount, device=self.dev)
            blockdiagidx = rangeblockcount.repeat(count)
            rangeunroll = torch.arange(self.columns * count, device=self.dev)
            diagidx = rangecolumns.repeat(count)
            paroffset = blockcount * rangecount
            expandrows = torch.arange(size, device=self.dev).unsqueeze(0).repeat(count, 1)
            expandrows += self.columns * rangecount.unsqueeze(1) 

            tick = time.time()

            for dropped in range(1, blockcount + 1):
                blocks = Hinv.reshape(count * blockcount, size, blockcount, size)
                blocks = blocks[rangeblockunroll, :, blockdiagidx, :]
                invblocks = torch.cholesky_inverse(torch.linalg.cholesky(blocks))
                w1 = w.reshape((count * blockcount, 1, size))
                lambd = torch.bmm(w1, invblocks)
                scores = torch.sum(lambd * w1, (1, 2))
                scores = scores.reshape((count, blockcount))
                scores[mask] = float('inf')
                j = torch.argmin(scores, 1)
                self.Losses[i1:i2, dropped] = scores[rangecount, j]

                tmp = (expandrows + size * j.unsqueeze(1)).flatten()
                rows = Hinv.reshape((-1, self.columns))[tmp]
                rows = rows.reshape((count, size, self.columns))
                tmp = paroffset + j
                d = invblocks[tmp]

                w -= torch.bmm(lambd[tmp], rows).squeeze(1)
                mask[rangecount, j] = True
                mask1[mask] = True
                tmp = mask1.flatten(1)
                w[mask1.flatten(1)] = 0
                Trace[dropped, :, :] = w

                if dropped == self.columns:
                    break
                Hinv -= torch.bmm(rows.transpose(1, 2), torch.bmm(d, rows))
                Hinv = Hinv.reshape((count * self.columns, self.columns))
                tmp = mask1.flatten()
                Hinv[rangeunroll[tmp], diagidx[tmp]] = 1
                Hinv = Hinv.reshape((count, self.columns, self.columns))
            self.Losses[i1:i2, :] /= 2
            Trace = Trace[:, :, torch.argsort(perm)]
            self.Traces.append(Trace.cpu())

            torch.cuda.synchronize()
            print('%04d %04d time %.2f' % (i1, i2, time.time() - tick))

    def prune_blocked(self, sparsities):
        parallel = self.Traces[0].shape[1]
        blockcount = self.Traces[0].shape[0] - 1
        losses = self.Losses[:, 1:].reshape(-1)
        order = torch.argsort(losses)
        Ws = [torch.zeros((self.rows, self.columns), device=self.dev) for _ in sparsities]
        losses = [0] * len(sparsities) 
        for i in range(self.rows):
            if i % parallel == 0:
                Trace = self.Traces[i // parallel].to(self.dev)
            for j, sparsity in enumerate(sparsities):
                count = int(math.ceil(self.rows * blockcount * sparsity))
                perrow = torch.sum(
                    torch.div(order[:count], blockcount, rounding_mode='trunc') == i
                ).item()
                losses[j] += torch.sum(self.Losses[i, :(perrow + 1)]).item()
                Ws[j][i, :] = Trace[perrow, i % parallel, :]
        for sparsity, loss in zip(sparsities, losses):
            print('%.4f error' % sparsity, loss)
            if DEBUG:
                tmp = self.layer.weight.data.clone()
                self.layer.weight.data = Ws[sparsities.index(sparsity)].reshape(self.layer.weight.shape) 
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2) / 128)
                self.layer.weight.data = tmp
        return Ws

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
