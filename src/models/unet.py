import math
from argparse import ArgumentParser

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .common import BasicBlock, DownBlock, UpBlock, convnd


class UNet(nn.Module):

    def __init__(self, layers, k1=7, k=3, cin=3, cout=3, nk=64, skip=True,
                 act='prelu', down='max', up='nearest', nrm='identity', G=2,
                 bias=False, normalize=False, **kwargs):
        super(UNet, self).__init__()

        self.layers = layers
        self.k = k
        self.G = G
        self.act = act
        self.skip = skip
        self.down = down
        self.up = up
        self.nrm = nrm
        self.bias = bias
        self.normalize = normalize

        chs = [(nk, nk)]
        chs += [(round(G**(i)*nk), round(G**(i+1)*nk)) for i in range(len(layers)-1)]

        self.conv1 = convnd((cin, nk), k=k1, bias=bias)
        self.enc = self._make_enc(layers, chs)
        self.dec = self._make_dec(layers, chs)
        self.conv2 = convnd((nk, cout), k=1, bias=bias)

        # init
        mo = 'fan_out'
        nl = 'leaky_relu' if act == 'prelu' else 'relu'

        nn.init.kaiming_uniform_(self.conv1.weight, mode=mo, nonlinearity='conv2d')
        nn.init.kaiming_uniform_(self.conv2.weight, mode=mo, nonlinearity=nl)
        with th.no_grad():
            self.conv2.weight /= np.sqrt(np.sum(layers + layers[:-1]))

    def _make_enc(self, layers, chs):
        enc = nn.ModuleList()
        for i in range(len(layers)):
            layer = []
            c1, c2 = chs[i]

            if i > 0:
                d = DownBlock(self.down, ch=(c1, c1), bias=self.bias)
                layer.append(d)

            for j in range(layers[i]):
                c = (c1, c2) if j == 0 else (c2, c2)
                l = BasicBlock(c, alpha=self.G, k=self.k, act=self.act,
                               nrm=self.nrm, bias=self.bias, skip=self.skip,
                               depth=i, L=layers[i])
                layer.append(l)

            enc.append(nn.Sequential(*layer))

        return enc

    def _make_dec(self, layers, chs):
        dec = nn.ModuleList()

        chs = chs[::-1]
        layers = layers[-2::-1]

        c2 = chs[0][1]
        u = UpBlock(self.up, ch=(c2, c2), bias=self.bias)
        dec.append(u)

        for i in range(len(layers)):
            layer = []
            c2_ = c2
            c2 = chs[i+1][1]
            for j in range(layers[i]):
                c = (c2 + c2_, c2) if j == 0 else (c2, c2)
                l = BasicBlock(c, k=self.k, alpha=self.G, act=self.act,
                               nrm=self.nrm, bias=self.bias, skip=self.skip,
                               depth=i, L=layers[i])
                layer.append(l)

            if i < len(layers)-1:
                u = UpBlock(self.up, ch=(c2, c2), bias=self.bias)
                layer.append(u)
            dec.append(nn.Sequential(*layer))

        return dec

    def pad(self, x):
        N = len(self.layers)-1
        b, c, h, w = x.shape

        w1 = ((w - 1) | (2**N - 1)) + 1
        h1 = ((h - 1) | (2**N - 1)) + 1

        dw = (w1 - w) / 2
        dh = (h1 - h) / 2

        if dw == 0 and dh == 0:
            p = (0, 0, 0, 0)
        else:
            p = math.floor(dw), math.ceil(dw)
            p += math.floor(dh), math.ceil(dh)
            x = F.pad(x, p)

        return x, p

    def unpad(self, x, p):
        if p != (0, 0, 0, 0):
            b, c, h, w = x.shape
            x = x[..., p[2]:h-p[3], p[0]:w-p[1]]
        return x

    def norm(self, x):
        if not self.normalize:
            return x, 0, 1
        else:
            b, c, h, w = x.shape
            mu = x.view(b, c, h * w).mean(dim=2).view(b, c, 1, 1)
            sigma = x.view(b, c, h * w).std(dim=2).view(b, c, 1, 1)
            return (x - mu) / sigma, mu, sigma

    def unnorm(self, x, mu, sigma):
        if not self.normalize:
            return x
        else:
            return x * sigma + mu

    def forward(self, x):
        identity = x

        x, pads = self.pad(x)
        x, mus, sigmas = self.norm(x)

        u = self.conv1(x)
        u = self.enc[0](u)

        es = []
        for i in range(len(self.enc)-1):
            es.append(u)
            u = self.enc[i+1](u)

        u = self.dec[0](u)
        for i in range(len(self.dec)-1):
            u = th.cat((u, es[-(i+1)]), 1)
            u = self.dec[i+1](u)

        u = self.conv2(u)
        u = self.unpad(u, pads)
        u = self.unnorm(u, mus, sigmas)
        return u + identity

    @staticmethod
    def add_model_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            '--layers',
            default = [1, 2, 3, 5, 8],
            nargs = '+',
            type = int,
        )

        parser.add_argument(
            '--cin',
            default = 3,
            type = int
        )

        parser.add_argument(
            '--cout',
            default = 3,
            type = int
        )

        parser.add_argument(
            '--nk',
            default = 64,
            type = int
        )

        parser.add_argument(
            '--k1',
            default = 7,
            type = int
        )

        parser.add_argument(
            '--k',
            default = 3,
            type = int
        )

        parser.add_argument(
            '--skip',
            default = 1,
            choices = [0, 1],
            type = int
        )

        parser.add_argument(
            '--normalize',
            default = False,
            action = 'store_true'
        )

        parser.add_argument(
            '--act',
            default = 'prelu',
            choices = ['relu', 'elu', 'selu', 'prelu', 'rrelu', 'leakyrelu'],
            type = str
        )

        parser.add_argument(
            '--down',
            default = 'max',
            choices = ['max', 'mean', 'norm2', 'conv'],
            type = str
        )

        parser.add_argument(
            '--up',
            default = 'nearest',
            choices = ['conv', 'nearest', 'bilinear', 'bicubic'],
            type = str
        )

        parser.add_argument(
            '--nrm',
            default = 'identity',
            choices = ['identity', 'batch', 'instance'],
            type = str
        )

        parser.add_argument(
            '--G',
            default = 2,
            type = int
        )

        parser.add_argument(
            '--bias',
            default = 0,
            choices = [0, 1],
            type = int
        )

        return parser


if __name__ == '__main__':
    parser = UNet.add_model_args(ArgumentParser(add_help=False))
    args = parser.parse_args()

    th.manual_seed(733)
    model = UNet(**vars(args))

    th.manual_seed(733)
    x = th.rand(2, 3, 128, 128)

    y = model(x)
