#!/usr/bin/env python
# encoding: utf-8
'''
@project : baseresample_likegsps
@file    : cvae.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-12-10 09:41
'''
import jittor as jt
from jittor import Module
from jittor.nn import Linear, Sequential

from .gcn_layers import GraphConv, GraphConvBlock, ResGCB

class CVAE(Module):
    def __init__(self, node_n=16, hidden_dim=256, z_dim=64, dct_n=10, dropout_rate=0):
        super(CVAE, self).__init__()

        self.node_n = node_n
        self.dct_n = dct_n
        self.z_dim = z_dim

        self.enc = Sequential(
            GraphConvBlock(in_len=3 * dct_n * 2, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n,
                           dropout_rate=dropout_rate, bias=True, residual=False),
            ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate,
                   bias=True, residual=True),
            ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate,
                   bias=True, residual=True),
            ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate,
                   bias=True, residual=True),
            ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate,
                   bias=True, residual=True),
        )

        self.mean = Linear(hidden_dim * node_n, z_dim)
        self.logvar = Linear(hidden_dim * node_n, z_dim)

        self.dec = Sequential(
            GraphConvBlock(in_len=3*dct_n+z_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=False),
            ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=True),
            ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=True),
            ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=True),
            ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=True),
            GraphConv(in_len=hidden_dim, out_len=3*dct_n, in_node_n=node_n, out_node_n=node_n, bias=True)
        )

    def _sample(self, mean, logvar):
        """Returns a sample from a Gaussian with diagonal covariance."""
        return jt.exp(0.5*logvar) * jt.randn_like(logvar) + mean

    def execute(self, condition, data):
        '''

        Args:
            condition: [b, 48, 25] / [b, 16, 30]
            data: [b, 48, 100] / [b, 16, 30]

        Returns:

        '''
        b, v, ct = condition.shape
        feature = self.enc(jt.concat((condition, data), dim=-1)) # b, 16, 60 -> b, 16, 256

        mean = self.mean(feature.view(b, -1))
        logvar = self.logvar(feature.view(b, -1))
        z = self._sample(mean, logvar)

        out = self.dec(jt.concat((condition, z.unsqueeze(dim=1).repeat([1, self.node_n, 1])), dim=-1))  # b, 16, 30+64
        out = out + condition
        return out, mean, logvar


    def inference(self, condition, z):
        '''

        Args:
            condition: [b, 48, 25] / [b, 16, 30]
            z: b, 64

        Returns:

        '''
        out = self.dec(jt.concat((condition, z.unsqueeze(dim=1).repeat([1, self.node_n, 1])), dim=-1))  # b, 16, 30+64
        out = out + condition
        return out



