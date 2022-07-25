#!/usr/bin/env python
# encoding: utf-8
'''
@project : gsps_reimplementation
@file    : gcn_layers.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-12-03 15:48
'''

import jittor as jt
from jittor import nn, Module
import math

class GraphConv(Module):
    """
        adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
        """

    def __init__(self, in_len, out_len, in_node_n=66, out_node_n=66, bias=True):
        super(GraphConv, self).__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.in_node_n = in_node_n
        self.out_node_n = out_node_n

        self.weight = jt.randn((in_len, out_len), requires_grad=True)
        self.att = jt.randn((in_node_n, out_node_n), requires_grad=True)

        if bias:
            self.bias = jt.randn(out_len, requires_grad=True)
        else:
            self.bias = None


        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.shape[1])
        nn.init.uniform_(self.weight, -stdv, stdv)
        nn.init.uniform_(self.att, -stdv, stdv)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -stdv, stdv)

    def execute(self, input):
        '''
        b, cv, t
        '''

        features = jt.matmul(input, self.weight)  # 35 -> 256
        output = jt.matmul(features.permute(0, 2, 1), self.att).permute(0, 2, 1)  # 66 -> 66
        if self.bias is not None:
            output = output + self.bias
        return output


class GraphConvBlock(Module):
    def __init__(self, in_len, out_len, in_node_n, out_node_n, dropout_rate=0, leaky=0.1, bias=True, residual=False):
        super(GraphConvBlock, self).__init__()
        self.dropout_rate = dropout_rate
        self.resual = residual

        self.out_len = out_len

        self.gcn = GraphConv(in_len, out_len, in_node_n=in_node_n, out_node_n=out_node_n, bias=bias)
        self.bn = nn.BatchNorm1d(out_node_n * out_len)
        self.act = nn.Tanh()
        if self.dropout_rate > 0:
            self.drop = nn.Dropout(dropout_rate)

    def execute(self, input):
        '''

        Args:
            input: b, cv, t

        Returns:

        '''
        x = self.gcn(input)
        b, vc, t = x.shape
        x = self.bn(x.reshape(b, -1)).reshape(b, vc, t)
        # x = self.bn(x.view(b, -1, 3, t).permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1, 3).contiguous().view(b, vc, t)
        x = self.act(x)
        if self.dropout_rate > 0:
            x = self.drop(x)

        if self.resual:
            return x + input
        else:
            return x


class ResGCB(Module):
    def __init__(self, in_len, out_len, in_node_n, out_node_n, dropout_rate=0, leaky=0.1, bias=True, residual=False):
        super(ResGCB, self).__init__()
        self.resual = residual
        self.gcb1 = GraphConvBlock(in_len, in_len, in_node_n=in_node_n, out_node_n=in_node_n, dropout_rate=dropout_rate, bias=bias, residual=False)
        self.gcb2 = GraphConvBlock(in_len, out_len, in_node_n=in_node_n, out_node_n=out_node_n, dropout_rate=dropout_rate, bias=bias, residual=False)


    def execute(self, input):
        '''

        Args:
            x: B,CV,T

        Returns:

        '''

        x = self.gcb1(input)
        x = self.gcb2(x)

        if self.resual:
            return x + input
        else:
            return x

if __name__ == '__main__':
    m = GraphConv(in_len=35, out_len=256, in_node_n=66, out_node_n=66, bias=True)
    x = jt.randn((4, 66, 35))
    y = m(x)
    pass