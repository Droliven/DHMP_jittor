#!/usr/bin/env python
# encoding: utf-8
'''
@project : diverse_sampling_jittor
@file    : lambda_lr.py
@author  : levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2022-07-24 16:56
'''
import jittor as jt
from jittor.optim import Optimizer
import math

class LambdaLR(object):
    '''
    自己实现的 LambdaLR
    '''
    def __init__(self, optimizer, origin_lr=1e-3, epoch_fix_t1=100, epoch_t1=1000, last_epoch=-1):
        self.optimizer = optimizer
        self.epoch_fix_t1 = epoch_fix_t1
        self.epoch_t1 = epoch_t1
        self.last_epoch = last_epoch
        self.origin_lr = origin_lr

    def get_lr(self):
        now_lr = self.optimizer.lr
        return now_lr


    def step(self):
        self.last_epoch += 1
        self.update_lr()


    def update_lr(self):
        # lambda rule
        gamma = 1.0 - max(0, self.last_epoch - self.epoch_fix_t1) / float(self.epoch_t1 - self.epoch_fix_t1 + 1)
        lr = self.origin_lr * gamma
        self.optimizer.lr = lr
        for i, param_group in enumerate(self.optimizer.param_groups):
            if param_group.get("lr") != None:
                param_group["lr"] = lr