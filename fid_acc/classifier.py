#!/usr/bin/env python
# encoding: utf-8
'''
@project : m3day32022
@file    : classifier_jittor.py
@author  : levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2022-06-08 15:37
'''
import jittor as jt
from jittor import Module
from jittor.nn import GRU, Linear

class ClassifierForAcc(Module):
    def __init__(self, input_size=48, hidden_size=128, hidden_layer=2, output_size=15, use_noise=None):
        super(ClassifierForAcc, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer
        self.use_noise = use_noise

        self.recurrent = GRU(input_size, hidden_size, hidden_layer)
        self.linear1 = Linear(hidden_size, 30)
        self.linear2 = Linear(30, output_size)


    def execute(self, motion_sequence, hidden_unit=None):
        '''
        motion_sequence: b, 48, 100
        hidden_unit:
        '''
        motion_sequence = motion_sequence.permute(2, 0, 1) # [100, b, 48]
        # dim (motion_length, num_samples, hidden_size)
        if hidden_unit is None:
            hidden_unit = self.initHidden(motion_sequence.size(1), self.hidden_layer) # [2, b, 128]

        gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit) # [100, b, 48]
        # dim (num_samples, 30)
        lin1 = self.linear1(gru_o[-1, :, :]) # [b, 48]
        lin1 = jt.tanh(lin1)
        # dim (num_samples, output_size)
        lin2 = self.linear2(lin1)
        return lin2


    def initHidden(self, num_samples, layer):
        return jt.randn((layer, num_samples, self.hidden_size), requires_grad=False)

class ClassifierForFID(Module):
    def __init__(self, input_size=48, hidden_size=128, hidden_layer=2, output_size=15, use_noise=None):
        super(ClassifierForFID, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer
        self.use_noise = use_noise

        self.recurrent = GRU(input_size, hidden_size, hidden_layer)
        self.linear1 = Linear(hidden_size, 30)
        self.linear2 = Linear(30, output_size)

    def execute(self, motion_sequence, hidden_unit=None):
        '''
        motion_sequence: b, 48, 100
        hidden_unit:
        '''
        motion_sequence = motion_sequence.permute(2, 0, 1)  # [100, b, 48]

        # dim (motion_length, num_samples, hidden_size)
        if hidden_unit is None:
            hidden_unit = self.initHidden(motion_sequence.size(1), self.hidden_layer)

        gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit)
        # dim (num_samples, 30)
        lin1 = self.linear1(gru_o[-1, :, :])
        lin1 = jt.tanh(lin1)
        return lin1


    def initHidden(self, num_samples, layer):
        return jt.randn((layer, num_samples, self.hidden_size), requires_grad=False)
