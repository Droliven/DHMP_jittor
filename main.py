#!/usr/bin/env python
# encoding: utf-8
'''
@project : diverse_sampling
@file    : main.py
@author  : levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2022-07-11 22:59
'''
# ****************************************************************************************************************
# *********************************************** Environments ***************************************************
# ****************************************************************************************************************

import numpy as np
import random
import jittor as jt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

jt.flags.use_cuda = 1

# ****************************************************************************************************************
# *********************************************** Main ***********************************************************
# ****************************************************************************************************************

import argparse

from h36m.runs import RunCVAE as RunCVAEH36m
from h36m.runs import RunDiverseSampling as RunDiverseSamplingH36m
from humaneva.runs import RunCVAE as RunCVAEHumaneva
from humaneva.runs import RunDiverseSampling as RunDiverseSamplingHumaneva

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--exp_name', type=str, default="h36m_t2", help="h36m_t1 / h36m_t2 / humaneva_t1 / humaneva_t2")
parser.add_argument('--is_train', type=bool, default='', help="")
parser.add_argument('--is_load', type=bool, default='', help="")
parser.add_argument('--is_debug', type=bool, default='', help="")

parser.add_argument('--model_path', type=str, default="", help="")

args = parser.parse_args()

if args.exp_name == "h36m_t1":
    args.model_path = os.path.join(r"./ckpt/pretrained_jittor", "h36m_t1.pkl")
    r = RunCVAEH36m(exp_name=args.exp_name, is_debug=args.is_debug, args=args)

elif args.exp_name == "h36m_t2":
    args.model_path = os.path.join(r"./ckpt/pretrained_jittor", "h36m_t2.pkl")
    r = RunDiverseSamplingH36m(exp_name=args.exp_name, is_debug=args.is_debug, args=args)

elif args.exp_name == "humaneva_t1":
    args.model_path = os.path.join(r"./ckpt/pretrained_jittor", "humaneva_t1.pkl")
    r = RunCVAEHumaneva(exp_name=args.exp_name, is_debug=args.is_debug, args=args)

elif args.exp_name == "humaneva_t2":
    args.model_path = os.path.join(r"./ckpt/pretrained_jittor", "humaneva_t2.pkl")
    r = RunDiverseSamplingHumaneva(exp_name=args.exp_name, is_debug=args.is_debug, args=args)

else:
    print("wrong exp_name!")


if args.is_load:
    r.restore(args.model_path)

if args.is_train:
    r.run()

else:
    diversity, ade, fde, mmade, mmfde = r.eval(epoch=-1, draw=True)
    print("\n Test -->  div {:.4f} -- ade {:.4f} --  fde {:.4f} --  mmade {:.4f} --  mmfde {:.4f} ".format(diversity,
                                                                                               ade,
                                                                                                fde,
                                                                                                mmade,
                                                                                               mmfde))

