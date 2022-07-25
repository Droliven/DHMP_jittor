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
import jittor as jt
import numpy as np
import random
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

jt.flags.use_cuda = 1

# ****************************************************************************************************************
# *********************************************** Main ***********************************************************
# ****************************************************************************************************************

import argparse
import pandas as pd
from pprint import pprint

from fid_acc import Evaluate_FID_ACC_H36m
from fid_acc import Evaluate_FID_ACC_Humaneva

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--exp_name', type=str, default="humaneva_t2", help="h36m_t2 / humaneva_t2")

args = parser.parse_args()

if args.exp_name == "h36m_t2":
    r = Evaluate_FID_ACC_H36m()
    r.restore(os.path.join("./ckpt/pretrained_jittor", "h36m_t2.pkl"))

elif args.exp_name == "humaneva_t2":
    r = Evaluate_FID_ACC_Humaneva()
    r.restore(os.path.join("./ckpt/pretrained_jittor", "humaneva_t2.pkl"))

else:
    print("wrong exp_name!")


fid, acc = r.compute_fid_and_acc()
print("\n Test -->  fid {:.4f} -- acc {:.4f}".format(fid, acc))


