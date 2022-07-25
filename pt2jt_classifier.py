import copy
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

import jittor as jt
jt.flags.use_cuda = 0

import argparse

# ======================================================================================================================
from fid_acc.classifier import ClassifierForAcc

exp_name = "h36m_classifier"
# exp_name = "humaneva_classifier"

if exp_name == "h36m_classifier":
    m_jt = ClassifierForAcc(input_size=48, hidden_size=128, hidden_layer=2,
                                                   output_size=15,
                                                   use_noise=None)
elif exp_name == "humaneva_classifier":
    m_jt = ClassifierForAcc(input_size=42, hidden_size=128, hidden_layer=2,
                                                   output_size=5,
                                                   use_noise=None)

stact_dict = torch.load(fr"E:\PythonWorkspace\stochastic_human_motion_prediction\diverse_sampling\ckpt\classifier\{exp_name}.pth", map_location="cpu")["model"]
pt_key = list(stact_dict.keys())

jt_dict = m_jt.state_dict()
jt_key = list(jt_dict.keys())

modified_pt_stact_dict = {}
for k in pt_key:
    if k in jt_key:
        modified_pt_stact_dict[k] = copy.deepcopy(stact_dict[k])
    elif k + ".weight" in jt_key:
        modified_pt_stact_dict[k + ".weight"] = copy.deepcopy(stact_dict[k])
    else:
        continue

is_shape_match = True
all_keys = jt_key + list(modified_pt_stact_dict.keys())
all_key = set(all_keys)
cnt = 0
for k in all_key:
    if (k in modified_pt_stact_dict and k in jt_key):
        cnt += 1
        if not list(modified_pt_stact_dict[k].shape) == list(jt_dict[k].shape):
            is_shape_match = False

assert is_shape_match and cnt == len(modified_pt_stact_dict.keys()) and cnt == len(jt_key)

m_jt.load_state_dict(modified_pt_stact_dict)
# jt.save({"model": m_jt.state_dict()}, os.path.join(rf"E:\PythonWorkspace\stochastic_human_motion_prediction\DHMP_jittor\ckpt\classifier_jittor\{exp_name}.pkl"))
