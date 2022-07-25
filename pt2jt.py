import copy
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

import jittor as jt
jt.flags.use_cuda = 0

import argparse

# ======================================================================================================================
from h36m.nets import CVAE as CVAE_h36m_jt
from h36m.nets import DiverseSampling as DiverseSampling_h36m_jt

from h36m.configs import ConfigCVAE as ConfigCVAE_h36m
from h36m.configs import ConfigDiverseSampling as ConfigDiverseSampling_h36m

# ======

from humaneva.nets import CVAE as CVAE_humaneva_jt
from humaneva.nets import DiverseSampling as DiverseSampling_humaneva_jt

from humaneva.configs import ConfigCVAE as ConfigCVAE_humaneva
from humaneva.configs import ConfigDiverseSampling as ConfigDiverseSampling_humaneva

# exp_name = "h36m_t2"
exp_name = "humaneva_t2"

if exp_name == "h36m_t1":
    cfg = ConfigCVAE_h36m(exp_name)
    m_jt = CVAE_h36m_jt(node_n=cfg.node_n, hidden_dim=cfg.hidden_dim, z_dim=cfg.z_dim, dct_n=cfg.dct_n, dropout_rate=cfg.dropout_rate)
elif exp_name == "h36m_t2":
    cfg = ConfigDiverseSampling_h36m(exp_name)
    m_jt = DiverseSampling_h36m_jt(node_n=cfg.node_n, hidden_dim=cfg.hidden_dim,
                                                      base_dim=cfg.base_dim, base_num_p1=cfg.base_num_p1,
                                                      z_dim=cfg.z_dim, dct_n=cfg.dct_n,
                                                      dropout_rate=cfg.dropout_rate)
elif exp_name == "humaneva_t1":
    cfg = ConfigCVAE_humaneva(exp_name)
    m_jt = CVAE_humaneva_jt(node_n=cfg.node_n, hidden_dim=cfg.hidden_dim, z_dim=cfg.z_dim, dct_n=cfg.dct_n, dropout_rate=cfg.dropout_rate)
elif exp_name == "humaneva_t2":
    cfg = ConfigDiverseSampling_humaneva(exp_name)
    m_jt = DiverseSampling_h36m_jt(node_n=cfg.node_n, hidden_dim=cfg.hidden_dim,
                                                      base_dim=cfg.base_dim, base_num_p1=cfg.base_num_p1,
                                                      z_dim=cfg.z_dim, dct_n=cfg.dct_n,
                                                      dropout_rate=cfg.dropout_rate)

stact_dict = torch.load(fr"E:\PythonWorkspace\stochastic_human_motion_prediction\diverse_sampling\ckpt\pretrained\{exp_name}.pth", map_location="cpu")["model"]

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

# 检查
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
jt.save({"model": m_jt.state_dict()}, os.path.join(rf"E:\PythonWorkspace\stochastic_human_motion_prediction\DHMP_jittor\ckpt\pretrained_jittor/{exp_name}.pkl"))
