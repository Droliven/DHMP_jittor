#!/usr/bin/env python
# encoding: utf-8
'''
@project : gsps_reimplementation
@file    : valid_angle_check.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-12-08 19:56
'''
import jittor as jt
import numpy as np


def humaneva_valid_angle_check(p3d):
    """
    p3d: [bs,14,3] or [bs,42]
    """
    if p3d.shape[-1] == 42:
        p3d = p3d.reshape([p3d.shape[0], 14, 3])

    cos_func = lambda p1, p2: np.sum(p1 * p2, axis=1) / np.linalg.norm(p1, axis=1) / np.linalg.norm(p2, axis=1)
    data_all = p3d
    valid_cos = {}

    # LHip2RHip
    p1 = data_all[:, 7]
    p2 = data_all[:, 10]
    cos_gt_l = cos_func(p1, p2)
    valid_cos['LHip2RHip'] = cos_gt_l

    # Neck2HipPlane
    p1 = data_all[:, 7]
    p2 = data_all[:, 10]
    n0 = np.cross(p1, p2)
    p3 = data_all[:, 0]
    cos_gt_l = cos_func(n0, p3)
    valid_cos['Neck2HipPlane'] = cos_gt_l

    # Head2Neck
    p1 = data_all[:, 13] - data_all[:, 0]
    p2 = data_all[:, 0]
    cos_gt_l = cos_func(p1, p2)
    valid_cos['Head2Neck'] = cos_gt_l

    # Shoulder2Shoulder
    p1 = data_all[:, 1] - data_all[:, 0]
    p2 = data_all[:, 4] - data_all[:, 0]
    cos_gt_l = cos_func(p1, p2)
    valid_cos['Shoulder2Shoulder'] = cos_gt_l

    # ShoulderPlane2HipPlane
    p1 = data_all[:, 7] - data_all[:, 0]
    p2 = data_all[:, 10] - data_all[:, 0]
    n0 = np.cross(p1, p2)
    p3 = data_all[:, 1]
    p4 = data_all[:, 4]
    n1 = np.cross(p3, p4)
    cos_gt_l = cos_func(n0, n1)
    valid_cos['ShoulderPlane2HipPlane'] = cos_gt_l

    # Shoulder2Neck
    p1 = data_all[:, 1] - data_all[:, 0]
    p2 = data_all[:, 0]
    cos_gt_l = cos_func(p1, p2)
    p1 = data_all[:, 4] - data_all[:, 0]
    p2 = data_all[:, 0]
    cos_gt_r = cos_func(p1, p2)
    valid_cos['Shoulder2Neck'] = np.vstack((cos_gt_l, cos_gt_r))

    # Leg2HipPlane
    p1 = data_all[:, 7]
    p2 = data_all[:, 10]
    n0 = np.cross(p1, p2)
    p3 = data_all[:, 8] - data_all[:, 7]
    cos_gt_l = cos_func(n0, p3)
    p3 = data_all[:, 11] - data_all[:, 10]
    cos_gt_r = cos_func(n0, p3)
    valid_cos['Leg2HipPlane'] = np.vstack((cos_gt_l, cos_gt_r))

    # Foot2LegPlane
    p1 = data_all[:, 7] - data_all[:, 10]
    p2 = data_all[:, 11] - data_all[:, 10]
    n0 = np.cross(p1, p2)
    p3 = data_all[:, 12] - data_all[:, 11]
    cos_gt_l = cos_func(n0, p3)
    p1 = data_all[:, 7] - data_all[:, 10]
    p2 = data_all[:, 8] - data_all[:, 7]
    n0 = np.cross(p1, p2)
    p3 = data_all[:, 9] - data_all[:, 8]
    cos_gt_r = cos_func(n0, p3)
    valid_cos['Foot2LegPlane'] = np.vstack((cos_gt_l, cos_gt_r))

    # ForeArm2ShoulderPlane
    p1 = data_all[:, 4] - data_all[:, 0]
    p2 = data_all[:, 5] - data_all[:, 4]
    n0 = np.cross(p1, p2)
    p3 = data_all[:, 6] - data_all[:, 5]
    cos_gt_l = cos_func(n0, p3)
    p1 = data_all[:, 1] - data_all[:, 0]
    p2 = data_all[:, 2] - data_all[:, 1]
    n0 = np.cross(p2, p1)
    p3 = data_all[:, 3] - data_all[:, 2]
    cos_gt_r = cos_func(n0, p3)
    valid_cos['ForeArm2ShoulderPlane'] = np.vstack((cos_gt_l, cos_gt_r))

    return valid_cos


def humaneva_valid_angle_check_cuda(p3d):
    """
    p3d: [bs,14,3] or [bs,42]
    """
    if p3d.shape[-1] == 42:
        p3d = p3d.reshape([p3d.shape[0], 14, 3])

    cos_func = lambda p1, p2: jt.sum(p1 * p2, dim=1) / jt.norm(p1, dim=1) / jt.norm(p2, dim=1)
    data_all = p3d
    valid_cos = {}

    # LHip2RHip
    p1 = data_all[:, 7]
    p2 = data_all[:, 10]
    cos_gt_l = cos_func(p1, p2)
    valid_cos['LHip2RHip'] = cos_gt_l

    # Neck2HipPlane
    p1 = data_all[:, 7]
    p2 = data_all[:, 10]
    n0 = jt.cross(p1, p2)
    p3 = data_all[:, 0]
    cos_gt_l = cos_func(n0, p3)
    valid_cos['Neck2HipPlane'] = cos_gt_l

    # Head2Neck
    p1 = data_all[:, 13] - data_all[:, 0]
    p2 = data_all[:, 0]
    cos_gt_l = cos_func(p1, p2)
    valid_cos['Head2Neck'] = cos_gt_l

    # Shoulder2Shoulder
    p1 = data_all[:, 1] - data_all[:, 0]
    p2 = data_all[:, 4] - data_all[:, 0]
    cos_gt_l = cos_func(p1, p2)
    valid_cos['Shoulder2Shoulder'] = cos_gt_l

    # ShoulderPlane2HipPlane
    p1 = data_all[:, 7] - data_all[:, 0]
    p2 = data_all[:, 10] - data_all[:, 0]
    n0 = jt.cross(p1, p2)
    p3 = data_all[:, 1]
    p4 = data_all[:, 4]
    n1 = jt.cross(p3, p4)
    cos_gt_l = cos_func(n0, n1)
    valid_cos['ShoulderPlane2HipPlane'] = cos_gt_l

    # Shoulder2Neck
    p1 = data_all[:, 1] - data_all[:, 0]
    p2 = data_all[:, 0]
    cos_gt_l = cos_func(p1, p2)
    p1 = data_all[:, 4] - data_all[:, 0]
    p2 = data_all[:, 0]
    cos_gt_r = cos_func(p1, p2)
    valid_cos['Shoulder2Neck'] = jt.stack((cos_gt_l, cos_gt_r), 0)

    # Leg2HipPlane
    p1 = data_all[:, 7]
    p2 = data_all[:, 10]
    n0 = jt.cross(p1, p2)
    p3 = data_all[:, 8] - data_all[:, 7]
    cos_gt_l = cos_func(n0, p3)
    p3 = data_all[:, 11] - data_all[:, 10]
    cos_gt_r = cos_func(n0, p3)
    valid_cos['Leg2HipPlane'] = jt.stack((cos_gt_l, cos_gt_r), 0)

    # Foot2LegPlane
    p1 = data_all[:, 7] - data_all[:, 10]
    p2 = data_all[:, 11] - data_all[:, 10]
    n0 = jt.cross(p1, p2)
    p3 = data_all[:, 12] - data_all[:, 11]
    cos_gt_l = cos_func(n0, p3)
    p1 = data_all[:, 7] - data_all[:, 10]
    p2 = data_all[:, 8] - data_all[:, 7]
    n0 = jt.cross(p1, p2)
    p3 = data_all[:, 9] - data_all[:, 8]
    cos_gt_r = cos_func(n0, p3)
    valid_cos['Foot2LegPlane'] = jt.stack((cos_gt_l, cos_gt_r), 0)

    # ForeArm2ShoulderPlane
    p1 = data_all[:, 4] - data_all[:, 0]
    p2 = data_all[:, 5] - data_all[:, 4]
    n0 = jt.cross(p1, p2)
    p3 = data_all[:, 6] - data_all[:, 5]
    cos_gt_l = cos_func(n0, p3)
    p1 = data_all[:, 1] - data_all[:, 0]
    p2 = data_all[:, 2] - data_all[:, 1]
    n0 = jt.cross(p2, p1)
    p3 = data_all[:, 3] - data_all[:, 2]
    cos_gt_r = cos_func(n0, p3)
    valid_cos['ForeArm2ShoulderPlane'] = jt.stack((cos_gt_l, cos_gt_r), 0)

    return valid_cos
