#!/usr/bin/env python
# encoding: utf-8
'''
@project : m3day32022
@file    : evaluate_fid_acc_humaneva.py
@author  : levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2022-06-08 21:52
'''

from humaneva.nets import DiverseSampling, CVAE
from humaneva.configs import ConfigDiverseSampling
from humaneva.datas import get_dct_matrix, reverse_dct_cuda, dct_transform_cuda

from .classifier import ClassifierForAcc, ClassifierForFID
from .humaneva_dataset_classifier import MaoweiGSPS_Dynamic_Seq_Classifier_Humaneva

import jittor as jt
import os
from tqdm import tqdm
from pprint import pprint
import random
import numpy as np
import json
import pickle
from scipy import linalg


class Evaluate_FID_ACC_Humaneva():
    def __init__(self):
        super(Evaluate_FID_ACC_Humaneva, self).__init__()

        self.cfg = ConfigDiverseSampling(exp_name="humaneva_t2")
        print("\n================== Configs =================")
        pprint(vars(self.cfg), indent=4)
        print("==========================================\n")

        # 模型
        self.classifier_for_acc = ClassifierForAcc(input_size=42, hidden_size=128, hidden_layer=2,
                                                   output_size=self.cfg.class_num,
                                                   use_noise=None)
        self.classifier_for_fid = ClassifierForFID(input_size=42, hidden_size=128, hidden_layer=2,
                                                   output_size=self.cfg.class_num,
                                                   use_noise=None)
        self.model_t1 = CVAE(node_n=self.cfg.node_n, hidden_dim=self.cfg.hidden_dim, z_dim=self.cfg.z_dim,
                             dct_n=self.cfg.dct_n, dropout_rate=self.cfg.dropout_rate)
        self.model = DiverseSampling(node_n=self.cfg.node_n, hidden_dim=self.cfg.hidden_dim,
                               base_dim=self.cfg.base_dim, base_num_p1=self.cfg.base_num_p1,
                               z_dim=self.cfg.z_dim, dct_n=self.cfg.dct_n,
                               dropout_rate=self.cfg.dropout_rate)

        print(">>> total params of {}: {:.6f}M\n".format("t1", sum(
            p.numel() for p in self.model_t1.parameters()) / 1000000.0))
        print(">>> total params of {}: {:.6f}M\n".format("t2",
                                                         sum(p.numel() for p in self.model.parameters()) / 1000000.0))


        # 导入参数并冻结
        model_t1_state = jt.load(self.cfg.model_path_t1)
        self.model_t1.load_state_dict(model_t1_state["model"])
        print("{} loaded from {}".format("model_t1", self.cfg.model_path_t1))
        for p in self.model_t1.parameters():
            p.requires_grad = False
        self.model_t1.eval()

        classifier_path = os.path.join("./ckpt/classifier_jittor", "humaneva_classifier.pkl")

        classifier_state = jt.load(classifier_path)
        self.classifier_for_acc.load_state_dict(classifier_state["model"])
        self.classifier_for_fid.load_state_dict(classifier_state["model"])
        self.classifier_for_acc.eval()
        self.classifier_for_fid.eval()
        print(
            f"classifier_jittor loaded from {classifier_path}")


        # 数据
        self.test_data = MaoweiGSPS_Dynamic_Seq_Classifier_Humaneva(data_path=self.cfg.base_data_dir,
                                                         similar_idx_path=self.cfg.similar_idx_path,
                                                         similar_pool_path=self.cfg.similar_pool_path,
                                                         t_his=self.cfg.t_his,
                                                         t_pred=self.cfg.t_pred, similar_cnt=0,
                                                         dynamic_sub_len=self.cfg.sub_len_train,
                                                         batch_size=self.cfg.test_batch_size,
                                                         joint_used_17=self.cfg.joint_used, subjects=self.cfg.subjects,
                                                         parents_17=self.cfg.parents,
                                                         mode="test",
                                                         multimodal_threshold=self.cfg.multimodal_threshold,
                                                         is_debug=False)
        self.test_data.get_test_similat_gt_like_dlow()

        self.valid_angle = pickle.load(open(self.cfg.valid_angle_path, "rb"))  # dict 13
        print(f"{'valid angle'} loaded from {self.cfg.valid_angle_path} !")

        ## dct
        self.dct_m, self.i_dct_m = get_dct_matrix(self.cfg.t_total)
        self.dct_m = jt.array(self.dct_m)
        self.i_dct_m = jt.array(self.i_dct_m)

    def restore(self, checkpoint_path):
        state = jt.load(checkpoint_path)
        self.model.load_state_dict(state["model"])
        print("load from {}".format(checkpoint_path))

    def _sample_weight_gumbel_softmax(self, logits, temperature=1, eps=1e-20):
        # b*h, 1, 10
        assert temperature > 0, "temperature must be greater than 0 !"

        U = jt.rand(logits.shape)
        g = -jt.log(-jt.log(U + eps) + eps)

        y = logits + g
        y = y / temperature
        y = jt.nn.softmax(y, dim=-1)
        return y

    def compute_fid_and_acc(self):
        self.model.eval()
        confusion = np.zeros((self.cfg.class_num, self.cfg.class_num))

        all_gt_activations = []
        all_pred_activations = []

        dg = self.test_data.onebyone_generator()
        generator_len = len(self.test_data.similat_gt_like_dlow)
        for i, (datas, action_idx_5) in enumerate(tqdm(dg, total=generator_len)):
            # b, 48, 125
            b, vc, t = datas.shape
            similars = self.test_data.similat_gt_like_dlow[i]  # 0/n, 48, 100
            if similars.shape[0] == 0:  # todo 这会淡化误差
                continue

            datas = jt.array(datas)

            with jt.no_grad():
                padded_inputs = datas[:, :, list(range(self.cfg.t_his)) + [self.cfg.t_his - 1] * self.cfg.t_pred]
                padded_inputs_dct = dct_transform_cuda(padded_inputs, self.dct_m, dct_n=self.cfg.dct_n)  # b, 48, 10
                padded_inputs_dct = padded_inputs_dct.view(b, -1, 3 * self.cfg.dct_n)  # # b, 16, 3*10

                # todo train generator eps 不共享
                repeated_eps_1 = jt.randn((b * self.cfg.nk, self.cfg.z_dim))
                logtics = jt.ones((b * self.cfg.nk, 1, self.cfg.base_num_p1)) / self.cfg.base_num_p1  # b*h, 1, 10
                many_weights = self._sample_weight_gumbel_softmax(logtics,
                                                                  temperature=self.cfg.temperature_p1)  # b*h, 1, 10

                all_z, all_mean_p1, all_logvar_p1 = self.model(condition=padded_inputs_dct,
                                                                  repeated_eps=repeated_eps_1,
                                                                  many_weights=many_weights,
                                                                  multi_modal_head=self.cfg.nk)  # b*(10), 128


                all_outs_dct = self.model_t1.inference(
                    condition=jt.misc.repeat_interleave(padded_inputs_dct, repeats=self.cfg.nk, dim=0),
                    z=all_z)  # b*h, 16, 30
                all_outs_dct = all_outs_dct.reshape(b * self.cfg.nk, -1, self.cfg.dct_n)  # b*h, 48, 10
                outputs = reverse_dct_cuda(all_outs_dct, self.i_dct_m, self.cfg.t_total)  # b*h, 48, 125
                outputs = outputs.view(self.cfg.nk, -1, self.cfg.t_total)[:, :, self.cfg.t_his:]  # 50, 48, 100

                probs = self.classifier_for_acc(motion_sequence=outputs)
                # batch_pred = probs.max(dim=1).indices.data
                batch_pred = np.argmax(probs.data, axis=1) # 50

                action_idx_5 = action_idx_5.reshape(-1).repeat(self.cfg.nk)

                for label, pred in zip(action_idx_5, batch_pred):
                    # print(label.data, pred.data)
                    confusion[label][pred] += 1

                pred_activations = self.classifier_for_fid(motion_sequence=outputs).data
                gt_activations = self.classifier_for_fid(
                    motion_sequence=datas[:, :, self.cfg.t_his:]).data

                all_gt_activations.append(gt_activations)
                all_pred_activations.append(pred_activations)

        acc = np.trace(confusion) / ((i + 1) * self.cfg.nk)

        all_gt_activations = np.concatenate(all_gt_activations, axis=0)  # n, 48
        all_pred_activations = np.concatenate(all_pred_activations, axis=0)  # n, 48
        gt_statistics = self._calculate_activation_statistics(all_gt_activations)
        pred_statiistics = self._calculate_activation_statistics(all_pred_activations)
        fid = self._calculate_fid_helper(gt_statistics, pred_statiistics)
        return fid, acc

    def _calculate_activation_statistics(self, activations):  # [b, 48]
        mu = np.mean(activations, axis=0)  # [48,]
        sigma = np.cov(activations, rowvar=False)  # [48, 48]

        return mu, sigma

    def _calculate_fid_helper(self, statistics_1, statistics_2):
        return self._calculate_frechet_distance(statistics_1[0], statistics_1[1],
                                                statistics_2[0], statistics_2[1])

    def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)
