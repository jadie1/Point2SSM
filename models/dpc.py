# DPC: Unsupervised Deep Point Correspondence via Cross and Self Construction
# Source: https://github.com/dvirginz/DPC

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_utils import calc_cd
from torch_cluster import knn
from models.dgcnn import DGCNN_encoder, get_graph_feature

import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(proj_dir, "utils/Pointnet2.PyTorch/pointnet2"))
from pointnet2_utils import grouping_operation


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.num_input = args.num_input_points
        self.latent_dim = args.latent_dim
        self.num_output = args.num_output_points
        self.train_loss = args.loss
        self.self_recon_lambda = args.self_recon_lambda
        self.cross_recon_lambda = args.cross_recon_lambda
        self.neigh_loss_lambda = args.neigh_loss_lambda
        self.device = args.device

        self.num_neighs = 27
        self.similarity_init = 'cosine'
        self.k_for_cross_recon = 10
        self.k_for_self_recon = 10
        self.sim_normalization = 'softmax'

        self.encoder = DGCNN_encoder(self.latent_dim)

    def measure_similarity(self, source_encoded, target_encoded):
        """
        Measure the similarity between two batched matrices vector by vector

        Args:
            similarity_init : The method to calculate similarity with(e.g cosine)
            source_encoded (BxNxF Tensor): The input 1 matrix
            target_encoded (BxNxF Tensor): The input 2 matrix
        """
        "multiplication", "cosine", "difference"
        if self.similarity_init == "cosine":
            a_norm = source_encoded / source_encoded.norm(dim=-1)[:, :, None]
            b_norm = target_encoded / target_encoded.norm(dim=-1)[:, :, None]
            return torch.bmm(a_norm, b_norm.transpose(1, 2))
        if self.similarity_init == "mult":
            return torch.bmm(source_encoded, target_encoded.transpose(1, 2))
        if self.similarity_init == "l2":
            diff = torch.cdist(source_encoded,target_encoded)
            return diff.max() - diff
        if self.similarity_init == "negative_l2":
            diff = -torch.cdist(source_encoded,target_encoded)
            return diff
        if self.similarity_init == "difference_exp":
            dist = torch.cdist(source_encoded.contiguous(), target_encoded.contiguous())
            return torch.exp(-dist * 2 * source_encoded.shape[-1])
        if self.similarity_init == "difference_inverse":
            # TODO maybe (max - tensor) instead of 1/tensor ?
            EPS = 1e-6
            return 1 / (torch.cdist(source_encoded.contiguous(), target_encoded.contiguous()) + EPS)
        if self.similarity_init == "difference_max_norm":
            dist = torch.cdist(source_encoded.contiguous(), target_encoded.contiguous())
            return (dist.max() - dist) / dist.max()
        if self.similarity_init == "multiplication":
            return torch.bmm(source_encoded, target_encoded.transpose(1, 2))

    @staticmethod
    def reconstruction(pos, nn_idx, nn_weight, k):
        nn_pos = get_graph_feature(pos.transpose(1, 2), k=k, idx=nn_idx, only_intrinsic='neighs', permute_feature=False)
    
        nn_weighted = nn_pos * nn_weight.unsqueeze(dim=3)
        recon = torch.sum(nn_weighted, dim=2)

        recon_hard = nn_pos[:, :, 0, :]
 
        return recon, recon_hard

    def chamfer_loss(self, pred, gt):
        cd_p, cd_t = calc_cd(pred, gt)
        if self.train_loss == 'cd_t':
            return cd_t
        elif self.train_loss == 'cd_p':
            return cd_p

    def forward_shape(self, pos, dense_output_features):
        P_self = self.measure_similarity(dense_output_features, dense_output_features)

        # measure self similarity
        nn_idx = None
        self_nn_weight, _, self_nn_idx, _, _, _ = \
            get_s_t_neighbors(self.k_for_self_recon + 1, P_self, sim_normalization=self.sim_normalization, s_only=True, ignore_first=True, nn_idx=nn_idx)

        # self reconstruction
        self_recon, _ = self.reconstruction(pos, self_nn_idx, self_nn_weight, self.k_for_self_recon)

        return self_recon, P_self

    @staticmethod
    def get_neighbor_loss(source, source_neigh_idxs, target_cross_recon, k):
        # source.shape[1] is the number of points

        if k < source_neigh_idxs.shape[2]:
            neigh_index_for_loss = source_neigh_idxs[:, :, :k]
        else:
            neigh_index_for_loss = source_neigh_idxs

        source_grouped = grouping_operation(source.transpose(1, 2).contiguous(), neigh_index_for_loss.int()).permute(0, 2, 3, 1)
        source_diff = source_grouped[:, :, 1:, :] - torch.unsqueeze(source, 2)  # remove fist grouped element, as it is the seed point itself
        source_square = torch.sum(source_diff ** 2, dim=-1)

        target_cr_grouped = grouping_operation(target_cross_recon.transpose(1, 2).contiguous(), neigh_index_for_loss.int()).permute(0, 2, 3, 1)
        target_cr_diff = target_cr_grouped[:, :, 1:, :] - torch.unsqueeze(target_cross_recon, 2)  # remove fist grouped element, as it is the seed point itself
        target_cr_square = torch.sum(target_cr_diff ** 2, dim=-1)

        GAUSSIAN_HEAT_KERNEL_T = 8.0
        gaussian_heat_kernel = torch.exp(-source_square/GAUSSIAN_HEAT_KERNEL_T)
        neighbor_loss_per_neigh = torch.mul(gaussian_heat_kernel, target_cr_square)

        neighbor_loss = torch.mean(neighbor_loss_per_neigh)

        return neighbor_loss

    def forward(self, source, target, gt, is_training=True, alpha=None):
        # get features (B, N, F)
        source_global, source_dense_output_features, source_neigh_idxs = self.encoder(source, return_neighs=True)
        target_global, target_dense_output_features, target_neigh_idxs = self.encoder(target, return_neighs=True)
        
        # measure cross similarity (B, N, N)
        P_non_normalized = self.measure_similarity(source_dense_output_features, target_dense_output_features) 
        # cross nearest neighbors and weights
        source_cross_nn_weight, source_cross_nn_sim, source_cross_nn_idx, target_cross_nn_weight, target_cross_nn_sim, target_cross_nn_idx =\
            get_s_t_neighbors(self.k_for_cross_recon, P_non_normalized, sim_normalization=self.sim_normalization)

        # cross reconstruction
        source_cross_recon, source_cross_recon_hard = self.reconstruction(source, target_cross_nn_idx, target_cross_nn_weight, self.k_for_cross_recon)
        target_cross_recon, target_cross_recon_hard = self.reconstruction(target, source_cross_nn_idx, source_cross_nn_weight, self.k_for_cross_recon)

        if is_training:
            # cross reconstruction losses
            source_cross_recon_loss = self.cross_recon_lambda * self.chamfer_loss(source, source_cross_recon)
            target_cross_recon_loss = self.cross_recon_lambda * self.chamfer_loss(target, target_cross_recon)

            # self reconstruction
            source_self_recon, P_self_source = self.forward_shape(source, source_dense_output_features)
            target_self_recon, P_self_target = self.forward_shape(target, target_dense_output_features)

            # self reconstruction losses
            source_self_recon_loss = self.self_recon_lambda * self.chamfer_loss(source, source_self_recon)
            target_self_recon_loss = self.self_recon_lambda * self.chamfer_loss(target, target_self_recon)

            # mapping losses
            neigh_loss_fwd = self.neigh_loss_lambda * self.get_neighbor_loss(source, source_neigh_idxs, target_cross_recon, self.k_for_cross_recon)
            neigh_loss_bac = self.neigh_loss_lambda * self.get_neighbor_loss(target, target_neigh_idxs, source_cross_recon, self.k_for_cross_recon)

            # total loss
            recon_loss = source_cross_recon_loss + target_cross_recon_loss + source_self_recon_loss + target_self_recon_loss
            loss = recon_loss.mean() + neigh_loss_fwd + neigh_loss_bac

            return source_cross_recon, loss
        else:
            cd_p, cd_t, f1 = calc_cd(source_cross_recon, gt, calc_f1=True)
            return {'recon': source_cross_recon, 'cd_p': cd_p, 'cd_t': cd_t, 'f1': f1}


def get_s_t_topk(P, k, s_only=False,nn_idx=None):
    """
    Get nearest neighbors per point (similarity value and index) for source and target shapes

    Args:
        P (BxNsxNb Tensor): Similarity matrix
        k: number of neighbors per point
    """
    if(nn_idx is not None):
        assert s_only, "Only for self-construction currently"
        s_nn_idx = nn_idx
        s_nn_val = P.gather(dim=2,index=nn_idx)
        t_nn_val = t_nn_idx = None
    else:
        s_nn_val, s_nn_idx = P.topk(k=min(k,P.shape[2]), dim=2)

        if not s_only:
            t_nn_val, t_nn_idx = P.topk(k=k, dim=1)

            t_nn_val = t_nn_val.transpose(2, 1)
            t_nn_idx = t_nn_idx.transpose(2, 1)
        else:
            t_nn_val = None
            t_nn_idx = None

    return s_nn_val, s_nn_idx, t_nn_val, t_nn_idx


def get_s_t_neighbors(k, P, sim_normalization, s_only=False, ignore_first=False,nn_idx=None):
    s_nn_sim, s_nn_idx, t_nn_sim, t_nn_idx = get_s_t_topk(P, k, s_only=s_only,nn_idx=nn_idx)
    if ignore_first:
        s_nn_sim, s_nn_idx = s_nn_sim[:, :, 1:], s_nn_idx[:, :, 1:]

    s_nn_weight = normalize_P(s_nn_sim, sim_normalization, dim=2)

    if not s_only:
        if ignore_first:
            t_nn_sim, t_nn_idx = t_nn_sim[:, :, 1:], t_nn_idx[:, :, 1:]

        t_nn_weight = normalize_P(t_nn_sim, sim_normalization, dim=2)
    else:
        t_nn_weight = None

    return s_nn_weight, s_nn_sim, s_nn_idx, t_nn_weight, t_nn_sim, t_nn_idx


def square_distance(src, dst):
    N, _ = src.shape
    M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(1,0))     
    dist += torch.sum(src ** 2, -1).view(N, 1)
    dist += torch.sum(dst ** 2, -1).view(1, M)
    return dist

def normalize_P(P, p_normalization, dim=None):
    """
    The method to normalize the P matrix to be "like" a statistical matrix.
    
    Here we assume that P is Ny times Nx, according to coup paper the columns (per x) should be statistical, hence normalize column wise
    """
    if dim is None:
        dim = 1 if len(P.shape) == 3 else 0

    if p_normalization == "no_normalize":
        return P
    if p_normalization == "l1":
        return F.normalize(P, dim=dim, p=1)
    if p_normalization == "l2":
        return F.normalize(P, dim=dim, p=2)
    if p_normalization == "softmax":
        return F.softmax(P, dim=dim)
    raise NameError


