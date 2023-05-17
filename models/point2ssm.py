# Proposed model - Point2SSM
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ae import PointNet_encoder
from models.dgcnn import DGCNN_encoder
from utils.model_utils import calc_cd
from torch_cluster import knn

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
        self.device = args.device
        self.alpha = args.alpha

        self.encoder_name = args.encoder

        if args.encoder == 'pn':
            self.encoder = PointNet_encoder(self.latent_dim)
        elif args.encoder == 'dgcnn':
            self.encoder = DGCNN_encoder(self.latent_dim)
        else:
            print("Unimplemented encoder: " + str(args.encoder))

        self.attn_module = Attention_Module(self.latent_dim, self.num_output)

    # all combos within batch
    def get_neighbor_loss(self, pred, k):
        edge_index = [knn(pred[i], pred[i], k,) for i in range(pred.shape[0])]
        neigh_idxs = torch.stack([edge_index[i][1].reshape(pred.shape[1], -1) for i in range(pred.shape[0])])
        batch_size = pred.shape[0]
        loss, count = 0, 0
        for source_index in range(batch_size):
            for target_index in range(batch_size):
                if source_index != target_index:
                    count += 1
                    loss += self.neighbor_loss_helper(pred[source_index].unsqueeze(0), neigh_idxs[source_index].unsqueeze(0), pred[target_index].unsqueeze(0), k)
        return loss/count

    def neighbor_loss_helper(self, source, source_neighs, target, k):
        source_grouped = grouping_operation(source.transpose(1, 2).contiguous(), source_neighs.int()).permute(0, 2, 3, 1)
        source_diff = source_grouped[:, :, 1:, :] - torch.unsqueeze(source, 2)  # remove fist grouped element, as it is the seed point itself
        source_square = torch.sum(source_diff ** 2, dim=-1)

        target_cr_grouped = grouping_operation(target.transpose(1, 2).contiguous(), source_neighs.int()).permute(0, 2, 3, 1)
        target_cr_diff = target_cr_grouped[:, :, 1:, :] - torch.unsqueeze(target, 2)  # remove fist grouped element, as it is the seed point itself
        target_cr_square = torch.sum(target_cr_diff ** 2, dim=-1)

        GAUSSIAN_HEAT_KERNEL_T = 8.0
        gaussian_heat_kernel = torch.exp(-source_square/GAUSSIAN_HEAT_KERNEL_T)
        neighbor_loss_per_neigh = torch.mul(gaussian_heat_kernel, target_cr_square)

        neighbor_loss = torch.sum(neighbor_loss_per_neigh)

        return neighbor_loss

    def forward(self, x, gt, is_training=True):
        z, features = self.encoder(x)

        if self.encoder_name == 'dgcnn':
            features = features.transpose(2,1)

        prob_map = self.attn_module(features)

        pred = torch.sum(prob_map[:, :, :, None] * x[:, None, :, :], dim=2)

        if is_training:
            cd_p, cd_t = calc_cd(pred, gt)
            if self.train_loss == 'cd_t':
                recon_loss = cd_t
            elif self.train_loss == 'cd_p':
                recon_loss = cd_p
            else:
                raise NotImplementedError('Only CD loss is supported')
            if self.alpha == None:
                self.alpha = 0 
            if self.alpha == 0:
                neigh_loss = 0
            else:
                neigh_loss = self.get_neighbor_loss(pred, 10)

            loss = recon_loss.mean() + self.alpha*neigh_loss.mean()

            return pred, loss
        else:
            cd_p, cd_t = calc_cd(pred, gt) 
            return {'recon': pred, 'cd_p': cd_p, 'cd_t': cd_t}


class Attention_Module(nn.Module):
    def __init__(self, latent_dim, num_output):
        super(Attention_Module, self).__init__()
        self.num_output = num_output
        self.latent_dim = latent_dim

        self.sa1 = cross_transformer(self.latent_dim,self.num_output)
        self.sa2 = cross_transformer(self.num_output,self.num_output)
        self.sa3 = cross_transformer(self.num_output,self.num_output)
        self.softmax = nn.Softmax(dim=2)        

    def forward(self, x):
        x = self.sa1(x,x)
        x = self.sa2(x,x)
        x = self.sa3(x,x)
        prob_map = self.softmax(x)
        return prob_map


# PointAttN: You Only Need Attention for Point Cloud Completion
# https://github.com/ohhhyeahhh/PointAttN
class cross_transformer(nn.Module):
    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)

        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)

        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        self.activation1 = torch.nn.GELU()

        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    # transformer
    def forward(self, src1, src2, if_act=False):
        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)

        b, c, _ = src1.shape

        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)

        src1 = self.norm13(src1)
        src2 = self.norm13(src2)

        src12 = self.multihead_attn1(query=src1,
                                     key=src2,
                                     value=src2)[0]

        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)

        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)

        src1 = src1.permute(1, 2, 0)

        return src1