import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ae import PointNet_encoder
from models.dgcnn import DGCNN_encoder
from utils.model_utils import calc_cd

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.num_input = args.num_input_points
        self.latent_dim = args.latent_dim
        self.num_output = args.num_output_points
        self.train_loss = args.loss
        self.device = args.device

        self.encoder_name = args.encoder

        if args.encoder == 'pn':
            self.encoder = PointNet_encoder(self.latent_dim)
        elif args.encoder == 'dgcnn':
            self.encoder = DGCNN_encoder(self.latent_dim)
        else:
            print("Unimplemented encoder: " + str(args.encoder))

        self.attn_module = Attention_Module(self.latent_dim, self.num_output)

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
            loss = recon_loss.mean()
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