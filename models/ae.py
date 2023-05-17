# Learning Representations and Generative Models for 3D Point Clouds
# Source: https://github.com/optas/latent_3d_points/blob/master/src/encoders_decoders.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_utils import calc_cd
from .dgcnn import DGCNN_encoder

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.num_input = args.num_input_points
        self.latent_dim = args.latent_dim
        self.num_output = args.num_output_points
        self.train_loss = args.loss
        self.device = args.device

        if args.encoder == 'pn':
            self.encoder = PointNet_encoder(self.latent_dim)
        elif args.encoder == 'dgcnn':
            self.encoder = DGCNN_encoder(self.latent_dim)
        else:
            print("Unimplemented encoder: " + str(args.encoder))

        self.decoder = FC_decoder(self.latent_dim, self.num_output)

    def forward(self, x, gt, is_training=True):
        z, per_point_features = self.encoder(x)
        pred = self.decoder(z)
        pred = pred.transpose(1, 2).contiguous()

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

class PointNet_encoder(nn.Module):
    def __init__(self, latent_dim):
        super(PointNet_encoder, self).__init__()
        if latent_dim > 256:
            input("Warning: Bottleneck is smaller than latent dimension.")
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, 256, 1)
        self.conv5 = nn.Conv1d(256, self.latent_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(self.latent_dim)

    def forward(self, x, use_bn=True):
        x = x.transpose(2, 1)
        batch_size, _, num_points = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        per_point_features = x
        global_feature, _ = torch.max(x, 2)
        global_feature = global_feature.view(batch_size, -1)
        return global_feature, per_point_features


class FC_decoder(nn.Module):
    def __init__(self, latent_dim, num_output):
        super(FC_decoder, self).__init__()
        self.num_output = num_output
        self.inter_dim = max(latent_dim, num_output//2)
        self.fc1 = nn.Linear(latent_dim, self.inter_dim)
        self.fc2 = nn.Linear(self.inter_dim, num_output)
        self.fc3 = nn.Linear(num_output, num_output * 3)

    def forward(self, x):
        batch_size = x.size()[0]
        output = F.relu(self.fc1(x))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).view(-1, 3, self.num_output)
        return output 