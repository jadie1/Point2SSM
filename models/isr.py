# Unsupervised Learning of Intrinsic Structural Representation (ISR) Points
# https://github.com/NolenChen/3DStructurePoints

import torch
import torch.nn as nn
from utils.model_utils import calc_cd

import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(proj_dir, "utils/Pointnet2.PyTorch/pointnet2"))
from pointnet2_modules import PointnetSAModuleMSG


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.num_input = args.num_input_points
        self.num_structure_points = args.num_output_points
        self.train_loss = args.loss
        self.device = args.device
        self.point_dim = 3
        self.SA_modules = nn.ModuleList()
        self.stpts_prob_map = None

        input_channels = 3
        use_xyz = True

        self.SA_module1 = PointnetSAModuleMSG(
                npoint=512,
                radii=[0.1, 0.2, 0.4],
                nsamples=[16, 32, 128],
                mlps=[
                    [input_channels, 32, 32, 64],
                    [input_channels, 64, 64, 128],
                    [input_channels, 64, 96, 128],
                ],
                use_xyz=True,
            )

        input_channels = 64 + 128 + 128
        self.SA_module2 = PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4, 0.8],
                nsamples=[32, 64, 128],
                mlps=[
                    [input_channels, 64, 64, 128],
                    [input_channels, 128, 128, 256],
                    [input_channels, 128, 128, 256],
                ],
                use_xyz=use_xyz,
                use_features=True,
            )

        conv1d_stpts_prob_modules = []
        if self.num_structure_points <= 128 + 256 + 256:
            conv1d_stpts_prob_modules.append(nn.Dropout(0.2))
            conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=128 + 256 + 256, out_channels=512, kernel_size=1))
            conv1d_stpts_prob_modules.append(nn.BatchNorm1d(512))
            conv1d_stpts_prob_modules.append(nn.ReLU())
            in_channels = 512
            while in_channels >= self.num_structure_points * 2:
                out_channels = int(in_channels / 2)
                conv1d_stpts_prob_modules.append(nn.Dropout(0.2))
                conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1))
                conv1d_stpts_prob_modules.append(nn.BatchNorm1d(out_channels))
                conv1d_stpts_prob_modules.append(nn.ReLU())
                in_channels = out_channels

            conv1d_stpts_prob_modules.append(nn.Dropout(0.2))
            conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=in_channels, out_channels=self.num_structure_points, kernel_size=1))

            conv1d_stpts_prob_modules.append(nn.BatchNorm1d(self.num_structure_points))
            conv1d_stpts_prob_modules.append(nn.Softmax(dim=2))
        else:
            conv1d_stpts_prob_modules.append(nn.Dropout(0.2))
            conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=128 + 256 + 256, out_channels=1024, kernel_size=1))
            conv1d_stpts_prob_modules.append(nn.BatchNorm1d(1024))
            conv1d_stpts_prob_modules.append(nn.ReLU())

            in_channels = 1024
            while in_channels <= self.num_structure_points / 2:
                out_channels = int(in_channels * 2)
                conv1d_stpts_prob_modules.append(nn.Dropout(0.2))
                conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1))
                conv1d_stpts_prob_modules.append(nn.BatchNorm1d(out_channels))
                conv1d_stpts_prob_modules.append(nn.ReLU())
                in_channels = out_channels

            conv1d_stpts_prob_modules.append(nn.Dropout(0.2))
            conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=in_channels, out_channels=self.num_structure_points, kernel_size=1))

            conv1d_stpts_prob_modules.append(nn.BatchNorm1d(self.num_structure_points))
            conv1d_stpts_prob_modules.append(nn.Softmax(dim=2))

        self.conv1d_stpts_prob = nn.Sequential(*conv1d_stpts_prob_modules)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud, gt, is_training=True, return_weighted_feature=False):
    # def forward(self, pointcloud, return_weighted_feature=False):
        '''
        :param pointcloud: input point cloud with shape (bn, num_of_pts, 3)
        :param return_weighted_feature: whether return features for the structure points or not
        :return:
        '''
        xyz, features = self._break_up_pc(pointcloud)
        xyz, features = self.SA_module1(xyz, features)
        xyz, features = self.SA_module2(xyz, features)


        self.stpts_prob_map = self.conv1d_stpts_prob(features)

        weighted_xyz = torch.sum(self.stpts_prob_map[:, :, :, None] * xyz[:, None, :, :], dim=2)
        if return_weighted_feature:
            weighted_features = torch.sum(self.stpts_prob_map[:, None, :, :] * features[:, :, None, :], dim=3)

        pred = weighted_xyz

        if is_training:
            cd_p, cd_t = calc_cd(pred, gt)
            if self.train_loss == 'cd_t':
                recon_loss = cd_t
            elif self.train_loss == 'cd_p':
                recon_loss = cd_p
                raise NotImplementedError('Only CD is supported')
            loss = recon_loss.mean()
            return pred, loss
        else:
            cd_p, cd_t = calc_cd(pred, gt) 
            return {'recon': pred, 'cd_p': cd_p, 'cd_t': cd_t}






 
