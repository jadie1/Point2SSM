# Learning 3D Dense Correspondence via Canonical Point Autoencoder
# Source: https://github.com/AnjieCheng/CanonicalPAE

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_utils import calc_cd, calc_emd
from torch.autograd import Variable
import os
import math
import scipy

criterion = torch.nn.MSELoss()

import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(proj_dir, "utils/emd"))
import emd_module as emd
sys.path.append(os.path.join(proj_dir, "utils/ChamferDistancePytorch"))
from chamfer3D import dist_chamfer_3D

def EMD_loss(esti_shapes, shapes):
    emd_dist = emd.emdModule()
    dist, assigment = emd_dist(esti_shapes, shapes, 0.005, 50)
    loss_emd = torch.sqrt(dist).mean(1).mean()
    return loss_emd

def CD_loss(esti_shapes, shapes):
    cf_dist = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, idx1, idx2 = cf_dist(esti_shapes, shapes)
    loss_cd = torch.mean(torch.sqrt(dist1)) + torch.mean(torch.sqrt(dist2))
    return loss_cd

def unfold_loss(esti_shapes, shapes, full_loss=False):
    cf_dist = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, idx1, idx2 = cf_dist(esti_shapes, shapes) # idx1[16, 2048] idx2[16, 2562]
    if full_loss:
        loss_cd = torch.mean(torch.sqrt(dist1)) + torch.mean(torch.sqrt(dist2))
    else:
        loss_cd = torch.mean(torch.sqrt(dist1))
    return loss_cd

def selfrec_loss(esti_shapes, shapes):
    return criterion(esti_shapes, shapes)

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.num_input = args.num_input_points
        self.latent_dim = args.latent_dim
        self.num_output = args.num_output_points
        self.train_loss = args.loss
        self.device = args.device
        self.loss_unfold_lambda = args.loss_unfold_lambda
        self.loss_mse_lambda = args.loss_mse_lambda
        self.loss_cd_lambda = args.loss_cd_lambda
        self.loss_emd_lambda = args.loss_emd_lambda
        self.loss_cross_lambda = args.loss_cross_lambda

        self.template = SphereTemplate(self.num_input, device=self.device)
        self.grid = self.template.get_regular_points(device=self.device).squeeze().transpose(0,1)

        self.encoder = PointNetfeat(npoint=self.num_input, c_dim=self.latent_dim)
        self.fold = ImplicitFun(z_dim=self.latent_dim)
        self.unfold = ImplicitFun(z_dim=self.latent_dim)

    def chamfer_loss(self, pred, gt):
        cd_p, cd_t = calc_cd(pred, gt)
        if self.train_loss == 'cd_t':
            return cd_t
        elif self.train_loss == 'cd_p':
            return cd_p

    # https://github.com/AnjieCheng/CanonicalPAE/blob/dc1806412050f4ae9f532fe5f246b04c90fcca4d/correspondence/unfoldnet/training.py#L192
    def forward(self, x, gt, is_training=True, epoch=0):
        # get features
        z = self.encoder(x)
        # unfold
        unfold_pts = self.unfold(z, x)
        # fold
        self_rec_shape = self.fold(z, unfold_pts)

        if is_training:
            # generate template
            batch_p_2d = batch_sample_from_2d_grid(self.grid, self.num_input, batch_size=x.shape[0], without_sample=True)
            # unfold loss
            use_full_loss = False if epoch >= 100 else True
            unfold_loss_alpha = 20 if epoch >= 100 else 10
            loss_unfold = unfold_loss_alpha*CD_loss(unfold_pts, batch_p_2d).mean()
            # recon loss
            loss_recon = 1000*selfrec_loss(self_rec_shape, x) + 10*CD_loss(self_rec_shape, x) + EMD_loss(self_rec_shape, x)
            # cross loss 
            if epoch >= 100:
                # cross-reconstruction loss
                cross_unfold_pts = torch.cat((unfold_pts[1:,:,:], unfold_pts[:1,:,:]), dim=0)
                cross_rec_shapes = self.fold(z, cross_unfold_pts)
                loss_cr = self.loss_cross_lambda*self.chamfer_loss(cross_rec_shapes, x).mean()
                loss_cr = 10 * CD_loss(cross_rec_shapes, x)
                loss = loss_unfold.mean() + loss_recon.mean() + loss_cr.mean() 
            else:
                loss = loss_unfold.mean() + loss_recon.mean()
            
            return self_rec_shape, loss
        else:
            cd_p, cd_t = calc_cd(self_rec_shape, gt) 
            return {'recon': self_rec_shape, 'cd_p': cd_p, 'cd_t': cd_t}

class PointNetfeat(nn.Module):
    def __init__(self, npoint = 2500, c_dim = 512):
        """Encoder""" 
        super(PointNetfeat, self).__init__()
        nlatent = c_dim
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, nlatent, 1)
        self.lin = nn.Linear(nlatent, nlatent)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(nlatent)
        self.bn4 = torch.nn.BatchNorm1d(nlatent)

        self.npoint = npoint
        self.nlatent = nlatent

    def forward(self, x):
        x = x.transpose(2,1).contiguous()
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, self.nlatent)
        x = F.relu(self.bn4(self.lin(x).unsqueeze(-1)))
        return x[...,0]

class ImplicitFun(nn.Module):
    def __init__(self, z_dim=256, num_branches=12):
        super(ImplicitFun, self).__init__()
        input_dim = z_dim+3

        self.unfold1 = mlpAdj(nlatent=input_dim)
        self.unfold2 = mlpAdj(nlatent=input_dim)

    def forward(self, z, points):

        num_pts = points.shape[1]
        z = z.unsqueeze(1).repeat(1, num_pts, 1)
        pointz = torch.cat((points, z), dim=2).float()

        x1 = self.unfold1(pointz)
        x2 = torch.cat((x1, z), dim=2)
        x3 = self.unfold2(x2)

        return x3


class mlpAdj(nn.Module):
    def __init__(self, nlatent = 1024):
        """Atlas decoder"""

        super(mlpAdj, self).__init__()
        self.nlatent = nlatent
        self.conv1 = torch.nn.Conv1d(self.nlatent, self.nlatent, 1)
        self.conv2 = torch.nn.Conv1d(self.nlatent, self.nlatent//2, 1)
        self.conv3 = torch.nn.Conv1d(self.nlatent//2, self.nlatent//4, 1)
        self.conv4 = torch.nn.Conv1d(self.nlatent//4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.nlatent)
        self.bn2 = torch.nn.BatchNorm1d(self.nlatent//2)
        self.bn3 = torch.nn.BatchNorm1d(self.nlatent//4)

    def forward(self, x):
        batchsize = x.size()[0]
        x = x.transpose(2,1).contiguous()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x.transpose(2,1)


class Template(object):
    def get_random_points(self):
        print("Please implement get_random_points ")

    def get_regular_points(self):
        print("Please implement get_regular_points ")

class SphereTemplate(Template):
    def __init__(self,  num_input, device=0, grain=6):
        self.device = device
        self.dim = 3
        self.npoints = 0
        self.num_input = num_input

    def get_random_points(self, shape, device="gpu0"):
        """
        Get random points on a Sphere
        Return Tensor of Size [x, 3, x ... x]
        """
        assert shape[1] == 3, "shape should have 3 in dim 1"
        rand_grid = torch.cuda.FloatTensor(shape).to(device).float()
        rand_grid.data.normal_(0, 1)
        rand_grid = rand_grid / torch.sqrt(torch.sum(rand_grid ** 2, dim=1, keepdim=True))
        return Variable(rand_grid) / 2

    def get_regular_points(self, npoints=None, device="gpu0"):
        """
        Get regular points on a Sphere
        Return Tensor of Size [x, 3]
        """
        if not self.npoints == npoints:
            # self.mesh = pymesh.meshutils.generate_icosphere(1, [0, 0, 0], 5)  # 2562 vertices
            # self.vertex = torch.from_numpy(self.mesh.vertices).to(device).float()
            # self.num_vertex = self.vertex.size(0)
            # self.vertex = self.vertex.transpose(0,1).contiguous().unsqueeze(0)
            # self.npoints = npoints

            # https://arxiv.org/pdf/0912.4540.pdf
            points = []
            phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

            for i in range(self.num_input):
                y = 1 - (i / float(self.num_input - 1)) * 2  # y goes from 1 to -1
                radius = math.sqrt(1 - y * y)  # radius at y

                theta = phi * i  # golden angle increment

                x = math.cos(theta) * radius
                z = math.sin(theta) * radius

                points.append((x, y, z))
            vertex = torch.from_numpy(np.array(points)).to(device).float()
            self.vertex = vertex.transpose(0,1).contiguous().unsqueeze(0)
            self.num_vertex = npoints
            self.npoints = npoints

        return Variable(self.vertex.to(device)) / 2

def batch_sample_from_2d_grid(grid, K, batch_size=1, without_sample=False):
    grid_size = grid.shape[0]

    grid = grid.unsqueeze(0)
    grid = grid.expand(batch_size, -1, -1) # BxNx2

    if without_sample:
        return grid
        
    assert(grid_size >= K)
    idx = torch.randint(
        low=0, high=grid_size,
        size=(batch_size, K),
    )

    idx, _ = torch.sort(idx, 1)

    idx = idx[:, :, None].expand(batch_size, K, 2)

    sampled_points = torch.gather(grid, dim=1, index=idx)
    assert(sampled_points.size() == (batch_size, K, 2))
    return sampled_points