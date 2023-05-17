# Source: https://github.com/paul007pl/VRCNet/blob/main/utils/model_utils.py
import torch
import math
import os
import sys
import torch.nn as nn
import torch.nn.functional as F

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(proj_dir, "utils/emd"))
import emd_module as emd
sys.path.append(os.path.join(proj_dir, "utils/ChamferDistancePytorch"))
from chamfer3D import dist_chamfer_3D
from fscore import fscore


def calc_cd(output, gt, calc_f1=False, f1_threshold=0.0001):
    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, _, _ = cham_loss(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    # cd_t = (dist1.mean(1) + dist2.mean(1))
    cd_t = (dist1.sum(1) + dist2.sum(1))
    if calc_f1:
        f1, _, _ = fscore(dist1, dist2, f1_threshold)
        return cd_p, cd_t, f1
    else:
        return cd_p, cd_t


def calc_emd(output, gt, eps=0.005, iterations=50):
    emd_loss = emd.emdModule()
    dist, _ = emd_loss(output, gt, eps, iterations)
    emd_out = torch.sqrt(dist).mean(1)
    return emd_out


def calc_cd_mean(output, gt, calc_f1=False, f1_threshold=0.0001):
    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, _, _ = cham_loss(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = (dist1.mean(1) + dist2.mean(1))
    # cd_t = (dist1.sum(1) + dist2.sum(1))
    if calc_f1:
        f1, _, _ = fscore(dist1, dist2, f1_threshold)
        return cd_p, cd_t, f1
    else:
        return cd_p, cd_t