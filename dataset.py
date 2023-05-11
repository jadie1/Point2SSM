import os
import torch 
import torch.utils.data as data
import numpy as np
import pyvista as pv
import random

from utils.model_utils import calc_cd
import pointnet2_cuda as pointnet2

class MeshDataset(data.Dataset):
    def __init__(self, args, set_type='test', scale_factor=None, sigma_squared=None):
        self.num_points = args.num_input_points
        self.mesh_dir = os.path.join('data', args.dataset, set_type+'_meshes/')
        self.missing_percent = args.missing_percent
        self.noise_level = args.noise_level
        if not self.noise_level:
            self.add_noise = False
        else:
            self.add_noise = True
        self.subsample = args.train_subset_size
        self.set_type = set_type

        self.point_sets = []
        self.names = []
        
        calc_scale_factor = 0
        min_points = 1e8
        for file in sorted(os.listdir(self.mesh_dir)):
            points = np.array(pv.read(self.mesh_dir+file).points)
            if np.max(np.abs(points)) > calc_scale_factor:
                calc_scale_factor = np.max(np.abs(points))
            if points.shape[0] < min_points:
                min_points = points.shape[0]
            self.point_sets.append(points)
            self.names.append(file.replace(".vtk",""))
        self.min_points = min_points

        if not scale_factor:
            self.scale_factor = float(calc_scale_factor)
        else:
            self.scale_factor = scale_factor

        if self.subsample != None and set_type=='train':
            if os.path.exists(self.mesh_dir + "../importance_sampling_indices.npy"):
                print("Using importance sampling.")
                sorted_indices = np.load(self.mesh_dir + "../importance_sampling_indices.npy")
                indices = sorted_indices[:int(self.subsample)]
                pts, nms = [], []
                for index in indices:
                    pts.append(self.point_sets[index])
                    nms.append(self.names[index])
            else:
                pts, nms = self.point_sets[:int(self.subsample)], self.names[:int(self.subsample)]
            self.point_sets = pts
            self.names = nms
        
        if self.add_noise and not sigma_squared:
            self.sigma_squared = self.calc_sigma_squared()
        else:
            self.sigma_squared = 0
            
    def get_scale_factor(self):
        return self.scale_factor

    def get_sigma_squared(self):
        return self.sigma_squared

    def calc_sigma_squared(self):
        cd_matrix = np.eye(len(self.point_sets))
        for i in range(len(self.point_sets)):
            for j in range(len(self.point_sets)):
                if i != j:
                    pc1 = torch.FloatTensor(self.point_sets[i]/ self.scale_factor)[None, :].cuda()
                    pc2 = torch.FloatTensor(self.point_sets[j]/ self.scale_factor)[None, :].cuda()
                    cd_l2, cd_l1 = calc_cd(pc1, pc2)
                    cd_matrix[i][j] = cd_l1.item()
        nn_indices = cd_matrix.argmin(axis=1)
        nearest_neighbors = []
        for i in range(len(self.point_sets)):
            nearest_neighbors.append(cd_matrix[i][nn_indices[i]])
        mean_nn = np.mean(nearest_neighbors)
        sigma_squared = mean_nn/(self.point_sets[0].shape[0]*self.point_sets[0].shape[1])
        print('sigma_squared', sigma_squared)
        return sigma_squared

    def __getitem__(self, index):
        full_point_set = self.point_sets[index]
        name = self.names[index]
        full_point_set = full_point_set / self.scale_factor
        
        # add missingness
        if not self.missing_percent or self.missing_percent == 0:
            partial_point_set = full_point_set
        else:
            if self.set_type == 'train':
                seed = np.random.randint(len(full_point_set))
            else:
                seed = 0 # consistent testing
            distances = np.linalg.norm(full_point_set - full_point_set[seed], axis=1)
            sorted_points = full_point_set[np.argsort(distances)]
            partial_point_set = sorted_points[int(len(full_point_set)*self.missing_percent):]

        # select subset
        if self.num_points > len(partial_point_set):
            replace = True
        else: 
            replace = False
        choice = np.random.choice(len(partial_point_set), self.num_points, replace=replace)
        partial = torch.FloatTensor(partial_point_set[choice, :])
        
        # add noise
        if self.add_noise:
            partial = partial + (self.sigma_squared**0.5)*torch.randn(partial.shape)
        
        # ground truth 
        choice = np.random.choice(len(full_point_set), self.min_points, replace=False)
        gt = torch.FloatTensor(full_point_set[choice, :])
        
        return partial, gt, name

    def __len__(self):
        return len(self.point_sets)

'''
If ref path is none it will use a random refs
'''
class DPC_Dataset(data.Dataset):
    def __init__(self, args, set_type='test', scale_factor=None, sigma_squared=None, ref_path=None):
        self.num_points = args.num_input_points
        self.mesh_dataset = MeshDataset(args, set_type, scale_factor, sigma_squared)
        self.scale_factor = self.mesh_dataset.scale_factor
        self.sigma_squared = self.mesh_dataset.sigma_squared
        if ref_path:
            ref_points = np.array(pv.read(ref_path).points)
            target_pc = torch.FloatTensor(ref_points/ self.scale_factor).to('cuda:0')
            self.target_pc = furthest_point_downsampling(target_pc[None,:], self.num_points).squeeze()
        else:
            self.target_pc = None
            
    def get_scale_factor(self):
        return self.scale_factor

    def get_sigma_squared(self):
        return self.sigma_squared

    def __getitem__(self, index):
        source_pc, source_gt, source_name = self.mesh_dataset.__getitem__(index)
        if self.target_pc == None:
            choices = list(range(0,index)) + list(range(index+1, len(self.mesh_dataset.point_sets)))
            target_index = random.choice(choices)
            target_pc, target_gt, target_name = self.mesh_dataset.__getitem__(target_index)
        else:
            target_pc = self.target_pc
        return source_pc, target_pc, source_gt, source_name

    def __len__(self):
        return len(self.mesh_dataset.point_sets)


def furthest_point_downsampling(points, npoint):
    xyz = points.contiguous()
    B, N, _ = xyz.size()
    output = torch.IntTensor(B, npoint).to(xyz.device)
    temp = torch.FloatTensor(B, N).fill_(1e10).to(xyz.device)
    pointnet2.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
    indices = output.cpu().numpy()
    subset = []
    for i in range(points.shape[0]):
        subset.append(points[i][indices[i], :].cpu().numpy())
    subset = np.array(subset)
    return torch.FloatTensor(subset).to('cuda:0')
