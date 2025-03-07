import os
import torch
import numpy as np
from torch.utils.data import Dataset
from dataset.fusion_utils import Voxelizer
from dataset.augmentation import ElasticDistortion, RandomHorizontalFlip, Compose
from utils.dataset_utils import load_gaussian_ply
import open3d as o3d

class GaussianDataset(Dataset):
    def __init__(self, gaussians_dir, label_dir, gaussian_iterations=10000, voxel_size=0.02, aug=False, feature_type="all"):
        self.aug = aug
        self.feature_type = feature_type
        self.scenes = os.listdir(gaussians_dir)
        self.scenes.sort()

        self.data = []
        for scene in self.scenes:
            ply_path = os.path.join(gaussians_dir, scene, "point_cloud", f"iteration_{gaussian_iterations}", "point_cloud.ply")
            label_path = os.path.join(label_dir, scene, "final_labels.npy")
            self.data.append([ply_path, label_path, 0])

        self.voxelizer = Voxelizer(voxel_size=voxel_size, clip_bound=None, use_augmentation=aug)
        self.prevoxel_transforms = Compose([ElasticDistortion(self.ELASTIC_DISTORT_PARAMS)])
        self.input_transforms = Compose([RandomHorizontalFlip(self.ROTATION_AXIS, is_temporal=False)])

    def __getitem__(self, index):
        with torch.no_grad():
            ply_path, label_path, head_id = self.data[index]
            locs, features = load_gaussian_ply(ply_path, self.feature_type)
            labels = np.load(label_path)  # 3DGS 매핑된 레이블

            # 증강
            if self.aug:
                locs = self.prevoxel_transforms(locs)

            # Voxelization
            locs, features, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                locs, features, labels, return_ind=True
            )

            vox_ind = torch.from_numpy(vox_ind)
            labels = torch.from_numpy(labels).long()

            if self.aug:
                locs, features, labels = self.input_transforms(locs, features, labels)

            locs = torch.from_numpy(locs).int()
            locs = torch.cat([torch.ones(locs.shape[0], 1, dtype=torch.int), locs], dim=1)
            features = torch.from_numpy(features).float()

        return locs, features, labels, head_id

    def __len__(self):
        return len(self.data)

# 모델 수정 예시
model_3d = mink_unet(in_channels=56, out_channels=20, D=3, arch=config.distill.model_3d).cuda()