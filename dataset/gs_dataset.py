# gs_dataset.py
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from dataset.fusion_utils import Voxelizer
from dataset.augmentation import ElasticDistortion, RandomHorizontalFlip, Compose
from utils.dataset_utils import load_gaussian_ply
import open3d as o3d

class GaussianDataset(Dataset):
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi, np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))
    ROTATION_AXIS = "z"

    def __init__(self, gaussians_dir, input_type="3dgs", gaussian_iterations=10000, voxel_size=0.02, aug=False, feature_type="all"):
        """
        Args:
            gaussians_dir (str): Point Cloud 파일이 저장된 디렉토리.
            input_type (str): "3dgs" (3DGS Point Cloud) 또는 "low" (Low Point Cloud).
            gaussian_iterations (int): 3DGS Point Cloud의 반복 횟수 (3DGS 전용).
            voxel_size (float): Voxelization 크기.
            aug (bool): 증강 여부.
            feature_type (str): 3DGS 피처 타입 ("all", "rgb", 등).
        """
        self.input_type = input_type
        self.aug = aug
        self.feature_type = feature_type
        self.scenes = os.listdir(gaussians_dir)
        self.scenes.sort()

        self.data = []
        for scene in self.scenes:
            if self.input_type == "3dgs":
                # 3DGS Point Cloud 경로
                ply_path = os.path.join(gaussians_dir, scene, "point_cloud", f"iteration_{gaussian_iterations}", "point_cloud.ply")
                label_path = os.path.join(gaussians_dir, scene, "labels.npy")
            else:
                # Low Point Cloud 경로
                ply_path = os.path.join(gaussians_dir, scene, "points3d.ply")
                label_path = os.path.join(gaussians_dir, scene, "labels.npy")
            
            if os.path.exists(ply_path) and os.path.exists(label_path):
                self.data.append([ply_path, label_path, 0])
            else:
                print(f"Warning: Skipping {scene} - Files not found: {ply_path}, {label_path}")

        self.voxelizer = Voxelizer(voxel_size=voxel_size if voxel_size is not None else None, clip_bound=None, use_augmentation=aug)
        self.prevoxel_transforms = Compose([ElasticDistortion(self.ELASTIC_DISTORT_PARAMS)]) if aug else None
        self.input_transforms = Compose([RandomHorizontalFlip(self.ROTATION_AXIS, is_temporal=False)]) if aug else None

    def __getitem__(self, index):
        with torch.no_grad():
            ply_path, label_path, head_id = self.data[index]

            if self.input_type == "3dgs":
                locs, features = load_gaussian_ply(ply_path, self.feature_type)
            else:
                pcd = o3d.io.read_point_cloud(ply_path)
                locs = np.asarray(pcd.points)  # [N, 3]
                features = np.asarray(pcd.colors)  # [N, 3] (RGB)

            labels = np.load(label_path)

            if self.aug and self.prevoxel_transforms:
                locs = self.prevoxel_transforms(locs)

            if self.voxelizer.voxel_size is not None:  # Voxelization 적용
                locs, features, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                    locs, features, labels, return_ind=True
                )
            else:  # Voxelization 미적용
                vox_ind = np.arange(len(locs))

            vox_ind = torch.from_numpy(vox_ind)
            labels = torch.from_numpy(labels).long()

            if self.aug and self.input_transforms:
                locs, features, labels = self.input_transforms(locs, features, labels)

            locs = torch.from_numpy(locs).int()
            locs = torch.cat([torch.ones(locs.shape[0], 1, dtype=torch.int), locs], dim=1)
            features = torch.from_numpy(features).float()

        return locs, features, labels, head_id

    def __len__(self):
        return len(self.data)