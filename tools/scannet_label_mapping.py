import os
import open3d as o3d
import numpy as np
import plyfile
from tqdm import tqdm

# ScanNet 20개 클래스 매핑
SCANNET_LABELS_MAP = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 12: 11, 14: 12, 16: 13, 24: 14, 28: 15, 33: 16, 34: 17, 36: 18, 39: 19
}

def remap_labels(labels):
    return np.array([SCANNET_LABELS_MAP[label] if label in SCANNET_LABELS_MAP else 255 for label in labels])

def map_labels_3dgs_to_scannet(gaussians_ply_path, scannet_ply_path, scannet_labels_path, output_labels_path):
    gaussians_pcd = o3d.io.read_point_cloud(gaussians_ply_path)
    gaussians_points = np.asarray(gaussians_pcd.points)

    scannet_pcd = o3d.io.read_point_cloud(scannet_ply_path)
    scannet_points = np.asarray(scannet_pcd.points)

    with open(scannet_labels_path, "rb") as f:
        plydata = plyfile.PlyData.read(f)
        scannet_labels = np.asarray(plydata['vertex']['label'])

    kdtree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(scannet_points)))
    labels = []
    for point in gaussians_points:
        _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
        labels.append(scannet_labels[idx[0]])
    labels = np.array(labels)

    # 레이블을 20개 클래스로 변환
    mapped_labels = remap_labels(labels)

    output_dir = os.path.dirname(output_labels_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    np.save(output_labels_path, mapped_labels)

    print(f"✅ Saved {len(mapped_labels)} labels to {output_labels_path}")
    print(f"🎯 Unique labels after mapping: {np.unique(mapped_labels)}")
    return mapped_labels

# 전체 샘플 데이터 처리
gaussians_dir = "/home/knuvi/Desktop/song/semantic-gaussians/dataset/scannet/scannet_samples_pre/valid"
scannet_dir = "/home/knuvi/Desktop/song/data/scannet_sample/valid"
scenes = os.listdir(scannet_dir)
scenes.sort()

for scene in tqdm(scenes, desc="Processing Scenes"):
    #gaussians_ply_path = os.path.join(gaussians_dir, scene, "point_cloud", "iteration_10000", "point_cloud.ply")
    gaussians_ply_path = os.path.join(gaussians_dir, scene, "points3d.ply")
    
    scannet_ply_path = os.path.join(scannet_dir, scene, f"{scene}_vh_clean_2.ply")
    scannet_labels_path = os.path.join(scannet_dir, scene, f"{scene}_vh_clean_2.labels.ply")
    output_labels_path = os.path.join(gaussians_dir, scene, "labels.npy")

    if not os.path.exists(gaussians_ply_path):
        print(f"Warning: {gaussians_ply_path} does not exist. Skipping...")
        continue
    if not os.path.exists(scannet_ply_path) or not os.path.exists(scannet_labels_path):
        print(f"Warning: ScanNet files for {scene} do not exist. Skipping...")
        continue

    labels = map_labels_3dgs_to_scannet(gaussians_ply_path, scannet_ply_path, scannet_labels_path, output_labels_path)

    # 클래스 분포 확인
    print(f"Scene: {scene}, Unique labels after mapping: {np.unique(labels)}")
    print(f"Label distribution: {np.bincount(labels[labels != 255])}")  # 255 제외하고 카운트
