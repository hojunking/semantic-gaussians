import open3d as o3d
import numpy as np
import plyfile
from pathlib import Path
import os

# ScanNet20 매핑
SCANNET_LABELS_MAP = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 12: 11, 14: 12, 16: 13, 24: 14, 28: 15, 33: 16, 34: 17, 36: 18, 39: 19
}

def remap_labels(labels):
    return np.array([SCANNET_LABELS_MAP[label] if label in SCANNET_LABELS_MAP else 255 for label in labels])

def map_labels_to_ply(input_ply_path, scannet_ply_path, scannet_labels_path, output_labels_path):
    # 입력 포인트 클라우드 로드
    input_pcd = o3d.io.read_point_cloud(input_ply_path)
    input_points = np.asarray(input_pcd.points)

    # ScanNet Mesh 로드
    scannet_pcd = o3d.io.read_point_cloud(scannet_ply_path)
    scannet_points = np.asarray(scannet_pcd.points)

    # ScanNet 레이블 로드 및 매핑
    with open(scannet_labels_path, "rb") as f:
        plydata = plyfile.PlyData.read(f)
        scannet_labels = np.asarray(plydata['vertex']['label'])
    scannet_labels = remap_labels(scannet_labels)

    # KDTree 생성
    kdtree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(scannet_points)))

    # 레이블 매핑
    labels = []
    for point in input_points:
        _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
        labels.append(scannet_labels[idx[0]])
    labels = np.array(labels)

    # 저장
    output_dir = os.path.dirname(output_labels_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    np.save(output_labels_path, labels)

    print(f"✅ Saved {len(labels)} labels to {output_labels_path}")
    print(f"🎯 Unique labels after mapping: {np.unique(labels)}")
    return labels

# 실행 예시
input_ply_path = "/home/knuvi/Desktop/song/semantic-gaussians/dataset/scannet/scannet_samples_pre/train/scene0017_00/points3d.ply"
scannet_ply_path = "/home/knuvi/Desktop/song/data/scannet_sample/train/scene0017_00/scene0017_00_vh_clean_2.ply"
scannet_labels_path = "/home/knuvi/Desktop/song/data/scannet_sample/train/scene0017_00/scene0017_00_vh_clean_2.labels.ply"
output_labels_path = "/home/knuvi/Desktop/song/semantic-gaussians/dataset/scannet/scannet_samples_pre/train/scene0017_00/init_labels.npy"

labels = map_labels_to_ply(input_ply_path, scannet_ply_path, scannet_labels_path, output_labels_path)

# 클래스 분포 확인
print(f"Scene: scene0017_00, Unique labels after mapping: {np.unique(labels)}")
print(f"Label distribution: {np.bincount(labels[labels != 255])}")