import open3d as o3d
import numpy as np
import plyfile

# ScanNet20 매핑
SCANNET_LABELS_MAP = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 12: 11, 14: 12, 16: 13, 24: 14, 28: 15, 33: 16, 34: 17, 36: 18, 39: 19
}

def remap_labels(labels):
    return np.array([SCANNET_LABELS_MAP[label] if label in SCANNET_LABELS_MAP else 255 for label in labels])

# 초기 포인트 레이블 생성
init_pcd = o3d.io.read_point_cloud("/home/knuvi/Desktop/song/semantic-gaussians/dataset/scannet/scannet_samples_pre/train/scene0017_00/points3d.ply")
init_points = np.asarray(init_pcd.points)
scannet_pcd = o3d.io.read_point_cloud("/home/knuvi/Desktop/song/data/scannet_sample/train/scene0017_00/scene0017_00_vh_clean_2.ply")
scannet_points = np.asarray(scannet_pcd.points)

with open("/home/knuvi/Desktop/song/data/scannet_sample/train/scene0017_00/scene0017_00_vh_clean_2.labels.ply", "rb") as f:
    plydata = plyfile.PlyData.read(f)
    scannet_labels = np.asarray(plydata['vertex']['label'])

# ScanNet20 매핑
scannet_labels = remap_labels(scannet_labels)

kdtree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(scannet_points)))
init_labels = []
for point in init_points:
    _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
    init_labels.append(scannet_labels[idx[0]])
init_labels = np.array(init_labels)

# 3DGS 레이블과 비교
final_labels = np.load("/home/knuvi/Desktop/song/semantic-gaussians/output/train_samples/scene0017_00/final_labels.npy")
gaussians_pcd = o3d.io.read_point_cloud("/home/knuvi/Desktop/song/semantic-gaussians/output/train_samples/scene0017_00/point_cloud/iteration_10000/point_cloud.ply")
gaussians_points = np.asarray(gaussians_pcd.points)

kdtree_init = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(init_points)))
matches = 0
total_valid = 0
for i, point in enumerate(gaussians_points):
    _, idx, _ = kdtree_init.search_knn_vector_3d(point, 1)
    init_label = init_labels[idx[0]]
    final_label = final_labels[i]
    # 255 제외하고 비교
    if init_label != 255 and final_label != 255:
        total_valid += 1
        if init_label == final_label:
            matches += 1

# 정확도 계산 (255 제외)
if total_valid > 0:
    accuracy = matches / total_valid * 100
    print(f"Label mapping accuracy (excluding 255): {accuracy:.2f}%")
    print(f"Total valid points (excluding 255): {total_valid}")
    print(f"Matches: {matches}")
else:
    print("No valid points for comparison (all labels are 255).")