import open3d as o3d
import numpy as np

input_ply = "/home/knuvi/Desktop/song/semantic-gaussians/dataset/scannet/scannet_samples_pre/scene0248_02/points3d.ply"
pcd = o3d.io.read_point_cloud(input_ply)
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors) * 255.0  # 0-1 → 0-255 변환

nan_count = np.sum(np.isnan(points))
inf_count = np.sum(np.isinf(points))
total_points = points.shape[0]
print(f"Total points: {total_points}")
print(f"NaN points: {nan_count}")
print(f"Inf points: {inf_count}")
if nan_count > 0 or inf_count > 0:
    print("⚠️ Invalid values detected!")
else:
    print("✅ No invalid values.")