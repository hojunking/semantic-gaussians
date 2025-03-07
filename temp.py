
import open3d as o3d
import numpy as np



# pcd = o3d.io.read_point_cloud("/home/knuvi/Desktop/song/semantic-gaussians/dataset/scannet/scans/scene0000_00/scene0000_00_vh_clean_2.labels.ply")
# print(pcd)  # PointCloud 속성 출력
# print(pcd.has_colors())  # colors 속성 여부
# print(pcd.has_normals())  # normals 속성 여부
# print(pcd.has_vertex_colors())  # vertex_colors (colors) 여부

# # colors에서 레이블 추출 (색상으로 인코딩된 경우)
# if pcd.has_colors():
#     colors = np.asarray(pcd.colors)
#     # ScanNet 레이블은 일반적으로 colors[0] * 255로 인코딩 (0~19 또는 0~199)
#     raw_labels = (colors[:, 0] * 255).astype(np.uint8)
#     print(f"Raw labels: {raw_labels}")
# else:
#     print("No colors found, check for 'label' attribute.")

import plyfile

# with open("/home/knuvi/Desktop/song/semantic-gaussians/dataset/scannet/scans/scene0000_00/scene0000_00_vh_clean_2.labels.ply", "rb") as f:
#     plydata = plyfile.PlyData.read(f)
#     if 'label' in plydata['vertex'].data.dtype.names:
#         raw_labels = np.asarray(plydata['vertex']['label'])
#         print(f"Labels from 'label' property: {raw_labels}")
#     else:
#         print("No 'label' property found, check colors or other attributes.")


def count_points_in_ply(ply_path):
    """
    Reads a .ply file and returns the number of points in the point cloud.
    
    Args:
        ply_path (str): Path to the .ply file.
    
    Returns:
        int: Number of points in the .ply file.
    """
    # Load PLY file
    ply_data = plyfile.PlyData.read(ply_path)

    # Get the number of points
    num_points = len(ply_data["vertex"])
    
    print(f"Number of points in {ply_path}: {num_points}")
    return num_points

# Example usage
ply_file_path = "/home/knuvi/Desktop/song/semantic-gaussians/output/scene0017_00/point_cloud/iteration_10000/point_cloud.ply"
count_points_in_ply(ply_file_path)