import os
import sys
import cv2
import struct

import numpy as np
import open3d as o3d
import plyfile
from pathlib import Path
from tqdm import tqdm
import argparse
import zlib
import imageio
# ScanNet20 매핑
SCANNET_LABELS_MAP = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 12: 11, 14: 12, 16: 13, 24: 14, 28: 15, 33: 16, 34: 17, 36: 18, 39: 19
}

def remap_labels(labels):
    return np.array([SCANNET_LABELS_MAP[label] if label in SCANNET_LABELS_MAP else 255 for label in labels])

COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {
    -1: "unknown",
    0: "raw_ushort",
    1: "zlib_ushort",
    2: "occi_ushort",
}

class RGBDFrame:
    def load(self, file_handle):
        self.camera_to_world = np.asarray(struct.unpack("f" * 16, file_handle.read(16 * 4)), dtype=np.float32).reshape(4, 4)
        self.timestamp_color = struct.unpack("Q", file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack("Q", file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.color_data = b"".join(struct.unpack("c" * self.color_size_bytes, file_handle.read(self.color_size_bytes)))
        self.depth_data = b"".join(struct.unpack("c" * self.depth_size_bytes, file_handle.read(self.depth_size_bytes)))

    def decompress_depth(self, compression_type):
        if compression_type == "zlib_ushort":
            return self.decompress_depth_zlib()
        else:
            raise

    def decompress_depth_zlib(self):
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        if compression_type == "jpeg":
            return self.decompress_color_jpeg()
        else:
            raise

    def decompress_color_jpeg(self):
        return imageio.v2.imread(self.color_data)

class SensorData:
    def __init__(self, filename):
        self.version = 4
        self.load(filename)

    def load(self, filename):
        with open(filename, "rb") as f:
            version = struct.unpack("I", f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack("Q", f.read(8))[0]
            self.sensor_name = b"".join(struct.unpack("c" * strlen, f.read(strlen)))
            self.intrinsic_color = np.asarray(struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_color = np.asarray(struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.intrinsic_depth = np.asarray(struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_depth = np.asarray(struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack("i", f.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack("i", f.read(4))[0]]
            self.color_width = struct.unpack("I", f.read(4))[0]
            self.color_height = struct.unpack("I", f.read(4))[0]
            self.depth_width = struct.unpack("I", f.read(4))[0]
            self.depth_height = struct.unpack("I", f.read(4))[0]
            self.depth_shift = struct.unpack("f", f.read(4))[0]
            num_frames = struct.unpack("Q", f.read(8))[0]
            self.frames = []
            for i in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)

    def depth_to_point_cloud(self, depth_map, K, c2w):
        """Depth map을 3D 포인트 클라우드로 변환"""
        h, w = depth_map.shape
        i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
        x = (i - K[0, 2]) * depth_map / K[0, 0]
        y = (j - K[1, 2]) * depth_map / K[1, 1]
        z = depth_map
        points = np.stack((x, y, z, np.ones_like(z)), axis=-1).reshape(-1, 4)
        points_world = (c2w @ points.T).T[:, :3]
        valid = (z > 0).flatten() & np.all(np.isfinite(points_world), axis=-1)
        return points_world[valid]

    def export_point_cloud(self, output_path, target_size=(648, 484), frame_skip=1, voxel_size=0.02):
        """Depth map과 Pose를 사용해 3D 포인트 클라우드 저장 및 레이블 매핑"""
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(f"Exporting 3D Point Cloud with target_size={target_size}, frame_skip={frame_skip}, voxel_size={voxel_size}...")

        all_points = []
        all_colors = []
        K = self.intrinsic_depth

        for f in tqdm(range(0, len(self.frames), frame_skip), desc="Processing Frames"):
            frame = self.frames[f]
            depth = np.frombuffer(frame.decompress_depth(self.depth_compression_type), 
                                dtype=np.uint16).reshape(self.depth_height, self.depth_width)
            depth = depth.astype(np.float32) / 1000.0
            depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)
            
            color = frame.decompress_color(self.color_compression_type)
            color = cv2.resize(color, target_size, interpolation=cv2.INTER_NEAREST)
            c2w = frame.camera_to_world
            
            points = self.depth_to_point_cloud(depth, K, c2w)
            if points.size == 0:
                print(f"Warning: Frame {f} generated no valid points")
                continue
            
            colors = color.reshape(-1, 3)[(depth > 0).flatten()]
            all_points.append(points)
            all_colors.append(colors)

        if not all_points:
            print("Error: No valid points generated")
            return

        point_cloud = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)

        # Open3D로 저장 및 다운샘플링
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        
        nan_count = np.sum(np.isnan(np.asarray(pcd.points)))
        if nan_count > 0:
            print(f"Error: {nan_count} NaN points remain after downsampling")
            return
        
        output_file = os.path.join(output_path, "low_points3d.ply")
        o3d.io.write_point_cloud(output_file, pcd)
        print(f"✅ Saved {len(np.asarray(pcd.points))} points to {output_file}")

        # 레이블 매핑
        scene_name = os.path.basename(output_path)
        scannet_ply_path = f"/home/knuvi/Desktop/song/data/scannet_sample/train/scene0017_00/scene0017_00_vh_clean_2.ply"
        scannet_labels_path = f"/home/knuvi/Desktop/song/data/scannet_sample/train/scene0017_00/scene0017_00_vh_clean_2.labels.ply"
        output_labels_path = f"/home/knuvi/Desktop/song/semantic-gaussians/output/train_samples/scene0017_00/raw_labels.npy"

        if not os.path.exists(scannet_ply_path) or not os.path.exists(scannet_labels_path):
            print(f"Warning: ScanNet files for {scene_name} do not exist. Skipping label mapping...")
            return

        # 포인트 클라우드 로드
        points = np.asarray(pcd.points)

        # ScanNet Mesh 로드
        scannet_pcd = o3d.io.read_point_cloud(scannet_ply_path)
        scannet_points = np.asarray(scannet_pcd.points)

        # ScanNet 레이블 로드 및 매핑
        with open(scannet_labels_path, "rb") as f:
            plydata = plyfile.PlyData.read(f)
            scannet_labels = np.asarray(plydata['vertex']['label'])
        scannet_labels = remap_labels(scannet_labels)

        # KDTree 생성 및 레이블 매핑
        kdtree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(scannet_points)))
        labels = []
        for point in points:
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
        print(f"255 proportion in raw_labels: {(labels == 255).mean() * 100:.2f}%")
        print(f"Label distribution (excluding 255): {np.bincount(labels[labels != 255])}")

if __name__ == "__main__":
    # params
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument(
        "--input_path",
        default='/home/knuvi/Desktop/song/data/scannet_sample/train/scene0017_00/',
        help="path to input folder (e.g. /home/knuvi/Desktop/song/data/scannet/scans/scene0017_00/)",
    )
    parser.add_argument(
        "--output_path",
        default='/home/knuvi/Desktop/song/semantic-gaussians/output/train_samples/scene0017_00/',
        help="path to output folder (e.g. /home/knuvi/Desktop/song/semantic-gaussians/output/train_samples/scene0017_00/)",
    )
    parser.add_argument("--export_width", default=648, type=int)
    parser.add_argument("--export_height", default=484, type=int)
    parser.add_argument("--frame_skip", default=1, type=int)  # 모든 프레임 사용
    parser.add_argument("--voxel_size", default=0.02, type=float)  # Voxelization 설정
    parser.set_defaults(
        export_point_cloud=True,
    )

    opt = parser.parse_args()
    print(opt)

    elm = opt.input_path.split("/")[-1]
    if elm == "":
        elm = opt.input_path.split("/")[-2]

    output_path = opt.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    # load the data
    sys.stdout.write(f"loading {elm}... ")
    sd = SensorData(os.path.join(opt.input_path, f"{elm}.sens"))
    sys.stdout.write(f"loaded! Frames count: {len(sd.frames)}\n")
    
    # 포인트 클라우드 생성 및 레이블 매핑
    if opt.export_point_cloud:
        sd.export_point_cloud(
            output_path,
            target_size=(opt.export_width, opt.export_height),
            frame_skip=opt.frame_skip,
            voxel_size=opt.voxel_size
        )

    print("Processing complete!")