"""Read .sens file to (color, depth, intrinsic, pose, and point cloud).
Code from https://github.com/ScanNet/ScanNet/tree/master/SensReader/python,
modified to support Python3, batch read, and point cloud export.
"""

import os
import sys
import math
import struct
import argparse
import numpy as np
import zlib
import imageio
import cv2
from pathlib import Path  # 추가된 부분
from tqdm import tqdm
import open3d as o3d

COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {
    -1: "unknown",
    0: "raw_ushort",
    1: "zlib_ushort",
    2: "occi_ushort",
}


class RGBDFrame:
    def load(self, file_handle):
        self.camera_to_world = np.asarray(struct.unpack("f" * 16, file_handle.read(16 * 4)), dtype=np.float32).reshape(
            4, 4
        )
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

    def export_depth_images(self, output_path, image_size=None, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print("exporting", len(self.frames) // frame_skip, " depth frames to", output_path)
        for f in range(0, len(self.frames), frame_skip):
            depth_data = self.frames[f].decompress_depth(self.depth_compression_type)
            depth = np.fromstring(depth_data, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
            if image_size is not None:
                depth = cv2.resize(
                    depth,
                    (image_size[1], image_size[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            imageio.imwrite(os.path.join(output_path, str(f) + ".png"), depth)
            # with open(os.path.join(output_path, str(f) + ".png"), "wb") as f:  # write 16-bit
            #     writer = png.Writer(width=depth.shape[1], height=depth.shape[0], bitdepth=16)
            #     depth = depth.reshape(-1, depth.shape[1]).tolist()
            #     writer.write(f, depth)

    def export_color_images(self, output_path, image_size=None, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print("exporting", len(self.frames) // frame_skip, "color frames to", output_path)
        for f in range(0, len(self.frames), frame_skip):
            color = self.frames[f].decompress_color(self.color_compression_type)
            if image_size is not None:
                color = cv2.resize(
                    color,
                    (image_size[1], image_size[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            imageio.imwrite(os.path.join(output_path, str(f) + ".jpg"), color)

    def save_mat_to_file(self, matrix, filename):
        with open(filename, "w") as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt="%f")

    def export_poses(self, output_path, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print("exporting", len(self.frames) // frame_skip, "camera poses to", output_path)
        for f in range(0, len(self.frames), frame_skip):
            self.save_mat_to_file(
                self.frames[f].camera_to_world,
                os.path.join(output_path, str(f) + ".txt"),
            )

    def export_intrinsics(self, output_path, image_size=None):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print("exporting camera intrinsics to", output_path)
        if image_size is not None:
            height, width = image_size
            self.intrinsic_color[0] = self.intrinsic_color[0] * (width - 0.5) / (self.intrinsic_color[0, 2] * 2)
            self.intrinsic_color[1] = self.intrinsic_color[1] * (height - 0.5) / (self.intrinsic_color[1, 2] * 2)
            self.intrinsic_depth[0] = self.intrinsic_depth[0] * (width - 0.5) / (self.intrinsic_depth[0, 2] * 2)
            self.intrinsic_depth[1] = self.intrinsic_depth[1] * (height - 0.5) / (self.intrinsic_depth[1, 2] * 2)
        self.save_mat_to_file(self.intrinsic_color, os.path.join(output_path, "intrinsic_color.txt"))
        self.save_mat_to_file(self.extrinsic_color, os.path.join(output_path, "extrinsic_color.txt"))
        self.save_mat_to_file(self.intrinsic_depth, os.path.join(output_path, "intrinsic_depth.txt"))
        self.save_mat_to_file(self.extrinsic_depth, os.path.join(output_path, "extrinsic_depth.txt"))

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

    def export_point_cloud(self, output_path, target_size=(648, 484), frame_skip=10):
        """Depth map과 Pose를 사용해 3D 포인트 클라우드 저장"""
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(f"Exporting 3D Point Cloud with target_size={target_size}, frame_skip={frame_skip}...")

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
        pcd = pcd.voxel_down_sample(voxel_size=0.02)  # 200만 목표
        
        nan_count = np.sum(np.isnan(np.asarray(pcd.points)))
        if nan_count > 0:
            print(f"Error: {nan_count} NaN points remain after downsampling")
            return
        
        output_file = os.path.join(output_path, "points3d.ply")
        o3d.io.write_point_cloud(output_file, pcd)
        print(f"✅ Saved {len(pcd.points)} points to {output_file}")


if __name__ == "__main__":
    # params
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument(
        "--input_path",
        required=True,
        help="path to input folder (e.g. ../scannet/train/scene0000_00/)",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="path to output folder (e.g. ../scannet_out/train/scene0000_00/)",
    )
    parser.add_argument("--not_export_depth_images", dest="export_depth_images", action="store_false")
    parser.add_argument("--not_export_color_images", dest="export_color_images", action="store_false")
    parser.add_argument("--not_export_poses", dest="export_poses", action="store_false")
    parser.add_argument("--not_export_intrinsics", dest="export_intrinsics", action="store_false")
    parser.add_argument("--export_point_cloud", dest="export_point_cloud", action="store_true")
    parser.add_argument("--export_width", default=648, type=int)
    parser.add_argument("--export_height", default=484, type=int)
    parser.add_argument("--frame_skip", default=10, type=int)  # 기본값을 10으로 변경
    parser.set_defaults(
        export_depth_images=True,
        export_color_images=True,
        export_poses=True,
        export_intrinsics=True,
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
    sys.stdout.write("loading %s..." % elm)
    sd = SensorData(os.path.join(opt.input_path, f"{elm}.sens"))
    sys.stdout.write("loaded!\n")
    if opt.export_depth_images:
        sd.export_depth_images(
            os.path.join(output_path, "depth"), (opt.export_height, opt.export_width), opt.frame_skip
        )
    if opt.export_color_images:
        sd.export_color_images(
            os.path.join(output_path, "color"), (opt.export_height, opt.export_width), opt.frame_skip
        )
    if opt.export_poses:
        sd.export_poses(os.path.join(output_path, "pose"), opt.frame_skip)
    if opt.export_intrinsics:
        sd.export_intrinsics(os.path.join(output_path, "intrinsic"), (opt.export_height, opt.export_width))
    if opt.export_point_cloud:
        sd.export_point_cloud(
            output_path,
            target_size=(opt.export_width, opt.export_height),  # 동적 크기 전달
            frame_skip=opt.frame_skip  # 동적 프레임 스킵 전달
        )

    print("Processing complete!")
