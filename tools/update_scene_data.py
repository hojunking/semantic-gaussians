#!/usr/bin/env python3

import os
import argparse
import shutil
import numpy as np
from pathlib import Path

def update_scene_data(raw_scannet_path, sample_root, scene_name):
    """
    원본 ScanNet에서 vh_clean_2.ply와 labels.ply를 가져와 sample 디렉토리에 업데이트.
    """
    # 입력 및 출력 경로 설정
    scene_dir = os.path.join(sample_root, scene_name)
    raw_scene_dir = os.path.join(raw_scannet_path, scene_name)

    if not os.path.exists(raw_scene_dir):
        print(f"Error: Raw scene directory {raw_scene_dir} does not exist. Skipping {scene_name}...")
        return False

    # vh_clean_2.ply 가져오기
    vh_clean_ply_src = os.path.join(raw_scene_dir, f"{scene_name}_vh_clean_2.ply")
    point3d_ply_dest = os.path.join(scene_dir, "points3d.ply")

    if not os.path.exists(vh_clean_ply_src):
        print(f"Error: {vh_clean_ply_src} not found. Skipping {scene_name}...")
        return False

    # 기존 point3d.ply 삭제
    if os.path.exists(point3d_ply_dest):
        os.remove(point3d_ply_dest)
        print(f"Removed existing {point3d_ply_dest}")

    #vh_clean_2.ply 복사 (point3d.ply로 저장)
    shutil.copy2(vh_clean_ply_src, point3d_ply_dest)
    print(f"Copied {vh_clean_ply_src} to {point3d_ply_dest}")

    # vh_clean_2.labels.ply 가져오기
    labels_ply_src = os.path.join(raw_scene_dir, f"{scene_name}_vh_clean_2.labels.ply")
    labels_ply_dest = os.path.join(scene_dir, "vh_clean_2.labels.ply")

    if os.path.exists(labels_ply_src):
        shutil.copy2(labels_ply_src, labels_ply_dest)
        print(f"Copied {labels_ply_src} to {labels_ply_dest}")
    else:
        print(f"Warning: {labels_ply_src} not found. Skipping labels for {scene_name}...")

    return True

def main():
    # 인자 파싱
    parser = argparse.ArgumentParser(description="Update point3d.ply with vh_clean_2.ply from raw ScanNet data.")
    parser.add_argument("--raw_scannet_path", default= '/media/song/Desk SSD/hojun/scans', help="Path to the raw ScanNet dataset")
    parser.add_argument("--sample_root", default="/home/song/Desktop/song/data/scans_preprocessed", help="Path to the sample train directory")
    args = parser.parse_args()
    
    # scene 목록 가져오기
    scene_dirs = [d for d in os.listdir(args.sample_root) if os.path.isdir(os.path.join(args.sample_root, d))]
    scene_dirs.sort()

    print(f"Found {len(scene_dirs)} scenes in {args.sample_root}")
    
    # from_scene_dirs = [d for d in os.listdir(args.raw_scannet_path) if os.path.isdir(os.path.join(args.raw_scannet_path, d))]
    # from_scene_dirs.sort()

    # print(f"Found {len(from_scene_dirs)} scenes in {args.raw_scannet_path}")

    # 각 scene 처리
    for scene_name in scene_dirs:
        print(f"\nProcessing {scene_name}...")
        success = update_scene_data(args.raw_scannet_path, args.sample_root, scene_name)
        if success:
            print(f"Successfully updated {scene_name}")
        else:
            print(f"Failed to update {scene_name}")

if __name__ == "__main__":
    main()