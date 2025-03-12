#!/usr/bin/env python3

import os
import sys
import glob
import json

def load_scene_list(base_dir, start_idx, end_idx):
    # scene 목록 로드 및 정렬 (glob 사용)
    pattern = os.path.join(base_dir, "scene[0-9][0-9][0-9][0-9]_[0-9][0-9]")
    scene_paths = glob.glob(pattern)
    scenes = [os.path.basename(path) for path in scene_paths if os.path.isdir(path)]
    scenes.sort()

    # 총 scene 수 출력
    total_scenes = len(scenes)
    print(f"Total scenes found: {total_scenes}", flush=True)

    # 범위 검증 (start_idx와 end_idx는 scene 번호 기준)
    if start_idx < 0 or end_idx > 9999 or start_idx > end_idx:
        print(f"Error: Invalid range. start_idx={start_idx}, end_idx={end_idx}, max_scene=9999", flush=True)
        sys.exit(1)

    # scene 번호(앞 4자리) 추출 및 필터링
    scenes_to_process = []
    for scene in scenes:
        scene_num = int(scene[5:9])  # sceneXXXX_YY에서 XXXX 부분 추출
        if start_idx <= scene_num <= end_idx:
            scenes_to_process.append(os.path.join(base_dir, scene))

    print(f"Loaded {len(scenes_to_process)} scenes for processing (scene {start_idx}xx to {end_idx}xx)", flush=True)
    return scenes_to_process

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python load_scenes.py <base_dir> <start_idx> <end_idx>")
        sys.exit(1)

    base_dir = sys.argv[1]
    start_idx = int(sys.argv[2])
    end_idx = int(sys.argv[3])

    scenes = load_scene_list(base_dir, start_idx, end_idx)
    # scene 리스트를 JSON 형식으로 출력
    print(json.dumps(scenes), flush=True)