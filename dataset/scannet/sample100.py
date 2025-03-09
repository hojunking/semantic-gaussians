# sample_scannet_scenes.py
import os
import random
import argparse

def sample_scenes_from_txt(input_file, num_samples, output_file):
    # scannetv2_train.txt 또는 scannetv2_val.txt 파일 로드
    with open(input_file, "r") as f:
        scene_names = [line.strip() for line in f if line.strip()]

    # 전체 scene 수 확인
    total_scenes = len(scene_names)
    print(f"Total scenes in {input_file}: {total_scenes}")

    # 요청된 샘플 수와 비교
    if num_samples > total_scenes:
        raise ValueError(f"Requested {num_samples} scenes, but only {total_scenes} scenes are available in {input_file}.")

    # 무작위 샘플링
    sampled_scenes = random.sample(scene_names, num_samples)
    sampled_scenes.sort()  # 정렬하여 일관성 유지

    # lr_file로 저장
    with open(output_file, "w") as f:
        for scene_name in sampled_scenes:
            f.write(f"{scene_name}\n")

    print(f"Sampled {len(sampled_scenes)} scenes and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_root", required=True, help="Path to the ScanNet meta data directory (e.g., pointcept/datasets/preprocessing/scannet/meta_data)")
    parser.add_argument("--output_dir", required=True, help="Output directory for lr_file")
    parser.add_argument("--num_train", type=int, default=100, help="Number of train scenes to sample")
    parser.add_argument("--num_valid", type=int, default=20, help="Number of valid scenes to sample")
    args = parser.parse_args()

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # Train 데이터 샘플링
    sample_scenes_from_txt(
        input_file=os.path.join(args.meta_root, "scannetv2_train.txt"),
        num_samples=args.num_train,
        output_file=os.path.join(args.output_dir, "sampled_train_scenes.txt")
    )

    # Valid 데이터 샘플링
    sample_scenes_from_txt(
        input_file=os.path.join(args.meta_root, "scannetv2_val.txt"),
        num_samples=args.num_valid,
        output_file=os.path.join(args.output_dir, "sampled_val_scenes.txt")
    )