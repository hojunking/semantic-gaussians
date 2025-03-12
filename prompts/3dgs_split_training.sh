#!/bin/bash

# 기본 설정
BASE_INPUT_DIR="/workdir/data/scans_preprocessed"  # 전처리된 ScanNet 데이터 디렉토리
OUTPUT_BASE_DIR="/workdir/semantic-gaussians/output"  # 3DGS 출력 디렉토리
CONFIG_TEMPLATE="/workdir/semantic-gaussians/config/official_train.yaml"  # 3DGS YAML 파일
TRAIN_SCRIPT="/workdir/semantic-gaussians/train.py"  # 3DGS 실행 스크립트
TEMP_CONFIG="temp_config.yaml"  # 임시 config 파일
LOG_DIR="/workdir/semantic-gaussians/3D_model_results/logs"  # 로그 저장 디렉토리
LOAD_SCENES_SCRIPT="prompts/load_scenes.py"  # scene 목록 로드 스크립트

# 로그 디렉토리 생성
mkdir -p ${LOG_DIR}

# 인자 파싱
START_IDX="${1:-0}"    # 시작 인덱스 (기본값: 0)
END_IDX="${2:-200}"    # 종료 인덱스 (기본값: 200)
if [ -z "$START_IDX" ] || [ -z "$END_IDX" ]; then
    echo "Usage: $0 [start_idx] [end_idx]"
    exit 1
fi

# Python 스크립트 호출하여 scene 목록 로드
SCENE_LIST=$(python3 ${LOAD_SCENES_SCRIPT} "${BASE_INPUT_DIR}" "${START_IDX}" "${END_IDX}")
if [ $? -ne 0 ]; then
    echo "Error: Failed to load scene list."
    exit 1
fi

# JSON 파싱하여 SCENES_TO_PROCESS 배열 생성 (sed와 grep 개선)
SCENES_TO_PROCESS=($(echo "${SCENE_LIST}" | grep -v "Total scenes" | grep -v "Loaded" | sed 's/\[\|\]//g' | sed 's/"//g' | sed 's/,/ /g'))

# 처리할 scene 수 확인
NUM_SCENES=${#SCENES_TO_PROCESS[@]}
echo "Processing scenes from index ${START_IDX} to ${END_IDX} (Total: ${NUM_SCENES} scenes)"
echo "Scenes to process: ${SCENES_TO_PROCESS[*]}"

# 3DGS 실행 함수
run_3dgs() {
    local idx=$1
    local scene_path=$2
    local scene=$(basename ${scene_path})

    # 입력 및 출력 경로 설정
    SCENE_PATH="${BASE_INPUT_DIR}/${scene}"
    EXP_NAME="${scene}"
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${scene}"
    LOG_FILE="${LOG_DIR}/${scene}.log"
    PLY_PATH="${OUTPUT_DIR}/point_cloud/iteration_10000/point_cloud.ply"

    # 입력 디렉토리 확인
    if [ ! -d "${SCENE_PATH}" ]; then
        echo "Error: ${SCENE_PATH} does not exist. Skipping..."
        return 1
    fi

    # 이미 처리된 scene 건너뛰기 (출력 폴더 존재 여부로 판단)
    if [ -d "${OUTPUT_DIR}" ]; then
        echo "Skipping ${scene}: Already processed (output directory ${OUTPUT_DIR} exists)."
        return 0
    fi

    # 출력 디렉토리 생성
    mkdir -p ${OUTPUT_DIR}

    # 임시 config 파일 생성
    cp ${CONFIG_TEMPLATE} ${TEMP_CONFIG}

    # YAML 파일 수정 (scene_path, exp_name 업데이트)
    sed -i "s|scene_path:.*|scene_path: \"${SCENE_PATH}\"|" ${TEMP_CONFIG}
    sed -i "s|exp_name:.*|exp_name: \"${EXP_NAME}\"|" ${TEMP_CONFIG}

    # 3DGS 실행 (출력을 터미널과 로그 파일에 모두 기록)
    echo "Running 3DGS for ${scene} (Index: ${idx}/${END_IDX})..."
    python ${TRAIN_SCRIPT} --config ${TEMP_CONFIG} 2>&1 | tee ${LOG_FILE}

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "Error: 3DGS failed for ${scene}. Check ${LOG_FILE} for details."
        return 1
    fi

    # 임시 config 파일 삭제
    rm ${TEMP_CONFIG}

    # 3DGS 출력 확인
    if [ -f "${PLY_PATH}" ]; then
        echo "Successfully generated 3DGS output for ${scene} at ${PLY_PATH}."
    else
        echo "Error: 3DGS output ${PLY_PATH} not found for ${scene}."
        return 1
    fi

    return 0
}

# 선택된 scene 처리
CURRENT_IDX=$START_IDX
for scene_path in "${SCENES_TO_PROCESS[@]}"; do
    run_3dgs $CURRENT_IDX "$scene_path"
    ((CURRENT_IDX++))
done

echo "All scenes from index ${START_IDX} to ${END_IDX} processed for 3DGS conversion!"