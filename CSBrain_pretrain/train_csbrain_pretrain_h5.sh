#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/vePFS-0x0d/home/ws2319/Tridim"
DATA_DIR="/vePFS-0x0d/eeg-h5/raw/merged_final_dataset"
IMAGE="tridim:tokenizer-h5py"

SAVE_DIR="${PROJECT_DIR}/csbrain_runs/pretrain_h5_depth20_heads8_mask0.5_standard1020_21"
LOG_DIR="${PROJECT_DIR}/logs"

CONTAINER_PROJECT_DIR="/workspace/Tridim"
CONTAINER_DATA_DIR="/workspace/data"


# FP1, FP2, F7, F3, FZ, F4, F8, T7, C3, CZ, C4, T8,
# P7, P3, PZ, P4, P8, O1, O2, A1, A2
SOURCE_N_CHANNELS=21
CHANNEL_INDICES="all"
N_CHANNELS=21
MONTAGE="standard1020_21"

CUDA=0
EPOCHS=40
BATCH_SIZE=256
NUM_WORKERS=8

LR=5e-4
WEIGHT_DECAY=5e-2
CLIP_VALUE=1

IN_DIM=200
OUT_DIM=200
D_MODEL=200
DIM_FEEDFORWARD=800
SEQ_LEN=30
N_LAYER=20
NHEAD=8
MASK_RATIO=0.5
WINDOW_STRIDE=6000
MAX_SUBJECTS=2000

mkdir -p "${SAVE_DIR}"
mkdir -p "${LOG_DIR}"

EXTRA_ARGS=()
if [ -n "${MAX_SUBJECTS}" ]; then
  EXTRA_ARGS+=(--max_subjects "${MAX_SUBJECTS}")
fi

docker run --rm -it --gpus all \
  --shm-size=32g \
  -e PYTHONUNBUFFERED=1 \
  -v "${PROJECT_DIR}:${CONTAINER_PROJECT_DIR}" \
  -v "${DATA_DIR}:${CONTAINER_DATA_DIR}" \
  -w "${CONTAINER_PROJECT_DIR}/CSBrain" \
  "${IMAGE}" \
  bash -lc "
    set -euo pipefail

    which python
    python -V
    python -c 'import sys; print(sys.executable)'
    python -c 'import h5py; print(\"h5py ok\")'
    python -m pip install -q einops threadpoolctl torchinfo ptflops

    python -u ${CONTAINER_PROJECT_DIR}/CSBrain/pretrain_csbrain_h5.py \
      --dataset_dir ${CONTAINER_DATA_DIR} \
      --model_dir ${CONTAINER_PROJECT_DIR}/csbrain_runs/pretrain_h5_depth12_heads8_mask0.5_standard1020_21 \
      --recursive \
      --cuda ${CUDA} \
      --epochs ${EPOCHS} \
      --batch_size ${BATCH_SIZE} \
      --num_workers ${NUM_WORKERS} \
      --lr ${LR} \
      --weight_decay ${WEIGHT_DECAY} \
      --clip_value ${CLIP_VALUE} \
      --dropout 0.1 \
      --in_dim ${IN_DIM} \
      --out_dim ${OUT_DIM} \
      --d_model ${D_MODEL} \
      --dim_feedforward ${DIM_FEEDFORWARD} \
      --seq_len ${SEQ_LEN} \
      --source_n_channels ${SOURCE_N_CHANNELS} \
      --channel_indices ${CHANNEL_INDICES} \
      --n_channels ${N_CHANNELS} \
      --n_layer ${N_LAYER} \
      --nhead ${NHEAD} \
      --mask_ratio ${MASK_RATIO} \
      --window_stride ${WINDOW_STRIDE} \
      --montage ${MONTAGE} \
      --cache_open_files \
      --max_open_files 4 \
      --skip_model_summary \
      \
      ${EXTRA_ARGS[*]}
  " 2>&1 | tee -a "${LOG_DIR}/train_csbrain_pretrain_h5_$(date +%F_%H-%M-%S).log"
