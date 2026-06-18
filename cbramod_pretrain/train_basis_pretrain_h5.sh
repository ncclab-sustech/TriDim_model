#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/vePFS-0x0d/home/ws2319/Tridim"
DATA_DIR="/vePFS-0x0d/eeg-h5/raw/merged_final_dataset"
IMAGE="tridim:tokenizer-h5py"

TRAIN_PY="${PROJECT_DIR}/CBraMod/train_pretrain_v11_v5.py"
SAVE_DIR="${PROJECT_DIR}/cbramod_runs/v11_blockswap_pretrain_h5_depth12_heads8_mask0.5_standard1020_21"
LOG_DIR="${PROJECT_DIR}/logs"

mkdir -p "${SAVE_DIR}"
mkdir -p "${LOG_DIR}"

# Still use gpu0 for now. Future multi-GPU can switch NPROC_PER_NODE > 1.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NPROC_PER_NODE=1

# ===== Data / protocol (true CBraMod patch protocol) =====
N_CHANNELS=21
SEQ_LEN=30
IN_DIM=200
WINDOW_STRIDE=6000
MASK_RATIO=0.5
MAX_SUBJECTS=2000

# ===== Model =====
OUT_DIM=200
D_MODEL=200
DIM_FEEDFORWARD=800
N_LAYER=12
NHEAD=8
DROPOUT=0.1

# ===== V11 block params =====
LAYER_SCALE_INIT=1e-2
DROP_PATH_C=0.0
DROP_PATH_K=0.0
DROP_PATH_T=0.0
DROP_PATH_MLP=0.0
DROP_PATH_SCHEDULE="linear"

# ===== Optimization =====
EPOCHS=40
BATCH_SIZE=128
LR=5e-4
WEIGHT_DECAY=5e-2

docker run --rm -it --gpus all \
  --shm-size=32g \
  -e PYTHONUNBUFFERED=1 \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  -v "${PROJECT_DIR}:${PROJECT_DIR}" \
  -v "${DATA_DIR}:${DATA_DIR}" \
  -w "${PROJECT_DIR}/CBraMod" \
  "${IMAGE}" \
  bash -lc '
    set -euo pipefail
    which python
    python -V
    python -c '"'"'import h5py; print("h5py ok")'"'"'
    python -c '"'"'import torch; print("torch ok")'"'"'

    torchrun --standalone --nproc_per_node='"${NPROC_PER_NODE}"' "'"${TRAIN_PY}"'" \
      --dataset_dir "'"${DATA_DIR}"'" \
      --model_dir "'"${SAVE_DIR}"'" \
      --recursive \
      --cuda 0 \
      --epochs "'"${EPOCHS}"'" \
      --batch_size "'"${BATCH_SIZE}"'" \
      --num_workers 8 \
      --lr "'"${LR}"'" \
      --weight_decay "'"${WEIGHT_DECAY}"'" \
      --clip_value 1.0 \
      --dropout "'"${DROPOUT}"'" \
      --in_dim "'"${IN_DIM}"'" \
      --out_dim "'"${OUT_DIM}"'" \
      --d_model "'"${D_MODEL}"'" \
      --dim_feedforward "'"${DIM_FEEDFORWARD}"'" \
      --seq_len "'"${SEQ_LEN}"'" \
      --n_channels "'"${N_CHANNELS}"'" \
      --n_layer "'"${N_LAYER}"'" \
      --nhead "'"${NHEAD}"'" \
      --mask_ratio "'"${MASK_RATIO}"'" \
      --window_stride "'"${WINDOW_STRIDE}"'" \
      --cache_open_files \
      --max_open_files 4 \
      --layer_scale_init "'"${LAYER_SCALE_INIT}"'" \
      --drop_path_c "'"${DROP_PATH_C}"'" \
      --drop_path_k "'"${DROP_PATH_K}"'" \
      --drop_path_t "'"${DROP_PATH_T}"'" \
      --drop_path_mlp "'"${DROP_PATH_MLP}"'" \
      --drop_path_schedule "'"${DROP_PATH_SCHEDULE}"'" \
      --lr_scheduler CosineAnnealingLR \
      --input_scale 100.0 \
      --need_mask \
      --max_subjects "'"${MAX_SUBJECTS}"'"
  ' 2>&1 | tee -a "${LOG_DIR}/train_basis_pretrain_$(date +%F_%H-%M-%S).log"
