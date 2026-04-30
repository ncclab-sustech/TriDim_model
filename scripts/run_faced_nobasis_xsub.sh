#!/usr/bin/env bash
# Cross-subject (4:3:3) sweep on FACED_new with the nobasis baseline.
# 5 seeds (42-46), label_order split, F1 model selection, no augmentation.
#
# Edit ROOT_PATH to point at your local FACED_new directory before running.

set -euo pipefail

ROOT_PATH="${ROOT_PATH:-/path/to/datasets/FACED_new}"
GPU="${GPU:-0}"

# Deterministic cuBLAS workspace (paired with torch.use_deterministic_algorithms in run.py).
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

python run.py \
    --model eeg_basis_mixer_v2_nobasis \
    --data FACED_new \
    --dataset_paths_yaml ./configs/datasets/FACED_new.yaml \
    --root_path "${ROOT_PATH}" \
    --gpu "${GPU}" --gpu_idx "${GPU}" \
    --num_workers 4 \
    --itr 5 --seed_start 42 \
    --split_mode label_order --train_ratio 0.4 --val_ratio 0.3 \
    --augmentations none --select_metric F1
