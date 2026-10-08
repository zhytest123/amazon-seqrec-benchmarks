#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

DATA_ROOT="../data/amazon_reviews_2023"
OUTPUT_DIR="outputs"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

{
  echo "===== CDs_and_Vinyl train restart $(date -Is) ====="
  python -u scripts/train_din.py \
    --category "CDs_and_Vinyl" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --epochs 20 \
    --batch-size 512 \
    --eval-batch-size 8 \
    --candidate-chunk-size 256 \
    --maxlen 200 \
    --embedding-dim 64 \
    --hidden-units 128,64 \
    --attention-units 64,16 \
    --dropout-rate 0.2 \
    --eval-every 1 \
    --eval-limit 5000 \
    --device cuda

  echo "===== CDs_and_Vinyl test start $(date -Is) ====="
  python -u scripts/eval_din.py \
    --category "CDs_and_Vinyl" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --checkpoint "${OUTPUT_DIR}/CDs_and_Vinyl/best_model.pt" \
    --split test \
    --batch-size 8 \
    --candidate-chunk-size 256 \
    --device cuda
  echo "===== CDs_and_Vinyl done $(date -Is) ====="
} 2>&1 | tee "${LOG_DIR}/CDs_and_Vinyl_gpu1_safe.log"

