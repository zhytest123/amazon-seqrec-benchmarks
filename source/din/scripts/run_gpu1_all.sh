#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

DATA_ROOT="../data/amazon_reviews_2023"
OUTPUT_DIR="outputs"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

run_category() {
  local category="$1"
  local maxlen="$2"
  local log_file="${LOG_DIR}/${category}_gpu1.log"

  {
    echo "===== ${category} train start $(date -Is) ====="
    python -u scripts/train_din.py \
      --category "${category}" \
      --data-root "${DATA_ROOT}" \
      --output-dir "${OUTPUT_DIR}" \
      --epochs 20 \
      --batch-size 512 \
      --eval-batch-size 64 \
      --candidate-chunk-size 2048 \
      --maxlen "${maxlen}" \
      --embedding-dim 64 \
      --hidden-units 128,64 \
      --attention-units 64,16 \
      --dropout-rate 0.2 \
      --eval-every 1 \
      --eval-limit 5000 \
      --device cuda

    echo "===== ${category} test start $(date -Is) ====="
    python -u scripts/eval_din.py \
      --category "${category}" \
      --data-root "${DATA_ROOT}" \
      --output-dir "${OUTPUT_DIR}" \
      --checkpoint "${OUTPUT_DIR}/${category}/best_model.pt" \
      --split test \
      --batch-size 64 \
      --candidate-chunk-size 2048 \
      --device cuda
    echo "===== ${category} done $(date -Is) ====="
  } 2>&1 | tee "${log_file}"
}

run_category "Industrial_and_Scientific" 100
run_category "Musical_Instruments" 100
run_category "CDs_and_Vinyl" 200

