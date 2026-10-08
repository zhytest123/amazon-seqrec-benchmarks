#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

DATA_ROOT="../data/amazon_reviews_2023"
OUTPUT_DIR="outputs"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

run_category() {
  local category="$1"
  local maxlen="$2"
  local log_file="${LOG_DIR}/${category}_gpu3.log"

  {
    echo "===== ${category} train start $(date -Is) ====="
    python -u scripts/train_sasrec.py \
      --category "${category}" \
      --data-root "${DATA_ROOT}" \
      --output-dir "${OUTPUT_DIR}" \
      --epochs 20 \
      --batch-size 512 \
      --eval-batch-size 512 \
      --maxlen "${maxlen}" \
      --hidden-units 64 \
      --num-blocks 2 \
      --num-heads 2 \
      --dropout-rate 0.2 \
      --eval-every 1 \
      --device cuda

    echo "===== ${category} test start $(date -Is) ====="
    python -u scripts/eval_sasrec.py \
      --category "${category}" \
      --data-root "${DATA_ROOT}" \
      --output-dir "${OUTPUT_DIR}" \
      --checkpoint "${OUTPUT_DIR}/${category}/best_model.pt" \
      --split test \
      --batch-size 512 \
      --device cuda
    echo "===== ${category} done $(date -Is) ====="
  } 2>&1 | tee "${log_file}"
}

run_category "Industrial_and_Scientific" 100
# Run the smaller category first as the baseline sanity target.
# Enable the following two lines only after Industrial_and_Scientific is stable.
# run_category "Musical_Instruments" 100
# run_category "CDs_and_Vinyl" 200
