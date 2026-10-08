# Amazon Sequential Recommendation Benchmarks

PyTorch implementations of three sequential recommender models on the Amazon
Reviews 2023 5-core recommendation splits:

- SASRec: self-attentive sequential recommendation.
- DIN: target-aware attention pooling over user behavior sequences.
- DIEN: GRU/AUGRU interest extraction and interest evolution.

The project uses full-catalog Top-10 evaluation. For each test sample, the
model ranks against the complete item vocabulary of that category, filters
items already present in the input history, and reports Hit@10 and NDCG@10.

## Repository Layout

```text
source/
  SASRec/      # SASRec model, training scripts, evaluation scripts
  din/         # DIN model, training scripts, evaluation scripts
  dien/        # DIEN model, training scripts, evaluation scripts
results/
  metrics_summary.csv
  metrics/     # Per-model, per-category test metrics
docs/
  experiment_report.md
requirements.txt
```

## Dataset

This repository does not include the original Amazon Reviews 2023 data.
Prepare the processed split files locally under:

```text
data/amazon_reviews_2023/
  Industrial_and_Scientific/
    train.csv.gz
    valid.csv.gz
    test.csv.gz
  Musical_Instruments/
    train.csv.gz
    valid.csv.gz
    test.csv.gz
  CDs_and_Vinyl/
    train.csv.gz
    valid.csv.gz
    test.csv.gz
```

Each CSV split is expected to contain:

```text
user_id,parent_asin,rating,timestamp,history
```

The current code uses the item ID sequence in `history` and the next interacted
item `parent_asin`. Ratings, timestamps, reviews, and item metadata are not used
in these baselines.

## Quick Start

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Train SASRec on one category:

```bash
cd source/SASRec
python scripts/train_sasrec.py \
  --category Industrial_and_Scientific \
  --data-root ../../data/amazon_reviews_2023 \
  --output-dir outputs \
  --epochs 20 \
  --maxlen 100 \
  --hidden-units 64 \
  --batch-size 512 \
  --device cuda
```

Evaluate a checkpoint:

```bash
python scripts/eval_sasrec.py \
  --category Industrial_and_Scientific \
  --data-root ../../data/amazon_reviews_2023 \
  --checkpoint outputs/Industrial_and_Scientific/best_model.pt \
  --split test \
  --device cuda
```

DIN and DIEN use the same data layout:

```bash
cd source/din
python scripts/train_din.py --category Industrial_and_Scientific \
  --data-root ../../data/amazon_reviews_2023 --output-dir outputs \
  --epochs 20 --maxlen 100 --embedding-dim 64 --batch-size 512 --device cuda

cd ../dien
python scripts/train_dien.py --category Industrial_and_Scientific \
  --data-root ../../data/amazon_reviews_2023 --output-dir outputs \
  --epochs 20 --maxlen 100 --embedding-dim 64 --batch-size 512 --device cuda
```

## Results

The reported metrics are from full-catalog test evaluation.

| Model | Category | Hit@10 | NDCG@10 |
|---|---|---:|---:|
| SASRec | Industrial_and_Scientific | 0.023752 | 0.012324 |
| SASRec | Musical_Instruments | 0.038040 | 0.019893 |
| SASRec | CDs_and_Vinyl | 0.026284 | 0.013095 |
| DIN | Industrial_and_Scientific | 0.028832 | 0.014630 |
| DIN | Musical_Instruments | 0.039085 | 0.020224 |
| DIN | CDs_and_Vinyl | 0.029360 | 0.014867 |
| DIEN | Industrial_and_Scientific | 0.027910 | 0.014210 |
| DIEN | Musical_Instruments | 0.042184 | 0.021803 |
| DIEN | CDs_and_Vinyl | 0.030426 | 0.015567 |

Best model by category:

| Category | Best Model | Hit@10 | NDCG@10 |
|---|---|---:|---:|
| Industrial_and_Scientific | DIN | 0.028832 | 0.014630 |
| Musical_Instruments | DIEN | 0.042184 | 0.021803 |
| CDs_and_Vinyl | DIEN | 0.030426 | 0.015567 |

## Notes

- `0` is reserved for padding.
- ID mappings include train, valid, and test items so cold test targets are
  measurable instead of being silently dropped.
- Training rows with empty history are skipped.
- The repository intentionally excludes raw data, checkpoints, logs, and large
  prediction CSV files.

