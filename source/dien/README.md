# DIEN baseline

This directory contains a lightweight DIEN implementation adapted from the
DeepCTR-Torch DIEN structure for the Amazon Reviews 2023 recommendation task.

The implementation is independent of DeepCTR-Torch. It keeps the main DIEN
ideas:

- GRU interest extractor over historical item embeddings.
- Auxiliary loss for next-click interest extraction.
- Query-aware interest evolution using a local activation unit and AUGRU.
- DNN scoring for candidate item ranking.

## Train

```bash
cd source/dien
python scripts/train_dien.py \
  --category Industrial_and_Scientific \
  --data-root ../../data/amazon_reviews_2023 \
  --output-dir outputs \
  --epochs 20 \
  --maxlen 100 \
  --embedding-dim 64 \
  --batch-size 256 \
  --eval-batch-size 64
```

## Evaluate

```bash
cd source/dien
python scripts/eval_dien.py \
  --category Industrial_and_Scientific \
  --data-root ../../data/amazon_reviews_2023 \
  --checkpoint outputs/Industrial_and_Scientific/best_model.pt \
  --split test
```

The evaluator ranks against the full item vocabulary for the category, filters
items already present in the input history, computes `NDCG@10` and `Hit@10`,
and writes `test_predictions.csv`.

## Notes

- DIEN uses right padding because GRU packing expects valid timesteps before
  padding.
- `0` is reserved for padding.
- ID mappings include items from train, valid, and test splits so cold targets
  are measurable instead of silently dropped.
- Training rows with empty history are skipped.
