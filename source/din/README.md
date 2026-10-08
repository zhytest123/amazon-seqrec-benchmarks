# DIN baseline

This directory contains a lightweight DIN implementation adapted from the
DeepCTR-Torch DIN structure for the Amazon Reviews 2023 recommendation task.

The implementation keeps the main DIN components:

- Shared item embedding for target item and historical item sequence.
- Local activation unit over `[query, key, query-key, query*key]`.
- Attention sequence pooling conditioned on the target item.
- DNN scorer for positive/negative item ranking.

## Train

```bash
cd source/din
python scripts/train_din.py \
  --category Industrial_and_Scientific \
  --data-root ../../data/amazon_reviews_2023 \
  --output-dir outputs \
  --epochs 20 \
  --maxlen 100 \
  --embedding-dim 64 \
  --batch-size 512 \
  --eval-batch-size 64
```

## Evaluate

```bash
cd source/din
python scripts/eval_din.py \
  --category Industrial_and_Scientific \
  --data-root ../../data/amazon_reviews_2023 \
  --checkpoint outputs/Industrial_and_Scientific/best_model.pt \
  --split test
```

The evaluator ranks against the full item vocabulary for the category, filters
items already present in the input history, computes `NDCG@10` and `Hit@10`,
and writes `test_predictions.csv`.

## Notes

- DIN uses right padding because attention masks valid prefix timesteps by
  sequence length.
- `0` is reserved for padding.
- ID mappings include items from train, valid, and test splits.
- Training rows with empty history are skipped.
