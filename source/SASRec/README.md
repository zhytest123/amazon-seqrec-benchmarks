# SASRec baseline

This directory contains a SASRec implementation for the Amazon Reviews 2023
splits in `../../data/amazon_reviews_2023`.

## Train

```bash
cd source/SASRec
python scripts/train_sasrec.py \
  --category Industrial_and_Scientific \
  --data-root ../../data/amazon_reviews_2023 \
  --output-dir outputs \
  --epochs 20 \
  --maxlen 100 \
  --hidden-units 64 \
  --num-blocks 2 \
  --num-heads 2 \
  --batch-size 512
```

The script writes checkpoints, mappings, and validation metrics under
`outputs/<category>/`.

## Evaluate

```bash
cd source/SASRec
python scripts/eval_sasrec.py \
  --category Industrial_and_Scientific \
  --data-root ../../data/amazon_reviews_2023 \
  --checkpoint outputs/Industrial_and_Scientific/best_model.pt \
  --split test
```

The evaluator ranks against the full item vocabulary for the category, filters
items already present in the input history, computes `NDCG@10` and `Hit@10`,
and writes `test_predictions.csv`.

## Notes

- `0` is reserved for padding.
- ID mappings include items from train, valid, and test splits so cold targets
  are measurable instead of silently dropped.
- Training rows with empty history are skipped because SASRec requires a
  sequence prefix.
- The code uses full-catalog evaluation, not the 100-negative sampled
  evaluation in the reference repository.
