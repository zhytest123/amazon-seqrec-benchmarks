# Experiment Report

## Task

This project studies sequential product recommendation on three Amazon Reviews
2023 5-core categories:

- `Industrial_and_Scientific`
- `Musical_Instruments`
- `CDs_and_Vinyl`

Given a user's historical item sequence, the task is to predict the next
interacted item and output a Top-10 recommendation list.

## Data Processing

The expected split files are `train.csv.gz`, `valid.csv.gz`, and `test.csv.gz`
for each category. Each row includes:

```text
user_id,parent_asin,rating,timestamp,history
```

The implemented pipeline:

1. Maps string user IDs and item IDs to contiguous integer IDs.
2. Reserves `0` as the padding ID.
3. Splits the whitespace-separated `history` field into an item sequence.
4. Uses `history` as model input and `parent_asin` as the positive target.
5. Skips empty-history training rows.
6. Uses the provided valid/test splits without re-splitting.
7. Filters items already present in the input history during evaluation.

Only ID-based sequential features are used in these baselines. Ratings,
timestamps, review texts, and item metadata are not used.

## Data Scale

| Category | Train Samples | Valid Samples | Test Samples | Items |
|---|---:|---:|---:|---:|
| Industrial_and_Scientific | 259,992 | 50,985 | 50,985 | 25,848 |
| Musical_Instruments | 339,519 | 57,439 | 57,439 | 24,587 |
| CDs_and_Vinyl | 1,181,136 | 123,876 | 123,876 | 89,370 |

The train sample counts exclude empty-history rows.

## Models

### SASRec

SASRec encodes historical item sequences with item embeddings, positional
embeddings, causal multi-head self-attention, feed-forward layers, LayerNorm,
and dropout. The final sequence representation is scored against candidate
item embeddings.

### DIN

DIN uses target-aware attention pooling. For each candidate item, the local
activation unit computes relevance between the target item and each historical
item using `[query, key, query-key, query*key]`, and then feeds the pooled
interest representation into an MLP scorer.

### DIEN

DIEN adds interest evolution modeling. It first extracts interest states with a
GRU and an auxiliary loss, then uses query-aware attention and AUGRU to evolve
historical interest for the target candidate item.

## Experimental Setup

Shared setup:

| Parameter | Value |
|---|---:|
| Embedding / hidden dim | 64 |
| Batch size | 512 |
| Epochs | 20 |
| Dropout | 0.2 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Negative samples | 1 |

Maximum history length:

| Category | maxlen |
|---|---:|
| Industrial_and_Scientific | 100 |
| Musical_Instruments | 100 |
| CDs_and_Vinyl | 200 |

Evaluation uses full-catalog ranking. For each valid/test sample, the model
ranks every item in the category, filters items already present in the input
history, and selects Top-10 items.

## Metrics

Hit@10 measures whether the ground-truth target appears in the Top-10 list.
NDCG@10 additionally considers the position of the ground-truth target.

This setup is stricter than sampled evaluation because each sample is ranked
against tens of thousands of items instead of a small set of random negatives.

## Results

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

DIN and DIEN outperform SASRec on all three categories. This suggests that
target-aware interest modeling is helpful for this task. DIEN performs best on
`Musical_Instruments` and `CDs_and_Vinyl`, where interest evolution appears to
bring additional gains.

## Limitations

- Side information such as item title, category, price, description, image, and
  review text is not used.
- Ratings and time intervals are not explicitly modeled.
- Training uses simple random negative sampling.
- DIN and DIEN are query-aware models, so full-catalog evaluation is more
  expensive than SASRec matrix scoring.

Future improvements can include side-feature fusion, text embeddings for item
semantic representation, popularity-aware or hard negative sampling, learning
rate scheduling, and a two-stage recall/ranking architecture.

