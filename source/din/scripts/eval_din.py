from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from din.data import SPLITS, build_mappings, encode_rows, read_split, save_mappings  # noqa: E402
from din.evaluate import evaluate_full  # noqa: E402
from din.model import DIN, DINConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DIN checkpoint.")
    parser.add_argument("--category", required=True)
    parser.add_argument("--data-root", default="../data/amazon_reviews_2023")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--candidate-chunk-size", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-filter-history", dest="filter_history", action="store_false")
    parser.set_defaults(filter_history=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    category_dir = Path(args.data_root) / args.category
    output_dir = Path(args.output_dir) / args.category
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config_dict = checkpoint["config"]
    config_dict["hidden_units"] = tuple(config_dict["hidden_units"])
    config_dict["attention_units"] = tuple(config_dict["attention_units"])
    config = DINConfig(**config_dict)
    model = DIN(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    raw_rows = {split: read_split(category_dir, split) for split in SPLITS}
    user2idx, item2idx = build_mappings(raw_rows, include_splits=SPLITS)
    save_mappings(output_dir, user2idx, item2idx)
    idx2item = {idx: item_id for item_id, idx in item2idx.items()}
    encoded_rows = encode_rows(raw_rows[args.split], item2idx)

    prediction_path = output_dir / f"{args.split}_predictions.csv"
    metrics = evaluate_full(
        model=model,
        rows=encoded_rows,
        idx2item=idx2item,
        maxlen=config.maxlen,
        device=device,
        batch_size=args.batch_size,
        candidate_chunk_size=args.candidate_chunk_size,
        output_path=prediction_path,
        limit=args.limit,
        filter_history=args.filter_history,
    )

    metrics_path = output_dir / f"{args.split}_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Predictions: {prediction_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()

