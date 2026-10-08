from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sasrec.data import (  # noqa: E402
    SPLITS,
    SequenceTrainDataset,
    build_mappings,
    encode_rows,
    read_split,
    save_mappings,
)
from sasrec.evaluate import evaluate_full  # noqa: E402
from sasrec.model import SASRec, SASRecConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SASRec on Amazon Reviews splits.")
    parser.add_argument("--category", required=True)
    parser.add_argument("--data-root", default="../data/amazon_reviews_2023")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--maxlen", type=int, default=100)
    parser.add_argument("--hidden-units", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--num-negatives", type=int, default=1)
    parser.add_argument("--l2-emb", type=float, default=0.0)
    parser.add_argument("--norm-first", action="store_true", default=True)
    parser.add_argument("--no-norm-first", dest="norm_first", action="store_false")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    category_dir = Path(args.data_root) / args.category
    if not category_dir.exists():
        raise FileNotFoundError(f"category directory not found: {category_dir}")

    output_dir = Path(args.output_dir) / args.category
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "train_args.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    print(f"Loading splits from {category_dir}")
    raw_rows = {split: read_split(category_dir, split) for split in SPLITS}
    user2idx, item2idx = build_mappings(raw_rows, include_splits=SPLITS)
    idx2item = {idx: item_id for item_id, idx in item2idx.items()}
    save_mappings(output_dir, user2idx, item2idx)

    encoded = {split: encode_rows(raw_rows[split], item2idx) for split in SPLITS}
    train_dataset = SequenceTrainDataset(
        encoded["train"],
        item_num=len(item2idx),
        maxlen=args.maxlen,
        num_negatives=args.num_negatives,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    config = SASRecConfig(
        item_num=len(item2idx),
        maxlen=args.maxlen,
        hidden_units=args.hidden_units,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        norm_first=args.norm_first,
    )
    model = SASRec(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    criterion = torch.nn.BCEWithLogitsLoss()

    print(
        "Prepared "
        f"users={len(user2idx):,}, items={len(item2idx):,}, "
        f"train_rows={len(train_dataset):,}, valid_rows={len(encoded['valid']):,}"
    )

    best_valid = -1.0
    metrics_log: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for seqs, pos_items, neg_items in train_loader:
            seqs = seqs.to(device, non_blocking=True)
            pos_items = pos_items.to(device, non_blocking=True)
            neg_items = neg_items.to(device, non_blocking=True)

            pos_logits, neg_logits = model(seqs, pos_items, neg_items)
            pos_labels = torch.ones_like(pos_logits)
            neg_labels = torch.zeros_like(neg_logits)

            loss = criterion(pos_logits, pos_labels)
            loss = loss + criterion(neg_logits, neg_labels)
            if args.l2_emb > 0:
                loss = loss + args.l2_emb * torch.sum(model.item_emb.weight**2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        print(f"epoch={epoch} train_loss={avg_loss:.6f}")

        if args.eval_every > 0 and epoch % args.eval_every == 0:
            metrics = evaluate_full(
                model=model,
                rows=encoded["valid"],
                idx2item=idx2item,
                maxlen=args.maxlen,
                device=device,
                batch_size=args.eval_batch_size,
                output_path=None,
                limit=args.eval_limit,
            )
            record = {"epoch": epoch, "train_loss": avg_loss, **metrics}
            metrics_log.append(record)
            print(
                f"epoch={epoch} valid_ndcg@10={metrics['ndcg@10']:.6f} "
                f"valid_hit@10={metrics['hit@10']:.6f} count={metrics['count']}"
            )
            if metrics["ndcg@10"] > best_valid:
                best_valid = metrics["ndcg@10"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config.__dict__,
                        "args": vars(args),
                        "epoch": epoch,
                        "valid_metrics": metrics,
                    },
                    output_dir / "best_model.pt",
                )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
                "args": vars(args),
                "epoch": epoch,
            },
            output_dir / "latest_model.pt",
        )
        with (output_dir / "metrics.jsonl").open("w") as f:
            for record in metrics_log:
                f.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"Done. Outputs are under {output_dir}")


if __name__ == "__main__":
    main()

