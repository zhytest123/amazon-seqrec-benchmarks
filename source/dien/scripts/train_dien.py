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

from dien.data import (  # noqa: E402
    SPLITS,
    DIENTrainDataset,
    build_mappings,
    encode_rows,
    read_split,
    save_mappings,
)
from dien.evaluate import evaluate_full  # noqa: E402
from dien.model import DIEN, DIENConfig  # noqa: E402


def parse_units(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(part) for part in value.split(",") if part)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DIEN on Amazon Reviews splits.")
    parser.add_argument("--category", required=True)
    parser.add_argument("--data-root", default="../data/amazon_reviews_2023")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--candidate-chunk-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--maxlen", type=int, default=100)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-units", type=parse_units, default=(128, 64))
    parser.add_argument("--attention-units", type=parse_units, default=(64, 16))
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--aux-loss-weight", type=float, default=0.1)
    parser.add_argument("--num-negatives", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=None)
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
    serializable_args = vars(args).copy()
    serializable_args["hidden_units"] = list(args.hidden_units)
    serializable_args["attention_units"] = list(args.attention_units)
    with (output_dir / "train_args.json").open("w") as f:
        json.dump(serializable_args, f, indent=2, sort_keys=True)

    print(f"Loading splits from {category_dir}")
    raw_rows = {split: read_split(category_dir, split) for split in SPLITS}
    user2idx, item2idx = build_mappings(raw_rows, include_splits=SPLITS)
    idx2item = {idx: item_id for item_id, idx in item2idx.items()}
    save_mappings(output_dir, user2idx, item2idx)

    encoded = {split: encode_rows(raw_rows[split], item2idx) for split in SPLITS}
    train_rows = encoded["train"]
    if args.train_limit is not None:
        train_rows = train_rows[: args.train_limit]
    train_dataset = DIENTrainDataset(
        train_rows,
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

    config = DIENConfig(
        item_num=len(item2idx),
        maxlen=args.maxlen,
        embedding_dim=args.embedding_dim,
        hidden_units=args.hidden_units,
        attention_units=args.attention_units,
        dropout_rate=args.dropout_rate,
        aux_loss_weight=args.aux_loss_weight,
    )
    model = DIEN(config).to(device)
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
        total_main_loss = 0.0
        total_aux_loss = 0.0
        total_batches = 0

        for seqs, lengths, pos_items, neg_items, aux_neg_seqs in train_loader:
            seqs = seqs.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            pos_items = pos_items.to(device, non_blocking=True)
            neg_items = neg_items.to(device, non_blocking=True)
            aux_neg_seqs = aux_neg_seqs.to(device, non_blocking=True)

            pos_logits, neg_logits, aux_loss = model(
                seqs,
                lengths,
                pos_items,
                neg_items,
                aux_neg_seqs,
            )
            pos_labels = torch.ones_like(pos_logits)
            neg_labels = torch.zeros_like(neg_logits)
            main_loss = criterion(pos_logits, pos_labels)
            main_loss = main_loss + criterion(neg_logits, neg_labels)
            loss = main_loss + args.aux_loss_weight * aux_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_main_loss += float(main_loss.detach().cpu())
            total_aux_loss += float(aux_loss.detach().cpu())
            total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        avg_main = total_main_loss / max(total_batches, 1)
        avg_aux = total_aux_loss / max(total_batches, 1)
        print(
            f"epoch={epoch} train_loss={avg_loss:.6f} "
            f"main_loss={avg_main:.6f} aux_loss={avg_aux:.6f}"
        )

        if args.eval_every > 0 and epoch % args.eval_every == 0:
            metrics = evaluate_full(
                model=model,
                rows=encoded["valid"],
                idx2item=idx2item,
                maxlen=args.maxlen,
                device=device,
                batch_size=args.eval_batch_size,
                candidate_chunk_size=args.candidate_chunk_size,
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
                        "args": serializable_args,
                        "epoch": epoch,
                        "valid_metrics": metrics,
                    },
                    output_dir / "best_model.pt",
                )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
                "args": serializable_args,
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
