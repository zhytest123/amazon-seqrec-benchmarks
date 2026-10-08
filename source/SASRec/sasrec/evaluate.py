from __future__ import annotations

import csv
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import EncodedRow, SequenceEvalDataset
from .model import SASRec


@torch.no_grad()
def evaluate_full(
    model: SASRec,
    rows: list[EncodedRow],
    idx2item: dict[int, str],
    maxlen: int,
    device: torch.device,
    batch_size: int = 128,
    output_path: Path | None = None,
    limit: int | None = None,
    filter_history: bool = True,
) -> dict[str, float]:
    model.eval()
    eval_rows = [row for row in rows if len(row.history) > 0]
    if limit is not None:
        eval_rows = eval_rows[:limit]

    dataset = SequenceEvalDataset(eval_rows, maxlen=maxlen)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    item_ids = torch.arange(1, model.item_num + 1, dtype=torch.long, device=device)
    item_labels = [idx2item[idx] for idx in range(1, model.item_num + 1)]

    total_ndcg = 0.0
    total_hit = 0.0
    total = 0

    writer = None
    out_file = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = output_path.open("w", newline="")
        writer = csv.writer(out_file)
        writer.writerow(["user_id", "target_asin", "rank", "ndcg@10", "top10"])

    try:
        for seqs, targets, row_indices in loader:
            seqs = seqs.to(device)
            targets = targets.to(device)
            scores = model.score_items(seqs, item_ids)

            batch_rows = [eval_rows[int(i)] for i in row_indices.tolist()]
            if filter_history:
                for row_idx, row in enumerate(batch_rows):
                    history_ids = [item - 1 for item in row.history if 1 <= item <= model.item_num]
                    if history_ids:
                        scores[row_idx, torch.tensor(history_ids, device=device)] = -torch.inf

            target_scores = scores.gather(1, (targets - 1).unsqueeze(1)).squeeze(1)
            ranks = (scores > target_scores.unsqueeze(1)).sum(dim=1)
            _, top_indices = torch.topk(scores, k=10, dim=1)

            ranks_cpu = ranks.cpu().tolist()
            top_indices_cpu = top_indices.cpu().tolist()
            for local_idx, rank in enumerate(ranks_cpu):
                ndcg = 0.0
                hit = 0.0
                if rank < 10:
                    hit = 1.0
                    ndcg = 1.0 / math.log2(rank + 2)
                total += 1
                total_hit += hit
                total_ndcg += ndcg

                if writer is not None:
                    row = batch_rows[local_idx]
                    top10 = [item_labels[item_idx] for item_idx in top_indices_cpu[local_idx]]
                    writer.writerow(
                        [
                            row.user_id,
                            row.target_asin,
                            int(rank) + 1,
                            f"{ndcg:.8f}",
                            " ".join(top10),
                        ]
                    )
    finally:
        if out_file is not None:
            out_file.close()

    if total == 0:
        return {"ndcg@10": 0.0, "hit@10": 0.0, "count": 0}
    return {
        "ndcg@10": total_ndcg / total,
        "hit@10": total_hit / total,
        "count": total,
    }
