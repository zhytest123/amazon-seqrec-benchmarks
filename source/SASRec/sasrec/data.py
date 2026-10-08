from __future__ import annotations

import csv
import gzip
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset


SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class RawRow:
    user_id: str
    target_asin: str
    history: tuple[str, ...]


@dataclass(frozen=True)
class EncodedRow:
    user_id: str
    target_asin: str
    target: int
    history: tuple[int, ...]


def read_split(category_dir: Path, split: str) -> list[RawRow]:
    path = category_dir / f"{split}.csv.gz"
    rows: list[RawRow] = []
    with gzip.open(path, "rt", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            history = tuple(row["history"].split()) if row["history"] else ()
            rows.append(
                RawRow(
                    user_id=row["user_id"],
                    target_asin=row["parent_asin"],
                    history=history,
                )
            )
    return rows


def build_mappings(
    split_rows: dict[str, list[RawRow]],
    include_splits: Iterable[str] = SPLITS,
) -> tuple[dict[str, int], dict[str, int]]:
    user_ids: set[str] = set()
    item_ids: set[str] = set()
    for split in include_splits:
        for row in split_rows[split]:
            user_ids.add(row.user_id)
            item_ids.add(row.target_asin)
            item_ids.update(row.history)

    user2idx = {user_id: idx for idx, user_id in enumerate(sorted(user_ids), start=1)}
    item2idx = {item_id: idx for idx, item_id in enumerate(sorted(item_ids), start=1)}
    return user2idx, item2idx


def encode_rows(rows: list[RawRow], item2idx: dict[str, int]) -> list[EncodedRow]:
    encoded: list[EncodedRow] = []
    for row in rows:
        if row.target_asin not in item2idx:
            continue
        history = tuple(item2idx[item_id] for item_id in row.history if item_id in item2idx)
        encoded.append(
            EncodedRow(
                user_id=row.user_id,
                target_asin=row.target_asin,
                target=item2idx[row.target_asin],
                history=history,
            )
        )
    return encoded


def save_mappings(output_dir: Path, user2idx: dict[str, int], item2idx: dict[str, int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "user2idx.json").open("w") as f:
        json.dump(user2idx, f, separators=(",", ":"))
    with (output_dir / "item2idx.json").open("w") as f:
        json.dump(item2idx, f, separators=(",", ":"))
    idx2item = {str(idx): item_id for item_id, idx in item2idx.items()}
    with (output_dir / "idx2item.json").open("w") as f:
        json.dump(idx2item, f, separators=(",", ":"))


def load_item_mapping(mapping_dir: Path) -> tuple[dict[str, int], dict[int, str]]:
    with (mapping_dir / "item2idx.json").open() as f:
        item2idx = json.load(f)
    idx2item = {idx: item_id for item_id, idx in item2idx.items()}
    return item2idx, idx2item


def make_sequence(history: tuple[int, ...], maxlen: int) -> list[int]:
    seq = [0] * maxlen
    truncated = history[-maxlen:]
    if truncated:
        seq[-len(truncated) :] = truncated
    return seq


class SequenceTrainDataset(Dataset):
    def __init__(
        self,
        rows: list[EncodedRow],
        item_num: int,
        maxlen: int,
        num_negatives: int = 1,
    ) -> None:
        self.rows = [row for row in rows if len(row.history) > 0]
        self.item_num = item_num
        self.maxlen = maxlen
        self.num_negatives = num_negatives

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        seen = set(row.history)
        seen.add(row.target)
        negatives = []
        while len(negatives) < self.num_negatives:
            item = random.randint(1, self.item_num)
            if item not in seen:
                negatives.append(item)

        seq = make_sequence(row.history, self.maxlen)
        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(row.target, dtype=torch.long),
            torch.tensor(negatives, dtype=torch.long),
        )


class SequenceEvalDataset(Dataset):
    def __init__(self, rows: list[EncodedRow], maxlen: int) -> None:
        self.rows = [row for row in rows if len(row.history) > 0]
        self.maxlen = maxlen

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.rows[idx]
        return (
            torch.tensor(make_sequence(row.history, self.maxlen), dtype=torch.long),
            torch.tensor(row.target, dtype=torch.long),
            idx,
        )
