from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(slots=True)
class TrainBatch:
    history_items: torch.Tensor
    history_mask: torch.Tensor
    targets: torch.Tensor


@dataclass(slots=True)
class EvalBatch:
    history_items: torch.Tensor
    history_mask: torch.Tensor
    targets: list[list[int]]


class SequenceDataset(Dataset[dict[str, Any]]):
    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def load_samples(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            samples.append(json.loads(line))
    return samples


def read_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def count_items(item_map_path: Path) -> int:
    max_item_id = 0
    with item_map_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            _raw_item, mapped_item = line.rstrip("\n").split(",", 1)
            max_item_id = max(max_item_id, int(mapped_item))
    return max_item_id + 1


def load_dataset(samples_path: Path) -> SequenceDataset:
    return SequenceDataset(load_samples(samples_path))


def _pad_sequence(sequence: list[int], cutoff: int, maxlen: int, pad_value: int = 0) -> list[int]:
    history = sequence[max(0, cutoff - maxlen) : cutoff]
    if len(history) < maxlen:
        history = history + [pad_value] * (maxlen - len(history))
    return history


def _history_mask(cutoff: int, sequence_length: int, maxlen: int) -> list[bool]:
    history_length = min(cutoff, maxlen, sequence_length)
    return [True] * history_length + [False] * (maxlen - history_length)


def collate_train_records(
    records: list[dict[str, Any]],
    *,
    maxlen: int,
    rng: random.Random,
) -> TrainBatch:
    history_items: list[list[int]] = []
    history_mask: list[list[bool]] = []
    targets: list[int] = []

    for record in records:
        sequence = list(record["sequence"])
        cutoff = rng.randrange(4, len(sequence))
        history_items.append(_pad_sequence(sequence, cutoff, maxlen))
        history_mask.append(_history_mask(cutoff, len(sequence), maxlen))
        targets.append(sequence[cutoff])

    return TrainBatch(
        history_items=torch.tensor(history_items, dtype=torch.long),
        history_mask=torch.tensor(history_mask, dtype=torch.float32),
        targets=torch.tensor(targets, dtype=torch.long),
    )


def collate_eval_records(records: list[dict[str, Any]], *, maxlen: int) -> EvalBatch:
    history_items: list[list[int]] = []
    history_mask: list[list[bool]] = []
    targets: list[list[int]] = []

    for record in records:
        sequence = list(record["sequence"])
        cutoff = int(len(sequence) * 0.8)
        history_items.append(_pad_sequence(sequence, cutoff, maxlen))
        history_mask.append(_history_mask(cutoff, len(sequence), maxlen))
        targets.append(sequence[cutoff:])

    return EvalBatch(
        history_items=torch.tensor(history_items, dtype=torch.long),
        history_mask=torch.tensor(history_mask, dtype=torch.float32),
        targets=targets,
    )


def create_train_loader(
    samples_path: Path,
    batch_size: int,
    seed: int,
    maxlen: int,
    num_workers: int = 0,
) -> DataLoader[dict[str, Any]]:
    dataset = load_dataset(samples_path)
    rng = random.Random(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=lambda records: collate_train_records(records, maxlen=maxlen, rng=rng),
    )


def create_eval_loader(
    samples_path: Path,
    batch_size: int,
    maxlen: int,
    num_workers: int = 0,
) -> DataLoader[dict[str, Any]]:
    dataset = load_dataset(samples_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda records: collate_eval_records(records, maxlen=maxlen),
    )
