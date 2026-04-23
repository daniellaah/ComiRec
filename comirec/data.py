from __future__ import annotations

import json
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
    """A single dataset class used by both training and evaluation loaders."""

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


def collate_train_records(records: list[dict[str, Any]]) -> TrainBatch:
    return TrainBatch(
        history_items=torch.tensor(
            [record["history_items"] for record in records],
            dtype=torch.long,
        ),
        history_mask=torch.tensor(
            [record["history_mask"] for record in records],
            dtype=torch.float32,
        ),
        targets=torch.tensor([record["target"] for record in records], dtype=torch.long),
    )


def collate_eval_records(records: list[dict[str, Any]]) -> EvalBatch:
    return EvalBatch(
        history_items=torch.tensor(
            [record["history_items"] for record in records],
            dtype=torch.long,
        ),
        history_mask=torch.tensor(
            [record["history_mask"] for record in records],
            dtype=torch.float32,
        ),
        targets=[list(record["targets"]) for record in records],
    )


def create_train_loader(
    samples_path: Path,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    dataset = load_dataset(samples_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        num_workers=num_workers,
        collate_fn=collate_train_records,
    )


def create_eval_loader(
    samples_path: Path,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader[dict[str, Any]]:
    dataset = load_dataset(samples_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_eval_records,
    )
