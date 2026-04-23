from __future__ import annotations

import json
from pathlib import Path

import pytest

from comirec.configs import DataConfig


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_processed_dataset(root: Path) -> DataConfig:
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    train_rows = [
        {"history_items": [0, 1, 2, 3], "history_mask": [False, True, True, True], "target": 4},
        {"history_items": [0, 2, 3, 4], "history_mask": [False, True, True, True], "target": 5},
        {"history_items": [1, 2, 4, 5], "history_mask": [True, True, True, True], "target": 6},
        {"history_items": [2, 3, 5, 6], "history_mask": [True, True, True, True], "target": 7},
    ]
    eval_rows = [
        {"history_items": [0, 1, 2, 3], "history_mask": [False, True, True, True], "targets": [4, 5]},
        {"history_items": [0, 2, 3, 4], "history_mask": [False, True, True, True], "targets": [5, 6]},
    ]

    _write_jsonl(processed / "train.jsonl", train_rows)
    _write_jsonl(processed / "valid.jsonl", eval_rows)
    _write_jsonl(processed / "test.jsonl", eval_rows)

    with (processed / "book_item_map.txt").open("w", encoding="utf-8") as handle:
        for item_id in range(1, 8):
            handle.write(f"item{item_id},{item_id}\n")

    with (processed / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "num_items": 8,
                "num_train_samples": len(train_rows),
                "num_valid_samples": len(eval_rows),
                "num_test_samples": len(eval_rows),
                "maxlen": 4,
                "min_count": 2,
                "seed": 7,
            },
            handle,
        )

    return DataConfig(processed_data_dir=processed, maxlen=4)


@pytest.fixture
def toy_data_config(tmp_path: Path) -> DataConfig:
    return _write_processed_dataset(tmp_path)
