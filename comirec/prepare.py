from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .configs import DataConfig, parse_prepare_args


def load_reviews(source: Path) -> tuple[dict[str, list[tuple[str, int]]], Counter[str]]:
    users: dict[str, list[tuple[str, int]]] = defaultdict(list)
    item_counts: Counter[str] = Counter()
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            user_id = record["reviewerID"]
            item_id = record["asin"]
            timestamp = int(record["unixReviewTime"])
            users[user_id].append((item_id, timestamp))
            item_counts[item_id] += 1
    return users, item_counts


def build_item_map(item_counts: Counter[str], min_count: int) -> dict[str, int]:
    ranked_items = sorted(item_counts.items(), key=lambda pair: pair[1], reverse=True)
    item_map: dict[str, int] = {}
    for index, (item_id, count) in enumerate(ranked_items, start=1):
        if count < min_count:
            break
        item_map[item_id] = index
    return item_map


def build_user_sequences(
    users: dict[str, list[tuple[str, int]]],
    item_map: dict[str, int],
    min_count: int,
) -> dict[str, list[int]]:
    sequences: dict[str, list[int]] = {}
    for user_id, interactions in users.items():
        mapped_sequence = [
            item_map[item_id]
            for item_id, _timestamp in sorted(interactions, key=lambda pair: pair[1])
            if item_id in item_map
        ]
        if len(mapped_sequence) >= min_count:
            sequences[user_id] = mapped_sequence
    return sequences


def split_users(user_ids: list[str], seed: int) -> tuple[list[str], list[str], list[str]]:
    shuffled = list(user_ids)
    random.Random(seed).shuffle(shuffled)
    split_train = int(len(shuffled) * 0.8)
    split_valid = int(len(shuffled) * 0.9)
    return (
        shuffled[:split_train],
        shuffled[split_train:split_valid],
        shuffled[split_valid:],
    )


def export_item_map(path: Path, item_map: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for raw_item_id, mapped_item_id in item_map.items():
            handle.write(f"{raw_item_id},{mapped_item_id}\n")


def save_samples(path: Path, payload: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in payload:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def build_split_records(
    user_ids: list[str],
    sequences: dict[str, list[int]],
    *,
    include_user_id: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for user_id in user_ids:
        record: dict[str, Any] = {"sequence": sequences[user_id]}
        if include_user_id:
            record["user_id"] = user_id
        records.append(record)
    return records


def prepare_books(data_config: DataConfig) -> dict[str, int | str]:
    if not data_config.reviews_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_config.reviews_path}. Download Amazon Books first."
        )

    data_config.processed_data_dir.mkdir(parents=True, exist_ok=True)
    users, item_counts = load_reviews(data_config.reviews_path)
    item_map = build_item_map(item_counts, min_count=data_config.min_count)
    sequences = build_user_sequences(users, item_map=item_map, min_count=data_config.min_count)

    kept_users = list(sequences)
    train_users, valid_users, test_users = split_users(kept_users, seed=data_config.split_seed)
    export_item_map(data_config.item_map_path, item_map)

    train_records = build_split_records(train_users, sequences, include_user_id=True)
    valid_records = build_split_records(valid_users, sequences, include_user_id=True)
    test_records = build_split_records(test_users, sequences, include_user_id=True)

    save_samples(data_config.train_samples_path, train_records)
    save_samples(data_config.valid_samples_path, valid_records)
    save_samples(data_config.test_samples_path, test_records)

    metadata = {
        "data_format_version": 2,
        "num_items": len(item_map) + 1,
        "num_train_users": len(train_records),
        "num_valid_users": len(valid_records),
        "num_test_users": len(test_records),
        "maxlen": data_config.maxlen,
        "min_count": data_config.min_count,
        "seed": data_config.split_seed,
        "padding_side": "right",
        "train_cutoff_strategy": "dynamic_random_cutoff_per_user_access",
        "eval_cutoff_strategy": "fixed_80_20_per_user",
    }
    with data_config.metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True)

    return {
        "raw_users": len(users),
        "raw_items": len(item_counts),
        "filtered_users": len(sequences),
        "filtered_items": len(item_map),
        "train_users": len(train_users),
        "valid_users": len(valid_users),
        "test_users": len(test_users),
        "train_records": metadata["num_train_users"],
        "valid_records": metadata["num_valid_users"],
        "test_records": metadata["num_test_users"],
        "output_dir": str(data_config.processed_data_dir),
    }


def main(argv: list[str] | None = None) -> None:
    data_config = parse_prepare_args(argv)
    stats = prepare_books(data_config)
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
