from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

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
        mapped = [
            item_map[item_id]
            for item_id, _timestamp in sorted(interactions, key=lambda pair: pair[1])
            if item_id in item_map
        ]
        if len(mapped) >= min_count:
            sequences[user_id] = mapped
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


def export_map(path: Path, mapping: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for key, value in mapping.items():
            handle.write(f"{key},{value}\n")


def save_samples(path: Path, payload: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in payload:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def pad_history(sequence: list[int], cutoff: int, maxlen: int) -> tuple[list[int], list[bool]]:
    history = sequence[max(0, cutoff - maxlen) : cutoff]
    mask = [True] * len(history)
    if len(history) < maxlen:
        pad = maxlen - len(history)
        history = [0] * pad + history
        mask = [False] * pad + mask
    return history, mask


def build_train_samples(
    train_users: list[str],
    sequences: dict[str, list[int]],
    maxlen: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    samples: list[dict[str, object]] = []
    for user_id in train_users:
        sequence = sequences[user_id]
        cutoff = rng.choice(range(4, len(sequence)))
        padded_history, mask = pad_history(sequence, cutoff, maxlen)
        samples.append(
            {
                "history_items": padded_history,
                "history_mask": mask,
                "target": sequence[cutoff],
            }
        )
    return samples


def build_eval_samples(
    eval_users: list[str],
    sequences: dict[str, list[int]],
    maxlen: int,
) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for user_id in eval_users:
        sequence = sequences[user_id]
        cutoff = int(len(sequence) * 0.8)
        padded_history, mask = pad_history(sequence, cutoff, maxlen)
        samples.append(
            {
                "history_items": padded_history,
                "history_mask": mask,
                "targets": sequence[cutoff:],
            }
        )
    return samples


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
    export_map(data_config.item_map_path, item_map)

    train_samples = build_train_samples(
        train_users,
        sequences,
        maxlen=data_config.maxlen,
        seed=data_config.split_seed,
    )
    valid_samples = build_eval_samples(valid_users, sequences, maxlen=data_config.maxlen)
    test_samples = build_eval_samples(test_users, sequences, maxlen=data_config.maxlen)

    save_samples(data_config.train_samples_path, train_samples)
    save_samples(data_config.valid_samples_path, valid_samples)
    save_samples(data_config.test_samples_path, test_samples)

    metadata = {
        "num_items": len(item_map) + 1,
        "num_train_samples": len(train_samples),
        "num_valid_samples": len(valid_samples),
        "num_test_samples": len(test_samples),
        "maxlen": data_config.maxlen,
        "min_count": data_config.min_count,
        "seed": data_config.split_seed,
        "train_cutoff_strategy": "one_fixed_random_cutoff_per_user",
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
        "train_samples": metadata["num_train_samples"],
        "valid_samples": metadata["num_valid_samples"],
        "test_samples": metadata["num_test_samples"],
        "output_dir": str(data_config.processed_data_dir),
    }


def ensure_prepared(data_config: DataConfig) -> None:
    required = (
        data_config.train_samples_path,
        data_config.valid_samples_path,
        data_config.test_samples_path,
        data_config.item_map_path,
        data_config.metadata_path,
    )
    if all(path.exists() for path in required):
        return
    prepare_books(data_config)


def main(argv: list[str] | None = None) -> None:
    data_config = parse_prepare_args(argv)
    stats = prepare_books(data_config)
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
