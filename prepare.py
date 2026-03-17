from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = ROOT / "data/amazon_books"
PROCESSED_DATA_DIR = ROOT / "data/processed"
TRAIN_SAMPLES_FILE = PROCESSED_DATA_DIR / "train.jsonl"
VALID_SAMPLES_FILE = PROCESSED_DATA_DIR / "valid.jsonl"
TEST_SAMPLES_FILE = PROCESSED_DATA_DIR / "test.jsonl"
ITEM_MAP_FILE = PROCESSED_DATA_DIR / "book_item_map.txt"
METADATA_FILE = PROCESSED_DATA_DIR / "metadata.json"
DEFAULT_MIN_COUNT = 5
DEFAULT_SPLIT_SEED = 55
DEFAULT_MAXLEN = 20


@dataclass
class Batch:
    history_items: torch.Tensor
    history_mask: torch.Tensor
    targets: torch.Tensor | list[list[int]]


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
    with path.open("w", encoding="utf-8") as handle:
        for key, value in mapping.items():
            handle.write(f"{key},{value}\n")


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


def save_samples(path: Path, payload: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in payload:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def load_samples(path: Path) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            samples.append(json.loads(line))
    return samples


def prepare_books(
    raw_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
    min_count: int = DEFAULT_MIN_COUNT,
    seed: int = DEFAULT_SPLIT_SEED,
    maxlen: int = DEFAULT_MAXLEN,
) -> dict[str, int | str]:
    reviews_path = raw_dir / "reviews_Books_5.json"
    if not reviews_path.exists():
        raise FileNotFoundError(
            f"Could not find {reviews_path}. Download Amazon Books first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    users, item_counts = load_reviews(reviews_path)
    item_map = build_item_map(item_counts, min_count=min_count)
    sequences = build_user_sequences(users, item_map=item_map, min_count=min_count)

    kept_users = list(sequences)
    train_users, valid_users, test_users = split_users(kept_users, seed=seed)
    export_map(output_dir / ITEM_MAP_FILE.name, item_map)

    train_samples = build_train_samples(train_users, sequences, maxlen=maxlen, seed=seed)
    valid_samples = build_eval_samples(valid_users, sequences, maxlen=maxlen)
    test_samples = build_eval_samples(test_users, sequences, maxlen=maxlen)

    save_samples(output_dir / TRAIN_SAMPLES_FILE.name, train_samples)
    save_samples(output_dir / VALID_SAMPLES_FILE.name, valid_samples)
    save_samples(output_dir / TEST_SAMPLES_FILE.name, test_samples)

    metadata = {
        "num_items": len(item_map) + 1,
        "num_train_samples": len(train_samples),
        "num_valid_samples": len(valid_samples),
        "num_test_samples": len(test_samples),
        "maxlen": maxlen,
        "min_count": min_count,
        "seed": seed,
        "train_cutoff_strategy": "one_fixed_random_cutoff_per_user",
    }
    with (output_dir / METADATA_FILE.name).open("w", encoding="utf-8") as handle:
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
        "output_dir": str(output_dir),
    }


def ensure_prepared() -> None:
    required = (
        TRAIN_SAMPLES_FILE,
        VALID_SAMPLES_FILE,
        TEST_SAMPLES_FILE,
        ITEM_MAP_FILE,
        METADATA_FILE,
    )
    if all(path.exists() for path in required):
        return
    stats = prepare_books()
    print("prepared_data=true")
    for key, value in stats.items():
        print(f"{key}={value}")


def count_items(item_map_path: Path = ITEM_MAP_FILE) -> int:
    max_item_id = 0
    with item_map_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            _raw_item, mapped_item = line.rstrip("\n").split(",", 1)
            max_item_id = max(max_item_id, int(mapped_item))
    return max_item_id + 1


class TrainIterator:
    def __init__(self, samples: list[dict[str, object]], batch_size: int, seed: int) -> None:
        self.samples = samples
        self.batch_size = batch_size
        self.rng = torch.Generator().manual_seed(seed)
        self.indices = torch.randperm(len(self.samples), generator=self.rng)
        self.cursor = 0

    def __iter__(self) -> TrainIterator:
        return self

    def __next__(self) -> Batch:
        if self.cursor + self.batch_size > len(self.samples):
            self.indices = torch.randperm(len(self.samples), generator=self.rng)
            self.cursor = 0

        batch_indices = self.indices[self.cursor : self.cursor + self.batch_size]
        self.cursor += self.batch_size
        batch_records = [self.samples[index] for index in batch_indices.tolist()]
        return Batch(
            history_items=torch.tensor(
                [record["history_items"] for record in batch_records],
                dtype=torch.long,
            ),
            history_mask=torch.tensor(
                [record["history_mask"] for record in batch_records],
                dtype=torch.float32,
            ),
            targets=torch.tensor(
                [record["target"] for record in batch_records],
                dtype=torch.long,
            ),
        )


class EvalIterator:
    def __init__(self, samples: list[dict[str, object]], batch_size: int) -> None:
        self.samples = samples
        self.batch_size = batch_size
        self.cursor = 0

    def __iter__(self) -> EvalIterator:
        return self

    def __next__(self) -> Batch:
        if self.cursor >= len(self.samples):
            self.cursor = 0
            raise StopIteration

        start = self.cursor
        end = min(start + self.batch_size, len(self.samples))
        self.cursor = end
        batch_records = self.samples[start:end]
        return Batch(
            history_items=torch.tensor(
                [record["history_items"] for record in batch_records],
                dtype=torch.long,
            ),
            history_mask=torch.tensor(
                [record["history_mask"] for record in batch_records],
                dtype=torch.float32,
            ),
            targets=[record["targets"] for record in batch_records],
        )


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested device 'mps', but MPS is not available.")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested device 'cuda', but CUDA is not available.")
    return torch.device(device_name)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_recall: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "best_recall": best_recall,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict[str, float | int]:
    checkpoint = torch.load(path, map_location=device or torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return {
        "step": int(checkpoint["step"]),
        "best_recall": float(checkpoint["best_recall"]),
    }


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def compute_ranking_metrics(predicted_items: list[int], true_items: list[int]) -> dict[str, float]:
    true_item_set = set(true_items)
    recall_hits = 0
    dcg = 0.0
    for rank, item_id in enumerate(predicted_items):
        if item_id in true_item_set:
            recall_hits += 1
            dcg += 1.0 / torch.log2(torch.tensor(rank + 2.0)).item()
    ideal_hits = min(len(true_items), len(predicted_items))
    idcg = sum(1.0 / torch.log2(torch.tensor(rank + 2.0)).item() for rank in range(ideal_hits))
    return {
        "recall": recall_hits / max(len(true_items), 1),
        "ndcg": (dcg / idcg) if recall_hits > 0 and idcg > 0 else 0.0,
        "hitrate": 1.0 if recall_hits > 0 else 0.0,
    }


def average_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {"recall": 0.0, "ndcg": 0.0, "hitrate": 0.0}
    keys = metrics[0].keys()
    return {key: sum(m[key] for m in metrics) / len(metrics) for key in keys}


def mask_history_items(
    item_scores: torch.Tensor,
    history_items: torch.Tensor,
    history_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, _num_interests, num_items = item_scores.shape
    seen_mask = torch.zeros((batch_size, num_items), dtype=torch.bool, device=item_scores.device)
    seen_mask.scatter_(1, history_items, history_mask.bool())
    seen_mask[:, 0] = True
    return item_scores.masked_fill(seen_mask.unsqueeze(1), float("-inf"))


def merge_topk_interests(
    top_item_ids: torch.Tensor,
    top_scores: torch.Tensor,
    topk: int,
) -> list[list[int]]:
    merged_rankings: list[list[int]] = []
    for user_item_ids, user_scores in zip(top_item_ids.tolist(), top_scores.tolist()):
        scored_pairs: list[tuple[int, float]] = []
        seen_items: set[int] = set()
        for interest_item_ids, interest_scores in zip(user_item_ids, user_scores):
            scored_pairs.extend(zip(interest_item_ids, interest_scores, strict=True))

        scored_pairs.sort(key=lambda pair: pair[1], reverse=True)
        ranking: list[int] = []
        for item_id, _score in scored_pairs:
            if item_id == 0 or item_id in seen_items:
                continue
            seen_items.add(item_id)
            ranking.append(item_id)
            if len(ranking) >= topk:
                break
        merged_rankings.append(ranking)
    return merged_rankings


def evaluate_full_ranking(
    model: torch.nn.Module,
    samples_path: Path,
    batch_size: int,
    topk: int,
    device: torch.device,
    max_users: int | None = None,
) -> dict[str, float]:
    samples = load_samples(samples_path)
    iterator = EvalIterator(samples=samples, batch_size=batch_size)
    metrics: list[dict[str, float]] = []
    processed_users = 0

    model.eval()
    with torch.no_grad():
        for batch in iterator:
            interest_embeddings = model.get_user_interest_embeddings(
                batch.history_items.to(device),
                batch.history_mask.to(device),
            )
            item_scores = model.score_all_items(interest_embeddings)
            item_scores = mask_history_items(
                item_scores=item_scores,
                history_items=batch.history_items.to(device),
                history_mask=batch.history_mask.to(device),
            )
            top_scores, top_item_ids = torch.topk(item_scores, k=topk, dim=-1)
            ranked_items = merge_topk_interests(
                top_item_ids=top_item_ids.cpu(),
                top_scores=top_scores.cpu(),
                topk=topk,
            )

            for predicted_items, true_items in zip(ranked_items, batch.targets):
                metrics.append(compute_ranking_metrics(predicted_items, true_items))
                processed_users += 1
                if max_users is not None and processed_users >= max_users:
                    return average_metrics(metrics)

    return average_metrics(metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare final Amazon Books samples.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--maxlen", type=int, default=DEFAULT_MAXLEN)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = prepare_books(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        min_count=args.min_count,
        seed=args.seed,
        maxlen=args.maxlen,
    )
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
