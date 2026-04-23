from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .configs import DataConfig, EvalConfig, ModelConfig, parse_eval_args
from .data import create_eval_loader, read_metadata
from .model import build_model
from .util import load_checkpoint, resolve_device


def compute_ndcg_at_k(predicted_items: list[int], true_items: list[int]) -> float:
    true_item_set = set(true_items)
    dcg = 0.0
    for rank, item_id in enumerate(predicted_items):
        if item_id in true_item_set:
            dcg += 1.0 / torch.log2(torch.tensor(rank + 2.0)).item()
    ideal_hits = min(len(true_items), len(predicted_items))
    idcg = sum(1.0 / torch.log2(torch.tensor(rank + 2.0)).item() for rank in range(ideal_hits))
    return (dcg / idcg) if dcg > 0 and idcg > 0 else 0.0


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


def evaluate_ndcg50(
    model: torch.nn.Module,
    samples_path: Path,
    batch_size: int,
    topk: int,
    device: torch.device,
    max_users: int | None = None,
) -> float:
    loader = create_eval_loader(samples_path=samples_path, batch_size=batch_size)
    ndcg_values: list[float] = []
    processed_users = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            interest_embeddings = model(
                batch.history_items.to(device),
                batch.history_mask.to(device),
            )
            item_scores = model.score_all_items(interest_embeddings)
            item_scores = mask_history_items(
                item_scores=item_scores,
                history_items=batch.history_items.to(device),
                history_mask=batch.history_mask.to(device),
            )
            effective_topk = min(topk, item_scores.size(-1))
            top_scores, top_item_ids = torch.topk(item_scores, k=effective_topk, dim=-1)
            ranked_items = merge_topk_interests(
                top_item_ids=top_item_ids.cpu(),
                top_scores=top_scores.cpu(),
                topk=effective_topk,
            )

            for predicted_items, true_items in zip(ranked_items, batch.targets):
                ndcg_values.append(compute_ndcg_at_k(predicted_items, true_items))
                processed_users += 1
                if max_users is not None and processed_users >= max_users:
                    return sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0.0

    return sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0.0


def _resolve_split_path(data_config: DataConfig, split: str) -> Path:
    if split == "valid":
        return data_config.valid_samples_path
    if split == "test":
        return data_config.test_samples_path
    raise ValueError(f"Unsupported split: {split}")


def _checkpoint_model_config(checkpoint: dict[str, Any]) -> ModelConfig:
    payload = checkpoint.get("model_config")
    if not isinstance(payload, dict):
        raise KeyError("Checkpoint is missing 'model_config'.")
    return ModelConfig(**payload)


def evaluate_split(data_config: DataConfig, eval_config: EvalConfig) -> float:
    device = resolve_device(eval_config.device)
    checkpoint = load_checkpoint(eval_config.checkpoint_path, map_location=device)
    model_config = _checkpoint_model_config(checkpoint)
    metadata = read_metadata(data_config.metadata_path)
    model = build_model(
        num_items=int(metadata["num_items"]),
        maxlen=int(metadata["maxlen"]),
        model_config=model_config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    score = evaluate_ndcg50(
        model=model,
        samples_path=_resolve_split_path(data_config, eval_config.split),
        batch_size=eval_config.batch_size,
        topk=eval_config.topk,
        device=device,
        max_users=eval_config.max_users,
    )
    return score


def main(argv: list[str] | None = None) -> None:
    eval_config, data_config = parse_eval_args(argv)
    score = evaluate_split(data_config=data_config, eval_config=eval_config)
    print(f"{eval_config.split}_ndcg{eval_config.topk}={score:.6f}")


if __name__ == "__main__":
    main()
