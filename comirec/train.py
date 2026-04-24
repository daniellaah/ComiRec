from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from .configs import DataConfig, ModelConfig, TrainConfig, parse_train_args
from .data import count_items, create_train_loader, read_metadata
from .eval import evaluate_ranking_metrics
from .model import SampledSoftmaxLoss, build_model
from .util import (
    count_parameters,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    seed_everything,
    synchronize_device,
)


def train_one_step(
    model: torch.nn.Module,
    batch: Any,
    optimizer: torch.optim.Optimizer,
    criterion: SampledSoftmaxLoss,
    device: torch.device,
) -> float:
    model.train()
    optimizer.zero_grad()

    history_items = batch.history_items.to(device)
    history_mask = batch.history_mask.to(device)
    targets = batch.targets.to(device)

    user_embeddings = model(history_items, history_mask, targets)
    loss = criterion(model, user_embeddings, targets)
    loss.backward()
    optimizer.step()
    synchronize_device(device)
    return float(loss.item())


def _print_metrics(prefix: str, metrics: dict[str, float]) -> None:
    for metric_name, value in sorted(metrics.items()):
        print(f"{prefix}_{metric_name.replace('@', '')}={value:.6f}")


def _load_model_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


def _checkpoint_extras(
    train_config: TrainConfig,
    model_config: ModelConfig,
    *,
    best_metric_k: int,
    best_metric_name: str,
) -> dict[str, Any]:
    return {
        "train_config": {
            "num_sampled": train_config.num_sampled,
            "best_metric_k": best_metric_k,
            "best_metric_name": best_metric_name,
            "max_iter_k": train_config.max_iter_k,
            "max_steps": train_config.max_steps,
            "test_every_steps": train_config.test_every_steps,
            "patience": train_config.patience,
            "metric_ks": list(train_config.metric_ks),
        },
        "model_config": {
            "embedding_dim": model_config.embedding_dim,
            "hidden_size": model_config.hidden_size,
            "num_interests": model_config.num_interests,
            "padding_idx": model_config.padding_idx,
        },
    }


def run_training(
    train_config: TrainConfig,
    data_config: DataConfig,
    model_config: ModelConfig,
) -> dict[str, Any]:
    seed_everything(train_config.seed)

    metadata = read_metadata(data_config.metadata_path)
    num_items = count_items(data_config.item_map_path)
    device = resolve_device(train_config.device)

    model = build_model(
        num_items=num_items,
        maxlen=int(metadata["maxlen"]),
        model_config=model_config,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    criterion = SampledSoftmaxLoss(num_items=num_items, num_sampled=train_config.num_sampled)
    train_loader = create_train_loader(
        samples_path=data_config.train_samples_path,
        batch_size=train_config.batch_size,
        seed=train_config.seed,
        maxlen=int(metadata["maxlen"]),
    )

    param_count = count_parameters(model)
    ema_loss: float | None = None
    loss_value = 0.0
    loss_window_sum = 0.0
    loss_window_count = 0
    step = 0
    eval_count = 0
    trials = 0
    early_stopped = False
    best_step = 0

    effective_max_steps = (
        train_config.max_steps
        if train_config.max_steps is not None
        else train_config.max_iter_k * 1000
    )
    best_metric_k = 50 if 50 in train_config.metric_ks else max(train_config.metric_ks)
    best_metric_name = f"recall@{best_metric_k}"
    best_metric = float("-inf")

    print(f"device={device}")
    print(f"num_items={num_items}")
    print(f"num_train_users={len(train_loader.dataset)}")
    print(f"num_parameters={param_count}")
    print(f"num_sampled_per_example={train_config.num_sampled}")
    print(f"max_steps={effective_max_steps}")
    print(f"test_every_steps={train_config.test_every_steps}")
    print(f"patience={train_config.patience}")

    started_at = time.perf_counter()
    train_loader_iter = iter(train_loader)
    while step < effective_max_steps:
        try:
            batch = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            batch = next(train_loader_iter)

        step += 1
        loss_value = train_one_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        loss_window_sum += loss_value
        loss_window_count += 1
        ema_loss = loss_value if ema_loss is None else 0.95 * ema_loss + 0.05 * loss_value

        if step == 1 or step % train_config.log_every == 0:
            print(
                f"step={step} "
                f"loss={loss_value:.6f} "
                f"ema_loss={ema_loss:.6f}"
            )

        if step % train_config.test_every_steps != 0 and step < effective_max_steps:
            continue

        eval_count += 1
        valid_metrics = evaluate_ranking_metrics(
            model=model,
            samples_path=data_config.valid_samples_path,
            batch_size=train_config.eval_batch_size,
            metric_ks=train_config.metric_ks,
            device=device,
            maxlen=int(metadata["maxlen"]),
            max_users=train_config.valid_max_users,
        )
        average_window_loss = loss_window_sum / max(1, loss_window_count)
        print(f"eval={eval_count} step={step} window_loss={average_window_loss:.6f}")
        _print_metrics("valid", valid_metrics)

        current_metric = float(valid_metrics[best_metric_name])
        if current_metric > best_metric:
            best_metric = current_metric
            best_step = step
            trials = 0
            save_checkpoint(
                path=train_config.checkpoint_path,
                model=model,
                optimizer=optimizer,
                step=step,
                best_metric=best_metric,
                extras=_checkpoint_extras(
                    train_config,
                    model_config,
                    best_metric_k=best_metric_k,
                    best_metric_name=best_metric_name,
                ),
            )
            print(f"checkpoint_saved={train_config.checkpoint_path}")
        else:
            trials += 1
            if trials > train_config.patience:
                early_stopped = True
                break

        loss_window_sum = 0.0
        loss_window_count = 0

    synchronize_device(device)
    training_seconds = time.perf_counter() - started_at

    if best_metric == float("-inf"):
        save_checkpoint(
            path=train_config.checkpoint_path,
            model=model,
            optimizer=optimizer,
            step=step,
            best_metric=float("nan"),
            extras=_checkpoint_extras(
                train_config,
                model_config,
                best_metric_k=best_metric_k,
                best_metric_name=best_metric_name,
            ),
        )
        print(f"checkpoint_saved={train_config.checkpoint_path}")

    _load_model_checkpoint(model, train_config.checkpoint_path, device)

    valid_metrics = evaluate_ranking_metrics(
        model=model,
        samples_path=data_config.valid_samples_path,
        batch_size=train_config.eval_batch_size,
        metric_ks=train_config.metric_ks,
        device=device,
        maxlen=int(metadata["maxlen"]),
        max_users=train_config.valid_max_users,
    )
    _print_metrics("valid", valid_metrics)

    test_metrics: dict[str, float] | None = None
    if train_config.run_test_eval:
        test_metrics = evaluate_ranking_metrics(
            model=model,
            samples_path=data_config.test_samples_path,
            batch_size=train_config.eval_batch_size,
            metric_ks=train_config.metric_ks,
            device=device,
            maxlen=int(metadata["maxlen"]),
            max_users=train_config.test_max_users,
        )
        _print_metrics("test", test_metrics)

    return {
        "steps_completed": step,
        "eval_count": eval_count,
        "best_step": best_step,
        "best_metric_name": best_metric_name,
        "best_metric_value": round(best_metric, 6) if best_metric != float("-inf") else None,
        "early_stopped": early_stopped,
        "training_seconds": round(training_seconds, 2),
        "final_loss": round(loss_value, 6),
        "final_ema_loss": round(ema_loss or loss_value, 6),
        "valid_metrics": {key: round(value, 6) for key, value in valid_metrics.items()},
        "test_metrics": (
            {key: round(value, 6) for key, value in test_metrics.items()}
            if test_metrics is not None
            else None
        ),
        "checkpoint_path": str(train_config.checkpoint_path),
    }


def main(argv: list[str] | None = None) -> None:
    train_config, data_config, model_config = parse_train_args(argv)
    run_training(
        train_config=train_config,
        data_config=data_config,
        model_config=model_config,
    )


if __name__ == "__main__":
    main()
