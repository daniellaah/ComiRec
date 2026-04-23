from __future__ import annotations

import time
from typing import Any

import torch

from .configs import DataConfig, ModelConfig, TrainConfig, parse_train_args
from .data import count_items, create_train_loader, read_metadata
from .eval import evaluate_ndcg50
from .model import InBatchSoftmaxLoss, build_model
from .prepare import ensure_prepared
from .util import count_parameters, resolve_device, save_checkpoint, seed_everything, synchronize_device


def train_one_step(
    model: torch.nn.Module,
    batch: Any,
    optimizer: torch.optim.Optimizer,
    criterion: InBatchSoftmaxLoss,
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


def run_training(
    train_config: TrainConfig,
    data_config: DataConfig,
    model_config: ModelConfig,
) -> dict[str, Any]:
    seed_everything(train_config.seed)
    ensure_prepared(data_config)

    metadata = read_metadata(data_config.metadata_path)
    num_items = count_items(data_config.item_map_path)
    device = resolve_device(train_config.device)

    model = build_model(
        num_items=num_items,
        maxlen=int(metadata["maxlen"]),
        model_config=model_config,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    criterion = InBatchSoftmaxLoss()
    train_loader = create_train_loader(
        samples_path=data_config.train_samples_path,
        batch_size=train_config.batch_size,
        seed=train_config.seed,
    )

    param_count = count_parameters(model)
    ema_loss: float | None = None
    loss_value = 0.0
    step = 0

    print(f"device={device}")
    print(f"num_items={num_items}")
    print(f"num_train_samples={len(train_loader.dataset)}")
    print(f"num_parameters={param_count}")
    print(f"num_epochs={train_config.num_epochs}")

    started_at = time.perf_counter()
    for epoch in range(train_config.num_epochs):
        for batch in train_loader:
            step += 1
            loss_value = train_one_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )
            ema_loss = loss_value if ema_loss is None else 0.95 * ema_loss + 0.05 * loss_value

            if step == 1 or step % train_config.log_every == 0:
                print(
                    f"epoch={epoch + 1} "
                    f"step={step} "
                    f"loss={loss_value:.6f} "
                    f"ema_loss={ema_loss:.6f}"
                )

    synchronize_device(device)
    training_seconds = time.perf_counter() - started_at

    valid_ndcg50 = evaluate_ndcg50(
        model=model,
        samples_path=data_config.valid_samples_path,
        batch_size=train_config.eval_batch_size,
        topk=train_config.eval_topk,
        device=device,
        max_users=train_config.valid_max_users,
    )
    best_metric = float(valid_ndcg50)
    print(f"valid_ndcg50={valid_ndcg50:.6f}")

    test_ndcg50: float | None = None
    if train_config.run_test_eval:
        test_ndcg50 = evaluate_ndcg50(
            model=model,
            samples_path=data_config.test_samples_path,
            batch_size=train_config.eval_batch_size,
            topk=train_config.eval_topk,
            device=device,
            max_users=train_config.test_max_users,
        )
        print(f"test_ndcg50={test_ndcg50:.6f}")

    save_checkpoint(
        path=train_config.checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=step,
        best_metric=best_metric,
        extras={
            "model_config": {
                "embedding_dim": model_config.embedding_dim,
                "hidden_size": model_config.hidden_size,
                "num_interests": model_config.num_interests,
                "padding_idx": model_config.padding_idx,
            },
        },
    )
    print(f"checkpoint_saved={train_config.checkpoint_path}")
    return {
        "steps_completed": step,
        "training_seconds": round(training_seconds, 2),
        "final_loss": round(loss_value, 6),
        "final_ema_loss": round(ema_loss or loss_value, 6),
        "valid_ndcg50": round(valid_ndcg50, 6),
        "test_ndcg50": round(test_ndcg50, 6) if test_ndcg50 is not None else None,
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
