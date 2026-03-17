from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

import prepare


# -----------------------------------------------------------------------------
# Stable infrastructure. Do not edit for experiments.
# These define the data version, time budget, seed, device, and evaluation
# protocol so that autoresearch runs stay directly comparable.
# -----------------------------------------------------------------------------

TRAIN_DATA = prepare.TRAIN_SAMPLES_FILE
VALID_DATA = prepare.VALID_SAMPLES_FILE
TEST_DATA = prepare.TEST_SAMPLES_FILE
ITEM_MAP_FILE = prepare.ITEM_MAP_FILE
METADATA_FILE = prepare.METADATA_FILE
DEVICE = "mps"
SEED = 55
TRAIN_TIME_LIMIT_SECONDS = 10 * 60
EVAL_TOPK = 50
EVAL_BATCH_SIZE = 128
VALID_MAX_USERS = 4096
RUN_TEST_EVAL = False
TEST_MAX_USERS: int | None = 4096

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Run
RUN_NAME = "run_10m_i8_h32_b64_lr5e4"
WARMUP_STEPS = 5
MAX_STEPS = 1_000_000
LOG_EVERY = 20

# Model architecture
EMBEDDING_DIM = 64
HIDDEN_SIZE = 32
NUM_INTERESTS = 8

# Optimization
BATCH_SIZE = 64
LEARNING_RATE = 5e-4

# Output
CHECKPOINT_PATH = Path(f"best_model/{RUN_NAME}.pt")
STEP_LOG_PATH: Path | None = None
RUN_LOG_PATH = Path("tmp/runs.jsonl")

class ComiRecSA(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        hidden_size: int,
        num_interests: int,
        maxlen: int,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.num_interests = num_interests
        self.padding_idx = padding_idx

        self.item_embeddings = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.register_buffer("item_bias", torch.zeros(num_items))
        self.position_embedding = nn.Parameter(torch.empty(1, maxlen, embedding_dim))
        self.attention_hidden = nn.Linear(embedding_dim, hidden_size * 4)
        self.attention_projection = nn.Linear(hidden_size * 4, num_interests)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.item_embeddings.weight)
        nn.init.xavier_uniform_(self.position_embedding)
        nn.init.xavier_uniform_(self.attention_hidden.weight)
        nn.init.zeros_(self.attention_hidden.bias)
        nn.init.xavier_uniform_(self.attention_projection.weight)
        nn.init.zeros_(self.attention_projection.bias)
        with torch.no_grad():
            self.item_embeddings.weight[self.padding_idx].zero_()

    def get_user_interest_embeddings(
        self,
        history_items: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        item_embeddings = self.item_embeddings(history_items)
        attention_input = item_embeddings + self.position_embedding[:, : history_items.size(1), :]
        attention_hidden = torch.tanh(self.attention_hidden(attention_input))
        attention_logits = self.attention_projection(attention_hidden).transpose(1, 2)
        attention_logits = attention_logits.masked_fill(
            (history_mask <= 0).unsqueeze(1),
            torch.finfo(attention_logits.dtype).min,
        )
        attention_weights = torch.softmax(attention_logits, dim=-1)
        return attention_weights @ item_embeddings

    def get_training_user_embeddings(
        self,
        history_items: torch.Tensor,
        history_mask: torch.Tensor,
        target_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        interest_embeddings = self.get_user_interest_embeddings(history_items, history_mask)
        target_embeddings = self.item_embeddings(target_item_ids)
        interest_scores = (interest_embeddings * target_embeddings.unsqueeze(1)).sum(dim=-1)
        selected_interest_idx = torch.argmax(interest_scores, dim=1)
        batch_idx = torch.arange(history_items.size(0), device=history_items.device)
        return interest_embeddings[batch_idx, selected_interest_idx]

    def score_all_items(self, interest_embeddings: torch.Tensor) -> torch.Tensor:
        return interest_embeddings @ self.item_embeddings.weight.T + self.item_bias.view(1, 1, -1)


class InBatchSoftmaxLoss(nn.Module):
    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(
        self,
        model: ComiRecSA,
        user_embeddings: torch.Tensor,
        positive_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        item_embeddings = model.item_embeddings(positive_item_ids)
        item_bias = model.item_bias[positive_item_ids]
        logits = (user_embeddings @ item_embeddings.T + item_bias.unsqueeze(0)) / self.temperature

        duplicate_targets = positive_item_ids.unsqueeze(0) == positive_item_ids.unsqueeze(1)
        diagonal = torch.eye(
            positive_item_ids.size(0),
            dtype=torch.bool,
            device=positive_item_ids.device,
        )
        logits = logits.masked_fill(
            duplicate_targets & ~diagonal,
            torch.finfo(logits.dtype).min,
        )

        labels = torch.arange(user_embeddings.size(0), device=user_embeddings.device)
        return F.cross_entropy(logits, labels)


def read_metadata(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    torch.manual_seed(SEED)
    prepare.ensure_prepared()

    train_samples = prepare.load_samples(TRAIN_DATA)
    metadata = read_metadata(METADATA_FILE)
    num_items = prepare.count_items(ITEM_MAP_FILE)
    device = prepare.resolve_device(DEVICE)
    maxlen = int(metadata["maxlen"])

    model = ComiRecSA(
        num_items=num_items,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_interests=NUM_INTERESTS,
        maxlen=maxlen,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = InBatchSoftmaxLoss()
    train_iterator = prepare.TrainIterator(
        samples=train_samples,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    run_started_at = utc_now_iso()
    train_py_hash = file_sha256(Path(__file__).resolve())
    prepare_py_hash = file_sha256(Path(prepare.__file__).resolve())
    param_count = count_parameters(model)
    ema_loss: float | None = None
    timed_training_started_at: float | None = None
    training_seconds = 0.0
    step = 0

    print(f"run_name={RUN_NAME}")
    print(f"run_started_at={run_started_at}")
    print(f"device={device}")
    print(f"num_items={num_items}")
    print(f"num_train_samples={len(train_samples)}")
    print(f"num_parameters={param_count}")
    print("model=ComiRec-SA")
    print("loss=in_batch")
    print("readout=hard")
    print(f"time_budget_seconds={TRAIN_TIME_LIMIT_SECONDS}")
    print(f"warmup_steps={WARMUP_STEPS}")
    print(f"valid_max_users={VALID_MAX_USERS}")
    print(f"run_test_eval={RUN_TEST_EVAL}")

    while step < MAX_STEPS:
        step += 1
        optimizer.zero_grad()
        batch = next(train_iterator)

        user_embeddings = model.get_training_user_embeddings(
            batch.history_items.to(device),
            batch.history_mask.to(device),
            batch.targets.to(device),
        )
        loss = criterion(model, user_embeddings, batch.targets.to(device))
        loss.backward()
        optimizer.step()
        synchronize_device(device)

        loss_value = float(loss.item())
        if ema_loss is None:
            ema_loss = loss_value
        else:
            ema_loss = 0.95 * ema_loss + 0.05 * loss_value

        now = time.perf_counter()
        if timed_training_started_at is None and step >= WARMUP_STEPS:
            timed_training_started_at = now
        if timed_training_started_at is not None:
            training_seconds = now - timed_training_started_at

        if step == 1 or step % LOG_EVERY == 0:
            remaining_seconds = max(TRAIN_TIME_LIMIT_SECONDS - training_seconds, 0.0)
            print(
                f"step={step} "
                f"loss={loss_value:.6f} "
                f"ema_loss={ema_loss:.6f} "
                f"train_seconds={training_seconds:.1f} "
                f"remaining_seconds={remaining_seconds:.1f}"
            )
            if STEP_LOG_PATH is not None:
                prepare.append_jsonl(
                    STEP_LOG_PATH,
                    {
                        "run_name": RUN_NAME,
                        "step": step,
                        "loss": round(loss_value, 6),
                        "ema_loss": round(ema_loss, 6),
                        "train_seconds": round(training_seconds, 2),
                        "remaining_seconds": round(remaining_seconds, 2),
                    },
                )

        if timed_training_started_at is not None and training_seconds >= TRAIN_TIME_LIMIT_SECONDS:
            break

    synchronize_device(device)
    training_finished_at = utc_now_iso()

    valid_ndcg50 = prepare.evaluate_ndcg50(
        model=model,
        samples_path=VALID_DATA,
        batch_size=EVAL_BATCH_SIZE,
        topk=EVAL_TOPK,
        device=device,
        max_users=VALID_MAX_USERS,
    )
    best_metric = float(valid_ndcg50)
    print(f"valid_ndcg50={valid_ndcg50:.6f}")

    test_ndcg50: float | None = None
    if RUN_TEST_EVAL:
        test_ndcg50 = prepare.evaluate_ndcg50(
            model=model,
            samples_path=TEST_DATA,
            batch_size=EVAL_BATCH_SIZE,
            topk=EVAL_TOPK,
            device=device,
            max_users=TEST_MAX_USERS,
        )
        print(f"test_ndcg50={test_ndcg50:.6f}")

    prepare.save_checkpoint(
        path=CHECKPOINT_PATH,
        model=model,
        optimizer=optimizer,
        step=step,
        best_metric=best_metric,
    )

    summary = {
        "run_name": RUN_NAME,
        "run_started_at": run_started_at,
        "training_finished_at": training_finished_at,
        "train_py_sha256": train_py_hash,
        "prepare_py_sha256": prepare_py_hash,
        "dataset": "amazon_books",
        "model": "ComiRec-SA",
        "loss": "in_batch",
        "readout": "hard",
        "primary_metric_key": "ndcg50",
        "primary_metric_label": "NDCG@50",
        "device": str(device),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "train_data": str(TRAIN_DATA),
        "valid_data": str(VALID_DATA),
        "test_data": str(TEST_DATA),
        "metadata_file": str(METADATA_FILE),
        "num_items": num_items,
        "num_parameters": param_count,
        "num_train_samples": len(train_samples),
        "config": {
            "batch_size": BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "embedding_dim": EMBEDDING_DIM,
            "hidden_size": HIDDEN_SIZE,
            "num_interests": NUM_INTERESTS,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "time_budget_seconds": TRAIN_TIME_LIMIT_SECONDS,
            "warmup_steps": WARMUP_STEPS,
            "max_steps": MAX_STEPS,
            "log_every": LOG_EVERY,
            "valid_max_users": VALID_MAX_USERS,
            "run_test_eval": RUN_TEST_EVAL,
            "test_max_users": TEST_MAX_USERS,
        },
        "prepared_data": metadata,
        "result": {
            "steps_completed": step,
            "training_seconds": round(training_seconds, 2),
            "examples_seen": step * BATCH_SIZE,
            "examples_per_second": round((step * BATCH_SIZE) / max(training_seconds, 1e-8), 2),
            "final_loss": round(loss_value, 6),
            "final_ema_loss": round(ema_loss or loss_value, 6),
            "valid_primary_metric": round(best_metric, 6),
            "valid_ndcg50": round(valid_ndcg50, 6),
            "test_primary_metric": round(test_ndcg50, 6) if test_ndcg50 is not None else None,
            "test_ndcg50": round(test_ndcg50, 6) if test_ndcg50 is not None else None,
        },
    }
    prepare.append_jsonl(RUN_LOG_PATH, summary)
    print(f"checkpoint_saved={CHECKPOINT_PATH}")
    print(f"run_log_appended={RUN_LOG_PATH}")


if __name__ == "__main__":
    main()
