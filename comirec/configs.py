from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


def _optional_int(value: str | None) -> int | None:
    if value in {None, "", "none", "null"}:
        return None
    return int(value)


def _bool_flag(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


@dataclass(slots=True)
class DataConfig:
    raw_data_dir: Path = PROJECT_ROOT / "data" / "amazon_books"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    min_count: int = 5
    split_seed: int = 55
    maxlen: int = 20

    @property
    def reviews_path(self) -> Path:
        return self.raw_data_dir / "reviews_Books_5.json"

    @property
    def train_samples_path(self) -> Path:
        return self.processed_data_dir / "train.jsonl"

    @property
    def valid_samples_path(self) -> Path:
        return self.processed_data_dir / "valid.jsonl"

    @property
    def test_samples_path(self) -> Path:
        return self.processed_data_dir / "test.jsonl"

    @property
    def item_map_path(self) -> Path:
        return self.processed_data_dir / "book_item_map.txt"

    @property
    def metadata_path(self) -> Path:
        return self.processed_data_dir / "metadata.json"


@dataclass(slots=True)
class ModelConfig:
    embedding_dim: int = 64
    hidden_size: int = 32
    num_interests: int = 8
    padding_idx: int = 0


@dataclass(slots=True)
class TrainConfig:
    device: str = "auto"
    seed: int = 55
    batch_size: int = 48
    learning_rate: float = 5e-4
    num_epochs: int = 1
    log_every: int = 20
    eval_topk: int = 50
    eval_batch_size: int = 128
    valid_max_users: int | None = 4096
    run_test_eval: bool = False
    test_max_users: int | None = 4096
    checkpoint_path: Path = PROJECT_ROOT / "best_model" / "comirec.pt"


@dataclass(slots=True)
class EvalConfig:
    checkpoint_path: Path
    split: str = "valid"
    device: str = "auto"
    batch_size: int = 128
    topk: int = 50
    max_users: int | None = 4096


def build_prepare_parser() -> argparse.ArgumentParser:
    defaults = DataConfig()
    parser = argparse.ArgumentParser(description="Prepare Amazon Books samples.")
    parser.add_argument("--raw-data-dir", type=Path, default=defaults.raw_data_dir)
    parser.add_argument("--processed-data-dir", type=Path, default=defaults.processed_data_dir)
    parser.add_argument("--min-count", type=int, default=defaults.min_count)
    parser.add_argument("--split-seed", type=int, default=defaults.split_seed)
    parser.add_argument("--maxlen", type=int, default=defaults.maxlen)
    return parser


def build_train_parser() -> argparse.ArgumentParser:
    data_defaults = DataConfig()
    model_defaults = ModelConfig()
    train_defaults = TrainConfig()

    parser = argparse.ArgumentParser(description="Train ComiRec-SA with PyTorch.")
    parser.add_argument("--device", default=train_defaults.device)
    parser.add_argument("--seed", type=int, default=train_defaults.seed)
    parser.add_argument("--batch-size", type=int, default=train_defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=train_defaults.learning_rate)
    parser.add_argument("--num-epochs", type=int, default=train_defaults.num_epochs)
    parser.add_argument("--log-every", type=int, default=train_defaults.log_every)
    parser.add_argument("--eval-topk", type=int, default=train_defaults.eval_topk)
    parser.add_argument("--eval-batch-size", type=int, default=train_defaults.eval_batch_size)
    parser.add_argument("--valid-max-users", type=_optional_int, default=train_defaults.valid_max_users)
    parser.add_argument("--run-test-eval", type=_bool_flag, default=train_defaults.run_test_eval)
    parser.add_argument("--test-max-users", type=_optional_int, default=train_defaults.test_max_users)
    parser.add_argument("--checkpoint-path", type=Path, default=train_defaults.checkpoint_path)

    parser.add_argument("--embedding-dim", type=int, default=model_defaults.embedding_dim)
    parser.add_argument("--hidden-size", type=int, default=model_defaults.hidden_size)
    parser.add_argument("--num-interests", type=int, default=model_defaults.num_interests)
    parser.add_argument("--padding-idx", type=int, default=model_defaults.padding_idx)

    parser.add_argument("--raw-data-dir", type=Path, default=data_defaults.raw_data_dir)
    parser.add_argument("--processed-data-dir", type=Path, default=data_defaults.processed_data_dir)
    parser.add_argument("--min-count", type=int, default=data_defaults.min_count)
    parser.add_argument("--split-seed", type=int, default=data_defaults.split_seed)
    parser.add_argument("--maxlen", type=int, default=data_defaults.maxlen)
    return parser


def build_eval_parser() -> argparse.ArgumentParser:
    data_defaults = DataConfig()
    parser = argparse.ArgumentParser(description="Evaluate a trained ComiRec-SA checkpoint.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--max-users", type=_optional_int, default=4096)
    parser.add_argument("--processed-data-dir", type=Path, default=data_defaults.processed_data_dir)
    return parser


def parse_prepare_args(argv: list[str] | None = None) -> DataConfig:
    args = build_prepare_parser().parse_args(argv)
    return DataConfig(
        raw_data_dir=args.raw_data_dir,
        processed_data_dir=args.processed_data_dir,
        min_count=args.min_count,
        split_seed=args.split_seed,
        maxlen=args.maxlen,
    )


def parse_train_args(
    argv: list[str] | None = None,
) -> tuple[TrainConfig, DataConfig, ModelConfig]:
    args = build_train_parser().parse_args(argv)
    train_config = TrainConfig(
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        log_every=args.log_every,
        eval_topk=args.eval_topk,
        eval_batch_size=args.eval_batch_size,
        valid_max_users=args.valid_max_users,
        run_test_eval=args.run_test_eval,
        test_max_users=args.test_max_users,
        checkpoint_path=args.checkpoint_path,
    )
    data_config = DataConfig(
        raw_data_dir=args.raw_data_dir,
        processed_data_dir=args.processed_data_dir,
        min_count=args.min_count,
        split_seed=args.split_seed,
        maxlen=args.maxlen,
    )
    model_config = ModelConfig(
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_interests=args.num_interests,
        padding_idx=args.padding_idx,
    )
    return train_config, data_config, model_config


def parse_eval_args(argv: list[str] | None = None) -> tuple[EvalConfig, DataConfig]:
    args = build_eval_parser().parse_args(argv)
    eval_config = EvalConfig(
        checkpoint_path=args.checkpoint_path,
        split=args.split,
        device=args.device,
        batch_size=args.batch_size,
        topk=args.topk,
        max_users=args.max_users,
    )
    data_config = DataConfig(processed_data_dir=args.processed_data_dir)
    return eval_config, data_config
