from __future__ import annotations

from comirec.configs import EvalConfig, ModelConfig, TrainConfig
from comirec.eval import evaluate_split
from comirec.train import run_training


def test_training_and_eval_smoke(toy_data_config, tmp_path) -> None:
    model_config = ModelConfig(embedding_dim=8, hidden_size=4, num_interests=2)
    train_config = TrainConfig(
        device="cpu",
        seed=7,
        batch_size=2,
        learning_rate=1e-3,
        num_sampled=3,
        max_steps=2,
        test_every_steps=1,
        patience=3,
        log_every=1,
        metric_ks=(3,),
        eval_batch_size=2,
        valid_max_users=2,
        run_test_eval=True,
        test_max_users=2,
        checkpoint_path=tmp_path / "checkpoints" / "smoke.pt",
    )

    summary = run_training(
        train_config=train_config,
        data_config=toy_data_config,
        model_config=model_config,
    )

    assert summary["steps_completed"] == 2
    assert train_config.checkpoint_path.exists()

    valid_score = evaluate_split(
        data_config=toy_data_config,
        eval_config=EvalConfig(
            checkpoint_path=train_config.checkpoint_path,
            split="valid",
            device="cpu",
            batch_size=2,
            metric_ks=(3,),
            max_users=2,
        ),
    )
    test_score = evaluate_split(
        data_config=toy_data_config,
        eval_config=EvalConfig(
            checkpoint_path=train_config.checkpoint_path,
            split="test",
            device="cpu",
            batch_size=2,
            metric_ks=(3,),
            max_users=2,
        ),
    )

    assert 0.0 <= valid_score["recall@3"] <= 1.0
    assert 0.0 <= valid_score["ndcg@3"] <= 1.0
    assert 0.0 <= valid_score["hitrate@3"] <= 1.0
    assert 0.0 <= test_score["recall@3"] <= 1.0
    assert 0.0 <= test_score["ndcg@3"] <= 1.0
    assert 0.0 <= test_score["hitrate@3"] <= 1.0
