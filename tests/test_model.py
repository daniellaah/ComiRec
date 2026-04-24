from __future__ import annotations

import torch

from comirec.configs import ModelConfig
from comirec.data import create_train_loader
from comirec.model import SampledSoftmaxLoss, build_model


def test_model_forward_and_loss_backward(toy_data_config) -> None:
    model_config = ModelConfig(embedding_dim=8, hidden_size=4, num_interests=2)
    batch = next(
        iter(create_train_loader(toy_data_config.train_samples_path, batch_size=2, seed=7, maxlen=4))
    )

    model = build_model(num_items=8, maxlen=4, model_config=model_config)
    interest_embeddings = model(batch.history_items, batch.history_mask)
    assert interest_embeddings.shape == (2, 2, 8)

    user_embeddings = model(batch.history_items, batch.history_mask, batch.targets)
    assert user_embeddings.shape == (2, 8)

    loss = SampledSoftmaxLoss(num_items=8, num_sampled=3)(model, user_embeddings, batch.targets)
    loss.backward()
    assert model.item_embeddings.weight.grad is not None


def test_sampled_softmax_unique_sampler_reports_unique_ids() -> None:
    criterion = SampledSoftmaxLoss(num_items=16, num_sampled=2)
    sampled_ids, num_tries = criterion._sample_unique_candidate_ids(
        device=torch.device("cpu"),
        num_sampled=6,
    )
    assert sampled_ids.shape == (6,)
    assert len(set(sampled_ids.tolist())) == 6
    assert num_tries >= 6


def test_expected_count_helper_matches_linear_shortcut() -> None:
    criterion = SampledSoftmaxLoss(num_items=16, num_sampled=2)
    probabilities = torch.tensor([0.1, 0.2], dtype=torch.float32)
    expected_counts = criterion._expected_counts(
        probabilities,
        batch_size=4,
        num_sampled=4,
    )
    assert torch.allclose(expected_counts, probabilities * 4)
