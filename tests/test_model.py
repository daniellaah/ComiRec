from __future__ import annotations

from comirec.configs import ModelConfig
from comirec.data import create_train_loader
from comirec.model import InBatchSoftmaxLoss, build_model


def test_model_forward_and_loss_backward(toy_data_config) -> None:
    model_config = ModelConfig(embedding_dim=8, hidden_size=4, num_interests=2)
    batch = next(iter(create_train_loader(toy_data_config.train_samples_path, batch_size=2, seed=7)))

    model = build_model(num_items=8, maxlen=4, model_config=model_config)
    interest_embeddings = model(
        batch.history_items,
        batch.history_mask,
    )
    assert interest_embeddings.shape == (2, 2, 8)

    user_embeddings = model(
        batch.history_items,
        batch.history_mask,
        batch.targets,
    )
    assert user_embeddings.shape == (2, 8)

    loss = InBatchSoftmaxLoss()(model, user_embeddings, batch.targets)
    loss.backward()
    assert model.item_embeddings.weight.grad is not None
