from __future__ import annotations

from comirec.data import create_eval_loader, create_train_loader


def test_data_loaders_emit_expected_shapes(toy_data_config) -> None:
    train_batch = next(
        iter(create_train_loader(toy_data_config.train_samples_path, batch_size=2, seed=7))
    )
    assert train_batch.history_items.shape == (2, 4)
    assert train_batch.history_mask.shape == (2, 4)
    assert train_batch.targets.shape == (2,)

    eval_batch = next(iter(create_eval_loader(toy_data_config.valid_samples_path, batch_size=2)))
    assert eval_batch.history_items.shape == (2, 4)
    assert eval_batch.history_mask.shape == (2, 4)
    assert len(eval_batch.targets) == 2
