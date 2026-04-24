from __future__ import annotations

import json

from comirec.configs import DataConfig
from comirec.prepare import prepare_books


def test_prepare_books_creates_processed_files(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)

    reviews = [
        {"reviewerID": "u1", "asin": "a1", "unixReviewTime": 1},
        {"reviewerID": "u1", "asin": "a2", "unixReviewTime": 2},
        {"reviewerID": "u1", "asin": "a3", "unixReviewTime": 3},
        {"reviewerID": "u1", "asin": "a4", "unixReviewTime": 4},
        {"reviewerID": "u1", "asin": "a5", "unixReviewTime": 5},
        {"reviewerID": "u2", "asin": "a1", "unixReviewTime": 1},
        {"reviewerID": "u2", "asin": "a2", "unixReviewTime": 2},
        {"reviewerID": "u2", "asin": "a3", "unixReviewTime": 3},
        {"reviewerID": "u2", "asin": "a4", "unixReviewTime": 4},
        {"reviewerID": "u2", "asin": "a5", "unixReviewTime": 5},
        {"reviewerID": "u3", "asin": "a1", "unixReviewTime": 1},
        {"reviewerID": "u3", "asin": "a2", "unixReviewTime": 2},
        {"reviewerID": "u3", "asin": "a3", "unixReviewTime": 3},
        {"reviewerID": "u3", "asin": "a4", "unixReviewTime": 4},
        {"reviewerID": "u3", "asin": "a5", "unixReviewTime": 5},
    ]
    with (raw_dir / "reviews_Books_5.json").open("w", encoding="utf-8") as handle:
        for record in reviews:
            handle.write(json.dumps(record) + "\n")

    data_config = DataConfig(
        raw_data_dir=raw_dir,
        processed_data_dir=processed_dir,
        min_count=1,
        split_seed=7,
        maxlen=4,
    )
    stats = prepare_books(data_config)

    assert stats["filtered_users"] == 3
    assert data_config.train_samples_path.exists()
    assert data_config.valid_samples_path.exists()
    assert data_config.test_samples_path.exists()
    assert data_config.item_map_path.exists()
    assert data_config.metadata_path.exists()

    with data_config.metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["data_format_version"] == 2
    assert metadata["padding_side"] == "right"
