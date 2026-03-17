# autoresearch

This repository is a minimal `autoresearch`-style experiment for recommender systems.

The current fixed setup is:

- dataset: Amazon Books
- model family: `ComiRec-SA`
- training loss: in-batch negatives
- readout: hard
- primary metric: `valid_ndcg50`
- device protocol: `mps`
- time budget: 5 minutes per run

The repository is intentionally small:

- `prepare.py`
  - fixed data preparation
  - final dataset generation
  - dataloading
  - checkpoint helpers
  - evaluation via `evaluate_ndcg50`
  - treat this file as stable infrastructure
- `train.py`
  - the main experiment file
  - model architecture
  - optimizer
  - training loop
  - the hyperparameter block at the top is the main place to edit
- `program.md`
  - instructions for an autonomous research agent
- `pyproject.toml`
  - dependencies only

## Prepare

Prepare the final datasets:

```bash
uv run python prepare.py
```

This writes the fixed experiment inputs to `data/processed/`:

- `train.jsonl`
- `valid.jsonl`
- `test.jsonl`
- `metadata.json`
- `book_item_map.txt`

Training data is fixed across runs:

- each training user contributes one fixed training sample
- the cutoff is chosen once in `prepare.py`
- the seed is fixed

## Train

Run one experiment:

```bash
uv run python train.py
```

The training script follows the `autoresearch` pattern:

- edit `train.py`
- run for a fixed 5-minute budget
- compare runs using `valid_ndcg50`
- write a checkpoint to `best_model/`
- append a structured run record to `tmp/runs.jsonl`

The key rule is:

- `prepare.py` defines the fixed protocol
- `train.py` is where experiments happen
