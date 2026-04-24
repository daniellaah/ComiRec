# ComiRec

![Python](https://img.shields.io/badge/python-3.13-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.10-red)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

A learning-oriented PyTorch implementation of **ComiRec-SA** for sequential recommendation.

This repository is designed to be small, readable, and practical. It keeps the full training pipeline in plain PyTorch while avoiding framework-heavy abstractions, so you can study how a real recommendation model is built, trained, and evaluated.

## Highlights

- Plain PyTorch training loop with `zero_grad()`, `backward()`, and `step()`
- Clear separation between preprocessing, datasets, model, training, and evaluation
- User-level sequence splits with dynamic training cutoffs and fixed 80/20 eval cutoffs
- Minimal project structure that is still close to a real-world repository
- Tests covering preprocessing, data loading, model forward/backward, and train/eval flow

## What This Repo Contains

- **Model**: `ComiRec-SA`
- **Dataset**: Amazon Books
- **Task**: sequential recommendation
- **Training objective**: sampled softmax with log-uniform negative sampling
- **Primary evaluation metrics**: `Recall`, `NDCG`, `HitRate` at `@20` and `@50`

## Repository Layout

```text
comirec/
  configs.py
  data.py
  eval.py
  model.py
  prepare.py
  train.py
  util.py
tests/
```

- [`comirec/model.py`](comirec/model.py): model definition and loss
- [`comirec/prepare.py`](comirec/prepare.py): raw data preprocessing and processed dataset export
- [`comirec/data.py`](comirec/data.py): `SequenceDataset`, batch collation, and `DataLoader`
- [`comirec/train.py`](comirec/train.py): training loop and checkpoint saving
- [`comirec/eval.py`](comirec/eval.py): offline evaluation
- [`comirec/configs.py`](comirec/configs.py): runtime configuration
- [`tests/`](tests): smoke tests for the full pipeline

## Quick Start

### 1. Install

```bash
uv sync
```

### 2. Prepare Data

```bash
uv run python -m comirec.prepare
```

This generates:

- `data/processed/train.jsonl`
- `data/processed/valid.jsonl`
- `data/processed/test.jsonl`
- `data/processed/metadata.json`
- `data/processed/book_item_map.txt`

The processed files store full user sequences. Training samples are drawn dynamically from those sequences, and histories are right-padded to match the official TensorFlow data iterator.
The default user split seed is `1230`, matching the official preprocessing script.
The default `--num-sampled 10` follows the official implementation's convention of 10 negatives per example, so a batch of 128 users samples up to 1280 negative classes.

### 3. Train

```bash
uv run python -m comirec.train
```

Training expects `data/processed/` to already exist. Run the prepare step explicitly before training.

Example:

```bash
uv run python -m comirec.train \
  --device cpu \
  --max-steps 5000 \
  --test-every-steps 1000 \
  --patience 50 \
  --metric-k 20 \
  --metric-k 50 \
  --run-test-eval true
```

The training loop is step-driven to match the official TensorFlow implementation more closely:

- it validates every `--test-every-steps`
- saves the best checkpoint by `Recall@50`
- stops early after `--patience` non-improving validations
- falls back to `--max-iter-k * 1000` total steps when `--max-steps` is not set

By default, training saves a checkpoint to:

```text
best_model/comirec.pt
```

### 4. Evaluate

```bash
uv run python -m comirec.eval --checkpoint-path best_model/comirec.pt
```

Evaluate the test split:

```bash
uv run python -m comirec.eval \
  --checkpoint-path best_model/comirec.pt \
  --split test
```

### 5. Run Tests

```bash
uv run --group dev python -m pytest
```
