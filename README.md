# ComiRec

![Python](https://img.shields.io/badge/python-3.13-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.10-red)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

A learning-oriented PyTorch implementation of **ComiRec-SA** for sequential recommendation.

This repository is designed to be small, readable, and practical. It keeps the full training pipeline in plain PyTorch while avoiding framework-heavy abstractions, so you can study how a real recommendation model is built, trained, and evaluated.

## Highlights

- Plain PyTorch training loop with `zero_grad()`, `backward()`, and `step()`
- Clear separation between preprocessing, datasets, model, training, and evaluation
- Minimal project structure that is still close to a real-world repository
- Tests covering preprocessing, data loading, model forward/backward, and train/eval flow

## What This Repo Contains

- **Model**: `ComiRec-SA`
- **Dataset**: Amazon Books
- **Task**: sequential recommendation
- **Training objective**: in-batch softmax negatives
- **Primary evaluation metric**: `NDCG@50`

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

### 3. Train

```bash
uv run python -m comirec.train
```

Example:

```bash
uv run python -m comirec.train \
  --device cpu \
  --batch-size 16 \
  --num-epochs 1 \
  --valid-max-users 256
```

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
uv run pytest
```
