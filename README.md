# ComiRec Autoresearch

Minimal PyTorch repo for learning recommender-system training with:

- dataset: Amazon Books
- model: `ComiRec-SA`
- loss: in-batch negatives
- readout: hard
- device default: `mps -> cuda -> cpu`

Files that matter:

- `prepare.py`
- `train.py`
- `program.md`
- `pyproject.toml`

Prepare final datasets:

```bash
uv run python prepare.py
```

This writes stable experiment inputs to `data/processed/`:

- `train.jsonl`
- `valid.jsonl`
- `test.jsonl`
- `metadata.json`

Current training-data policy:

- each training user contributes exactly one fixed training sample
- the cutoff is sampled once in `prepare.py` with a fixed seed
- training data is therefore stable across runs

Train:

```bash
uv run python train.py
```

`train.py` now follows the `autoresearch` pattern:

- edit the config block at the top of the file
- train for a fixed wall-clock budget
- run one standardized validation pass at the end
- append one structured run record to `tmp/runs.jsonl`
- keep data paths and prepared-data-dependent settings outside the editable block
- keep seed, device, time budget, and evaluation protocol fixed across runs

Important training knobs now include:

- `RUN_NAME`
- `WARMUP_STEPS`
- `BATCH_SIZE`
- `LEARNING_RATE`
- `CHECKPOINT_PATH`

Not experiment knobs:

- dataset paths
- `maxlen`
- `topk`
- seed
- device
- train time budget
- validation/test user caps
- evaluation batch size

These are treated as stable infrastructure and fixed as part of the project protocol.
