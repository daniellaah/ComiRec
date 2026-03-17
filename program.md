# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify prepared data exists**: Check that these files exist:
   - `data/processed/train.jsonl`
   - `data/processed/valid.jsonl`
   - `data/processed/test.jsonl`
   - `data/processed/metadata.json`
   - `data/processed/book_item_map.txt`
   If any are missing, tell the human to run `uv run python prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on `mps`. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. The default target is the hyperparameter block at the top, but you may also modify the model architecture or training logic inside `train.py` if that is the experimental idea.
- For each experiment, set a unique `RUN_NAME` in `train.py` so checkpoints and run records do not overwrite each other.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed data preparation, data loading, batching, checkpoint helpers, and offline evaluation.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness or prepared data protocol. The `evaluate_ndcg50` function in `prepare.py` is the ground truth evaluation entry point.
- Modify the stable infrastructure section at the top of `train.py`. Data paths, seed, device, training time budget, and evaluation protocol must stay fixed so runs remain comparable.

**The goal is simple: get the highest `valid_ndcg@50`.** This is the one primary metric used for autoresearch comparisons. `Recall@50` and `HitRate@50` may still be logged for context, but keep/discard decisions should be based on validation NDCG under the same 5-minute budget. The only constraint is that the code runs without crashing and finishes within the time budget.

**Memory** is a soft constraint. Some increase is acceptable for meaningful NDCG gains, but it should not blow up dramatically or make the run unstable on the current machine.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A tiny NDCG gain that adds a lot of hacky code is probably not worth it. A tiny gain from deleting code probably is.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
valid_ndcg50:     0.023753
training_seconds: 300.1
num_steps:        953
num_params_M:     23.6
run_name:         baseline
checkpoint:       best_model/baseline.pt
```


Notes:

- The key comparison field is always `valid_ndcg50`.
- `training_seconds` and `num_steps` help judge training efficiency under the fixed 5-minute budget.
- `num_params_M` helps compare model size when architecture changes.
- `tmp/runs.jsonl` remains the source of truth for the full structured record.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	valid_ndcg50	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. `valid_ndcg50` achieved (e.g. 0.023753) — use 0.000000 for crashes
3. peak memory in GB, round to .1f if you measured it; otherwise use 0.0. Use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	valid_ndcg50	memory_gb	status	description
a1b2c3d	0.023753	0.0	keep	baseline
b2c3d4e	0.024981	0.0	keep	increase number of interests to 6
c3d4e5f	0.022410	0.0	discard	double learning rate
d4e5f6g	0.000000	0.0	crash	double embedding dimension (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^valid_ndcg50=" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If `valid_ndcg50` improved (higher), you "advance" the branch, keeping the git commit
9. If `valid_ndcg50` is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
