from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Autoresearch Experiment Analysis
# Analysis of autonomous hyperparameter tuning results from results.tsv.

# Load the TSV (tab-separated, 5 columns: commit, valid_ndcg50, memory_gb, status, description)
df = pd.read_csv("results.tsv", sep="\t")
df["valid_ndcg50"] = pd.to_numeric(df["valid_ndcg50"], errors="coerce")
df["memory_gb"] = pd.to_numeric(df["memory_gb"], errors="coerce")
df["status"] = df["status"].str.strip().str.upper()

print(f"Total experiments: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(df.head(10).to_string(index=False))

counts = df["status"].value_counts()
print("Experiment outcomes:")
print(counts.to_string())

n_keep = counts.get("KEEP", 0)
n_discard = counts.get("DISCARD", 0)
n_crash = counts.get("CRASH", 0)
n_decided = n_keep + n_discard
if n_decided > 0:
    print(f"\nKeep rate: {n_keep}/{n_decided} = {n_keep / n_decided:.1%}")

# Show all KEPT experiments (the improvements that stuck)
kept = df[df["status"] == "KEEP"].copy()
print(f"KEPT experiments ({len(kept)} total):\n")
for i, row in kept.iterrows():
    ndcg50 = row["valid_ndcg50"]
    desc = row["description"]
    print(f"  #{i:3d}  ndcg50={ndcg50:.6f}  mem={row['memory_gb']:.1f}GB  {desc}")

# NDCG@50 Over Time
# Track how the best kept valid_ndcg50 evolves as experiments progress.

fig, ax = plt.subplots(figsize=(16, 8))

# Filter out crashes for plotting
valid = df[df["status"] != "CRASH"].copy()
valid = valid.reset_index(drop=True)

baseline_ndcg50 = valid.loc[0, "valid_ndcg50"]

# Only plot points at or above baseline (the interesting region)
above = valid[valid["valid_ndcg50"] >= baseline_ndcg50 - 0.0005]

# Plot discarded as faint background dots
disc = above[above["status"] == "DISCARD"]
ax.scatter(
    disc.index,
    disc["valid_ndcg50"],
    c="#cccccc",
    s=12,
    alpha=0.5,
    zorder=2,
    label="Discarded",
)

# Plot kept experiments as prominent green dots
kept_v = above[above["status"] == "KEEP"]
ax.scatter(
    kept_v.index,
    kept_v["valid_ndcg50"],
    c="#2ecc71",
    s=50,
    zorder=4,
    label="Kept",
    edgecolors="black",
    linewidths=0.5,
)

# Running maximum step line
kept_mask = valid["status"] == "KEEP"
kept_idx = valid.index[kept_mask]
kept_ndcg50 = valid.loc[kept_mask, "valid_ndcg50"]
running_max = kept_ndcg50.cummax()
ax.step(
    kept_idx,
    running_max,
    where="post",
    color="#27ae60",
    linewidth=2,
    alpha=0.7,
    zorder=3,
    label="Running best",
)

# Label each kept experiment with its description
for idx, ndcg50 in zip(kept_idx, kept_ndcg50):
    desc = str(valid.loc[idx, "description"]).strip()
    if len(desc) > 45:
        desc = desc[:42] + "..."

    ax.annotate(
        desc,
        (idx, ndcg50),
        textcoords="offset points",
        xytext=(6, 6),
        fontsize=8.0,
        color="#1a7a3a",
        alpha=0.9,
        rotation=30,
        ha="left",
        va="bottom",
    )

n_total = len(df)
n_kept = len(df[df["status"] == "KEEP"])
ax.set_xlabel("Experiment #", fontsize=12)
ax.set_ylabel("Validation NDCG@50 (higher is better)", fontsize=12)
ax.set_title(
    f"Autoresearch Progress: {n_total} Experiments, {n_kept} Kept Improvements",
    fontsize=14,
)
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.2)

# Y-axis: from just below baseline to just above best
best_ndcg50 = kept["valid_ndcg50"].max()
margin = max((best_ndcg50 - baseline_ndcg50) * 0.15, 0.0005)
ax.set_ylim(baseline_ndcg50 - margin, best_ndcg50 + margin)

plt.tight_layout()
plt.savefig("progress.png", dpi=150, bbox_inches="tight")
print("Saved to progress.png")

# Summary Statistics
kept = df[df["status"] == "KEEP"].copy()
baseline_ndcg50 = df.iloc[0]["valid_ndcg50"]
best_ndcg50 = kept["valid_ndcg50"].max()
best_row = kept.loc[kept["valid_ndcg50"].idxmax()]

print(f"Baseline valid_ndcg50:  {baseline_ndcg50:.6f}")
print(f"Best valid_ndcg50:      {best_ndcg50:.6f}")
print(
    "Total improvement:      "
    f"{best_ndcg50 - baseline_ndcg50:.6f} "
    f"({(best_ndcg50 - baseline_ndcg50) / baseline_ndcg50 * 100:.2f}%)"
)
print(f"Best experiment:        {best_row['description']}")
print()

# How many experiments to find each improvement
print("Cumulative effort per improvement:")
kept_sorted = kept.reset_index()
for _, row in kept_sorted.iterrows():
    desc = str(row["description"]).strip()
    print(f"  Experiment #{row['index']:3d}: ndcg50={row['valid_ndcg50']:.6f}  {desc}")

# Top Hits (Kept Experiments by Improvement)
# Each kept experiment's delta is measured vs the previous kept experiment's ndcg50
kept = df[df["status"] == "KEEP"].copy()
kept["prev_ndcg50"] = kept["valid_ndcg50"].shift(1)
kept["delta"] = kept["valid_ndcg50"] - kept["prev_ndcg50"]

# Drop baseline (no delta)
hits = kept.iloc[1:].copy()

# Sort by delta improvement (biggest first)
hits = hits.sort_values("delta", ascending=False)

print(f"{'Rank':>4}  {'Delta':>8}  {'NDCG50':>10}  Description")
print("-" * 80)
for rank, (_, row) in enumerate(hits.iterrows(), 1):
    print(f"{rank:4d}  {row['delta']:+.6f}  {row['valid_ndcg50']:.6f}  {row['description']}")

print(f"\n{'':>4}  {hits['delta'].sum():+.6f}  {'':>10}  TOTAL improvement over baseline")
