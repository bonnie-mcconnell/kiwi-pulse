"""
scripts/decision_simulation.py

Tests whether using the credible interval to gate decisions
produces better accuracy than always committing to a prediction.

Decision rule (uncertainty-aware):
  - lower_bound > 0  → predict POSITIVE
  - upper_bound < 0  → predict NEGATIVE
  - otherwise        → ABSTAIN  (too uncertain to call)

Baseline rule (always commit):
  - posterior mean > 0 → predict POSITIVE
  - posterior mean < 0 → predict NEGATIVE

The key question: does abstaining on uncertain cases
improve accuracy on the cases where we do commit?

Run from the project root:
    python scripts/decision_simulation.py
"""

import os

import logging
import numpy as np
import matplotlib.pyplot as plt

from core.bayesian_model import estimate_market

logging.disable(logging.WARNING)  # suppress soft-clamp warnings in bulk runs

# ----- parameters ----------------------------------------------------

N_RUNS    = 2000
N_OBS     = 10
NOISE_STD = 0.4
SEED      = 7

# ----- simulation ----------------------------------------------------

rng = np.random.default_rng(SEED)

# Track per-run outcomes
decisions   = []   # "positive", "negative", "abstain"
baselines   = []   # "positive", "negative" (always commits)
true_labels = []   # "positive" or "negative" (ground truth)

for _ in range(N_RUNS):
    true_mu = rng.uniform(-1.0, 1.0)
    raw     = rng.normal(loc=true_mu, scale=NOISE_STD, size=N_OBS)
    scores  = np.clip(raw, -1.0, 1.0).tolist()

    result  = estimate_market(scores)

    # Ground truth label
    true_labels.append("positive" if true_mu > 0 else "negative")

    # Uncertainty-aware decision
    if result.lower_bound > 0:
        decisions.append("positive")
    elif result.upper_bound < 0:
        decisions.append("negative")
    else:
        decisions.append("abstain")

    # Baseline: always commit to the mean
    baselines.append("positive" if result.mean > 0 else "negative")

# ----- metrics -------------------------------------------------------

def compute_metrics(preds: list[str], labels: list[str]) -> dict:
    n = len(preds)
    committed    = [(p, l) for p, l in zip(preds, labels) if p != "abstain"]
    abstained    = sum(1 for p in preds if p == "abstain")
    correct      = sum(1 for p, l in committed if p == l)
    false_pos    = sum(1 for p, l in committed if p == "positive" and l == "negative")
    false_neg    = sum(1 for p, l in committed if p == "negative" and l == "positive")

    n_committed  = len(committed)
    accuracy     = correct / n_committed if n_committed else float("nan")
    abstain_rate = abstained / n

    return {
        "accuracy":     accuracy,
        "abstain_rate": abstain_rate,
        "n_committed":  n_committed,
        "false_pos":    false_pos,
        "false_neg":    false_neg,
    }

aware    = compute_metrics(decisions, true_labels)
baseline = compute_metrics(baselines, true_labels)

# ----- print ---------------------------------------------------------

print(f"{'Metric':<28} {'Uncertainty-aware':>18}  {'Baseline (always commit)':>24}")
print("-" * 74)
print(f"{'Accuracy (when committed)':<28} {aware['accuracy']:>17.3f}  {baseline['accuracy']:>24.3f}")
print(f"{'Abstain rate':<28} {aware['abstain_rate']:>17.3f}  {baseline['abstain_rate']:>24.3f}")
print(f"{'Decisions made':<28} {aware['n_committed']:>17}  {baseline['n_committed']:>24}")
print(f"{'False positives':<28} {aware['false_pos']:>17}  {baseline['false_pos']:>24}")
print(f"{'False negatives':<28} {aware['false_neg']:>17}  {baseline['false_neg']:>24}")
print()
acc_lift = (aware["accuracy"] - baseline["accuracy"]) * 100
print(f"Accuracy lift from abstaining: +{acc_lift:.1f} percentage points")
print(f"Cost: abstained on {aware['abstain_rate']*100:.1f}% of cases")

# ----- plot ----------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.suptitle(
    "Does Uncertainty-Aware Decision Making Improve Accuracy?",
    fontsize=13, fontweight="bold",
)

# Left: accuracy comparison
ax = axes[0]
labels_bar = ["Uncertainty-aware\n(abstains when unsure)", "Baseline\n(always commits)"]
accs       = [aware["accuracy"], baseline["accuracy"]]
colors     = ["#2ecc71", "#95a5a6"]

bars = ax.bar(labels_bar, accs, color=colors, width=0.4,
              edgecolor="white", linewidth=1.5)

for bar, acc in zip(bars, accs):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{acc:.3f}",
        ha="center", va="bottom", fontsize=12, fontweight="bold",
    )

ax.set_ylim(0, 1.0)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("Accuracy When Decision Made", fontsize=11)
ax.axhline(0.5, color="#cccccc", linewidth=1, linestyle="--", label="Random chance")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# Right: decision breakdown (stacked bar showing what happened)
ax = axes[1]

# Count outcomes for aware strategy
a_correct  = sum(1 for p, l in zip(decisions, true_labels) if p != "abstain" and p == l)
a_wrong    = sum(1 for p, l in zip(decisions, true_labels) if p != "abstain" and p != l)
a_abstain  = sum(1 for p in decisions if p == "abstain")

# Count outcomes for baseline
b_correct  = sum(1 for p, l in zip(baselines, true_labels) if p == l)
b_wrong    = sum(1 for p, l in zip(baselines, true_labels) if p != l)

x       = np.arange(2)
correct = [a_correct, b_correct]
wrong   = [a_wrong,   b_wrong]
abstain = [a_abstain, 0]

ax.bar(x, correct, label="Correct",  color="#2ecc71", edgecolor="white")
ax.bar(x, wrong,   label="Wrong",    color="#e74c3c", bottom=correct, edgecolor="white")
ax.bar(x, abstain, label="Abstain",  color="#bdc3c7",
       bottom=[c + w for c, w in zip(correct, wrong)], edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(["Uncertainty-aware", "Baseline"], fontsize=10)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Decision Breakdown (all runs)", fontsize=11)
ax.legend(fontsize=9, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "decision_simulation.png")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved to {OUTPUT_PATH}")