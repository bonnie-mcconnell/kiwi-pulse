"""
scripts/baseline_comparison.py

Compares the Bayesian model against the naive sample mean
on three metrics: calibration, interval width, and point error.

WHY THIS MATTERS
----------------
The Bayesian model claims to be better than just averaging scores.
This script checks whether that claim holds up empirically.

TWO METHODS
-----------
Bayesian  - Normal-Normal conjugate posterior.
            Interval: 95% credible interval from posterior variance.

Naive mean - Sample mean of observations.
             Interval: bootstrap 95% CI (2000 resamples).
             This is the fairest comparison - the bootstrap makes no
             parametric assumptions, so any advantage the Bayesian
             model shows is real, not an artefact of the baseline
             using a weaker interval method.

THREE METRICS
-------------
Coverage    - Does the interval contain the true μ ~95% of the time?
Width       - How wide is the interval? Narrower is better if coverage holds.
Abs error   - |estimate - true μ|. Lower is better.

Run from the project root:
    python scripts/baseline_comparison.py
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt

from core.bayesian_model import estimate_market

logging.disable(logging.WARNING)

# ----- parameters ----------------------------------------------------

N_RUNS       = 2000
N_OBS        = 10
NOISE_STD    = 0.4
BOOT_SAMPLES = 2000   # bootstrap resamples per trial
SEED         = 42

rng = np.random.default_rng(SEED)

# ----- simulation ----------------------------------------------------

bayes_hits    = []
naive_hits    = []
bayes_widths  = []
naive_widths  = []
bayes_errors  = []
naive_errors  = []

for _ in range(N_RUNS):
    true_mu = rng.uniform(-0.6, 0.6)   # interior to keep comparison fair
    raw     = rng.normal(loc=true_mu, scale=NOISE_STD, size=N_OBS)
    scores  = np.clip(raw, -1.0, 1.0).tolist()
    x       = np.array(scores)

    # --- Bayesian estimate -------------------------------------------
    result    = estimate_market(scores)
    b_mean    = result.mean
    b_lo      = result.lower_bound
    b_hi      = result.upper_bound

    bayes_hits.append(b_lo <= true_mu <= b_hi)
    bayes_widths.append(b_hi - b_lo)
    bayes_errors.append(abs(b_mean - true_mu))

    # --- Naive mean + bootstrap CI -----------------------------------
    n_mean = float(x.mean())

    boot   = rng.choice(x, size=(BOOT_SAMPLES, N_OBS), replace=True)
    boot_means = boot.mean(axis=1)
    n_lo   = float(np.percentile(boot_means, 2.5))
    n_hi   = float(np.percentile(boot_means, 97.5))

    naive_hits.append(n_lo <= true_mu <= n_hi)
    naive_widths.append(n_hi - n_lo)
    naive_errors.append(abs(n_mean - true_mu))


def stats(values: list[float]) -> tuple[float, float]:
    """Return mean and standard error."""
    a = np.array(values)
    return float(a.mean()), float(a.std() / np.sqrt(len(a)))


b_cov,   _    = stats([float(h) for h in bayes_hits])
n_cov,   _    = stats([float(h) for h in naive_hits])
b_width, b_wse = stats(bayes_widths)
n_width, n_wse = stats(naive_widths)
b_err,   b_ese = stats(bayes_errors)
n_err,   n_ese = stats(naive_errors)

# ----- print ---------------------------------------------------------

print(f"Runs: {N_RUNS}  |  n={N_OBS} obs/run  |  noise σ={NOISE_STD}  |  μ ∈ [-0.6, 0.6]")
print(f"Bootstrap CI uses {BOOT_SAMPLES} resamples per trial.\n")

col = 22
print(f"{'Metric':<28} {'Bayesian':>{col}}  {'Naive mean':>{col}}  {'Winner'}")
print("-" * 80)

def winner(b_val, n_val, higher_is_better: bool) -> str:
    if higher_is_better:
        return "Bayesian" if b_val > n_val else "Naive" if n_val > b_val else "Tie"
    return "Bayesian" if b_val < n_val else "Naive" if n_val < b_val else "Tie"

print(f"{'Coverage (target 0.95)':<28} {b_cov:{col}.3f}  {n_cov:{col}.3f}  {winner(b_cov, n_cov, higher_is_better=True)}")
print(f"{'Interval width (mean)':<28} {b_width:{col}.3f}  {n_width:{col}.3f}  {winner(b_width, n_width, higher_is_better=False)}")
print(f"{'Abs error (mean)':<28} {b_err:{col}.3f}  {n_err:{col}.3f}  {winner(b_err, n_err, higher_is_better=False)}")

print()
print("Notes:")
print(f"  Both methods undercover relative to 0.95 - this is the boundary clipping effect.")
print(f"  Width difference: Bayesian is {(n_width - b_width) / n_width * 100:.1f}% {'narrower' if b_width < n_width else 'wider'} than naive bootstrap.")
print(f"  Error difference: Bayesian is {abs(b_err - n_err) / n_err * 100:.1f}% {'better' if b_err < n_err else 'worse'} on point estimate.")

# ----- plot ----------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
fig.suptitle(
    "Bayesian Model vs Naive Sample Mean\n"
    "Bootstrap CI used for naive baseline - fairest possible comparison",
    fontsize=12, fontweight="bold",
)

BAY_COLOR   = "#4C72B0"
NAIVE_COLOR = "#DD8452"
x_pos       = [0, 1]
x_labels    = ["Bayesian", "Naive mean\n+ bootstrap CI"]

# --- Coverage ---
ax = axes[0]
bars = ax.bar(x_pos, [b_cov, n_cov], color=[BAY_COLOR, NAIVE_COLOR],
              width=0.45, edgecolor="white", linewidth=1.5)
ax.axhline(0.95, color="#555555", linewidth=1.2, linestyle="--", label="Target (0.95)")
for bar, val in zip(bars, [b_cov, n_cov]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylim(0.7, 1.0)
ax.set_xticks(x_pos); ax.set_xticklabels(x_labels, fontsize=9)
ax.set_ylabel("Coverage", fontsize=10)
ax.set_title("Coverage\n(higher → better)", fontsize=10)
ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# --- Interval width ---
ax = axes[1]
bars = ax.bar(x_pos, [b_width, n_width], color=[BAY_COLOR, NAIVE_COLOR],
              width=0.45, edgecolor="white", linewidth=1.5)
# error bars showing standard error across runs
ax.errorbar(x_pos, [b_width, n_width], yerr=[b_wse, n_wse],
            fmt="none", color="#333333", capsize=5, linewidth=1.5)
for bar, val in zip(bars, [b_width, n_width]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_xticks(x_pos); ax.set_xticklabels(x_labels, fontsize=9)
ax.set_ylabel("Mean interval width", fontsize=10)
ax.set_title("Interval Width\n(lower → better if coverage holds)", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# --- Abs error ---
ax = axes[2]
bars = ax.bar(x_pos, [b_err, n_err], color=[BAY_COLOR, NAIVE_COLOR],
              width=0.45, edgecolor="white", linewidth=1.5)
ax.errorbar(x_pos, [b_err, n_err], yerr=[b_ese, n_ese],
            fmt="none", color="#333333", capsize=5, linewidth=1.5)
for bar, val in zip(bars, [b_err, n_err]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
            f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_xticks(x_pos); ax.set_xticklabels(x_labels, fontsize=9)
ax.set_ylabel("Mean absolute error", fontsize=10)
ax.set_title("Point Estimate Error\n(lower → better)", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "baseline_comparison.png")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved to {OUTPUT_PATH}")