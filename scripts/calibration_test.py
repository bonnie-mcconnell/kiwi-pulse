"""
scripts/calibration_test.py

Evaluates whether the Bayesian credible intervals are well-calibrated,
and isolates boundary exposure as the causal mechanism of miscalibration.

ALL THREE EXPERIMENTS USE CLIPPED DATA.
The difference is how often clipping is asymmetric - i.e. how often
the true μ is close enough to a boundary that noise gets truncated
in one direction more than the other.

Asymmetric truncation causes the model to underestimate σ², which
produces intervals that are too narrow, which reduces coverage below
the nominal 95%.

THREE REGIMES
-------------
A) Interior μ   - μ ~ Uniform(-0.6, 0.6), noise_std=0.4
   True sentiment kept away from boundaries.
   Clipping occurs but is roughly symmetric.
   → Coverage should be close to 0.95.

B) Full domain  - μ ~ Uniform(-1.0, 1.0), noise_std=0.4
   True sentiment can land near boundaries.
   Clipping is frequently asymmetric.
   → Coverage drops meaningfully below 0.95.

C) Strong clipping regime - μ ~ Uniform(-0.95, 0.95), noise_std=0.6
   Higher noise means more observations hit the boundary.
   Even μ values away from the edges produce asymmetric truncation.
   → Coverage degrades further.

Comparing A → B → C shows a monotonic degradation primarily driven
by the degree of boundary interaction under current model assumptions.

Two additional sources of miscalibration are present in all regimes
and are not isolated by this experiment:
  - Gaussian likelihood mismatch: the truncated Normal data-generating
    process is not identical to the untruncated Normal the model assumes.
  - Empirical Bayes σ² instability: with small n, sample variance is a
    noisy estimator of true observation noise, which adds further error.

Run from the project root:
    python scripts/calibration_test.py
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt

from core.bayesian_model import estimate_market

logging.disable(logging.WARNING)

# ----- parameters ----------------------------------------------------

N_RUNS = 2000
N_OBS  = 10
SEED   = 99

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "calibration.png")

# ----- simulation ----------------------------------------------------

def run_experiment(mu_lo: float, mu_hi: float, noise_std: float, seed: int) -> dict:
    """
    Run N_RUNS calibration trials and return coverage metrics.

    Both bounds and clipping are always applied - the only variable
    across experiments is how much boundary interaction occurs.

    Parameters
    ----------
    mu_lo, mu_hi : float
        Range from which true sentiment μ is drawn uniformly.
    noise_std : float
        Standard deviation of observation noise before clipping.
    seed : int
        Offset added to global SEED for reproducibility across experiments.
    """
    rng  = np.random.default_rng(SEED + seed)
    hits = []

    for _ in range(N_RUNS):
        true_mu = rng.uniform(mu_lo, mu_hi)
        raw     = rng.normal(loc=true_mu, scale=noise_std, size=N_OBS)
        scores  = np.clip(raw, -1.0, 1.0).tolist()  # always clipped

        result = estimate_market(scores)
        hits.append(result.lower_bound <= true_mu <= result.upper_bound)

    coverage = sum(hits) / N_RUNS
    return {
        "coverage": coverage,
        "hits":     sum(hits),
        "misses":   N_RUNS - sum(hits),
    }


EXPERIMENTS = [
    {
        "label":      "A) Interior μ\n(low boundary interaction)",
        "mu_lo":      -0.6,
        "mu_hi":       0.6,
        "noise_std":   0.4,
        "seed":        0,
        "note":        "μ ∈ [-0.6, 0.6]\nnoise σ = 0.4\nClipping mostly symmetric",
    },
    {
        "label":      "B) Full domain μ\n(high boundary interaction)",
        "mu_lo":      -1.0,
        "mu_hi":       1.0,
        "noise_std":   0.4,
        "seed":        1,
        "note":        "μ ∈ [-1.0, 1.0]\nnoise σ = 0.4\nClipping often asymmetric",
    },
    {
        "label":      "C) Strong clipping regime\n(high noise + near-boundary μ)",
        "mu_lo":      -0.95,
        "mu_hi":       0.95,
        "noise_std":   0.6,
        "seed":        2,
        "note":        "μ ∈ [-0.95, 0.95]\nnoise σ = 0.6\nHeavy asymmetric truncation",
    },
]

results = [run_experiment(**{k: v for k, v in e.items() if k in
           ("mu_lo", "mu_hi", "noise_std", "seed")}) for e in EXPERIMENTS]

# ----- print ---------------------------------------------------------

print("NOTE: All experiments use clipped data.")
print("      Difference is boundary exposure, not clipping presence.\n")
print(f"{'Experiment':<45} {'Coverage':>10}  {'Hits':>6}  {'Misses':>6}")
print("-" * 72)

for exp, res in zip(EXPERIMENTS, results):
    short_label = exp["label"].replace("\n", " ")
    gap = res["coverage"] - 0.95
    gap_str = f"({gap:+.3f} vs 0.95)"
    print(f"{short_label:<45} {res['coverage']:>10.3f}  {res['hits']:>6}  {res['misses']:>6}  {gap_str}")

coverages = [r["coverage"] for r in results]
if coverages[0] > coverages[1] > coverages[2]:
    print("\nConclusion: monotonic degradation confirmed.")
    print("Increased boundary interaction → underestimated σ² → intervals too narrow → lower coverage.")
    print("Note: Gaussian likelihood mismatch and empirical Bayes σ² instability also")
    print("      contribute to miscalibration but are not isolated by this experiment.")
else:
    print("\nConclusion: results non-monotonic - check experiment parameters.")

# ----- plot ----------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
fig.suptitle(
    "Calibration Across Boundary Exposure Regimes\n"
    "All experiments use clipped data - boundary interaction is the primary variable",
    fontsize=12, fontweight="bold",
)

COLORS = ["#2ecc71", "#e67e22", "#e74c3c"]

for ax, exp, res, color in zip(axes, EXPERIMENTS, results, COLORS):
    hits   = res["hits"]
    misses = res["misses"]
    cov    = res["coverage"]

    bars = ax.bar(
        ["Hit", "Miss"],
        [hits, misses],
        color=[color, "#dddddd"],
        width=0.45,
        edgecolor="white",
        linewidth=1.5,
    )

    for bar, count in zip(bars, [hits, misses]):
        pct = count / N_RUNS * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 12,
            f"{pct:.1f}%",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#333333",
        )

    ax.axhline(N_RUNS * 0.95, color="#555555", linewidth=1.2,
               linestyle="--", label="95% target")

    gap     = cov - 0.95
    gap_str = f"gap: {gap:+.3f}"
    ax.set_title(f"{exp['label']}\nCoverage: {cov:.3f}  ({gap_str})",
                 fontsize=9.5, pad=10)

    ax.text"""
scripts/calibration_test.py

Evaluates whether the Bayesian credible intervals are well-calibrated,
and isolates boundary exposure as the causal mechanism of miscalibration.

ALL THREE EXPERIMENTS USE CLIPPED DATA.
The difference is how often clipping is asymmetric - i.e. how often
the true μ is close enough to a boundary that noise gets truncated
in one direction more than the other.

Asymmetric truncation causes the model to underestimate σ², which
produces intervals that are too narrow, which reduces coverage below
the nominal 95%.

THREE REGIMES
-------------
A) Interior μ   - μ ~ Uniform(-0.6, 0.6), noise_std=0.4
   True sentiment kept away from boundaries.
   Clipping occurs but is roughly symmetric.
   → Coverage should be close to 0.95.

B) Full domain  - μ ~ Uniform(-1.0, 1.0), noise_std=0.4
   True sentiment can land near boundaries.
   Clipping is frequently asymmetric.
   → Coverage drops meaningfully below 0.95.

C) Strong clipping regime - μ ~ Uniform(-0.95, 0.95), noise_std=0.6
   Higher noise means more observations hit the boundary.
   Even μ values away from the edges produce asymmetric truncation.
   → Coverage degrades further.

Comparing A → B → C shows a monotonic degradation primarily driven
by the degree of boundary interaction under current model assumptions.

Two additional sources of miscalibration are present in all regimes
and are not isolated by this experiment:
  - Gaussian likelihood mismatch: the truncated Normal data-generating
    process is not identical to the untruncated Normal the model assumes.
  - Empirical Bayes σ² instability: with small n, sample variance is a
    noisy estimator of true observation noise, which adds further error.

Run from the project root:
    python scripts/calibration_test.py
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt

from core.bayesian_model import estimate_market

logging.disable(logging.WARNING)

# ----- parameters ----------------------------------------------------

N_RUNS = 2000
N_OBS  = 10
SEED   = 99

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "calibration.png")

# ----- simulation ----------------------------------------------------

def run_experiment(mu_lo: float, mu_hi: float, noise_std: float, seed: int) -> dict:
    """
    Run N_RUNS calibration trials and return coverage metrics.

    Both bounds and clipping are always applied - the only variable
    across experiments is how much boundary interaction occurs.

    Parameters
    ----------
    mu_lo, mu_hi : float
        Range from which true sentiment μ is drawn uniformly.
    noise_std : float
        Standard deviation of observation noise before clipping.
    seed : int
        Offset added to global SEED for reproducibility across experiments.
    """
    rng  = np.random.default_rng(SEED + seed)
    hits = []

    for _ in range(N_RUNS):
        true_mu = rng.uniform(mu_lo, mu_hi)
        raw     = rng.normal(loc=true_mu, scale=noise_std, size=N_OBS)
        scores  = np.clip(raw, -1.0, 1.0).tolist()  # always clipped

        result = estimate_market(scores)
        hits.append(result.lower_bound <= true_mu <= result.upper_bound)

    coverage = sum(hits) / N_RUNS
    return {
        "coverage": coverage,
        "hits":     sum(hits),
        "misses":   N_RUNS - sum(hits),
    }


EXPERIMENTS = [
    {
        "label":      "A) Interior μ\n(low boundary interaction)",
        "mu_lo":      -0.6,
        "mu_hi":       0.6,
        "noise_std":   0.4,
        "seed":        0,
        "note":        "μ ∈ [-0.6, 0.6]\nnoise σ = 0.4\nClipping mostly symmetric",
    },
    {
        "label":      "B) Full domain μ\n(high boundary interaction)",
        "mu_lo":      -1.0,
        "mu_hi":       1.0,
        "noise_std":   0.4,
        "seed":        1,
        "note":        "μ ∈ [-1.0, 1.0]\nnoise σ = 0.4\nClipping often asymmetric",
    },
    {
        "label":      "C) Strong clipping regime\n(high noise + near-boundary μ)",
        "mu_lo":      -0.95,
        "mu_hi":       0.95,
        "noise_std":   0.6,
        "seed":        2,
        "note":        "μ ∈ [-0.95, 0.95]\nnoise σ = 0.6\nHeavy asymmetric truncation",
    },
]

results = [run_experiment(**{k: v for k, v in e.items() if k in
           ("mu_lo", "mu_hi", "noise_std", "seed")}) for e in EXPERIMENTS]

# ----- print ---------------------------------------------------------

print("NOTE: All experiments use clipped data.")
print("      Difference is boundary exposure, not clipping presence.\n")
print(f"{'Experiment':<45} {'Coverage':>10}  {'Hits':>6}  {'Misses':>6}")
print("-" * 72)

for exp, res in zip(EXPERIMENTS, results):
    short_label = exp["label"].replace("\n", " ")
    gap = res["coverage"] - 0.95
    gap_str = f"({gap:+.3f} vs 0.95)"
    print(f"{short_label:<45} {res['coverage']:>10.3f}  {res['hits']:>6}  {res['misses']:>6}  {gap_str}")

coverages = [r["coverage"] for r in results]
if coverages[0] > coverages[1] > coverages[2]:
    print("\nConclusion: monotonic degradation confirmed.")
    print("Increased boundary interaction → underestimated σ² → intervals too narrow → lower coverage.")
    print("Note: Gaussian likelihood mismatch and empirical Bayes σ² instability also")
    print("      contribute to miscalibration but are not isolated by this experiment.")
else:
    print("\nConclusion: results non-monotonic - check experiment parameters.")

# ----- plot ----------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
fig.suptitle(
    "Calibration Across Boundary Exposure Regimes\n"
    "All experiments use clipped data - boundary interaction is the primary variable",
    fontsize=12, fontweight="bold",
)

COLORS = ["#2ecc71", "#e67e22", "#e74c3c"]

for ax, exp, res, color in zip(axes, EXPERIMENTS, results, COLORS):
    hits   = res["hits"]
    misses = res["misses"]
    cov    = res["coverage"]

    bars = ax.bar(
        ["Hit", "Miss"],
        [hits, misses],
        color=[color, "#dddddd"],
        width=0.45,
        edgecolor="white",
        linewidth=1.5,
    )

    for bar, count in zip(bars, [hits, misses]):
        pct = count / N_RUNS * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 12,
            f"{pct:.1f}%",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#333333",
        )

    ax.axhline(N_RUNS * 0.95, color="#555555", linewidth=1.2,
               linestyle="--", label="95% target")

    gap     = cov - 0.95
    gap_str = f"gap: {gap:+.3f}"
    ax.set_title(f"{exp['label']}\nCoverage: {cov:.3f}  ({gap_str})",
                 fontsize=9.5, pad=10)

    ax.text(
        0.97, 0.97, exp["note"],
        transform=ax.transAxes,
        fontsize=8, va="top", ha="right", color="#444444",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f9f9f9",
                  edgecolor="#cccccc", linewidth=0.8),
    )

    ax.set_ylim(0, N_RUNS * 1.15)
    if ax == axes[0]:
        ax.set_ylabel("Count", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved to {OUTPUT_PATH}")(
        0.97, 0.97, exp["note"],
        transform=ax.transAxes,
        fontsize=8, va="top", ha="right", color="#444444",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f9f9f9",
                  edgecolor="#cccccc", linewidth=0.8),
    )

    ax.set_ylim(0, N_RUNS * 1.15)
    if ax == axes[0]:
        ax.set_ylabel("Count", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved to {OUTPUT_PATH}")