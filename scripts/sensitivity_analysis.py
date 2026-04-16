"""
scripts/sensitivity_analysis.py

Tests how calibration, interval width, and point error change
across the three key parameters of the model:

  τ²  - prior variance (how strongly the prior pulls toward 0)
  σ   - observation noise (how noisy each LLM score is)
  n   - sample size (articles per estimate)

This is a prior sensitivity + robustness analysis. It answers:
  "Does the model's behaviour depend heavily on our assumptions?"

For a well-behaved model we expect:
  - Higher τ² → slightly wider intervals, less prior pull
  - Higher σ  → wider intervals (more observation noise)
  - Higher n  → narrower intervals, lower error (more data)

Deviations from this pattern indicate instability worth reporting.

Run from the project root:
    python scripts/sensitivity_analysis.py
"""

import os
import sys
import logging
import itertools
import numpy as np
import matplotlib.pyplot as plt

# Patch prior constants at runtime - we need to vary τ² across runs.
# We do this by importing the module and overriding its constants directly,
# which is cleaner than adding a parameter to estimate_market() for a
# sensitivity analysis that shouldn't affect production code.
import core.bayesian_model as bm

logging.disable(logging.WARNING)

# ----- parameter grid ------------------------------------------------

PRIOR_VARS  = [0.25, 1.0, 4.0]
OBS_NOISES  = [0.2,  0.4, 0.6]
SAMPLE_SIZES = [5,   10,  20]

N_RUNS = 1000
MU_LO  = -0.6
MU_HI  =  0.6
SEED   = 77

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs",
                           "sensitivity_analysis.png")

# ----- simulation ----------------------------------------------------

def run_cell(prior_var: float, obs_noise: float, n: int, rng) -> dict:
    """
    Run N_RUNS calibration trials with given hyperparameters.
    Temporarily patches the module-level prior variance constant.
    """
    original_prior_var = bm.PRIOR_VAR
    bm.PRIOR_VAR = prior_var

    hits   = []
    widths = []
    errors = []

    for _ in range(N_RUNS):
        true_mu = rng.uniform(MU_LO, MU_HI)
        raw     = rng.normal(loc=true_mu, scale=obs_noise, size=n)
        scores  = np.clip(raw, -1.0, 1.0).tolist()

        result = bm.estimate_market(scores)

        hits.append(result.lower_bound <= true_mu <= result.upper_bound)
        widths.append(result.upper_bound - result.lower_bound)
        errors.append(abs(result.mean - true_mu))

    bm.PRIOR_VAR = original_prior_var   # always restore

    return {
        "coverage": float(np.mean(hits)),
        "width":    float(np.mean(widths)),
        "error":    float(np.mean(errors)),
    }


# Pre-seed each cell with a fixed offset so results are reproducible
# regardless of iteration order.
base_rng = np.random.default_rng(SEED)

results = {}
total   = len(PRIOR_VARS) * len(OBS_NOISES) * len(SAMPLE_SIZES)
done    = 0

print(f"Running {total} cells × {N_RUNS} trials each ...", flush=True)

for pv, sigma, n in itertools.product(PRIOR_VARS, OBS_NOISES, SAMPLE_SIZES):
    cell_rng = np.random.default_rng(SEED + done)
    results[(pv, sigma, n)] = run_cell(pv, sigma, n, cell_rng)
    done += 1

print(f"Done.\n")

# ----- print table ---------------------------------------------------

METRICS = [("coverage", "Coverage", ".3f"),
           ("width",    "Width",    ".3f"),
           ("error",    "Error",    ".3f")]

for metric_key, metric_label, fmt in METRICS:
    print(f"── {metric_label} (target: coverage≈0.95, width/error: lower=better) ──\n")
    # Header: n values
    header = ('  ' + 'τ²\σ'.ljust(10)) + "".join(f"  σ={s}" for s in OBS_NOISES)
    sub    = f"  {'':10}" + "".join(
        "  " + "  ".join(f"n={n:2d}" for n in SAMPLE_SIZES)
        for _ in OBS_NOISES
    )
    col_heads = ('  ' + 'τ²\σ'.ljust(10))
    for sigma in OBS_NOISES:
        for n in SAMPLE_SIZES:
            col_heads += f"  σ={sigma}/n={n:2d}"
    print(col_heads)
    print("  " + "-" * (len(col_heads) - 2))

    for pv in PRIOR_VARS:
        row = f"  τ²={pv:<7}"
        for sigma in OBS_NOISES:
            for n in SAMPLE_SIZES:
                val = results[(pv, sigma, n)][metric_key]
                row += f"  {val:{fmt}}"
        print(row)
    print()

# ----- summary observations ------------------------------------------

print("── Key observations ──\n")

# 1. Effect of n (averaging over τ² and σ)
for n in SAMPLE_SIZES:
    avg_cov = np.mean([results[(pv, s, n)]["coverage"]
                       for pv, s in itertools.product(PRIOR_VARS, OBS_NOISES)])
    avg_w   = np.mean([results[(pv, s, n)]["width"]
                       for pv, s in itertools.product(PRIOR_VARS, OBS_NOISES)])
    print(f"  n={n:2d}:  avg coverage={avg_cov:.3f}  avg width={avg_w:.3f}")

print()

# 2. Effect of τ² (averaging over σ and n)
for pv in PRIOR_VARS:
    avg_cov = np.mean([results[(pv, s, n)]["coverage"]
                       for s, n in itertools.product(OBS_NOISES, SAMPLE_SIZES)])
    avg_w   = np.mean([results[(pv, s, n)]["width"]
                       for s, n in itertools.product(OBS_NOISES, SAMPLE_SIZES)])
    print(f"  τ²={pv}:  avg coverage={avg_cov:.3f}  avg width={avg_w:.3f}")

print()

# 3. Worst cell
worst_cov_key = min(results, key=lambda k: results[k]["coverage"])
best_cov_key  = max(results, key=lambda k: results[k]["coverage"])
wc = results[worst_cov_key]
bc = results[best_cov_key]
print(f"  Worst coverage: τ²={worst_cov_key[0]}, σ={worst_cov_key[1]}, "
      f"n={worst_cov_key[2]}  → {wc['coverage']:.3f}")
print(f"  Best  coverage: τ²={best_cov_key[0]},  σ={best_cov_key[1]}, "
      f"n={best_cov_key[2]}  → {bc['coverage']:.3f}")

# ----- heatmap -------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle(
    "Sensitivity Analysis - Coverage, Interval Width, Point Error\n"
    f"N={N_RUNS} trials per cell  |  μ ∈ [{MU_LO}, {MU_HI}]",
    fontsize=12, fontweight="bold",
)

# For each metric, build a 2D array: rows = (τ², σ) combos, cols = n
# We'll use a flat layout: rows = σ values, cols = n values,
# one subplot per τ².

cmaps = {"coverage": "RdYlGn", "width": "RdYlGn_r", "error": "RdYlGn_r"}
titles = {"coverage": "Coverage (→ 0.95)",
          "width":    "Interval Width (lower=better)",
          "error":    "Point Error (lower=better)"}

for ax, (metric_key, metric_label, _) in zip(axes, METRICS):
    # One heatmap: rows = obs noise, cols = sample size
    # Averaged over τ² to keep it readable
    grid = np.zeros((len(OBS_NOISES), len(SAMPLE_SIZES)))
    for i, sigma in enumerate(OBS_NOISES):
        for j, n in enumerate(SAMPLE_SIZES):
            grid[i, j] = np.mean([results[(pv, sigma, n)][metric_key]
                                   for pv in PRIOR_VARS])

    im = ax.imshow(grid, cmap=cmaps[metric_key], aspect="auto")
    ax.set_xticks(range(len(SAMPLE_SIZES)))
    ax.set_xticklabels([f"n={n}" for n in SAMPLE_SIZES], fontsize=9)
    ax.set_yticks(range(len(OBS_NOISES)))
    ax.set_yticklabels([f"σ={s}" for s in OBS_NOISES], fontsize=9)
    ax.set_title(f"{titles[metric_key]}\n(averaged over τ²)", fontsize=9)

    for i in range(len(OBS_NOISES)):
        for j in range(len(SAMPLE_SIZES)):
            ax.text(j, i, f"{grid[i, j]:.3f}",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if grid[i, j] < grid.mean() else "#222222")

    fig.colorbar(im, ax=ax, shrink=0.85)

plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved to {OUTPUT_PATH}")