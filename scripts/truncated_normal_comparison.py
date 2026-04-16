"""
scripts/truncated_normal_comparison.py

Compares Gaussian vs truncated Normal Bayesian inference on calibration,
interval width, and point error - across interior and boundary conditions.

THE HYPOTHESIS
--------------
The Gaussian model should undercover near boundaries (data compressed
by clamping → variance underestimated → intervals too narrow).
The truncated Normal corrects for the boundary directly via the
normalising constant Z(μ, σ), and should show improved coverage,
particularly when true μ is close to ±1.

TWO CONDITIONS
--------------
Interior: μ ~ Uniform(-0.6, 0.6) - matches our standard calibration test
Boundary: μ ~ Uniform(-1.0, 1.0) - includes boundary cases

WITHIN EACH: we compare Gaussian (floored) vs truncated Normal.

Run from the project root:
    python scripts/truncated_normal_comparison.py
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt

from core.bayesian_model import estimate_market
from core.truncated_normal_model import estimate_market_truncated

logging.disable(logging.WARNING)

# ----- parameters ----------------------------------------------------

N_RUNS    = 2000
N_OBS     = 10
NOISE_STD = 0.4
SEED      = 42

CONDITIONS = [
    ("Interior  μ ∈ [-0.6, 0.6]", -0.6,  0.6),
    ("Boundary  μ ∈ [-1.0, 1.0]", -1.0,  1.0),
]

MODELS = [
    ("Gaussian (floored)",  estimate_market),
    ("Truncated Normal",    estimate_market_truncated),
]

# ----- simulation ----------------------------------------------------

results = {}   # (condition, model) → {coverage, width, error}

for cond_label, mu_lo, mu_hi in CONDITIONS:
    for model_label, fn in MODELS:
        rng   = np.random.default_rng(SEED)
        hits, widths, errors = [], [], []

        for _ in range(N_RUNS):
            true_mu = rng.uniform(mu_lo, mu_hi)
            raw     = rng.normal(true_mu, NOISE_STD, N_OBS)
            scores  = np.clip(raw, -1.0, 1.0).tolist()

            r = fn(scores)

            hits.append(r.lower_bound <= true_mu <= r.upper_bound)
            widths.append(r.upper_bound - r.lower_bound)
            errors.append(abs(r.mean - true_mu))

        results[(cond_label, model_label)] = {
            "coverage": float(np.mean(hits)),
            "width":    float(np.mean(widths)),
            "error":    float(np.mean(errors)),
        }

# ----- print ---------------------------------------------------------

print(f"N={N_RUNS} runs  |  n={N_OBS} obs/run  |  noise σ={NOISE_STD}\n")

for cond_label, mu_lo, mu_hi in CONDITIONS:
    print(f"── {cond_label} ──\n")
    print(f"  {'Model':<26} {'Coverage':>10}  {'Width':>10}  {'Abs error':>10}")
    print("  " + "-" * 60)
    for model_label, _ in MODELS:
        r = results[(cond_label, model_label)]
        cov_flag = " ✓" if abs(r['coverage'] - 0.95) < 0.03 else " ⚠"
        print(f"  {model_label:<26} {r['coverage']:>10.3f}{cov_flag}"
              f"  {r['width']:>10.3f}  {r['error']:>10.3f}")

    # Coverage lift
    g_cov = results[(cond_label, "Gaussian (floored)")]["coverage"]
    t_cov = results[(cond_label, "Truncated Normal")]["coverage"]
    lift  = (t_cov - g_cov) * 100
    print(f"\n  Coverage lift from truncated Normal: {lift:+.1f}pp\n")

# ----- plot ----------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.suptitle(
    "Gaussian vs Truncated Normal - Calibration Comparison\n"
    f"N={N_RUNS} trials  |  n={N_OBS} obs/trial  |  noise σ={NOISE_STD}",
    fontsize=12, fontweight="bold",
)

metrics = [
    ("coverage", "Coverage",       "Coverage (target: 0.95)", True),
    ("width",    "Interval Width", "Mean interval width",     False),
    ("error",    "Point Error",    "Mean absolute error",     False),
]

COLORS   = {"Gaussian (floored)": "#4C72B0", "Truncated Normal": "#2ecc71"}
cond_labels = [c[0] for c in CONDITIONS]
x = np.arange(len(cond_labels))
bar_w = 0.3

for ax, (key, title, ylabel, add_target) in zip(axes, metrics):
    for i, (model_label, _) in enumerate(MODELS):
        vals   = [results[(c, model_label)][key] for c in cond_labels]
        offset = (i - 0.5) * bar_w
        bars   = ax.bar(x + offset, vals, bar_w,
                        label=model_label,
                        color=COLORS[model_label],
                        edgecolor="white", linewidth=1.2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.004,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")

    if add_target:
        ax.axhline(0.95, color="#333333", linewidth=1.2,
                   linestyle="--", label="Target (0.95)")
        ax.set_ylim(0.80, 1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(["Interior\nμ∈[-0.6,0.6]",
                         "Boundary\nμ∈[-1.0,1.0]"], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs",
                           "truncated_normal_comparison.png")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved to {OUTPUT_PATH}")