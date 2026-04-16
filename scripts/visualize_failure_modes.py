"""
scripts/visualize_failure_modes.py

Demonstrates three cases where the Normal-Normal conjugate model
produces outputs that are technically correct but potentially misleading.

These are not bugs - they are fundamental limitations of the model
assumptions. Understanding them is as important as understanding
what the model does well.

Case 1 - Bimodal data
    Two clusters of opinion at opposite ends of the scale.
    The Gaussian likelihood cannot represent this. The posterior
    mean lands near zero, suggesting neutrality, when the data
    actually shows strong polarisation.

Case 2 - Extreme outlier
    One highly negative score among consistent positives.
    The model treats all observations as equally valid. A single
    outlier pulls the mean and inflates σ², widening the interval.
    The model has no concept of source reliability.

Case 3 - Tiny sample (n=1)
    With one observation, we cannot estimate σ² from data.
    We fall back to maximum assumed noise (σ²=1.0), which keeps
    the posterior close to the prior. The interval is wide by design.

Run from the project root:
    python scripts/visualize_failure_modes.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.patches as mpatches

from core.bayesian_model import estimate_market

# ----- cases ---------------------------------------------------------

CASES = [
    {
        "label":      "Case 1: Bimodal",
        "scores":     [-0.8, -0.7, 0.7, 0.8],
        "annotation": "Two opposing clusters.\nMean ≈ 0 suggests neutrality,\nbut the data shows polarisation.\nGaussian cannot model this.",
    },
    {
        "label":      "Case 2: Outlier",
        "scores":     [0.5, 0.5, 0.5, -1.0],
        "annotation": "One extreme score inflates σ²\nand pulls the mean down.\nNo source weighting -\nall observations treated equally.",
    },
    {
        "label":      "Case 3: Tiny sample",
        "scores":     [0.9],
        "annotation": "n=1: σ² cannot be estimated.\nFalls back to σ²=1.0 (max noise).\nPrior dominates - posterior\nstays near 0, not 0.9.",
    },
]

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "failure_modes.png")

# ----- run model -----------------------------------------------------

results = [estimate_market(c["scores"]) for c in CASES]

# ----- plot ----------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
fig.suptitle(
    "Model Failure Modes - Where the Gaussian Assumption Breaks Down",
    fontsize=13, fontweight="bold", y=1.01,
)

INTERVAL_COLOR = "#4C72B0"
POINT_COLOR    = "#e67e22"
MISS_COLOR     = "#e74c3c"

for ax, case, result in zip(axes, CASES, results):
    scores = case["scores"]
    mean   = result.mean
    lo     = result.lower_bound
    hi     = result.upper_bound

    # --- credible interval bar ---
    ax.plot([lo, hi], [0.5, 0.5],
            color=INTERVAL_COLOR, linewidth=4,
            solid_capstyle="round", zorder=2,
            label="95% credible interval")

    # --- posterior mean marker ---
    ax.scatter([mean], [0.5],
               color=INTERVAL_COLOR, s=140, zorder=4,
               edgecolors="white", linewidths=1.8,
               label=f"Posterior mean ({mean:.2f})")

    # --- raw scores as jittered points ---
    jitter = np.linspace(0.62, 0.72, len(scores))
    ax.scatter(scores, jitter,
               color=POINT_COLOR, s=55, zorder=3,
               alpha=0.85, label="Raw scores")

    # --- vertical line at zero (prior mean) ---
    ax.axvline(0, color="#aaaaaa", linewidth=1.0,
               linestyle=":", zorder=1, label="Prior mean (0)")

    # --- annotation box ---
    ax.text(
        0.5, 0.08,
        case["annotation"],
        transform=ax.transAxes,
        fontsize=8.5,
        va="bottom", ha="center",
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                  edgecolor="#cccccc", linewidth=0.8),
    )

    # --- formatting ---
    ax.set_title(case["label"], fontsize=11, fontweight="bold", pad=10)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Sentiment", fontsize=10)
    ax.set_yticks([])

    # interval width label
    ax.text(
        lo, 0.42, f"[{lo:.2f},",
        fontsize=8, color=INTERVAL_COLOR, ha="right", va="top"
    )
    ax.text(
        hi, 0.42, f"{hi:.2f}]",
        fontsize=8, color=INTERVAL_COLOR, ha="left", va="top"
    )

    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

# shared legend at bottom
handles = [
    matplotlib.lines.Line2D([0], [0], color=INTERVAL_COLOR, linewidth=3, label="95% credible interval"),
    plt.scatter([], [], color=INTERVAL_COLOR, s=80, label="Posterior mean"),
    plt.scatter([], [], color=POINT_COLOR, s=60, label="Raw scores"),
    matplotlib.lines.Line2D([0], [0], color="#aaaaaa", linewidth=1.2,
               linestyle=":", label="Prior mean (0)"),
]
fig.legend(
    handles=handles,
    loc="lower center",
    ncol=4,
    fontsize=9,
    bbox_to_anchor=(0.5, -0.08),
    frameon=False,
)

plt.tight_layout()

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved to {OUTPUT_PATH}")