"""
scripts/visualize_uncertainty.py

Shows how input disagreement drives uncertainty in the posterior.

Two datasets with identical means but different variance:
  - Consensus:    all scores near 0.5  → tight credible interval
  - Disagreement: scores spread across [-0.9, 0.9] → wide interval

This directly demonstrates why the empirical Bayes estimate of σ²
matters: disagreement is not averaged away, it is modelled as noise.

Run from the project root:
    python scripts/visualize_uncertainty.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from core.bayesian_model import estimate_market

# ----- datasets ------------------------------------------------------

DATASETS = {
    "Consensus":     [0.5, 0.5, 0.5, 0.5, 0.5],
    "Disagreement":  [-0.9, -0.5, 0.0, 0.5, 0.9],
}

COLORS = {
    "Consensus":    "#2ecc71",
    "Disagreement": "#e74c3c",
}

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "uncertainty.png")

# ----- run model -----------------------------------------------------

results = {label: estimate_market(scores) for label, scores in DATASETS.items()}

# ----- plot ----------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 3.5))

y_positions = {"Consensus": 1, "Disagreement": 0}

for label, result in results.items():
    y     = y_positions[label]
    color = COLORS[label]
    mean  = result.mean
    lo    = result.lower_bound
    hi    = result.upper_bound

    # Interval line
    ax.plot([lo, hi], [y, y], color=color, linewidth=3, solid_capstyle="round", zorder=2)

    # Mean marker
    ax.scatter([mean], [y], color=color, s=120, zorder=3, edgecolors="white", linewidths=1.5)

    # Interval width annotation (right of bar)
    width = hi - lo
    ax.text(
        hi + 0.03, y,
        f"width = {width:.2f}",
        va="center", ha="left",
        fontsize=10, color=color,
    )

    # Raw scores as small ticks below/above the bar
    raw = DATASETS[label]
    jitter_y = y - 0.18 if label == "Consensus" else y + 0.18
    ax.scatter(raw, [jitter_y] * len(raw),
               color=color, alpha=0.5, s=30, zorder=2, marker="|", linewidths=1.5)

# y-axis labels
ax.set_yticks(list(y_positions.values()))
ax.set_yticklabels(list(y_positions.keys()), fontsize=12)

# x-axis
ax.set_xlim(-1.25, 1.55)
ax.set_xlabel("Sentiment", fontsize=12)
ax.axvline(0, color="#cccccc", linewidth=1, linestyle="--", zorder=1)

ax.set_title(
    "Uncertainty Driven by Input Disagreement\n"
    "Same number of articles - wider spread means wider credible interval",
    fontsize=12,
    pad=12,
)

# Legend
handles = [
    mpatches.Patch(color=COLORS[label], label=label)
    for label in DATASETS
]
ax.legend(handles=handles, loc="upper left", fontsize=10)

ax.spines[["top", "right", "left"]].set_visible(False)
ax.yaxis.set_tick_params(length=0)
ax.grid(axis="x", linestyle="--", alpha=0.3)

plt.tight_layout()

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved to {OUTPUT_PATH}")