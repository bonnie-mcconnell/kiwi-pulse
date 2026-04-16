"""
scripts/demo_real_articles.py

End-to-end demo using realistic article headlines.

Runs 5 simulations of the same articles with small Gaussian noise
added to each score - mimicking the natural variability you'd get
from running an LLM multiple times on the same text.

The point: even though individual scores shift between runs,
the Bayesian posterior mean and interval stay stable.
That's the model doing its job.

Run from the project root:
    python scripts/demo_real_articles.py
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt

from core.bayesian_model import estimate_market

logging.disable(logging.WARNING)

# ----- articles + hand-scored sentiment ------------------------------
#
# Scores represent what a well-calibrated LLM should return.
# We add noise in the simulation loop to mimic real LLM variance.

ARTICLES = [
    {"title": "Fonterra lifts forecast as dairy prices surge on strong Asian demand",  "score":  0.8},
    {"title": "NZ GDP contracts for second consecutive quarter amid weak construction", "score": -0.7},
    {"title": "Reserve Bank holds OCR at 5.5%, signals cuts may come earlier",         "score":  0.4},
    {"title": "GlobalDairyTrade auction falls 3.2% - third consecutive decline",       "score": -0.5},
    {"title": "Tourism numbers recover to pre-Covid levels for first time",             "score":  0.6},
    {"title": "Westpac economists warn of elevated mortgage stress into 2025",          "score": -0.4},
    {"title": "NZX 50 closes at six-month high on positive offshore sentiment",         "score":  0.5},
    {"title": "Unemployment rises to 4.6%, highest since 2021",                        "score": -0.6},
    {"title": "Government infrastructure spend - economists split on impact",           "score":  0.1},
    {"title": "ANZ Business Confidence index rebounds sharply in October survey",      "score":  0.7},
]

N_SIMS     = 5
NOISE_STD  = 0.15
SEED       = 21

rng = np.random.default_rng(SEED)
base_scores = [a["score"] for a in ARTICLES]

# ----- run simulations -----------------------------------------------

sim_scores  = []   # scores used in each sim
sim_results = []   # MarketEstimate for each sim

for i in range(N_SIMS):
    noisy = np.clip(
        rng.normal(loc=base_scores, scale=NOISE_STD),
        -1.0, 1.0,
    ).tolist()
    sim_scores.append(noisy)
    sim_results.append(estimate_market(noisy))

# ----- print ---------------------------------------------------------

WIDTH = 72
print("=" * WIDTH)
print("  KiwiPulse - Demo: Robustness to LLM Scoring Noise")
print(f"  {len(ARTICLES)} articles  |  {N_SIMS} simulations  |  noise σ={NOISE_STD}")
print("=" * WIDTH)

print("\nNote: same inputs → different outputs due to scoring noise.")
print("      Watch how the posterior interval stays stable.\n")

print(f"  {'Article':<52}  {'Base':>5}  " + "  ".join(f"S{i+1:1d}" for i in range(N_SIMS)))
print("  " + "-" * (52 + 7 + N_SIMS * 5 + N_SIMS * 2 - 1))

for j, article in enumerate(ARTICLES):
    base  = base_scores[j]
    noisy_vals = [f"{sim_scores[i][j]:+.2f}" for i in range(N_SIMS)]
    title = article["title"][:52]
    print(f"  {title:<52}  {base:+.2f}  " + "  ".join(noisy_vals))

print()
print(f"  {'':52}  {'':5}  " + "  ".join(f"S{i+1}" for i in range(N_SIMS)))
print("-" * WIDTH)

means  = [r.mean         for r in sim_results]
lowers = [r.lower_bound  for r in sim_results]
uppers = [r.upper_bound  for r in sim_results]
widths = [r.upper_bound - r.lower_bound for r in sim_results]

print(f"  {'Posterior mean':<22} " + "  ".join(f"{m:+.3f}" for m in means))
print(f"  {'Lower bound (95% CI)':<22} " + "  ".join(f"{l:+.3f}" for l in lowers))
print(f"  {'Upper bound (95% CI)':<22} " + "  ".join(f"{u:+.3f}" for u in uppers))
print(f"  {'Interval width':<22} " + "  ".join(f" {w:.3f}" for w in widths))

mean_range  = max(means)  - min(means)
width_range = max(widths) - min(widths)

print()
print(f"  Mean spread across sims:            {mean_range:.3f}")
print(f"  Interval width spread across sims:  {width_range:.3f}")
print()
print("  Despite per-article score noise, the posterior stays consistent.")
print("  That's the aggregation doing its job - noise averages out.")
print()
print("=" * WIDTH)

# ----- plot ----------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle(
    "Robustness to LLM Scoring Noise\n"
    "Same articles, 5 simulations with σ=0.15 score noise",
    fontsize=12, fontweight="bold",
)

sim_labels = [f"Sim {i+1}" for i in range(N_SIMS)]
colors     = plt.cm.Blues(np.linspace(0.45, 0.85, N_SIMS))

# Left: posterior means + intervals across sims
ax = axes[0]
for i, (result, color) in enumerate(zip(sim_results, colors)):
    ax.plot([result.lower_bound, result.upper_bound], [i, i],
            color=color, linewidth=4, solid_capstyle="round")
    ax.scatter([result.mean], [i], color=color, s=100,
               zorder=3, edgecolors="white", linewidths=1.5)

ax.axvline(0, color="#cccccc", linewidth=1, linestyle="--")
ax.set_yticks(range(N_SIMS))
ax.set_yticklabels(sim_labels, fontsize=10)
ax.set_xlabel("Sentiment estimate", fontsize=10)
ax.set_title("Posterior mean + 95% CI per simulation", fontsize=10)
ax.set_xlim(-1.1, 1.1)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.3)

# Right: per-article score variability across sims
ax = axes[1]
x = np.arange(len(ARTICLES))
for i, (scores, color) in enumerate(zip(sim_scores, colors)):
    ax.scatter(x, scores, color=color, alpha=0.7, s=30,
               label=f"Sim {i+1}", zorder=2)
ax.scatter(x, base_scores, color="#333333", s=50, marker="D",
           zorder=3, label="Base score", linewidths=0.5)

ax.axhline(0, color="#cccccc", linewidth=1, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels([f"A{j+1}" for j in range(len(ARTICLES))], fontsize=8)
ax.set_xlabel("Article", fontsize=10)
ax.set_ylabel("Score", fontsize=10)
ax.set_title("Per-article score variability across simulations", fontsize=10)
ax.legend(fontsize=8, loc="lower right", ncol=3)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "demo_noise_robustness.png")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"Plot saved to {OUTPUT_PATH}")