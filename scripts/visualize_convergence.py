"""
scripts/visualize_convergence.py

Demonstrates how the Bayesian model converges as sample size grows.

We fix a true underlying sentiment, generate noisy observations around
it, and show how the posterior mean and credible interval behave as
more data arrives. This is the core statistical property of the model.

Run from the project root:
    python scripts/visualize_convergence.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from core.bayesian_model import estimate_market

# ----- experiment parameters -----------------------------------------

TRUE_SENTIMENT = 0.6      # the "ground truth" we're trying to recover
NOISE_STD      = 0.4      # how noisy each LLM score is (realistic assumption)
SAMPLE_SIZES   = [1, 2, 5, 10, 20, 50]
RANDOM_SEED    = 42       # reproducibility

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "convergence.png")

# ----- generate data -------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)

# Draw the largest batch upfront, then slice - ensures each smaller
# dataset is a prefix of the larger one (fair comparison).
max_n = max(SAMPLE_SIZES)
all_scores_raw = rng.normal(loc=TRUE_SENTIMENT, scale=NOISE_STD, size=max_n)

# Clamp to [-1, 1] - same constraint the real pipeline enforces
all_scores = np.clip(all_scores_raw, -1.0, 1.0).tolist()

# ----- run model for each sample size --------------------------------

means        = []
lower_bounds = []
upper_bounds = []

for n in SAMPLE_SIZES:
    result = estimate_market(all_scores[:n])
    means.append(result.mean)
    lower_bounds.append(result.lower_bound)
    upper_bounds.append(result.upper_bound)

# ----- plot ----------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))

# Shaded credible interval
ax.fill_between(
    SAMPLE_SIZES,
    lower_bounds,
    upper_bounds,
    alpha=0.25,
    color="#4C72B0",
    label="95% credible interval",
)

# Posterior mean
ax.plot(
    SAMPLE_SIZES,
    means,
    color="#4C72B0",
    linewidth=2,
    marker="o",
    markersize=5,
    label="Posterior mean",
    zorder=3,
)

# True sentiment reference
ax.axhline(
    TRUE_SENTIMENT,
    color="#C44E52",
    linewidth=1.5,
    linestyle="--",
    label=f"True sentiment ({TRUE_SENTIMENT})",
)

# Prior mean reference
ax.axhline(
    0.0,
    color="#888888",
    linewidth=1.0,
    linestyle=":",
    label="Prior mean (0.0 - neutral)",
)

# Formatting
ax.set_xscale("log")   # log scale shows early convergence more clearly
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xticks(SAMPLE_SIZES)

ax.set_xlabel("Number of articles (n)", fontsize=12)
ax.set_ylabel("Sentiment estimate", fontsize=12)
ax.set_title(
    "Bayesian Posterior Convergence\n"
    "Credible interval narrows as more articles are observed",
    fontsize=13,
    pad=14,
)

ax.set_ylim(-1.05, 1.05)
ax.legend(loc="lower right", fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved to {OUTPUT_PATH}")