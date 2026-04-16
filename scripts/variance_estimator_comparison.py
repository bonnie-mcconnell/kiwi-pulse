"""
scripts/variance_estimator_comparison.py

Compares three variance estimation strategies for the Bayesian model,
focusing on stability at small sample sizes.

THE PROBLEM
-----------
The current model estimates σ² directly from sample variance (empirical Bayes):

    σ²_hat = Var(x_1, ..., x_n)

This is unbiased but highly variable at small n. With n=3, the sample
variance of a single draw can range from near-zero to very large,
causing the posterior to swing between overconfident and very uncertain.

THREE STRATEGIES
----------------
1. Empirical Bayes (current)
   σ² = sample variance, with a hard floor at EPSILON.
   Unstable at small n. Baseline.

2. Floored estimator
   σ² = max(sample_variance, FLOOR)
   FLOOR = 0.1  (corresponds to std ≈ 0.316)
   Simple, principled. Prevents near-zero variance that produces
   falsely tight intervals. The floor is a domain assumption: we
   believe LLM scores always have at least some irreducible noise.

3. Hierarchical (pooled) estimator
   Maintain a running estimate of the population variance across
   all batches seen so far. Blend the current sample variance toward
   this global estimate, weighted by how many batches we've seen.

       σ²_hier = (n_batches * σ²_global + n * σ²_sample) /
                 (n_batches + n)

   With few batches, falls back to the global prior (σ²_global = 0.16).
   With many batches, converges toward the true population variance.
   This is empirical Bayes at the batch level, not the observation level.

EVALUATION
----------
For each strategy, we measure across n ∈ {2, 5, 10, 20}:
  - Coverage   (target: 0.95)
  - Width      (narrower is better if coverage holds)
  - Variance of σ² estimates (stability - lower = more stable)

Run from the project root:
    python scripts/variance_estimator_comparison.py
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

import core.bayesian_model as bm

logging.disable(logging.WARNING)

# ----- parameters ----------------------------------------------------

N_RUNS       = 2000
NOISE_STD    = 0.4
MU_LO, MU_HI = -0.6, 0.6
SAMPLE_SIZES = [2, 5, 10, 20]
SEED         = 88

# Floored estimator floor value
VAR_FLOOR = 0.10

# Hierarchical prior: initial belief about population variance
# 0.16 = std of 0.4, matching our known observation noise
HIER_PRIOR_VAR    = 0.16
HIER_PRIOR_WEIGHT = 5   # equivalent to having seen 5 "prior batches"

# ----- variance estimators -------------------------------------------

def empirical_var(x: np.ndarray) -> float:
    """Current approach: raw sample variance with epsilon floor."""
    n = len(x)
    if n == 1:
        return bm._SINGLE_OBS_VAR
    v = float(np.var(x, ddof=1))
    return v if v > 0 else bm.EPSILON_ZERO_VAR


def floored_var(x: np.ndarray) -> float:
    """Sample variance with a principled minimum floor."""
    n = len(x)
    if n == 1:
        return max(bm._SINGLE_OBS_VAR, VAR_FLOOR)
    return max(float(np.var(x, ddof=1)), VAR_FLOOR)


class HierarchicalVarianceEstimator:
    """
    Maintains a running population variance estimate across batches.
    Blends each new sample variance toward the population estimate.

    This converts the per-batch variance problem into a pooling problem:
    early batches lean on the prior; later batches trust the data more.
    """

    def __init__(self, prior_var: float = HIER_PRIOR_VAR,
                 prior_weight: int = HIER_PRIOR_WEIGHT):
        self.prior_var    = prior_var
        self.prior_weight = prior_weight
        self._sum_var     = prior_var * prior_weight
        self._n_batches   = prior_weight

    def estimate(self, x: np.ndarray) -> float:
        n = len(x)
        if n == 1:
            # Can't estimate from one point - use current global estimate
            sample_var = self._sum_var / self._n_batches
        else:
            sample_var = max(float(np.var(x, ddof=1)), bm.EPSILON_ZERO_VAR)
            # Update running estimate (online mean of variances)
            self._sum_var  += sample_var
            self._n_batches += 1

        # Blend: current sample toward population estimate
        global_var = self._sum_var / self._n_batches
        blended    = (self._n_batches * global_var + n * sample_var) / \
                     (self._n_batches + n)
        return float(blended)

    def reset(self):
        self._sum_var   = self.prior_var * self.prior_weight
        self._n_batches = self.prior_weight


# ----- core inference (bypasses estimate_market to swap variance) ----

def posterior(x: np.ndarray, obs_var: float) -> tuple[float, float]:
    """
    Compute posterior mean and variance given pre-computed obs_var.
    Uses the same Normal-Normal conjugate as bayesian_model.py.
    """
    n              = len(x)
    prior_prec     = 1.0 / bm.PRIOR_VAR
    data_prec      = n   / obs_var
    post_var       = 1.0 / (prior_prec + data_prec)
    post_mean      = post_var * (bm.PRIOR_MEAN * prior_prec +
                                 float(x.sum()) / obs_var)
    return post_mean, post_var


def interval(post_mean: float, post_var: float) -> tuple[float, float]:
    margin = 1.96 * np.sqrt(post_var)
    return (max(-1.0, post_mean - margin),
            min( 1.0, post_mean + margin))


# ----- simulation ----------------------------------------------------

STRATEGIES = ["Empirical Bayes", "Floored (floor=0.10)", "Hierarchical"]

results = {s: {n: {"hits": [], "widths": [], "obs_vars": []}
               for n in SAMPLE_SIZES}
           for s in STRATEGIES}

hier = HierarchicalVarianceEstimator()

for n in SAMPLE_SIZES:
    rng  = np.random.default_rng(SEED + n)
    hier.reset()   # fresh estimator per sample-size experiment

    for _ in range(N_RUNS):
        true_mu = rng.uniform(MU_LO, MU_HI)
        raw     = rng.normal(true_mu, NOISE_STD, n)
        x       = np.clip(raw, -1.0, 1.0)

        for strategy in STRATEGIES:
            if strategy == "Empirical Bayes":
                obs_var = empirical_var(x)
            elif strategy == "Floored (floor=0.10)":
                obs_var = floored_var(x)
            else:
                obs_var = hier.estimate(x)

            pm, pv    = posterior(x, obs_var)
            lo, hi    = interval(pm, pv)

            results[strategy][n]["hits"].append(lo <= true_mu <= hi)
            results[strategy][n]["widths"].append(hi - lo)
            results[strategy][n]["obs_vars"].append(obs_var)

# ----- print ---------------------------------------------------------

print(f"N={N_RUNS} runs per cell  |  noise σ={NOISE_STD}  |  μ ∈ [{MU_LO}, {MU_HI}]\n")

for metric, label, fmt in [
    ("hits",     "Coverage  (target 0.95)",       ".3f"),
    ("widths",   "Interval width  (lower=better)", ".3f"),
    ("obs_vars", "σ² estimate std (lower=stable)", ".4f"),
]:
    print(f"── {label} ──\n")
    header = f"  {'Strategy':<28}" + "".join(f"  n={n:>2}" for n in SAMPLE_SIZES)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for s in STRATEGIES:
        row = f"  {s:<28}"
        for n in SAMPLE_SIZES:
            vals = results[s][n][metric]
            if metric == "hits":
                val = float(np.mean(vals))
            elif metric == "obs_vars":
                val = float(np.std(vals))      # stability = low std of estimates
            else:
                val = float(np.mean(vals))
            row += f"  {val:{fmt}}"
        print(row)
    print()

# Key finding: at n=2, which strategy is most stable?
print("── Key finding: n=2 stability ──\n")
for s in STRATEGIES:
    std_v = float(np.std(results[s][2]["obs_vars"]))
    cov   = float(np.mean(results[s][2]["hits"]))
    print(f"  {s:<30}  σ²_std={std_v:.4f}  coverage={cov:.3f}")

# ----- plot ----------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.suptitle(
    "Variance Estimator Comparison - Stability at Small n\n"
    f"N={N_RUNS} trials per cell  |  True σ={NOISE_STD}",
    fontsize=12, fontweight="bold",
)

COLORS = {"Empirical Bayes": "#4C72B0",
          "Floored (floor=0.10)": "#DD8452",
          "Hierarchical": "#55a868"}

metric_specs = [
    ("hits",     "Coverage",             "Interval Width",     0.70, 1.0),
    ("widths",   "Interval Width",       "Mean interval width", 0.0,  1.0),
    ("obs_vars", "σ² Estimate Std Dev",  "Std dev of σ²",      0.0,  None),
]

for ax, (metric, title, ylabel, ylo, yhi) in zip(axes, metric_specs):
    for s in STRATEGIES:
        vals = []
        for n in SAMPLE_SIZES:
            v = results[s][n][metric]
            if metric == "hits":
                vals.append(float(np.mean(v)))
            elif metric == "obs_vars":
                vals.append(float(np.std(v)))
            else:
                vals.append(float(np.mean(v)))
        ax.plot(SAMPLE_SIZES, vals, marker="o", linewidth=2,
                markersize=6, label=s, color=COLORS[s])

    if metric == "hits":
        ax.axhline(0.95, color="#333333", linewidth=1.2,
                   linestyle="--", label="Target (0.95)")

    ax.set_xlabel("Sample size n", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(SAMPLE_SIZES)
    if yhi:
        ax.set_ylim(ylo, yhi)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs",
                           "variance_estimator_comparison.png")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved to {OUTPUT_PATH}")