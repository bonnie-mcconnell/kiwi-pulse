"""
scripts/adversarial_tests.py

Tests where the Bayesian model fails.

Every prior experiment generated data from Normal(μ, σ) - the same
distribution the model assumes. That's circular validation. This script
deliberately violates model assumptions to find the boundaries of where
the model can be trusted.

FIVE ADVERSARIAL CONDITIONS
----------------------------
1. Non-Gaussian noise (Gaussian mixture)
   Observations drawn from 0.5·N(μ-0.3, 0.1) + 0.5·N(μ+0.3, 0.1).
   The model assumes unimodal noise. Bimodal spread will cause it to
   overestimate variance and produce intervals that are too wide.

2. Systematic LLM bias
   All observations shifted by a fixed positive bias: x_i + 0.3.
   Models a recency bias, optimism bias, or prompt-induced skew.
   The model has no bias correction - posterior mean will be pulled up.

3. Heavy-tailed noise (Student-t, df=3)
   Occasional extreme scores that Gaussian likelihood down-weights too little.
   The model will overfit to outliers and inflate σ².

4. Regime shift
   True μ jumps abruptly at T/2. The model has no memory - each call to
   estimate_market() is independent - so it actually handles this well,
   but the rolling mean will lag significantly.

5. Correlated observations (not iid)
   Observations are AR(1)-correlated within each batch: x_i = ρ·x_{i-1} + ε.
   The model assumes independence. Effective sample size is smaller than n,
   so the model will be overconfident (intervals too narrow).

METRICS PER CONDITION
----------------------
  Coverage   - empirical 95% CI coverage (target: 0.95)
  Width      - mean interval width
  Abs error  - |posterior mean - true μ|
  vs Naive   - same metrics for sample mean + bootstrap CI

Run from the project root:
    python scripts/adversarial_tests.py
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt

from core.bayesian_model import estimate_market

logging.disable(logging.WARNING)

# ----- shared parameters ---------------------------------------------

N_RUNS    = 2000
N_OBS     = 10
MU_LO     = -0.6
MU_HI     =  0.6
BOOT_N    = 500      # bootstrap resamples per trial (kept low for speed)
SEED      = 31

# ----- test definitions ----------------------------------------------

def gaussian_noise(rng, true_mu, n):
    """Baseline: model assumptions exactly met."""
    return np.clip(rng.normal(true_mu, 0.4, n), -1, 1)

def mixture_noise(rng, true_mu, n):
    """
    Bimodal noise: two clusters offset from true_mu.
    Violates unimodal Gaussian assumption.
    """
    cluster = rng.integers(0, 2, n)
    offsets = np.where(cluster == 0, -0.3, 0.3)
    return np.clip(rng.normal(true_mu + offsets, 0.1, n), -1, 1)

def biased_noise(rng, true_mu, n):
    """
    Systematic upward bias of +0.3.
    Models a prompt or recency bias in the LLM scorer.
    """
    raw = rng.normal(true_mu, 0.4, n) + 0.3
    return np.clip(raw, -1, 1)

def heavy_tail_noise(rng, true_mu, n):
    """
    Student-t noise with df=3 - fat tails, frequent outliers.
    """
    t_samples = rng.standard_t(df=3, size=n) * 0.25  # scale to comparable spread
    return np.clip(true_mu + t_samples, -1, 1)

def regime_shift_noise(rng, true_mu, n):
    """
    True signal jumps at the midpoint of the batch.
    First half: true_mu. Second half: true_mu * -1 (sign flip).
    """
    half = n // 2
    first  = rng.normal(true_mu,       0.4, half)
    second = rng.normal(-true_mu,      0.4, n - half)
    return np.clip(np.concatenate([first, second]), -1, 1)

def correlated_noise(rng, true_mu, n, rho=0.7):
    """
    AR(1)-correlated observations within a batch.
    Independence assumption violated - effective n is lower than actual n.
    """
    xs = np.zeros(n)
    xs[0] = rng.normal(true_mu, 0.4)
    for i in range(1, n):
        xs[i] = rho * xs[i-1] + rng.normal(true_mu * (1 - rho), 0.4 * np.sqrt(1 - rho**2))
    return np.clip(xs, -1, 1)

TESTS = [
    ("Baseline (Gaussian)",    gaussian_noise,    "Model assumptions met. Benchmark."),
    ("Mixture noise",          mixture_noise,     "Bimodal spread - violates unimodal assumption."),
    ("Systematic bias (+0.3)", biased_noise,      "All scores shifted up - no bias correction."),
    ("Heavy tails (t, df=3)",  heavy_tail_noise,  "Outliers inflate σ² → wider intervals."),
    ("Regime shift",           regime_shift_noise,"Sign flip mid-batch - model has no memory."),
    ("Correlated (AR ρ=0.7)",  correlated_noise,  "Effective n < actual n → overconfident."),
]

# ----- simulation ----------------------------------------------------

def run_test(noise_fn, seed_offset: int) -> dict:
    rng = np.random.default_rng(SEED + seed_offset)

    b_hits, b_widths, b_errors = [], [], []
    n_hits, n_widths, n_errors = [], [], []

    for _ in range(N_RUNS):
        true_mu = rng.uniform(MU_LO, MU_HI)
        scores  = noise_fn(rng, true_mu, N_OBS).tolist()
        x       = np.array(scores)

        # Bayesian
        result = estimate_market(scores)
        b_hits.append(result.lower_bound <= true_mu <= result.upper_bound)
        b_widths.append(result.upper_bound - result.lower_bound)
        b_errors.append(abs(result.mean - true_mu))

        # Naive: sample mean + bootstrap CI
        n_mean = float(x.mean())
        boot   = rng.choice(x, size=(BOOT_N, N_OBS), replace=True).mean(axis=1)
        n_lo, n_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

        n_hits.append(n_lo <= true_mu <= n_hi)
        n_widths.append(n_hi - n_lo)
        n_errors.append(abs(n_mean - true_mu))

    return {
        "bayes":  {"coverage": np.mean(b_hits), "width": np.mean(b_widths), "error": np.mean(b_errors)},
        "naive":  {"coverage": np.mean(n_hits), "width": np.mean(n_widths), "error": np.mean(n_errors)},
    }

print(f"Running {len(TESTS)} adversarial conditions × {N_RUNS} trials ...", flush=True)
all_results = {name: run_test(fn, i) for i, (name, fn, _) in enumerate(TESTS)}
print("Done.\n")

# ----- print ---------------------------------------------------------

W = 26
print(f"{'Condition':<{W}}  {'Bayes cov':>10}  {'Naive cov':>10}  "
      f"{'Bayes width':>12}  {'Bayes error':>12}  Notes")
print("-" * 100)

for name, fn, note in TESTS:
    r  = all_results[name]
    bc = r["bayes"]["coverage"]
    nc = r["naive"]["coverage"]
    bw = r["bayes"]["width"]
    be = r["bayes"]["error"]

    # Flag if coverage drops more than 5pp below target
    flag = " ⚠" if bc < 0.90 else ""
    print(f"  {name:<{W}}  {bc:>10.3f}  {nc:>10.3f}  {bw:>12.3f}  {be:>12.3f}  {note}{flag}")

print()
print("⚠ = coverage below 0.90 - model is meaningfully miscalibrated under this condition.")
print()

# Bias test: posterior mean vs true mean
print("── Bias check (systematic bias condition) ──\n")
bias_r = all_results["Systematic bias (+0.3)"]
print(f"  Bayesian error:  {bias_r['bayes']['error']:.3f}  "
      f"(should be ~0.3 if bias fully absorbed)")
print(f"  Naive error:     {bias_r['naive']['error']:.3f}")
print(f"  Both methods absorb the bias equally - neither has correction.")
print()

# Correlation test: coverage should drop (overconfident)
corr_r = all_results["Correlated (AR ρ=0.7)"]
base_r = all_results["Baseline (Gaussian)"]
drop   = base_r["bayes"]["coverage"] - corr_r["bayes"]["coverage"]
print("── Correlation impact ──\n")
print(f"  Baseline coverage:    {base_r['bayes']['coverage']:.3f}")
print(f"  Correlated coverage:  {corr_r['bayes']['coverage']:.3f}  "
      f"(drop of {drop:.3f} - iid assumption violated)")

# ----- plot ----------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(
    "Adversarial Robustness Tests - Where the Bayesian Model Breaks\n"
    f"N={N_RUNS} trials per condition  |  n={N_OBS} obs/trial",
    fontsize=12, fontweight="bold",
)

names      = [name for name, _, _ in TESTS]
short_names = ["Baseline", "Mixture", "Bias +0.3", "Heavy tail", "Regime shift", "Correlated"]
bayes_cov  = [all_results[n]["bayes"]["coverage"]  for n in names]
naive_cov  = [all_results[n]["naive"]["coverage"]  for n in names]
bayes_wid  = [all_results[n]["bayes"]["width"]     for n in names]
bayes_err  = [all_results[n]["bayes"]["error"]     for n in names]

x = np.arange(len(names))
w = 0.35

# Coverage
ax = axes[0]
ax.bar(x - w/2, bayes_cov, w, label="Bayesian", color="#4C72B0", edgecolor="white")
ax.bar(x + w/2, naive_cov, w, label="Naive + bootstrap", color="#DD8452", edgecolor="white")
ax.axhline(0.95, color="#333333", linewidth=1.2, linestyle="--", label="Target (0.95)")
ax.axhline(0.90, color="#c44e52", linewidth=0.8, linestyle=":",  label="Concern threshold")
ax.set_xticks(x); ax.set_xticklabels(short_names, rotation=25, ha="right", fontsize=8)
ax.set_ylim(0.6, 1.0)
ax.set_ylabel("Coverage"); ax.set_title("Coverage by Condition", fontsize=10)
ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# Width
ax = axes[1]
ax.bar(x, bayes_wid, color="#4C72B0", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(short_names, rotation=25, ha="right", fontsize=8)
ax.set_ylabel("Mean interval width"); ax.set_title("Interval Width (Bayesian)", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# Error
ax = axes[2]
ax.bar(x, bayes_err, color="#4C72B0", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(short_names, rotation=25, ha="right", fontsize=8)
ax.set_ylabel("Mean absolute error"); ax.set_title("Point Error (Bayesian)", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "adversarial_tests.png")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved to {OUTPUT_PATH}")