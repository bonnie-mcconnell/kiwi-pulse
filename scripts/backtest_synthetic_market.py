"""
scripts/backtest_synthetic_market.py

Simulates a market sentiment time series and compares four strategies
on their ability to predict the *direction of change* in sentiment.

TARGET
------
At each timestep t, we predict:
    sign(μ_{t+1} - μ_t)   - is sentiment rising or falling?

This is harder and more realistic than predicting sign(μ_{t+1}).
A model that just learns "sentiment is usually positive" would
score well on the sign task but fail here.

FOUR STRATEGIES
---------------
1. Bayesian          - always predict from posterior mean
2. Rolling mean      - mean of last ROLL_WIN posterior means
3. EWMA              - exponentially weighted mean of scores (α=0.3)
4. Filtered Bayesian - only predict when |posterior mean| > THRESHOLD
                       abstains otherwise; precision vs coverage tradeoff

DATA GENERATING PROCESS
------------------------
True sentiment follows AR(1) with mean reversion:
    μ_t = ρ * μ_{t-1} + ε,  ε ~ Normal(0, σ_drift²)
Observations are noisy:
    x_i ~ Normal(μ_t, σ_obs), clipped to [-1, 1]

Run from the project root:
    python scripts/backtest_synthetic_market.py
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt

from core.bayesian_model import estimate_market

logging.disable(logging.WARNING)

# ----- parameters ----------------------------------------------------

T           = 300
N_OBS       = 8
RHO         = 0.75
SIGMA_DRIFT = 0.15
SIGMA_OBS   = 0.35
ROLL_WIN    = 3
EWMA_ALPHA  = 0.3
THRESHOLD   = 0.2     # filtered Bayesian: abstain if |mean| <= this
SEED        = 55

# ----- generate true sentiment ---------------------------------------

rng     = np.random.default_rng(SEED)
true_mu = np.zeros(T)
true_mu[0] = rng.uniform(-0.3, 0.3)

for t in range(1, T):
    shock      = rng.normal(0, SIGMA_DRIFT)
    true_mu[t] = np.clip(RHO * true_mu[t - 1] + shock, -1.0, 1.0)

# direction of change: +1 if rising, -1 if falling
# defined for t = 0..T-2 (predict change from t to t+1)
true_direction = np.sign(true_mu[1:] - true_mu[:-1])

# ----- generate observations and estimates ---------------------------

post_means = []
post_vars  = []
ewma_vals  = []
roll_means = []
ewma_cur   = 0.0

for t in range(T):
    raw    = rng.normal(loc=true_mu[t], scale=SIGMA_OBS, size=N_OBS)
    scores = np.clip(raw, -1.0, 1.0).tolist()

    result = estimate_market(scores)
    post_means.append(result.mean)
    post_vars.append(result.variance)

    # EWMA over posterior means - smooth the signal across timesteps
    ewma_cur = EWMA_ALPHA * result.mean + (1 - EWMA_ALPHA) * ewma_cur
    ewma_vals.append(ewma_cur)

    window = post_means[max(0, t - ROLL_WIN + 1): t + 1]
    roll_means.append(float(np.mean(window)))

post_means = np.array(post_means)
post_vars  = np.array(post_vars)
ewma_vals  = np.array(ewma_vals)
roll_means = np.array(roll_means)

# ----- prediction: sign of change ------------------------------------
# Predict sign(μ_{t+1} - μ_t) using estimate at t.
# We use the *change in the estimate* as a proxy for change in true μ.

def predict_direction(estimates: np.ndarray) -> np.ndarray:
    """Predict direction of change: sign(estimate[t+1] - estimate[t])."""
    return np.sign(np.diff(estimates))   # length T-1

bayes_pred    = predict_direction(post_means)
roll_pred     = predict_direction(roll_means)
ewma_pred     = predict_direction(ewma_vals)

# Filtered Bayesian: only predict when the estimate is strong enough
# to be trustworthy. Abstain (0) when |mean| <= THRESHOLD.
filtered_pred = np.where(
    np.abs(post_means[:-1]) > THRESHOLD,
    bayes_pred,
    0,   # abstain
)

# ----- metrics -------------------------------------------------------

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """
    preds:   array of {-1, 0, 1}, length T-1. 0 = abstain.
    targets: array of {-1, 0, 1}, true direction of change.
    """
    mask       = preds != 0                          # timesteps where decision made
    n_decided  = int(mask.sum())
    n_total    = len(preds)

    if n_decided == 0:
        return dict(accuracy=float("nan"), precision=float("nan"),
                    decision_rate=0.0, n_decided=0)

    p = preds[mask]
    t = targets[mask]

    correct        = (p == t).sum()
    true_pos       = ((p == 1) & (t == 1)).sum()
    predicted_pos  = (p == 1).sum()

    accuracy       = correct / n_decided
    precision      = true_pos / predicted_pos if predicted_pos > 0 else float("nan")
    decision_rate  = n_decided / n_total

    return dict(
        accuracy=float(accuracy),
        precision=float(precision),
        decision_rate=float(decision_rate),
        n_decided=n_decided,
    )

strategies = {
    "Bayesian":           compute_metrics(bayes_pred,    true_direction),
    "Rolling mean":       compute_metrics(roll_pred,     true_direction),
    "EWMA (α=0.3)":       compute_metrics(ewma_pred,     true_direction),
    "Filtered Bayesian":  compute_metrics(filtered_pred, true_direction),
}

# ----- print ---------------------------------------------------------

print(f"T={T} steps  |  n={N_OBS} obs/step  |  AR(1) ρ={RHO}  "
      f"|  obs noise σ={SIGMA_OBS}  |  filter threshold={THRESHOLD}\n")
print(f"Target: predict sign(μ_{{t+1}} - μ_t)  - direction of change\n")

col = 20
print(f"{'Strategy':<22} {'Dir. accuracy':>{col}}  {'Precision':>{col}}  {'Decision rate':>{col}}  {'N decided':>{col}}")
print("-" * (22 + col * 4 + 8))

for name, m in strategies.items():
    acc  = f"{m['accuracy']:.3f}"   if not np.isnan(m['accuracy'])   else "N/A"
    prec = f"{m['precision']:.3f}"  if not np.isnan(m['precision'])  else "N/A"
    dr   = f"{m['decision_rate']:.3f}"
    nd   = str(m['n_decided'])
    print(f"  {name:<20} {acc:>{col}}  {prec:>{col}}  {dr:>{col}}  {nd:>{col}}")

print()
print(f"Baseline (random):  accuracy ≈ 0.500, precision ≈ 0.500")
print()

# Key observations
b  = strategies["Bayesian"]
fb = strategies["Filtered Bayesian"]
if not np.isnan(fb["precision"]) and not np.isnan(b["precision"]):
    if fb["precision"] > b["precision"]:
        print(f"Filtered Bayesian improves precision by "
              f"{(fb['precision'] - b['precision'])*100:.1f}pp "
              f"at a decision rate of {fb['decision_rate']:.0%}.")
    else:
        print(f"Filtering did not improve precision at threshold={THRESHOLD}. "
              f"Consider adjusting the threshold.")

# ----- plot ----------------------------------------------------------

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
fig.suptitle(
    f"Synthetic Market Backtest - Predicting Direction of Change\n"
    f"T={T}  |  AR(1) ρ={RHO}  |  Four strategies compared",
    fontsize=12, fontweight="bold",
)

t_axis = np.arange(T)

# --- Panel 1: true sentiment + all estimates -------------------------
ax = axes[0]
ax.plot(t_axis, true_mu,    color="#333333", linewidth=1.4,
        label="True μ", zorder=4)
ax.plot(t_axis, post_means, color="#4C72B0", linewidth=1.0,
        alpha=0.85, label="Bayesian posterior mean")
ax.plot(t_axis, roll_means, color="#DD8452", linewidth=1.0,
        alpha=0.75, linestyle="--", label=f"Rolling mean (w={ROLL_WIN})")
ax.plot(t_axis, ewma_vals,  color="#55a868", linewidth=1.0,
        alpha=0.75, linestyle="-.", label=f"EWMA (α={EWMA_ALPHA})")
ax.axhline(0, color="#cccccc", linewidth=0.8, linestyle=":")
ax.set_ylabel("Sentiment", fontsize=10)
ax.set_title("Sentiment Estimates Over Time", fontsize=10)
ax.legend(fontsize=8, loc="upper right", ncol=4)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.25)

# --- Panel 2: true direction of change + predictions -----------------
ax = axes[1]
t_pred = t_axis[:-1]

ax.step(t_pred, true_direction, color="#333333", linewidth=1.0,
        where="post", label="True direction", alpha=0.6, zorder=4)
ax.step(t_pred, bayes_pred,    color="#4C72B0", linewidth=1.0,
        where="post", label="Bayesian", alpha=0.7, linestyle="--")
ax.step(t_pred, ewma_pred,     color="#55a868", linewidth=1.0,
        where="post", label="EWMA", alpha=0.6, linestyle="-.")

# Mark abstentions for filtered Bayesian
abstain_t = t_pred[filtered_pred == 0]
ax.scatter(abstain_t, np.zeros(len(abstain_t)),
           color="#c44e52", s=8, alpha=0.5, zorder=3,
           label=f"Filtered Bayesian abstains (n={len(abstain_t)})")

ax.set_yticks([-1, 0, 1])
ax.set_yticklabels(["Falling", "Abstain", "Rising"], fontsize=8)
ax.set_ylabel("Predicted direction", fontsize=10)
ax.set_title("Direction Predictions vs Truth", fontsize=10)
ax.legend(fontsize=8, loc="upper right", ncol=2)
ax.spines[["top", "right"]].set_visible(False)

# --- Panel 3: rolling directional accuracy (40-step window) ----------
ax = axes[2]
win = 40

def rolling_dir_acc(preds, targets, w):
    out = []
    for t in range(w, len(preds)):
        p = preds[t - w:t]
        tgt = targets[t - w:t]
        mask = p != 0
        if mask.sum() == 0:
            out.append(float("nan"))
        else:
            out.append(float((p[mask] == tgt[mask]).mean()))
    return np.array(out)

t_roll = t_axis[win:-1]
for name, preds, color, ls in [
    ("Bayesian",          bayes_pred,    "#4C72B0", "-"),
    ("Rolling mean",      roll_pred,     "#DD8452", "--"),
    ("EWMA",              ewma_pred,     "#55a868", "-."),
    ("Filtered Bayesian", filtered_pred, "#c44e52", ":"),
]:
    racc = rolling_dir_acc(preds, true_direction, win)
    ax.plot(t_roll, racc, color=color, linewidth=1.5,
            linestyle=ls, label=name, alpha=0.85)

ax.axhline(0.5, color="#cccccc", linewidth=1.0, linestyle=":",
           label="Random chance")
ax.set_ylim(0.2, 0.9)
ax.set_ylabel("Dir. accuracy", fontsize=10)
ax.set_xlabel("Timestep", fontsize=10)
ax.set_title(f"Rolling Directional Accuracy ({win}-step window)", fontsize=10)
ax.legend(fontsize=8, loc="lower right", ncol=3)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.25)

plt.tight_layout()

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs",
                           "backtest_synthetic_market.png")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved to {OUTPUT_PATH}")