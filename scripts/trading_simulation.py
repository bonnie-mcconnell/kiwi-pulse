"""
scripts/trading_simulation.py

Simulates a simple long/short strategy driven by the Bayesian model
on the 30-day real-world dataset from real_world_validation.py.

STRATEGY
--------
  lower_bound > 0  → long SPY (size proportional to posterior mean)
  upper_bound < 0  → short SPY (size proportional to |posterior mean|)
  CI crosses zero  → no position (abstain)

Position sizing is confidence-weighted: a posterior mean of 0.8
takes a full position; a mean of 0.2 takes a 25% position.
This is the simplest form of uncertainty-aware sizing.

TRANSACTION COSTS
-----------------
5 basis points per trade (0.05%), applied on position changes.
This is conservative for a daily strategy on a liquid ETF.

BASELINES
---------
1. Always-long SPY (buy and hold)
2. Naive mean signal (always commits, equal sizing)

IMPORTANT LIMITATIONS
---------------------
- N=30 trading days. No Sharpe ratio is statistically meaningful
  at this sample size. Results are illustrative only.
- Sentiment scores were hand-assigned with date knowledge, which
  may introduce look-ahead bias (see real_world_validation.py).
- This is a demonstration of the decision framework, not a
  validated trading strategy.

Run from the project root:
    python scripts/trading_simulation.py
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt

from core.bayesian_model import estimate_market

logging.disable(logging.WARNING)

# ----- same dataset as real_world_validation.py ----------------------
# Reproduced here so this script is self-contained.

TRADING_DAYS = [
    # (date, scores, spy_next_day_return)
    ("2025-01-02", [0.5, -0.2, -0.1],   0.0056),
    ("2025-01-03", [0.6,  0.7,  0.5],  -0.0021),
    ("2025-01-06", [-0.4, 0.3, -0.5],   0.0074),
    ("2025-01-07", [-0.7, -0.6, -0.7], -0.0118),
    ("2025-01-08", [0.1, -0.1,  0.5],   0.0043),
    ("2025-01-09", [-0.4, 0.3, -0.3],  -0.0016),
    ("2025-01-10", [-0.6, -0.5, -0.7], -0.0089),
    ("2025-01-13", [0.4,  0.7,  0.8],   0.0112),
    ("2025-01-14", [0.5,  0.4,  0.2],   0.0034),
    ("2025-01-15", [0.5,  0.6,  0.7],   0.0098),
    ("2025-01-16", [-0.4, -0.5, -0.6], -0.0031),
    ("2025-01-17", [0.0, -0.1,  0.1],   0.0015),
    ("2025-01-21", [0.6,  0.7,  0.4],   0.0134),
    ("2025-01-22", [-0.6, -0.7, -0.2], -0.0055),
    ("2025-01-23", [0.7,  0.6,  0.6],   0.0088),
    ("2025-01-24", [0.6,  0.5,  0.6],   0.0041),
    ("2025-01-27", [-0.8, -0.9, -0.8], -0.0152),
    ("2025-01-28", [0.3,  0.0,  0.4],   0.0063),
    ("2025-01-29", [-0.2, -0.3,  0.1], -0.0022),
    ("2025-01-30", [0.3,  0.9,  0.4],   0.0071),
    ("2025-01-31", [-0.7, -0.8, -0.6], -0.0108),
    ("2025-02-03", [-0.6, -0.5,  0.2],  0.0039),
    ("2025-02-04", [0.4,  0.5,  0.6],   0.0067),
    ("2025-02-05", [0.4, -0.2,  0.5],   0.0028),
    ("2025-02-06", [0.3, -0.2, -0.2],  -0.0019),
    ("2025-02-07", [-0.5, -0.6, -0.7], -0.0091),
    ("2025-02-10", [-0.6, -0.5,  0.4], -0.0044),
    ("2025-02-11", [-0.2, -0.1, -0.3],  0.0018),
    ("2025-02-12", [-0.8, -0.7, -0.8], -0.0134),
    ("2025-02-13", [-0.5, -0.4, -0.1],  0.0055),
]

COST_BPS = 5 / 10_000   # 5 basis points per trade

# ----- run strategies ------------------------------------------------

def run_strategy(positions: list[float], returns: list[float]) -> dict:
    """
    Simulate a strategy given daily position sizes and actual returns.

    position: +1.0 = fully long, -1.0 = fully short, 0 = flat
    Cost applied on any change in position.
    """
    portfolio   = 1.0
    daily_pnl   = []
    prev_pos    = 0.0

    for pos, ret in zip(positions, returns):
        cost   = abs(pos - prev_pos) * COST_BPS
        pnl    = pos * ret - cost
        portfolio *= (1 + pnl)
        daily_pnl.append(pnl)
        prev_pos = pos

    pnl_arr   = np.array(daily_pnl)
    total_ret = portfolio - 1.0
    sharpe    = (pnl_arr.mean() / pnl_arr.std() * np.sqrt(252)
                 if pnl_arr.std() > 0 else float("nan"))
    max_dd    = _max_drawdown(pnl_arr)
    win_rate  = (pnl_arr > 0).mean()

    return dict(
        total_return=total_ret,
        sharpe=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        daily_pnl=daily_pnl,
        n_trades=int(sum(1 for i in range(1, len(positions))
                         if positions[i] != positions[i-1])),
        n_flat=int(sum(1 for p in positions if p == 0)),
    )


def _max_drawdown(pnl: np.ndarray) -> float:
    cum   = np.cumprod(1 + pnl)
    peak  = np.maximum.accumulate(cum)
    dd    = (cum - peak) / peak
    return float(dd.min())


# ----- compute positions for each strategy ---------------------------

dates   = [d[0] for d in TRADING_DAYS]
scores  = [d[1] for d in TRADING_DAYS]
returns = [d[2] for d in TRADING_DAYS]

bayes_pos = []
naive_pos = []

for day_scores in scores:
    r          = estimate_market(day_scores)
    mean_score = float(np.mean(day_scores))

    # Bayesian: abstain if CI crosses zero, else size by posterior mean
    if r.lower_bound > 0:
        bayes_pos.append(r.mean)    # long, confidence-weighted
    elif r.upper_bound < 0:
        bayes_pos.append(r.mean)    # short (negative mean), confidence-weighted
    else:
        bayes_pos.append(0.0)       # abstain

    # Naive: always commit with unit size in direction of mean
    naive_pos.append(1.0 if mean_score > 0 else -1.0)

# Buy-and-hold: always +1
bah_pos = [1.0] * len(returns)

bayes_result = run_strategy(bayes_pos, returns)
naive_result = run_strategy(naive_pos, returns)
bah_result   = run_strategy(bah_pos,  returns)

# ----- print ---------------------------------------------------------

print("=" * 66)
print("  Trading Simulation - Uncertainty-Aware Decision Making")
print(f"  {len(TRADING_DAYS)} trading days  |  {COST_BPS*10000:.0f}bps transaction cost")
print()
print("  ⚠  N=30. No metric here is statistically reliable.")
print("  ⚠  Scores hand-assigned - possible look-ahead bias.")
print("  ⚠  Results illustrate the framework, not a real edge.")
print("=" * 66)
print()

print(f"{'Metric':<26} {'Bayesian':>12}  {'Naive mean':>12}  {'Buy & Hold':>12}")
print("-" * 68)

rows = [
    ("Total return",   "total_return", ".2%"),
    ("Sharpe (ann.)*", "sharpe",       ".2f"),
    ("Max drawdown",   "max_drawdown", ".2%"),
    ("Win rate",       "win_rate",     ".1%"),
    ("Trades made",    "n_trades",     "d"),
    ("Days flat",      "n_flat",       "d"),
]

for label, key, fmt in rows:
    bv = bayes_result[key]
    nv = naive_result[key]
    hv = bah_result[key]
    print(f"  {label:<24} {bv:>12{fmt}}  {nv:>12{fmt}}  {hv:>12{fmt}}")

print()
print("* Sharpe annualised from 30 daily returns - not meaningful at this n.")
print()

# Key insight
b_ret = bayes_result["total_return"]
n_ret = naive_result["total_return"]
h_ret = bah_result["total_return"]
flat  = bayes_result["n_flat"]

print("Key observation:")
if b_ret > n_ret:
    print(f"  Bayesian strategy outperformed naive mean by "
          f"{(b_ret - n_ret)*100:.1f}pp total return,")
    print(f"  while sitting flat on {flat}/{len(TRADING_DAYS)} days "
          f"({flat/len(TRADING_DAYS):.0%} abstain rate).")
    print(f"  Fewer, higher-confidence decisions outperformed more frequent ones.")
else:
    print(f"  Naive mean outperformed Bayesian by "
          f"{(n_ret - b_ret)*100:.1f}pp total return.")
    print(f"  At N=30 this is within noise - insufficient data to conclude.")

# ----- plot ----------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    "Trading Simulation - Bayesian Abstention vs Always-Commit\n"
    "⚠ N=30, hand-scored data - illustrative only, not a validated backtest",
    fontsize=11, fontweight="bold",
)

# Cumulative return
ax = axes[0]
cum_bayes = np.cumprod([1 + p for p in bayes_result["daily_pnl"]])
cum_naive = np.cumprod([1 + p for p in naive_result["daily_pnl"]])
cum_bah   = np.cumprod([1 + p for p in bah_result["daily_pnl"]])

t = range(len(dates))
ax.plot(t, cum_bayes - 1, color="#2ecc71", linewidth=2,
        label=f"Bayesian (abstain={flat}d)", zorder=3)
ax.plot(t, cum_naive - 1, color="#4C72B0", linewidth=2,
        linestyle="--", label="Naive mean")
ax.plot(t, cum_bah   - 1, color="#888888", linewidth=1.5,
        linestyle=":", label="Buy & hold")
ax.axhline(0, color="#cccccc", linewidth=0.8)
ax.set_xticks(list(t)[::5])
ax.set_xticklabels([dates[i] for i in range(0, len(dates), 5)],
                   rotation=30, ha="right", fontsize=7)
ax.set_ylabel("Cumulative return")
ax.set_title("Cumulative Return Over Time", fontsize=10)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# Daily position bar chart (Bayesian)
ax = axes[1]
pos_colors = ["#2ecc71" if p > 0 else "#e74c3c" if p < 0 else "#cccccc"
              for p in bayes_pos]
ax.bar(t, bayes_pos, color=pos_colors, edgecolor="white", linewidth=0.5)
ax.axhline(0, color="#333333", linewidth=0.8)
ax.set_xticks(list(t)[::5])
ax.set_xticklabels([dates[i] for i in range(0, len(dates), 5)],
                   rotation=30, ha="right", fontsize=7)
ax.set_ylabel("Position size")
ax.set_title("Bayesian Daily Positions\n(green=long, red=short, grey=abstain)",
             fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs",
                           "trading_simulation.png")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved to {OUTPUT_PATH}")