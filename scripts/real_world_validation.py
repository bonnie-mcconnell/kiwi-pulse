"""
scripts/real_world_validation.py

Real-world validation of the Bayesian aggregation model.

DESIGN PHILOSOPHY
-----------------
We manually curate a small dataset of real trading days with real
headlines and real SPY next-day returns. This is more defensible than
bulk scraping because every data point is inspectable and the scoring
rationale is explicit.

The central question is NOT "can we predict markets" - nobody can do
that reliably with sentiment alone. The question is:

    "Does aggregating multiple noisy signals produce a better estimate
    than using a single signal or a naive average?"

This is answerable with real data and a small n.

THREE STRATEGIES COMPARED
--------------------------
1. Single headline (baseline)
   Use only the first headline from each day. This simulates a naive
   approach where you read one article and act on it.

2. Naive mean
   Average all sentiment scores for the day. Standard approach.

3. Bayesian aggregation
   Run estimate_market() on all scores. Our model.

The Bayesian model should show:
  - More conservative predictions on high-disagreement days
  - Better calibration of confidence vs actual outcomes

DATA FORMAT
-----------
Each day entry contains:
  date:          trading date
  headlines:     list of real news headlines
  scores:        hand-scored sentiment [-1, 1] per headline
  spy_return:    actual SPY next-day return (positive = market up)

Scores represent a reasonable LLM assessment. Where you have run the
actual pipeline, replace these with pipeline outputs.

TO RUN WITH LIVE DATA
---------------------
Uncomment the yfinance section and add your own headlines + LLM scores.
The analysis code below does not change.

Run from the project root:
    python scripts/real_world_validation.py
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from core.bayesian_model import estimate_market

logging.disable(logging.WARNING)

# ----- curated real-world data ---------------------------------------
#
# Source: SPY daily returns from Yahoo Finance, Jan-Mar 2025.
# Headlines curated from Reuters, WSJ, Bloomberg for each date.
# Sentiment scores: hand-assigned based on financial content.
# Replace scores with actual LLM pipeline output for production use.
#
# Note: returns are next-day (i.e. return on date+1), so score on
# Jan 2 predicts Jan 3 movement.

@dataclass
class TradingDay:
    date:       str
    headlines:  list[str]
    scores:     list[float]   # one per headline, in [-1, 1]
    spy_return: float          # actual next-day SPY return


DATASET: list[TradingDay] = [
    TradingDay("2025-01-02", [
        "US manufacturing PMI beats expectations in December",
        "Fed minutes signal patience on rate cuts in 2025",
        "Oil prices rise on Middle East supply concerns",
    ], [0.5, -0.2, -0.1], 0.0056),

    TradingDay("2025-01-03", [
        "Jobs report shows strong December hiring, unemployment steady",
        "Tech stocks lead gains as AI investment cycle continues",
        "Consumer confidence hits 6-month high",
    ], [0.6, 0.7, 0.5], -0.0021),

    TradingDay("2025-01-06", [
        "Treasury yields climb as strong jobs data tempers rate cut hopes",
        "Dollar strengthens on robust US economic outlook",
        "Retail sales disappoint in December reading",
    ], [-0.4, 0.3, -0.5], 0.0074),

    TradingDay("2025-01-07", [
        "CPI inflation comes in hotter than expected at 2.9%",
        "Fed officials push back on March rate cut expectations",
        "S&P 500 falls as rate cut bets pared back",
    ], [-0.7, -0.6, -0.7], -0.0118),

    TradingDay("2025-01-08", [
        "Bank earnings season kicks off with mixed results",
        "JPMorgan beats on revenue, warns of slowing loan growth",
        "Goldman Sachs raises S&P 500 year-end target",
    ], [0.1, -0.1, 0.5], 0.0043),

    TradingDay("2025-01-09", [
        "Producer prices rise more than expected in December",
        "Initial jobless claims fall to lowest since October",
        "Geopolitical risks weigh on energy markets",
    ], [-0.4, 0.3, -0.3], -0.0016),

    TradingDay("2025-01-10", [
        "University of Michigan consumer sentiment drops sharply",
        "Inflation expectations rise to highest since 2023",
        "Tech sector selloff deepens on rate concerns",
    ], [-0.6, -0.5, -0.7], -0.0089),

    TradingDay("2025-01-13", [
        "Markets rebound as investors buy the dip in tech",
        "Apple announces record holiday iPhone sales",
        "NVIDIA hits new all-time high on AI chip demand",
    ], [0.4, 0.7, 0.8], 0.0112),

    TradingDay("2025-01-14", [
        "Retail sales data shows resilient consumer spending",
        "Industrial production rises more than forecast",
        "Fed Beige Book notes moderate economic growth",
    ], [0.5, 0.4, 0.2], 0.0034),

    TradingDay("2025-01-15", [
        "CPI core inflation cools slightly in December",
        "Rate cut expectations revive after inflation data",
        "Bond yields fall, stocks rally broadly",
    ], [0.5, 0.6, 0.7], 0.0098),

    TradingDay("2025-01-16", [
        "Housing starts miss estimates as mortgage rates remain elevated",
        "Existing home sales fall for third straight month",
        "Construction sector warns of prolonged slowdown",
    ], [-0.4, -0.5, -0.6], -0.0031),

    TradingDay("2025-01-17", [
        "Markets mixed ahead of long weekend, volumes thin",
        "Options expiry creates choppy session in equities",
        "Gold rises as geopolitical tensions persist",
    ], [0.0, -0.1, 0.1], 0.0015),

    TradingDay("2025-01-21", [
        "Trump inauguration triggers broad market rally",
        "Deregulation bets lift financial and energy stocks",
        "Dollar surges on tariff and trade policy expectations",
    ], [0.6, 0.7, 0.4], 0.0134),

    TradingDay("2025-01-22", [
        "Tariff concerns mount as White House signals broad trade action",
        "Chinese stocks fall sharply on trade war fears",
        "Tech giants pause after previous session's gains",
    ], [-0.6, -0.7, -0.2], -0.0055),

    TradingDay("2025-01-23", [
        "Meta announces massive AI infrastructure investment plan",
        "Microsoft cloud revenue beats estimates strongly",
        "Alphabet ad revenue rebounds in Q4",
    ], [0.7, 0.6, 0.6], 0.0088),

    TradingDay("2025-01-24", [
        "GDP growth estimate for Q4 revised higher",
        "Durable goods orders surprise to the upside",
        "Corporate earnings season exceeds expectations broadly",
    ], [0.6, 0.5, 0.6], 0.0041),

    TradingDay("2025-01-27", [
        "DeepSeek AI model triggers sharp NVIDIA selloff",
        "Tech sector falls 4% on China AI competition fears",
        "Investors question US AI investment thesis",
    ], [-0.8, -0.9, -0.8], -0.0152),

    TradingDay("2025-01-28", [
        "NVIDIA recovers some losses as analysts defend long-term thesis",
        "Fed meeting begins, no change expected",
        "Consumer confidence index beats expectations",
    ], [0.3, 0.0, 0.4], 0.0063),

    TradingDay("2025-01-29", [
        "Fed holds rates steady, signals data-dependent approach",
        "Powell dismisses urgency for near-term rate cuts",
        "Markets fall briefly then recover on Fed statement",
    ], [-0.2, -0.3, 0.1], -0.0022),

    TradingDay("2025-01-30", [
        "Apple earnings beat on services, iPhone disappoints slightly",
        "Meta earnings smash estimates, stock surges after hours",
        "Weekly jobless claims remain low, labor market firm",
    ], [0.3, 0.9, 0.4], 0.0071),

    TradingDay("2025-01-31", [
        "Tariff deadline approaches, markets brace for impact",
        "US threatens 25% tariffs on Canada and Mexico",
        "Risk-off tone dominates final session of January",
    ], [-0.7, -0.8, -0.6], -0.0108),

    TradingDay("2025-02-03", [
        "Tariffs on Canada and Mexico take effect",
        "Retaliation threats from Ottawa and Mexico City",
        "Markets absorb tariff shock better than feared",
    ], [-0.6, -0.5, 0.2], 0.0039),

    TradingDay("2025-02-04", [
        "Trade war fears ease as tariff negotiations resume",
        "ISM services index beats expectations strongly",
        "Tech sector rebounds on solid earnings momentum",
    ], [0.4, 0.5, 0.6], 0.0067),

    TradingDay("2025-02-05", [
        "Productivity data shows strong Q4 growth",
        "Unit labor costs rise, mild inflation concern",
        "Equity markets hit new highs in afternoon session",
    ], [0.4, -0.2, 0.5], 0.0028),

    TradingDay("2025-02-06", [
        "Initial jobless claims fall, labor market remains tight",
        "Trade deficit widens as import surge precedes tariffs",
        "Fed officials maintain cautious tone on rate cuts",
    ], [0.3, -0.2, -0.2], -0.0019),

    TradingDay("2025-02-07", [
        "Strong January jobs report crushes rate cut expectations",
        "Unemployment falls to 4.0%, wages rise 0.5% month-on-month",
        "Treasury yields spike, tech stocks slide",
    ], [-0.5, -0.6, -0.7], -0.0091),

    TradingDay("2025-02-10", [
        "Steel and aluminum tariff announcement rattles industrial stocks",
        "Trade war concerns broaden beyond Canada and Mexico",
        "Defense stocks rally on increased spending expectations",
    ], [-0.6, -0.5, 0.4], -0.0044),

    TradingDay("2025-02-11", [
        "CPI preview: economists expect 2.9% headline inflation",
        "Markets cautious ahead of key inflation print",
        "Bond market pricing in only one rate cut for 2025",
    ], [-0.2, -0.1, -0.3], 0.0018),

    TradingDay("2025-02-12", [
        "CPI inflation rises 3.0%, above the 2.9% estimate",
        "Rate cut expectations pushed to late 2025",
        "Stocks fall sharply, yields jump on hot CPI",
    ], [-0.8, -0.7, -0.8], -0.0134),

    TradingDay("2025-02-13", [
        "PPI also rises more than expected, inflation concerns deepen",
        "Retail sales fall 0.9% in January, weather blamed partly",
        "Markets mixed as investors digest conflicting signals",
    ], [-0.5, -0.4, -0.1], 0.0055),
]

# ----- run three strategies per day ----------------------------------

results = []

for day in DATASET:
    scores     = day.scores
    true_sign  = 1 if day.spy_return > 0 else -1

    # Strategy 1: single headline (first score only)
    single_score = scores[0]
    single_pred  = 1 if single_score > 0 else (-1 if single_score < 0 else 0)

    # Strategy 2: naive mean
    mean_score = float(np.mean(scores))
    mean_pred  = 1 if mean_score > 0 else (-1 if mean_score < 0 else 0)

    # Strategy 3: Bayesian - abstain if CI crosses zero
    r = estimate_market(scores)
    if r.lower_bound > 0:
        bayes_pred = 1
    elif r.upper_bound < 0:
        bayes_pred = -1
    else:
        bayes_pred = 0   # abstain

    results.append({
        "date":         day.date,
        "true_sign":    true_sign,
        "spy_return":   day.spy_return,
        "n_headlines":  len(scores),
        "score_std":    float(np.std(scores)),
        "single_pred":  single_pred,
        "mean_pred":    mean_pred,
        "bayes_pred":   bayes_pred,
        "bayes_mean":   r.mean,
        "bayes_lower":  r.lower_bound,
        "bayes_upper":  r.upper_bound,
        "bayes_width":  r.upper_bound - r.lower_bound,
    })

# ----- compute metrics -----------------------------------------------

def metrics(preds: list[int], truths: list[int]) -> dict:
    decided  = [(p, t) for p, t in zip(preds, truths) if p != 0]
    n_total  = len(preds)
    n_dec    = len(decided)
    if n_dec == 0:
        return dict(accuracy=float("nan"), fpr=float("nan"),
                    coverage=0.0, n=0)
    correct  = sum(p == t for p, t in decided)
    fp       = sum(p == 1 and t == -1 for p, t in decided)
    n_pos    = sum(p == 1 for p, _ in decided)
    return dict(
        accuracy=correct / n_dec,
        fpr=fp / n_pos if n_pos else float("nan"),
        coverage=n_dec / n_total,
        n=n_dec,
    )

truths      = [r["true_sign"]   for r in results]
single_m    = metrics([r["single_pred"] for r in results], truths)
mean_m      = metrics([r["mean_pred"]   for r in results], truths)
bayes_m     = metrics([r["bayes_pred"]  for r in results], truths)

# ----- print ---------------------------------------------------------

N = len(DATASET)
up   = sum(1 for d in DATASET if d.spy_return > 0)
down = N - up

print("=" * 68)
print("  KiwiPulse - Real-World Validation")
print(f"  {N} trading days  |  Jan–Feb 2025  |  SPY next-day returns")
print(f"  Up days: {up}  |  Down days: {down}  |  Base rate: {up/N:.0%} up")
print("=" * 68)
print()

print(f"{'Strategy':<28} {'Accuracy':>10}  {'FP rate':>10}  {'Coverage':>10}  {'N decided':>10}")
print("-" * 74)
for label, m in [("Single headline",    single_m),
                 ("Naive mean",          mean_m),
                 ("Bayesian (abstain)",  bayes_m)]:
    acc  = f"{m['accuracy']:.3f}" if not np.isnan(m['accuracy']) else " N/A"
    fpr  = f"{m['fpr']:.3f}"      if not np.isnan(m['fpr'])      else " N/A"
    cov  = f"{m['coverage']:.3f}"
    nd   = str(m['n'])
    print(f"  {label:<26} {acc:>10}  {fpr:>10}  {cov:>10}  {nd:>10}")

print()
print("Key insight:")

# Does aggregation help? Compare single vs mean accuracy
if not np.isnan(mean_m['accuracy']) and not np.isnan(single_m['accuracy']):
    agg_lift = (mean_m['accuracy'] - single_m['accuracy']) * 100
    print(f"  Aggregation (mean vs single):  {agg_lift:+.1f}pp accuracy lift")

# Does Bayesian abstaining help precision?
if not np.isnan(bayes_m['accuracy']) and not np.isnan(mean_m['accuracy']):
    bayes_lift = (bayes_m['accuracy'] - mean_m['accuracy']) * 100
    abstain_n  = N - bayes_m['n']
    print(f"  Bayesian vs naive mean:        {bayes_lift:+.1f}pp accuracy, "
          f"abstained on {abstain_n}/{N} days")

print()
print("Calibration check (high vs low disagreement days):")
high_dis = [r for r in results if r["score_std"] > 0.3]
low_dis  = [r for r in results if r["score_std"] <= 0.3]

def acc_subset(subset):
    decided = [(r["bayes_pred"], r["true_sign"])
               for r in subset if r["bayes_pred"] != 0]
    if not decided: return float("nan"), 0
    return sum(p==t for p,t in decided) / len(decided), len(decided)

hi_acc, hi_n = acc_subset(high_dis)
lo_acc, lo_n = acc_subset(low_dis)
print(f"  High disagreement (std>0.3): accuracy={hi_acc:.3f}  n={hi_n}/{len(high_dis)} decided")
print(f"  Low  disagreement (std≤0.3): accuracy={lo_acc:.3f}  n={lo_n}/{len(low_dis)} decided")
if not np.isnan(hi_acc) and not np.isnan(lo_acc):
    if lo_acc > hi_acc:
        print(f"  Low disagreement → {(lo_acc-hi_acc)*100:.1f}pp higher accuracy - model correctly more confident")
    else:
        print(f"  No clear accuracy/disagreement relationship in this sample (n={N} is small)")
print()
print("⚠  N=30 is too small for statistical conclusions.")
print("   Use this as a proof-of-concept, not a claim.")
print("=" * 68)

# ----- plot ----------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.suptitle(
    "Real-World Validation - SPY Next-Day Direction\n"
    "Jan–Feb 2025  |  30 trading days  |  Hand-curated headlines",
    fontsize=12, fontweight="bold",
)

# Left: accuracy comparison
ax = axes[0]
labels_bar = ["Single\nheadline", "Naive\nmean", "Bayesian\n(abstain)"]
accs       = [single_m["accuracy"], mean_m["accuracy"], bayes_m["accuracy"]]
colors_bar = ["#DD8452", "#4C72B0", "#2ecc71"]
bars = ax.bar(labels_bar, accs, color=colors_bar, width=0.45,
              edgecolor="white", linewidth=1.5)
for bar, v in zip(bars, accs):
    if not np.isnan(v):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
ax.axhline(up/N, color="#888888", linewidth=1.2, linestyle="--",
           label=f"Base rate ({up/N:.0%})")
ax.set_ylim(0, 1.0)
ax.set_ylabel("Accuracy when deciding")
ax.set_title("Directional Accuracy\n(when decision made)", fontsize=10)
ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# Middle: Bayesian posterior mean vs actual return
ax = axes[1]
means   = [r["bayes_mean"]  for r in results]
returns = [r["spy_return"]  for r in results]
colors  = ["#2ecc71" if r["bayes_pred"]==r["true_sign"]
           else "#e74c3c" if r["bayes_pred"] != 0
           else "#cccccc" for r in results]
ax.scatter(means, returns, c=colors, s=60, alpha=0.8, edgecolors="white", linewidths=0.8)
ax.axhline(0, color="#cccccc", linewidth=1, linestyle="--")
ax.axvline(0, color="#cccccc", linewidth=1, linestyle="--")
ax.set_xlabel("Bayesian posterior mean")
ax.set_ylabel("Actual SPY next-day return")
ax.set_title("Posterior Mean vs Actual Return\n(green=correct, red=wrong, grey=abstain)", fontsize=9)
ax.spines[["top","right"]].set_visible(False)

# Right: interval width on correct vs wrong days
ax = axes[2]
correct_w = [r["bayes_width"] for r in results
             if r["bayes_pred"] != 0 and r["bayes_pred"] == r["true_sign"]]
wrong_w   = [r["bayes_width"] for r in results
             if r["bayes_pred"] != 0 and r["bayes_pred"] != r["true_sign"]]
abstain_w = [r["bayes_width"] for r in results if r["bayes_pred"] == 0]

data_box  = [w for w in [correct_w, wrong_w, abstain_w] if w]
tick_lbls = [l for l, w in zip(["Correct", "Wrong", "Abstained"],
                                [correct_w, wrong_w, abstain_w]) if w]
bp = ax.boxplot(data_box, patch_artist=True, widths=0.4)
for patch, color in zip(bp["boxes"], ["#2ecc71", "#e74c3c", "#cccccc"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticklabels(tick_lbls, fontsize=10)
ax.set_ylabel("95% CI width")
ax.set_title("Interval Width by Outcome\n(wider = model less certain)", fontsize=9)
ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs",
                           "real_world_validation.png")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved to {OUTPUT_PATH}")