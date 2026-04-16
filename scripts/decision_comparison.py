"""
scripts/decision_comparison.py

Directly answers: "Does Bayesian uncertainty improve decisions?"

Two strategies are compared on identical simulated data:

  Bayesian (abstain rule)
      If the 95% credible interval excludes zero, commit to the
      sign of the posterior mean. Otherwise abstain.

  Naive mean (always decides)
      Always commit to the sign of the sample mean. No abstention.

The Bayesian strategy trades coverage for precision - it decides
less often but should be right more often when it does decide.
Whether that tradeoff is worth it depends on the cost structure
of the application. This script measures both sides of the tradeoff.

METRICS
-------
  Accuracy when acting    - of decisions made, what fraction were correct?
  Overall accuracy        - abstains counted as incorrect (harshest comparison)
  False positive rate     - predicted positive when true was negative
  False negative rate     - predicted negative when true was positive
  Decision coverage       - fraction of cases where a decision was made

Run from the project root:
    python scripts/decision_comparison.py
"""

import logging
import numpy as np

from core.bayesian_model import estimate_market

logging.disable(logging.WARNING)

# ----- parameters ----------------------------------------------------

N_RUNS    = 5000
N_OBS     = 10
NOISE_STD = 0.4
MU_LO     = -0.6    # interior domain - fair to both methods
MU_HI     =  0.6
SEED      = 13

# ----- simulation ----------------------------------------------------

rng = np.random.default_rng(SEED)

# Accumulators
b_decided_correct   = 0
b_decided_wrong     = 0
b_abstained         = 0
b_false_pos         = 0
b_false_neg         = 0

n_correct           = 0
n_wrong             = 0
n_false_pos         = 0
n_false_neg         = 0

for _ in range(N_RUNS):
    true_mu  = rng.uniform(MU_LO, MU_HI)
    true_dir = 1 if true_mu > 0 else -1

    raw    = rng.normal(loc=true_mu, scale=NOISE_STD, size=N_OBS)
    scores = np.clip(raw, -1.0, 1.0).tolist()
    x      = np.array(scores)

    # --- Bayesian decision -------------------------------------------
    result = estimate_market(scores)

    if result.lower_bound > 0:
        b_pred = 1
    elif result.upper_bound < 0:
        b_pred = -1
    else:
        b_pred = 0    # abstain

    if b_pred == 0:
        b_abstained += 1
    elif b_pred == true_dir:
        b_decided_correct += 1
    else:
        b_decided_wrong += 1
        if b_pred == 1:
            b_false_pos += 1
        else:
            b_false_neg += 1

    # --- Naive decision ----------------------------------------------
    n_pred = 1 if float(x.mean()) > 0 else -1

    if n_pred == true_dir:
        n_correct += 1
    else:
        n_wrong += 1
        if n_pred == 1:
            n_false_pos += 1
        else:
            n_false_neg += 1

# ----- compute metrics -----------------------------------------------

b_decided = b_decided_correct + b_decided_wrong

b_acc_when_acting = b_decided_correct / b_decided if b_decided else float("nan")
b_acc_overall     = b_decided_correct / N_RUNS   # abstains = wrong
b_fpr             = b_false_pos / b_decided      if b_decided else float("nan")
b_fnr             = b_false_neg / b_decided      if b_decided else float("nan")
b_coverage        = b_decided / N_RUNS

n_decided         = n_correct + n_wrong
n_acc_when_acting = n_correct / N_RUNS           # never abstains, so same
n_acc_overall     = n_correct / N_RUNS
n_fpr             = n_false_pos / N_RUNS
n_fnr             = n_false_neg / N_RUNS
n_coverage        = 1.0

# ----- print ---------------------------------------------------------

W1, W2 = 34, 18

print(f"Runs: {N_RUNS}  |  n={N_OBS} obs/run  |  noise σ={NOISE_STD}  |  μ ∈ [{MU_LO}, {MU_HI}]")
print()
print(f"{'Metric':<{W1}} {'Bayesian (abstain)':>{W2}}  {'Naive mean':>{W2}}")
print("-" * (W1 + W2 * 2 + 4))
print(f"{'Accuracy when acting':<{W1}} {b_acc_when_acting:>{W2}.3f}  {n_acc_when_acting:>{W2}.3f}")
print(f"{'Overall accuracy (abstain=wrong)':<{W1}} {b_acc_overall:>{W2}.3f}  {n_acc_overall:>{W2}.3f}")
print(f"{'False positive rate':<{W1}} {b_fpr:>{W2}.3f}  {n_fpr:>{W2}.3f}")
print(f"{'False negative rate':<{W1}} {b_fnr:>{W2}.3f}  {n_fnr:>{W2}.3f}")
print(f"{'Decision coverage':<{W1}} {b_coverage:>{W2}.3f}  {n_coverage:>{W2}.3f}")
print(f"{'Decisions made':<{W1}} {b_decided:>{W2}}  {N_RUNS:>{W2}}")
print(f"{'Abstentions':<{W1}} {b_abstained:>{W2}}  {'0':>{W2}}")
print()

# Plain-English verdict
print("Verdict:")

if b_acc_when_acting > n_acc_when_acting + 0.01:
    print(f"  When the Bayesian model acts, it is more accurate by "
          f"{(b_acc_when_acting - n_acc_when_acting)*100:.1f}pp.")
else:
    print(f"  Accuracy when acting is similar for both methods.")

if b_fpr < n_fpr - 0.01:
    print(f"  False positive rate is {(n_fpr - b_fpr)*100:.1f}pp lower - "
          f"the abstain rule filters out most uncertain positive calls.")
else:
    print(f"  False positive rates are similar.")

print(f"  Cost: the Bayesian model abstains on "
      f"{b_abstained} / {N_RUNS} cases ({(1 - b_coverage)*100:.1f}%).")

if b_acc_overall > n_acc_overall:
    print(f"  Overall accuracy (abstains penalised) is still "
          f"{(b_acc_overall - n_acc_overall)*100:.1f}pp higher.")
elif b_acc_overall < n_acc_overall - 0.005:
    print(f"  Overall accuracy (abstains penalised) is "
          f"{(n_acc_overall - b_acc_overall)*100:.1f}pp lower - "
          f"abstaining has a net cost at this sample size.")
else:
    print(f"  Overall accuracy (abstains penalised) is roughly equal.")

print()
print("Answer: does Bayesian uncertainty improve decisions?")
if b_acc_when_acting > n_acc_when_acting and b_fpr < n_fpr:
    print("  Yes - higher precision and lower false positive rate when acting,")
    print("  at the cost of lower coverage. The tradeoff is explicit and tunable.")
else:
    print("  Mixed - see metrics above. Benefit depends on cost structure.")