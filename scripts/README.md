# Scripts

All scripts run from the project root after `pip install -e .`. No API key needed.

## Start here

**`demo_real_articles.py`** - runs the full pipeline on 10 realistic headlines with simulated LLM noise. Good first thing to run to see the system working end-to-end.

**`visualize_convergence.py`** - shows how the posterior narrows as sample size grows. The core statistical property of the model in one plot.

## Understanding the model

**`visualize_uncertainty.py`** - same n, different spread. Demonstrates that interval width comes from disagreement between sources, not from sample size alone.

**`visualize_failure_modes.py`** - three cases where the Gaussian assumption breaks: bimodal data, outliers, and single observations. The model is technically correct in each case but the output can be misleading.

## Testing the model's claims

**`calibration_test.py`** - checks whether the 95% credible interval actually contains the true value ~95% of the time. Three boundary-exposure regimes compared. The gap between nominal and empirical coverage is quantified here.

**`adversarial_tests.py`** - six conditions that violate model assumptions: mixture noise, systematic bias, heavy tails, regime shifts, correlated observations. This is where the model actually breaks.

**`variance_estimator_comparison.py`** - compares raw empirical Bayes, floored estimator, and hierarchical pooling across n ∈ {2, 5, 10, 20}. Shows why the floor matters at small n.

**`truncated_normal_comparison.py`** - tests the theoretically correct fix for bounded data. Result: no meaningful improvement over the floored Gaussian. Explains why.

## Comparing against baselines

**`baseline_comparison.py`** - Bayesian model vs naive mean + bootstrap CI. Coverage, width, and point error across 2000 runs.

**`sensitivity_analysis.py`** - coverage and width across a grid of τ² ∈ {0.25, 1.0, 4.0}, σ ∈ {0.2, 0.4, 0.6}, n ∈ {5, 10, 20}. Shows the prior choice isn't load-bearing.

**`backtest_synthetic_market.py`** - AR(1) time series backtest. Four strategies compared on directional accuracy predicting sign(μ_{t+1} - μ_t).

## Decision making

**`decision_simulation.py`** - accuracy when the model abstains on uncertain cases vs always committing.

**`decision_comparison.py`** - full decision quality table: accuracy when acting, overall accuracy, false positive rate, coverage.

## Real-world

**`real_world_validation.py`** - 30 trading days, Jan–Feb 2025, real SPY returns. Scores hand-assigned (see caveat in script). Proof-of-concept, not a backtest.

**`trading_simulation.py`** - simple long/short strategy on the same 30-day dataset. Cumulative return, Sharpe, max drawdown. Same caveats apply.
