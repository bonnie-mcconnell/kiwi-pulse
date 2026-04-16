"""
core/bayesian_model.py

Bayesian inference over noisy sentiment observations.

Model
-----
We treat each LLM-produced sentiment score as a noisy measurement
of a true latent market sentiment μ:

    x_i ~ Normal(μ, σ²)       likelihood
    μ   ~ Normal(0, 1)         prior  (neutral, weakly informative)

σ² is estimated from data using a floored empirical Bayes estimator:

    σ² = max(Var(x_1, ..., x_n), VAR_FLOOR)

The floor encodes a domain assumption: LLM scores always carry at
least some irreducible noise. Without it, small samples with low
observed variance produce falsely tight intervals - calibration
testing showed 69.6% coverage at n=2 with raw sample variance,
vs 92.6% with the floor applied (target: 95%).

The Normal-Normal conjugate gives a closed-form posterior:

    τ_n² = 1 / (1/τ² + n/σ²)
    μ_n  = τ_n² * (μ₀/τ² + Σx_i/σ²)

We then read off a 95% credible interval as:

    μ_n ± 1.96 * sqrt(τ_n²)

Why not just take the sample mean?
  - The sample mean has no uncertainty quantification.
  - It ignores prior knowledge and behaves poorly for small n.
  - As n → ∞ and τ² → ∞, the posterior mean converges to the
    sample mean - the Bayesian approach strictly generalises it.

Known limitations
-----------------
  - Gaussian likelihood: breaks under systematic bias or correlated
    observations (see scripts/adversarial_tests.py for quantification).
  - Bounded domain: asymmetric clamping near [-1, 1] causes mild
    undercoverage (~92% vs nominal 95% on full domain).
  - VAR_FLOOR is a domain assumption, not estimated from data.
    A hierarchical estimator would be more principled for production.
"""

import logging

import numpy as np

from src.schema.models import MarketEstimate

logger = logging.getLogger(__name__)

# Prior hyperparameters - kept as module-level constants so they are
# easy to find and justify in a code review or interview.
PRIOR_MEAN: float = 0.0   # neutral sentiment a priori
PRIOR_VAR: float  = 1.0   # weakly informative over [-1, 1]

# Minimum observation variance floor.
# Encodes the assumption that LLM scores always have irreducible noise.
# Calibration analysis: raw sample variance gives 69.6% coverage at n=2;
# this floor brings it to 92.6%. Value corresponds to std ≈ 0.316.
VAR_FLOOR: float = 0.10

# Fallback variance when n == 1. We have no information about spread
# from a single observation, so we assume maximum plausible noise.
_SINGLE_OBS_VAR: float = 1.0

# Substituted when all scores are identical (zero empirical variance)
# and below the floor. Near-zero rather than zero prevents division by zero.
EPSILON_ZERO_VAR: float = 1e-9


def estimate_market(sentiments: list[float]) -> MarketEstimate:
    """
    Run Bayesian update and return a posterior market estimate.

    Parameters
    ----------
    sentiments:
        List of sentiment scores in [-1, 1], one per article.
        Produced by the LLM scoring module.

    Returns
    -------
    MarketEstimate
        Posterior mean, 95% credible interval, variance, sample size.

    Raises
    ------
    ValueError
        If the list is empty or any score is outside [-1, 1].
    """
    _validate(sentiments)

    x = np.array(sentiments, dtype=float)
    n = len(x)

    # --- floored empirical Bayes: estimate observation noise ---------
    if n == 1:
        # With one point we cannot estimate spread.
        # Fall back to maximum plausible variance so the posterior
        # stays close to the prior rather than over-committing.
        observation_var = _SINGLE_OBS_VAR
    else:
        raw_var = float(np.var(x, ddof=1))   # unbiased sample variance

        if raw_var == 0.0:
            # All scores identical - genuine signal, near-zero noise.
            # Floor still applies: we don't trust perfect agreement.
            observation_var = VAR_FLOOR
        else:
            # Apply floor. Prevents falsely tight intervals when small
            # samples happen to show low spread by chance.
            observation_var = max(raw_var, VAR_FLOOR)

    # --- Normal-Normal conjugate update ------------------------------
    # Using precision (1/variance) form for numerical stability and clarity
    prior_precision = 1.0 / PRIOR_VAR
    data_precision  = n   / observation_var

    posterior_var  = 1.0 / (prior_precision + data_precision)
    x_sum = float(x.sum())
    posterior_mean = posterior_var * (
        PRIOR_MEAN * prior_precision + x_sum / observation_var
    )

    # --- 95% credible interval ---------------------------------------
    margin = 1.96 * np.sqrt(posterior_var)
    lower  = posterior_mean - margin
    upper  = posterior_mean + margin

    # The domain of individual observations is [-1, 1], but the posterior
    # interval can legitimately exceed this with strong or sparse data.
    # We soft-clamp and log a warning so callers are aware rather than
    # silently truncating valid uncertainty information.
    if lower < -1.0 or upper > 1.0:
        logger.warning(
            "Credible interval [%.3f, %.3f] extends outside [-1, 1]. "
            "This reflects genuine model uncertainty, not a bug.",
            lower, upper,
        )
        lower = max(-1.0, lower)
        upper = min(1.0, upper)
        posterior_mean = max(-1.0, min(1.0, posterior_mean))

    return MarketEstimate(
        mean=round(posterior_mean, 6),
        lower_bound=round(float(lower), 6),
        upper_bound=round(float(upper), 6),
        variance=round(posterior_var, 6),
        sample_size=n,
    )


# ----- helpers -------------------------------------------------------

def _validate(sentiments: list[float]) -> None:
    if not sentiments:
        raise ValueError("sentiments list must not be empty")
    out_of_range = [s for s in sentiments if not -1.0 <= s <= 1.0]
    if out_of_range:
        raise ValueError(f"scores must be in [-1, 1]; got {out_of_range}")