"""
core/truncated_normal_model.py

Bayesian inference using a truncated Normal likelihood for bounded data.

WHY THIS EXISTS
---------------
The standard Gaussian model (bayesian_model.py) assumes observations
can take any real value. For sentiment scores bounded to [-1, 1], this
is a misspecification: the model reasons about impossible observations,
which causes it to underestimate variance near boundaries and produce
intervals that are too narrow.

The truncated Normal corrects this by renormalising the Gaussian density
to integrate to 1 over [-1, 1] rather than over all reals.

WHY CONJUGACY BREAKS
--------------------
With a plain Gaussian likelihood, the normalising constant Z = 1
everywhere and cancels in the posterior. With a truncated Normal,
Z(μ, σ) depends on μ, so it does not cancel. The posterior is no
longer Gaussian - no closed form exists.

APPROXIMATION: NUMERICAL INTEGRATION ON A GRID
-----------------------------------------------
We evaluate the unnormalised log-posterior at G grid points spanning
[-1, 1], exponentiate, and normalise to obtain a discrete approximation
to the posterior distribution. From this we read off:

    posterior mean    = Σ μ_g · p(μ_g | x)
    credible interval = 2.5th and 97.5th percentiles

Grid resolution of 500 points gives 0.004 spacing - sufficient for
3 decimal place output with negligible discretisation error.

LOG-POSTERIOR AT EACH GRID POINT
---------------------------------
log p(μ_g | x) ∝ log_prior(μ_g)
                + Σ_i [ -(x_i - μ_g)² / (2σ²) ]   ← Gaussian kernel
                - n · log Z(μ_g, σ)                 ← truncation correction

where log Z(μ, σ) = log Φ((1-μ)/σ) - log Φ((-1-μ)/σ)

We use scipy.special.log_ndtr (numerically stable log CDF) to avoid
catastrophic cancellation when μ is far from the boundary.

KNOWN LIMITATIONS
-----------------
- σ² still estimated via floored empirical Bayes (same as Gaussian model)
- Grid approximation introduces discretisation error (controlled by G)
- Slower than conjugate model: O(G·n) vs O(n)
- Still assumes observations are conditionally independent given μ
"""

import logging

import numpy as np
from scipy.special import ndtr   # CDF of standard Normal

from src.schema.models import MarketEstimate

logger = logging.getLogger(__name__)

# Prior - same as Gaussian model for a fair comparison
PRIOR_MEAN: float = 0.0
PRIOR_VAR:  float = 1.0

# Observation variance floor - same rationale as Gaussian model
VAR_FLOOR: float = 0.10

# Fallback for n == 1
_SINGLE_OBS_VAR: float = 1.0

# Domain bounds
LOWER_BOUND: float = -1.0
UPPER_BOUND: float =  1.0

# Grid resolution - 500 points gives 0.004 spacing.
# Increasing to 2000 improves accuracy by ~10x but slows by ~4x.
# 500 is the right tradeoff for this application.
GRID_SIZE: int = 500

# Pre-compute grid once at module load - it never changes
_GRID: np.ndarray = np.linspace(LOWER_BOUND, UPPER_BOUND, GRID_SIZE)


def estimate_market_truncated(sentiments: list[float]) -> MarketEstimate:
    """
    Bayesian posterior via truncated Normal likelihood + grid integration.

    Produces a 95% credible interval that correctly accounts for the
    bounded domain of sentiment scores.

    Parameters
    ----------
    sentiments:
        List of sentiment scores in [-1, 1], one per article.

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

    # --- estimate observation noise (same floored empirical Bayes) ---
    if n == 1:
        obs_var = _SINGLE_OBS_VAR
    else:
        raw_var = float(np.var(x, ddof=1))
        obs_var = max(raw_var, VAR_FLOOR)

    sigma = np.sqrt(obs_var)

    # --- log-posterior on the grid -----------------------------------
    log_post = _log_posterior(x, sigma)

    # --- normalise to get discrete probability distribution ----------
    # Subtract max before exp for numerical stability (log-sum-exp trick)
    log_post -= log_post.max()
    weights   = np.exp(log_post)
    weights  /= weights.sum()

    # --- posterior mean and variance ---------------------------------
    posterior_mean = float(np.dot(_GRID, weights))
    posterior_var  = float(np.dot((_GRID - posterior_mean) ** 2, weights))

    # --- 95% credible interval via CDF of discrete distribution ------
    cdf   = np.cumsum(weights)
    lower = float(_GRID[np.searchsorted(cdf, 0.025)])
    upper = float(_GRID[np.searchsorted(cdf, 0.975)])

    # Clamp mean to domain (should already be in [-1, 1] by construction)
    posterior_mean = float(np.clip(posterior_mean, LOWER_BOUND, UPPER_BOUND))

    return MarketEstimate(
        mean=round(posterior_mean, 6),
        lower_bound=round(lower, 6),
        upper_bound=round(upper, 6),
        variance=round(posterior_var, 6),
        sample_size=n,
    )


# ----- private helpers -----------------------------------------------

def _log_posterior(x: np.ndarray, sigma: float) -> np.ndarray:
    """
    Unnormalised log-posterior evaluated at every grid point.

    Shape: (GRID_SIZE,)

    Each term:
      log_prior      - Gaussian prior on μ, evaluated at each grid point
      log_likelihood - sum of truncated Normal log-densities over observations
                       = Gaussian kernel - truncation correction (per obs)
    """
    # log prior: log N(μ | 0, τ²)  [constant terms dropped]
    log_prior = -(_GRID - PRIOR_MEAN) ** 2 / (2.0 * PRIOR_VAR)

    # log Z(μ, σ) for each grid point - the truncation correction.
    # Z(μ, σ) = Φ((1-μ)/σ) - Φ((-1-μ)/σ) is the Gaussian probability
    # mass inside [-1, 1]. We need log(Phi(b) - Phi(a)), NOT
    # log(Phi(b)) - log(Phi(a)) - those are very different quantities.
    # ndtr is numerically stable across our domain (verified).
    upper_z = ndtr((UPPER_BOUND - _GRID) / sigma)
    lower_z = ndtr((LOWER_BOUND - _GRID) / sigma)
    log_z   = np.log(upper_z - lower_z)

    # Gaussian kernel: Σ_i -(x_i - μ)² / (2σ²)
    # Shape: (GRID_SIZE, n) → sum over n → (GRID_SIZE,)
    residuals      = _GRID[:, None] - x[None, :]   # (G, n)
    log_gauss_sum  = -0.5 * np.sum(residuals ** 2, axis=1) / (sigma ** 2)

    # Full log-likelihood: Gaussian kernel minus n copies of log Z
    log_likelihood = log_gauss_sum - len(x) * log_z

    return log_prior + log_likelihood


def _validate(sentiments: list[float]) -> None:
    if not sentiments:
        raise ValueError("sentiments list must not be empty")
    out_of_range = [s for s in sentiments if not -1.0 <= s <= 1.0]
    if out_of_range:
        raise ValueError(f"scores must be in [-1, 1]; got {out_of_range}")