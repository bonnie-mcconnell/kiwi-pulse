"""
tests/test_bayesian_model.py

Tests for the Bayesian inference core.

We test behavior, not implementation — meaning we check that outputs
satisfy statistical properties, not that specific intermediate values
match. This makes tests robust to refactoring.
"""

import pytest

from core.bayesian_model import estimate_market


class TestSingleObservation:
    """With one data point, the posterior should stay close to the prior."""

    def test_returns_valid_market_estimate(self):
        result = estimate_market([0.5])
        assert result.sample_size == 1
        assert -1.0 <= result.mean <= 1.0
        assert result.variance > 0

    def test_interval_is_wide(self):
        # With n=1 we have almost no information — interval should be wide.
        # "Wide" here means > 1.0 total width, which is half the domain.
        result = estimate_market([0.5])
        width = result.upper_bound - result.lower_bound
        assert width > 1.0, f"Expected wide interval for n=1, got width={width:.3f}"

    def test_prior_pull(self):
        # With one extreme observation, prior (mean=0) should pull the
        # posterior mean back toward zero — not sit at the raw score.
        result = estimate_market([1.0])
        assert result.mean < 1.0, "Posterior should be pulled toward prior"
        assert result.mean > 0.0, "Posterior should still reflect the positive signal"


class TestIdenticalScores:
    """All identical scores = zero empirical variance = near-zero observation noise."""

    def test_tight_interval(self):
        # If every article agrees, uncertainty should be low.
        result = estimate_market([0.6, 0.6, 0.6, 0.6, 0.6])
        width = result.upper_bound - result.lower_bound
        assert width < 0.5, f"Expected tight interval for identical scores, got {width:.3f}"

    def test_mean_near_score(self):
        # With many identical observations, posterior should commit to that value.
        result = estimate_market([0.6] * 10)
        assert abs(result.mean - 0.6) < 0.1


class TestHighDisagreement:
    """Spread-out scores should produce a wide credible interval."""

    def test_wide_interval(self):
        scores = [-0.9, -0.5, 0.0, 0.5, 0.9]
        result = estimate_market(scores)
        low_disagreement = estimate_market([0.3, 0.3, 0.3, 0.3, 0.3])

        assert result.variance > low_disagreement.variance, (
            "High disagreement should produce higher variance than consensus"
        )

    def test_mean_near_zero(self):
        # Symmetric disagreement should produce a mean near the prior (0).
        scores = [-0.8, -0.4, 0.0, 0.4, 0.8]
        result = estimate_market(scores)
        assert abs(result.mean) < 0.2


class TestConvergence:
    """As n grows, the posterior should converge and tighten."""

    def test_interval_narrows_with_more_data(self):
        few   = estimate_market([0.5, 0.5, 0.5])
        many  = estimate_market([0.5] * 30)
        assert many.variance < few.variance

    def test_mean_converges_toward_data(self):
        # With many consistent positive observations, mean should approach
        # the observed value and move away from the prior (0).
        result = estimate_market([0.7] * 50)
        assert result.mean > 0.6


class TestBoundsInvariant:
    """lower_bound <= mean <= upper_bound must always hold."""

    @pytest.mark.parametrize("scores", [
        [0.0],
        [-1.0, 1.0],
        [0.1, 0.2, 0.3],
        [-0.5, -0.5, -0.5],
    ])
    def test_bounds_ordering(self, scores):
        result = estimate_market(scores)
        assert result.lower_bound <= result.mean <= result.upper_bound


class TestValidation:
    """Invalid inputs should raise immediately, not produce bad output."""

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            estimate_market([])

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            estimate_market([0.5, 1.5])

    def test_single_invalid_value_raises(self):
        with pytest.raises(ValueError):
            estimate_market([-2.0])