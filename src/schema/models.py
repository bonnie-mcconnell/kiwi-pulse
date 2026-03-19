"""
schema/models.py

Core data contracts for KiwiPulse.
Represents the shapes data takes as it moves through the pipeline.
"""

from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class RawTextInput(BaseModel):
    """A single raw article or news item before any processing."""

    model_config = {"extra": "forbid"}

    id: UUID = Field(default_factory=uuid4)
    title: str = Field(..., min_length=1, description="Article headline")
    content: str = Field(..., min_length=10, description="Article body text")
    source: str = Field(..., min_length=1, description="Origin of the article, e.g 'Reuters'")
    timestamp: datetime = Field(..., description="Publication time (UTC)")

    @field_validator("timestamp")
    @classmethod
    def timestamp_not_future(cls, v: datetime) -> datetime:
        # Normalise to UTC for comparison
        now = datetime.now(timezone.utc)
        v_aware = v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        if v_aware > now:
            raise ValueError("timestamp cannot be in the future")
        return v_aware

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


class SentimentObservation(BaseModel):
    """
    LLM-extracted sentiment for a single article.

    Confidence is intentionally excluded here.
    Uncertainty is derived statistically from the distribution
    of scores, not from the LLM, which produces uncalibrated outputs.
    """

    model_config = {"extra": "forbid"}

    id: UUID = Field(default_factory=uuid4)
    raw_id: UUID = Field(..., description="References the RawTextInput this was derived from")
    sentiment_score: float = Field(..., description="Sentiment in [-1.0, 1.0]. -1 = very bearish, 1 = very bullish")
    reasoning: str = Field(..., min_length=1, description="LLM explanation for the assigned score")

    @field_validator("sentiment_score")
    @classmethod
    def score_in_range(cls, v: float) -> float:
        if not -1.0 <= v <= 1.0:
            raise ValueError(f"sentiment_score must be in [-1, 1], got {v}")
        return round(v, 4)  # normalise float precision from LLM JSON serialisation

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


class MarketEstimate(BaseModel):
    """
    Output of the Bayesian inference step.

    Represents our best estimate of true underlying market sentiment,
    along with a 95% credible interval and variance.

    Invariant: lower_bound <= mean <= upper_bound
    """

    model_config = {"extra": "forbid"}

    mean: float = Field(..., description="Posterior mean — best estimate of true sentiment")
    lower_bound: float = Field(..., description="Lower bound of 95% credible interval")
    upper_bound: float = Field(..., description="Upper bound of 95% credible interval")
    variance: float = Field(..., ge=0.0, description="Posterior variance - measure of remaining uncertainty. " \
    "Near-zero values are valid and handled in bayesian_model.py.")
    sample_size: int = Field(..., ge=1, description="Number of sentiment observations used")

    @model_validator(mode="after")
    def bounds_are_consistent(self) -> "MarketEstimate":
        if not (self.lower_bound <= self.mean <= self.upper_bound):
            raise ValueError(
                f"Expected lower_bound <= mean <= upper_bound, "
                f"got [{self.lower_bound}, {self.mean}, {self.upper_bound}]"
            )
        return self

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

