"""
api/routes.py

HTTP interface for KiwiPulse.

Single endpoint: POST /analyze
  - Accepts a list of raw articles
  - Scores each with the LLM sentiment module
  - Feeds scores into the Bayesian model
  - Returns a posterior market estimate
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.core.bayesian_model import estimate_market
from src.llm.sentiment import analyze_sentiment
from src.schema.models import MarketEstimate, RawTextInput

router = APIRouter()

_MAX_ARTICLES = 50  # guard against accidental large payloads


class AnalyzeRequest(BaseModel):
    articles: list[RawTextInput] = Field(..., min_length=1, max_length=_MAX_ARTICLES)


class AnalyzeResponse(BaseModel):
    estimate: MarketEstimate
    observations: int  # how many articles contributed - useful for the caller


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Score a batch of articles and return a Bayesian market estimate.

    Each article is scored independently by the LLM. The scores are
    then passed to the Normal-Normal conjugate model which returns a
    posterior mean and 95% credible interval.

    Errors in any individual article cause the whole request to fail.
    This is intentional: a partial estimate is misleading.
    """
    scores = _score_articles(request.articles)
    estimate = estimate_market(scores)

    return AnalyzeResponse(estimate=estimate, observations=len(scores))


# ----- private helpers -----------------------------------------------

def _score_articles(articles: list[RawTextInput]) -> list[float]:
    """
    Run sentiment analysis on each article.

    Raises HTTP 422 if an individual article fails to score - keeping
    the error at the request boundary rather than leaking internal
    exceptions to the caller.

    Note: This is currently sequential and may be slow for large batches.
    In production, this would be parallelised (e.g asyncio.gather).
    """
    scores: list[float] = []

    for article in articles:
        try:
            observation = analyze_sentiment(article)
            scores.append(observation.sentiment_score)
        except ValueError as e:
            # Parsing / validation failure - bad input or unexpected LLM output
            raise HTTPException(
                status_code=422,
                detail=f"Failed to score article '{article.title}': {e}",
            ) from e
        except RuntimeError as e:
            # API-level failure - surface as 503, not 500
            raise HTTPException(
                status_code=503,
                detail=f"LLM service unavailable while scoring '{article.title}': {e}",
            ) from e

    return scores