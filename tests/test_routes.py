"""
tests/test_routes.py

Integration tests for the /analyze endpoint.

We mock analyze_sentiment so these tests cover:
- request validation
- routing logic
- error handling
...without making real API calls.
"""

from datetime import datetime, timezone
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from main import app
from schema.models import MarketEstimate, SentimentObservation

client = TestClient(app)

# ----- fixtures ------------------------------------------------------

def _make_article(title: str = "Test article", content: str = "x" * 20) -> dict:
    """Minimal valid article payload."""
    return {
        "title": title,
        "content": content,
        "source": "TestSource",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _make_observation(score: float = 0.5) -> SentimentObservation:
    return SentimentObservation(
        raw_id=uuid4(),
        sentiment_score=score,
        reasoning="Test reasoning.",
    )


# ----- success cases -------------------------------------------------

class TestSuccessfulAnalysis:
    def test_single_article_returns_estimate(self):
        with patch("api.routes.analyze_sentiment", return_value=_make_observation(0.5)):
            response = client.post("/analyze", json={"articles": [_make_article()]})

        assert response.status_code == 200
        body = response.json()
        assert "estimate" in body
        assert "observations" in body
        assert body["observations"] == 1

    def test_multiple_articles_counted_correctly(self):
        with patch("api.routes.analyze_sentiment", return_value=_make_observation(0.3)):
            response = client.post("/analyze", json={
                "articles": [_make_article(f"Article {i}") for i in range(5)]
            })

        assert response.status_code == 200
        assert response.json()["observations"] == 5

    def test_estimate_has_required_fields(self):
        with patch("api.routes.analyze_sentiment", return_value=_make_observation(0.5)):
            response = client.post("/analyze", json={"articles": [_make_article()]})

        estimate = response.json()["estimate"]
        for field in ["mean", "lower_bound", "upper_bound", "variance", "sample_size"]:
            assert field in estimate, f"Missing field: {field}"


# ----- validation errors ---------------------------------------------

class TestRequestValidation:
    def test_empty_articles_list_rejected(self):
        response = client.post("/analyze", json={"articles": []})
        assert response.status_code == 422

    def test_missing_articles_field_rejected(self):
        response = client.post("/analyze", json={})
        assert response.status_code == 422

    def test_article_missing_required_field_rejected(self):
        bad_article = {"title": "No content or source or timestamp"}
        response = client.post("/analyze", json={"articles": [bad_article]})
        assert response.status_code == 422

    def test_future_timestamp_rejected(self):
        article = _make_article()
        article["timestamp"] = "2099-01-01T00:00:00Z"
        response = client.post("/analyze", json={"articles": [article]})
        assert response.status_code == 422


# ----- downstream failures -------------------------------------------

class TestDownstreamErrors:
    def test_llm_parse_failure_returns_422(self):
        with patch("api.routes.analyze_sentiment", side_effect=ValueError("Bad output")):
            response = client.post("/analyze", json={"articles": [_make_article()]})

        assert response.status_code == 422
        assert "Failed to score" in response.json()["detail"]

    def test_llm_api_failure_returns_503(self):
        with patch("api.routes.analyze_sentiment", side_effect=RuntimeError("Timeout")):
            response = client.post("/analyze", json={"articles": [_make_article()]})

        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"]