"""
llm/sentiment.py

Converts raw article text into a structured sentiment score.

Design notes
------------
We use OpenAI's response_format json_object to enforce structured output
rather than parsing free text. More reliable and easier to validate.

The score is a point estimate only. Confidence is intentionally not
derived from the LLM — LLM self-reported confidence is uncalibrated.
Uncertainty is handled downstream by the Bayesian model.

Known limitation
----------------
No retry logic. Transient failures raise RuntimeError immediately.
A production system would add exponential backoff via tenacity.
"""

import json
import os

from openai import OpenAI

from src.schema.models import RawTextInput, SentimentObservation

MODEL_NAME = "gpt-4o-mini"

# Lazy-initialised so importing this module doesn't require OPENAI_API_KEY.
# Tests that mock _call_llm never touch the client at all.
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


_SYSTEM_PROMPT = """\
You are a financial sentiment analyser. Your job is to read a news article
and return a single JSON object with exactly two fields:

  "score"     - a float in [-1.0, 1.0]
                -1.0 = extremely bearish
                 0.0 = neutral
                 1.0 = extremely bullish

  "reasoning" - one sentence (max 20 words) explaining the score

Rules:
- Return ONLY the JSON object. No preamble, no markdown.
- Score must be a single number, not a range.
- Do not hedge or qualify the score with words like "approximately".
- If the article is unrelated to markets or economics, return score 0.0.
"""

_USER_TEMPLATE = """\
Article title: {title}

Article text:
{content}
"""


def analyze_sentiment(article: RawTextInput) -> SentimentObservation:
    """
    Call the LLM and return a validated SentimentObservation.

    Parameters
    ----------
    article:
        A RawTextInput that has already been ingested and validated.

    Returns
    -------
    SentimentObservation
        Structured sentiment with score and reasoning.

    Raises
    ------
    ValueError
        If the LLM returns output that cannot be parsed or validated.
    RuntimeError
        If the API call itself fails or returns an unexpected structure.
    """
    raw_response = _call_llm(article.title, article.content)
    score, reasoning = _parse_and_validate(raw_response)

    return SentimentObservation(
        raw_id=article.id,
        sentiment_score=score,
        reasoning=reasoning,
    )


# ----- private helpers -----------------------------------------------

def _call_llm(title: str, content: str) -> str:
    """Make the API call. Isolated so it can be mocked in tests."""
    response = _get_client().chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        timeout=10,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _USER_TEMPLATE.format(
                title=title,
                content=content[:2000],
            )},
        ],
    )

    if not response.choices or not response.choices[0].message:
        raise RuntimeError("Invalid LLM response structure")

    return response.choices[0].message.content or ""


def _parse_and_validate(raw: str) -> tuple[float, str]:
    """
    Parse JSON response and validate fields.

    JSON mode reduces formatting errors but doesn't guarantee correctness.
    We validate all fields explicitly rather than trusting the model output.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned non-JSON output: {raw!r}") from e

    if "score" not in data or "reasoning" not in data:
        raise ValueError(f"Missing required fields in LLM response: {data}")

    try:
        score = float(data["score"])
    except (TypeError, ValueError) as e:
        raise ValueError(f"score is not a valid float: {data['score']!r}") from e

    if not -1.0 <= score <= 1.0:
        raise ValueError(f"score out of range [-1, 1]: {score}")

    reasoning = str(data["reasoning"]).strip()
    if not reasoning:
        raise ValueError("reasoning must not be empty")

    return score, reasoning
