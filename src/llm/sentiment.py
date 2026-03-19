"""
llm/sentiment.py

Converts raw article text into a structured sentiment score.

Design notes
------------
We use OpenAI's `response_format={"type": "json_object"}` to enforce
structured output rather than parsing free text with regex or split().
Its more reliable and easier to validate.

The score is a *point estimate only*. Confidence/uncertainty is
intentionally not derived from the LLM. LLM logit probabilities are
not exposed here and self-reported confidence is uncalibrated.
Uncertainty is handled downstream by the Bayesian model.
"""

import json
import os

from openai import OpenAI

from schema.models import SentimentObservation, RawTextInput

_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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
        If the API call itself fails.
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
    response = _client.chat.completions.create(
        model="gpt-4o-mini",          # cheap, fast, good enough for scoring
        temperature=0,                # deterministic - we want consistency
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _USER_TEMPLATE.format(
                title=title,
                content=content[:2000],   # guard against token blowout
            )},
        ],
    )
    return response.choices[0].message.content


def _parse_and_validate(raw: str) -> tuple[float, str]:
    """
    Parse JSON response and validate fields.

    We do not trust the LLM to stay within bounds, so we validate
    explicitly rather than relying on prompt instructions alone.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned non-JSON output: {raw!r}") from e

    # Check required keys exist
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