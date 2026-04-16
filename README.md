# KiwiPulse - Probabilistic Market Intelligence Engine

Most sentiment pipelines return a number. This one returns a number, a confidence interval, and a principled measure of how much to trust it. Instead of averaging scores, this models uncertainty explicitly.

---

## The Problem With Naive Sentiment Analysis

The standard approach:

```
articles → LLM → average score → "market is bullish: 0.72"
```

This fails in three ways:

- **No uncertainty.** A score from 2 articles and a score from 50 articles look identical.
- **No disagreement signal.** Sources that wildly contradict each other produce the same output as sources that agree - just noisier.
- **Uncalibrated confidence.** LLMs will report confidence scores, but those scores are not statistically meaningful.

The result is false precision: a single number that hides everything you actually need to know.

---

## The Solution

Treat each LLM sentiment score as a **noisy observation of a latent true signal**, and use Bayesian inference to recover that signal with explicit uncertainty.

**Model:**

```
x_i ~ Normal(μ, σ²)    - each score is a noisy measurement
μ   ~ Normal(0, 1)     - prior: neutral, weakly informative
σ²  = Var(x_1..x_n)   - estimated from data (empirical Bayes)
```

**Output:**

```json
{
  "mean": 0.41,
  "lower_bound": 0.18,
  "upper_bound": 0.64,
  "variance": 0.014,
  "sample_size": 8
}
```

A point estimate, a 95% credible interval, and a variance - all derived analytically, no sampling required.

---

## Key Insight

Uncertainty is not added after the fact - it emerges directly from disagreement in the data.

---

## Architecture

```
POST /analyze
      │
      ▼
 RawTextInput[]          ← Pydantic v2, strict validation, UTC timestamps
      │
      ▼
 LLM Scoring             ← GPT-4o-mini, temperature=0, JSON mode enforced
 (per article)              score ∈ [-1, 1], validated explicitly
      │
      ▼
 Bayesian Update         ← Normal-Normal conjugate, closed-form posterior
 estimate_market()          empirical Bayes for σ², soft-clamped output
      │
      ▼
 MarketEstimate          ← mean, lower_bound, upper_bound, variance, n
```

```
src/
├── core/bayesian_model.py   # inference logic - the statistical core
├── llm/sentiment.py         # LLM boundary - isolated and mockable
├── schema/models.py         # data contracts - Pydantic v2
├── api/routes.py            # FastAPI - request validation + error routing
└── main.py
```

---

## Key Technical Decisions

**Empirical Bayes for σ²**
Rather than fixing observation noise, we estimate it from the sample variance of incoming scores. High disagreement between articles → high σ² → lower data precision → wider credible interval. The model is uncertainty-aware by construction.

**No LLM confidence scores**
LLM self-reported confidence is not calibrated. We do not use it. Uncertainty is derived entirely from the statistical distribution of scores. This is explicitly documented in the codebase and a deliberate design choice.

**Normal-Normal conjugate**
Gives a closed-form posterior - no MCMC, no sampling, no black box. The update equations are four lines of arithmetic. Every output is fully explainable.

**Soft clamping at boundaries**
The posterior interval can legitimately exceed [-1, 1] with sparse or extreme data. We log a warning and soft-clamp rather than silently truncating, preserving statistical honesty while enforcing domain constraints at the API boundary.

**Isolated LLM boundary**
`_call_llm` is separated from `_parse_and_validate`. This means the entire parsing and validation logic is unit-testable without any API calls or mocking the OpenAI client.

**422 vs 503 error routing**
Parsing failures return 422 (bad input or unexpected LLM output). API-level failures return 503 (dependency unavailable). These are different failure modes and should not be conflated.

---

## Example

**Request:**
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "articles": [
      {
        "title": "Dairy prices surge on strong Chinese demand",
        "content": "GlobalDairyTrade index rose 4.2% overnight...",
        "source": "Reuters",
        "timestamp": "2025-03-15T08:00:00Z"
      },
      {
        "title": "Fonterra revises forecast amid supply concerns",
        "content": "Fonterra lowered its farmgate milk price...",
        "source": "NZHerald",
        "timestamp": "2025-03-15T09:30:00Z"
      }
    ]
  }'
```

**Response:**
```json
{
  "estimate": {
    "mean": 0.312,
    "lower_bound": -0.089,
    "upper_bound": 0.713,
    "variance": 0.041,
    "sample_size": 2
  },
  "observations": 2
}
```

The wide interval reflects low sample size - exactly correct behavior with two articles.

---

## Visualizations

**Posterior convergence** - credible interval narrows as n increases:

![Convergence](outputs/convergence.png)

The posterior starts near the prior (0.0) and converges toward the true signal as data accumulates. Log-scaled x-axis shows the dramatic early convergence clearly.

**Uncertainty from disagreement** - same n, different spread:

![Uncertainty](outputs/uncertainty.png)

Both datasets have five articles. The consensus case produces a tight interval; the disagreement case produces a wide one. The difference is driven entirely by σ² - the model correctly represents that conflicting sources should produce lower confidence.

---

## Running Locally

```bash
# Install
pip install fastapi uvicorn openai pydantic numpy

# Set API key
export OPENAI_API_KEY=your_key_here

# Run
uvicorn main:app --reload

# Tests (no API key required)
pytest tests/

# Visualizations
python scripts/visualize_convergence.py
python scripts/visualize_uncertainty.py
```

---

## What I'd Build Next

**Calibration validation** - run the model against historical data with known outcomes and check whether the 95% credible interval actually contains the true value ~95% of the time. This is the difference between a model that *claims* uncertainty and one that *earns* it.

**Prior sensitivity analysis** - show how results change under different prior choices (τ² = 0.25 vs 1.0 vs 4.0). Demonstrates awareness of prior dependency and builds trust in the chosen hyperparameters.

**Parallel LLM scoring** - current implementation is sequential; latency scales linearly with article count. `asyncio.gather` would parallelise the independent API calls with minimal structural change.

**Temporal weighting** - recent articles should contribute more signal than older ones. A natural extension is exponential decay on the likelihood weights, which preserves the conjugate structure.

**Source reliability priors** - if Reuters and an anonymous blog both publish articles, they should not carry equal weight. A hierarchical model could learn per-source noise parameters from data.

---

## Limitations

- Assumes Gaussian noise - may break under multimodal sentiment
- Treats all sources equally (no credibility weighting)
- Empirical Bayes σ² can be unstable for very small n
- LLM scores themselves are not validated against ground truth
- Calibration testing shows ~89% empirical coverage vs the nominal 95%, caused by asymmetric clamping near the domain boundaries
- Miscalibration is partly structural due to bounded domain and Gaussian assumptions - boundary interaction, likelihood mismatch, and empirical Bayes σ² instability all contribute

This is a statistical aggregation layer, not a ground-truth oracle.

---

## Does This Actually Help Decisions?

We simulated a decision system where predictions are only made when the 95% credible interval does not cross zero - i.e. when the model is confident enough to commit.

```
Decision rule:
  lower_bound > 0  → predict positive
  upper_bound < 0  → predict negative
  otherwise        → abstain
```

Result across 2,000 simulated runs:

| | Bayesian (uncertainty-aware) | Baseline (always commits) |
|---|---|---|
| Accuracy when deciding | **near-perfect under simulation** | 95.0% |
| Abstain rate | 22.6% | 0% |
| False positives | 3 | 50 |

Results are from synthetic data generated under model assumptions. Performance will degrade on real-world data with model mismatch - the calibration analysis shows this explicitly.

This demonstrates a key property: uncertainty is not just descriptive - it can be used to control risk. The model knows when it doesn't know, and that knowledge is actionable.

This system does not try to be right all the time, it tries to be right when it chooses to act.

This model is intentionally simple. Its value is not in complexity, but in making uncertainty explicit and usable.

---

## Why Bayesian Over a Simple Average?

| | Sample Mean | Posterior Mean |
|---|---|---|
| Uncertainty estimate | ✗ | ✓ |
| Small sample behavior | Overconfident | Pulled toward prior |
| Disagreement signal | Lost | Encoded in σ² |
| Explainability | Trivial | Fully analytical |

As n → ∞ with an uninformative prior, the posterior mean converges to the sample mean. The Bayesian approach strictly generalises the naive approach and is never worse.