# What I Learned Building a Bayesian Sentiment Engine

*A technical account of what worked, what didn't, and what the results actually mean.*

---

## The Problem

Most sentiment analysis pipelines return a number. A score of 0.72 looks precise. But precision without uncertainty is false confidence. A score from two articles and a score from fifty articles look identical, and sources that violently disagree produce the same output as sources that agree, just noisier.

The core question I wanted to answer: **can you build a system that knows when it doesn't know?**

---

## The Initial Approach

I framed sentiment scoring as a noisy measurement problem. Each LLM-produced score $x_i$ is a noisy observation of some true latent market sentiment $\mu$:

$$x_i \sim \mathcal{N}(\mu, \sigma^2), \quad \mu \sim \mathcal{N}(0, 1)$$

The Normal-Normal conjugate model gives a closed-form posterior: four lines of arithmetic, fully explainable, no black box. Uncertainty emerges directly from disagreement between sources: high variance in scores produces high $\sigma^2$, which widens the credible interval. No uncertainty is added after the fact; it's a direct consequence of the data.

The prior $\mu \sim \mathcal{N}(0, 1)$ encodes a reasonable starting assumption: markets are neutral before we see any evidence. As data accumulates, the posterior pulls away from this prior at a rate controlled by how much the data actually tells us.

---

## The First Real Problem: Small Sample Instability

Early calibration testing revealed that empirical Bayes - estimating $\sigma^2$ directly from sample variance - produces only 69.6% coverage at $n=2$. The nominal target is 95%.

The cause: sample variance of two points swings wildly. When it lands near zero by chance, the model produces a falsely tight interval. The posterior commits strongly to a value it has almost no evidence for.

The fix was a variance floor:

```python
observation_var = max(float(np.var(x, ddof=1)), VAR_FLOOR)  # VAR_FLOOR = 0.10
```

This encodes a domain assumption: LLM scores always carry at least some irreducible noise. A floor of 0.10 (std ≈ 0.316) improved coverage at $n=2$ from 69.6% to 92.6%. One line, principled, defensible.

---

## The Calibration Gap

With the floor in place, coverage across conditions settled around 88–92% against a 95% target. Three sources contribute to this gap:

**Asymmetric clamping near boundaries.** When the true $\mu$ is near ±1, observations get clamped asymmetrically. Scores that would have been 1.1 or 1.2 become 1.0, compressing the apparent variance. The model underestimates $\sigma^2$ and produces intervals that are too narrow.

**Gaussian likelihood mismatch.** The model assumes observations can take any real value. Sentiment scores are bounded to [−1, 1]. These are different worlds.

**Empirical Bayes $\sigma^2$ instability.** Sample variance is a noisy estimator, especially at small $n$.

Calibration tests across three boundary exposure regimes confirmed monotonic degradation: the more often the true $\mu$ lands near the boundary, the worse the coverage.

---

## The Attempted Fix: Truncated Normal Likelihood

The theoretically correct response to bounded data is a truncated Normal likelihood, which renormalises the Gaussian density to integrate to 1 over [−1, 1]:

$$p(x_i \mid \mu, \sigma) = \frac{\phi\left(\frac{x_i - \mu}{\sigma}\right)}{Z(\mu, \sigma)}, \quad Z(\mu, \sigma) = \Phi\left(\frac{1-\mu}{\sigma}\right) - \Phi\left(\frac{-1-\mu}{\sigma}\right)$$

This breaks conjugacy - the normalising constant $Z(\mu, \sigma)$ depends on $\mu$ in a way that doesn't cancel in the posterior. I implemented numerical grid integration (500 points over [−1, 1]), evaluating the unnormalised log-posterior at each point and reading off the credible interval from the discrete CDF.

**It didn't help.**

Calibration comparison across interior and boundary conditions showed no meaningful improvement. In the interior case it was slightly worse (90.6% vs 94.0%). In the boundary case it was essentially identical (91.3% vs 91.2%).

The reason: the variance floor had already addressed the primary source of miscalibration by a different mechanism. The two corrections are partially redundant. The floor adds conservative bias that protects against narrow intervals near boundaries; the truncated Normal corrects the likelihood shape. When the floor is already doing its job, the likelihood correction has less room to show improvement.

**This was the most valuable finding in the project.** Implementing something more principled, testing it rigorously, and discovering it doesn't improve on a simpler fix - and then being able to explain why - is the actual work of statistical engineering. The truncated Normal implementation is correct. Its failure to improve calibration is informative, not embarrassing.

---

## Adversarial Testing

Synthetic data generates from the same distribution the model assumes, which makes calibration experiments circular. I broke this by designing six adversarial conditions:

| Condition | Coverage | Notes |
|-----------|----------|-------|
| Baseline (Gaussian) | 0.919 | Benchmark |
| Mixture noise | 0.917 | Robust - σ² adapts |
| Systematic bias (+0.3) | 0.346 ⚠ | Catastrophic |
| Heavy tails (t, df=3) | 0.934 | Robust - σ² inflates |
| Regime shift | 0.509 ⚠ | Model has no memory |
| Correlated (AR ρ=0.7) | 0.519 ⚠ | iid assumption violated |

The model is surprisingly robust to mixture noise and heavy tails - because empirical Bayes $\sigma^2$ adapts automatically to both. It fails badly under three conditions: systematic bias (no correction mechanism), regime shifts mid-batch (no temporal model), and correlated observations (iid assumption violated, effective $n$ much smaller than actual $n$).

The correlation finding is the most operationally important. In a real news pipeline, articles from the same day react to the same events and are correlated. The model treats them as independent and produces intervals that are too narrow. This is the failure mode most likely to occur in production.

---

## Decision Making Under Uncertainty

The project includes a decision rule that uses the credible interval as a gate:

```
lower_bound > 0  → predict positive
upper_bound < 0  → predict negative
otherwise        → abstain
```

Simulation over 5,000 runs showed:

- Uncertainty-aware: **99.2% accuracy** when acting, **40% abstain rate**
- Always-commit baseline: **92.0% accuracy**, 0% abstain rate

The model doesn't try to be right all the time. It tries to be right when it chooses to act. Whether that tradeoff is worth it depends entirely on the cost structure of the application - which is the correct answer, not a hedge.

---

## Real-World Validation

I curated 30 trading days (Jan–Feb 2025) with real headlines and actual SPY next-day returns. Sentiment scores were hand-assigned, which introduces a methodological limitation: scoring with knowledge of the dates creates the possibility of look-ahead bias, even unintentional. This is a proof-of-concept, not a rigorous backtest.

Results across three strategies:

| Strategy | Accuracy | Coverage |
|----------|----------|----------|
| Single headline | 79.3% | 96.7% |
| Naive mean | 82.8% | 96.7% |
| Bayesian (abstain) | 94.1% | 56.7% |

Aggregation (mean vs single) showed +3.4pp - real but modest. The abstain rule showed +11.4pp over naive mean, at the cost of 43% of decisions. The Bayesian model abstained on all six high-disagreement days (score std > 0.3), declining to act precisely when sources disagreed most.

These results are directionally consistent with the synthetic experiments. They are not statistically significant at N=30.

---

## What the Variance Estimator Comparison Showed

A systematic comparison of three σ² estimators across $n \in \{2, 5, 10, 20\}$ produced the clearest finding in the project:

| Estimator | Coverage at n=2 | σ² std dev at n=2 |
|-----------|-----------------|-------------------|
| Empirical Bayes | 0.696 | 0.1887 |
| Floored (0.10) | 0.926 | 0.1652 |
| Hierarchical pooling | 0.949 | 0.0088 |

The hierarchical estimator - which pools σ² across batches rather than re-estimating per batch - is 21x more stable at n=2 and essentially hits the 95% nominal coverage. The floor is a good fix. Hierarchical pooling is the right production upgrade.

---

## What I Would Do Differently

**Use a hierarchical variance estimator from the start.** The per-batch empirical Bayes was always going to be unstable at small n. The fix was known in the literature before I started.

**Collect scores before seeing outcomes.** The real-world validation is methodologically compromised by hand-scoring with date knowledge. The correct experiment runs the LLM pipeline on archived articles before the market close, stores scores, then evaluates after. That's a two-week experiment, not a one-afternoon exercise.

**Model temporal dynamics.** The current model treats each batch as independent. Real sentiment has autocorrelation - the DeepSeek selloff on Jan 27 influenced sentiment for days. An AR(1) model on the posterior means, or exponential decay weighting of recent batches, would be a more honest representation of the data-generating process.

---

## The Central Finding

**The value is in the abstention, not the prediction.**

A system that predicts when confident and declines when uncertain is more useful than one that always predicts. The credible interval makes the confidence threshold explicit and tunable - not a hidden hyperparameter or a gut feeling, but a number with a statistical interpretation.

This applies well beyond sentiment analysis. Any system that aggregates noisy, conflicting signals (survey responses, sensor readings, expert opinions) faces the same problem. The Bayesian framework makes the uncertainty visible. What you do with it is a product decision, not a modeling decision.

