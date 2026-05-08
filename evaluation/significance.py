"""
Statistical significance tests for pairwise pipeline comparisons.

Two complementary tests are exposed:

1. **Paired bootstrap** on accuracy.
   For each of B bootstrap resamples (with replacement) of the example indices,
   compute the accuracy of pipeline A and pipeline B on the same resampled
   indices, and record the difference. The 95 % CI of the difference and the
   two-sided bootstrap p-value (fraction of resamples where the sign flips
   relative to the observed mean) summarise the comparison.

2. **McNemar's exact test** on the 2×2 contingency table of agreement.
   Standard test for paired binary outcomes (correct / incorrect). Uses the
   exact binomial form so it's valid for small N.

Both tests assume the two pipelines were evaluated on the *same* examples in
the same order — which is how ExperimentRunner produces them.
"""

from __future__ import annotations

import math
import random
from typing import Iterable


def _correct_vector(predictions: list[str], truths: list[str]) -> list[int]:
    if len(predictions) != len(truths):
        raise ValueError(
            f"Length mismatch: {len(predictions)} preds vs {len(truths)} truths"
        )
    return [1 if p == t else 0 for p, t in zip(predictions, truths)]


# ── Paired bootstrap ──────────────────────────────────────────────────────────

def paired_bootstrap(
    correct_a: list[int],
    correct_b: list[int],
    n_iter: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """
    Paired bootstrap on accuracy difference (A − B).

    Returns
    -------
    dict with:
      delta_accuracy  : observed mean of (A − B)
      ci_low, ci_high : confidence interval bounds for the difference
      p_value         : two-sided bootstrap p-value
      n               : sample size
    """
    n = len(correct_a)
    if n != len(correct_b):
        raise ValueError("correct_a and correct_b must have the same length")
    if n == 0:
        return {
            "delta_accuracy": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_value": 1.0,
            "n": 0,
        }

    rng = random.Random(seed)
    observed_delta = sum(correct_a) / n - sum(correct_b) / n

    deltas: list[float] = []
    for _ in range(n_iter):
        idx = [rng.randrange(n) for _ in range(n)]
        a = sum(correct_a[i] for i in idx) / n
        b = sum(correct_b[i] for i in idx) / n
        deltas.append(a - b)

    deltas.sort()
    alpha = (1 - confidence) / 2
    lo = deltas[int(alpha * n_iter)]
    hi = deltas[int((1 - alpha) * n_iter) - 1]

    # Two-sided p: fraction of bootstrap samples whose delta has the opposite
    # sign to the observed mean (or is exactly zero), times 2.
    if observed_delta == 0:
        p_value = 1.0
    elif observed_delta > 0:
        p_value = min(1.0, 2 * sum(1 for d in deltas if d <= 0) / n_iter)
    else:
        p_value = min(1.0, 2 * sum(1 for d in deltas if d >= 0) / n_iter)

    return {
        "delta_accuracy": observed_delta,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": p_value,
        "n": n,
    }


# ── McNemar's test ────────────────────────────────────────────────────────────

def mcnemar(correct_a: list[int], correct_b: list[int]) -> dict[str, float]:
    """
    Exact McNemar test on paired binary outcomes.

    b = examples where A is correct, B is wrong
    c = examples where A is wrong,   B is correct

    Under H0 (no difference), b and c follow Binomial(b+c, 0.5).
    Two-sided exact p-value = 2 * P(X <= min(b,c)).
    """
    if len(correct_a) != len(correct_b):
        raise ValueError("correct_a and correct_b must have the same length")
    b = sum(1 for a, c in zip(correct_a, correct_b) if a == 1 and c == 0)
    c = sum(1 for a, ci in zip(correct_a, correct_b) if a == 0 and ci == 1)
    n = b + c

    if n == 0:
        return {"b": 0, "c": 0, "n_disagreements": 0, "p_value": 1.0}

    k = min(b, c)
    # P(X <= k) under Binomial(n, 0.5)
    log_half_n = -n * math.log(2)
    cdf = 0.0
    for i in range(0, k + 1):
        # math.comb is exact integer; combine with log_half_n for stability
        cdf += math.comb(n, i) * math.exp(log_half_n)
    p_value = min(1.0, 2 * cdf)

    return {"b": b, "c": c, "n_disagreements": n, "p_value": p_value}


# ── High-level convenience: compare pipelines vs a baseline ───────────────────

def compare_pipelines(
    pipeline_results: dict[str, list[dict]],
    baseline: str = "vector_rag",
    n_iter: int = 5000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """
    For each non-baseline pipeline, compute paired bootstrap + McNemar vs baseline.

    Each result dict in pipeline_results[name] must have 'predicted_label' and
    'true_label'. The function will line up examples by `doc_id` if present
    (defensive — runs through ExperimentRunner already align by index).
    """
    if baseline not in pipeline_results:
        return {}

    base = pipeline_results[baseline]
    base_correct = _correct_vector(
        [r.get("predicted_label", "") for r in base],
        [r.get("true_label", "") for r in base],
    )

    out: dict[str, dict[str, float]] = {}
    for pname, results in pipeline_results.items():
        if pname == baseline:
            continue
        if len(results) != len(base):
            # If lengths differ, attempt to align by doc_id
            base_by_doc = {r.get("doc_id", i): r for i, r in enumerate(base)}
            aligned_a, aligned_b = [], []
            for r in results:
                doc = r.get("doc_id")
                if doc in base_by_doc:
                    aligned_a.append(r)
                    aligned_b.append(base_by_doc[doc])
            if not aligned_a:
                continue
            a_correct = _correct_vector(
                [r.get("predicted_label", "") for r in aligned_a],
                [r.get("true_label", "") for r in aligned_a],
            )
            b_correct = _correct_vector(
                [r.get("predicted_label", "") for r in aligned_b],
                [r.get("true_label", "") for r in aligned_b],
            )
        else:
            a_correct = _correct_vector(
                [r.get("predicted_label", "") for r in results],
                [r.get("true_label", "") for r in results],
            )
            b_correct = base_correct

        boot = paired_bootstrap(a_correct, b_correct, n_iter=n_iter, seed=seed)
        mcn = mcnemar(a_correct, b_correct)
        out[pname] = {
            **boot,
            "mcnemar_b": mcn["b"],
            "mcnemar_c": mcn["c"],
            "mcnemar_p": mcn["p_value"],
        }
    return out
