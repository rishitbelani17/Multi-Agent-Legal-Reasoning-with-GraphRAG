"""
Evaluation metrics: accuracy, F1, precision, recall.

Supports:
  - Multi-class classification  (LEDGAR clause type)
  - Multiple-choice              (CaseHOLDER)
  - Binary classification        (ECtHR violation prediction)

All functions operate on lists of strings/ints so they work regardless of
how the LLM answer was parsed.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


# ── Answer parsing ────────────────────────────────────────────────────────────

def parse_answer(raw_answer: str, label_space: list[str] | None = None) -> str:
    """
    Extract the predicted label from an LLM-generated answer string.

    Strategy:
    1. Look for 'ANSWER: <text>' in the response.
    2. If label_space provided, find the best fuzzy match.
    3. Fallback: return first token of the answer.
    """
    # 1. Try to find explicit ANSWER: tag
    m = re.search(r"ANSWER\s*:\s*(.+?)(?:\n|$)", raw_answer, re.IGNORECASE)
    answer_str = m.group(1).strip() if m else raw_answer.strip().split("\n")[0]

    if not label_space:
        return answer_str

    # 2. Exact match (case-insensitive)
    answer_lower = answer_str.lower()
    for label in label_space:
        if label.lower() == answer_lower:
            return label

    # 3. Substring match
    for label in label_space:
        if label.lower() in answer_lower:
            return label

    # 4. Partial match – return closest label by character overlap
    best_label = label_space[0]
    best_score = 0
    for label in label_space:
        overlap = _char_overlap(label.lower(), answer_lower)
        if overlap > best_score:
            best_score = overlap
            best_label = label

    return best_label


def _char_overlap(a: str, b: str) -> float:
    """Jaccard character-bigram overlap between two strings."""
    def bigrams(s: str) -> set:
        return {s[i:i+2] for i in range(len(s) - 1)}
    ba, bb = bigrams(a), bigrams(b)
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / len(ba | bb)


# ── Per-example metrics ───────────────────────────────────────────────────────

def compute_metrics(
    predictions: list[str | int],
    ground_truths: list[str | int],
    dataset: str = "ledgar",
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute classification metrics for a list of predictions.

    Returns a dict with accuracy, macro F1, weighted F1, precision, recall,
    and a full sklearn classification report.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs "
            f"{len(ground_truths)} ground truths"
        )

    avg = "binary" if dataset == "ecthr" else "macro"

    acc = accuracy_score(ground_truths, predictions)
    f1_macro = f1_score(ground_truths, predictions, average="macro", zero_division=0)
    f1_weighted = f1_score(ground_truths, predictions, average="weighted", zero_division=0)
    prec = precision_score(ground_truths, predictions, average=avg, zero_division=0)
    rec = recall_score(ground_truths, predictions, average=avg, zero_division=0)

    report = classification_report(
        ground_truths,
        predictions,
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision": float(prec),
        "recall": float(rec),
        "classification_report": report,
        "n_samples": len(predictions),
    }


# ── Aggregate across pipelines ────────────────────────────────────────────────

def consistency_score(
    runs: list[list[str]],
) -> dict[str, float]:
    """
    Inter-run consistency across N repeated runs of the same pipeline on
    the same examples (proposal §2.4).

    Parameters
    ----------
    runs:
        A list of length N. Each entry is the predictions from one run,
        in the same example order. Lengths must match across runs.

    Returns
    -------
    dict with:
        agreement_rate : fraction of examples where all N runs agree
        majority_rate  : fraction of examples where the modal answer was given
                         by > N/2 of the runs
        mean_pairwise  : mean pairwise agreement across all run pairs
    """
    if not runs or len(runs) < 2:
        return {"agreement_rate": 1.0, "majority_rate": 1.0, "mean_pairwise": 1.0}

    n_runs = len(runs)
    n_examples = len(runs[0])
    if any(len(r) != n_examples for r in runs):
        raise ValueError("All runs must have the same number of predictions.")

    full_agree = 0
    majority_agree = 0
    for i in range(n_examples):
        col = [runs[r][i] for r in range(n_runs)]
        counts: dict[str, int] = defaultdict(int)
        for v in col:
            counts[v] += 1
        top = max(counts.values())
        if top == n_runs:
            full_agree += 1
        if top > n_runs / 2:
            majority_agree += 1

    # Mean pairwise agreement
    pair_total, pair_match = 0, 0
    for a in range(n_runs):
        for b in range(a + 1, n_runs):
            for i in range(n_examples):
                pair_total += 1
                if runs[a][i] == runs[b][i]:
                    pair_match += 1

    return {
        "agreement_rate": full_agree / n_examples,
        "majority_rate": majority_agree / n_examples,
        "mean_pairwise": pair_match / pair_total if pair_total else 1.0,
        "n_runs": n_runs,
        "n_examples": n_examples,
    }


def aggregate_results(
    pipeline_results: dict[str, list[dict[str, Any]]],
    dataset: str = "ledgar",
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Given {pipeline_name: [result_dict, ...]}, compute and return a summary
    table suitable for logging / plotting.

    Each result_dict must have keys: 'predicted_label', 'true_label',
    'latency_s', 'input_tokens', 'output_tokens'.
    """
    summary: dict[str, Any] = {}

    for pipeline_name, results in pipeline_results.items():
        preds = [r["predicted_label"] for r in results]
        truths = [r["true_label"] for r in results]
        metrics = compute_metrics(preds, truths, dataset=dataset, label_names=label_names)

        latencies = [r.get("latency_s", 0.0) for r in results]
        in_tokens = [r.get("input_tokens", 0) for r in results]
        out_tokens = [r.get("output_tokens", 0) for r in results]

        summary[pipeline_name] = {
            **metrics,
            "avg_latency_s": float(np.mean(latencies)),
            "total_input_tokens": int(sum(in_tokens)),
            "total_output_tokens": int(sum(out_tokens)),
            "avg_input_tokens": float(np.mean(in_tokens)),
            "avg_output_tokens": float(np.mean(out_tokens)),
        }

    return summary
