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

    Strategy (in order; first match wins):
    0. Look for 'FINAL ANSWER:' (the strict format used by the JudgeAgent).
    1. Look for 'ANSWER:' (the looser format used by single-LLM pipelines).
    2. Anchored regex for known label patterns (e.g. 'holding_2', 'violated').
       This is dataset-aware and avoids fuzzy fallback errors where the Judge's
       prose accidentally substring-matched the wrong label.
    3. If label_space provided: exact match (case-insensitive).
    4. Substring match. Skipped for CaseHOLD because labels are too similar
       ('holding_2' is a substring of 'holding_24', and rhetorical phrases like
       'interested in the outcome' coincidentally match candidate text).
    5. **Body-vote fallback** (NEW): the Judge often hits its max_tokens cap
       before producing the four-line tail block. In that case, look at every
       holding_X mention in the body, weighted by position (later mentions
       count more — they're closer to the conclusion). Return the most-cited
       label. This is what rescues truncated multi_agent_vector transcripts
       without needing a re-run.
    6. Anchored "concluding verb" patterns near the end of the body
       ("the answer is holding_2", "I therefore hold holding_3", etc.).
    7. Fall back to the literal answer string. We DO NOT do bigram fuzzy
       matching anymore — it produced spurious matches between e.g. holding_2
       and holding_3.
    """
    # 0. Strictest: FINAL ANSWER on its own line (Judge format)
    m = re.search(
        r"FINAL\s*ANSWER\s*[:\-]\s*([^\n]+?)\s*$",
        raw_answer,
        re.IGNORECASE | re.MULTILINE,
    )
    if not m:
        # 1. Looser: any ANSWER: line
        m = re.search(r"ANSWER\s*[:\-]\s*([^\n]+)", raw_answer, re.IGNORECASE)

    answer_str = m.group(1).strip() if m else raw_answer.strip().split("\n")[0]
    # Strip common ornamentation: leading/trailing quotes, parens, asterisks
    answer_str = answer_str.strip(" .;,'\"`*()[]{}")

    # 2. Anchored regex for canonical label patterns. Far more reliable than
    #    fuzzy substring on multi-class spaces.
    canonical_patterns = [
        # CaseHOLD: 'holding_2', tolerant of 'holding 2', 'Holding_2'
        (r"\bholding[\s_-]?([0-4])\b", lambda mm: f"holding_{mm.group(1)}"),
        # ECtHR
        (r"\b(violated|not[\s_-]?violated|no[\s_-]?violation)\b", _ecthr_norm),
    ]
    for pattern, normalizer in canonical_patterns:
        m2 = re.search(pattern, answer_str, re.IGNORECASE)
        if m2:
            candidate = normalizer(m2)
            if not label_space or candidate in label_space:
                return candidate

    if not label_space:
        return answer_str

    # 3. Exact (case-insensitive)
    answer_lower = answer_str.lower()
    for label in label_space:
        if label.lower() == answer_lower:
            return label

    # 4. Substring match — only for label spaces where labels are not
    #    near-prefixes of each other (so e.g. LEDGAR clause types are fine but
    #    'holding_2' / 'holding_24' would be ambiguous).
    if not _labels_are_near_prefixes(label_space):
        for label in label_space:
            if label.lower() in answer_lower:
                return label

    # 5. Body-vote fallback for truncated Judge transcripts. We use the FULL
    #    raw answer here (not just the first line) because at this point the
    #    Judge clearly didn't produce a parseable FINAL ANSWER line — but its
    #    reasoning still names a specific holding_X repeatedly.
    #
    #    Concluding-verb patterns near the end win first; otherwise a
    #    position-weighted vote across all label mentions.
    body_pred = _body_vote_label(raw_answer, label_space)
    if body_pred is not None:
        return body_pred

    # 6. Last resort. If the literal extracted answer doesn't look like a
    #    candidate (markdown headers, prose snippets, etc.) AND we had a
    #    label_space to compare against, return '' so that the metrics
    #    classification report doesn't get polluted with stray strings like
    #    '## Judicial Analysis'. The empty prediction will count as wrong,
    #    which is the correct behaviour for an unparseable response.
    if not _looks_like_label(answer_str):
        return ""
    return answer_str


def _looks_like_label(s: str) -> bool:
    """Cheap sanity check that ``s`` plausibly matches the shape of a label
    rather than markdown / prose. Used to keep the classification report clean
    when the parser truly couldn't recover an answer."""
    if not s:
        return False
    s = s.strip()
    if not s:
        return False
    if s.startswith(("#", "*", ">", "-", "`")):
        return False
    if len(s) > 80:
        return False
    # Reject anything with multiple sentences or markdown headings inside it
    if "\n" in s or s.count(" ") > 6:
        return False
    return True


def _body_vote_label(raw_answer: str, label_space: list[str]) -> str | None:
    """
    Recover a label from a long, truncated Judge body that never produced
    'FINAL ANSWER:'. Two strategies, in order:

    a) Concluding-verb anchors near the tail (e.g. 'the correct holding is
       holding_2', 'therefore I find holding_3', 'choose violated').
    b) Position-weighted mention vote: every occurrence of a known label
       contributes a weight that grows linearly with its character position
       in the body, so the holding the Judge keeps returning to at the end
       wins. This is empirically much more reliable than a top-mention vote
       on debate transcripts.
    """
    if not raw_answer or not label_space:
        return None

    label_lower = [l.lower() for l in label_space]
    label_set = set(label_lower)

    # ── (a) Concluding-verb anchors ──────────────────────────────────────────
    tail = raw_answer[-3500:]
    # CaseHOLD-style anchors
    cas = [
        r"correct\s+(?:holding|answer|choice|option)\s+is\s+holding[\s_-]?([0-4])",
        r"\banswer\s+is\s+holding[\s_-]?([0-4])",
        r"holding[\s_-]?([0-4])\s+is\s+(?:the\s+)?correct",
        r"\bselect\s+holding[\s_-]?([0-4])",
        r"\bchoose\s+holding[\s_-]?([0-4])",
        r"(?:therefore|accordingly|thus|hence)\s+(?:I\s+)?(?:find|conclude|hold|select|rule)[^.\n]{0,80}holding[\s_-]?([0-4])",
    ]
    for pat in cas:
        m = re.search(pat, tail, re.IGNORECASE)
        if m:
            cand = f"holding_{m.group(1)}"
            if cand in label_set:
                return cand

    # ECtHR-style anchors
    ec = [
        r"(?:therefore|accordingly|thus|hence)[^.\n]{0,80}\b(violated|not[\s_-]?violated|no[\s_-]?violation)\b",
        r"\b(?:I|the\s+court)\s+(?:find|conclude|hold)[^.\n]{0,80}\b(violated|not[\s_-]?violated|no[\s_-]?violation)\b",
    ]
    for pat in ec:
        m = re.search(pat, tail, re.IGNORECASE)
        if m:
            v = m.group(1).lower().replace(" ", "_").replace("-", "_")
            if v in {"not_violated", "no_violation"}:
                v = "not_violated"
            else:
                v = "violated"
            if v in label_set:
                return v

    # ── (b) Position-weighted mention vote ───────────────────────────────────
    body_len = max(1, len(raw_answer))
    scores: dict[str, float] = defaultdict(float)

    # CaseHOLD
    for m in re.finditer(r"\bholding[\s_-]?([0-4])\b", raw_answer, re.IGNORECASE):
        cand = f"holding_{m.group(1)}"
        if cand not in label_set:
            continue
        rel_pos = m.start() / body_len  # 0 = early, 1 = late
        scores[cand] += 1.0 + 3.0 * rel_pos

    # ECtHR
    for m in re.finditer(r"\b(violated|not[\s_-]?violated|no[\s_-]?violation)\b",
                         raw_answer, re.IGNORECASE):
        v = m.group(1).lower().replace(" ", "_").replace("-", "_")
        if v in {"not_violated", "no_violation"}:
            v = "not_violated"
        else:
            v = "violated"
        if v not in label_set:
            continue
        rel_pos = m.start() / body_len
        scores[v] += 1.0 + 3.0 * rel_pos

    if not scores:
        return None
    # Map back to canonical (preserving case from label_space)
    top = max(scores, key=scores.get)
    for orig in label_space:
        if orig.lower() == top:
            return orig
    return top


def _ecthr_norm(m: re.Match) -> str:
    s = m.group(1).lower().replace(" ", "_").replace("-", "_")
    if s in {"not_violated", "no_violation"}:
        return "not_violated"
    return "violated"


def _labels_are_near_prefixes(label_space: list[str]) -> bool:
    """True if any label is a substring of another (e.g. 'holding_2' / 'holding_24')."""
    lower = [l.lower() for l in label_space]
    for i, a in enumerate(lower):
        for j, b in enumerate(lower):
            if i != j and a in b:
                return True
    return False


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

    # Build the label set actually present in y_true ∪ y_pred so sklearn
    # doesn't crash when failed pipelines emit empty strings or labels outside
    # the canonical label_space (e.g. when the API returns an error and we
    # store predicted_label='').
    all_labels = sorted({str(x) for x in list(ground_truths) + list(predictions)})
    target_names_aligned = (
        label_names if label_names and len(label_names) == len(all_labels) else all_labels
    )

    report = classification_report(
        ground_truths,
        predictions,
        labels=all_labels,
        target_names=target_names_aligned,
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
