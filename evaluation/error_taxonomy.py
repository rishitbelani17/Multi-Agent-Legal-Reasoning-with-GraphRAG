"""
Error taxonomy for legal reasoning failures.

Classifies model errors into interpretable categories, enabling
qualitative analysis of where each pipeline fails.

Error categories (inspired by the proposal and legal AI literature):
  1. RETRIEVAL_MISS      – Correct answer required evidence not retrieved
  2. REASONING_GAP       – Evidence retrieved but logic is flawed
  3. WRONG_LABEL         – Correct reasoning but wrong final label (e.g., off-by-one)
  4. OVERCONFIDENT       – Wrong answer stated with High confidence
  5. UNDERCONFIDENT      – Correct answer but marked Low confidence
  6. CITATION_HALLUC     – Cites passages that don't support the claim
  7. CONTEXT_OVERFLOW    – Answer ignores relevant retrieved context
  8. AMBIGUOUS_ANSWER    – Answer is unclear / could map to multiple labels
  9. CORRECT             – No error (control category)
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any


class ErrorType(str, Enum):
    RETRIEVAL_MISS = "retrieval_miss"
    REASONING_GAP = "reasoning_gap"
    WRONG_LABEL = "wrong_label"
    OVERCONFIDENT = "overconfident"
    UNDERCONFIDENT = "underconfident"
    CITATION_HALLUC = "citation_hallucination"
    CONTEXT_OVERFLOW = "context_overflow"
    AMBIGUOUS_ANSWER = "ambiguous_answer"
    CORRECT = "correct"


class ErrorTaxonomy:
    """
    Classifies a single model output into an error category.

    Heuristic rules are applied in order; the first matching rule wins.
    For more precise classification in production, pass an LLM judge.
    """

    def classify(
        self,
        result: dict[str, Any],
        true_label: str | int,
        label_space: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Classify the error type for a pipeline result.

        Parameters
        ----------
        result:      Pipeline result dict (must have 'answer', 'retrieved_chunks').
        true_label:  Ground-truth label.
        label_space: List of valid label strings.

        Returns
        -------
        dict with 'error_type', 'confidence', and 'explanation'.
        """
        answer = result.get("answer", "")
        pred_label = result.get("predicted_label", "")
        chunks = result.get("retrieved_chunks", [])
        confidence = _extract_confidence(answer)
        is_correct = str(pred_label).lower() == str(true_label).lower()

        if is_correct:
            if confidence == "Low":
                return _make(ErrorType.UNDERCONFIDENT, confidence,
                             "Correct prediction but expressed low confidence.")
            return _make(ErrorType.CORRECT, confidence, "Prediction is correct.")

        # Wrong prediction – diagnose why
        if not chunks:
            return _make(ErrorType.RETRIEVAL_MISS, confidence,
                         "No chunks retrieved; evidence was unavailable.")

        if _has_citation_hallucination(answer, chunks):
            return _make(ErrorType.CITATION_HALLUC, confidence,
                         "Answer cites passage numbers that don't match retrieved context.")

        if confidence == "High":
            return _make(ErrorType.OVERCONFIDENT, confidence,
                         "Wrong answer stated with high confidence.")

        if _is_ambiguous_answer(answer, label_space):
            return _make(ErrorType.AMBIGUOUS_ANSWER, confidence,
                         "Predicted label is ambiguous or matches multiple categories.")

        if _context_not_used(answer, chunks):
            return _make(ErrorType.CONTEXT_OVERFLOW, confidence,
                         "Retrieved context appears to have been ignored in the answer.")

        # Default: retrieved evidence was present but reasoning was flawed
        return _make(ErrorType.REASONING_GAP, confidence,
                     "Evidence was retrieved but the logical argument is incorrect.")

    def batch_classify(
        self,
        results: list[dict[str, Any]],
        true_labels: list[str | int],
        label_space: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return [
            self.classify(r, t, label_space)
            for r, t in zip(results, true_labels)
        ]

    @staticmethod
    def summary(classifications: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in classifications:
            et = c["error_type"]
            counts[et] = counts.get(et, 0) + 1
        return counts


# ── Standalone function (convenience) ────────────────────────────────────────

def classify_error(
    result: dict[str, Any],
    true_label: str | int,
    label_space: list[str] | None = None,
) -> dict[str, Any]:
    return ErrorTaxonomy().classify(result, true_label, label_space)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make(error_type: ErrorType, confidence: str, explanation: str) -> dict[str, Any]:
    return {
        "error_type": error_type.value,
        "confidence": confidence,
        "explanation": explanation,
    }


def _extract_confidence(answer: str) -> str:
    m = re.search(r"CONFIDENCE\s*:\s*(High|Medium|Low)", answer, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    return "Unknown"


def _has_citation_hallucination(answer: str, chunks: list[dict]) -> bool:
    """
    Check if the answer references passage numbers that don't exist.
    Simple heuristic: extract [N] references and verify N <= len(chunks).
    """
    cited = re.findall(r"\[(\d+)\]", answer)
    n_chunks = len(chunks)
    for c in cited:
        if int(c) > n_chunks:
            return True
    return False


def _is_ambiguous_answer(answer: str, label_space: list[str] | None) -> bool:
    """Check if multiple labels appear in the ANSWER: field."""
    if not label_space:
        return False
    m = re.search(r"ANSWER\s*:\s*(.+?)(?:\n|$)", answer, re.IGNORECASE)
    if not m:
        return False
    answer_str = m.group(1).lower()
    matches = sum(1 for label in label_space if label.lower() in answer_str)
    return matches > 1


def _context_not_used(answer: str, chunks: list[dict]) -> bool:
    """
    Heuristic: if the answer contains no passage references and
    is shorter than 50 words, the context may have been ignored.
    """
    has_citation = bool(re.search(r"\[\d+\]", answer))
    word_count = len(answer.split())
    return not has_citation and word_count < 50
