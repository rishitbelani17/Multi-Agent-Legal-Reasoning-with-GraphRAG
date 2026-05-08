"""
Automated qualitative error analysis on a saved ExperimentRunner run.

Reads the saved ``*_raw_results.json`` and writes a markdown report that:
  - Lists per-pipeline accuracy / F1
  - Identifies multi-agent failure cases
  - Classifies each failure into one of:
        retrieval_miss     – gold doc not in retrieved chunks at all
        plaintiff_wrong    – plaintiff's POSITION already disagrees with gold
        defense_flipped    – plaintiff was right, but defense's COUNTER_POSITION = gold-disagreer
                             AND judge ended up agreeing with defense
        judge_overrode     – plaintiff was right, defense raised a wrong counter,
                             but judge picked yet a third holding
        parser_glitch      – judge's reasoning suggests the right answer but
                             the parsed FINAL ANSWER is something else

Usage
-----
  python3 -m tools.analyze_errors ./results/caseholder_20260505_154054
  python3 -m tools.analyze_errors ./results/caseholder_20260505_154054 --pipeline multi_agent
  python3 -m tools.analyze_errors ./results/caseholder_20260505_154054 --top 10

Outputs
-------
  <prefix>_error_analysis.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Any


# ── Failure classification heuristics ────────────────────────────────────────

def _extract_position(text: str, kind: str = "POSITION") -> str | None:
    """Pull the holding label out of a `POSITION:` or `COUNTER_POSITION:` line."""
    if not text:
        return None
    m = re.search(rf"{kind}\s*[:\-]\s*([^\n]+)", text, re.IGNORECASE)
    if not m:
        return None
    val = m.group(1).strip().strip(" .,'\"`*()")
    m2 = re.search(r"holding[\s_-]?([0-4])", val, re.IGNORECASE)
    if m2:
        return f"holding_{m2.group(1)}"
    m2 = re.search(r"\b(violated|not[\s_-]?violated)\b", val, re.IGNORECASE)
    if m2:
        return m2.group(1).lower().replace(" ", "_").replace("-", "_")
    return val.lower()


def _scan_judge_reasoning_for_label(text: str, candidates: list[str]) -> str | None:
    """Return the candidate label most frequently named in the Judge's body
    (a heuristic for catching parser_glitch where the reasoning clearly
    favours one holding but the FINAL ANSWER line contradicts it)."""
    if not text:
        return None
    body = re.split(r"FINAL\s*ANSWER\s*[:\-]", text, flags=re.IGNORECASE)[0]
    counts = Counter()
    for cand in candidates:
        counts[cand] = len(re.findall(rf"\b{re.escape(cand)}\b", body, re.IGNORECASE))
    if not counts or max(counts.values()) == 0:
        return None
    top, n = counts.most_common(1)[0]
    runners_up = [c for c, k in counts.items() if c != top and k == n]
    return top if not runners_up else None


def _retrieved_doc_ids(chunks: list[dict]) -> set[str]:
    out = set()
    for c in chunks:
        d = c.get("doc_id")
        if not d and c.get("chunk_id", ""):
            head, _, tail = c["chunk_id"].rpartition("_c")
            d = head if tail.isdigit() else c["chunk_id"]
        if d:
            out.add(d)
    return out


def classify_failure(
    record: dict[str, Any],
    candidates: list[str],
) -> tuple[str, dict[str, Any]]:
    """Return (failure_class, details) for one failed prediction."""
    true_label = record.get("true_label", "")
    pred_label = record.get("predicted_label", "")
    doc_id = record.get("doc_id", "")
    chunks = record.get("retrieved_chunks", [])

    # 1. retrieval_miss
    retrieved = _retrieved_doc_ids(chunks)
    if chunks and doc_id not in retrieved:
        return "retrieval_miss", {
            "retrieved_docs": sorted(retrieved)[:5],
            "gold_doc": doc_id,
        }

    # Inspect the agent transcript if present
    rounds = record.get("round_logs", [])
    plaintiff_pos: list[str] = []
    defense_pos: list[str] = []
    judge_text = ""
    for entry in rounds:
        stage = entry.get("stage", "")
        content = entry.get("content", "")
        if stage.startswith("plaintiff"):
            pos = _extract_position(content, "POSITION")
            if pos:
                plaintiff_pos.append(pos)
        elif stage.startswith("defense"):
            pos = _extract_position(content, "COUNTER_POSITION")
            if pos:
                defense_pos.append(pos)
        elif stage == "judge":
            judge_text = content

    # 2. parser_glitch
    judge_body_label = _scan_judge_reasoning_for_label(judge_text, candidates)
    if judge_body_label and judge_body_label != pred_label and judge_body_label == true_label:
        return "parser_glitch", {
            "judge_body_favored": judge_body_label,
            "parsed_as": pred_label,
        }

    final_plaintiff = plaintiff_pos[-1] if plaintiff_pos else None
    final_defense = defense_pos[-1] if defense_pos else None

    # 3. plaintiff_wrong
    if final_plaintiff and final_plaintiff != true_label:
        return "plaintiff_wrong", {
            "plaintiff_position": final_plaintiff,
            "defense_position": final_defense,
            "judge_picked": pred_label,
        }

    # 4. defense_flipped
    if (
        final_plaintiff == true_label
        and final_defense
        and pred_label == final_defense
    ):
        return "defense_flipped", {
            "plaintiff_position": final_plaintiff,
            "defense_position": final_defense,
            "judge_picked": pred_label,
        }

    # 5. judge_overrode
    if (
        final_plaintiff == true_label
        and final_defense
        and pred_label not in {final_plaintiff, final_defense}
    ):
        return "judge_overrode", {
            "plaintiff_position": final_plaintiff,
            "defense_position": final_defense,
            "judge_picked": pred_label,
        }

    return "unclassified", {
        "plaintiff_position": final_plaintiff,
        "defense_position": final_defense,
        "judge_picked": pred_label,
    }


# ── Main report generation ───────────────────────────────────────────────────

def build_report(
    raw_results: dict[str, list[dict]],
    metrics: dict[str, dict],
    pipeline: str,
    top_k: int = 5,
) -> str:
    if pipeline not in raw_results:
        return f"# Error analysis\n\nPipeline `{pipeline}` not found in results.\n"

    results = raw_results[pipeline]
    failures = [r for r in results if r.get("predicted_label") != r.get("true_label")]
    n_total = len(results)
    n_fail = len(failures)
    candidates = sorted({r.get("true_label", "") for r in results} |
                        {r.get("predicted_label", "") for r in results})

    classes = Counter()
    classified: list[tuple[str, dict, dict]] = []
    for r in failures:
        cls, det = classify_failure(r, candidates)
        classes[cls] += 1
        classified.append((cls, det, r))

    lines: list[str] = []
    lines.append(f"# Error Analysis — `{pipeline}`")
    lines.append("")
    lines.append(f"- Total examples: **{n_total}**")
    lines.append(f"- Failures: **{n_fail}** "
                 f"({100 * n_fail / n_total:.1f}%)" if n_total else "")
    lines.append("")
    if pipeline in metrics:
        m = metrics[pipeline]
        lines.append("## Headline metrics")
        lines.append("")
        lines.append(
            f"- accuracy = {m.get('accuracy', 0):.3f}, "
            f"F1-macro = {m.get('f1_macro', 0):.3f}, "
            f"F1-weighted = {m.get('f1_weighted', 0):.3f}"
        )
        ret = m.get("retrieval", {})
        if ret:
            lines.append(
                f"- retrieval P/R/MRR = {ret.get('mean_precision', 0):.2f} / "
                f"{ret.get('mean_recall', 0):.2f} / {ret.get('mean_mrr', 0):.2f}"
            )
        sig = m.get("significance", {})
        if sig:
            lines.append(
                f"- vs `{sig.get('baseline')}`: "
                f"Δacc = {sig.get('delta_accuracy', 0):+.3f} "
                f"(95 % CI [{sig.get('ci_low', 0):+.3f}, {sig.get('ci_high', 0):+.3f}], "
                f"bootstrap p = {sig.get('p_value', 1):.3f}, "
                f"McNemar p = {sig.get('mcnemar_p', 1):.3f})"
            )
        lines.append("")

    lines.append("## Failure classification")
    lines.append("")
    if not classes:
        lines.append("_No failures._")
    else:
        for cls, n in classes.most_common():
            pct = 100 * n / n_fail if n_fail else 0
            lines.append(f"- `{cls}`: **{n}** ({pct:.1f}% of failures)")
    lines.append("")

    lines.append("## Selected failure cases")
    lines.append("")
    for cls, det, r in classified[:top_k]:
        lines.append(f"### {r.get('doc_id', '?')} — `{cls}`")
        lines.append("")
        lines.append(f"- predicted: **{r.get('predicted_label')}**, true: **{r.get('true_label')}**")
        for k, v in det.items():
            lines.append(f"- {k}: `{v}`")
        # Short snippet of the judge body to document why
        judge_text = ""
        for entry in r.get("round_logs", []):
            if entry.get("stage") == "judge":
                judge_text = entry.get("content", "")
        if judge_text:
            snippet = judge_text.replace("\n", " ").strip()
            if len(snippet) > 350:
                snippet = snippet[:350] + "…"
            lines.append("")
            lines.append("> Judge: " + snippet)
        lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    if classes.get("retrieval_miss", 0) >= max(1, n_fail // 4):
        lines.append(
            "- Retrieval is the dominant failure mode. Consider raising "
            "`GRAPH_MAX_HOPS`, `VECTOR_TOP_K`, or expanding sub-query decomposition."
        )
    if classes.get("defense_flipped", 0) >= max(1, n_fail // 4):
        lines.append(
            "- Defense rhetoric is flipping correct Plaintiff positions. "
            "Strengthen the Judge's evidence-anchor instruction or reduce "
            "`N_DEBATE_ROUNDS` to 1."
        )
    if classes.get("parser_glitch", 0) >= max(1, n_fail // 8):
        lines.append(
            "- Parser glitches detected: tighten the Judge's strict output "
            "format and verify the `parse_answer` anchored regex covers all label types."
        )
    if classes.get("plaintiff_wrong", 0) >= max(1, n_fail // 4):
        lines.append(
            "- Plaintiff is opening with the wrong position frequently. "
            "Improve the Plaintiff prompt's evidence-grounding step."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate a qualitative error-analysis report from saved results."
    )
    p.add_argument("results_prefix",
                   help="Path prefix used by ExperimentRunner, e.g. ./results/caseholder_20260505_154054")
    p.add_argument("--pipeline", default="multi_agent",
                   help="Pipeline to analyse (default: multi_agent)")
    p.add_argument("--top", type=int, default=5,
                   help="Number of failure cases to include verbatim")
    p.add_argument("--out", default=None,
                   help="Output markdown path (default: <prefix>_error_analysis.md)")
    args = p.parse_args()

    raw_path = f"{args.results_prefix}_raw_results.json"
    metrics_path = f"{args.results_prefix}_metrics.json"
    if not os.path.exists(raw_path):
        print(f"Cannot find {raw_path}", file=sys.stderr)
        return 1

    with open(raw_path) as f:
        raw = json.load(f)
    metrics: dict = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    report = build_report(raw, metrics, args.pipeline, top_k=args.top)
    out_path = args.out or f"{args.results_prefix}_error_analysis.md"
    with open(out_path, "w") as f:
        f.write(report)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
