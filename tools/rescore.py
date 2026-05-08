"""
Offline re-scorer for ExperimentRunner runs.

Re-parses every saved row in ``*_raw_results.json`` using the current
``evaluation.metrics.parse_answer`` and recomputes the metrics file. No
LLM calls, no network, no cost.

This is the cheap rescue path for runs that suffered a parser glitch (e.g.
the Judge hit max_tokens before reaching FINAL ANSWER). The heuristic
"body-vote" fallback in parse_answer can recover a label from a truncated
Judge transcript by scanning every holding_X mention in the reasoning,
weighting later mentions more heavily.

Usage
-----
  python3 -m tools.rescore ./results/caseholder_20260506_031041
  python3 -m tools.rescore ./results/caseholder_20260506_031041 --suffix _rescored

Outputs (suffix defaults to '_rescored')
----------------------------------------
  <prefix>_metrics_rescored.json
  <prefix>_raw_results_rescored.json   (with new predicted_label values)
  Side-by-side diff printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any

from evaluation.metrics import parse_answer, compute_metrics


def _label_space(rows: list[dict]) -> list[str]:
    """Build the canonical label space from the ground truths of a pipeline."""
    return sorted({r.get("true_label", "") for r in rows if r.get("true_label")})


def _judge_text(row: dict) -> str:
    """Return the saved Judge transcript content, or '' if not present."""
    for entry in row.get("round_logs") or []:
        if entry.get("stage") == "judge":
            return entry.get("content", "") or ""
    # For non-debate pipelines the answer field is the full LLM response.
    return row.get("answer", "") or ""


def rescore_pipeline(
    pipeline_name: str,
    rows: list[dict],
    dataset: str = "caseholder",
) -> tuple[list[dict], dict[str, Any], dict[str, int]]:
    """Return (new_rows, new_metrics, change_summary)."""
    label_space = _label_space(rows)
    new_rows: list[dict] = []
    changed = 0
    flips_to_correct = 0
    flips_to_wrong = 0
    new_correct = 0
    old_correct = 0

    for r in rows:
        old_pred = r.get("predicted_label", "")
        truth = r.get("true_label", "")
        # Re-parse from the FULL judge body (not just the .answer field, which
        # for the multi_agent_* pipelines is the raw judge output).
        text = _judge_text(r)
        new_pred = parse_answer(text, label_space) if text else old_pred

        new_r = dict(r)
        new_r["predicted_label"] = new_pred
        new_r["predicted_label_old"] = old_pred
        new_rows.append(new_r)

        if new_pred != old_pred:
            changed += 1
            if new_pred == truth and old_pred != truth:
                flips_to_correct += 1
            elif old_pred == truth and new_pred != truth:
                flips_to_wrong += 1

        if new_pred == truth:
            new_correct += 1
        if old_pred == truth:
            old_correct += 1

    metrics = compute_metrics(
        [r["predicted_label"] for r in new_rows],
        [r["true_label"] for r in new_rows],
        dataset=dataset,
    )

    summary = {
        "n_rows": len(rows),
        "labels_changed": changed,
        "flips_to_correct": flips_to_correct,
        "flips_to_wrong": flips_to_wrong,
        "old_correct": old_correct,
        "new_correct": new_correct,
        "old_accuracy": old_correct / len(rows) if rows else 0.0,
        "new_accuracy": new_correct / len(rows) if rows else 0.0,
    }
    return new_rows, metrics, summary


def main() -> int:
    p = argparse.ArgumentParser(
        description="Re-parse saved Judge transcripts and recompute metrics — no LLM calls."
    )
    p.add_argument("results_prefix",
                   help="ExperimentRunner prefix, e.g. ./results/caseholder_20260506_031041")
    p.add_argument("--dataset", default="caseholder",
                   help="caseholder|ledgar|ecthr (default: caseholder)")
    p.add_argument("--suffix", default="_rescored",
                   help="Suffix for output files (default: _rescored)")
    args = p.parse_args()

    raw_path = f"{args.results_prefix}_raw_results.json"
    if not os.path.exists(raw_path):
        print(f"Cannot find {raw_path}", file=sys.stderr)
        return 1

    with open(raw_path) as f:
        raw: dict[str, list[dict]] = json.load(f)

    new_raw: dict[str, list[dict]] = {}
    new_metrics: dict[str, dict] = {}

    print(f"Rescoring {raw_path}\n")
    print(f"{'pipeline':<24}  {'old':>6}  {'new':>6}  {'+correct':>8}  {'-correct':>8}  changed")
    print("-" * 70)

    for pipeline_name, rows in raw.items():
        rescored, metrics, summ = rescore_pipeline(
            pipeline_name, rows, dataset=args.dataset
        )
        new_raw[pipeline_name] = rescored
        new_metrics[pipeline_name] = metrics
        print(
            f"{pipeline_name:<24}  "
            f"{summ['old_accuracy']:>6.3f}  "
            f"{summ['new_accuracy']:>6.3f}  "
            f"{summ['flips_to_correct']:>8d}  "
            f"{summ['flips_to_wrong']:>8d}  "
            f"{summ['labels_changed']:>5d}"
        )

    out_metrics = f"{args.results_prefix}_metrics{args.suffix}.json"
    out_raw = f"{args.results_prefix}_raw_results{args.suffix}.json"
    with open(out_metrics, "w") as f:
        json.dump(new_metrics, f, indent=2)
    with open(out_raw, "w") as f:
        json.dump(new_raw, f, indent=2)
    print()
    print(f"Wrote {out_metrics}")
    print(f"Wrote {out_raw}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
