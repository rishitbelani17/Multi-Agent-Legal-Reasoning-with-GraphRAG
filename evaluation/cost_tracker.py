"""
Cost & latency tracker.

Tracks token usage and wall-clock time across pipeline runs and computes
estimated API costs (using Claude Haiku pricing as default).
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Claude model pricing per 1M tokens (as of early 2026, adjust as needed)
_PRICING_PER_1M: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-6":         {"input": 3.00, "output": 15.00},
    "claude-opus-4-6":           {"input": 15.00, "output": 75.00},
}
_DEFAULT_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class CallRecord:
    pipeline: str
    query_id: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    model: str = _DEFAULT_MODEL
    metadata: dict[str, Any] = field(default_factory=dict)


class CostTracker:
    """
    Accumulates token counts and latencies across experiment runs.

    Usage
    -----
    tracker = CostTracker(model="claude-haiku-4-5-20251001")
    tracker.record(pipeline="multi_agent_graph_rag", query_id="q1",
                   input_tokens=500, output_tokens=200, latency_s=1.2)
    print(tracker.summary())
    """

    def __init__(self, model: str = _DEFAULT_MODEL):
        self.model = model
        self._records: list[CallRecord] = []

    def record(
        self,
        pipeline: str,
        query_id: str,
        input_tokens: int,
        output_tokens: int,
        latency_s: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._records.append(
            CallRecord(
                pipeline=pipeline,
                query_id=query_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_s=latency_s,
                model=self.model,
                metadata=metadata or {},
            )
        )

    def record_from_result(self, result: dict[str, Any], query_id: str) -> None:
        """Convenience method – extract fields directly from a pipeline result dict."""
        self.record(
            pipeline=result.get("pipeline", "unknown"),
            query_id=query_id,
            input_tokens=result.get("input_tokens", 0),
            output_tokens=result.get("output_tokens", 0),
            latency_s=result.get("latency_s", 0.0),
        )

    # ── Summaries ─────────────────────────────────────────────────────────────

    def cost_usd(self, record: CallRecord) -> float:
        pricing = _PRICING_PER_1M.get(record.model, _PRICING_PER_1M[_DEFAULT_MODEL])
        return (
            record.input_tokens * pricing["input"] / 1_000_000
            + record.output_tokens * pricing["output"] / 1_000_000
        )

    def summary(self) -> dict[str, Any]:
        """Return per-pipeline and overall cost/latency summary."""
        by_pipeline: dict[str, list[CallRecord]] = defaultdict(list)
        for r in self._records:
            by_pipeline[r.pipeline].append(r)

        result: dict[str, Any] = {}
        for pipeline, records in by_pipeline.items():
            in_toks = [r.input_tokens for r in records]
            out_toks = [r.output_tokens for r in records]
            lats = [r.latency_s for r in records]
            costs = [self.cost_usd(r) for r in records]
            result[pipeline] = {
                "n_calls": len(records),
                "total_input_tokens": sum(in_toks),
                "total_output_tokens": sum(out_toks),
                "avg_input_tokens": float(np.mean(in_toks)),
                "avg_output_tokens": float(np.mean(out_toks)),
                "avg_latency_s": float(np.mean(lats)),
                "p50_latency_s": float(np.percentile(lats, 50)),
                "p95_latency_s": float(np.percentile(lats, 95)),
                "total_cost_usd": float(sum(costs)),
                "avg_cost_usd": float(np.mean(costs)),
            }

        # Overall
        all_costs = [self.cost_usd(r) for r in self._records]
        result["_total"] = {
            "n_calls": len(self._records),
            "total_cost_usd": float(sum(all_costs)),
        }
        return result

    def to_dataframe(self):
        """Export all records as a pandas DataFrame."""
        import pandas as pd  # type: ignore
        rows = [
            {
                "pipeline": r.pipeline,
                "query_id": r.query_id,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "latency_s": r.latency_s,
                "cost_usd": self.cost_usd(r),
                "model": r.model,
            }
            for r in self._records
        ]
        return pd.DataFrame(rows)
