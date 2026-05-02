"""
Visualization utilities for experiment results.

Generates:
  1. Bar chart: accuracy / F1 comparison across pipelines
  2. Scatter plot: accuracy vs. latency (cost-performance tradeoff)
  3. Stacked bar: error taxonomy per pipeline
  4. Box plot: per-example latency distribution
  5. Heatmap: confusion matrix per pipeline
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

from utils.helpers import ensure_dir

sns.set_theme(style="whitegrid", palette="muted")

PIPELINE_COLORS = {
    "vector_rag": "#4C72B0",
    "graph_rag": "#55A868",
    "single_agent": "#C44E52",
    "multi_agent": "#8172B2",
}
PIPELINE_LABELS = {
    "vector_rag": "P1: Vector RAG",
    "graph_rag": "P2: Graph RAG",
    "single_agent": "P3: Single Agent",
    "multi_agent": "P4: Multi-Agent",
}


# ── 1. Accuracy / F1 bar chart ───────────────────────────���────────────────────

def plot_metrics_bar(
    metrics: dict[str, dict],
    output_path: str,
    title: str = "Pipeline Performance Comparison",
) -> None:
    pipelines = list(metrics.keys())
    acc = [metrics[p].get("accuracy", 0) for p in pipelines]
    f1_mac = [metrics[p].get("f1_macro", 0) for p in pipelines]
    f1_wt = [metrics[p].get("f1_weighted", 0) for p in pipelines]

    x = np.arange(len(pipelines))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width, acc, width, label="Accuracy", color="#4C72B0")
    bars2 = ax.bar(x, f1_mac, width, label="F1 (Macro)", color="#55A868")
    bars3 = ax.bar(x + width, f1_wt, width, label="F1 (Weighted)", color="#C44E52")

    ax.set_xticks(x)
    ax.set_xticklabels([PIPELINE_LABELS.get(p, p) for p in pipelines], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    _annotate_bars(ax, [bars1, bars2, bars3])
    fig.tight_layout()
    ensure_dir(os.path.dirname(output_path) or ".")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ── 2. Accuracy vs. latency scatter ───────────────────────────────────────────

def plot_accuracy_vs_latency(
    metrics: dict[str, dict],
    output_path: str,
    title: str = "Accuracy vs. Latency Trade-off",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for pname, m in metrics.items():
        ax.scatter(
            m.get("avg_latency_s", 0),
            m.get("accuracy", 0),
            s=150,
            color=PIPELINE_COLORS.get(pname, "gray"),
            label=PIPELINE_LABELS.get(pname, pname),
            zorder=5,
        )
        ax.annotate(
            PIPELINE_LABELS.get(pname, pname),
            (m.get("avg_latency_s", 0), m.get("accuracy", 0)),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
        )

    ax.set_xlabel("Avg Latency (s)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ── 3. Error taxonomy stacked bar ─────────────────────────────────────────────

def plot_error_taxonomy(
    errors: dict[str, dict[str, int]],
    output_path: str,
    title: str = "Error Taxonomy by Pipeline",
) -> None:
    from evaluation.error_taxonomy import ErrorType

    all_error_types = [e.value for e in ErrorType]
    pipelines = list(errors.keys())

    data = {et: [errors[p].get(et, 0) for p in pipelines] for et in all_error_types}
    total = [sum(errors[p].values()) for p in pipelines]

    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(pipelines))
    palette = sns.color_palette("tab10", len(all_error_types))

    for i, et in enumerate(all_error_types):
        counts = np.array(data[et])
        fracs = np.where(np.array(total) > 0, counts / np.array(total), 0)
        ax.bar(
            pipelines, fracs, bottom=bottom,
            color=palette[i], label=et.replace("_", " ").title()
        )
        bottom += fracs

    ax.set_xticklabels([PIPELINE_LABELS.get(p, p) for p in pipelines], rotation=15, ha="right")
    ax.set_ylabel("Fraction of Examples")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 4. Latency box plot ───────────────────────────────────────────────────────

def plot_latency_boxplot(
    pipeline_results: dict[str, list[dict]],
    output_path: str,
    title: str = "Per-Example Latency Distribution",
) -> None:
    data = {
        PIPELINE_LABELS.get(p, p): [r.get("latency_s", 0) for r in results]
        for p, results in pipeline_results.items()
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data.values(), labels=data.keys(), patch_artist=True)
    ax.set_ylabel("Latency (s)")
    ax.set_title(title)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ── 5. Token usage bar ───────────────────��────────────────────────────────────

def plot_token_usage(
    metrics: dict[str, dict],
    output_path: str,
    title: str = "Average Token Usage per Query",
) -> None:
    pipelines = list(metrics.keys())
    in_toks = [metrics[p].get("avg_input_tokens", 0) for p in pipelines]
    out_toks = [metrics[p].get("avg_output_tokens", 0) for p in pipelines]

    x = np.arange(len(pipelines))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, in_toks, width, label="Input tokens", color="#4C72B0")
    ax.bar(x + width / 2, out_toks, width, label="Output tokens", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels([PIPELINE_LABELS.get(p, p) for p in pipelines], rotation=15, ha="right")
    ax.set_ylabel("Avg Tokens / Query")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ── Batch generate all plots ───────────────────────────────��──────────────────

def generate_all_plots(
    metrics: dict[str, dict],
    errors: dict[str, dict],
    pipeline_results: dict[str, list[dict]],
    output_dir: str,
    dataset_name: str = "ledgar",
) -> list[str]:
    """
    Generate all standard plots and save them to output_dir.
    Returns list of saved file paths.
    """
    ensure_dir(output_dir)
    saved: list[str] = []

    def _path(name: str) -> str:
        return os.path.join(output_dir, f"{dataset_name}_{name}.png")

    plot_metrics_bar(metrics, _path("metrics_bar"), title=f"{dataset_name.upper()} – Pipeline Metrics")
    saved.append(_path("metrics_bar"))

    plot_accuracy_vs_latency(metrics, _path("accuracy_vs_latency"))
    saved.append(_path("accuracy_vs_latency"))

    if errors:
        plot_error_taxonomy(errors, _path("error_taxonomy"))
        saved.append(_path("error_taxonomy"))

    if pipeline_results:
        plot_latency_boxplot(pipeline_results, _path("latency_boxplot"))
        saved.append(_path("latency_boxplot"))

        plot_token_usage(metrics, _path("token_usage"))
        saved.append(_path("token_usage"))

    return saved


# ── Helpers ──────────────────────────────────────────────────────────��────────

def _annotate_bars(ax: plt.Axes, bar_groups) -> None:
    for bars in bar_groups:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.annotate(
                    f"{h:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
