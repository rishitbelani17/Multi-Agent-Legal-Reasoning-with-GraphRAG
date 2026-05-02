#!/usr/bin/env python3
"""
Multi-Agent Legal Reasoning with GraphRAG
==========================================
Rishit Belani & Sushmita Korishetty – Columbia University, Spring 2026

Entry point for running the full experiment suite.

Usage
-----
# Run all 4 pipelines on LEDGAR (default)
python main.py

# Run on CaseHOLDER, only pipelines 1 and 4
python main.py --dataset caseholder --pipelines vector_rag multi_agent

# Run on all datasets
python main.py --dataset all

# Quick smoke-test (tiny subset, 1 pipeline)
python main.py --dataset ledgar --pipelines vector_rag --subset 5

Environment
-----------
Set ANTHROPIC_API_KEY in a .env file or as an environment variable.
"""

from __future__ import annotations

import argparse
import logging
import sys

import config
from experiments.run_experiments import ExperimentRunner
from experiments.visualize import generate_all_plots
from utils.helpers import ensure_dir, load_json, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-Agent Legal Reasoning with GraphRAG – experiment runner"
    )
    p.add_argument(
        "--dataset",
        default="caseholder",
        choices=["caseholder", "ecthr", "ledgar", "all"],
        help="Dataset to evaluate on (default: caseholder, the proposal's primary benchmark)",
    )
    p.add_argument(
        "--pipelines",
        nargs="+",
        default=[
            "single_llm",
            "vector_rag",
            "multi_agent_vector",
            "multi_agent",
        ],
        choices=[
            "single_llm",
            "vector_rag",
            "graph_rag",
            "single_agent",
            "multi_agent_vector",
            "multi_agent",
        ],
        help=(
            "Which pipelines to run. Default = the proposal's four ablation conditions: "
            "single_llm (no retrieval), vector_rag (single-agent + vector RAG), "
            "multi_agent_vector (multi-agent + vector RAG), multi_agent (multi-agent + GraphRAG)."
        ),
    )
    p.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Override dataset subset size for quick testing",
    )
    p.add_argument(
        "--model",
        default=config.LLM_MODEL,
        help=f"Anthropic model ID (default: {config.LLM_MODEL})",
    )
    p.add_argument(
        "--results-dir",
        default=config.RESULTS_DIR,
        help="Directory to save results (default: ./results)",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )
    p.add_argument(
        "--log-level",
        default=config.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def run_dataset(
    dataset_name: str,
    args: argparse.Namespace,
) -> dict:
    # Apply subset override if provided
    if args.subset is not None:
        if dataset_name == "ledgar":
            config.LEDGAR_SUBSET_SIZE = args.subset
        elif dataset_name == "caseholder":
            config.CASEHOLDER_SUBSET_SIZE = args.subset
        elif dataset_name == "ecthr":
            config.ECTHR_SUBSET_SIZE = args.subset

    runner = ExperimentRunner(
        dataset_name=dataset_name,
        pipelines=args.pipelines,
        results_dir=args.results_dir,
        model=args.model,
    )
    result = runner.run()

    if not args.no_plots:
        plots_dir = ensure_dir(f"{args.results_dir}/plots/{dataset_name}")
        # Load raw results for the box plot
        raw_prefix = result["output_prefix"]
        try:
            raw_results = load_json(f"{raw_prefix}_raw_results.json")
        except FileNotFoundError:
            raw_results = {}

        saved_plots = generate_all_plots(
            metrics=result["metrics"],
            errors=result["errors"],
            pipeline_results=raw_results,
            output_dir=plots_dir,
            dataset_name=dataset_name,
        )
        logger.info("Plots saved: %s", saved_plots)

    return result


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    if not config.ANTHROPIC_API_KEY:
        logger.error(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to a .env file or export it as an environment variable."
        )
        return 1

    datasets = (
        ["caseholder", "ecthr", "ledgar"]
        if args.dataset == "all"
        else [args.dataset]
    )

    for ds in datasets:
        logger.info("Running experiments on dataset: %s", ds)
        run_dataset(ds, args)

    logger.info("All experiments complete. Results in: %s", args.results_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
