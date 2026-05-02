"""
ExperimentRunner – orchestrates the proposal's ablation experiments over a dataset.

Experimental conditions (proposal §2.1)
---------------------------------------
  single_llm           : Single LLM, no retrieval                    (Configuration 1)
  vector_rag           : Single-Agent + standard vector RAG          (Configuration 2)
  multi_agent_vector   : Multi-Agent debate + vector RAG             (Configuration 3)
  multi_agent          : Multi-Agent debate + GraphRAG (full system) (Configuration 4)

Two additional pipelines are also exposed for diagnostic use:
  graph_rag    : Single-call GraphRAG without agents
  single_agent : Single-agent + GraphRAG (no debate)

For each condition, the runner:
  1. Loops over all examples in the dataset.
  2. Calls the appropriate pipeline.
  3. Parses the predicted label.
  4. Records metrics, cost, and error taxonomy.
  5. Saves results to disk (JSON + CSV).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import config
from data.loader import DataLoader
from data.preprocessor import Preprocessor
from evaluation.cost_tracker import CostTracker
from evaluation.error_taxonomy import ErrorTaxonomy
from evaluation.metrics import aggregate_results, parse_answer
from graph.builder import GraphBuilder
from graph.retriever import GraphRetriever
from pipelines.baseline_rag import run_baseline_rag
from pipelines.graph_rag import run_graph_rag
from pipelines.multi_agent import run_multi_agent
from pipelines.multi_agent_vector import run_multi_agent_vector
from pipelines.single_agent import run_single_agent
from pipelines.single_llm import run_single_llm
from rag.vector_rag import VectorIndex
from utils.embedder import get_embedder
from utils.helpers import ensure_dir, save_json
from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Runs all 4 pipeline conditions on a given dataset split.

    Parameters
    ----------
    dataset_name:   "ledgar" | "caseholder" | "ecthr"
    pipelines:      Which pipelines to run (default: all 4).
    results_dir:    Output directory for JSON / CSV results.
    graph_cache:    Path to save/load the pre-built graph (avoids rebuilding).
    """

    # Default ablation set per proposal §2.1
    ALL_PIPELINES = [
        "single_llm",
        "vector_rag",
        "multi_agent_vector",
        "multi_agent",
    ]

    def __init__(
        self,
        dataset_name: str = "caseholder",
        pipelines: list[str] | None = None,
        results_dir: str = config.RESULTS_DIR,
        graph_cache: str | None = None,
        model: str = config.LLM_MODEL,
    ):
        self.dataset_name = dataset_name
        self.pipelines = pipelines or self.ALL_PIPELINES
        self.results_dir = results_dir
        self.graph_cache = graph_cache or os.path.join(
            config.GRAPH_CACHE_DIR, f"{dataset_name}_graph.pkl"
        )
        self.model = model

        ensure_dir(results_dir)
        ensure_dir(config.GRAPH_CACHE_DIR)

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self) -> dict[str, Any]:
        """
        Execute all configured experiments and return a results summary.
        """
        logger.info("=" * 60)
        logger.info("Starting experiments: dataset=%s, pipelines=%s",
                    self.dataset_name, self.pipelines)
        logger.info("=" * 60)

        # ── Setup shared components ───────────────────────────��───────────────
        llm = LLMClient(model=self.model)
        embedder = get_embedder()
        preprocessor = Preprocessor()
        loader = DataLoader(datasets=[self.dataset_name])

        logger.info("Loading dataset: %s …", self.dataset_name)
        docs = loader.load(self.dataset_name)
        label_names = list({d["label_str"] for d in docs})
        logger.info("Loaded %d examples. Label space size: %d",
                    len(docs), len(label_names))

        # ── Chunk documents ───────────────────────────────────────────────────
        logger.info("Chunking documents …")
        chunks = preprocessor.process_batch(docs)
        logger.info("Created %d chunks from %d documents.", len(chunks), len(docs))

        # ── Build or load vector index ────────────────────────────────────────
        vector_index = VectorIndex(embedder)
        vector_index.build(chunks)

        # ── Build or load knowledge graph ─────────────────────────────────────
        if os.path.exists(self.graph_cache):
            logger.info("Loading graph from cache: %s", self.graph_cache)
            graph = GraphBuilder.load(self.graph_cache)
        else:
            logger.info("Building knowledge graph …")
            builder = GraphBuilder(embedder)
            graph = builder.build(chunks)
            GraphBuilder.save(graph, self.graph_cache)
            stats = GraphBuilder.stats(graph)
            logger.info("Graph stats: %s", stats)

        graph_retriever = GraphRetriever(graph=graph, embedder=embedder)

        # ── Run each pipeline ─────────────────────────────────────────────────
        pipeline_results: dict[str, list[dict]] = {}
        cost_tracker = CostTracker(model=self.model)
        error_tax = ErrorTaxonomy()

        for pipeline_name in self.pipelines:
            logger.info("-" * 40)
            logger.info("Running pipeline: %s", pipeline_name)
            results = self._run_pipeline(
                pipeline_name=pipeline_name,
                docs=docs,
                label_names=label_names,
                vector_index=vector_index,
                graph_retriever=graph_retriever,
                llm=llm,
                cost_tracker=cost_tracker,
                error_tax=error_tax,
            )
            pipeline_results[pipeline_name] = results
            logger.info(
                "Pipeline %s complete: %d examples processed.", pipeline_name, len(results)
            )

        # ── Aggregate metrics ─────────────────────────────────────────────────
        metrics_summary = aggregate_results(
            pipeline_results, dataset=self.dataset_name, label_names=label_names
        )
        cost_summary = cost_tracker.summary()

        # Attach cost data to metrics
        for pname in metrics_summary:
            if pname in cost_summary:
                metrics_summary[pname]["cost"] = cost_summary[pname]

        # ── Error taxonomy summary ────────────────────────────────────────────
        error_summary: dict[str, dict] = {}
        for pname, results in pipeline_results.items():
            classifications = [r.get("error_classification", {}) for r in results]
            error_summary[pname] = error_tax.summary(classifications)

        # ── Save results ──────────────────────────────────────────────────────
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_prefix = os.path.join(self.results_dir, f"{self.dataset_name}_{timestamp}")

        save_json(pipeline_results, f"{out_prefix}_raw_results.json")
        save_json(metrics_summary, f"{out_prefix}_metrics.json")
        save_json(error_summary, f"{out_prefix}_errors.json")
        save_json(cost_summary, f"{out_prefix}_cost.json")

        self._print_summary_table(metrics_summary, error_summary)

        return {
            "metrics": metrics_summary,
            "cost": cost_summary,
            "errors": error_summary,
            "output_prefix": out_prefix,
        }

    # ── Pipeline dispatch ─────────────────────────────────────────────────────

    def _run_pipeline(
        self,
        pipeline_name: str,
        docs: list[dict],
        label_names: list[str],
        vector_index: VectorIndex,
        graph_retriever: GraphRetriever,
        llm: LLMClient,
        cost_tracker: CostTracker,
        error_tax: ErrorTaxonomy,
    ) -> list[dict]:
        results = []

        # Per-dataset query cap. CaseHOLD examples concatenate the citing
        # prompt with 5 candidate holdings and routinely exceed 500 chars;
        # truncating would hide the choices from the LLM.
        QUERY_CAPS = {"caseholder": 4000, "ledgar": 1500, "ecthr": 4000}
        cap = QUERY_CAPS.get(self.dataset_name, 2000)

        for i, doc in enumerate(docs):
            query = doc["text"][:cap]
            true_label = doc["label_str"]

            try:
                if pipeline_name == "single_llm":
                    raw = run_single_llm(query, llm, self.dataset_name)
                elif pipeline_name == "vector_rag":
                    raw = run_baseline_rag(query, vector_index, llm, self.dataset_name)
                elif pipeline_name == "graph_rag":
                    raw = run_graph_rag(query, graph_retriever, llm, self.dataset_name)
                elif pipeline_name == "single_agent":
                    raw = run_single_agent(query, graph_retriever, llm, self.dataset_name)
                elif pipeline_name == "multi_agent_vector":
                    raw = run_multi_agent_vector(query, vector_index, llm, self.dataset_name)
                elif pipeline_name == "multi_agent":
                    raw = run_multi_agent(query, graph_retriever, llm, self.dataset_name)
                else:
                    raise ValueError(f"Unknown pipeline: {pipeline_name}")

            except Exception as e:
                logger.error("Pipeline %s failed on doc %s: %s", pipeline_name, doc["id"], e)
                raw = {
                    "pipeline": pipeline_name,
                    "query": query,
                    "answer": "",
                    "latency_s": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }

            # Parse predicted label
            predicted_label = parse_answer(raw.get("answer", ""), label_names)
            raw["predicted_label"] = predicted_label
            raw["true_label"] = true_label
            raw["doc_id"] = doc["id"]

            # Error classification
            raw["error_classification"] = error_tax.classify(
                raw, true_label, label_names
            )

            # Record cost
            cost_tracker.record_from_result(raw, query_id=doc["id"])

            results.append(raw)

            if (i + 1) % 10 == 0:
                logger.info(
                    "  [%s] %d/%d done (latest: pred=%r, true=%r)",
                    pipeline_name, i + 1, len(docs), predicted_label, true_label,
                )

        return results

    # ── Display ───────────────────────────────────────────────────────────────

    @staticmethod
    def _print_summary_table(
        metrics: dict[str, Any],
        errors: dict[str, Any],
    ) -> None:
        header = f"\n{'Pipeline':<25} {'Acc':>6} {'F1-Mac':>8} {'F1-Wt':>8} {'AvgLat':>8} {'TotCost':>9}"
        logger.info(header)
        logger.info("-" * len(header))
        for pname, m in metrics.items():
            cost_usd = m.get("cost", {}).get("total_cost_usd", 0.0)
            logger.info(
                "%-25s %6.3f %8.3f %8.3f %8.2fs %9.4f$",
                pname,
                m.get("accuracy", 0),
                m.get("f1_macro", 0),
                m.get("f1_weighted", 0),
                m.get("avg_latency_s", 0),
                cost_usd,
            )
