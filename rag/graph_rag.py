"""
GraphRAG pipeline without agents (Pipeline 2).

Uses the GraphRetriever to fetch semantically + structurally relevant chunks,
then formats them with provenance paths before calling the LLM.
This baseline isolates the contribution of the graph structure vs. agents.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import config
from graph.retriever import GraphRetriever
from rag.vector_rag import _legal_system_prompt

logger = logging.getLogger(__name__)


class GraphRAGPipeline:
    """
    Pipeline 2 – Graph-based RAG (no agent layer).

    graph-retrieve → provenance-aware context → LLM answer
    """

    def __init__(
        self,
        retriever: GraphRetriever,
        llm_client,
        top_n: int = config.VECTOR_TOP_K,
    ):
        self.retriever = retriever
        self.llm = llm_client
        self.top_n = top_n

    def run(self, query: str, dataset: str = "ledgar") -> dict[str, Any]:
        t0 = time.perf_counter()

        # ── Graph retrieval ───────────────────────────────────────────────────
        result = self.retriever.retrieve_with_context(query)
        chunks = result["chunks"][: self.top_n]

        # ── Build provenance-annotated context ────────────────────────────────
        context_blocks: list[str] = []
        for i, rc in enumerate(chunks):
            path_str = " → ".join(rc.path) if rc.path else rc.chunk_id
            edge_str = ", ".join(rc.edge_types) if rc.edge_types else "direct"
            header = (
                f"[{i+1}] score={rc.score:.3f} | hop={rc.hop_distance} | "
                f"edge_types=[{edge_str}] | doc={rc.doc_id}\n"
                f"Provenance path: {path_str}"
            )
            context_blocks.append(f"{header}\n\n{rc.text}")

        context = "\n\n---\n\n".join(context_blocks)

        # ── Prompt ────────────────────────────────────────────────────────────
        system_prompt = _legal_system_prompt(dataset)
        user_prompt = _graph_rag_user_prompt(query, context)

        # ── LLM call ──────────────────────────────────────────────────────────
        response = self.llm.call(system_prompt, user_prompt)

        elapsed = time.perf_counter() - t0
        return {
            "pipeline": "graph_rag",
            "query": query,
            "answer": response["content"],
            "retrieved_chunks": [
                {
                    "chunk_id": rc.chunk_id,
                    "score": rc.score,
                    "hop_distance": rc.hop_distance,
                    "path": rc.path,
                    "edge_types": rc.edge_types,
                    "text": rc.text,
                }
                for rc in chunks
            ],
            "latency_s": elapsed,
            "input_tokens": response["input_tokens"],
            "output_tokens": response["output_tokens"],
        }


def _graph_rag_user_prompt(query: str, context: str) -> str:
    return (
        f"## Query\n{query}\n\n"
        f"## Graph-Retrieved Context (with provenance paths)\n{context}\n\n"
        "## Instructions\n"
        "Using ONLY the retrieved context above, answer the query. "
        "The provenance paths show how each passage is connected in the legal knowledge graph. "
        "Use these connections to build a coherent, well-grounded legal argument. "
        "Cite specific passages and explain any cross-document relationships that support your reasoning."
    )
