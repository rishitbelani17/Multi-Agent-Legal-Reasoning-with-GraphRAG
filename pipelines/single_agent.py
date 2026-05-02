"""
Pipeline 3 – Single Agent + GraphRAG.

A single LLM agent performs retrieval AND reasoning in one pass,
using GraphRAG for its retrieval step.

This pipeline tests whether the agent scaffolding (without multi-agent
collaboration) adds value over the bare GraphRAG pipeline.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import config
from graph.retriever import GraphRetriever
from rag.vector_rag import _legal_system_prompt

logger = logging.getLogger(__name__)


def run_single_agent(
    query: str,
    retriever: GraphRetriever,
    llm_client,
    dataset: str = "ledgar",
) -> dict[str, Any]:
    """
    Run the single-agent GraphRAG pipeline for a single query.

    The agent is given the full graph context and asked to:
      1. Reason about the evidence structure.
      2. Form a legal argument.
      3. Produce a final classified answer.
    """
    t0 = time.perf_counter()

    # ── Graph retrieval ───────────────────────────────────────────────────────
    result = retriever.retrieve_with_context(query)
    chunks = result["chunks"][: config.VECTOR_TOP_K]

    context_blocks = []
    for i, rc in enumerate(chunks):
        path_str = " → ".join(rc.path[-3:]) if rc.path else rc.chunk_id
        edge_str = ", ".join(rc.edge_types[-2:]) if rc.edge_types else "direct"
        context_blocks.append(
            f"[{i+1}] score={rc.score:.3f} | hop={rc.hop_distance} | "
            f"edge=[{edge_str}] | doc={rc.doc_id}\n"
            f"Provenance: {path_str}\n\n{rc.text}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    # ── Single-agent prompt ───────────────────────────────────────────────────
    system_prompt = (
        _legal_system_prompt(dataset) + "\n\n"
        "You are operating as a single autonomous agent. "
        "You must:\n"
        "1. Analyse the retrieved evidence including graph provenance paths.\n"
        "2. Reason step-by-step about the legal question.\n"
        "3. Produce a final structured answer.\n\n"
        "Structure your response as:\n"
        "ANALYSIS: [step-by-step reasoning]\n"
        "ANSWER: [final answer]\n"
        "CONFIDENCE: [High / Medium / Low]\n"
        "CITATIONS: [passage numbers used]"
    )

    user_prompt = (
        f"## Legal Query\n{query}\n\n"
        f"## Graph-Retrieved Context\n{context}\n\n"
        "Provide your full legal analysis and final answer."
    )

    response = llm_client.call(system_prompt, user_prompt)
    elapsed = time.perf_counter() - t0

    return {
        "pipeline": "single_agent_graph_rag",
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
