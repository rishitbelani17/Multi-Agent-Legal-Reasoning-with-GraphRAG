"""
Pipeline 1 – Standard Vector RAG (baseline).

No graph structure.  Flat cosine-similarity retrieval → LLM answer.
"""

from __future__ import annotations

from typing import Any

from rag.vector_rag import VectorIndex, VectorRAG


def run_baseline_rag(
    query: str,
    index: VectorIndex,
    llm_client,
    dataset: str = "ledgar",
) -> dict[str, Any]:
    """
    Run the baseline vector RAG pipeline for a single query.

    Parameters
    ----------
    query:      The legal question / clause text to classify.
    index:      Pre-built VectorIndex over the document chunks.
    llm_client: LLMClient instance.
    dataset:    Dataset name for prompt tailoring.

    Returns
    -------
    Result dict with answer, retrieved chunks, latency and token counts.
    """
    pipeline = VectorRAG(index=index, llm_client=llm_client)
    return pipeline.run(query=query, dataset=dataset)
