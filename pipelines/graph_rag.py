"""
Pipeline 2 – GraphRAG (no agent layer).

Graph-based retrieval with provenance paths → LLM answer.
Isolates the contribution of graph structure vs. flat vector retrieval.
"""

from __future__ import annotations

from typing import Any

from graph.retriever import GraphRetriever
from rag.graph_rag import GraphRAGPipeline


def run_graph_rag(
    query: str,
    retriever: GraphRetriever,
    llm_client,
    dataset: str = "ledgar",
) -> dict[str, Any]:
    """
    Run the GraphRAG pipeline (no agents) for a single query.

    Parameters
    ----------
    query:      The legal question / clause text to classify.
    retriever:  Pre-built GraphRetriever.
    llm_client: LLMClient instance.
    dataset:    Dataset name for prompt tailoring.
    """
    pipeline = GraphRAGPipeline(retriever=retriever, llm_client=llm_client)
    return pipeline.run(query=query, dataset=dataset)
