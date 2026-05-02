"""
Pipeline 3b – Multi-Agent + Vector RAG.

Same four-agent debate pipeline as Multi-Agent + GraphRAG, but with the
RetrieverAgent backed by a flat vector index (no graph traversal).

This is the critical ablation pair for the proposal: comparing it against
``multi_agent`` (GraphRAG) isolates the contribution of structured retrieval
while holding agent count, prompts, and judge logic constant.
"""

from __future__ import annotations

from typing import Any

from agents.defense_agent import DefenseAgent
from agents.judge_agent import JudgeAgent
from agents.orchestrator import DebateOrchestrator
from agents.plaintiff_agent import PlaintiffAgent
from agents.vector_retriever_agent import VectorRetrieverAgent
from rag.vector_rag import VectorIndex


def run_multi_agent_vector(
    query: str,
    index: VectorIndex,
    llm_client,
    dataset: str = "caseholder",
) -> dict[str, Any]:
    """
    Run the four-agent debate pipeline backed by vector RAG retrieval.
    """
    retriever_agent = VectorRetrieverAgent(
        llm_client=llm_client,
        index=index,
        dataset=dataset,
    )
    plaintiff_agent = PlaintiffAgent(llm_client=llm_client, dataset=dataset)
    defense_agent = DefenseAgent(llm_client=llm_client, dataset=dataset)
    judge_agent = JudgeAgent(llm_client=llm_client, dataset=dataset)

    orchestrator = DebateOrchestrator(
        retriever_agent=retriever_agent,
        plaintiff_agent=plaintiff_agent,
        defense_agent=defense_agent,
        judge_agent=judge_agent,
        retrieval_mode="vector",
    )
    return orchestrator.run(query=query, dataset=dataset)
