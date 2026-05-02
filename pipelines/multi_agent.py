"""
Pipeline 4 – Multi-Agent + GraphRAG (full proposed system).

Sequential debate-style pipeline:
    RetrieverAgent (graph) → PlaintiffAgent → DefenseAgent → JudgeAgent

This is the primary system being evaluated in the proposal (§2.2).
"""

from __future__ import annotations

from typing import Any

from agents.defense_agent import DefenseAgent
from agents.judge_agent import JudgeAgent
from agents.orchestrator import DebateOrchestrator
from agents.plaintiff_agent import PlaintiffAgent
from agents.retriever_agent import RetrieverAgent
from graph.retriever import GraphRetriever


def run_multi_agent(
    query: str,
    retriever: GraphRetriever,
    llm_client,
    dataset: str = "caseholder",
) -> dict[str, Any]:
    """
    Run the full Multi-Agent + GraphRAG debate pipeline for a single query.
    """
    retriever_agent = RetrieverAgent(
        llm_client=llm_client,
        retriever=retriever,
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
        retrieval_mode="graph",
    )
    return orchestrator.run(query=query, dataset=dataset)
