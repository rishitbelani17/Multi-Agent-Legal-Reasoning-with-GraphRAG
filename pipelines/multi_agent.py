"""
Pipeline 4 – Multi-Agent + GraphRAG (full proposed system).

Sequential debate-style pipeline:
    RetrieverAgent (graph) → PlaintiffAgent → DefenseAgent → JudgeAgent

This is the primary system being evaluated in the proposal (§2.2).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import config
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
    n_debate_rounds: Optional[int] = None,
    on_message: Optional[Callable] = None,
    judge_llm_client=None,
) -> dict[str, Any]:
    """
    Run the full Multi-Agent + GraphRAG debate pipeline for a single query.

    Parameters
    ----------
    n_debate_rounds:
        Override config.N_DEBATE_ROUNDS for this call.
    on_message:
        Streaming callback fn(stage, AgentMessage) called after each agent turn.
    judge_llm_client:
        Optional dedicated client for the Judge (use a stronger model than the
        workers). If None, the JudgeAgent picks one from ``config.LLM_MODEL_JUDGE``.
    """
    retriever_agent = RetrieverAgent(
        llm_client=llm_client,
        retriever=retriever,
        dataset=dataset,
    )
    plaintiff_agent = PlaintiffAgent(llm_client=llm_client, dataset=dataset)
    defense_agent = DefenseAgent(llm_client=llm_client, dataset=dataset)
    judge_agent = JudgeAgent(
        llm_client=llm_client,
        dataset=dataset,
        judge_llm_client=judge_llm_client,
    )

    orchestrator = DebateOrchestrator(
        retriever_agent=retriever_agent,
        plaintiff_agent=plaintiff_agent,
        defense_agent=defense_agent,
        judge_agent=judge_agent,
        retrieval_mode="graph",
        n_debate_rounds=n_debate_rounds if n_debate_rounds is not None else config.N_DEBATE_ROUNDS,
        on_message=on_message,
    )
    return orchestrator.run(query=query, dataset=dataset)
