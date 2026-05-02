"""
Pipeline 0 – Single LLM (no retrieval) baseline.

Calls the LLM directly on the query with no retrieved evidence and no agent
scaffolding. This is the proposal's pure baseline (Configuration 1: "Single LLM,
no retrieval, 1 agent") used to establish the floor performance against which all
retrieval-augmented and multi-agent variants are compared.
"""

from __future__ import annotations

import time
from typing import Any

from rag.vector_rag import _legal_system_prompt


def run_single_llm(
    query: str,
    llm_client,
    dataset: str = "caseholder",
) -> dict[str, Any]:
    """
    Run a single, retrieval-free LLM call.

    Parameters
    ----------
    query:      The legal question / clause text.
    llm_client: LLMClient instance.
    dataset:    Dataset name for prompt tailoring.

    Returns
    -------
    Result dict with answer, latency, and token counts.
    """
    t0 = time.perf_counter()

    system_prompt = (
        _legal_system_prompt(dataset)
        + "\n\nYou have NO retrieved evidence. Answer using only your own knowledge.\n"
        "Structure your response as:\n"
        "ANALYSIS: [step-by-step legal reasoning]\n"
        "ANSWER: [final classification / holding / verdict]\n"
        "CONFIDENCE: [High / Medium / Low]"
    )

    user_prompt = (
        f"## Legal Query\n{query}\n\n"
        "Provide your structured legal analysis and final answer."
    )

    response = llm_client.call(system_prompt, user_prompt)
    elapsed = time.perf_counter() - t0

    return {
        "pipeline": "single_llm",
        "query": query,
        "answer": response["content"],
        "retrieved_chunks": [],
        "latency_s": elapsed,
        "input_tokens": response["input_tokens"],
        "output_tokens": response["output_tokens"],
    }
