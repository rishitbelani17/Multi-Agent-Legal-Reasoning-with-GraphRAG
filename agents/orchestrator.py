"""
Multi-Agent Orchestrators.

Two orchestrators are provided:

1. ``DebateOrchestrator`` (proposal-aligned)
   Sequential debate-style pipeline with four specialized agents:
       RetrieverAgent → PlaintiffAgent → DefenseAgent → JudgeAgent
   The Judge produces the final answer. This is the architecture described in
   §2.2 of the project proposal.

2. ``MultiAgentOrchestrator`` (legacy critic-revision loop)
   Kept for backward compatibility with earlier experiments. Uses
   Reasoner + Critic with an iterative revision loop.

Both orchestrators track total token usage and latency across all agents.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import config
from agents.base_agent import AgentMessage
from agents.critic_agent import CriticAgent, ReasonerWithRevision
from agents.defense_agent import DefenseAgent
from agents.judge_agent import JudgeAgent
from agents.plaintiff_agent import PlaintiffAgent
from agents.reasoner_agent import ReasonerAgent
from agents.retriever_agent import RetrieverAgent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Proposal-aligned debate orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class DebateOrchestrator:
    """
    Sequential debate-style multi-agent pipeline.

    Round structure
    ---------------
        Retriever → Plaintiff → Defense → Judge

    The Judge's output is the final answer. There is no critic-revision loop;
    the Defense itself plays the adversarial role, and the Judge synthesizes.
    """

    def __init__(
        self,
        retriever_agent: RetrieverAgent,
        plaintiff_agent: PlaintiffAgent,
        defense_agent: DefenseAgent,
        judge_agent: JudgeAgent,
        retrieval_mode: str = "graph",   # "graph" | "vector"  – for logging only
    ):
        self.retriever = retriever_agent
        self.plaintiff = plaintiff_agent
        self.defense = defense_agent
        self.judge = judge_agent
        self.retrieval_mode = retrieval_mode

    def run(self, query: str, dataset: str = "caseholder") -> dict[str, Any]:
        t_start = time.perf_counter()
        history: list[AgentMessage] = []
        round_logs: list[dict] = []

        # ── Stage 1: Retrieve ────────────────────────────────────────────────
        logger.info("[Debate] Stage 1: RetrieverAgent (%s) …", self.retrieval_mode)
        retriever_msg = self.retriever.run(query=query, history=history)
        history.append(retriever_msg)
        round_logs.append(_log_msg("retrieval", retriever_msg))

        # ── Stage 2: Plaintiff ───────────────────────────────────────────────
        logger.info("[Debate] Stage 2: PlaintiffAgent …")
        plaintiff_msg = self.plaintiff.run(query=query, history=history)
        history.append(plaintiff_msg)
        round_logs.append(_log_msg("plaintiff", plaintiff_msg))

        # ── Stage 3: Defense ─────────────────────────────────────────────────
        logger.info("[Debate] Stage 3: DefenseAgent …")
        defense_msg = self.defense.run(query=query, history=history)
        history.append(defense_msg)
        round_logs.append(_log_msg("defense", defense_msg))

        # ── Stage 4: Judge ───────────────────────────────────────────────────
        logger.info("[Debate] Stage 4: JudgeAgent …")
        judge_msg = self.judge.run(query=query, history=history)
        history.append(judge_msg)
        round_logs.append(_log_msg("judge", judge_msg))

        total_latency = time.perf_counter() - t_start
        total_in, total_out = _sum_tokens(history)

        return {
            "pipeline": f"multi_agent_{self.retrieval_mode}_rag",
            "query": query,
            "answer": judge_msg.content,        # Judge produces final ANSWER
            "stages": ["retrieval", "plaintiff", "defense", "judge"],
            "round_logs": round_logs,
            "retrieved_chunks": retriever_msg.metadata.get("chunks", []),
            "sub_queries": retriever_msg.metadata.get("sub_queries", []),
            "latency_s": total_latency,
            "input_tokens": total_in,
            "output_tokens": total_out,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Legacy critic-revision orchestrator (kept for backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────

class MultiAgentOrchestrator:
    """
    Legacy 3-agent + revision-loop orchestrator.

    Round structure
    ---------------
    Round 1:  RetrieverAgent → ReasonerAgent → CriticAgent
    Round 2+: ReasonerRevised → CriticAgent    (if critic says REVISION_NEEDED)
    Stop when: critic says APPROVED  OR  max_rounds reached.
    """

    def __init__(
        self,
        retriever_agent: RetrieverAgent,
        reasoner_agent: ReasonerAgent,
        critic_agent: CriticAgent,
        max_rounds: int = config.MAX_AGENT_ROUNDS,
    ):
        self.retriever = retriever_agent
        self.reasoner = reasoner_agent
        self.critic = critic_agent
        self.max_rounds = max_rounds
        self._reviser = ReasonerWithRevision(
            llm_client=reasoner_agent.llm,
            dataset=reasoner_agent.dataset,
        )

    def run(self, query: str, dataset: str = "ledgar") -> dict[str, Any]:
        t_start = time.perf_counter()
        history: list[AgentMessage] = []
        round_logs: list[dict] = []

        # ── Round 0: Retrieve ─────────────────────────────────────────────────
        logger.info("[Orchestrator] Round 0: RetrieverAgent …")
        retriever_msg = self.retriever.run(query=query, history=history)
        history.append(retriever_msg)
        round_logs.append(_log_msg("retrieval", retriever_msg))

        # ── Round 1: Reason ───────────────────────────────────────────────────
        logger.info("[Orchestrator] Round 1: ReasonerAgent …")
        reasoner_msg = self.reasoner.run(query=query, history=history)
        history.append(reasoner_msg)
        round_logs.append(_log_msg("reasoning_round_1", reasoner_msg))

        # ── Critic loop ───────────────────────────────────────────────────────
        approved = False
        final_answer_msg = reasoner_msg

        for round_num in range(1, self.max_rounds + 1):
            logger.info("[Orchestrator] Round %d: CriticAgent …", round_num)
            critic_msg = self.critic.run(query=query, history=history)
            history.append(critic_msg)
            round_logs.append(_log_msg(f"critique_round_{round_num}", critic_msg))

            if critic_msg.metadata.get("approved", False):
                logger.info("[Orchestrator] CriticAgent approved at round %d.", round_num)
                approved = True
                break

            if round_num < self.max_rounds:
                logger.info("[Orchestrator] Revision requested – round %d …", round_num + 1)
                revised_msg = self._reviser.run(query=query, history=history)
                history.append(revised_msg)
                round_logs.append(_log_msg(f"revision_round_{round_num+1}", revised_msg))
                final_answer_msg = revised_msg

        total_latency = time.perf_counter() - t_start
        total_in, total_out = _sum_tokens(history)

        return {
            "pipeline": "multi_agent_graph_rag_legacy",
            "query": query,
            "answer": final_answer_msg.content,
            "approved_by_critic": approved,
            "rounds_completed": len([l for l in round_logs if "critique" in l["stage"]]),
            "round_logs": round_logs,
            "retrieved_chunks": retriever_msg.metadata.get("chunks", []),
            "sub_queries": retriever_msg.metadata.get("sub_queries", []),
            "latency_s": total_latency,
            "input_tokens": total_in,
            "output_tokens": total_out,
        }


# ── helpers ───────────────────────────────────────────────────────────────────

def _log_msg(stage: str, msg: AgentMessage) -> dict[str, Any]:
    return {
        "stage": stage,
        "sender": msg.sender,
        "content": msg.content,
        "input_tokens": msg.input_tokens,
        "output_tokens": msg.output_tokens,
        "latency_s": msg.latency_s,
    }


def _sum_tokens(history: list[AgentMessage]) -> tuple[int, int]:
    return (
        sum(m.input_tokens for m in history),
        sum(m.output_tokens for m in history),
    )
