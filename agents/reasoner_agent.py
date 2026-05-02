"""
Reasoner Agent – synthesises evidence from the RetrieverAgent into a
coherent legal argument and produces a structured answer.
"""

from __future__ import annotations

from typing import Any

from agents.base_agent import AgentMessage, BaseAgent
from rag.vector_rag import answer_format_instructions


class ReasonerAgent(BaseAgent):
    """
    Agent that:
    1. Reads the retriever's evidence summary.
    2. Constructs a step-by-step legal argument.
    3. Produces a final answer with confidence and cited passages.
    """

    def __init__(self, llm_client, dataset: str = "ledgar"):
        super().__init__(
            name="ReasonerAgent",
            llm_client=llm_client,
            role_description="Synthesises evidence into a legal argument.",
        )
        self.dataset = dataset

    @property
    def system_prompt(self) -> str:
        dataset_instructions = {
            "ledgar": (
                "You are a legal reasoning expert specialising in contract law. "
                "Given retrieved evidence, classify the contract clause type and "
                "build a step-by-step legal argument for your classification."
            ),
            "caseholder": (
                "You are a legal reasoning expert specialising in case law. "
                "Given retrieved evidence and candidate holdings, select the correct "
                "holding using a step-by-step chain-of-thought."
            ),
            "ecthr": (
                "You are a legal reasoning expert specialising in human rights law. "
                "Given retrieved evidence, predict whether a Convention article was "
                "violated and provide a structured legal analysis."
            ),
        }
        base = dataset_instructions.get(self.dataset, dataset_instructions["ledgar"])
        return (
            base + "\n\n"
            "Structure your response as:\n"
            "ANALYSIS: [step-by-step legal reasoning]\n"
            "ANSWER: [final classification / holding / verdict, in the exact label format below]\n"
            "CONFIDENCE: [High / Medium / Low]\n"
            "CITATIONS: [list of passage numbers used]\n\n"
            + answer_format_instructions(self.dataset)
        )

    def run(
        self,
        query: str,
        context: str = "",
        history: list[AgentMessage] | None = None,
        **kwargs: Any,
    ) -> AgentMessage:
        history = history or []
        history_str = self._format_history(history)

        # Prefer the RetrieverAgent's raw_context if available
        raw_context = ""
        for msg in reversed(history):
            if msg.sender == "RetrieverAgent" and "raw_context" in msg.metadata:
                raw_context = msg.metadata["raw_context"]
                break

        retriever_summary = ""
        for msg in history:
            if msg.sender == "RetrieverAgent":
                retriever_summary = msg.content
                break

        user_prompt = (
            f"## Legal Query\n{query}\n\n"
            + (f"## Evidence Summary (from RetrieverAgent)\n{retriever_summary}\n\n" if retriever_summary else "")
            + (f"## Retrieved Passages\n{raw_context}\n\n" if raw_context else "")
            + (f"{history_str}\n\n" if history_str else "")
            + "Provide your structured legal analysis and final answer."
        )

        return self._call_llm(user_prompt)
