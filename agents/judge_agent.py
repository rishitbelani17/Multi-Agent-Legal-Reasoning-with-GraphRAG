"""
Judge Agent – synthesizes the Plaintiff's and Defense's arguments and
produces the final reasoned answer.

Sequential debate role: Retriever → Plaintiff → Defense → **Judge**.
"""

from __future__ import annotations

from typing import Any

from agents.base_agent import AgentMessage, BaseAgent
from rag.vector_rag import answer_format_instructions


class JudgeAgent(BaseAgent):
    """
    Agent that weighs both sides and renders the final decision.

    The Judge is the only agent that produces the structured ANSWER that the
    evaluation pipeline parses.
    """

    def __init__(self, llm_client, dataset: str = "caseholder"):
        super().__init__(
            name="JudgeAgent",
            llm_client=llm_client,
            role_description="Synthesizes both arguments and produces the final answer.",
        )
        self.dataset = dataset

    @property
    def system_prompt(self) -> str:
        dataset_instructions = {
            "caseholder": (
                "You are the JUDGE in a legal reasoning debate. "
                "Two advocates have argued for different holdings. Weigh the arguments "
                "and the underlying evidence, then select the correct holding."
            ),
            "ledgar": (
                "You are the JUDGE in a contract-classification debate. "
                "Weigh the Plaintiff's and Defense's classifications against the "
                "evidence and select the correct clause type."
            ),
            "ecthr": (
                "You are the JUDGE in a human-rights-law debate. "
                "Weigh both sides and decide whether a Convention article was violated."
            ),
        }
        base = dataset_instructions.get(self.dataset, dataset_instructions["caseholder"])
        return (
            base + "\n\n"
            "Be impartial. Ground your decision in the evidence, not in which "
            "advocate was more eloquent. Flag any unsupported claims by either side.\n\n"
            "Structure your response as:\n"
            "REASONING: [why you weighed the arguments as you did]\n"
            "ANSWER: [final classification / holding / verdict, in the exact label format below]\n"
            "CONFIDENCE: [High / Medium / Low]\n"
            "CITATIONS: [passage numbers that determined the outcome]\n"
            "DISSENT_NOTES: [any merit in the losing side worth noting]\n\n"
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

        retriever_summary = ""
        raw_context = ""
        plaintiff_argument = ""
        defense_argument = ""
        for msg in history:
            if msg.sender == "RetrieverAgent":
                retriever_summary = msg.content
                raw_context = msg.metadata.get("raw_context", "")
            if msg.sender == "PlaintiffAgent":
                plaintiff_argument = msg.content
            if msg.sender == "DefenseAgent":
                defense_argument = msg.content

        user_prompt = (
            f"## Legal Query\n{query}\n\n"
            + (f"## Evidence Summary\n{retriever_summary}\n\n" if retriever_summary else "")
            + (f"## Retrieved Passages\n{raw_context}\n\n" if raw_context else "")
            + f"## Plaintiff's Argument\n{plaintiff_argument}\n\n"
            + f"## Defense's Counter-Argument\n{defense_argument}\n\n"
            "Render your final, reasoned decision."
        )

        return self._call_llm(user_prompt)
