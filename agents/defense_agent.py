"""
Defense Agent – identifies weak reasoning, missing evidence, or
inconsistencies in the Plaintiff's argument and offers the strongest
counter-argument.

Sequential debate role: Retriever → Plaintiff → **Defense** → Judge.
"""

from __future__ import annotations

from typing import Any

from agents.base_agent import AgentMessage, BaseAgent
from rag.vector_rag import answer_format_instructions


class DefenseAgent(BaseAgent):
    """
    Agent that critiques the Plaintiff's argument and proposes a counter-position.

    Unlike the previous CriticAgent (which only approved/rejected), the Defense
    must articulate the *opposing* legal position so that the Judge has a real
    debate to synthesize.
    """

    def __init__(self, llm_client, dataset: str = "caseholder"):
        super().__init__(
            name="DefenseAgent",
            llm_client=llm_client,
            role_description="Builds the opposing legal counter-argument.",
        )
        self.dataset = dataset

    @property
    def system_prompt(self) -> str:
        dataset_instructions = {
            "caseholder": (
                "You are the DEFENSE in a legal reasoning debate. "
                "The Plaintiff has selected a holding and argued for it. Your job is to "
                "(a) identify weaknesses in their argument, (b) propose the strongest "
                "alternative holding from the candidates, and (c) build an opposing case."
            ),
            "ledgar": (
                "You are the DEFENSE in a contract-classification debate. "
                "Identify weaknesses in the Plaintiff's clause classification and propose "
                "the strongest alternative classification with supporting evidence."
            ),
            "ecthr": (
                "You are the DEFENSE in a human-rights-law debate. "
                "Challenge the Plaintiff's violation/no-violation conclusion with the "
                "strongest opposing argument grounded in the retrieved precedents."
            ),
        }
        base = dataset_instructions.get(self.dataset, dataset_instructions["caseholder"])
        return (
            base + "\n\n"
            "Cite specific passages by number. Be precise and adversarial but fair.\n\n"
            "Structure your response as:\n"
            "WEAKNESSES: [numbered list of flaws in Plaintiff's argument]\n"
            "COUNTER_POSITION: [the alternative answer you propose, in the exact label format below]\n"
            "COUNTER_ARGUMENT: [step-by-step opposing case]\n"
            "EVIDENCE_CITED: [list of passage numbers]\n"
            "STRONGEST_POINT: [the single most damaging point against Plaintiff]\n\n"
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
        for msg in history:
            if msg.sender == "RetrieverAgent":
                retriever_summary = msg.content
                raw_context = msg.metadata.get("raw_context", "")
            if msg.sender == "PlaintiffAgent":
                plaintiff_argument = msg.content

        user_prompt = (
            f"## Legal Query\n{query}\n\n"
            + (f"## Evidence Summary (RetrieverAgent)\n{retriever_summary}\n\n" if retriever_summary else "")
            + (f"## Retrieved Passages\n{raw_context}\n\n" if raw_context else "")
            + f"## Plaintiff's Argument\n{plaintiff_argument}\n\n"
            "Critique the Plaintiff's argument and present the strongest opposing case."
        )

        return self._call_llm(user_prompt)
