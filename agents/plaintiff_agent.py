"""
Plaintiff Agent – constructs the affirmative legal argument.

Per the proposal's debate-style pipeline, the Plaintiff reads the
RetrieverAgent's evidence summary and builds the strongest possible
affirmative case for one of the candidate answers / labels / holdings.
"""

from __future__ import annotations

from typing import Any

from agents.base_agent import AgentMessage, BaseAgent
from rag.vector_rag import answer_format_instructions


class PlaintiffAgent(BaseAgent):
    """
    Agent that constructs the affirmative argument from retrieved evidence.

    Sequential debate role: Retriever → **Plaintiff** → Defense → Judge.
    """

    def __init__(self, llm_client, dataset: str = "caseholder"):
        super().__init__(
            name="PlaintiffAgent",
            llm_client=llm_client,
            role_description="Builds the affirmative legal argument.",
        )
        self.dataset = dataset

    @property
    def system_prompt(self) -> str:
        dataset_instructions = {
            "caseholder": (
                "You are the PLAINTIFF in a legal reasoning debate. "
                "Given the retrieved evidence and the candidate holdings, choose the "
                "holding that is best supported by the evidence and build the strongest "
                "possible affirmative argument for it."
            ),
            "ledgar": (
                "You are the PLAINTIFF in a contract-classification debate. "
                "Given the retrieved evidence, choose the contract clause type that is "
                "best supported and build the strongest affirmative argument for that "
                "classification."
            ),
            "ecthr": (
                "You are the PLAINTIFF in a human-rights-law debate. "
                "Given the retrieved facts and precedents, build the strongest "
                "affirmative argument for whether a Convention article was violated."
            ),
        }
        base = dataset_instructions.get(self.dataset, dataset_instructions["caseholder"])
        return (
            base + "\n\n"
            "Cite specific passages by number. Be persuasive but evidentiary.\n\n"
            "Structure your response as:\n"
            "POSITION: [the answer you are arguing for, in the exact label format below]\n"
            "ARGUMENT: [step-by-step affirmative case]\n"
            "EVIDENCE_CITED: [list of passage numbers]\n"
            "STRONGEST_POINT: [the single most persuasive piece of evidence]\n\n"
            + answer_format_instructions(self.dataset)
        )

    def run(
        self,
        query: str,
        context: str = "",
        history: list[AgentMessage] | None = None,
        debate_round: int = 1,
        total_rounds: int = 1,
        **kwargs: Any,
    ) -> AgentMessage:
        history = history or []

        retriever_summary = ""
        raw_context = ""
        prior_plaintiff: list[str] = []
        prior_defense: list[str] = []
        for msg in history:
            if msg.sender == "RetrieverAgent" and not retriever_summary:
                retriever_summary = msg.content
                raw_context = msg.metadata.get("raw_context", "")
            elif msg.sender == "PlaintiffAgent":
                prior_plaintiff.append(msg.content)
            elif msg.sender == "DefenseAgent":
                prior_defense.append(msg.content)

        is_rebuttal = debate_round > 1 and len(prior_defense) > 0

        prompt_parts = [f"## Legal Query\n{query}\n"]
        if retriever_summary:
            prompt_parts.append(f"## Evidence Summary (RetrieverAgent)\n{retriever_summary}\n")
        if raw_context:
            prompt_parts.append(f"## Retrieved Passages\n{raw_context}\n")

        if is_rebuttal:
            # Show the most recent Defense rebuttal so Plaintiff can directly counter it
            last_defense = prior_defense[-1]
            prompt_parts.append(f"## Defense's Most Recent Argument\n{last_defense}\n")
            if prior_plaintiff:
                prompt_parts.append(
                    f"## Your Previous Argument (do not repeat verbatim)\n{prior_plaintiff[-1]}\n"
                )
            prompt_parts.append(
                f"### Round {debate_round} of {total_rounds} — REBUTTAL\n"
                "The Defense has challenged your position. Refine your affirmative case: "
                "directly address each weakness Defense raised, strengthen your strongest "
                "evidence, and concede any point that is genuinely correct. "
                "If your position has not changed, restate POSITION but provide a *new* ARGUMENT."
            )
        else:
            prompt_parts.append(
                f"### Round {debate_round} of {total_rounds} — OPENING\n"
                "Construct your affirmative legal argument."
            )

        user_prompt = "\n".join(prompt_parts)
        return self._call_llm(user_prompt)
