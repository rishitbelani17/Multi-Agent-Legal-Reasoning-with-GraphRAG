"""
Critic Agent – reviews the ReasonerAgent's output for logical errors,
unsupported claims, and missing counter-arguments.

The critic can either approve the reasoning or request a revision.
This adversarial review loop improves answer quality and interpretability.
"""

from __future__ import annotations

import re
from typing import Any

from agents.base_agent import AgentMessage, BaseAgent


class CriticAgent(BaseAgent):
    """
    Agent that:
    1. Reads the reasoner's analysis and answer.
    2. Identifies logical gaps, unsupported claims, or contradictions.
    3. Returns either APPROVED or REVISION_NEEDED with specific feedback.
    """

    def __init__(self, llm_client, dataset: str = "ledgar"):
        super().__init__(
            name="CriticAgent",
            llm_client=llm_client,
            role_description="Reviews legal reasoning for errors and unsupported claims.",
        )
        self.dataset = dataset

    @property
    def system_prompt(self) -> str:
        return (
            "You are a senior legal reviewer with expertise in logical argumentation. "
            "Your role is to critically evaluate a legal reasoning chain and identify:\n"
            "1. Unsupported claims (conclusions not backed by cited evidence)\n"
            "2. Logical fallacies or non-sequiturs\n"
            "3. Missing counter-arguments or alternative interpretations\n"
            "4. Misapplication of legal principles\n"
            "5. Overconfident or underconfident conclusions\n\n"
            "Structure your response as:\n"
            "VERDICT: [APPROVED | REVISION_NEEDED]\n"
            "ISSUES: [numbered list of problems, or 'None' if approved]\n"
            "SUGGESTIONS: [specific improvements to address each issue]\n"
            "REVISED_CONFIDENCE: [High / Medium / Low – your assessment of the answer quality]"
        )

    def run(
        self,
        query: str,
        context: str = "",
        history: list[AgentMessage] | None = None,
        **kwargs: Any,
    ) -> AgentMessage:
        history = history or []

        reasoner_output = ""
        for msg in reversed(history):
            if msg.sender == "ReasonerAgent":
                reasoner_output = msg.content
                break

        retriever_summary = ""
        for msg in history:
            if msg.sender == "RetrieverAgent":
                retriever_summary = msg.content
                break

        user_prompt = (
            f"## Original Legal Query\n{query}\n\n"
            + (f"## Evidence Available\n{retriever_summary}\n\n" if retriever_summary else "")
            + f"## Reasoning to Review\n{reasoner_output}\n\n"
            "Critically evaluate the reasoning above. Be specific and constructive."
        )

        msg = self._call_llm(user_prompt)
        msg.metadata["approved"] = _is_approved(msg.content)
        return msg


class ReasonerWithRevision(BaseAgent):
    """
    Extended reasoner that incorporates critic feedback to produce
    a revised answer.  Used in the iterative multi-agent loop.
    """

    def __init__(self, llm_client, dataset: str = "ledgar"):
        super().__init__(
            name="ReasonerRevised",
            llm_client=llm_client,
            role_description="Revises legal reasoning based on critic feedback.",
        )
        self.dataset = dataset

    @property
    def system_prompt(self) -> str:
        return (
            "You are a legal reasoning expert. "
            "You have received feedback from a senior reviewer on your previous analysis. "
            "Revise your reasoning to address all identified issues while maintaining "
            "evidentiary grounding. "
            "Structure your response identically to before:\n"
            "ANALYSIS: ...\nANSWER: ...\nCONFIDENCE: ...\nCITATIONS: ..."
        )

    def run(
        self,
        query: str,
        context: str = "",
        history: list[AgentMessage] | None = None,
        **kwargs: Any,
    ) -> AgentMessage:
        history = history or []

        reasoner_output = ""
        critic_output = ""
        raw_context = ""

        for msg in history:
            if msg.sender == "RetrieverAgent" and "raw_context" in msg.metadata:
                raw_context = msg.metadata["raw_context"]
            if msg.sender == "ReasonerAgent":
                reasoner_output = msg.content
            if msg.sender == "CriticAgent":
                critic_output = msg.content

        user_prompt = (
            f"## Original Query\n{query}\n\n"
            + (f"## Retrieved Evidence\n{raw_context}\n\n" if raw_context else "")
            + f"## Your Previous Analysis\n{reasoner_output}\n\n"
            + f"## Critic Feedback\n{critic_output}\n\n"
            "Revise your analysis addressing all critic issues."
        )

        return self._call_llm(user_prompt)


def _is_approved(critic_output: str) -> bool:
    """Parse critic verdict from output."""
    m = re.search(r"VERDICT\s*:\s*(APPROVED|REVISION_NEEDED)", critic_output, re.IGNORECASE)
    if m:
        return m.group(1).upper() == "APPROVED"
    # Fallback heuristic
    lower = critic_output.lower()
    if "approved" in lower and "revision" not in lower:
        return True
    return False
