"""
Judge Agent – synthesizes the Plaintiff's and Defense's arguments and
produces the final reasoned answer.

Sequential debate role: Retriever → Plaintiff → Defense → **Judge**.
"""

from __future__ import annotations

from typing import Any

import config
from agents.base_agent import AgentMessage, BaseAgent
from rag.vector_rag import answer_format_instructions
from utils.llm_client import LLMClient


class JudgeAgent(BaseAgent):
    """
    Agent that weighs both sides and renders the final decision.

    The Judge is the only agent that produces the structured ANSWER that the
    evaluation pipeline parses.

    Unlike the Plaintiff / Defense / Retriever (which are generators), the
    Judge is a *discriminator* over a long debate transcript. We therefore
    let it run on a separate, stronger model — controlled by
    ``config.LLM_MODEL_JUDGE`` — so workers stay cheap while the final
    decision benefits from a stronger model. Pass ``judge_llm_client`` to
    override; otherwise we instantiate one from config.
    """

    def __init__(
        self,
        llm_client,
        dataset: str = "caseholder",
        judge_llm_client: LLMClient | None = None,
    ):
        # If LLM_MODEL_JUDGE is set and differs from the worker model, build
        # a dedicated client. Otherwise reuse the shared client.
        if judge_llm_client is not None:
            client = judge_llm_client
        elif (
            getattr(config, "LLM_MODEL_JUDGE", "")
            and config.LLM_MODEL_JUDGE != getattr(llm_client, "model", config.LLM_MODEL)
        ):
            client = LLMClient(
                model=config.LLM_MODEL_JUDGE,
                max_tokens=getattr(config, "LLM_MAX_TOKENS_JUDGE", 2048),
            )
        else:
            client = llm_client

        super().__init__(
            name="JudgeAgent",
            llm_client=client,
            role_description="Synthesizes both arguments and produces the final answer.",
        )
        self.dataset = dataset

    @property
    def system_prompt(self) -> str:
        dataset_instructions = {
            "caseholder": (
                "You are the JUDGE in a legal reasoning debate. "
                "Two advocates have argued for different holdings across one or more "
                "rounds. Weigh the arguments AND the underlying retrieved evidence, "
                "then select the correct holding."
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
            "## Decision rules (apply in order)\n"
            "1. **Evidence anchor.** Your primary anchor is the retrieved passages, "
            "NOT the rhetorical strength of either advocate. A weak argument for the "
            "evidence-supported answer beats an eloquent argument for an unsupported one.\n"
            "2. **Tie-breaker.** If both positions seem equally plausible after debate, "
            "default to the position whose `EVIDENCE_CITED` passages most directly "
            "answer the query as posed.\n"
            "3. **Reject rhetorical tricks.** Discount any point that depends on "
            "speculation about evidence that isn't in front of you, or on "
            "characterising the *opposing* argument unfairly.\n"
            "4. **Pick from the candidates only.** Your final ANSWER MUST be one of "
            "the candidate labels — never a label not on the list.\n"
            "5. **No abstention.** You must pick one label even if confidence is Low.\n\n"
            "## Output format (STRICT — the parser depends on this)\n"
            "Begin your response with EXACTLY these four lines, each on its own line, "
            "in this order, BEFORE any reasoning prose:\n\n"
            "FINAL ANSWER: <the literal label, e.g. holding_3>\n"
            "CONFIDENCE: <High|Medium|Low>\n"
            "CITATIONS: <comma-separated passage numbers that determined the outcome>\n"
            "REASONING_SUMMARY: <2-3 sentence summary of why this answer wins>\n\n"
            "After this four-line header, you may write extended reasoning. The "
            "`FINAL ANSWER:` line MUST be the FIRST non-empty line of your response. "
            "This placement is critical: if your response is truncated mid-paragraph "
            "the parser still recovers the answer. Do not wrap the label in quotes, "
            "parentheses, or any other formatting. Do not write 'holding 3' or "
            "'Holding_3' — only the exact lowercase token.\n\n"
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
        plaintiff_turns: list[str] = []
        defense_turns: list[str] = []
        for msg in history:
            if msg.sender == "RetrieverAgent" and not retriever_summary:
                retriever_summary = msg.content
                raw_context = msg.metadata.get("raw_context", "")
            elif msg.sender == "PlaintiffAgent":
                plaintiff_turns.append(msg.content)
            elif msg.sender == "DefenseAgent":
                defense_turns.append(msg.content)

        # Format the FULL debate history so the Judge sees how positions evolved
        # across rounds, not just the final exchange. This was a real bug: the
        # original Judge only saw the last Plaintiff + Defense turn, missing
        # earlier concessions and any drift introduced by counter-rebuttals.
        debate_blocks: list[str] = []
        n_rounds = max(len(plaintiff_turns), len(defense_turns))
        for r in range(n_rounds):
            if r < len(plaintiff_turns):
                debate_blocks.append(
                    f"### Plaintiff — Round {r+1}\n{plaintiff_turns[r]}"
                )
            if r < len(defense_turns):
                debate_blocks.append(
                    f"### Defense — Round {r+1}\n{defense_turns[r]}"
                )
        full_debate = "\n\n".join(debate_blocks) if debate_blocks else "(no debate turns)"

        user_prompt = (
            f"## Legal Query\n{query}\n\n"
            + (f"## Evidence Summary\n{retriever_summary}\n\n" if retriever_summary else "")
            + (f"## Retrieved Passages\n{raw_context}\n\n" if raw_context else "")
            + f"## Full Debate Transcript ({n_rounds} round(s))\n{full_debate}\n\n"
            "Render your final, reasoned decision. START with the four-line header "
            "specified in your instructions — `FINAL ANSWER:` MUST be the first line "
            "so the answer is recoverable even if your reasoning is truncated."
        )

        # Judge needs a larger output budget than the default 1024 — it
        # produces a four-line header plus extended reasoning. Empirically the
        # default cap was truncating Judge bodies before they could reach the
        # FINAL ANSWER line, which is why we now lead with the answer AND give
        # extra room for the prose that follows. The dedicated Judge client
        # already defaults to LLM_MAX_TOKENS_JUDGE; pass an explicit cap so a
        # shared client (worker model fallback) still gets the right budget.
        return self._call_llm(
            user_prompt,
            max_tokens=getattr(config, "LLM_MAX_TOKENS_JUDGE", 2048),
        )
