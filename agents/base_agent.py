"""
Base agent infrastructure.

Each agent is a stateful object that:
  - receives AgentMessage objects
  - calls the LLM with a role-specific system prompt
  - returns an AgentMessage with its response and token usage
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Structured message passed between agents."""
    sender: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


class BaseAgent:
    """
    Abstract base for all agents.

    Subclasses must implement:
      - system_prompt (property)
      - run(query, context, history) -> AgentMessage
    """

    def __init__(self, name: str, llm_client, role_description: str = ""):
        self.name = name
        self.llm = llm_client
        self.role_description = role_description
        self._message_history: list[AgentMessage] = []

    @property
    def system_prompt(self) -> str:
        raise NotImplementedError

    def run(
        self,
        query: str,
        context: str = "",
        history: list[AgentMessage] | None = None,
        **kwargs: Any,
    ) -> AgentMessage:
        raise NotImplementedError

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _call_llm(self, user_content: str, max_tokens: int | None = None) -> AgentMessage:
        t0 = time.perf_counter()
        response = self.llm.call(
            self.system_prompt, user_content, max_tokens=max_tokens
        )
        elapsed = time.perf_counter() - t0
        return AgentMessage(
            sender=self.name,
            content=response["content"],
            input_tokens=response["input_tokens"],
            output_tokens=response["output_tokens"],
            latency_s=elapsed,
        )

    @staticmethod
    def _format_history(history: list[AgentMessage]) -> str:
        if not history:
            return ""
        lines = ["## Prior Agent Outputs"]
        for msg in history:
            lines.append(f"\n### {msg.sender}\n{msg.content}")
        return "\n".join(lines)
