"""
Thin wrapper around the Anthropic Messages API.

Handles:
  - Single call with system + user prompt
  - Automatic retries on rate-limit errors
  - Token usage extraction
  - Response normalisation
"""

from __future__ import annotations

import logging
import time
from typing import Any

import anthropic

import config

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Wrapper around anthropic.Anthropic for uniform LLM access.

    All agents and pipelines use this client so token tracking
    and retry logic is centralised.

    Parameters
    ----------
    model:       Anthropic model ID.
    max_tokens:  Max output tokens.
    temperature: Sampling temperature (0.0 = deterministic).
    max_retries: Number of retries on transient API errors.
    """

    def __init__(
        self,
        model: str = config.LLM_MODEL,
        max_tokens: int = config.LLM_MAX_TOKENS,
        temperature: float = config.LLM_TEMPERATURE,
        max_retries: int = 3,
        api_key: str | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = anthropic.Anthropic(api_key=api_key or config.ANTHROPIC_API_KEY)

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        extra_messages: list[dict] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Call the Anthropic Messages API.

        Parameters
        ----------
        system_prompt:   System message content.
        user_prompt:     User message content.
        extra_messages:  Optional additional messages for multi-turn context.
        max_tokens:      Optional per-call override of self.max_tokens. Used by
                         agents that need a longer output budget (e.g. the
                         JudgeAgent, which produces a four-line header followed
                         by extended reasoning and was being truncated at the
                         default 1024 cap).

        Returns
        -------
        dict with keys: content, input_tokens, output_tokens, stop_reason
        """
        messages: list[dict] = []
        if extra_messages:
            messages.extend(extra_messages)
        messages.append({"role": "user", "content": user_prompt})

        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        for attempt in range(self.max_retries):
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=effective_max_tokens,
                    temperature=self.temperature,
                    system=system_prompt,
                    messages=messages,
                )
                content = response.content[0].text if response.content else ""
                return {
                    "content": content,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "stop_reason": response.stop_reason,
                }
            except anthropic.RateLimitError as e:
                wait = 2 ** attempt
                logger.warning("Rate limit hit (attempt %d/%d). Waiting %ds …",
                               attempt + 1, self.max_retries, wait)
                time.sleep(wait)
            except anthropic.APIError as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning("API error (attempt %d/%d): %s", attempt + 1, self.max_retries, e)
                time.sleep(1)

        raise RuntimeError("LLM call failed after all retries.")

    def __repr__(self) -> str:
        return f"LLMClient(model={self.model!r}, max_tokens={self.max_tokens})"
