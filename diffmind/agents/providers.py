from __future__ import annotations

import os
from typing import List

from .base import ChatMessage, LLMClient


class OpenAIChat(LLMClient):  # pragma: no cover - optional dependency
    def __init__(self, *, api_key: str, base_url: str | None, model: str) -> None:
        try:
            import openai  # type: ignore
        except Exception as e:
            raise RuntimeError("openai package is required for OpenAIChat") from e
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def complete_chat(self, messages: List[ChatMessage], temperature: float = 0.2) -> str:
        payload = [{"role": m.role, "content": m.content} for m in messages]
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=payload,
                temperature=temperature,
            )
        except Exception as e:
            # Some models (e.g., gpt-5-nano) don't allow custom temperature.
            msg = str(e).lower()
            if "temperature" in msg and ("unsupported" in msg or "does not support" in msg):
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=payload,
                )
            else:
                raise
        return (resp.choices[0].message.content or "").strip()


class NoopLLM(LLMClient):
    """LLM stub for simple mode. Returns a helpful offline message."""

    def complete_chat(self, messages: List[ChatMessage], temperature: float = 0.0) -> str:  # noqa: ARG002
        # Always return a final action in the orchestrator dialect
        return (
            '{"action": "final", "output": "AI is unavailable in simple mode. '
            'Set OPENAI_API_KEY and switch provider via: diffmind init"}'
        )
