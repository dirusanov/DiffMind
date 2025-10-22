from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ChatMessage:
    role: str  # system | user | assistant
    content: str


class LLMClient:
    """Minimal chat LLM interface used by the agent orchestrator."""

    def complete_chat(self, messages: List[ChatMessage], temperature: float = 0.2) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class ToolContext:
    """Context passed to tools (config, repo, etc.)."""

    def __init__(self, *, cfg: Any = None, repo: Any = None, cwd: Optional[str] = None) -> None:
        self.cfg = cfg
        self.repo = repo
        self.cwd = cwd


class Tool:
    """Tool interface compatible with ReAct-like loops.

    Each tool gets a short name, description and an optional schema description
    and returns a textual observation.
    """

    name: str = "tool"
    description: str = ""
    schema: Optional[Dict[str, Any]] = None

    def run(self, ctx: ToolContext, tool_input: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError

