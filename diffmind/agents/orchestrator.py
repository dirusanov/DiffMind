from __future__ import annotations

import json
from typing import List, Optional
import re

from .base import ChatMessage, LLMClient, Tool, ToolContext


SYSTEM_PROMPT = (
    "You are a senior Git assistant. You can think step-by-step, use tools, and produce precise commands. "
    "When the question is Russian, answer in Russian; otherwise match the user's language. "
    "Keep answers concise with numbered steps and exact commands."
)


INSTRUCTIONS = (
    "You may use tools to gather context or compute.\n"
    "Available tools are described below.\n"
    "Respond using a single-line JSON object at each step in one of the forms:\n"
    '{"action": "tool", "tool": "<name>", "input": "<string>"}\n'
    '{"action": "<tool_name>", "input": "<string>"}  (shorthand also accepted)\n'
    '{"action": "final", "output": "<your final helpful answer>"}\n'
    "After a tool call, you will receive an Observation with the tool's result. Use it to decide next step.\n"
    "If the Observation already contains the necessary answer, respond with a final action.\n"
    "Do at most 4 steps unless necessary. Prefer a single tool call when enough."
)


def _tool_catalog(tools: List[Tool]) -> str:
    lines = []
    for t in tools:
        lines.append(f"- {t.name}: {t.description}")
    return "\n".join(lines)


class AgentOrchestrator:
    def __init__(self, llm: LLMClient, tools: List[Tool], max_steps: int = 4) -> None:
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps

    def run_git_assistant(self, question: str, ctx: ToolContext, history: Optional[List[ChatMessage]] = None) -> str:
        # Heuristic fast-path for common git queries
        quick = _quick_answer_with_tools(question, ctx, self.tools)
        if quick:
            return quick

        msgs: List[ChatMessage] = [
            ChatMessage(
                role="system",
                content=SYSTEM_PROMPT + "\n\n" + INSTRUCTIONS + "\n\nTools:\n" + _tool_catalog(list(self.tools.values())),
            ),
        ]
        if history:
            # keep only user/assistant roles from history
            msgs.extend([m for m in history if m.role in {"user", "assistant"}])
        msgs.append(ChatMessage(role="user", content=question.strip()))
        last_observation: Optional[str] = None
        for _ in range(self.max_steps):
            out = self.llm.complete_chat(msgs, temperature=0.2)
            # Be robust: extract an action JSON from the response
            obj = _extract_action_json(out)
            if obj is None:
                # If model answered directly as final text
                return out.strip()
            if not isinstance(obj, dict) or "action" not in obj:
                return out.strip()
            if obj.get("action") == "final":
                return str(obj.get("output", "")).strip()
            if obj.get("action") == "tool":
                tool_name = str(obj.get("tool", "")).strip()
                tool_input = str(obj.get("input", ""))
                tool = self.tools.get(tool_name)
                if not tool:
                    # Inform model about bad tool name
                    msgs.append(ChatMessage(role="assistant", content=json.dumps(obj)))
                    msgs.append(ChatMessage(role="user", content=f"Observation: error: unknown tool '{tool_name}'"))
                    continue
                try:
                    observation = tool.run(ctx, tool_input)
                except Exception as e:  # pragma: no cover - safety
                    observation = f"error: {e}"
                last_observation = observation
                msgs.append(ChatMessage(role="assistant", content=json.dumps(obj)))
                msgs.append(ChatMessage(role="user", content=f"Observation: {observation}"))
                continue
            # Fallback: treat action as a tool name (shorthand)
            fallback_tool = str(obj.get("action", "")).strip()
            if fallback_tool in self.tools:
                tool = self.tools[fallback_tool]
                # Derive input from common fields
                tool_input = ""
                for key in ["input", "query", "args", "param", "params"]:
                    if key in obj and isinstance(obj[key], str):
                        tool_input = obj[key]
                        break
                # Specialization for common patterns (e.g., git_commit {"hash": "short"})
                if not tool_input and any(k in obj for k in ["hash", "short", "subject", "author", "date"]):
                    parts = []
                    if isinstance(obj.get("hash"), str):
                        parts.append(f"hash {obj['hash']}")
                    if obj.get("short"):
                        parts.append("short")
                    if obj.get("subject"):
                        parts.append("subject")
                    if obj.get("author"):
                        parts.append("author")
                    if obj.get("date"):
                        parts.append("date")
                    tool_input = " ".join(parts)
                try:
                    observation = tool.run(ctx, tool_input)
                except Exception as e:  # pragma: no cover - safety
                    observation = f"error: {e}"
                last_observation = observation
                msgs.append(ChatMessage(role="assistant", content=json.dumps(obj)))
                msgs.append(ChatMessage(role="user", content=f"Observation: {observation}"))
                continue
            # Unknown action: return raw
            return out.strip()
        # Fallback if no final in budget
        if last_observation:
            return last_observation.strip()
        return "Could not complete within the step limit. Please clarify your request."


def _extract_action_json(text: str) -> Optional[dict]:
    s = (text or "").strip()
    # 1) Try single-line JSON on any line
    for ln in s.splitlines():
        ln = ln.strip()
        if ln.startswith("{") and ln.endswith("}"):
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict) and isinstance(obj.get("action"), str):
                    return obj
            except Exception:
                pass
    # 2) Try fenced blocks ```json ... ```
    fence = re.compile(r"```(json)?\s*([\s\S]*?)```", re.IGNORECASE)
    for m in fence.finditer(s):
        block = m.group(2).strip()
        try:
            obj = json.loads(block)
            if isinstance(obj, dict) and isinstance(obj.get("action"), str):
                return obj
        except Exception:
            continue
    # 3) Try to locate a minimal JSON object substring
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        snippet = s[first : last + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict) and isinstance(obj.get("action"), str):
                return obj
        except Exception:
            pass
    return None


def _quick_answer_with_tools(question: str, ctx: ToolContext, tools: dict[str, Tool]) -> Optional[str]:
    q = (question or "").strip().lower()
    # Count commits
    if ("сколько" in q and "коммит" in q) or ("how many" in q and "commit" in q) or ("count" in q and "commit" in q):
        t = tools.get("git_repo")
        if t:
            try:
                return t.run(ctx, "count")
            except Exception:
                pass
    # First commit
    if ("перв" in q and "коммит" in q) or ("first" in q and "commit" in q):
        t = tools.get("git_repo")
        if t:
            try:
                return t.run(ctx, "first")
            except Exception:
                pass
    # Last commit (subject/author/date)
    if ("послед" in q and "коммит" in q) or ("last" in q and "commit" in q):
        t = tools.get("git_commit")
        if t:
            try:
                return t.run(ctx, "")
            except Exception:
                pass
    # Last commit hash
    if ("хеш" in q or "hash" in q) and ("коммит" in q or "commit" in q) and ("first" not in q and "перв" not in q):
        t = tools.get("git_commit")
        if t:
            try:
                return t.run(ctx, "hash short")
            except Exception:
                pass
    return None
