from __future__ import annotations

from typing import Optional, Dict, List

from .config import DiffMindConfig
from .providers.base import Message
from .agents import AgentOrchestrator, OpenAIChat, NoopLLM, GitDiffTool, CalculatorTool, GitCommitTool, GitRepoTool, GitRunTool
from .agents.base import ToolContext


def refine_with_openai(subject: str, body: Optional[str], diff_text: str, feedback: str, cfg: DiffMindConfig) -> Optional[Message]:  # pragma: no cover - optional
    try:
        import openai  # type: ignore
    except Exception:
        return None

    api_key = cfg.openai_api_key
    if not api_key:
        import os

        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    client = openai.OpenAI(api_key=api_key, base_url=cfg.openai_base_url)
    sys = (
        "You are a helpful assistant that rewrites Git commit messages based on user feedback. "
        + ("Use emojis; " if cfg.emojis else "Avoid emojis; ")
        + "keep subject under 72 chars; use Conventional Commit type when applicable."
    )
    user = (
        "Rewrite this commit message according to FEEDBACK. Return two lines: subject then body.\n\n"
        f"Current subject: {subject}\nCurrent body:\n{body or ''}\n\nGit diff:\n{diff_text}\n\nFEEDBACK: {feedback}"
    )
    try:
        resp = client.chat.completions.create(
            model=cfg.openai_model,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.2,
        )
    except Exception as e:
        m = str(e).lower()
        if "temperature" in m and ("unsupported" in m or "does not support" in m):
            resp = client.chat.completions.create(
                model=cfg.openai_model,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            )
        else:
            raise
    content = resp.choices[0].message.content.strip()
    lines = content.splitlines()
    new_subject = lines[0].strip()
    new_body = "\n".join(lines[1:]).strip() or None
    return Message(subject=new_subject, body=new_body)


def answer_git_question(question: str, cfg: DiffMindConfig) -> Optional[str]:  # pragma: no cover - optional
    """Answer a Git-related question via an agent loop with tool use.

    Falls back to None when provider is misconfigured (CLI will show guidance).
    """
    provider = (cfg.provider or "simple").lower()
    llm = None
    if provider == "openai":
        import os as _os

        key = cfg.openai_api_key or _os.getenv("OPENAI_API_KEY")
        if not key:
            return None
        try:
            llm = OpenAIChat(api_key=key, base_url=cfg.openai_base_url, model=cfg.openai_model)
        except Exception:
            return None
    else:
        # Simple mode stub LLM that returns a friendly final answer
        llm = NoopLLM()

    tools = [GitDiffTool(), GitCommitTool(), GitRepoTool(), GitRunTool(), CalculatorTool()]
    agent = AgentOrchestrator(llm, tools, max_steps=4)
    ctx = ToolContext(cfg=cfg, repo=None)
    try:
        answer = agent.run_git_assistant(question, ctx)
        return answer.strip()
    except Exception:
        return None
