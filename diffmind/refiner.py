from __future__ import annotations

from typing import Optional

from .config import DiffMindConfig
from .providers.base import Message


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
        "Keep subject under 72 chars; use emojis and Conventional Commit type if applicable."
    )
    user = (
        "Rewrite this commit message according to FEEDBACK. Return two lines: subject then body.\n\n"
        f"Current subject: {subject}\nCurrent body:\n{body or ''}\n\nGit diff:\n{diff_text}\n\nFEEDBACK: {feedback}"
    )
    resp = client.chat.completions.create(
        model=cfg.openai_model,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    content = resp.choices[0].message.content.strip()
    lines = content.splitlines()
    new_subject = lines[0].strip()
    new_body = "\n".join(lines[1:]).strip() or None
    return Message(subject=new_subject, body=new_body)


def heuristic_refine(subject: str, body: Optional[str], feedback: str, max_len: int = 72) -> Message:
    # Very simple non-AI fallback: if 'short'/'короче' present, shorten subject; otherwise append feedback to body.
    f = feedback.lower()
    subj = subject
    b = body or ""
    if any(k in f for k in ["short", "короче", "shorter", "короч"]):
        if len(subj) > max_len:
            subj = subj[: max_len - 1].rstrip() + "…"
    else:
        note = f"Note: {feedback.strip()}" if feedback.strip() else ""
        b = (b + "\n" + note).strip() if b else note
    return Message(subject=subj, body=b or None)

