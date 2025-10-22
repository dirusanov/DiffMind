from __future__ import annotations

from typing import Optional

from git import Repo  # type: ignore

from ..base import Tool, ToolContext


class GitCommitTool(Tool):
    name = "git_commit"
    description = "Get latest commit info (hash short/full, subject, author, date). Input: optional 'hash', 'short', 'subject', 'author', 'date', or empty."

    def run(self, ctx: ToolContext, tool_input: str) -> str:
        repo = _ensure_repo(ctx.repo)
        if repo is None:
            return "no git repo detected"
        try:
            c = repo.head.commit
        except Exception:
            return "no commits found"
        short = c.hexsha[:7]
        full = c.hexsha
        subject = (c.message.splitlines()[0] if c.message else "").strip()
        author = getattr(getattr(c, "author", None), "name", "") or ""
        try:
            date = c.authored_datetime.isoformat()
        except Exception:
            date = ""
        q = (tool_input or "").strip().lower()
        if q.startswith("hash"):
            if "short" in q:
                return f"hash_short: {short}"
            return f"hash_short: {short}\nhash: {full}"
        if q.startswith("short"):
            return f"hash_short: {short}"
        if q.startswith("subject"):
            return f"subject: {subject}"
        if q.startswith("author"):
            return f"author: {author}"
        if q.startswith("date"):
            return f"date: {date}"
        return f"hash_short: {short}\nhash: {full}\nsubject: {subject}\nauthor: {author}\ndate: {date}"


def _ensure_repo(repo: Optional[Repo]) -> Optional[Repo]:
    if repo is not None:
        return repo
    try:
        return Repo(search_parent_directories=True)
    except Exception:
        return None
