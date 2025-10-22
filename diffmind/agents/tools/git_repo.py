from __future__ import annotations

from typing import Optional

from git import Repo  # type: ignore

from ..base import Tool, ToolContext


class GitRepoTool(Tool):
    name = "git_repo"
    description = "Repository info: commit count and first commit. Input: 'count' or 'first'."

    def run(self, ctx: ToolContext, tool_input: str) -> str:
        repo = _ensure_repo(ctx.repo)
        if repo is None:
            return "no git repo detected"
        q = (tool_input or "").strip().lower()
        if q.startswith("count") or "count" in q:
            try:
                count = int(repo.git.rev_list("--count", "HEAD").strip())
            except Exception:
                try:
                    count = sum(1 for _ in repo.iter_commits("HEAD"))
                except Exception:
                    return "count: 0"
            return f"count: {count}"
        if q.startswith("first") or "first" in q:
            try:
                roots = repo.git.rev_list("--max-parents=0", "HEAD").split()
                if not roots:
                    return "no commits found"
                sha = roots[0]
                c = repo.commit(sha)
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
            return f"hash_short: {short}\nhash: {full}\nsubject: {subject}\nauthor: {author}\ndate: {date}"
        return "unknown query"


def _ensure_repo(repo: Optional[Repo]) -> Optional[Repo]:
    if repo is not None:
        return repo
    try:
        return Repo(search_parent_directories=True)
    except Exception:
        return None

