from __future__ import annotations

from typing import Optional

from git import Repo  # type: ignore

from ..base import Tool, ToolContext
from ...utils.git import get_staged_diff_text


class GitDiffTool(Tool):
    name = "git_diff"
    description = "Return staged git diff with small context and repo info (branch)."

    def run(self, ctx: ToolContext, tool_input: str) -> str:  # noqa: ARG002
        repo = _ensure_repo(ctx.repo)
        if repo is None:
            return "no git repo detected"
        diff = get_staged_diff_text(repo, unified=3) or "(no staged diff)"
        try:
            branch = repo.active_branch.name
        except Exception:
            branch = "(detached)"
        return f"branch: {branch}\n{diff}"


def _ensure_repo(repo: Optional[Repo]) -> Optional[Repo]:
    if repo is not None:
        return repo
    try:
        return Repo(search_parent_directories=True)
    except Exception:
        return None

