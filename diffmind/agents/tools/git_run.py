from __future__ import annotations

import shlex
import subprocess
from typing import Optional

from git import Repo  # type: ignore

from ..base import Tool, ToolContext


class GitRunTool(Tool):
    name = "git_run"
    description = (
        "Execute a safe git command in the current repository. "
        "Input: a single command WITHOUT the leading 'git', e.g. 'add -A', 'commit -m \"msg\"'. "
        "Supported: add, commit, restore, reset, checkout, switch, stash, rm, mv, tag."
    )

    _WHITELIST = {
        "add",
        "commit",
        "restore",
        "reset",
        "checkout",
        "switch",
        "stash",
        "rm",
        "mv",
        "tag",
    }

    def run(self, ctx: ToolContext, tool_input: str) -> str:
        repo = _ensure_repo(ctx.repo)
        if repo is None:
            return "error: no git repo detected"

        cmd = (tool_input or "").strip()
        if not cmd:
            return "error: empty command"
        # Strip leading 'git ' if present
        if cmd.lower().startswith("git "):
            cmd = cmd[4:].strip()
        try:
            parts = shlex.split(cmd)
        except Exception:
            return "error: could not parse command"
        if not parts:
            return "error: empty command"
        verb = parts[0].lower()
        if verb not in self._WHITELIST:
            return f"error: command '{verb}' is not allowed"
        try:
            proc = subprocess.run(
                ["git", *parts],
                cwd=repo.working_tree_dir or None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            return f"error: {e}"
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if proc.returncode != 0:
            return (f"exit_code: {proc.returncode}\n" + (err or out or "error")).strip()
        # success
        return out or "ok"


def _ensure_repo(repo: Optional[Repo]) -> Optional[Repo]:
    if repo is not None:
        return repo
    try:
        return Repo(search_parent_directories=True)
    except Exception:
        return None
