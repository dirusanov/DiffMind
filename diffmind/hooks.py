from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from typing import Optional

from git import Repo

from .config import DiffMindConfig
from .generator import generate_commit_message, NoChangesError
from .utils.git import has_non_comment_content, write_commit_message


HOOK_NAME = "prepare-commit-msg"


def hook_file_path(repo: Repo) -> Path:
    return Path(repo.git_dir) / "hooks" / HOOK_NAME


def install_hook(repo: Repo) -> Path:
    path = hook_file_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    script = f"""#!/usr/bin/env sh
# DiffMind auto-generated hook: {HOOK_NAME}

set -e

if command -v diffmind >/dev/null 2>&1; then
  exec diffmind hook run "$@"
fi

# Fallback to python module
PYTHON_BIN="${{PYTHON_BIN:-python3}}"
exec "$PYTHON_BIN" -m diffmind.cli hook run "$@"
"""
    path.write_text(script, encoding="utf-8")
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def uninstall_hook(repo: Repo) -> None:
    path = hook_file_path(repo)
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def run_prepare_commit_msg(repo: Repo, message_file: str, commit_source: Optional[str], sha1: Optional[str]) -> int:
    # If message already has content (non-comment), respect it and exit
    path = Path(message_file)
    if has_non_comment_content(path):
        return 0

    cfg = DiffMindConfig.load()
    try:
        msg = generate_commit_message(repo, cfg)
    except NoChangesError:
        return 0
    write_commit_message(path, msg.subject, msg.body)
    return 0
