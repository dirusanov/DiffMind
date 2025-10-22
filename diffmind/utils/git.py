from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from git import Repo, InvalidGitRepositoryError


EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"  # magic empty tree


def find_repo(start: Optional[Path] = None) -> Repo:
    try:
        return Repo(search_parent_directories=True, path=str(start or Path.cwd()))
    except InvalidGitRepositoryError as e:  # pragma: no cover
        raise RuntimeError("Not inside a Git repository") from e


def repo_root(repo: Repo) -> Path:
    return Path(repo.git.rev_parse("--show-toplevel"))


def is_initial_commit(repo: Repo) -> bool:
    try:
        _ = repo.head.commit
        return False
    except Exception:
        return True


def get_staged_diff_text(repo: Repo, unified: int = 3) -> str:
    args = [
        "git",
        "-c",
        "core.pager=cat",
        "diff",
        "--cached",
        f"-U{unified}",
        "--no-color",
    ]
    return subprocess.run(args, capture_output=True, text=True, cwd=str(repo.working_dir)).stdout


@dataclass
class FileChange:
    path: str
    change_type: str  # 'A', 'M', 'D', 'R', 'T'
    additions: int
    deletions: int


def get_staged_changes(repo: Repo) -> List[FileChange]:
    """Return a summary of staged changes vs HEAD (or empty tree)."""
    base = EMPTY_TREE if is_initial_commit(repo) else "HEAD"
    diff_index = repo.index.diff(base, create_patch=True, cached=True)
    items: List[FileChange] = []
    for d in diff_index:
        stats = d.diff.decode(errors="ignore") if isinstance(d.diff, bytes) else str(d.diff)
        additions = stats.count("\n+")
        deletions = stats.count("\n-")
        items.append(
            FileChange(
                path=(d.b_path or d.a_path or ""),
                change_type=d.change_type,
                additions=max(0, additions - 1),  # ignore file header line starting with '+'
                deletions=max(0, deletions - 1),
            )
        )
    return items


def has_non_comment_content(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:  # pragma: no cover
        return False
    for line in text.splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            return True
    return False


def write_commit_message(path: Path, subject: str, body: Optional[str]) -> None:
    lines = [subject.strip()]
    if body:
        lines.append("")
        lines.extend(body.rstrip().splitlines())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

