from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..emoji import EMOJI_BY_TYPE, FALLBACK_EMOJI
from ..utils.git import FileChange
from .base import CommitMessageProvider, Message


KEYWORDS = {
    "feat": ["add", "introduce", "create", "enable", "support", "implement"],
    "fix": ["fix", "bug", "issue", "error", "fail", "broken"],
    "docs": ["doc", "readme", "guide", "docs", "spec"],
    "refactor": ["refactor", "cleanup", "restructure", "rename", "simplify"],
    "perf": ["perf", "faster", "optimiz"],
    "test": ["test", "pytest", "unittest", "coverage"],
    "style": ["format", "lint", "style", "typo"],
    "build": ["build", "deps", "dependency", "poetry", "setup", "package"],
    "ci": ["ci", "workflow", "github", "actions", "pipeline"],
    "chore": ["chore", "update", "bump"],
    "security": ["security", "vuln", "cve", "patch"],
}

FILE_HINTS = {
    "docs": [".md", "docs/", "doc/"],
    "test": ["tests/", "test_", "_test.py"],
    "build": ["pyproject.toml", "setup.py", "Dockerfile", ".github/", ".gitlab-ci"],
}


def _pick_type(changes: Sequence[FileChange], diff_text: str) -> str:
    # Hints from file paths
    type_scores: Dict[str, int] = defaultdict(int)
    for ch in changes:
        path = ch.path.lower()
        for t, patterns in FILE_HINTS.items():
            if any(p in path for p in patterns):
                type_scores[t] += 2

    # Heuristics from change types
    for ch in changes:
        if ch.change_type == "A":
            type_scores["feat"] += 2
        elif ch.change_type == "D":
            type_scores["chore"] += 1
        elif ch.change_type in {"R", "T"}:
            type_scores["refactor"] += 2

    # Keyword scan in diff
    low = diff_text.lower()
    for t, kws in KEYWORDS.items():
        type_scores[t] += sum(low.count(k) for k in kws)

    if not type_scores:
        return "chore"
    return max(type_scores.items(), key=lambda kv: kv[1])[0]


def _scope_from_paths(changes: Sequence[FileChange]) -> Optional[str]:
    # Choose most common top-level directory or module
    scopes = []
    for ch in changes:
        parts = ch.path.split("/")
        if len(parts) > 1 and not parts[0].startswith("."):
            scopes.append(parts[0])
        elif len(parts) == 1 and parts[0].endswith(".py"):
            scopes.append(Path(parts[0]).stem)
    if not scopes:
        return None
    return Counter(scopes).most_common(1)[0][0]


def _summarize(changes: Sequence[FileChange]) -> str:
    # Brief, human-ish summary
    added = [c for c in changes if c.change_type == "A"]
    modified = [c for c in changes if c.change_type == "M"]
    deleted = [c for c in changes if c.change_type == "D"]
    renamed = [c for c in changes if c.change_type == "R"]

    def name_list(items: List[FileChange], n: int = 2) -> str:
        names = [Path(c.path).name for c in items][:n]
        if len(items) > n:
            names.append(f"+{len(items) - n}")
        return ", ".join(names)

    parts: List[str] = []
    if added:
        parts.append(f"add {name_list(added)}")
    if modified:
        parts.append(f"update {name_list(modified)}")
    if deleted:
        parts.append(f"remove {name_list(deleted)}")
    if renamed:
        parts.append(f"rename {name_list(renamed)}")
    return ", ".join(parts) or "update project"



class SimpleProvider(CommitMessageProvider):
    def generate(self, diff_text: str, changes: Sequence[FileChange]) -> Message:
        ctype = _pick_type(changes, diff_text)
        scope = _scope_from_paths(changes)
        summary = _summarize(changes)

        emoji = EMOJI_BY_TYPE.get(ctype, FALLBACK_EMOJI)
        if scope:
            subject = f"{emoji} {ctype}: {summary} ({scope})"
        else:
            subject = f"{emoji} {ctype}: {summary}"

        # Body with stats
        lines = []
        for ch in changes:
            plus = f"+{ch.additions}" if ch.additions else ""
            minus = f" -{ch.deletions}" if ch.deletions else ""
            lines.append(f"- {ch.path}: {plus}{minus}".rstrip())
        body = "\n".join(lines) if lines else None
        return Message(subject=subject, body=body)
