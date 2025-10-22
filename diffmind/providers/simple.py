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

    # Heuristics from change types & size
    total_add = sum(c.additions for c in changes)
    total_del = sum(c.deletions for c in changes)
    for ch in changes:
        if ch.change_type in {"A", "C"}:
            type_scores["feat"] += 2
        elif ch.change_type == "D":
            type_scores["chore"] += 1
        elif ch.change_type in {"R", "T"}:
            type_scores["refactor"] += 2

    # Diff content patterns
    low = diff_text.lower()
    for t, kws in KEYWORDS.items():
        type_scores[t] += sum(low.count(k) for k in kws)

    # Structure-aware boosts
    has_new_defs = "+def " in low or "+class " in low
    if total_add > 0 and total_del == 0 and has_new_defs:
        type_scores["feat"] += 3
    if total_add > 0 and total_del > 0 and (total_add + total_del) >= 20:
        type_scores["refactor"] += 3
    if "todo" in low or "fixme" in low:
        type_scores["chore"] += 1

    if not type_scores:
        return "chore"
    return max(type_scores.items(), key=lambda kv: kv[1])[0]


def _scope_from_paths(changes: Sequence[FileChange]) -> Optional[str]:
    # Choose most common meaningful directory or module (skip generic names)
    skip = {"src", "app", "apps", "lib", "core", "pkg", "package", "packages"}
    scopes = []
    for ch in changes:
        parts = [p for p in ch.path.split("/") if p]
        # find first non-generic directory
        chosen = None
        for p in parts[:-1]:  # skip filename
            if p.startswith("."):
                continue
            if p.lower() in skip:
                continue
            chosen = p
            break
        if chosen:
            scopes.append(chosen)
        else:
            # fallback to top-level dir or file stem
            if len(parts) > 1 and not parts[0].startswith("."):
                scopes.append(parts[0])
            elif parts:
                name = parts[-1]
                scopes.append(Path(name).stem)
    if not scopes:
        return None
    return Counter(scopes).most_common(1)[0][0]


def _summarize(changes: Sequence[FileChange]) -> str:
    # Brief, human-ish summary
    # Treat type-changes ('T') as modifications; copies ('C') as additions
    added = [c for c in changes if c.change_type in {"A", "C"}]
    modified = [c for c in changes if c.change_type in {"M", "T"}]
    deleted = [c for c in changes if c.change_type == "D"]
    renamed = [c for c in changes if c.change_type == "R"]

    def name_list(items: List[FileChange], n: int = 2) -> str:
        # Prefer concise stems: transaction.py -> transaction
        names = [Path(c.path).stem for c in items][:n]
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
    return ", ".join(parts) or "update code"



def _analyze_diff(diff_text: str) -> Dict[str, object]:
    endpoints: List[str] = []
    added_funcs: List[str] = []
    added_classes: List[str] = []
    guards: List[str] = []

    ep_decorator = re.compile(r"^\+\s*@(?:app|router)\.(get|post|put|patch|delete)\(\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
    add_api_route = re.compile(r"^\+\s*router\.add_api_route\(\s*['\"]([^'\"]+)['\"].*?methods\s*=\s*\[([^\]]+)\]", re.IGNORECASE)
    func_re = re.compile(r"^\+\s*def\s+([A-Za-z_][\w]*)\s*\(")
    class_re = re.compile(r"^\+\s*class\s+([A-Za-z_][\w]*)\b")
    except_re = re.compile(r"^\+\s*except\s+([A-Za-z_][\w]*)")
    none_guard_re = re.compile(r"^\+\s*if\s+.*?\bis\s+None\b|^\+\s*if\s+.*?\bis\s+not\s+None\b")
    try_re = re.compile(r"^\+\s*try\s*:\s*$")

    for raw in diff_text.splitlines():
        # Skip file headers
        if raw.startswith("+++") or raw.startswith("---"):
            continue
        if not raw.startswith("+"):
            continue

        if m := ep_decorator.match(raw):
            method, path = m.group(1).upper(), m.group(2)
            endpoints.append(f"{method} {path}")
            continue
        if m := add_api_route.match(raw):
            path = m.group(1)
            methods = ",".join([x.strip().strip("'\"").upper() for x in m.group(2).split(",") if x.strip()])
            endpoints.append(f"{methods} {path}")
            continue
        if m := func_re.match(raw):
            added_funcs.append(m.group(1))
            continue
        if m := class_re.match(raw):
            added_classes.append(m.group(1))
            continue
        if m := except_re.match(raw):
            guards.append(m.group(1))
            continue
        if try_re.match(raw):
            guards.append("try/except")
            continue
        if none_guard_re.match(raw):
            guards.append("None check")
            continue

    def _dedup(xs: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for x in xs:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    return {
        "endpoints": _dedup(endpoints),
        "added_symbols": _dedup([*(f + "()" for f in added_funcs), *added_classes]),
        "guards": _dedup(guards),
    }


def _insight_summary(insights: Dict[str, object]) -> Optional[str]:
    eps: List[str] = insights.get("endpoints") or []  # type: ignore
    syms: List[str] = insights.get("added_symbols") or []  # type: ignore
    guards: List[str] = insights.get("guards") or []  # type: ignore

    if eps:
        if len(eps) == 1:
            return f"add {eps[0]}"
        if len(eps) >= 2:
            return f"add endpoints {eps[0]}, {eps[1]}"
    if syms:
        if len(syms) == 1:
            return f"add {syms[0]}"
        return f"add {syms[0]}, {syms[1]}"
    if guards:
        specific = next((g for g in guards if g != "try/except"), None)
        if specific:
            return f"handle {specific.lower()}"
        return "add error handling"
    return None


class SimpleProvider(CommitMessageProvider):
    def generate(self, diff_text: str, changes: Sequence[FileChange]) -> Message:
        ctype = _pick_type(changes, diff_text)
        scope = _scope_from_paths(changes)

        insights = _analyze_diff(diff_text)
        insight_summary = _insight_summary(insights)
        summary = insight_summary or _summarize(changes)

        if (not insights.get("endpoints") and not insights.get("added_symbols")) and insights.get("guards"):
            ctype = "fix"

        emoji = EMOJI_BY_TYPE.get(ctype, FALLBACK_EMOJI)
        if scope:
            subject = f"{emoji} {ctype}: {summary} ({scope})"
        else:
            subject = f"{emoji} {ctype}: {summary}"

        def _size(c: FileChange) -> int:
            return (c.additions or 0) + (c.deletions or 0)

        lines = []
        for ch in sorted(changes, key=_size, reverse=True)[:6]:
            plus = f"+{ch.additions}" if ch.additions else ""
            minus = f" -{ch.deletions}" if ch.deletions else ""
            lines.append(f"- {ch.path}: {plus}{minus}".rstrip())
        body = "\n".join(lines) if lines else None
        return Message(subject=subject, body=body)
