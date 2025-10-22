from __future__ import annotations

from typing import Optional, Dict, List, Tuple

from .config import DiffMindConfig
from .providers.base import Message
from .agents import AgentOrchestrator, OpenAIChat, NoopLLM, GitDiffTool, CalculatorTool, GitCommitTool, GitRepoTool
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
        "Keep subject under 72 chars; use emojis and Conventional Commit type if applicable."
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

    tools = [GitDiffTool(), GitCommitTool(), GitRepoTool(), CalculatorTool()]
    agent = AgentOrchestrator(llm, tools, max_steps=4)
    ctx = ToolContext(cfg=cfg, repo=None)
    try:
        answer = agent.run_git_assistant(question, ctx)
        return answer.strip()
    except Exception:
        return None


def heuristic_refine(
    subject: str,
    body: Optional[str],
    feedback: str,
    max_len: int = 72,
    diff_text: Optional[str] = None,
) -> Message:
    """Heuristic refinement without OpenAI.

    Supports intents: make more detailed, shorter, change type to fix/refactor/feat, remove emoji.
    When detail requested and diff is available, extract endpoints, added symbols, and summarize changed files.
    """
    f = (feedback or "").strip()
    fl = f.lower()

    # Parse current subject structure: [emoji] type: summary (scope)
    emoji = ""
    ctype = ""
    summary = subject
    scope = ""
    s = subject.strip()
    # Emoji prefix (single grapheme like '✨' or ':emoji:')
    if s and not s[0].isalnum():
        parts = s.split(" ", 2)
        if len(parts) >= 2 and parts[1].endswith(":"):
            emoji, rest = parts[0], " ".join(parts[1:])
            s = rest
    # Type prefix
    if ":" in s:
        maybe_type, rest = s.split(":", 1)
        ctype = maybe_type.strip()
        summary = rest.strip()
    # Scope suffix
    if summary.endswith(")") and "(" in summary:
        i = summary.rfind("(")
        scope = summary[i + 1 : -1]
        summary = summary[:i].strip()

    def _trim(x: str) -> str:
        return x[: max_len - 1].rstrip() + "…" if len(x) > max_len else x

    # Intent: shorter
    if any(k in fl for k in ["short", "shorter", "короч", "короче", "кратче", "кратко"]):
        new_subject = _trim(subject)
        return Message(subject=new_subject, body=(body or None))

    # Intent: change type
    for t in ["feat", "fix", "refactor", "chore", "docs", "perf", "test", "build", "ci", "style", "security"]:
        if f"{t}" in fl and any(k in fl for k in ["type", "тип", t]):
            ctype = t
            break

    # Intent: remove emoji
    if any(k in fl for k in ["no emoji", "без эмодзи", "без эмоджи", "remove emoji"]):
        emoji = ""

    # Intent: more detail
    wants_detail = any(k in fl for k in ["detail", "подроб", "подробнее", "распиши", "подробней", "более подробно"])

    new_summary = summary
    new_body_lines: List[str] = []

    if wants_detail and diff_text:
        details = _extract_details(diff_text)
        # Subject: prefer endpoints or new symbols
        if details["endpoints"]:
            eps = details["endpoints"]
            new_summary = f"add {eps[0]}" if len(eps) == 1 else f"add endpoints {eps[0]}, {eps[1]}"
            ctype = ctype or "feat"
        elif details["symbols"]:
            syms = details["symbols"]
            new_summary = f"add {syms[0]}" if len(syms) == 1 else f"add {syms[0]}, {syms[1]}"
            ctype = ctype or "feat"
        elif details["guards"] and not details["symbols"]:
            g = next((g for g in details["guards"] if g != "try/except"), None)
            new_summary = f"handle {g.lower()}" if g else "add error handling"
            ctype = "fix"
        else:
            # If only deletions across files, prefer cleanup
            try:
                add_total, del_total = [int(x) for x in details.get("totals", ["0", "0"])]
            except Exception:
                add_total, del_total = 0, 0
            if add_total == 0 and del_total > 0:
                ctype = "chore"
                # pick top file
                top_file = None
                if details["files"]:
                    # lines in form "- path: +A -D"
                    first = details["files"][0]
                    colon = first.find(":")
                    if colon > 2:
                        top_file = first[2:colon].strip()
                if top_file:
                    import os as _os
                    stem = _os.path.splitext(_os.path.basename(top_file))[0]
                    new_summary = f"cleanup {stem}"
                else:
                    new_summary = "cleanup code"

        # Body: top files and highlights
        for line in details["files"]:
            new_body_lines.append(line)
        if details["symbols"]:
            new_body_lines.append("Symbols: " + ", ".join(details["symbols"][:5]))
        if details["endpoints"]:
            new_body_lines.append("Endpoints: " + ", ".join(details["endpoints"][:5]))
        if details["guards"]:
            new_body_lines.append("Guards: " + ", ".join(details["guards"][:3]))

    # Rebuild subject
    rebuilt = (
        ((emoji + " ") if emoji else "")
        + (ctype + ": " if ctype else "")
        + (new_summary or summary)
        + (f" ({scope})" if scope else "")
    )
    rebuilt = _trim(rebuilt)

    # Body: merge with previous body
    new_body = None
    if new_body_lines:
        stats = "\n".join(new_body_lines)
        if body:
            new_body = (body.rstrip() + "\n" + stats).strip()
        else:
            new_body = stats
    else:
        new_body = body or None

    # If no explicit intent matched and no diff, append note
    if not wants_detail and subject == rebuilt and (new_body or "") == (body or ""):
        note = f"Note: {f}" if f else ""
        new_body = ((body or "") + ("\n" + note if note else "")).strip() or None

    return Message(subject=rebuilt, body=new_body)


def _extract_details(diff_text: str) -> Dict[str, List[str]]:
    """Extract endpoints, symbols, guards and top files from a diff."""
    endpoints: List[str] = []
    added_funcs: List[str] = []
    added_classes: List[str] = []
    guards: List[str] = []
    file_changes: Dict[str, Tuple[int, int]] = {}

    import re as _re

    ep_decorator = _re.compile(r"^\+\s*@(?:app|router)\.(get|post|put|patch|delete)\(\s*['\"]([^'\"]+)['\"]", _re.IGNORECASE)
    add_api_route = _re.compile(r"^\+\s*router\.add_api_route\(\s*['\"]([^'\"]+)['\"].*?methods\s*=\s*\[([^\]]+)\]", _re.IGNORECASE)
    func_re = _re.compile(r"^\+\s*def\s+([A-Za-z_][\w]*)\s*\(")
    class_re = _re.compile(r"^\+\s*class\s+([A-Za-z_][\w]*)\b")
    except_re = _re.compile(r"^\+\s*except\s+([A-Za-z_][\w]*)")
    none_guard_re = _re.compile(r"^\+\s*if\s+.*?\bis\s+None\b|^\+\s*if\s+.*?\bis\s+not\s+None\b")
    try_re = _re.compile(r"^\+\s*try\s*:\s*$")
    hunk_file_re = _re.compile(r"^\+\+\+\s+b/(.+)$|^---\s+a/(.+)$")

    current_file = None
    for raw in diff_text.splitlines():
        if raw.startswith("+++") or raw.startswith("---"):
            m = hunk_file_re.match(raw)
            if m:
                current_file = m.group(1) or m.group(2)
            continue
        if raw.startswith("@@"):
            continue
        if not raw.startswith("+") and not raw.startswith("-"):
            continue

        # Track file change stats
        if current_file:
            add, delete = file_changes.get(current_file, (0, 0))
            if raw.startswith("+"):
                add += 1
            elif raw.startswith("-"):
                delete += 1
            file_changes[current_file] = (add, delete)

        if not raw.startswith("+"):
            continue

        if (m := ep_decorator.match(raw)):
            method, path = m.group(1).upper(), m.group(2)
            endpoints.append(f"{method} {path}")
            continue
        if (m := add_api_route.match(raw)):
            path = m.group(1)
            methods = ",".join([x.strip().strip("'\"").upper() for x in m.group(2).split(",") if x.strip()])
            endpoints.append(f"{methods} {path}")
            continue
        if (m := func_re.match(raw)):
            added_funcs.append(m.group(1))
            continue
        if (m := class_re.match(raw)):
            added_classes.append(m.group(1))
            continue
        if (m := except_re.match(raw)):
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

    files = sorted(file_changes.items(), key=lambda kv: (kv[1][0] + kv[1][1]), reverse=True)
    file_lines = [f"- {path}: +{a} -{d}" for path, (a, d) in files[:6]]
    total_add = sum(a for _, (a, _) in files)
    total_del = sum(d for _, (_, d) in files)

    return {
        "endpoints": _dedup(endpoints),
        "symbols": _dedup([*(f + "()" for f in added_funcs), *added_classes]),
        "guards": _dedup(guards),
        "files": file_lines,
        "totals": [str(total_add), str(total_del)],
    }
