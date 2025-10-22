from __future__ import annotations

from typing import Optional

from .config import DiffMindConfig
from .providers.base import CommitMessageProvider, Message
from .providers.simple import SimpleProvider
from .utils.git import Repo, get_staged_changes, get_staged_diff_text


def _select_provider(cfg: DiffMindConfig) -> CommitMessageProvider:
    name = (cfg.provider or "simple").strip().lower()
    if name in {"simple", "builtin"}:
        return SimpleProvider()
    elif name in {"openai", "gpt"}:  # pragma: no cover - optional
        from .providers.openai_provider import OpenAIProvider

        return OpenAIProvider(cfg)
    else:
        raise ValueError(f"Unknown provider: {cfg.provider}")


def generate_commit_message(repo: Repo, cfg: Optional[DiffMindConfig] = None) -> Message:
    cfg = cfg or DiffMindConfig.load()
    provider = _select_provider(cfg)
    changes = get_staged_changes(repo)
    diff_text = get_staged_diff_text(repo, unified=0)
    msg = provider.generate(diff_text=diff_text, changes=changes)

    # Post-process subject length
    if cfg.max_subject_length and len(msg.subject) > cfg.max_subject_length:
        msg.subject = msg.subject[: cfg.max_subject_length - 1].rstrip() + "…"
    return msg

