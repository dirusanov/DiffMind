from __future__ import annotations

from typing import Optional

from .config import DiffMindConfig
from .providers.base import CommitMessageProvider, Message
from .providers.simple import SimpleProvider
from .utils.git import Repo, get_staged_changes
from .cache import current_diff_hash, load as cache_load, save as cache_save


def _select_provider(cfg: DiffMindConfig) -> CommitMessageProvider:
    name = (cfg.provider or "simple").strip().lower()
    if name in {"simple", "builtin"}:
        return SimpleProvider()
    elif name in {"openai", "gpt"}:  # pragma: no cover - optional
        from .providers.openai_provider import OpenAIProvider

        return OpenAIProvider(cfg)
    else:
        raise ValueError(f"Unknown provider: {cfg.provider}")


class NoChangesError(Exception):
    """Raised when there are no staged changes to describe."""


def generate_commit_message(
    repo: Repo,
    cfg: Optional[DiffMindConfig] = None,
    force_regen: bool = False,
) -> Message:
    cfg = cfg or DiffMindConfig.load()
    provider = _select_provider(cfg)
    changes = get_staged_changes(repo)
    # Use small context to improve heuristics without overwhelming providers
    diff_hash, diff_text = current_diff_hash(repo, unified=3)

    # Skip generation entirely when nothing staged
    if not changes or not (diff_text or "").strip():
        raise NoChangesError("no staged changes")

    # Try cache first unless forced regeneration
    if not force_regen:
        cached = cache_load(repo, diff_hash)
        if cached is not None:
            msg = cached
        else:
            msg = provider.generate(diff_text=diff_text, changes=changes)
            # Save to cache with minimal metadata
            meta = {"provider": (cfg.provider or "simple")}
            if (cfg.provider or "").lower() == "openai":
                meta.update({"model": cfg.openai_model})
            cache_save(repo, diff_hash, msg, meta)
    else:
        msg = provider.generate(diff_text=diff_text, changes=changes)
        meta = {"provider": (cfg.provider or "simple")}
        if (cfg.provider or "").lower() == "openai":
            meta.update({"model": cfg.openai_model, "refresh": True})
        cache_save(repo, diff_hash, msg, meta)

    # Post-process subject length
    if cfg.max_subject_length and len(msg.subject) > cfg.max_subject_length:
        msg.subject = msg.subject[: cfg.max_subject_length - 1].rstrip() + "…"
    # Respect emojis setting by stripping any leading emoji-like prefix
    if not cfg.emojis:
        s = msg.subject.strip()
        if s and not s[0].isalnum():
            # Remove first non-alnum token and following space
            parts = s.split(" ", 1)
            if len(parts) == 2:
                msg.subject = parts[1].strip()
            else:
                # Fallback: keep as-is if unexpected structure
                msg.subject = s
    return msg
