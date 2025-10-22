from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from git import Repo  # type: ignore

from .providers.base import Message
from .utils.git import get_staged_diff_text


def _cache_dir(repo: Repo) -> Path:
    root = Path(repo.git_dir).parent
    d = root / ".diffmind" / "cache" / "commits"
    d.mkdir(parents=True, exist_ok=True)
    return d


def current_diff_hash(repo: Repo, unified: int = 3) -> Tuple[str, str]:
    """Return (hash, diff_text) of the current staged diff.

    The hash is SHA-256 of the text returned by get_staged_diff_text with the
    same unified context that providers see.
    """
    diff_text = get_staged_diff_text(repo, unified=unified)
    h = hashlib.sha256(diff_text.encode("utf-8")).hexdigest()
    return h, diff_text


def load(repo: Repo, diff_hash: str) -> Optional[Message]:
    path = _cache_dir(repo) / f"{diff_hash}.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:
        return None
    subject = raw.get("subject") or ""
    body = raw.get("body") or None
    if not subject:
        return None
    return Message(subject=subject, body=body)


def save(repo: Repo, diff_hash: str, msg: Message, meta: Optional[Dict[str, Any]] = None) -> Path:
    path = _cache_dir(repo) / f"{diff_hash}.json"
    payload: Dict[str, Any] = {
        "subject": msg.subject,
        "body": msg.body,
        "created_at": int(time.time()),
    }
    if meta:
        payload.update(meta)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_for_current_diff(repo: Repo, msg: Message, meta: Optional[Dict[str, Any]] = None) -> Path:
    h, _ = current_diff_hash(repo)
    return save(repo, h, msg, meta)

