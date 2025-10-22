from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from git import Repo  # type: ignore

from .base import ChatMessage


def _now_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _sessions_dir(repo: Optional[Repo]) -> Path:
    if repo is not None:
        root = Path(repo.git_dir).parent
        base = root / ".diffmind" / "sessions"
    else:
        base = Path.home() / ".config" / "diffmind" / "sessions"
    base.mkdir(parents=True, exist_ok=True)
    return base


@dataclass
class ChatSession:
    session_id: str
    path: Path
    messages: List[ChatMessage]

    @classmethod
    def create(cls, repo: Optional[Repo]) -> "ChatSession":
        sid = _now_id()
        p = _sessions_dir(repo) / f"{sid}.json"
        return cls(session_id=sid, path=p, messages=[])

    @classmethod
    def load(cls, path: Path) -> "ChatSession":
        raw = json.loads(path.read_text("utf-8"))
        msgs = [ChatMessage(**m) for m in raw.get("messages", [])]
        sid = raw.get("session_id") or path.stem
        return cls(session_id=sid, path=path, messages=msgs)

    def save(self) -> None:
        payload = {
            "session_id": self.session_id,
            "messages": [asdict(m) for m in self.messages],
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_user(self, content: str) -> None:
        self.messages.append(ChatMessage(role="user", content=content))

    def add_assistant(self, content: str) -> None:
        self.messages.append(ChatMessage(role="assistant", content=content))

    @staticmethod
    def list_sessions(repo: Optional[Repo]) -> List[Path]:
        d = _sessions_dir(repo)
        return sorted(d.glob("*.json"), reverse=True)

