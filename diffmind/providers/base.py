from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..utils.git import FileChange


@dataclass
class Message:
    subject: str
    body: Optional[str] = None


class CommitMessageProvider(ABC):
    @abstractmethod
    def generate(self, diff_text: str, changes: Sequence[FileChange]) -> Message:  # pragma: no cover - interface
        raise NotImplementedError


class ProviderConfigError(Exception):
    """Raised when a provider is selected but misconfigured (e.g., missing API key)."""

