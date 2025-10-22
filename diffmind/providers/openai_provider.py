from __future__ import annotations

import os
from typing import Sequence

from ..config import DiffMindConfig
from ..emoji import EMOJI_BY_TYPE, FALLBACK_EMOJI
from ..utils.git import FileChange
from .base import CommitMessageProvider, Message, ProviderConfigError


class OpenAIProvider(CommitMessageProvider):
    def __init__(self, cfg: DiffMindConfig):
        self.cfg = cfg
        try:
            import openai  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ProviderConfigError(
                "OpenAI provider selected but 'openai' package is not installed.\n"
                "Install with: 'pip install openai' or 'poetry add openai --group ai'"
            ) from e

        api_key = cfg.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderConfigError(
                "OpenAI API key is missing. Set OPENAI_API_KEY env or configure via 'diffmind config wizard'."
            )
        self._client = openai.OpenAI(api_key=api_key, base_url=cfg.openai_base_url)

    def generate(self, diff_text: str, changes: Sequence[FileChange]) -> Message:
        # Keep prompt compact; ask for subject + body with emojis and conventional type
        sys = (
            "You generate excellent, concise Git commit messages with emojis and "
            "Conventional Commit types. Subject max 72 chars. Body as bullet list with file stats."
        )
        user = (
            "Create a commit message from this staged git diff. "
            "Return as two lines: first the subject, then the body.\n\n" + diff_text
        )

        resp = self._client.chat.completions.create(
            model=self.cfg.openai_model,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.3,
        )
        content = resp.choices[0].message.content.strip()
        lines = content.splitlines()
        subject = lines[0].strip()
        body = "\n".join(lines[1:]).strip() or None
        return Message(subject=subject, body=body)
