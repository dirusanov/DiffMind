from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

try:  # Python 3.11+
    import tomllib  # type: ignore
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore


USER_CONFIG_PATH = Path.home() / ".config" / "diffmind" / "config.toml"

DEFAULT_CONFIG_LOCATIONS = [
    Path(".diffmind.toml"),
    Path.cwd() / ".diffmind.toml",
    USER_CONFIG_PATH,
]


@dataclass
class DiffMindConfig:
    provider: str = "auto"  # auto | simple | openai
    conventional: bool = True
    emojis: bool = True
    max_subject_length: int = 72
    scope_strategy: str = "topdir"  # topdir | none
    language: str = "auto"  # auto | en | ru

    openai_model: str = "gpt-4o-mini"
    openai_base_url: Optional[str] = None
    openai_api_key: Optional[str] = None
    auto_install_openai: bool = False

    @classmethod
    def load(cls, overrides: Optional[Dict[str, str]] = None) -> "DiffMindConfig":
        data: Dict[str, object] = {}
        for path in DEFAULT_CONFIG_LOCATIONS:
            if path.is_file():
                try:
                    with path.open("rb") as f:
                        doc = tomllib.load(f)
                        if isinstance(doc, dict):
                            data.update(doc)
                except Exception:
                    pass

        # Env overrides
        env_provider = os.getenv("DIFFMIND_PROVIDER")
        if env_provider:
            data["provider"] = env_provider
        env_openai_key = os.getenv("OPENAI_API_KEY")
        if env_openai_key:
            data["openai_api_key"] = env_openai_key
        env_emojis = os.getenv("DIFFMIND_EMOJIS")
        if env_emojis is not None:
            data["emojis"] = env_emojis.lower() in {"1", "true", "yes", "on"}
        env_conventional = os.getenv("DIFFMIND_CONVENTIONAL")
        if env_conventional is not None:
            data["conventional"] = env_conventional.lower() in {"1", "true", "yes", "on"}
        env_auto_install = os.getenv("DIFFMIND_AUTO_INSTALL_OPENAI")
        if env_auto_install is not None:
            data["auto_install_openai"] = env_auto_install.lower() in {"1", "true", "yes", "on"}

        if overrides:
            data.update(overrides)

        # Auto-detect provider if not explicitly set or set to 'auto'
        explicit_provider = "provider" in data and str(data["provider"]).strip() not in {"", "auto"}
        desired_provider = str(data.get("provider", "auto")).strip()
        if not explicit_provider or desired_provider == "auto":
            # Prefer OpenAI when key exists and package is installed
            has_key = bool(data.get("openai_api_key") or os.getenv("OPENAI_API_KEY"))
            has_pkg = False
            try:
                import openai  # type: ignore

                has_pkg = True
            except Exception:
                has_pkg = False
            data["provider"] = "openai" if has_key and has_pkg else "simple"

        cfg = cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
        return cfg

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        # remove None values to keep config clean
        return {k: v for k, v in d.items() if v is not None}


def preferred_write_path(repo_root: Optional[Path], scope: str = "user") -> Path:
    if scope == "repo" and repo_root:
        return repo_root / ".diffmind.toml"
    return USER_CONFIG_PATH


def save_config(cfg: DiffMindConfig, path: Optional[Path] = None) -> Path:
    path = path or USER_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _to_toml(cfg.to_dict())
    path.write_text(content, encoding="utf-8")
    return path


def _to_toml(d: Dict[str, object]) -> str:
    lines = []
    for k, v in d.items():
        if isinstance(v, bool):
            lines.append(f"{k} = {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            lines.append(f"{k} = {v}")
        else:
            s = str(v).replace("\n", "\\n").replace("\r", "")
            s = s.replace("\"", "\\\"")
            lines.append(f"{k} = \"{s}\"")
    return "\n".join(lines) + "\n"
