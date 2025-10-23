from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

import typer
from git import Repo
from rich.console import Console
from rich.prompt import Confirm
from rich.panel import Panel
from rich import box

from . import __version__
from .config import DiffMindConfig, DEFAULT_CONFIG_LOCATIONS, preferred_write_path, save_config
from .generator import generate_commit_message, NoChangesError
from .cache import save_for_current_diff as _cache_save_current
from .hooks import HOOK_NAME, install_hook, run_prepare_commit_msg, uninstall_hook
from .ui import banner, print_message, tip, openai_help_panel, status_panel, simple_mode_warning
from .agents import AgentOrchestrator, OpenAIChat, NoopLLM, GitDiffTool, CalculatorTool, GitCommitTool, GitRepoTool
from .agents.base import ToolContext, ChatMessage
from .agents.session import ChatSession
from .providers.base import ProviderConfigError, Message
from .refiner import refine_with_openai, heuristic_refine, answer_git_question


console = Console()
# Disable Typer rich help formatting to avoid Click 8.1+ API mismatch
app = typer.Typer(
    add_completion=False,
    help="AI commit message generator (CLI + git hook)",
    rich_markup_mode=None,
)
hook_app = typer.Typer(help="Manage git hooks", rich_markup_mode=None)
config_app = typer.Typer(help="Configure DiffMind", rich_markup_mode=None)
app.add_typer(hook_app, name="hook")
app.add_typer(config_app, name="config")

# Sentinels for prompt handling
SENT_SLASH = "\x00__SLASH_START__\x00"
SENT_MENU = "\x00__OPEN_MENU__\x00"


# Internal: minimal wrapper to make InquirerPy select menus cancellable via Esc and Ctrl+Q
def _select_menu(message: str, choices, default=None):
    last_exc: Exception | None = None
    try:
        from InquirerPy import inquirer as _inq  # type: ignore

        # Ensure Esc does NOT abort; only Ctrl+C cancels.
        try:
            return _inq.select(
                message=message,
                choices=choices,
                default=default,
                keybindings={"abort": [{"key": "c-c"}]},
            ).execute()
        except (TypeError, ValueError, AttributeError):
            # Older InquirerPy without keybindings support
            return _inq.select(message=message, choices=choices, default=default).execute()
    except KeyboardInterrupt:
        # Treat Ctrl+C as cancel
        return None
    except Exception as exc:  # pragma: no cover - fallback path
        last_exc = exc

    # Inline fallback that works even when InquirerPy is missing/broken.
    try:
        return _inline_arrow_menu(message, choices, default=default)
    except KeyboardInterrupt:
        return None
    except Exception:
        if last_exc is not None:
            raise last_exc
        # Propagate when both primary and fallback failed
        raise


def _inline_arrow_menu(message: str, choices, default=None):
    """Render a lightweight inline selector with arrow-key navigation."""
    import sys as _sys
    import termios as _termios
    import tty as _tty
    import select as _select

    def _normalize_choice(choice):
        if isinstance(choice, dict):
            name = str(choice.get("name") or choice.get("value") or choice)
            value = choice.get("value", choice.get("name", choice))
            disabled = bool(choice.get("disabled"))
            return {"name": name, "value": value, "disabled": disabled}
        return {"name": str(choice), "value": choice, "disabled": False}

    base_choices = list(choices or [])
    normalized = [_normalize_choice(ch) for ch in base_choices]
    if not normalized:
        return None

    stdin = _sys.stdin
    stdout = _sys.stdout
    if not hasattr(stdin, "fileno"):
        raise RuntimeError("stdin has no fileno()")
    try:
        fd = stdin.fileno()
    except (AttributeError, ValueError, OSError):
        raise RuntimeError("stdin is not a valid TTY")

    if fd < 0 or not os.isatty(fd) or not stdout.isatty():  # type: ignore[attr-defined]
        raise RuntimeError("interactive terminal not available")

    line_count = len(normalized) + 1

    default_index = 0
    if default is not None:
        for i, ch in enumerate(normalized):
            if ch.get("value") == default:
                default_index = i
                break

    index = default_index
    if normalized[index].get("disabled"):
        for i, entry in enumerate(normalized):
            if not entry.get("disabled"):
                index = i
                break
        else:
            raise RuntimeError("no selectable choices")
    selected_index = None
    cancelled = False
    rendered_once = False

    def _move_cursor_up(lines: int) -> None:
        if lines > 0:
            stdout.write(f"\x1b[{lines}A")

    def _clear_line() -> None:
        stdout.write("\r\x1b[2K")

    def _render() -> None:
        nonlocal rendered_once
        if rendered_once:
            _move_cursor_up(line_count)
        else:
            rendered_once = True
        _clear_line()
        stdout.write(f"? {message}\n")
        for pos, item in enumerate(normalized):
            marker = "▸" if pos == index else " "
            label = item.get("name", "")
            line = f"  {marker} {label}"
            _clear_line()
            stdout.write(f"{line}\n")
        stdout.flush()

    def _wait_for_input(timeout: float | None) -> bool:
        if timeout is not None and timeout < 0:
            timeout = 0
        try:
            rlist, _, _ = _select.select([fd], [], [], timeout)
        except Exception:
            return False
        return bool(rlist)

    def _read_char(timeout: float | None = None) -> str:
        if timeout is not None and not _wait_for_input(timeout):
            return ""
        try:
            data = os.read(fd, 1)
        except InterruptedError:
            return ""
        except Exception:
            return ""
        if not data:
            return ""
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    def _read_key() -> str | None:
        ch = _read_char()
        if not ch:
            return None
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch in {"\x04", "\x1a"}:
            return "escape"
        if ch in {"\r", "\n"}:
            return "enter"
        if ch == "\t":
            return "down"
        if ch in {"j", "J", "\x0e"}:  # j / Ctrl+N
            return "down"
        if ch in {"k", "K", "\x10"}:  # k / Ctrl+P
            return "up"
        if ch == "\x1b":
            ch2 = _read_char(0.25)
            if not ch2:
                return "escape"
            if ch2 == "[":
                seq = ""
                while True:
                    ch3 = _read_char(0.04)
                    if not ch3:
                        break
                    seq += ch3
                    if ch3 in {"A", "B", "C", "D", "F", "H", "~"}:
                        break
                if not seq:
                    return None
                final = seq[-1]
                if final == "A":
                    return "up"
                if final == "B":
                    return "down"
                if final == "Z":
                    return "up"
                if final in {"H"}:
                    return "home"
                if final in {"F"}:
                    return "end"
                if final == "~":
                    if seq.startswith("1") or seq.startswith("7"):
                        return "home"
                    if seq.startswith("4") or seq.startswith("8"):
                        return "end"
                    return None
                return None
            if ch2 == "O":
                ch3 = _read_char(0.04)
                if ch3 == "A":
                    return "up"
                if ch3 == "B":
                    return "down"
                if ch3 == "H":
                    return "home"
                if ch3 == "F":
                    return "end"
                return None
            return "escape"
        return None

    def _advance(delta: int) -> None:
        nonlocal index
        total = len(normalized)
        original = index
        for _ in range(total):
            index = (index + delta) % total
            if not normalized[index].get("disabled"):
                return
        index = original

    try:
        old_attrs = _termios.tcgetattr(fd)
    except Exception as exc:  # pragma: no cover - platform dependent
        raise RuntimeError("failed to read terminal settings") from exc

    try:
        _tty.setraw(fd)
        stdout.write("\x1b[?25l")  # hide cursor
        stdout.flush()
        _render()
        while True:
            key = _read_key()
            if key is None:
                continue
            if key == "enter":
                if normalized[index].get("disabled"):
                    continue
                selected_index = index
                break
            if key == "escape":
                cancelled = True
                break
            if key == "up":
                _advance(-1)
                _render()
                continue
            if key == "down":
                _advance(1)
                _render()
                continue
            if key == "home":
                index = 0
                _render()
                continue
            if key == "end":
                index = len(normalized) - 1
                _render()
                continue
    finally:
        try:
            _termios.tcsetattr(fd, _termios.TCSADRAIN, old_attrs)
        except Exception:
            pass
        stdout.write("\x1b[?25h")  # show cursor
        stdout.flush()

    _clear_last_prompt_lines(line_count)
    if selected_index is not None:
        choice = normalized[selected_index]
        label = choice.get("name", "")
        stdout.write(f"? {message} {label}\n")
        stdout.flush()
        return choice.get("value")
    if cancelled:
        return None
    raise RuntimeError("menu aborted")


def _clear_last_prompt_lines(n: int = 1) -> None:
    try:
        import sys as _sys
        for _ in range(max(1, n)):
            _sys.stdout.write("\x1b[1A\x1b[2K")  # up 1, clear line
        _sys.stdout.flush()
    except Exception:
        pass


def _stable_select(message: str, choices, default=None, *, min_duration: float = 0.15):
    """Call _select_menu and guard against immediate auto-close (e.g., stray Enter).

    If the menu returns too quickly (likely consumed the previous Enter), re-open once.
    """
    import time as _time
    _flush_stdin()

    t0 = _time.monotonic()
    # Small pause to avoid propagating the Enter that closed previous prompt
    _time.sleep(0.03)
    try:
        res = _select_menu(message, choices, default)
    except BaseException:
        return None
    dt = _time.monotonic() - t0
    # If it returned almost instantly, it's likely the previous Enter leaked.
    # Treat both None and default quick-accept as spurious and retry once.
    if (dt < min_duration) and (res is None or (default is not None and res == default)):
        # Likely instant abort; clean artifacts and retry once
        _time.sleep(0.09)
        try:
            res = _select_menu(message, choices, default)
        except BaseException:
            return None
    return res


def _flush_stdin() -> None:
    try:
        import sys as _sys
        import termios as _termios
        import os as _os
        try:
            fd = _sys.stdin.fileno()
            if _os.isatty(fd):
                _termios.tcflush(fd, _termios.TCIFLUSH)
        except Exception:
            pass
    except Exception:
        pass


def _ptk_select_menu(message: str, choices, default=None):
    # Disabled per request for minimal, inline menus only
    return None


def _numeric_menu(message: str, choices: list[dict], default_index: int = 0):
    """Very small fallback: show numbered list and read selection from input()."""
    try:
        print(message)
        for i, ch in enumerate(choices, start=1):
            name = ch.get("name", str(ch))
            print(f"  {i}. {name}")
        raw = input("Select number (empty to cancel): ").strip()
        if not raw:
            return None
        idx = int(raw)
        if not (1 <= idx <= len(choices)):
            return None
        return choices[idx - 1].get("value")
    except Exception:
        return None


@app.callback(invoke_without_command=True)
def _default(ctx: typer.Context):
    """Start interactive session when running plain `diffmind`."""
    if ctx.invoked_subcommand is not None:
        return
    try:
        repo = _repo()
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)
    cfg = DiffMindConfig.load()
    with console.status("Preparing suggestion…", spinner="dots"):
        try:
            msg = generate_commit_message(repo, cfg)
        except ProviderConfigError as e:
            banner("[b]Provider not configured[/b]")
            console.print(f"[red]{e}[/red]")
            openai_help_panel(
                missing_pkg="package is not installed" in str(e).lower(),
                missing_key="api key" in str(e).lower(),
            )
            raise typer.Exit(code=2)
        except NoChangesError:
            banner("[b]No staged changes[/b]")
            tip("Stage files first (git add) or use 'diffmind commit -a'.")
            # Continue into interactive mode even without staged changes
            msg = Message(subject="", body=None)
    _interactive_suggest(repo, cfg, msg)


def _repo() -> Repo:
    return Repo(search_parent_directories=True)


def _install_openai() -> bool:
    console.print("Installing 'openai' package…", style="yellow")
    res = subprocess.run([sys.executable, "-m", "pip", "install", "openai"])  # same env
    if res.returncode == 0:
        console.print("✔ Installed 'openai'", style="green")
        return True
    console.print("✖ Failed to install 'openai'", style="red")
    return False


def _save_user_config(cfg: DiffMindConfig, repo: Optional[Repo]) -> None:
    rr = None
    try:
        rr = Path(repo.git_dir).parent if repo else None  # type: ignore[attr-defined]
    except Exception:
        rr = None
    path = save_config(cfg, preferred_write_path(rr, scope="user"))
    console.print(f"✔ Saved configuration to {path}")


def _choose_provider_mode(cfg: DiffMindConfig, repo: Optional[Repo]) -> None:
    """Switch provider: auto/simple/openai (installs package when possible)."""
    try:
        sel = _select_menu(
            "Choose provider mode",
            [
                {"name": "auto — detect automatically", "value": "auto"},
                {"name": "simple — local heuristics (no AI)", "value": "simple"},
                {"name": "openai — OpenAI (GPT via API)", "value": "openai"},
            ],
            default=(cfg.provider or "auto"),
        )
    except Exception:
        sel = typer.prompt("Provider [auto/simple/openai]", default=(cfg.provider or "auto"))
    if not sel:
        tip("Mode unchanged.")
        return
    sel = str(sel).strip().lower()
    if sel == "openai":
        missing_key = not (cfg.openai_api_key or os.getenv("OPENAI_API_KEY"))
        missing_pkg = False
        try:
            import openai  # type: ignore
            _ = openai
        except Exception:
            missing_pkg = True
        if missing_pkg and not missing_key:
            if cfg.auto_install_openai or Confirm.ask("Install 'openai' package now?", default=True):
                if not _install_openai():
                    tip("Keeping simple mode (install failed).")
                    return
            else:
                tip("Keeping simple mode (install canceled).")
                return
        if missing_key:
            openai_help_panel(missing_pkg=missing_pkg, missing_key=True)
            tip("Provider unchanged: missing API key.")
            return
    cfg.provider = sel
    _save_user_config(cfg, repo)


def _toggle_emojis(cfg: DiffMindConfig, repo: Optional[Repo]) -> None:
    cfg.emojis = not bool(cfg.emojis)
    _save_user_config(cfg, repo)
    console.print(f"Emojis: {'on' if cfg.emojis else 'off'}")


def _choose_emojis(cfg: DiffMindConfig, repo: Optional[Repo]) -> None:
    """Choose emojis on/off via a minimal selector (fallback to typed)."""
    default = "on" if cfg.emojis else "off"
    try:
        sel = _select_menu(
            "Emojis",
            [{"name": "on", "value": "on"}, {"name": "off", "value": "off"}],
            default=default,
        )
    except Exception:
        try:
            sel = typer.prompt("Emojis [on/off]", default=default)
        except Exception:
            sel = default
    val = str(sel or default).strip().lower()
    cfg.emojis = (val == "on")
    _save_user_config(cfg, repo)
    console.print(f"Emojis: {'on' if cfg.emojis else 'off'}")


def _apply_models_token(token: str, cfg: DiffMindConfig, repo: Optional[Repo]) -> bool:
    """Apply model change from a token like '2' or 'gpt-5-mini' or 'mini'.

    Returns True if a model was applied and saved.
    """
    t = (token or "").strip().lower()
    if not t:
        return False
    choices = _available_openai_models()
    # numeric index (1-based)
    if t.isdigit():
        idx = int(t)
        if 1 <= idx <= len(choices):
            cfg.openai_model = choices[idx - 1]["value"]
            cfg.provider = "openai"
            _save_user_config(cfg, repo)
            tip(f"Using OpenAI model: {cfg.openai_model}")
            return True
        return False
    # match by exact id or substring
    for ch in choices:
        val = ch.get("value", "").lower()
        name = ch.get("name", "").lower()
        if t == val or t == name or t in val or t in name:
            cfg.openai_model = ch["value"]
            cfg.provider = "openai"
            _save_user_config(cfg, repo)
            tip(f"Using OpenAI model: {cfg.openai_model}")
            return True
    return False


def _apply_config_from_text(text: str, cfg: DiffMindConfig, repo: Optional[Repo]) -> bool:
    """Parse natural-language config commands. Returns True if applied and saved.

    Supports:
    - set model (e.g. 'use gpt-5-mini', 'модель gpt-5-mini')
    - switch mode/provider ('use openai', 'режим simple')
    - toggle emojis ('включи эмодзи', 'disable emojis', 'без эмодзи')
    """
    t = (text or "").strip().lower()
    if not t:
        return False
    import re as _re

    # emojis toggle
    if any(k in t for k in ["emoji", "эмодзи", "эмоджи"]):
        if any(k in t for k in ["off", "disable", "выключ", "без "]):
            cfg.emojis = False
        elif any(k in t for k in ["on", "enable", "включ", "с эмодзи"]):
            cfg.emojis = True
        else:
            cfg.emojis = not bool(cfg.emojis)
        _save_user_config(cfg, repo)
        return True

    # provider mode
    if "openai" in t or "simple" in t or "auto" in t or "режим" in t or "mode" in t:
        if "openai" in t:
            # try to ensure availability
            missing_key = not (cfg.openai_api_key or os.getenv("OPENAI_API_KEY"))
            missing_pkg = False
            try:
                import openai  # type: ignore
                _ = openai
            except Exception:
                missing_pkg = True
            if missing_pkg and not missing_key:
                if cfg.auto_install_openai or Confirm.ask("Install 'openai' package now?", default=True):
                    if not _install_openai():
                        return False
                else:
                    return False
            if missing_key:
                openai_help_panel(missing_pkg=missing_pkg, missing_key=True)
                return False
            cfg.provider = "openai"
            _save_user_config(cfg, repo)
            return True
        if "simple" in t:
            cfg.provider = "simple"
            _save_user_config(cfg, repo)
            return True
        if "auto" in t:
            cfg.provider = "auto"
            _save_user_config(cfg, repo)
            return True

    # model id (gpt-.....)
    m = _re.search(r"\bgpt-[\w-]+\b", t)
    if m:
        cfg.openai_model = m.group(0)
        cfg.provider = "openai"
        # Ensure package/key
        missing_key = not (cfg.openai_api_key or os.getenv("OPENAI_API_KEY"))
        missing_pkg = False
        try:
            import openai  # type: ignore
            _ = openai
        except Exception:
            missing_pkg = True
        if missing_pkg and not missing_key:
            if cfg.auto_install_openai or Confirm.ask("Install 'openai' package now?", default=True):
                if not _install_openai():
                    return False
            else:
                return False
        if missing_key:
            openai_help_panel(missing_pkg=missing_pkg, missing_key=True)
            return False
        _save_user_config(cfg, repo)
        return True

    return False


@app.command()
def version():
    """Show DiffMind version."""
    console.print(f"DiffMind v{__version__}")


@app.command()
def suggest(
    provider: Optional[str] = typer.Option(None, help="Provider: auto|simple|openai"),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        is_flag=True,
        flag_value=True,
        help="Enable interactive mode with arrows & input",
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        is_flag=True,
        flag_value=True,
        help="Ignore cache and generate anew",
    ),
):
    """Suggest a commit message from staged changes."""
    repo = _repo()
    cfg = DiffMindConfig.load({"provider": provider} if provider else None)
    try:
        msg = generate_commit_message(repo, cfg, force_regen=bool(refresh))
    except ProviderConfigError as e:
        text = str(e).lower()
        missing_pkg = "package is not installed" in text
        missing_key = "api key" in text
        # If only the package is missing and we have/will have a key, offer to install now
        if missing_pkg and not missing_key:
            if cfg.auto_install_openai or Confirm.ask("Install 'openai' package now?", default=True):
                if _install_openai():
                    cfg = DiffMindConfig.load()
                    msg = generate_commit_message(repo, cfg, force_regen=True)
                else:
                    banner("[b]Provider not configured[/b]")
                    console.print(f"[red]{e}[/red]")
                    openai_help_panel(missing_pkg=True, missing_key=False)
                    raise typer.Exit(code=2)
            else:
                banner("[b]Provider not configured[/b]")
                console.print(f"[red]{e}[/red]")
                openai_help_panel(missing_pkg=True, missing_key=False)
                tip("Run 'diffmind init' for quick OpenAI and hooks setup.")
                raise typer.Exit(code=2)
        else:
            banner("[b]Provider not configured[/b]")
            console.print(f"[red]{e}[/red]")
            openai_help_panel(missing_pkg=missing_pkg, missing_key=missing_key)
            tip("Run 'diffmind init' for quick OpenAI and hooks setup.")
            raise typer.Exit(code=2)
    except NoChangesError:
        banner("[b]No staged changes[/b]")
        tip("Stage files first (git add) or pass -a in 'diffmind commit'.")
        raise typer.Exit(code=0)
    except NoChangesError:
        banner("[b]No staged changes[/b]")
        tip("Nothing to commit. Use -a to stage all or 'git add'.")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)

    # Choose interactive by default when TTY unless explicitly disabled
    if not interactive:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()

    if interactive:
        _interactive_suggest(repo, cfg, msg)
        return

    banner("[b]Commit Message Suggestion[/b]")
    print_message(msg.subject, msg.body)
    if (cfg.provider or "simple").lower() == "simple":
        # Friendly notice about AI availability
        simple_mode_warning()
        missing_key = not (cfg.openai_api_key or os.getenv("OPENAI_API_KEY"))
        missing_pkg = False
        try:
            import openai  # type: ignore
            _ = openai
        except Exception:
            missing_pkg = True
        if not missing_key and missing_pkg:
            if cfg.auto_install_openai or Confirm.ask("Install 'openai' package now?", default=True):
                if _install_openai():
                    cfg = DiffMindConfig.load()
                    msg = generate_commit_message(repo, cfg, force_regen=True)
                    print_message(msg.subject, msg.body)
                    tip("Switched to OpenAI provider.")
                    return
        if missing_key or missing_pkg:
            openai_help_panel(missing_pkg=missing_pkg, missing_key=missing_key)
            tip("Quick setup: run 'diffmind init'.")
    tip("Use `diffmind commit` to commit with this message.")


@app.command()
def commit(
    all: bool = typer.Option(False, "-a", "--all", help="Stage all changes before committing"),
    no_verify: bool = typer.Option(
        False,
        "--no-verify",
        is_flag=True,
        flag_value=True,
        help="Pass --no-verify to git commit",
    ),
    amend: bool = typer.Option(
        False,
        "--amend",
        is_flag=True,
        flag_value=True,
        help="Amend the previous commit",
    ),
    provider: Optional[str] = typer.Option(None, help="Provider: auto|simple|openai"),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        is_flag=True,
        flag_value=True,
        help="Only print the message, do not commit",
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        is_flag=True,
        flag_value=True,
        help="Ignore cache and generate anew",
    ),
):
    """Generate and commit with the suggested message."""
    repo = _repo()
    cfg = DiffMindConfig.load({"provider": provider} if provider else None)
    if all:
        subprocess.run(["git", "add", "-A"], check=False)

    try:
        msg = generate_commit_message(repo, cfg, force_regen=bool(refresh))
    except ProviderConfigError as e:
        text = str(e).lower()
        missing_pkg = "package is not installed" in text
        missing_key = "api key" in text
        if missing_pkg and not missing_key:
            if cfg.auto_install_openai or Confirm.ask("Install 'openai' package now?", default=True):
                if _install_openai():
                    cfg = DiffMindConfig.load()
                    msg = generate_commit_message(repo, cfg, force_regen=True)
                else:
                    banner("[b]Provider not configured[/b]")
                    console.print(f"[red]{e}[/red]")
                    openai_help_panel(missing_pkg=True, missing_key=False)
                    raise typer.Exit(code=2)
            else:
                banner("[b]Provider not configured[/b]")
                console.print(f"[red]{e}[/red]")
                openai_help_panel(missing_pkg=True, missing_key=False)
                tip("Run 'diffmind init' for quick OpenAI setup.")
                raise typer.Exit(code=2)
        else:
            banner("[b]Provider not configured[/b]")
            console.print(f"[red]{e}[/red]")
            openai_help_panel(missing_pkg=missing_pkg, missing_key=missing_key)
            tip("Run 'diffmind init' for quick OpenAI setup.")
            raise typer.Exit(code=2)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)
    banner("[b]Generated Commit[/b]")
    print_message(msg.subject, msg.body)
    if (cfg.provider or "simple").lower() == "simple":
        # Friendly notice about AI availability
        simple_mode_warning()
        missing_key = not (cfg.openai_api_key or os.getenv("OPENAI_API_KEY"))
        missing_pkg = False
        try:
            import openai  # type: ignore
            _ = openai
        except Exception:
            missing_pkg = True
        if not missing_key and missing_pkg:
            if cfg.auto_install_openai or Confirm.ask("Install 'openai' package now?", default=True):
                if _install_openai():
                    cfg = DiffMindConfig.load()
                    msg = generate_commit_message(repo, cfg, force_regen=True)
                    print_message(msg.subject, msg.body)
                    tip("Switched to OpenAI provider.")
        if missing_key or missing_pkg:
            openai_help_panel(missing_pkg=missing_pkg, missing_key=missing_key)
            tip("Quick setup: 'diffmind init'.")

    if dry_run:
        return

    if not Confirm.ask("Proceed with commit?", default=True):
        raise typer.Abort()

    cmd = ["git", "commit", "-m", msg.subject]
    if msg.body:
        cmd.extend(["-m", msg.body])
    if no_verify:
        cmd.append("--no-verify")
    if amend:
        cmd.append("--amend")
    res = subprocess.run(cmd)
    raise typer.Exit(code=res.returncode)


@hook_app.command("install")
def hook_install():
    """Install the prepare-commit-msg hook to this repo."""
    repo = _repo()
    path = install_hook(repo)
    console.print(f"Installed hook: {HOOK_NAME} → {path}")


@hook_app.command("uninstall")
def hook_uninstall():
    """Remove the prepare-commit-msg hook from this repo."""
    repo = _repo()
    uninstall_hook(repo)
    console.print(f"Uninstalled hook: {HOOK_NAME}")


@hook_app.command("run")
def hook_run(
    message_file: str = typer.Argument(..., help="Path to commit message file"),
    commit_source: Optional[str] = typer.Argument(None),
    sha1: Optional[str] = typer.Argument(None),
):
    """Internal: executed by the git hook."""
    repo = _repo()
    code = run_prepare_commit_msg(repo, message_file, commit_source, sha1)
    raise typer.Exit(code)


@app.command()
def doctor():
    """Run basic checks and show status."""
    banner("[b]DiffMind Doctor[/b]")
    try:
        repo = _repo()
        console.print("✔ Found Git repository", style="green")
        hook_path = Path(repo.git_dir) / "hooks" / HOOK_NAME
        if hook_path.is_file():
            console.print(f"✔ Hook installed at {hook_path}", style="green")
        else:
            console.print("• Hook not installed (run: diffmind hook install)", style="yellow")
    except Exception as e:
        console.print(f"✖ {e}", style="red")

    cfg = DiffMindConfig.load()
    console.print(f"Provider: {cfg.provider}")
    # OpenAI checks
    if (cfg.provider or "").lower() == "openai":
        missing_pkg = False
        try:
            import openai  # type: ignore
        except Exception:
            missing_pkg = True
        api_key = cfg.openai_api_key or os.getenv("OPENAI_API_KEY")
        if missing_pkg or not api_key:
            openai_help_panel(missing_pkg=missing_pkg, missing_key=not api_key)
    tip("Configure via .diffmind.toml or ~/.config/diffmind/config.toml or run 'diffmind config wizard'.")


@app.command()
def init(
    ai: bool = typer.Option(
        False, "--ai", is_flag=True, flag_value=True, help="Set up OpenAI now (auto-detect if not provided)"
    ),
    hook: bool = typer.Option(
        True,
        "--hook/--no-hook",
        is_flag=True,
        flag_value=True,
        help="Install git hook after setup",
    ),
    scope: str = typer.Option("user", help="Where to save config: user|repo"),
    openai_api_key: Optional[str] = typer.Option(None, help="Provide OpenAI API key (sk-...)")
):
    """One-shot setup: configure provider and install git hook."""
    banner("[b]DiffMind Init[/b]")
    cfg = DiffMindConfig.load()

    # Decide AI setup
    has_pkg = True
    try:
        import openai  # type: ignore
        _ = openai
    except Exception:
        has_pkg = False
    key = openai_api_key or cfg.openai_api_key or os.getenv("OPENAI_API_KEY")

    if not ai:
        # Auto-detect when --ai not explicitly provided
        ai = bool(key)

    if ai:
        if not has_pkg:
            if Confirm.ask("Install 'openai' package now?", default=True):
                if not _install_openai():
                    console.print("[red]OpenAI not installed. Skipping AI setup.[/red]")
                    ai = False
                    has_pkg = False
                else:
                    has_pkg = True
        if not key:
            if Confirm.ask("No OPENAI_API_KEY found. Enter it now?", default=True):
                key = typer.prompt("OpenAI API key (sk-...)", hide_input=True)
        if key:
            cfg.openai_api_key = key
        if has_pkg and key:
            cfg.provider = "openai"
        else:
            cfg.provider = "simple"
    else:
        cfg.provider = "simple"

    # Save config
    repo = None
    try:
        repo = _repo()
    except Exception:
        pass
    rr = Path(repo.git_dir).parent if repo else None
    path = save_config(cfg, preferred_write_path(rr, scope=scope))
    console.print(f"✔ Saved configuration to {path}")

    # Hook installation
    if hook and repo:
        p = install_hook(repo)
        console.print(f"✔ Installed hook: {p}")
    elif hook and not repo:
        console.print("ℹ Run 'diffmind hook install' inside a Git repo to add the hook.")

    console.print(f"Provider: {cfg.provider}")
    if cfg.provider == "openai":
        console.print("Using OpenAI provider ✅", style="green")
    else:
        console.print("Using simple provider (local heuristics) ✅", style="green")


@app.command()
def session(
    provider: Optional[str] = typer.Option(None, help="Provider: auto|simple|openai"),
):
    """Interactive session (arrows + free-text instructions) to refine and commit."""
    repo = _repo()
    cfg = DiffMindConfig.load({"provider": provider} if provider else None)
    try:
        msg = generate_commit_message(repo, cfg)
    except ProviderConfigError as e:
        banner("[b]Provider not configured[/b]")
        console.print(f"[red]{e}[/red]")
        openai_help_panel(missing_pkg="package is not installed" in str(e).lower(), missing_key="api key" in str(e).lower())
        raise typer.Exit(code=2)
    except NoChangesError:
        banner("[b]No staged changes[/b]")
        tip("Stage files first (git add) or use 'diffmind commit -a'.")
        raise typer.Exit(code=0)
    _interactive_suggest(repo, cfg, msg)


def _edit_in_editor(initial: str) -> str:
    import tempfile
    import os
    import shlex
    import subprocess

    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".commitmsg", encoding="utf-8") as tf:
        tf.write(initial)
        tf.flush()
        path = tf.name
    editor = os.environ.get("EDITOR", "vi")
    try:
        subprocess.run([editor, path])
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


def _available_openai_models() -> list[dict[str, str]]:
    """Curated list of OpenAI chat models to pick from."""
    return [
        {"name": "gpt-5-nano — default (fast, low-cost)", "value": "gpt-5-nano"},
        {"name": "gpt-5-mini — balanced", "value": "gpt-5-mini"},
        {"name": "gpt-5 — highest quality", "value": "gpt-5"},
    ]


def _choose_openai_model(cfg: DiffMindConfig, repo: Optional[Repo]) -> None:
    """Minimal model picker: arrow keys, Enter to save, Esc to cancel."""
    default_model = (cfg.openai_model or "gpt-5-nano").strip()
    # Prefer InquirerPy minimal select (no extra dialogs)
    try:
        choices = _available_openai_models()
        preselect = default_model if any(c["value"] == default_model for c in choices) else choices[0]["value"]
        sel = _select_menu(
            "Choose OpenAI model",
            choices,
            default=preselect,
        )
        if not sel:
            tip("Model unchanged.")
            return
        cfg.openai_model = sel
    except KeyboardInterrupt:
        tip("Model unchanged.")
        return
    except Exception:
        # Fallback: numeric menu (arrow-key-free)
        try:
            choices = _available_openai_models()
            try:
                default_index = next(i for i, c in enumerate(choices) if c.get("value") == default_model)
            except StopIteration:
                default_index = 0
            sel = _numeric_menu("Choose OpenAI model", choices, default_index=default_index)
            if sel:
                cfg.openai_model = sel
            else:
                tip("Model unchanged.")
                return
        except Exception:
            # Last resort: typed
            try:
                model = typer.prompt("OpenAI model id", default=default_model).strip()
            except Exception:
                model = ""
            if not model:
                tip("Model unchanged.")
                return
            cfg.openai_model = model

    # Force OpenAI provider and save to user config immediately
    cfg.provider = "openai"
    rr = None
    try:
        rr = Path(repo.git_dir).parent if repo else None  # type: ignore[attr-defined]
    except Exception:
        rr = None
    path = save_config(cfg, preferred_write_path(rr, scope="user"))
    console.print(f"✔ Saved configuration to {path}")
    tip(f"Using OpenAI model: {cfg.openai_model}")
    return


class _SlashCompleter:  # lightweight prompt_toolkit completer
    def __init__(self, commands: list[tuple[str, str]]):
        # commands: list of (value, description)
        self._commands = commands

    def __pt_comp__(self):  # pragma: no cover - prompt_toolkit integration
        return self

    def get_completions(self, document, complete_event):  # type: ignore[override]
        try:
            from prompt_toolkit.completion import Completion  # type: ignore
        except Exception:
            return  # no completions available
        text = document.text_before_cursor or ""
        if not text.startswith("/"):
            return
        prefix = text
        for value, desc in self._commands:
            if value.startswith(prefix):
                yield Completion(
                    value,
                    start_position=-len(prefix),
                    display=value,
                    display_meta=desc,
                )

    async def get_completions_async(self, document, complete_event):  # pragma: no cover
        # Wrap sync completions for prompt_toolkit's async path
        for c in self.get_completions(document, complete_event):
            yield c


def _prompt_with_slash(
    message: str,
    commands: list[tuple[str, str]],
    *,
    placeholder: Optional[str] = None,
    placeholder_style: str = "fg:#666666 italic",
) -> Optional[str]:
    """Prompt for a single line. If user types '/', show commands dropdown immediately.

    - `message` renders as the left-hand prompt label (e.g., "▌ ").
    - `placeholder` renders inside the input when empty and disappears on typing.
    """
    try:
        from prompt_toolkit import PromptSession  # type: ignore
        from prompt_toolkit.key_binding import KeyBindings  # type: ignore
        from prompt_toolkit.styles import Style  # type: ignore
    except Exception:
        return None
    try:
        completer = _SlashCompleter(commands)

        # First pass: no bottom reserve to keep layout compact.
        # If user presses '/' on empty line, exit and re-open with extra space and '/' prefilled.
        kb = KeyBindings()

        @kb.add("/")
        def _(event):  # type: ignore
            buf = event.current_buffer
            doc = buf.document
            if (doc.text or "") == "":
                event.app.exit(result=SENT_SLASH)
            else:
                buf.insert_text("/")

        # Enter on empty input opens menu instead of returning empty string
        @kb.add("enter")
        def _enter(event):  # type: ignore
            buf = event.current_buffer
            if not (buf.text or "").strip():
                event.app.exit(result=SENT_MENU)
            else:
                event.app.exit(result=buf.text)

        # Swallow ESC and Ctrl+Q at the input prompt to avoid printing '^[...'
        @kb.add("escape")
        def _esc_ignore(event):  # type: ignore
            # Do nothing, just prevent literal ESC from appearing in the buffer
            return

        @kb.add("c-q")
        def _ctrl_q_ignore(event):  # type: ignore
            return

        # Swallow arrow keys so they don't print escape sequences in input
        @kb.add("up")
        def _up_ignore(event):  # type: ignore
            return

        @kb.add("down")
        def _down_ignore(event):  # type: ignore
            return

        session = PromptSession()
        style = Style.from_dict({
            # Make placeholder look like background hint
            "placeholder": placeholder_style,
        })
        text = session.prompt(
            message,
            completer=completer,
            complete_while_typing=True,
            reserve_space_for_menu=0,
            key_bindings=kb,
            placeholder=placeholder,
            style=style,
        )
        if text == SENT_SLASH:
            # Re-open with extra space and '/' prefilled so the dropdown is fully visible
            reserve = min(10, max(5, len(commands) + 2))
            # Clean up the previous prompt line to avoid duplicate prompt text
            try:
                import sys as _sys
                _sys.stdout.write("\x1b[1A\x1b[2K")  # move cursor up 1, clear line
                _sys.stdout.flush()
            except Exception:
                pass
            session2 = PromptSession()
            return session2.prompt(
                message,
                default="/",
                completer=completer,
                complete_while_typing=True,
                reserve_space_for_menu=reserve,
                pre_run=lambda: session2.app.current_buffer.start_completion(),
                placeholder=placeholder,
                style=style,
            )
        return text
    except Exception:
        return None


def _interpret_instruction(text: str) -> Optional[str]:
    t = text.strip().lower()
    commit_words = {
        "commit",
        "commit this",
        "ok",
        "looks good",
        "lgtm",
        "ship it",
        "закоммить",
        "коммит",
        "коммитим",
        "давай коммит",
        "всё ок",
        "все ок",
        "готово",
        "сделай коммит",
        "закомить",
        "закоммить это",
        "commit it",
    }
    regen_words = {"regen", "regenerate", "again", "ещё", "еще", "перегенерируй", "перегенери", "снова", "другой", "ещё раз"}
    diff_words = {"diff", "show diff", "покажи дифф", "дифф", "show changes", "покажи изменения"}
    if t in commit_words:
        return "commit"
    if t in regen_words:
        return "regen"
    if t in diff_words:
        return "diff"
    return None


def _looks_like_question(text: str) -> bool:
    t = (text or "").strip().lower()
    if "?" in t:
        return True
    prefixes = (
        # Russian interrogatives
        "как ", "что ", "что такое ", "что это ", "что делает ",
        "почему ", "зачем ", "когда ", "где ", "сколько ", "кто ",
        "какой ", "какая ", "какие ",
        "о чем ", "о чём ", "в чем ", "в чём ",
        # Russian imperative info-requests
        "расскажи", "объясни", "подскажи", "покажи", "научи", "перечисли", "опиши",
        # English interrogatives/modal questions
        "how ", "what ", "why ", "when ", "where ", "which ", "who ",
        "can ", "could ", "should ", "would ", "will ", "is ", "are ", "am ", "do ", "does ", "did ",
        # English imperative info-requests
        "tell me", "show me", "explain", "list ", "describe ", "help me", "please ",
    )
    return t.startswith(prefixes)


def _do_commit(msg, ask_options: bool = False):
    args = ["git", "commit", "-m", msg.subject]
    if msg.body:
        args += ["-m", msg.body]
    if ask_options:
        from InquirerPy import inquirer as _inq

        if _inq.confirm(message="--amend?", default=False).execute():
            args.append("--amend")
        if _inq.confirm(message="--no-verify?", default=False).execute():
            args.append("--no-verify")
    res = subprocess.run(args)
    raise typer.Exit(code=res.returncode)


def _interactive_suggest(repo: Repo, cfg: DiffMindConfig, msg):
    try:
        from InquirerPy import inquirer
    except Exception:
        # Fallback to non-interactive
        print_message(msg.subject, msg.body)
        tip("Interactive prompts unavailable. Install InquirerPy or run without --interactive.")
        return

    from .utils.git import get_staged_diff_text
    # Prepare agent for inline Q&A with a persisted session
    inline_session = ChatSession.create(repo)
    llm, tools, _misconfigured = _build_llm_and_tools(cfg)
    inline_agent = AgentOrchestrator(llm, tools, max_steps=4)

    def _exit_interactive_session() -> None:
        if inline_session.messages:
            try:
                inline_session.save()
            except Exception:
                pass
        console.print("Leaving interactive session without committing.", style="dim")

    def _handle_action(action: str, *, ask_options: bool = False) -> Optional[bool]:
        """Handle core actions. Returns True to continue loop, False to exit, None if unhandled."""

        nonlocal msg, cfg
        act = (action or "").strip().lower()
        if not act:
            return None
        if act == "commit":
            if ask_options:
                _do_commit(msg, ask_options=True)
            else:
                with console.status("Committing…", spinner="dots"):
                    _do_commit(msg)
            return False
        if act == "regen":
            src = (cfg.provider or "simple")
            with console.status(f"Regenerating suggestion ({src})…", spinner="dots"):
                msg = generate_commit_message(repo, cfg, force_regen=True)
            print_message(msg.subject, msg.body)
            return True
        if act == "diff":
            diff = get_staged_diff_text(repo, unified=3) or "(no staged diff)"
            console.print(Panel(diff, title="Staged Diff", border_style="magenta", box=box.ROUNDED))
            return True
        if act == "edit":
            subj = inquirer.text(message="Subject", default=msg.subject).execute()
            body = inquirer.text(message="Body (empty — keep as is)", default=msg.body or "").execute()
            msg.subject, msg.body = subj, (body or None)
            try:
                _cache_save_current(repo, msg, {"source": "edit"})
            except Exception:
                pass
            print_message(msg.subject, msg.body)
            return True
        if act == "editor":
            content = msg.subject + ("\n\n" + msg.body if msg.body else "")
            new = _edit_in_editor(content)
            parts = new.splitlines()
            new_subject = (parts[0].strip() if parts else msg.subject) or msg.subject
            new_body = "\n".join(parts[1:]).strip() or None
            msg.subject, msg.body = new_subject, new_body
            try:
                _cache_save_current(repo, msg, {"source": "editor"})
            except Exception:
                pass
            print_message(msg.subject, msg.body)
            return True
        if act == "add":
            line = inquirer.text(message="Bullet line", default="").execute()
            if line.strip():
                if msg.body:
                    msg.body = (msg.body + "\n- " + line.strip()).rstrip()
                else:
                    msg.body = "- " + line.strip()
            try:
                _cache_save_current(repo, msg, {"source": "add"})
            except Exception:
                pass
            print_message(msg.subject, msg.body)
            return True
        if act == "wizard":
            config_wizard()
            cfg = DiffMindConfig.load()
            status_panel(cfg)
            return True
        if act == "chat":
            _chat_loop(repo, cfg)
            return True
        if act == "model":
            _choose_openai_model(cfg, repo)
            status_panel(cfg)
            return True
        return None

    # Render initial suggestion once (banner + status)
    banner("[b]Commit Message Suggestion[/b]")
    status_panel(cfg)
    if (cfg.provider or "simple").lower() == "simple":
        simple_mode_warning()
    print_message(msg.subject, msg.body)

    while True:

        # Prompt with live slash palette when available (minimal palette)
        res = _prompt_with_slash(
            "\u258c ",
            [
                ("/models", "choose what model to use"),
                ("/mode", "switch provider: auto/simple/openai"),
                ("/emojis", "choose emojis on/off"),
                ("/help", "show this help"),
            ],
            placeholder="Enter instruction (or press Enter to open menu):",
            placeholder_style="fg:#333333 italic",
        )
        # If user pressed Enter on empty input in the prompt, open the action menu
        if res == SENT_MENU:
            choices = [
                {"name": "🔁 Regenerate", "value": "regen"},
                {"name": "✏️  Edit subject/body", "value": "edit"},
                {"name": "📝 Open in $EDITOR", "value": "editor"},
                {"name": "📄 Show staged diff", "value": "diff"},
            ]
            action = _stable_select("Choose an action", choices, default="regen")
            if action is None:
                continue
            # Map selection to actions below by setting a synthetic instruction
            instr = action or ""
        else:
            instr = res if res is not None else inquirer.text(message="\u258c ").execute()

        if instr and instr.strip():
            t = instr.strip()
            lowered = t.lower()
            normalized = lowered[1:] if lowered.startswith("/") else lowered
            if normalized in {"quit", "exit", "leave"}:
                _exit_interactive_session()
                return
            handled_direct = _handle_action(normalized)
            if handled_direct is True:
                continue
            if handled_direct is False:
                return
            # Slash commands
            if t in {"/help", "/?", "/"}:
                # Minimal command palette
                try:
                    sel = _stable_select(
                        "/ commands",
                        [
                            {"name": "/models     choose what model to use", "value": "models"},
                            {"name": "/mode       switch provider (auto/simple/openai)", "value": "mode"},
                            {"name": "/emojis     choose emojis on/off", "value": "emojis"},
                            {"name": "/help       show this help", "value": "help"},
                        ],
                        default="models",
                    )
                    if sel is None:
                        continue
                    if sel == "models":
                        _choose_openai_model(cfg, repo)
                        status_panel(cfg)
                    elif sel == "mode":
                        _choose_provider_mode(cfg, repo)
                        status_panel(cfg)
                    elif sel == "emojis":
                        _choose_emojis(cfg, repo)
                        status_panel(cfg)
                    elif sel == "help":
                        from rich.panel import Panel as _Panel

                        console.print(
                            _Panel(
                                "Available commands:\n"
                                "  /models — choose OpenAI model\n"
                                "  /mode   — switch provider (auto/simple/openai)\n"
                                "  /emojis — choose emojis on/off\n"
                                "  /help   — show this help",
                                title="Commands",
                                border_style="blue",
                                box=box.ROUNDED,
                            )
                        )
                    else:
                        # Closed or no selection
                        pass
                except Exception:
                    from rich.panel import Panel as _Panel

                    console.print(
                        _Panel(
                            "Available commands:\n"
                            "  /models — choose OpenAI model\n"
                            "  /mode   — switch provider (auto/simple/openai)\n"
                            "  /emojis — choose emojis on/off\n"
                            "  /help   — show this help",
                            title="Commands",
                            border_style="blue",
                            box=box.ROUNDED,
                        )
                    )
                continue
            if t == "/models" or t.startswith("/models"):
                parts = t.split(maxsplit=1)
                if len(parts) > 1 and _apply_models_token(parts[1], cfg, repo):
                    # Reprint status to reflect model change
                    status_panel(cfg)
                    continue
                _choose_openai_model(cfg, repo)
                # Reprint status to reflect model change
                status_panel(cfg)
                continue
            if t == "/mode" or t.startswith("/mode"):
                _choose_provider_mode(cfg, repo)
                status_panel(cfg)
                continue
            if t == "/emojis" or t.startswith("/emojis"):
                # Support '/emojis on' or '/emojis off'; otherwise show selector
                parts = t.split()
                if len(parts) >= 2 and parts[1].lower() in {"on", "off"}:
                    cfg.emojis = (parts[1].lower() == "on")
                    _save_user_config(cfg, repo)
                else:
                    _choose_emojis(cfg, repo)
                status_panel(cfg)
                continue
            # Natural command shortcuts
            action = _interpret_instruction(instr)
            if action:
                handled = _handle_action(action)
                if handled is True:
                    continue
                if handled is False:
                    return
            # Natural-language config changes
            if not action and _apply_config_from_text(instr, cfg, repo):
                status_panel(cfg)
                with console.status("Regenerating suggestion…", spinner="dots"):
                    try:
                        msg = generate_commit_message(repo, cfg, force_regen=True)
                    except Exception:
                        pass
                # Show updated message only
                print_message(msg.subject, msg.body)
                continue
            # Git Q&A: detect git-related questions and answer with OpenAI when available
            is_question = _looks_like_question(instr)
            if is_question:
                with console.status("Answering your question…", spinner="dots"):
                    try:
                        inline_session.add_user(instr)
                        ans = inline_agent.run_git_assistant(
                            instr, ToolContext(cfg=cfg, repo=repo), history=inline_session.messages[:-1]
                        )
                        inline_session.add_assistant(ans)
                        inline_session.save()
                    except Exception as e:
                        ans = f"error: {e}"
                console.print(Panel(_sanitize_agent_output(ans or ""), title="Agent", border_style="cyan", box=box.ROUNDED))
                continue
            diff = get_staged_diff_text(repo, unified=3)
            with console.status("Refining message…", spinner="dots"):
                refined = refine_with_openai(msg.subject, msg.body, diff, instr, cfg)
            if refined is None:
                with console.status("Refining locally…", spinner="dots"):
                    refined = heuristic_refine(
                        msg.subject, msg.body, instr, max_len=cfg.max_subject_length, diff_text=diff
                    )
            msg = refined
            try:
                _cache_save_current(repo, msg, {"source": "refine"})
            except Exception:
                pass
            # Show updated message only
            print_message(msg.subject, msg.body)
            continue

        choices = [
            {"name": "🔁 Regenerate", "value": "regen"},
            {"name": "✏️  Edit subject/body", "value": "edit"},
            {"name": "📝 Open in $EDITOR", "value": "editor"},
            {"name": "📄 Show staged diff", "value": "diff"},
        ]
        action = _stable_select("Choose an action", choices, default="regen")
        if action is None:
            # User canceled or menu failed; just return to input
            continue
        handled = _handle_action(action, ask_options=(action == "commit"))
        if handled is False:
            return
        if handled is True or handled is None:
            continue


@config_app.command("show")
def config_show():
    """Show effective configuration."""
    cfg = DiffMindConfig.load()
    console.print(cfg.to_dict())


@config_app.command("path")
def config_path(scope: str = typer.Option("user", help="user|repo")):
    repo = None
    try:
        repo = _repo()
    except Exception:
        pass
    rr = Path(repo.git_dir).parent if repo else None
    console.print(preferred_write_path(rr, scope=scope))


@config_app.command("set")
def config_set(
    provider: Optional[str] = typer.Option(None, help="simple|openai"),
    openai_api_key: Optional[str] = typer.Option(None, help="OpenAI API key (sk-...)"),
    openai_model: Optional[str] = typer.Option(None, help="OpenAI model id"),
    scope: str = typer.Option("user", help="Where to save: user|repo"),
):
    """Set configuration values and save file."""
    cfg = DiffMindConfig.load()
    if provider:
        cfg.provider = provider
    if openai_api_key:
        cfg.openai_api_key = openai_api_key
    if openai_model:
        cfg.openai_model = openai_model
    repo = None
    try:
        repo = _repo()
    except Exception:
        pass
    rr = Path(repo.git_dir).parent if repo else None
    path = save_config(cfg, preferred_write_path(rr, scope=scope))
    console.print(f"Saved config to {path}")


@config_app.command("wizard")
def config_wizard():
    """Interactive setup for configuring provider and API keys."""
    banner("[b]DiffMind Setup[/b]")
    # Prefer arrow-key selection via InquirerPy; fallback to typed prompt
    prov: str
    try:
        prov = _select_menu(
            "Choose provider",
            [
                {"name": "auto — detect automatically", "value": "auto"},
                {"name": "simple — local heuristics (no AI)", "value": "simple"},
                {"name": "openai — OpenAI (GPT via API)", "value": "openai"},
            ],
            default="auto",
        )
    except Exception:
        prov = typer.prompt("Choose provider [auto/simple/openai]", default="auto")

    cfg = DiffMindConfig.load({"provider": prov})
    if not prov:
        tip("Setup canceled.")
        return
    if prov.lower() == "openai":
        key = typer.prompt("Enter OpenAI API key (sk-...)", hide_input=True)
        cfg.openai_api_key = key
        model = typer.prompt("Model", default=cfg.openai_model)
        cfg.openai_model = model
    auto_install = Confirm.ask("Auto-install 'openai' package when missing?", default=True)
    cfg.auto_install_openai = bool(auto_install)
    try:
        scope = _select_menu(
            "Where to save the config?",
            [
                {"name": "user — ~/.config/diffmind/config.toml", "value": "user"},
                {"name": "repo — ./.diffmind.toml", "value": "repo"},
            ],
            default="user",
        )
    except Exception:
        scope = typer.prompt("Save to [user/repo]", default="user")
    if not scope:
        scope = "user"
    repo = None
    try:
        repo = _repo()
    except Exception:
        pass
    rr = Path(repo.git_dir).parent if repo else None
    path = save_config(cfg, preferred_write_path(rr, scope=scope))
    console.print(f"✔ Saved configuration to {path}")


def _build_llm_and_tools(cfg: DiffMindConfig):
    provider = (cfg.provider or "simple").lower()
    tools = [GitDiffTool(), GitCommitTool(), GitRepoTool(), CalculatorTool()]
    if provider == "openai":
        import os as _os

        key = cfg.openai_api_key or _os.getenv("OPENAI_API_KEY")
        if not key:
            return NoopLLM(), tools, True
        try:
            return OpenAIChat(api_key=key, base_url=cfg.openai_base_url, model=cfg.openai_model), tools, False
        except Exception:
            return NoopLLM(), tools, True
    return NoopLLM(), tools, False


def _sanitize_agent_output(text: str) -> str:
    # Remove fenced code block markers like ```bash ... ``` while keeping content
    lines = (text or "").splitlines()
    out = []
    in_fence = False
    for ln in lines:
        s = ln.strip()
        if s.startswith("```"):
            in_fence = not in_fence
            continue
        out.append(ln)
    s = "\n".join(out)
    # Collapse excessive blank lines
    import re as _re
    s = _re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


def _chat_loop(repo: Optional[Repo], cfg: DiffMindConfig, session: Optional[ChatSession] = None) -> None:
    # Create/find session
    sess = session or ChatSession.create(repo)
    llm, tools, misconfigured = _build_llm_and_tools(cfg)
    agent = AgentOrchestrator(llm, tools, max_steps=4)
    banner("[b]DiffMind Chat[/b]")
    console.print(f"Session: {sess.session_id}")
    if misconfigured:
        console.print(Panel("AI not configured. Answers will be limited. Use 'diffmind init' to enable.", border_style="yellow", box=box.ROUNDED))

    def _print_last(n: int = 6):
        for m in sess.messages[-n:]:
            role = "You" if m.role == "user" else "Agent"
            style = "cyan" if m.role == "user" else "green"
            console.print(Panel(m.content, title=role, border_style=style, box=box.ROUNDED))

    _print_last()

    while True:
        # Input
        # Prompt with live slash palette when available
        try:
            txt = _prompt_with_slash("Agent>", [
                ("/models", "choose what model to use"),
                ("/mode", "switch provider: auto/simple/openai"),
                ("/emojis", "choose emojis on/off"),
                ("/help", "show this help"),
            ])
            if txt is not None:
                if txt == SENT_MENU:
                    # Open a minimal action menu for chat? Keep behavior simple: just continue
                    continue
                text = txt
            else:
                from InquirerPy import inquirer as _inq  # type: ignore

                text = _inq.text(message="Agent>", multiline=False).execute()
        except Exception:
            text = input("Agent> ")
        if not text:
            continue
        t = text.strip()
        if t in {"/exit", "/quit"}:
            sess.save()
            return
        if t == "/save":
            sess.save()
            console.print(f"Saved: {sess.path}")
            continue
        if t == "/new":
            sess.save()
            sess = ChatSession.create(repo)
            console.print(f"New session: {sess.session_id}")
            continue
        if t in {"/help", "/?", "/"}:
            # Minimal command palette for chat
            try:
                sel = _stable_select(
                    "/ commands",
                    [
                        {"name": "/models     choose what model to use", "value": "models"},
                        {"name": "/mode       switch provider (auto/simple/openai)", "value": "mode"},
                        {"name": "/emojis     choose emojis on/off", "value": "emojis"},
                        {"name": "/help       show this help", "value": "help"},
                    ],
                    default="models",
                )
                if sel is None:
                    continue
                if sel == "models":
                    _choose_openai_model(cfg, repo)
                    llm, tools, misconfigured = _build_llm_and_tools(cfg)
                    agent = AgentOrchestrator(llm, tools, max_steps=4)
                    status_panel(cfg)
                elif sel == "mode":
                    _choose_provider_mode(cfg, repo)
                    llm, tools, misconfigured = _build_llm_and_tools(cfg)
                    agent = AgentOrchestrator(llm, tools, max_steps=4)
                    status_panel(cfg)
                elif sel == "emojis":
                    _choose_emojis(cfg, repo)
                    status_panel(cfg)
                else:
                    # Closed or no selection
                    pass
            except Exception:
                console.print(Panel("/models — choose OpenAI model\n/mode   — switch provider (auto/simple/openai)\n/emojis — choose emojis on/off\n/help   — show this help", title="Commands", border_style="blue", box=box.ROUNDED))
            continue
        if t == "/models" or t.startswith("/models"):
            parts = t.split(maxsplit=1)
            if len(parts) > 1 and _apply_models_token(parts[1], cfg, repo):
                # Rebuild LLM with new model
                llm, tools, misconfigured = _build_llm_and_tools(cfg)
                agent = AgentOrchestrator(llm, tools, max_steps=4)
                status_panel(cfg)
                continue
            _choose_openai_model(cfg, repo)
            # Rebuild LLM with new model
            llm, tools, misconfigured = _build_llm_and_tools(cfg)
            agent = AgentOrchestrator(llm, tools, max_steps=4)
            status_panel(cfg)
            continue
        if t == "/mode" or t.startswith("/mode"):
            _choose_provider_mode(cfg, repo)
            llm, tools, misconfigured = _build_llm_and_tools(cfg)
            agent = AgentOrchestrator(llm, tools, max_steps=4)
            status_panel(cfg)
            continue
        if t == "/emojis" or t.startswith("/emojis"):
            parts = t.split()
            if len(parts) >= 2 and parts[1].lower() in {"on", "off"}:
                cfg.emojis = (parts[1].lower() == "on")
                _save_user_config(cfg, repo)
            else:
                _choose_emojis(cfg, repo)
            status_panel(cfg)
            continue

        # Natural-language config changes
        if _apply_config_from_text(t, cfg, repo):
            llm, tools, misconfigured = _build_llm_and_tools(cfg)
            agent = AgentOrchestrator(llm, tools, max_steps=4)
            status_panel(cfg)
            continue

        # Ask agent
        sess.add_user(t)
        ctx = ToolContext(cfg=cfg, repo=repo)
        try:
            answer = agent.run_git_assistant(t, ctx, history=sess.messages[:-1])
        except Exception as e:
            answer = f"error: {e}"
        sess.add_assistant(_sanitize_agent_output(answer))
        sess.save()
        console.print(Panel(_sanitize_agent_output(answer), title="Agent", border_style="green", box=box.ROUNDED))


@app.command()
def chat(
    provider: Optional[str] = typer.Option(None, help="Provider: auto|simple|openai"),
    session_id: Optional[str] = typer.Option(None, help="Resume session by id (YYYYMMDD-HHMMSS) or pass a .json path"),
):
    """Open an agent chat REPL with tool use and session persistence."""
    cfg = DiffMindConfig.load({"provider": provider} if provider else None)
    # Repo is optional for chat
    repo: Optional[Repo] = None
    try:
        repo = _repo()
    except Exception:
        repo = None

    session: Optional[ChatSession] = None
    if session_id:
        p = Path(session_id)
        if p.exists():
            session = ChatSession.load(p)
        else:
            # Look up by id in sessions dir
            for f in ChatSession.list_sessions(repo):
                if f.stem == session_id:
                    session = ChatSession.load(f)
                    break
    _chat_loop(repo, cfg, session=session)

if __name__ == "__main__":  # pragma: no cover
    app()
