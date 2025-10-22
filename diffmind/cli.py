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
from .generator import generate_commit_message
from .hooks import HOOK_NAME, install_hook, run_prepare_commit_msg, uninstall_hook
from .ui import banner, print_message, tip, openai_help_panel, status_panel, simple_mode_warning
from .agents import AgentOrchestrator, OpenAIChat, NoopLLM, GitDiffTool, CalculatorTool, GitCommitTool, GitRepoTool
from .agents.base import ToolContext, ChatMessage
from .agents.session import ChatSession
from .providers.base import ProviderConfigError
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
):
    """Suggest a commit message from staged changes."""
    repo = _repo()
    cfg = DiffMindConfig.load({"provider": provider} if provider else None)
    try:
        msg = generate_commit_message(repo, cfg)
    except ProviderConfigError as e:
        text = str(e).lower()
        missing_pkg = "package is not installed" in text
        missing_key = "api key" in text
        # If only the package is missing and we have/will have a key, offer to install now
        if missing_pkg and not missing_key:
            if cfg.auto_install_openai or Confirm.ask("Install 'openai' package now?", default=True):
                if _install_openai():
                    cfg = DiffMindConfig.load()
                    msg = generate_commit_message(repo, cfg)
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
                    msg = generate_commit_message(repo, cfg)
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
):
    """Generate and commit with the suggested message."""
    repo = _repo()
    cfg = DiffMindConfig.load({"provider": provider} if provider else None)
    if all:
        subprocess.run(["git", "add", "-A"], check=False)

    try:
        msg = generate_commit_message(repo, cfg)
    except ProviderConfigError as e:
        text = str(e).lower()
        missing_pkg = "package is not installed" in text
        missing_key = "api key" in text
        if missing_pkg and not missing_key:
            if cfg.auto_install_openai or Confirm.ask("Install 'openai' package now?", default=True):
                if _install_openai():
                    cfg = DiffMindConfig.load()
                    msg = generate_commit_message(repo, cfg)
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
                    msg = generate_commit_message(repo, cfg)
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
        from InquirerPy import inquirer as _inq  # type: ignore
        choices = _available_openai_models()
        preselect = default_model if any(c["value"] == default_model for c in choices) else choices[0]["value"]
        sel = _inq.select(
            message="Choose OpenAI model",
            choices=choices,
            default=preselect,
        ).execute()
        if not sel:
            tip("Model unchanged.")
            return
        cfg.openai_model = sel
    except KeyboardInterrupt:
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


def _prompt_with_slash(message: str, commands: list[tuple[str, str]]) -> Optional[str]:
    """Prompt for a single line. If user types '/', show commands dropdown immediately."""
    try:
        from prompt_toolkit import PromptSession  # type: ignore
    except Exception:
        return None
    try:
        session = PromptSession()
        completer = _SlashCompleter(commands)
        # complete_while_typing shows menu as the user types
        text = session.prompt(
            message,
            completer=completer,
            complete_while_typing=True,
            reserve_space_for_menu=2,
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

    # Render initial suggestion once (banner + status)
    banner("[b]Commit Message Suggestion[/b]")
    status_panel(cfg)
    if (cfg.provider or "simple").lower() == "simple":
        simple_mode_warning()
    print_message(msg.subject, msg.body)

    while True:

        # Prompt with live slash palette when available (minimal palette)
        instr = _prompt_with_slash("Enter instruction (or press Enter to open menu):", [
            ("/models", "choose what model to use"),
            ("/help", "show this help"),
        ]) or inquirer.text(message="Enter instruction (or press Enter to open menu):").execute()
        if instr and instr.strip():
            t = instr.strip()
            # Slash commands
            if t in {"/help", "/?", "/"}:
                # Minimal command palette
                try:
                    from InquirerPy import inquirer as _inq  # type: ignore

                    sel = _inq.select(
                        message="/ commands",
                        choices=[
                            {"name": "/models     choose what model to use", "value": "models"},
                            {"name": "/help       show this help", "value": "help"},
                        ],
                        default="models",
                    ).execute()
                    if sel == "models":
                        _choose_openai_model(cfg, repo)
                        status_panel(cfg)
                    else:
                        from rich.panel import Panel as _Panel

                        console.print(
                            _Panel(
                                "Available commands:\n"
                                "  /models — choose OpenAI model\n"
                                "  /help   — show this help",
                                title="Commands",
                                border_style="blue",
                                box=box.ROUNDED,
                            )
                        )
                except Exception:
                    from rich.panel import Panel as _Panel

                    console.print(
                        _Panel(
                            "Available commands:\n"
                            "  /models — choose OpenAI model\n"
                            "  /help   — show this help",
                            title="Commands",
                            border_style="blue",
                            box=box.ROUNDED,
                        )
                    )
                continue
            if t == "/models" or t.startswith("/models"):
                _choose_openai_model(cfg, repo)
                # Reprint status to reflect model change
                status_panel(cfg)
                continue
            # Natural command shortcuts
            action = _interpret_instruction(instr)
            if action == "commit":
                with console.status("Committing…", spinner="dots"):
                    _do_commit(msg)
                return
            if action == "regen":
                with console.status(
                    f"Regenerating suggestion ({cfg.provider or 'simple'})…", spinner="dots"
                ):
                    msg = generate_commit_message(repo, cfg)
                # Show updated message (avoid re-printing banner/status)
                print_message(msg.subject, msg.body)
                continue
            if action == "diff":
                diff = get_staged_diff_text(repo, unified=3) or "(no staged diff)"
                console.print(Panel(diff, title="Staged Diff", border_style="magenta", box=box.ROUNDED))
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
            # Show updated message only
            print_message(msg.subject, msg.body)
            continue

        choices = [
            {"name": "✅ Commit", "value": "commit"},
            {"name": "🔁 Regenerate", "value": "regen"},
            {"name": "✏️  Edit subject/body", "value": "edit"},
            {"name": "📝 Open in $EDITOR", "value": "editor"},
            {"name": "➕ Add bullet to body", "value": "add"},
            {"name": "🤖 Agent chat", "value": "chat"},
            {"name": "📄 Show staged diff", "value": "diff"},
            {"name": "⚙️  Config wizard", "value": "wizard"},
            {"name": "🚪 Quit", "value": "quit"},
        ]
        action = inquirer.select(message="Choose an action", choices=choices, default="commit").execute()
        if action == "quit":
            return
        if action == "diff":
            diff = get_staged_diff_text(repo, unified=3) or "(no staged diff)"
            console.print(Panel(diff, title="Staged Diff", border_style="magenta", box=box.ROUNDED))
            continue
        if action == "regen":
            msg = generate_commit_message(repo, cfg)
            continue
        if action == "edit":
            subj = inquirer.text(message="Subject", default=msg.subject).execute()
            body = inquirer.text(message="Body (empty — keep as is)", default=msg.body or "").execute()
            msg.subject, msg.body = subj, (body or None)
            print_message(msg.subject, msg.body)
            continue
        if action == "editor":
            content = msg.subject + ("\n\n" + msg.body if msg.body else "")
            new = _edit_in_editor(content)
            parts = new.splitlines()
            new_subject = (parts[0].strip() if parts else msg.subject) or msg.subject
            new_body = "\n".join(parts[1:]).strip() or None
            msg.subject, msg.body = new_subject, new_body
            print_message(msg.subject, msg.body)
            continue
        if action == "add":
            line = inquirer.text(message="Bullet line", default="").execute()
            if line.strip():
                if msg.body:
                    msg.body = (msg.body + "\n- " + line.strip()).rstrip()
                else:
                    msg.body = "- " + line.strip()
            print_message(msg.subject, msg.body)
            continue
        if action == "wizard":
            config_wizard()
            cfg = DiffMindConfig.load()  # reload
            continue
        if action == "model":
            _choose_openai_model(cfg, repo)
            status_panel(cfg)
            continue
        if action == "commit":
            _do_commit(msg, ask_options=True)
        if action == "chat":
            _chat_loop(repo, cfg)


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
        from InquirerPy import inquirer as _inq  # type: ignore

        prov = _inq.select(
            message="Choose provider",
            choices=[
                {"name": "auto — detect automatically", "value": "auto"},
                {"name": "simple — local heuristics (no AI)", "value": "simple"},
                {"name": "openai — OpenAI (GPT via API)", "value": "openai"},
            ],
            default="auto",
        ).execute()
    except Exception:
        prov = typer.prompt("Choose provider [auto/simple/openai]", default="auto")

    cfg = DiffMindConfig.load({"provider": prov})
    if prov.lower() == "openai":
        key = typer.prompt("Enter OpenAI API key (sk-...)", hide_input=True)
        cfg.openai_api_key = key
        model = typer.prompt("Model", default=cfg.openai_model)
        cfg.openai_model = model
    auto_install = Confirm.ask("Auto-install 'openai' package when missing?", default=True)
    cfg.auto_install_openai = bool(auto_install)
    try:
        from InquirerPy import inquirer as _inq  # type: ignore

        scope = _inq.select(
            message="Where to save the config?",
            choices=[
                {"name": "user — ~/.config/diffmind/config.toml", "value": "user"},
                {"name": "repo — ./.diffmind.toml", "value": "repo"},
            ],
            default="user",
        ).execute()
    except Exception:
        scope = typer.prompt("Save to [user/repo]", default="user")
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
                ("/help", "show this help"),
            ])
            if txt is not None:
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
                from InquirerPy import inquirer as _inq  # type: ignore

                sel = _inq.select(
                    message="/ commands",
                    choices=[
                        {"name": "/models     choose what model to use", "value": "models"},
                        {"name": "/help       show this help", "value": "help"},
                    ],
                    default="models",
                ).execute()
                if sel == "models":
                    _choose_openai_model(cfg, repo)
                    llm, tools, misconfigured = _build_llm_and_tools(cfg)
                    agent = AgentOrchestrator(llm, tools, max_steps=4)
                    status_panel(cfg)
                else:
                    console.print(Panel("/models — choose OpenAI model\n/help — show this help", title="Commands", border_style="blue", box=box.ROUNDED))
            except Exception:
                console.print(Panel("/models — choose OpenAI model\n/help — show this help", title="Commands", border_style="blue", box=box.ROUNDED))
            continue
        if t == "/models" or t.startswith("/models"):
            _choose_openai_model(cfg, repo)
            # Rebuild LLM with new model
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
