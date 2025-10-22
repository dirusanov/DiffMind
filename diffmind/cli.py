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
from .ui import banner, print_message, tip, openai_help_panel
from .providers.base import ProviderConfigError
from .refiner import refine_with_openai, heuristic_refine


console = Console()
app = typer.Typer(add_completion=False, help="AI commit message generator (CLI + git hook)")
hook_app = typer.Typer(help="Manage git hooks")
config_app = typer.Typer(help="Configure DiffMind")
app.add_typer(hook_app, name="hook")
app.add_typer(config_app, name="config")


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
    interactive: Optional[bool] = typer.Option(None, help="Interactive mode with arrows & input"),
):
    """Suggest a commit message from staged changes."""
    repo = _repo()
    cfg = DiffMindConfig.load({"provider": provider} if provider else None)
    banner("[b]Commit Message Suggestion[/b]")
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

    # Choose interactive by default when TTY
    if interactive is None:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()

    if interactive:
        _interactive_suggest(repo, cfg, msg)
        return

    print_message(msg.subject, msg.body)
    if (cfg.provider or "simple").lower() == "simple":
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
    no_verify: bool = typer.Option(False, help="Pass --no-verify to git commit"),
    amend: bool = typer.Option(False, help="Amend the previous commit"),
    provider: Optional[str] = typer.Option(None, help="Provider: auto|simple|openai"),
    dry_run: bool = typer.Option(False, help="Only print the message, do not commit"),
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
    ai: Optional[bool] = typer.Option(None, help="Set up OpenAI now (auto-detect by default)"),
    hook: bool = typer.Option(True, help="Install git hook after setup"),
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

    if ai is None:
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


def _interpret_instruction(text: str) -> Optional[str]:
    t = text.strip().lower()
    commit_words = {"commit", "commit this", "ok", "looks good", "lgtm", "ship it", "закоммить", "коммит", "коммитим", "давай коммит", "всё ок", "все ок", "готово"}
    regen_words = {"regen", "regenerate", "again", "ещё", "еще", "перегенерируй", "перегенери", "снова", "другой"}
    diff_words = {"diff", "show diff", "покажи дифф", "дифф"}
    if t in commit_words:
        return "commit"
    if t in regen_words:
        return "regen"
    if t in diff_words:
        return "diff"
    return None


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

    while True:
        banner("[b]Commit Message Suggestion[/b]")
        print_message(msg.subject, msg.body)

        instr = inquirer.text(message="Enter instruction (or press Enter to open menu):").execute()
        if instr and instr.strip():
            # Natural command shortcuts
            action = _interpret_instruction(instr)
            if action == "commit":
                _do_commit(msg)
                return
            if action == "regen":
                msg = generate_commit_message(repo, cfg)
                continue
            if action == "diff":
                diff = get_staged_diff_text(repo, unified=3) or "(no staged diff)"
                console.print(Panel(diff, title="Staged Diff", border_style="magenta", box=box.ROUNDED))
                continue
            diff = get_staged_diff_text(repo, unified=0)
            refined = refine_with_openai(msg.subject, msg.body, diff, instr, cfg)
            if refined is None:
                refined = heuristic_refine(msg.subject, msg.body, instr, max_len=cfg.max_subject_length)
            msg = refined
            continue

        choices = [
            {"name": "✅ Commit", "value": "commit"},
            {"name": "🔁 Regenerate", "value": "regen"},
            {"name": "✏️  Edit subject/body", "value": "edit"},
            {"name": "📝 Open in $EDITOR", "value": "editor"},
            {"name": "➕ Add bullet to body", "value": "add"},
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
            continue
        if action == "editor":
            content = msg.subject + ("\n\n" + msg.body if msg.body else "")
            new = _edit_in_editor(content)
            parts = new.splitlines()
            new_subject = (parts[0].strip() if parts else msg.subject) or msg.subject
            new_body = "\n".join(parts[1:]).strip() or None
            msg.subject, msg.body = new_subject, new_body
            continue
        if action == "add":
            line = inquirer.text(message="Bullet line", default="").execute()
            if line.strip():
                if msg.body:
                    msg.body = (msg.body + "\n- " + line.strip()).rstrip()
                else:
                    msg.body = "- " + line.strip()
            continue
        if action == "wizard":
            config_wizard()
            cfg = DiffMindConfig.load()  # reload
            continue
        if action == "commit":
            _do_commit(msg, ask_options=True)


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


if __name__ == "__main__":  # pragma: no cover
    app()
