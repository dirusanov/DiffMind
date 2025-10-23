from __future__ import annotations

from typing import Optional
from .config import DiffMindConfig

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


console = Console()


def banner(title: str) -> None:
    # Print a simple title line without box borders
    t = Text.from_markup(title)
    t.stylize("magenta")
    console.print(t)


def print_message(subject: str, body: Optional[str]) -> None:
    # Avoid rendering an empty panel when nothing to show
    if not (subject or body):
        return
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="left")
    if subject:
        table.add_row(Text(subject, style="bold green"))
    if body:
        table.add_row(Text(body, style="dim"))
    console.print(Panel.fit(table, border_style="green", box=box.ROUNDED))


def tip(text: str) -> None:
    console.print(Text(f"💡 {text}", style="italic dim"))


def simple_mode_warning() -> None:
    """Show a friendly warning that AI features are disabled in simple mode."""
    body = (
        "AI features are unavailable in simple mode.\n"
        "Set OPENAI_API_KEY and switch provider to enable AI."
    )
    console.print(Panel(Text(body), title="AI Disabled", border_style="yellow", box=box.ROUNDED))


def openai_help_panel(missing_pkg: bool = False, missing_key: bool = True) -> None:
    lines = []
    if missing_pkg:
        lines.append("• Install the package: 'pip install openai' or 'poetry add openai --group ai'")
    if missing_key:
        lines.append("• Set API key: export OPENAI_API_KEY=sk-... (temporary for current session)")
        lines.append("• Or save in config: 'diffmind config wizard' or 'diffmind config set --provider openai --openai-api-key sk-... --scope user'")
    lines.append("• After setup, try: 'diffmind suggest --provider openai'")
    body = "\n".join(lines)
    console.print(Panel(Text(body), title="OpenAI Setup", border_style="yellow", box=box.ROUNDED))


def status_panel(cfg: DiffMindConfig) -> None:
    from rich.table import Table as _Table

    table = _Table.grid(padding=(0, 2))
    table.add_column(justify="left", style="bold cyan")
    table.add_column(justify="left")
    mode = (cfg.provider or "simple").lower()
    table.add_row("Mode", mode)
    if mode == "openai":
        table.add_row("Model", cfg.openai_model or "(default)")
    table.add_row("Emojis", "on" if cfg.emojis else "off")
    console.print(Panel.fit(table, title="Status", border_style="blue", box=box.ROUNDED))
