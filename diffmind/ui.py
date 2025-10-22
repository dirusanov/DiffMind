from __future__ import annotations

from typing import Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


console = Console()


def banner(title: str) -> None:
    console.print(Panel.fit(Text.from_markup(title), border_style="magenta", title="DiffMind", box=box.ROUNDED))


def print_message(subject: str, body: Optional[str]) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="left")
    table.add_row(Text(subject, style="bold green"))
    if body:
        table.add_row(Text(body, style="dim"))
    console.print(Panel.fit(table, border_style="green", box=box.ROUNDED))


def tip(text: str) -> None:
    console.print(Text(f"💡 {text}", style="italic dim"))


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
