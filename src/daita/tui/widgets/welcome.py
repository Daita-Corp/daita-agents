"""Branded startup and empty-chat welcome presentation."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from ..sanitization import sanitize_terminal_text

_WIDE_LOGO = """\
██████╗  █████╗ ██╗████████╗ █████╗
██╔══██╗██╔══██╗██║╚══██╔══╝██╔══██╗
██║  ██║███████║██║   ██║   ███████║
██║  ██║██╔══██║██║   ██║   ██╔══██║
██████╔╝██║  ██║██║   ██║   ██║  ██║
╚═════╝ ╚═╝  ╚═╝╚═╝   ╚═╝   ╚═╝  ╚═╝"""


class WelcomeView(Static):
    """Responsive Daita identity shown at process start and in an empty chat."""

    def __init__(self, *, booting: bool = False, id: str | None = None) -> None:
        super().__init__("", id=id, markup=False)
        self._booting = booting
        self._agent = "your agent"
        self._model = "model"
        self._source = "data source"

    def on_mount(self) -> None:
        self._refresh_content()

    def on_resize(self) -> None:
        self._refresh_content()

    def update_identity(self, *, agent: str, model: str, source: str) -> None:
        self._agent = sanitize_terminal_text(
            agent, maximum=32, preserve_lines=False, fallback="agent"
        )
        self._model = sanitize_terminal_text(
            model, maximum=48, preserve_lines=False, fallback="model"
        )
        self._source = sanitize_terminal_text(
            source, maximum=48, preserve_lines=False, fallback="source"
        )
        self._refresh_content()

    def _refresh_content(self) -> None:
        width = self.size.width or 80
        content = Text(justify="center")
        if width >= 58:
            content.append(_WIDE_LOGO, style="bold #ACFD21")
        else:
            content.append("D  A  I  T  A", style="bold #ACFD21")
        content.append("\n\nDAITA · Your persistent data agent", style="bold")
        if self._booting:
            content.append("\nStarting your workspace…", style="dim")
        else:
            content.append(
                f"\n{self._agent}  ·  {self._model}  ·  {self._source}",
                style="dim",
            )
            content.append("\n\nAsk a question to begin", style="#ACFD21")
            content.append("\nType / for commands  ·  @ for sources", style="dim")
        self.update(content)


__all__ = ["WelcomeView"]
