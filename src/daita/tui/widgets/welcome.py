"""Render branded startup and empty-chat welcome content."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from daita import __version__

from ..sanitization import sanitize_terminal_text

_LOGO_WIDTH = 15
_LOGO_GAP = 5
_WIDE_MINIMUM = 64
_SCENE_WIDTH = 20
_SCENE_GAP = 5
_SCENE_MINIMUM = 76
_LOGO_LINES = (
    ("████████████▄", "bold #ACFD21"),
    ("█████      ███▄", "bold #ACFD21"),
    ("█████       ███", "bold #70DF43"),
    ("█████       ███", "bold #70DF43"),
    ("█████       ███", "bold #70DF43"),
    ("█████       ███", "bold #46BF69"),
    ("█████      ███▀", "bold #46BF69"),
    ("████████████▀", "bold #46BF69"),
)
_SCENE_STYLES = {
    "T": "#ACFD21",
    "M": "#70DF43",
    "B": "#46BF69",
}
_BRAILLE_SCENE = (
    ("⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣤⡀", "TTTTTTTTTTTTTTTTTTT"),
    ("⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄", "TTTTTTTTTTTTTTTTTTTT"),
    ("⠿⠿⠿⠿⠿⠿⠏      ⠹⠿⠿⠿⠿⠿⠿", "TTTTTTT      TTTTTTT"),
    ("⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤", "MMMMMMMMMMMMMMMMMMMM"),
    ("⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿", "MMMMMMMMMMMMMMMMMMMM"),
    ("⣿⣿⣿⣿⣿⣿⡿⠛⠛⠛⠛⠛⠛⢿⣿⣿⣿⣿⣿⣿", "MMMMMMMMMMMMMMMMMMMM"),
    ("⠛⠛⠛⠛⠛⠛⠃      ⠘⠛⠛⠛⠛⠛⠛", "MMMMMMM      MMMMMMM"),
    ("⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶", "BBBBBBBBBBBBBBBBBBBB"),
    ("⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃", "BBBBBBBBBBBBBBBBBBBB"),
    ("⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠛⠁", "BBBBBBBBBBBBBBBBBBB"),
)


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

    def _detail_lines(self) -> tuple[Text, ...]:
        heading = Text("DAITA", style="bold")
        heading.append(f"  {__version__}", style="dim")
        lines = [
            heading,
            Text(),
            Text("Your persistent data agent", style="bold #ACFD21"),
        ]
        if self._booting:
            lines.extend((Text(), Text("Starting your workspace…", style="dim")))
        else:
            lines.extend(
                (
                    Text(
                        f"{self._agent}  ·  {self._model}  ·  {self._source}",
                        style="dim",
                    ),
                    Text(),
                    Text(
                        "Ask about workspace files; a source is optional", style="bold"
                    ),
                    Text(
                        "Type / for commands  ·  /files for workspace  ·  @ for sources",
                        style="dim",
                    ),
                )
            )
        while len(lines) < len(_LOGO_LINES):
            lines.append(Text())
        return tuple(lines)

    def _wide_content(self, width: int) -> Text:
        available = max(1, width - 4)
        detail_width = min(52, available - _LOGO_WIDTH - _LOGO_GAP)
        content = Text(no_wrap=True, overflow="crop")
        details = self._detail_lines()
        for index, ((logo, logo_style), detail) in enumerate(
            zip(_LOGO_LINES, details, strict=True)
        ):
            content.append(logo.ljust(_LOGO_WIDTH), style=logo_style)
            content.append(" " * _LOGO_GAP)
            detail.truncate(detail_width, overflow="ellipsis")
            content.append_text(detail)
            if index < len(_LOGO_LINES) - 1:
                content.append("\n")
        return content

    def _scene_content(self, width: int) -> Text:
        available = max(1, width - 4)
        detail_width = min(52, available - _SCENE_WIDTH - _SCENE_GAP)
        content = Text(no_wrap=True, overflow="crop")
        detail_lines = self._detail_lines()
        top_padding = max(0, (len(_BRAILLE_SCENE) - len(detail_lines)) // 2)
        details = (Text(),) * top_padding + detail_lines
        details += (Text(),) * (len(_BRAILLE_SCENE) - len(details))
        for index, ((characters, palette), detail) in enumerate(
            zip(_BRAILLE_SCENE, details, strict=True)
        ):
            characters = characters.ljust(_SCENE_WIDTH)
            palette = palette.ljust(_SCENE_WIDTH)
            run_start = 0
            for position in range(1, _SCENE_WIDTH + 1):
                if position < _SCENE_WIDTH and palette[position] == palette[run_start]:
                    continue
                content.append(
                    characters[run_start:position],
                    style=_SCENE_STYLES.get(palette[run_start]),
                )
                run_start = position
            content.append(" " * _SCENE_GAP)
            detail.truncate(detail_width, overflow="ellipsis")
            content.append_text(detail)
            if index < len(_BRAILLE_SCENE) - 1:
                content.append("\n")
        return content

    def _compact_content(self) -> Text:
        content = Text(justify="center", no_wrap=True, overflow="crop")
        for index, (line, style) in enumerate(_LOGO_LINES):
            content.append(line, style=style)
            if index < len(_LOGO_LINES) - 1:
                content.append("\n")
        content.append("\n\nDAITA", style="bold")
        content.append(f"  {__version__}", style="dim")
        if self._booting:
            content.append("\nStarting your workspace…", style="dim")
        else:
            content.append(
                "\nAsk about workspace files; a source is optional", style="bold"
            )
            content.append(
                "\nType / for commands  ·  /files for workspace", style="dim"
            )
        return content

    def _refresh_content(self) -> None:
        width = self.size.width or 80
        if width >= _SCENE_MINIMUM:
            content = self._scene_content(width)
        elif width >= _WIDE_MINIMUM:
            content = self._wide_content(width)
        else:
            content = self._compact_content()
        self.update(content)


__all__ = ["WelcomeView"]
