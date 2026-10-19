"""Render branded startup and empty-chat welcome content."""

from __future__ import annotations

import asyncio
import math
import random

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
    "████████████▄",
    "█████      ███▄",
    "█████       ███",
    "█████       ███",
    "█████       ███",
    "█████       ███",
    "█████      ███▀",
    "████████████▀",
)
_BRAILLE_SCENE = (
    "⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣤⡀",
    "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄",
    "⠿⠿⠿⠿⠿⠿⠏      ⠹⠿⠿⠿⠿⠿⠿",
    "⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤",
    "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿",
    "⣿⣿⣿⣿⣿⣿⡿⠛⠛⠛⠛⠛⠛⢿⣿⣿⣿⣿⣿⣿",
    "⠛⠛⠛⠛⠛⠛⠃      ⠘⠛⠛⠛⠛⠛⠛",
    "⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶",
    "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃",
    "⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠛⠁",
)

_INTRO_FRAMES = 24
_INTRO_FPS = 30
# Braille dot bits by (column, row) inside one 2x4 cell.
_BRAILLE_BITS = {
    (0, 0): 0x01,
    (0, 1): 0x02,
    (0, 2): 0x04,
    (1, 0): 0x08,
    (1, 1): 0x10,
    (1, 2): 0x20,
    (0, 3): 0x40,
    (1, 3): 0x80,
}


def _intro_paths() -> tuple[tuple[float, float, float, float, float], ...]:
    """Give every scene dot a start angle, start radius, delay, and target."""

    center_x, center_y = _SCENE_WIDTH, len(_BRAILLE_SCENE) * 2
    rng = random.Random(7)  # Fixed seed: same swirl on every start.
    paths = []
    for row, line in enumerate(_BRAILLE_SCENE):
        for column, character in enumerate(line):
            mask = ord(character) - 0x2800 if "\u2800" <= character <= "\u28ff" else 0
            for (dx, dy), bit in _BRAILLE_BITS.items():
                if mask & bit:
                    x, y = column * 2 + dx - center_x, row * 4 + dy - center_y
                    angle = math.atan2(y, x)
                    paths.append(
                        (
                            angle - math.pi * rng.uniform(0.8, 1.4),
                            rng.uniform(30, 44),
                            rng.uniform(0, 0.35),
                            angle,
                            math.hypot(x, y),
                        )
                    )
    return tuple(paths)


_INTRO_PATHS = _intro_paths()


def _intro_scene(progress: float) -> tuple[str, ...]:
    """Draw the braille scene with every dot partway along its inward spiral."""

    if progress >= 1:
        return _BRAILLE_SCENE
    center_x, center_y = _SCENE_WIDTH, len(_BRAILLE_SCENE) * 2
    cells = [[0] * _SCENE_WIDTH for _ in _BRAILLE_SCENE]
    for start_angle, start_radius, delay, angle, radius in _INTRO_PATHS:
        step = min(1.0, max(0.0, (progress - delay) / (1 - delay)))
        ease = 1 - (1 - step) ** 3
        turn = start_angle + (angle - start_angle) * ease
        reach = start_radius + (radius - start_radius) * ease
        x = round(center_x + reach * math.cos(turn))
        y = round(center_y + reach * math.sin(turn))
        if 0 <= x < _SCENE_WIDTH * 2 and 0 <= y < len(_BRAILLE_SCENE) * 4:
            cells[y // 4][x // 2] |= _BRAILLE_BITS[(x % 2, y % 4)]
    return tuple(
        "".join(chr(0x2800 + mask) if mask else " " for mask in row) for row in cells
    )


class WelcomeView(Static):
    """Responsive Daita identity shown at process start and in an empty chat."""

    def __init__(self, *, booting: bool = False, id: str | None = None) -> None:
        super().__init__("", id=id, markup=False)
        self._booting = booting
        self._agent = "your agent"
        self._model = "model"
        self._source = "data source"
        self._intro_frame = _INTRO_FRAMES
        self._intro_done = asyncio.Event()
        self._intro_done.set()

    def on_mount(self) -> None:
        if self._booting and self.app.animation_level != "none":
            self._intro_frame = 0
            self._intro_done.clear()
            self._intro_timer = self.set_interval(
                1 / _INTRO_FPS, self._advance_intro, name="daita-welcome-intro"
            )
        self._refresh_content()

    async def intro_finished(self) -> None:
        """Wait until the startup swirl has settled into the logo."""

        await self._intro_done.wait()

    def _advance_intro(self) -> None:
        self._intro_frame += 1
        # Only the wide braille scene animates; narrower layouts skip ahead.
        if 0 < self.size.width < _SCENE_MINIMUM:
            self._intro_frame = _INTRO_FRAMES
        if self._intro_frame >= _INTRO_FRAMES:
            self._intro_timer.stop()
            self._intro_done.set()
        self._refresh_content()

    def on_unmount(self) -> None:
        self._intro_done.set()

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
            Text("Your persistent data agent", style="bold"),
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
                        "Ask across connections or files; sources are optional",
                        style="bold",
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
        for index, (logo, detail) in enumerate(zip(_LOGO_LINES, details, strict=True)):
            content.append(logo.ljust(_LOGO_WIDTH), style="bold")
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
        scene = _intro_scene(self._intro_frame / _INTRO_FRAMES)
        for index, (characters, detail) in enumerate(zip(scene, details, strict=True)):
            content.append(characters.ljust(_SCENE_WIDTH), style="bold")
            content.append(" " * _SCENE_GAP)
            detail.truncate(detail_width, overflow="ellipsis")
            content.append_text(detail)
            if index < len(_BRAILLE_SCENE) - 1:
                content.append("\n")
        return content

    def _compact_content(self) -> Text:
        content = Text(justify="center", no_wrap=True, overflow="crop")
        for index, line in enumerate(_LOGO_LINES):
            content.append(line, style="bold")
            if index < len(_LOGO_LINES) - 1:
                content.append("\n")
        content.append("\n\nDAITA", style="bold")
        content.append(f"  {__version__}", style="dim")
        if self._booting:
            content.append("\nStarting your workspace…", style="dim")
        else:
            content.append(
                "\nAsk across connections or files; sources are optional", style="bold"
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
