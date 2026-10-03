"""Browse and manage the open agent's registered artifacts through owner APIs."""

from __future__ import annotations

import asyncio
import csv
import json
import re
from collections.abc import Sequence
from io import StringIO

from rich import box
from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Label, OptionList, Static
from textual.widgets.option_list import Option

from daita.artifacts import ArtifactError, ArtifactPayload, ArtifactRef
from daita.artifacts.renderers import (
    HTML_MEDIA_TYPE,
    TEXT_EDIT_MEDIA_TYPES,
    XLSX_MEDIA_TYPE,
    read_exact_xlsx_data,
)

from ..sanitization import safe_display, sanitize_terminal_text
from .confirm import ConfirmScreen

PAGE_SIZE = 50
MAX_PREVIEW_BYTES = 16_384
MAX_PREVIEW_ROWS = 20
MAX_PREVIEW_COLUMNS = 8


class ArtifactsScreen(ModalScreen[None]):
    """Human controls over current inventory, independent of conversation history."""

    BINDINGS = [
        Binding("escape", "close", "Back", priority=True),
        Binding("r", "refresh", "Refresh", priority=True),
        Binding("v", "preview", "Preview", priority=True),
        Binding("s", "save", "Save copy", priority=True),
        Binding("d", "delete", "Delete", priority=True),
        Binding("left", "previous", "Previous page", priority=True),
        Binding("right", "next", "Next page", priority=True),
        Binding("up", "select_previous", "Select previous", priority=True, show=False),
        Binding("down", "select_next", "Select next", priority=True, show=False),
        Binding("i", "details", "Details", priority=True),
        Binding(
            "pageup",
            "scroll_preview_up",
            "Scroll preview up",
            priority=True,
            show=False,
        ),
        Binding(
            "pagedown",
            "scroll_preview_down",
            "Scroll preview down",
            priority=True,
            show=False,
        ),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._items: tuple[ArtifactRef, ...] = ()
        self._offset = 0
        self._has_next = False
        self._busy = False

    def compose(self) -> ComposeResult:
        with Vertical(id="artifacts-manager"):
            yield Label("Artifacts", classes="title", markup=False)
            yield Static(
                "Loading stored artifacts…", id="artifacts-summary", markup=False
            )
            yield Static(render_artifact_list_row(None), id="artifacts-list-heading")
            yield OptionList(id="artifacts-list")
            yield Static("", id="artifacts-overview", markup=False)
            with VerticalScroll(id="artifacts-detail-scroll"):
                yield Static("", id="artifacts-detail", markup=False)
                with Collapsible(title="Details", id="artifacts-details"):
                    yield Static("", id="artifacts-metadata", markup=False)
                yield Static("", id="artifacts-notice", markup=False)
                yield Static("", id="artifacts-error", markup=False)
            with Horizontal(id="artifacts-actions"):
                yield Button("Preview", id="artifacts-preview")
                yield Button("Save copy", id="artifacts-save")
                yield Button("Delete", id="artifacts-delete", variant="error")
            with Horizontal(id="artifacts-navigation"):
                yield Button("Previous", id="artifacts-previous")
                yield Button("Next", id="artifacts-next")
                yield Button("Refresh", id="artifacts-refresh")
                yield Button("Close", id="artifacts-close")
            yield Static(
                "↑↓ Select · Enter Preview · i Details · PgUp/Dn Scroll · Esc",
                id="artifacts-help",
                markup=False,
            )

    def on_mount(self) -> None:
        self.on_resize()
        self._schedule("refresh")

    def on_resize(self) -> None:
        self.set_class(self.app.size.height < 30, "-compact")

    def action_close(self) -> None:
        if not self._busy:
            self.dismiss(None)

    def action_refresh(self) -> None:
        self._schedule("refresh")

    def action_preview(self) -> None:
        self._schedule("preview")

    def action_save(self) -> None:
        self._schedule("save")

    def action_delete(self) -> None:
        self._schedule("delete")

    def action_previous(self) -> None:
        if self._offset:
            self._schedule("previous")

    def action_next(self) -> None:
        if self._has_next:
            self._schedule("next")

    def action_select_previous(self) -> None:
        if not self._busy:
            listing = self.query_one("#artifacts-list", OptionList)
            listing.action_cursor_up()
            listing.focus()

    def action_select_next(self) -> None:
        if not self._busy:
            listing = self.query_one("#artifacts-list", OptionList)
            listing.action_cursor_down()
            listing.focus()

    def action_details(self) -> None:
        if not self._busy and self._selected() is not None:
            details = self.query_one("#artifacts-details", Collapsible)
            details.collapsed = not details.collapsed
            if not details.collapsed:
                details.scroll_visible(animate=False, top=True)

    def action_scroll_preview_up(self) -> None:
        self.query_one("#artifacts-detail-scroll", VerticalScroll).scroll_page_up(
            animate=False
        )

    def action_scroll_preview_down(self) -> None:
        self.query_one("#artifacts-detail-scroll", VerticalScroll).scroll_page_down(
            animate=False
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        action = (event.button.id or "").removeprefix("artifacts-")
        if action == "close":
            self.action_close()
        elif action == "previous":
            self.action_previous()
        elif action == "next":
            self.action_next()
        else:
            self._schedule(action)

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        if not self._busy:
            self._render_selected()
            self._update_actions()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.action_preview()

    def _selected(self) -> ArtifactRef | None:
        index = self.query_one("#artifacts-list", OptionList).highlighted
        return (
            self._items[index]
            if index is not None and index < len(self._items)
            else None
        )

    def _schedule(self, action: str) -> None:
        if self._busy:
            return
        selected = self._selected()
        if action in {"preview", "save", "delete"} and selected is None:
            return
        self._busy = True
        self._update_actions()
        self.run_worker(
            self._handle_action(action, selected),
            group="artifact-action",
            exclusive=True,
        )

    async def _handle_action(self, action: str, selected: ArtifactRef | None) -> None:
        self.query_one("#artifacts-error", Static).update("")
        self.query_one("#artifacts-notice", Static).update("")
        try:
            if action in {"refresh", "previous", "next"}:
                offset = self._offset
                if action == "previous":
                    offset -= PAGE_SIZE
                elif action == "next":
                    offset += PAGE_SIZE
                await self._load(offset, selected)
                return
            assert selected is not None
            controller = self.app.controller  # type: ignore[attr-defined]
            if action == "preview":
                self.query_one("#artifacts-detail", Static).update("Loading preview…")
                payload = await controller.read_artifact(selected.artifact_id)
                preview = await asyncio.to_thread(render_artifact_preview, payload)
                self.query_one("#artifacts-detail", Static).update(preview)
                self.query_one("#artifacts-detail-scroll", VerticalScroll).scroll_home(
                    animate=False
                )
                return
            if action == "save":
                destination = await controller.artifact_export_destination()
                accepted = await self.app.push_screen_wait(
                    ConfirmScreen(
                        f"Save a copy of {selected.filename} to {destination.display_name}? "
                        "Existing files will not be overwritten."
                    )
                )
                if accepted:
                    receipt = await controller.save_artifact(selected.artifact_id)
                    self.query_one("#artifacts-notice", Static).update(
                        "Saved copy: "
                        + safe_display(
                            receipt.saved_path, fallback="saved", maximum=512
                        )
                    )
                    self.query_one("#artifacts-notice", Static).scroll_visible(
                        animate=False, top=True
                    )
                return
            if action == "delete":
                accepted = await self.app.push_screen_wait(
                    ConfirmScreen(
                        f"Permanently delete {selected.filename}\n{selected.artifact_id}?\n"
                        "This removes the stored artifact. Conversation history and saved copies remain."
                    )
                )
                if accepted:
                    deleted = await controller.delete_artifact(selected.artifact_id)
                    await self._load(self._offset, selected)
                    self.query_one("#artifacts-notice", Static).update(
                        "Artifact deleted."
                        if deleted
                        else "Artifact is already unavailable."
                    )
                    self.query_one("#artifacts-notice", Static).scroll_visible(
                        animate=False, top=True
                    )
        except (ArtifactError, ValueError, RuntimeError, OSError) as error:
            if action == "preview":
                self.query_one("#artifacts-detail", Static).update(
                    "Preview is unavailable."
                )
            # Cleanup failures can hide the artifact before physical deletion completes.
            if action == "delete":
                try:
                    await self._load(self._offset, selected)
                except (ArtifactError, ValueError, RuntimeError, OSError):
                    pass
            message = str(error)
            if isinstance(error, ArtifactError):
                message = f"{error.code}: {message}"
                if error.code == "artifact_busy":
                    message += " Wait for the active job attempt to finish, then retry."
                elif error.details.get("stage") == "delete_cleanup":
                    message += " Reopen the agent to resume cleanup."
            self.query_one("#artifacts-error", Static).update(
                safe_display(message, fallback="Artifact action failed.", maximum=768)
            )
            self.query_one("#artifacts-error", Static).scroll_visible(
                animate=False, top=True
            )
        finally:
            if self.is_mounted:
                self._busy = False
                self._update_actions()
                self.query_one("#artifacts-list", OptionList).focus()

    async def _load(self, offset: int, selected: ArtifactRef | None) -> None:
        # One extra row determines whether the next page exists without a count query.
        items = await self.app.controller.list_artifacts(limit=PAGE_SIZE + 1, offset=offset)  # type: ignore[attr-defined]
        if not items and offset:
            await self._load(max(0, offset - PAGE_SIZE), selected)
            return
        self._offset = offset
        self._has_next = len(items) > PAGE_SIZE
        self._items = items[:PAGE_SIZE]
        listing = self.query_one("#artifacts-list", OptionList)
        listing.clear_options()
        for index, item in enumerate(self._items, offset + 1):
            listing.add_option(
                Option(
                    render_artifact_list_row(item, index),
                    id=item.artifact_id,
                )
            )
        if self._items:
            listing.highlighted = next(
                (
                    index
                    for index, item in enumerate(self._items)
                    if selected and item.artifact_id == selected.artifact_id
                ),
                0,
            )
        self.query_one("#artifacts-summary", Static).update(
            f"All conversations · {len(self._items)} stored artifacts"
            if self._items and offset == 0 and not self._has_next
            else (
                f"All conversations · artifacts {offset + 1}–{offset + len(self._items)} · page {offset // PAGE_SIZE + 1}"
                if self._items
                else "No stored artifacts."
            )
        )
        self._render_selected()

    def _render_selected(self) -> None:
        selected = self._selected()
        self.query_one("#artifacts-overview", Static).update(
            render_artifact_overview(selected) if selected else ""
        )
        details = self.query_one("#artifacts-details", Collapsible)
        details.collapsed = True
        details.display = selected is not None
        self.query_one("#artifacts-metadata", Static).update(
            render_artifact_details(selected) if selected else ""
        )
        self.query_one("#artifacts-detail", Static).update(
            "Press Enter or choose Preview to read this file."
            if selected
            else "Generated reports and exports will appear here, even after history is cleared."
        )
        self.query_one("#artifacts-detail-scroll", VerticalScroll).scroll_home(
            animate=False
        )

    def _update_actions(self) -> None:
        if not self.is_mounted:
            return
        self.query_one("#artifacts-list", OptionList).disabled = self._busy
        for action in ("preview", "save", "delete"):
            self.query_one(f"#artifacts-{action}", Button).disabled = (
                self._busy or self._selected() is None
            )
        for action in ("refresh", "close"):
            self.query_one(f"#artifacts-{action}", Button).disabled = self._busy
        self.query_one("#artifacts-previous", Button).disabled = (
            self._busy or self._offset == 0
        )
        self.query_one("#artifacts-next", Button).disabled = (
            self._busy or not self._has_next
        )
        for action in ("previous", "next"):
            self.query_one(f"#artifacts-{action}", Button).display = (
                self._offset > 0 or self._has_next
            )


def _display_name(ref: ArtifactRef) -> str:
    for prefix, label in (
        ("data-profile", "Data profile"),
        ("graph-result", "Job result"),
    ):
        if re.fullmatch(prefix + r"-job-[0-9a-f]{32}\.json", ref.filename):
            return label
    return ref.filename.rsplit(".", 1)[0] if "." in ref.filename else ref.filename


def _format_label(ref: ArtifactRef) -> str:
    if ref.media_type == XLSX_MEDIA_TYPE:
        return "XLSX"
    extension = (
        ref.filename.rsplit(".", 1)[-1].upper() if "." in ref.filename else "File"
    )
    return safe_display(extension, fallback="File", maximum=6)


def _file_size(byte_size: int) -> str:
    if byte_size < 1024:
        return f"{byte_size} B"
    if byte_size < 1024 * 1024:
        return f"{byte_size / 1024:.1f} KB"
    return f"{byte_size / (1024 * 1024):.1f} MB"


def render_artifact_list_row(ref: ArtifactRef | None, index: int = 0) -> Table:
    """Use a flexible name column so identifiers cannot push metadata offscreen."""
    row = Table.grid(padding=(0, 1), expand=True)
    row.add_column(width=3, justify="right", no_wrap=True)
    row.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    row.add_column(width=6, no_wrap=True)
    row.add_column(width=7, justify="right", no_wrap=True)
    row.add_column(width=21, no_wrap=True)
    values = (
        ("#", "Name", "Type", "Size", "Created (local)")
        if ref is None
        else (
            str(index),
            _display_name(ref),
            _format_label(ref),
            _file_size(ref.byte_size),
            ref.created_at.astimezone().strftime("%b %d %I:%M:%S %p"),
        )
    )
    row.add_row(
        *(Text(safe_display(value, fallback="", maximum=256)) for value in values)
    )
    return row


def render_artifact_overview(ref: ArtifactRef) -> str:
    created = ref.created_at.astimezone().strftime("%b %d, %Y · %I:%M %p %Z")
    return sanitize_terminal_text(
        f"{_display_name(ref)}\n{_format_label(ref)} · {_file_size(ref.byte_size)} · {created} · {ref.sensitivity.value.title()}",
        maximum=512,
        preserve_lines=True,
        fallback="Artifact",
    )


def render_artifact_details(ref: ArtifactRef) -> str:
    lines = [
        ref.filename,
        f"ID: {ref.artifact_id}",
        f"Format: {ref.media_type} · {ref.byte_size:,} bytes",
        f"Created: {ref.created_at.isoformat()}",
        f"Sensitivity: {ref.sensitivity.value}",
        f"Origin: {ref.provenance.authorship.value.replace('_', ' ')}",
        f"Conversation: {ref.conversation_id}",
        f"Run: {ref.run_id}",
        f"Checksum: {ref.sha256}",
    ]
    if ref.provenance.derived_from_artifact_id:
        lines.append(f"Parent artifact: {ref.provenance.derived_from_artifact_id}")
    return sanitize_terminal_text(
        "\n".join(lines),
        maximum=4096,
        preserve_lines=True,
        fallback="Artifact details unavailable.",
    )


def render_artifact_preview(payload: ArtifactPayload) -> RenderableType:
    """Format bounded content without executing markup, files or links."""
    if payload.ref.media_type == XLSX_MEDIA_TYPE:
        data = read_exact_xlsx_data(
            payload.content, expected_authorship=payload.ref.provenance.authorship
        )
        return _table_preview(
            data.columns,
            data.rows[:MAX_PREVIEW_ROWS],
            truncated=len(data.rows) > MAX_PREVIEW_ROWS,
        )
    elif payload.ref.media_type in {*TEXT_EDIT_MEDIA_TYPES, HTML_MEDIA_TYPE}:
        text = payload.content[:MAX_PREVIEW_BYTES].decode("utf-8", errors="replace")
        truncated = len(payload.content) > MAX_PREVIEW_BYTES
    else:
        return "No text preview for this format. Save a copy to open it locally."
    if payload.ref.media_type == "application/json" and not truncated:
        try:
            text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
        except (ValueError, RecursionError):
            pass
    elif payload.ref.media_type in {"text/csv", "text/tab-separated-values"}:
        try:
            reader = csv.reader(
                StringIO(text),
                delimiter=(
                    "\t"
                    if payload.ref.media_type == "text/tab-separated-values"
                    else ","
                ),
                strict=True,
            )
            columns = next(reader, [])
            rows: list[list[str]] = []
            for row in reader:
                if len(rows) == MAX_PREVIEW_ROWS:
                    truncated = True
                    break
                rows.append(row)
            if columns:
                return _table_preview(columns, rows, truncated=truncated)
        except csv.Error:
            pass
    truncated |= len(text) > MAX_PREVIEW_BYTES
    preview = sanitize_terminal_text(
        text,
        maximum=MAX_PREVIEW_BYTES,
        preserve_lines=True,
        fallback="(empty artifact)",
    )
    if payload.ref.media_type == "text/markdown":
        rendered: RenderableType = Markdown(preview, hyperlinks=False)
    else:
        rendered = Text(preview)
    return (
        Group(
            rendered,
            Text(
                "\nPreview truncated. Save a copy to view the complete artifact.",
                style="dim",
            ),
        )
        if truncated
        else rendered
    )


def _table_preview(
    columns: Sequence[str], rows: Sequence[Sequence[object]], *, truncated: bool
) -> RenderableType:
    """Render at most eight columns and twenty rows as literal cells."""
    table = Table(
        box=box.SIMPLE_HEAD,
        expand=True,
        header_style="bold",
        border_style="dim",
        show_edge=False,
    )
    truncated |= len(columns) > MAX_PREVIEW_COLUMNS or len(rows) > MAX_PREVIEW_ROWS
    for column in columns[:MAX_PREVIEW_COLUMNS]:
        truncated |= len(column) > 128
        table.add_column(
            Text(safe_display(column, fallback="", maximum=128)), overflow="fold"
        )
    for row in rows[:MAX_PREVIEW_ROWS]:
        cells = []
        truncated |= len(row) > len(table.columns)
        for value in row[: len(table.columns)]:
            cell = "" if value is None else str(value)
            truncated |= len(cell) > 64
            cells.append(Text(safe_display(cell, fallback="", maximum=64)))
        table.add_row(*cells)
    return (
        Group(
            table,
            Text(
                "\nPreview truncated. Save a copy to view the complete artifact.",
                style="dim",
            ),
        )
        if truncated
        else table
    )
