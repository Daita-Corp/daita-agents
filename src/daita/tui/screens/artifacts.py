"""Browse and manage the open agent's registered artifacts through owner APIs."""

from __future__ import annotations

import asyncio
import csv
from io import StringIO

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Label, OptionList, Static
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
            yield OptionList(id="artifacts-list")
            with VerticalScroll(id="artifacts-detail-scroll"):
                yield Static("", id="artifacts-detail", markup=False)
            yield Static("", id="artifacts-notice", markup=False)
            with Horizontal(id="artifacts-actions"):
                yield Button("Preview", id="artifacts-preview")
                yield Button("Save copy", id="artifacts-save")
                yield Button("Delete", id="artifacts-delete", variant="error")
            with Horizontal(id="artifacts-navigation"):
                yield Button("Previous", id="artifacts-previous")
                yield Button("Next", id="artifacts-next")
                yield Button("Refresh", id="artifacts-refresh")
                yield Button("Close", id="artifacts-close")
            yield Static("", id="artifacts-error", markup=False)
            yield Footer()

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
                payload = await controller.read_artifact(selected.artifact_id)
                preview = await asyncio.to_thread(render_artifact_preview, payload)
                self.query_one("#artifacts-detail", Static).update(
                    render_artifact_details(selected) + "\n\nPreview\n" + preview
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
        except (ArtifactError, ValueError, RuntimeError, OSError) as error:
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
        for item in self._items:
            label = f"{item.filename} · {item.byte_size:,} bytes · {item.created_at:%Y-%m-%d} · {item.artifact_id[-8:]}"
            listing.add_option(
                Option(
                    Text(safe_display(label, fallback="Artifact", maximum=512)),
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
            f"All conversations · artifacts {offset + 1}–{offset + len(self._items)} · page {offset // PAGE_SIZE + 1}"
            if self._items
            else "No stored artifacts."
        )
        self._render_selected()

    def _render_selected(self) -> None:
        selected = self._selected()
        self.query_one("#artifacts-detail", Static).update(
            render_artifact_details(selected)
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


def render_artifact_preview(payload: ArtifactPayload) -> str:
    """Render bounded literal text or the verified Daita XLSX Data worksheet."""
    if payload.ref.media_type == XLSX_MEDIA_TYPE:
        data = read_exact_xlsx_data(
            payload.content, expected_authorship=payload.ref.provenance.authorship
        )
        buffer = StringIO()
        writer = csv.writer(buffer)
        truncated = len(data.rows) > 20
        # Bound each cell and stop building the preview once its display is full.
        for row in (data.columns, *data.rows[:20]):
            cells = tuple("" if value is None else str(value) for value in row)
            truncated |= any(len(cell) > 512 for cell in cells)
            writer.writerow(tuple(cell[:512] for cell in cells))
            if buffer.tell() > MAX_PREVIEW_BYTES:
                truncated = True
                break
        text = buffer.getvalue()
    elif payload.ref.media_type in {*TEXT_EDIT_MEDIA_TYPES, HTML_MEDIA_TYPE}:
        text = payload.content[:MAX_PREVIEW_BYTES].decode("utf-8", errors="replace")
        truncated = len(payload.content) > MAX_PREVIEW_BYTES
    else:
        return "No text preview for this format. Save a copy to open it locally."
    preview = sanitize_terminal_text(
        text,
        maximum=MAX_PREVIEW_BYTES,
        preserve_lines=True,
        fallback="(empty artifact)",
    )
    if truncated:
        preview += "\n\nPreview truncated. Save a copy to view the complete artifact."
    return preview
