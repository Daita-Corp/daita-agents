"""Atomic source-connection editing inside the Textual lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, Label, Select, Static

from ..models import SSL_MODES, PickerOption
from ..sanitization import sanitize_terminal_text
from .confirm import ConfirmScreen
from .selection import SelectionScreen


class SourceEditScreen(Screen[bool]):
    """Edit one active source only after the public API produces a preview."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self) -> None:
        super().__init__()
        self._source: Any = None
        self._adapter_id: str | None = None

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="source-edit"):
            yield Label("Edit a source connection", id="onboard-title", markup=False)
            yield Static(
                "The current connection remains active until its replacement "
                "validates, catalogs, and is confirmed.",
                id="source-edit-help",
                markup=False,
            )
            yield Button("Choose source", id="edit-source-choose")
            yield Input(placeholder="Display name", id="edit-source-name")
            yield Input(
                placeholder="SQLite file or local directory", id="edit-source-path"
            )
            yield Input(placeholder="Host", id="edit-pg-host")
            yield Input(placeholder="Port", id="edit-pg-port")
            yield Input(placeholder="Database", id="edit-pg-database")
            yield Input(placeholder="Username", id="edit-pg-username")
            yield Input(
                placeholder="New password (blank keeps current)",
                id="edit-pg-password",
                password=True,
            )
            yield Input(placeholder="Schemas (comma-separated)", id="edit-pg-schemas")
            yield Select(
                ((mode, mode) for mode in sorted(SSL_MODES)),
                prompt="SSL mode",
                id="edit-pg-ssl",
            )
            yield Label("", id="source-edit-error", markup=False)
            yield Button(
                "Validate and review", id="edit-source-apply", variant="primary"
            )
            yield Footer()

    async def on_mount(self) -> None:
        controller = self.app.controller  # type: ignore[attr-defined]
        source = await controller.active_source()
        if source is not None:
            self._load_source(source)
            return
        await self._choose_source()

    def action_cancel(self) -> None:
        self.dismiss(False)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "edit-source-choose":
            await self._choose_source()
            return
        if event.button.id != "edit-source-apply":
            return
        await self._apply()

    async def _choose_source(self) -> None:
        controller = self.app.controller  # type: ignore[attr-defined]
        sources = tuple(
            source for source in await controller.list_sources() if source.active
        )
        options = tuple(
            PickerOption(source.id, source.display_name, source.adapter_id)
            for source in sources
        )
        if not options:
            self._show_error("No active source is available to edit.")
            return
        selected = await self.app._await_modal(  # type: ignore[attr-defined]
            SelectionScreen(title="Choose a source to edit", options=options)
        )
        if selected is None:
            return
        source = next(item for item in sources if item.id == selected[0])
        self._load_source(source)

    def _load_source(self, source: Any) -> None:
        defaults = self.app.controller.source_edit_defaults(source)  # type: ignore[attr-defined]
        self._source = source
        self._adapter_id = str(defaults["adapter_id"])
        self.query_one("#edit-source-name", Input).value = str(defaults["name"])
        self.query_one("#edit-source-path", Input).value = str(defaults.get("path", ""))
        self.query_one("#edit-pg-host", Input).value = str(defaults.get("host", ""))
        self.query_one("#edit-pg-port", Input).value = str(defaults.get("port", ""))
        self.query_one("#edit-pg-database", Input).value = str(
            defaults.get("database", "")
        )
        self.query_one("#edit-pg-username", Input).value = str(
            defaults.get("username", "")
        )
        schemas = defaults.get("schemas", ())
        self.query_one("#edit-pg-schemas", Input).value = ", ".join(schemas)
        ssl_mode = defaults.get("ssl_mode")
        if isinstance(ssl_mode, str):
            self.query_one("#edit-pg-ssl", Select).value = ssl_mode
        self.query_one("#edit-pg-password", Input).clear()
        self.query_one("#source-edit-help", Static).update(
            sanitize_terminal_text(
                f"Editing {source.display_name} ({source.adapter_id}). "
                "The current connection remains active until confirmation.",
                maximum=512,
                preserve_lines=False,
                fallback="Editing source.",
            )
        )
        self._show_error("")

    async def _apply(self) -> None:
        if self._source is None or self._adapter_id is None:
            self._show_error("Choose a source first.")
            return
        name = self.query_one("#edit-source-name", Input).value.strip()
        if not name:
            self._show_error("Display name cannot be empty.")
            return
        password_input = self.query_one("#edit-pg-password", Input)
        password = password_input.value or None
        password_input.clear()
        try:
            if self._adapter_id in {"sqlite", "local-directory"}:
                raw_path = self.query_one("#edit-source-path", Input).value.strip()
                if not raw_path:
                    raise ValueError("Enter the replacement path.")
                result = await self.app.controller.edit_source_connection(  # type: ignore[attr-defined]
                    self._source,
                    name=name,
                    path=Path(raw_path).expanduser().resolve(strict=False),
                    confirmation_handler=self._confirm_preview,
                )
            else:
                try:
                    port = int(self.query_one("#edit-pg-port", Input).value)
                except ValueError as error:
                    raise ValueError(
                        "Port must be an integer from 1 through 65535."
                    ) from error
                if not 1 <= port <= 65_535:
                    raise ValueError("Port must be an integer from 1 through 65535.")
                schemas = tuple(
                    item.strip()
                    for item in self.query_one("#edit-pg-schemas", Input).value.split(
                        ","
                    )
                    if item.strip()
                )
                result = await self.app.controller.edit_source_connection(  # type: ignore[attr-defined]
                    self._source,
                    name=name,
                    host=self.query_one("#edit-pg-host", Input).value.strip(),
                    port=port,
                    database=self.query_one("#edit-pg-database", Input).value.strip(),
                    username=self.query_one("#edit-pg-username", Input).value.strip(),
                    password=password,
                    schemas=schemas,
                    ssl_mode=str(
                        self.query_one("#edit-pg-ssl", Select).value or "require"
                    ),
                    confirmation_handler=self._confirm_preview,
                )
        except Exception as error:
            self._show_error(
                sanitize_terminal_text(
                    str(error),
                    maximum=512,
                    preserve_lines=False,
                    fallback="Source edit failed.",
                )
            )
            return
        finally:
            password = None
        if result is None:
            self._show_error("Connection changes were not applied.")
            return
        self.dismiss(True)

    async def _confirm_preview(self, preview: Any) -> bool:
        if preview.read_mode.value == "all":
            read_summary = "all cataloged resources"
        elif preview.read_mode.value == "none":
            read_summary = "none"
        else:
            read_summary = (
                f"{preview.preserved_read_resource_count} exact matching resources"
            )
        lines = [
            "Apply this source connection change?",
            f"Catalog: {preview.resource_count} resources, "
            f"{preview.relationship_count} relationships",
            f"Read access carried forward: {read_summary}",
        ]
        if preview.omitted_read_resources:
            shown = ", ".join(preview.omitted_read_resources[:5])
            suffix = "" if len(preview.omitted_read_resources) <= 5 else ", ..."
            lines.append(f"Not carried forward: {shown}{suffix}")
        if preview.adapter_id == "postgresql":
            lines.append(
                "PostgreSQL update access: none; exact scopes must be enabled again"
            )
        lines.append("A new conversation will start; existing history is retained.")
        accepted = await self.app._await_modal(  # type: ignore[attr-defined]
            ConfirmScreen("\n".join(lines))
        )
        return bool(accepted)

    def _show_error(self, text: str) -> None:
        self.query_one("#source-edit-error", Label).update(text)
