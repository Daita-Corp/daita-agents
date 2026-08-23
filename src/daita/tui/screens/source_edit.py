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
        self._preview_error: str | None = None

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="source-edit", classes="control-panel"):
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

    def on_mount(self) -> None:
        self.run_worker(
            self._load_initial_source(),
            name="source-edit-initial-source",
            group="source-edit-interaction",
            exclusive=True,
        )

    async def _load_initial_source(self) -> None:
        controller = self.app.controller  # type: ignore[attr-defined]
        source = await controller.active_source()
        if source is not None:
            self._load_source(source)
            return
        await self._choose_source()

    def action_cancel(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "edit-source-choose":
            self.run_worker(
                self._choose_source(),
                name="source-edit-selection",
                group="source-edit-interaction",
                exclusive=True,
            )
            return
        if event.button.id == "edit-source-apply":
            self.run_worker(
                self._apply(),
                name="source-edit-apply",
                group="source-edit-interaction",
                exclusive=True,
            )

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
        self._preview_error = None
        self._show_error("")
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
                entered_schemas = tuple(
                    item.strip()
                    for item in self.query_one("#edit-pg-schemas", Input).value.split(
                        ","
                    )
                    if item.strip()
                )
                host = self.query_one("#edit-pg-host", Input).value.strip()
                database = self.query_one("#edit-pg-database", Input).value.strip()
                username = self.query_one("#edit-pg-username", Input).value.strip()
                ssl_mode = str(
                    self.query_one("#edit-pg-ssl", Select).value or "require"
                )
                self._show_help("Testing connection and inspecting schemas…")
                probe = await self.app.controller.probe_postgresql_source(  # type: ignore[attr-defined]
                    self._source,
                    host=host,
                    port=port,
                    database=database,
                    username=username,
                    password=password,
                    ssl_mode=ssl_mode,
                )
                visible_schemas = tuple(item.name for item in probe.schemas)
                table_schemas = tuple(
                    item.name for item in probe.schemas if item.has_base_tables
                )
                if not visible_schemas:
                    raise ValueError(
                        "The connection succeeded, but no non-system schemas are visible."
                    )
                initial = tuple(
                    item for item in entered_schemas if item in visible_schemas
                )
                if entered_schemas == ("public",) and table_schemas:
                    initial = table_schemas
                elif not initial:
                    initial = table_schemas or visible_schemas
                options = tuple(
                    PickerOption(
                        item.name,
                        item.name,
                        "contains tables" if item.has_base_tables else "empty",
                    )
                    for item in probe.schemas
                )
                selected = await self.app._await_modal(  # type: ignore[attr-defined]
                    SelectionScreen(
                        title="Select PostgreSQL schemas",
                        options=options,
                        multi=True,
                        initial_selected=initial,
                    )
                )
                if selected is None:
                    self._show_help(
                        "Connection unchanged. Select schemas to continue reviewing it."
                    )
                    return
                schemas = tuple(selected)
                self.query_one("#edit-pg-schemas", Input).value = ", ".join(schemas)
                self._show_help("Cataloging selected schemas…")
                result = await self.app.controller.edit_source_connection(  # type: ignore[attr-defined]
                    self._source,
                    name=name,
                    host=host,
                    port=port,
                    database=database,
                    username=username,
                    password=password,
                    schemas=schemas,
                    ssl_mode=ssl_mode,
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
            password_input.clear()
        if result is None:
            if self._preview_error is None:
                self._show_error("Connection changes were not applied.")
            return
        self.dismiss(True)

    async def _confirm_preview(self, preview: Any) -> bool:
        if preview.resource_count == 0:
            self._preview_error = (
                "Connection succeeded, but the selected schemas contain no "
                "catalogable tables. Choose schemas marked as containing tables."
            )
            self._show_error(self._preview_error)
            return False
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

    def _show_help(self, text: str) -> None:
        self.query_one("#source-edit-help", Static).update(
            sanitize_terminal_text(
                text,
                maximum=512,
                preserve_lines=False,
                fallback="Editing source.",
            )
        )

    def _show_error(self, text: str) -> None:
        self.query_one("#source-edit-error", Label).update(text)
