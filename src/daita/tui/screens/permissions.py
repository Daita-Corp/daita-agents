"""Read and PostgreSQL update scope review."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Label, Static

from ..models import PickerOption
from ..sanitization import sanitize_terminal_text
from .selection import SelectionScreen

if TYPE_CHECKING:
    from ..app import DaitaApp


class PermissionsScreen(Screen[bool]):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, *, source_id: str | None = None) -> None:
        super().__init__()
        self._source_id = source_id
        self._preview: Any = None

    def compose(self) -> ComposeResult:
        with Vertical(id="permissions"):
            yield Label("Source permissions", id="onboard-title", markup=False)
            yield Static(
                "Review exact read and update scopes.", id="perm-help", markup=False
            )
            with VerticalScroll(id="perm-preview"):
                yield Static("", id="perm-body", markup=False)
            yield Button("Choose source", id="perm-source")
            yield Button("Read: all resources", id="perm-read-all")
            yield Button("Read: selected resources", id="perm-read-selected")
            yield Button("Read: none", id="perm-read-none")
            yield Button("Apply", id="perm-apply", variant="primary")
            yield Footer()

    def action_cancel(self) -> None:
        self.dismiss(False)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        app = self.app
        controller = app.controller  # type: ignore[attr-defined]
        if event.button.id == "perm-source":
            await self._pick_source(controller)
            return
        if self._source_id is None:
            self.query_one("#perm-help", Static).update("Choose a source first.")
            return
        if event.button.id == "perm-read-all":
            await self._preview_mode(controller, "all", ())
        elif event.button.id == "perm-read-selected":
            await self._preview_selected(controller)
        elif event.button.id == "perm-read-none":
            await self._preview_mode(controller, "none", ())
        elif event.button.id == "perm-apply":
            if self._preview is None:
                self.query_one("#perm-help", Static).update("Preview a change first.")
                return
            await controller.apply_source_permissions(
                source_id=self._source_id,
                confirmation_fingerprint=self._preview.confirmation_fingerprint,
            )
            self.dismiss(True)

    async def _pick_source(self, controller: Any) -> None:
        sources = await controller.list_sources()
        options = tuple(
            PickerOption(source.id, source.display_name, source.adapter_id)
            for source in sources
            if source.active
        )
        selected = await self.app._await_modal(  # type: ignore[attr-defined]
            SelectionScreen(title="Choose a source", options=options)
        )
        if selected is None:
            return
        self._source_id = selected[0]
        inspection = await controller.inspect_source_permissions(self._source_id)
        self.query_one("#perm-body", Static).update(self._inspection_text(inspection))

    async def _preview_selected(self, controller: Any) -> None:
        inspection = await controller.inspect_source_permissions(self._source_id)
        options = tuple(
            PickerOption(resource.resource_id, resource.display_name)
            for resource in inspection.resources
        )
        selected = await self.app._await_modal(  # type: ignore[attr-defined]
            SelectionScreen(
                title="Select readable resources",
                options=options,
                multi=True,
            )
        )
        if selected is None:
            return
        await self._preview_mode(controller, "selected", selected)

    async def _preview_mode(
        self,
        controller: Any,
        read_mode: str,
        resource_ids: tuple[str, ...],
    ) -> None:
        inspection = await controller.inspect_source_permissions(self._source_id)
        updates = {
            scope.resource_id: scope.allowed_assignment_columns
            for scope in inspection.state.postgresql_update_scopes
        }
        self._preview = await controller.preview_source_permissions(
            source_id=self._source_id,
            read_mode=read_mode,
            read_resource_ids=resource_ids,
            postgresql_update_scopes=updates,
        )
        self.query_one("#perm-body", Static).update(self._preview_text(self._preview))

    def _inspection_text(self, inspection: Any) -> str:
        return sanitize_terminal_text(
            f"{inspection.source_display_name}\n"
            f"Read mode: {inspection.state.read_scope.mode.value}\n"
            f"Resources: {len(inspection.resources)}",
            maximum=2_048,
            preserve_lines=True,
            fallback="permissions",
        )

    def _preview_text(self, preview: Any) -> str:
        return sanitize_terminal_text(
            "Before → after\n"
            f"Read: {preview.before.read_scope.mode.value} → {preview.after.read_scope.mode.value}\n"
            f"Fingerprint: {preview.confirmation_fingerprint}",
            maximum=2_048,
            preserve_lines=True,
            fallback="preview",
        )
