"""Read and PostgreSQL update scope review."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Label, Static

from ..models import PickerOption
from ..sanitization import sanitize_terminal_text
from .selection import SelectionScreen


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
            yield Button("PostgreSQL update access", id="perm-update")
            yield Button("Apply", id="perm-apply", variant="primary")
            yield Footer()

    def action_cancel(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id is None:
            return
        self.run_worker(
            self._handle_button(button_id),
            name="permissions-action",
            group="permissions-interaction",
            exclusive=True,
        )

    async def _handle_button(self, button_id: str) -> None:
        app = self.app
        controller = app.controller  # type: ignore[attr-defined]
        if button_id == "perm-source":
            await self._pick_source(controller)
            return
        if self._source_id is None:
            self.query_one("#perm-help", Static).update("Choose a source first.")
            return
        if button_id == "perm-read-all":
            await self._preview_mode(controller, "all", ())
        elif button_id == "perm-read-selected":
            await self._preview_selected(controller)
        elif button_id == "perm-read-none":
            await self._preview_mode(controller, "none", ())
        elif button_id == "perm-update":
            await self._preview_updates(controller)
        elif button_id == "perm-apply":
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
        self._preview = None
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

    async def _preview_updates(self, controller: Any) -> None:
        inspection = await controller.inspect_source_permissions(self._source_id)
        if inspection.adapter_id != "postgresql":
            self.query_one("#perm-help", Static).update(
                "Update access is available only for PostgreSQL sources."
            )
            return
        eligible = tuple(
            resource
            for resource in inspection.resources
            if resource.postgresql_update_eligible
        )
        access_options = [
            PickerOption("none", "No update access", "Remove all update scopes")
        ]
        if eligible:
            access_options.extend(
                (
                    PickerOption(
                        "selected",
                        "Selected current tables",
                        "Choose exact tables",
                    ),
                    PickerOption(
                        "all",
                        "All current eligible tables",
                        "Future tables remain excluded",
                    ),
                )
            )
        access = await self.app._await_modal(  # type: ignore[attr-defined]
            SelectionScreen(
                title="PostgreSQL update access",
                options=tuple(access_options),
            )
        )
        if access is None:
            return
        if access[0] == "none":
            await self._preview_update_mapping(controller, inspection, {})
            return

        selected_resources = eligible
        if access[0] == "selected":
            selected = await self.app._await_modal(  # type: ignore[attr-defined]
                SelectionScreen(
                    title="Select update tables",
                    options=tuple(
                        PickerOption(
                            resource.resource_id,
                            resource.display_name,
                            f"{len(resource.eligible_assignment_columns)} eligible columns",
                        )
                        for resource in eligible
                    ),
                    multi=True,
                )
            )
            if selected is None:
                return
            selected_ids = set(selected)
            selected_resources = tuple(
                resource
                for resource in eligible
                if resource.resource_id in selected_ids
            )

        advanced_required = any(
            resource.requires_advanced_column_selection
            for resource in selected_resources
        )
        column_mode = "advanced"
        if not advanced_required:
            selected_mode = await self.app._await_modal(  # type: ignore[attr-defined]
                SelectionScreen(
                    title="Choose assignment columns",
                    options=(
                        PickerOption(
                            "all",
                            "All eligible columns",
                            "Broadest scope for the selected tables",
                        ),
                        PickerOption(
                            "advanced",
                            "Advanced column selection",
                            "Choose an exact subset for each table",
                        ),
                    ),
                )
            )
            if selected_mode is None:
                return
            column_mode = selected_mode[0]

        updates: dict[str, tuple[str, ...]] = {}
        for resource in selected_resources:
            columns = resource.eligible_assignment_columns
            if column_mode == "advanced":
                selected_columns = await self.app._await_modal(  # type: ignore[attr-defined]
                    SelectionScreen(
                        title=f"Select update columns: {resource.display_name}",
                        options=tuple(
                            PickerOption(column, column) for column in columns
                        ),
                        multi=True,
                    )
                )
                if selected_columns is None:
                    return
                columns = selected_columns
            updates[resource.resource_id] = columns
        await self._preview_update_mapping(controller, inspection, updates)

    async def _preview_update_mapping(
        self,
        controller: Any,
        inspection: Any,
        updates: dict[str, tuple[str, ...]],
    ) -> None:
        proposal = self._proposal_state(inspection)
        await self._preview_permissions(
            controller,
            inspection,
            read_mode=proposal.read_scope.mode.value,
            read_resource_ids=proposal.read_scope.resource_ids,
            updates=updates,
        )

    async def _preview_mode(
        self,
        controller: Any,
        read_mode: str,
        resource_ids: tuple[str, ...],
    ) -> None:
        inspection = await controller.inspect_source_permissions(self._source_id)
        proposal = self._proposal_state(inspection)
        updates = {
            scope.resource_id: scope.allowed_assignment_columns
            for scope in proposal.postgresql_update_scopes
        }
        await self._preview_permissions(
            controller,
            inspection,
            read_mode=read_mode,
            read_resource_ids=resource_ids,
            updates=updates,
        )

    async def _preview_permissions(
        self,
        controller: Any,
        inspection: Any,
        *,
        read_mode: str,
        read_resource_ids: tuple[str, ...],
        updates: dict[str, tuple[str, ...]],
    ) -> None:
        self._preview = await controller.preview_source_permissions(
            source_id=self._source_id,
            read_mode=read_mode,
            read_resource_ids=read_resource_ids,
            postgresql_update_scopes=updates,
        )
        self.query_one("#perm-body", Static).update(
            self._preview_text(self._preview, inspection)
        )

    def _proposal_state(self, inspection: Any) -> Any:
        if (
            self._preview is not None
            and self._preview.source_id == inspection.source_id
            and self._preview.catalog_generation == inspection.catalog_generation
        ):
            return self._preview.after
        return inspection.state

    def _inspection_text(self, inspection: Any) -> str:
        update_lines = self._update_scope_lines(
            inspection.state.postgresql_update_scopes,
            inspection,
        )
        return sanitize_terminal_text(
            f"{inspection.source_display_name}\n"
            f"Read mode: {inspection.state.read_scope.mode.value}\n"
            f"Resources: {len(inspection.resources)}\n"
            f"PostgreSQL update tables: "
            f"{len(inspection.state.postgresql_update_scopes)}"
            f"{update_lines}",
            maximum=2_048,
            preserve_lines=True,
            fallback="permissions",
        )

    def _preview_text(self, preview: Any, inspection: Any) -> str:
        update_lines = self._update_scope_lines(
            preview.after.postgresql_update_scopes,
            inspection,
        )
        return sanitize_terminal_text(
            "Before → after\n"
            f"Read: {preview.before.read_scope.mode.value} → {preview.after.read_scope.mode.value}\n"
            "PostgreSQL update tables: "
            f"{len(preview.before.postgresql_update_scopes)} → "
            f"{len(preview.after.postgresql_update_scopes)}"
            f"{update_lines}\n"
            f"Fingerprint: {preview.confirmation_fingerprint}",
            maximum=2_048,
            preserve_lines=True,
            fallback="preview",
        )

    def _update_scope_lines(self, scopes: tuple[Any, ...], inspection: Any) -> str:
        names = {
            resource.resource_id: resource.display_name
            for resource in inspection.resources
        }
        if not scopes:
            return ""
        lines = [
            "\n  "
            + names.get(scope.resource_id, scope.resource_id)
            + ": "
            + ", ".join(scope.allowed_assignment_columns)
            for scope in scopes[:5]
        ]
        if len(scopes) > 5:
            lines.append(f"\n  +{len(scopes) - 5} more tables")
        return "".join(lines)
