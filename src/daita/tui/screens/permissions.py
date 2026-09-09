"""Author exact read, update and upsert permissions through preview/apply."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, Label, Static

from daita.capabilities import render_approval_arguments

from ..models import PickerOption
from ..sanitization import sanitize_terminal_text
from .selection import SelectionScreen


class PermissionsScreen(Screen[bool]):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, *, source_id: str | None = None) -> None:
        super().__init__()
        self._source_id = source_id
        self._preview: Any = None
        self._reviewable = False

    def compose(self) -> ComposeResult:
        with Vertical(id="permissions", classes="control-panel"):
            yield Label("Source permissions", id="onboard-title", markup=False)
            yield Static(
                "Choose exact read, update or upsert scopes. Permissions do not execute a write.",
                id="perm-help",
                markup=False,
            )
            with VerticalScroll(id="perm-preview"):
                yield Static("", id="perm-body", markup=False)
            yield Button("Choose source", id="perm-source")
            with Horizontal(classes="permission-actions"):
                yield Button("Read: all resources", id="perm-read-all")
                yield Button("Read: selected resources", id="perm-read-selected")
                yield Button("Read: none", id="perm-read-none")
            yield Label(
                "Maximum rows per call (update ≤10,000; upsert ≤1,000)", markup=False
            )
            yield Input(
                value="100",
                placeholder="Maximum rows per call",
                id="perm-max-rows",
                type="integer",
            )
            with Horizontal(classes="permission-actions"):
                yield Button("Edit table write access", id="perm-write")
                yield Button("Apply", id="perm-apply", variant="primary")
            yield Footer()

    def action_cancel(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id is None:
            return
        self.run_worker(
            self._handle_safely(button_id),
            name="permissions-action",
            group="permissions-interaction",
            exclusive=True,
        )

    async def _handle_safely(self, button_id: str) -> None:
        try:
            await self._handle_button(button_id)
        except (ValueError, RuntimeError, PermissionError, OSError) as error:
            self._preview = None
            self._reviewable = False
            self.query_one("#perm-help", Static).update(
                sanitize_terminal_text(
                    str(error),
                    maximum=2048,
                    preserve_lines=True,
                    fallback="Permission change failed. Preview again.",
                )
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
        elif button_id == "perm-write":
            await self._preview_write(controller)
        elif button_id == "perm-apply":
            if self._preview is None or not self._reviewable:
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

    async def _preview_write(self, controller: Any) -> None:
        inspection = await controller.inspect_source_permissions(self._source_id)
        if inspection.adapter_id != "postgresql":
            raise ValueError(
                "Native write access is available only for PostgreSQL sources."
            )
        state = self._proposal_state(inspection)
        scopes = {
            scope.resource_id: {
                key: value
                for key, value in scope.constraints().items()
                if key != "resource_revision"
            }
            for scope in state.relational_write_scopes
        }
        resources = tuple(
            resource
            for resource in inspection.resources
            if resource.resource_kind == "table"
        )
        chosen = await self.app._await_modal(  # type: ignore[attr-defined]
            SelectionScreen(
                title="Choose an exact table",
                options=tuple(
                    PickerOption(resource.resource_id, resource.display_name)
                    for resource in resources
                ),
            )
        )
        if chosen is None:
            return
        resource = next(item for item in resources if item.resource_id == chosen[0])
        options = [
            PickerOption(
                "none",
                "No write access",
                "Remove this table's update and upsert permission",
            )
        ]
        if resource.relational_update_eligible:
            options.append(
                PickerOption(
                    "update", "Update existing rows", "Insertion remains forbidden"
                )
            )
        if resource.upsert_conflict_keys:
            options.append(
                PickerOption(
                    "upsert",
                    "Upsert rows",
                    "Insert missing keys and update existing rows",
                )
            )
            if (
                resource.key_columns in resource.upsert_conflict_keys
                and resource.relational_update_eligible
            ):
                options.append(
                    PickerOption(
                        "both", "Update and upsert", "Explicitly permit both operations"
                    )
                )
        operation = await self.app._await_modal(  # type: ignore[attr-defined]
            SelectionScreen(
                title=f"Write operations: {resource.display_name}",
                options=tuple(options),
            )
        )
        if operation is None:
            return
        if operation[0] == "none":
            scopes.pop(resource.resource_id, None)
        else:
            operations = ("update", "upsert") if operation[0] == "both" else operation
            keys = resource.key_columns
            inserts: tuple[str, ...] = ()
            identities: tuple[str, ...] = ()
            columns = resource.eligible_assignment_columns
            if "upsert" in operations:
                if "update" not in operations:
                    selected_key = await self.app._await_modal(  # type: ignore[attr-defined]
                        SelectionScreen(
                            title="Choose a supported unique conflict key",
                            options=tuple(
                                PickerOption(str(index), ", ".join(key))
                                for index, key in enumerate(
                                    resource.upsert_conflict_keys
                                )
                            ),
                        )
                    )
                    if selected_key is None:
                        return
                    keys = resource.upsert_conflict_keys[int(selected_key[0])]
                selected_inserts = await self.app._await_modal(  # type: ignore[attr-defined]
                    SelectionScreen(
                        title="Insert columns (include every conflict key)",
                        multi=True,
                        options=tuple(
                            PickerOption(column, column)
                            for column in resource.eligible_insert_columns
                        ),
                    )
                )
                if selected_inserts is None:
                    return
                inserts = selected_inserts
                columns = tuple(
                    column
                    for column in resource.eligible_upsert_update_columns
                    if column in inserts and column not in keys
                )
                if resource.generated_identity_columns:
                    identity_choice = await self.app._await_modal(  # type: ignore[attr-defined]
                        SelectionScreen(
                            title="Permit identity generation for missing rows?",
                            options=(
                                PickerOption("cancel", "Cancel upsert authoring"),
                                PickerOption(
                                    "allow",
                                    "Allow database-generated identities",
                                    ", ".join(resource.generated_identity_columns),
                                ),
                            ),
                        )
                    )
                    if identity_choice != ("allow",):
                        return
                    identities = resource.generated_identity_columns
            selected_columns = await self.app._await_modal(  # type: ignore[attr-defined]
                SelectionScreen(
                    title=f"Update columns: {resource.display_name}",
                    multi=True,
                    options=tuple(PickerOption(column, column) for column in columns),
                )
            )
            if selected_columns is None:
                return
            max_rows = int(self.query_one("#perm-max-rows", Input).value)
            scopes[resource.resource_id] = {
                "allowed_operations": operations,
                "allowed_insert_columns": inserts,
                "allowed_update_columns": selected_columns,
                "key_columns": keys,
                "generated_identity_columns": identities,
                "max_rows": max_rows,
            }
        await self._preview_permissions(
            controller,
            inspection,
            read_mode=state.read_scope.mode.value,
            read_resource_ids=state.read_scope.resource_ids,
            updates=scopes,
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
            scope.resource_id: {
                key: value
                for key, value in scope.constraints().items()
                if key != "resource_revision"
            }
            for scope in proposal.relational_write_scopes
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
        updates: dict[str, dict[str, object]],
    ) -> None:
        self._preview = await controller.preview_source_permissions(
            source_id=self._source_id,
            read_mode=read_mode,
            read_resource_ids=read_resource_ids,
            relational_write_scopes=updates,
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
            inspection.state.relational_write_scopes,
            inspection,
        )
        return sanitize_terminal_text(
            f"{inspection.source_display_name}\n"
            f"Read mode: {inspection.state.read_scope.mode.value}\n"
            f"Resources: {len(inspection.resources)}\n"
            f"Relational write tables: "
            f"{len(inspection.state.relational_write_scopes)}"
            f"{update_lines}",
            maximum=32768,
            preserve_lines=True,
            fallback="permissions",
        )

    def _preview_text(self, preview: Any, inspection: Any) -> str:
        names = {
            resource.resource_id: resource.display_name
            for resource in inspection.resources
        }

        def state_document(state: Any) -> dict[str, object]:
            return {
                "read_mode": state.read_scope.mode.value,
                "read_resource_ids": state.read_scope.resource_ids,
                "write_scopes": tuple(
                    {
                        "resource_id": scope.resource_id,
                        "table": names.get(scope.resource_id, scope.resource_id),
                        **scope.constraints(),
                    }
                    for scope in state.relational_write_scopes
                ),
            }

        document = render_approval_arguments(
            {
                "before": state_document(preview.before),
                "after": state_document(preview.after),
                "confirmation_fingerprint": preview.confirmation_fingerprint,
            }
        )
        self._reviewable = document is not None
        return (
            "Before → after. Apply authorizes exactly these permissions; it executes no write. "
            "Review read additions as well as operations, keys, columns, identities and row ceilings.\n\n"
            + document
            if document is not None
            else "Exact permission details exceed the review bound. Apply is unavailable; choose a smaller scope."
        )

    def _update_scope_lines(self, scopes: Any, inspection: Any) -> str:
        names = {
            resource.resource_id: resource.display_name
            for resource in inspection.resources
        }
        return "".join(
            f"\n  {names.get(scope.resource_id, scope.resource_id)}: {', '.join(scope.allowed_operations)}; "
            f"keys: {', '.join(scope.key_columns)}; update: {', '.join(scope.allowed_update_columns)}; "
            f"insert: {', '.join(scope.allowed_insert_columns)}; identities: {', '.join(scope.generated_identity_columns)}; "
            f"max rows: {scope.max_rows}"
            for scope in scopes
        )
