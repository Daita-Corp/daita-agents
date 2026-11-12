"""Choose what the agent may read and change in a source, through preview/apply."""

from __future__ import annotations

from typing import Any, cast

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Label, Static

from daita.capabilities import render_approval_arguments

from ..models import PickerOption
from ..sanitization import sanitize_terminal_text
from .selection import SelectionScreen

# shortcut: one fixed ceiling per change instead of a typed limit; every change is
# still previewed with its exact row count. Make it settable if 1,000 proves tight.
_DEFAULT_MAX_ROWS = 1_000


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
                "Choose what the agent may read and change. The source login still "
                "limits what is possible, and every change is previewed for your "
                "approval. Permissions do not execute a write.",
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
            with Horizontal(classes="permission-actions"):
                yield Button("Edit table write access", id="perm-write")
                yield Button("Apply", id="perm-apply", variant="primary")
            yield Footer()

    def on_mount(self) -> None:
        if self._source_id is not None:
            self.run_worker(
                self._load_selected_source(),
                name="permissions-initial-source",
                group="permissions-interaction",
                exclusive=True,
            )

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
        await self._render_selected_source(controller)

    async def _load_selected_source(self) -> None:
        try:
            await self._render_selected_source(self.app.controller)  # type: ignore[attr-defined]
        except (ValueError, RuntimeError, PermissionError, OSError) as error:
            self.query_one("#perm-help", Static).update(
                sanitize_terminal_text(
                    str(error),
                    maximum=2048,
                    preserve_lines=True,
                    fallback="Could not load source permissions.",
                )
            )

    async def _render_selected_source(self, controller: Any) -> None:
        if self._source_id is None:
            return
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
        if not inspection.supports_relational_writes:
            raise ValueError(
                "Native write access needs a source type that supports writes."
            )
        state = self._proposal_state(inspection)
        scopes = _scope_constraints(state)
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
        stored = scopes.get(resource.display_name)
        operations = await self.app._await_modal(  # type: ignore[attr-defined]
            SelectionScreen(
                title=f"Changes the agent may propose (none removes access): "
                f"{resource.display_name}",
                multi=True,
                allow_empty=True,
                initial_selected=(
                    ()
                    if stored is None
                    else tuple(cast(list[str], stored["allowed_operations"]))
                ),
                options=(
                    PickerOption(
                        "update", "Update rows", "Any column the login may update"
                    ),
                    PickerOption(
                        "insert", "Insert rows", "Any column the login may insert"
                    ),
                    PickerOption(
                        "delete", "Delete rows", "Never granted unless chosen here"
                    ),
                ),
            )
        )
        if operations is None:
            return
        if not operations:
            scopes.pop(resource.display_name, None)
        else:
            # Columns come from the login's grants when the host previews this.
            scopes[resource.display_name] = {
                "allowed_operations": operations,
                "max_rows": max(
                    _DEFAULT_MAX_ROWS,
                    0 if stored is None else cast(int, stored["max_rows"]),
                ),
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
        updates = _scope_constraints(proposal)
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

        def state_document(state: Any) -> dict[str, object]:
            return {
                "read_mode": state.read_scope.mode.value,
                "read_resource_ids": state.read_scope.resource_ids,
                "write_scopes": tuple(
                    scope.constraints() for scope in state.relational_write_scopes
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
            "Review read additions as well as tables, operations, columns and row ceilings.\n\n"
            + document
            if document is not None
            else "Exact permission details exceed the review bound. Apply is unavailable; choose a smaller scope."
        )

    def _update_scope_lines(self, scopes: Any, inspection: Any) -> str:
        return "".join(
            f"\n  {scope.table}: {', '.join(scope.allowed_operations)}; "
            f"update: {', '.join(scope.allowed_update_columns)}; "
            f"insert: {', '.join(scope.allowed_insert_columns)}; "
            f"max rows: {scope.max_rows}"
            for scope in scopes
        )


def _scope_constraints(state: Any) -> dict[str, dict[str, object]]:
    """The stored write scopes in the shape a permission preview takes."""

    return {
        scope.table: {
            key: value for key, value in scope.constraints().items() if key != "table"
        }
        for scope in state.relational_write_scopes
    }
