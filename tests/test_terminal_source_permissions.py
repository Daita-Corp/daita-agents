from __future__ import annotations

import io
from collections.abc import Mapping
from datetime import UTC, datetime

from daita import terminal, terminal_tui
from daita.adapters.models import SourceRegistration
from daita.catalog.models import ResourceKind, catalog_resource_id
from daita.storage.sqlite_records import (
    PostgreSQLUpdateScope,
    SourcePermissionResource,
    SourcePermissionsInspection,
    SourcePermissionsPreview,
    SourcePermissionState,
    SourcePermissionSummary,
    SourceReadMode,
    SourceReadScope,
)

NOW = datetime(2026, 8, 12, tzinfo=UTC)
AGENT_ID = "agent-terminal-permissions"


class _PermissionAgent:
    def __init__(self) -> None:
        self.source = SourceRegistration.build(
            agent_id=AGENT_ID,
            adapter_id="postgresql",
            native_identity="postgresql:large-fixture-terminal-permissions",
            display_name="Large PostgreSQL write canary",
            configuration={},
            attached_at=NOW,
        )
        self.resources = (
            SourcePermissionResource(
                resource_id=catalog_resource_id(
                    self.source.id,
                    ResourceKind.TABLE,
                    "support.tickets",
                ),
                display_name="support.tickets",
                resource_kind="table",
                eligible_assignment_columns=("priority",),
            ),
            SourcePermissionResource(
                resource_id=catalog_resource_id(
                    self.source.id,
                    ResourceKind.TABLE,
                    "support.ticket_events",
                ),
                display_name="support.ticket_events",
                resource_kind="table",
            ),
        )
        self.state = SourcePermissionState(
            SourceReadScope.allow_all(
                agent_id=AGENT_ID,
                source_id=self.source.id,
            ),
            (),
        )
        self.preview_calls: list[dict[str, object]] = []
        self.apply_calls: list[dict[str, object]] = []

    async def list_sources(self):
        return (self.source,)

    async def inspect_source_permissions(self, source_id: str):
        assert source_id == self.source.id
        return SourcePermissionsInspection(
            source_id=self.source.id,
            source_display_name=self.source.display_name,
            adapter_id="postgresql",
            catalog_generation="catalog-terminal",
            state=self.state,
            resources=self.resources,
        )

    async def preview_source_permissions(self, **kwargs: object):
        self.preview_calls.append(dict(kwargs))
        raw_read_mode = kwargs["read_mode"]
        assert isinstance(raw_read_mode, (SourceReadMode, str))
        read_mode = SourceReadMode(raw_read_mode)
        raw_read_ids = kwargs["read_resource_ids"]
        assert isinstance(raw_read_ids, tuple)
        read_ids = raw_read_ids
        raw_updates = kwargs["postgresql_update_scopes"]
        assert isinstance(raw_updates, Mapping)
        updates = dict(raw_updates)
        read_scope = SourceReadScope(
            agent_id=AGENT_ID,
            source_id=self.source.id,
            mode=read_mode,
            resource_ids=read_ids,
        )
        update_scopes = tuple(
            PostgreSQLUpdateScope(
                agent_id=AGENT_ID,
                source_id=self.source.id,
                resource_id=resource_id,
                allowed_assignment_columns=tuple(columns),
                authorization_fingerprint="sha256:" + "a" * 64,
            )
            for resource_id, columns in updates.items()
        )
        after = SourcePermissionState(read_scope, update_scopes)
        names = {
            resource.resource_id: resource.display_name for resource in self.resources
        }
        return SourcePermissionsPreview(
            source_id=self.source.id,
            catalog_generation="catalog-terminal",
            before=self.state,
            after=after,
            automatic_read_additions=(),
            dependent_update_revocations=(),
            summary=SourcePermissionSummary(
                source_display_name=self.source.display_name,
                read_mode=read_scope.mode,
                selected_read_resource_count=(
                    len(self.resources)
                    if read_scope.mode is SourceReadMode.ALL
                    else len(read_scope.resource_ids)
                ),
                postgresql_update_table_count=len(update_scopes),
                postgresql_update_table_examples=tuple(
                    names[scope.resource_id] for scope in update_scopes
                ),
                automatic_read_addition_examples=(),
                dependent_update_revocation_examples=(),
            ),
            confirmation_fingerprint="sha256:" + "b" * 64,
        )

    async def apply_source_permissions(self, **kwargs: object):
        self.apply_calls.append(dict(kwargs))
        return await self.inspect_source_permissions(self.source.id)


async def test_terminal_read_permissions_select_many_summarize_and_apply_once():
    agent = _PermissionAgent()
    output = io.StringIO()

    await terminal._configure_source_permissions(
        agent,  # type: ignore[arg-type]
        input_stream=io.StringIO("1\n1\n2\n1-2\ny\n"),
        output_stream=output,
    )

    assert len(agent.preview_calls) == 1
    assert agent.preview_calls[0]["read_mode"] == "selected"
    raw_read_ids = agent.preview_calls[0]["read_resource_ids"]
    assert isinstance(raw_read_ids, tuple)
    assert set(raw_read_ids) == {resource.resource_id for resource in agent.resources}
    assert len(agent.apply_calls) == 1
    rendered = output.getvalue()
    assert "Review source permissions" in rendered
    assert "Future PostgreSQL tables are not automatically write-enabled" in rendered
    assert "No external database mutation was executed" in rendered


async def test_terminal_write_permissions_use_all_columns_and_cancel_changes_nothing():
    agent = _PermissionAgent()
    output = io.StringIO()

    await terminal._configure_source_permissions(
        agent,  # type: ignore[arg-type]
        input_stream=io.StringIO("1\n2\n2\nall\n1\nn\n"),
        output_stream=output,
    )

    assert len(agent.preview_calls) == 1
    assert agent.preview_calls[0]["postgresql_update_scopes"] == {
        agent.resources[0].resource_id: ("priority",)
    }
    assert agent.apply_calls == []
    assert "Source permissions were not changed" in output.getvalue()


async def test_advanced_write_permissions_can_select_more_than_32_columns():
    agent = _PermissionAgent()
    columns = tuple(f"column_{index}" for index in range(33))
    wide_resource = SourcePermissionResource(
        resource_id=agent.resources[0].resource_id,
        display_name=agent.resources[0].display_name,
        resource_kind="table",
        eligible_assignment_columns=columns,
    )
    agent.resources = (wide_resource, agent.resources[1])
    inspection = await agent.inspect_source_permissions(agent.source.id)

    selected = await terminal._advanced_update_columns(
        (wide_resource,),
        inspection,
        input_stream=io.StringIO("all\n"),
        output_stream=io.StringIO(),
        selection_input=None,
        selection_output=None,
    )

    assert len(selected[wide_resource.resource_id]) == 33
    assert set(selected[wide_resource.resource_id]) == set(columns)


async def test_legacy_source_command_is_unknown_and_routes_to_current_usage():
    agent = _PermissionAgent()
    output = io.StringIO()

    await terminal._handle_local_command(
        "/source " + "config",
        agent=agent,  # type: ignore[arg-type]
        root=None,
        input_stream=io.StringIO(),
        output_stream=output,
        hidden_input=lambda _prompt: "",
        keychain=None,
        model_validator=None,
        approval_handler=None,
        conversation_id=None,
        validated=True,
    )

    assert "/source permissions" in output.getvalue()
    assert agent.preview_calls == []
    assert agent.apply_calls == []


def test_permission_command_uses_external_prompt_path_and_slash_completion():
    assert terminal._command_uses_terminal_prompts("/source permissions") is True
    assert "/source permissions" in {
        display
        for _insertion, display, _description in terminal_tui._SLASH_COMMAND_COMPLETIONS
    }
