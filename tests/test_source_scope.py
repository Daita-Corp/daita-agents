from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from _workspace_support import workspace_for

from daita import Agent, SQLiteSource
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelResponse,
    ModelSensitivity,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.llm.routing import ModelProviderRegistration, ModelRouter, RetryPolicy
from daita.loop.models import LoopExitKind, RunInput
from daita.scope import resolve_effective_source_scope


def _database(path: Path, table: str) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            f'CREATE TABLE "{table}" (id INTEGER PRIMARY KEY)'  # noqa: S608
        )


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=32_000,
        max_output_tokens=1_000,
        supports_tools=True,
    )


def _stop(text: str) -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text)


def _user_texts(request: object) -> tuple[str, ...]:
    messages = getattr(request, "messages")
    return tuple(
        block.text
        for message in messages
        if message.role is MessageRole.USER
        for block in message.content
        if isinstance(block, TextBlock)
    )


async def test_admitted_sources_persist_without_active_selection(
    tmp_path: Path,
):
    first_path = tmp_path / "first.sqlite"
    second_path = tmp_path / "second.sqlite"
    _database(first_path, "first_records")
    _database(second_path, "second_records")
    agent = await Agent.create(
        "source-persistence", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    first = await agent.attach(SQLiteSource(first_path, name="First Source"))
    second = await agent.attach(SQLiteSource(second_path, name="Second Source"))

    assert set(item.id for item in await agent.list_sources()) == {first.id, second.id}
    assert await agent.resolve_source("second-source") == second
    await agent.close()

    reopened = await Agent.open(
        "source-persistence", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        assert len(await reopened.list_sources()) == 2
        await reopened.detach(second.id)
        assert tuple(
            item.id for item in await reopened.list_sources() if item.active
        ) == (first.id,)
    finally:
        await reopened.close()


async def test_detached_source_can_be_attached_again(tmp_path: Path):
    database = tmp_path / "reattach.sqlite"
    _database(database, "records")
    agent = await Agent.create(
        "source-reattach", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        original = await agent.attach(SQLiteSource(database, name="Original"))
        detached = await agent.detach(original.id)

        reattached = await agent.attach(SQLiteSource(database, name="Reattached"))

        assert reattached.id == original.id
        assert reattached.active
        assert reattached.display_name == "Reattached"
        assert detached.detached_at is not None
        assert await agent.list_sources() == (reattached,)
        assert len(await agent.list_catalog_resources(source_id=reattached.id)) == 1
        with pytest.raises(ValueError, match="source registration already exists"):
            await agent.attach(SQLiteSource(database, name="Duplicate"))
    finally:
        await agent.close()


async def test_source_edit_preserves_selected_reads_and_switches_atomically(
    tmp_path: Path,
) -> None:
    current_path = tmp_path / "current.sqlite"
    edited_path = tmp_path / "edited.sqlite"
    _database(current_path, "records")
    _database(edited_path, "records")
    agent = await Agent.create(
        "source-edit", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    previews: list[object] = []
    try:
        current = await agent.attach(SQLiteSource(current_path, name="Warehouse"))
        current_resources = await agent.list_catalog_resources(source_id=current.id)
        permission_preview = await agent.preview_source_permissions(
            source_id=current.id,
            read_mode="selected",
            read_resource_ids=(current_resources[0].id,),
            postgresql_update_scopes={},
        )
        await agent.apply_source_permissions(
            source_id=current.id,
            confirmation_fingerprint=permission_preview.confirmation_fingerprint,
        )

        async def confirm(preview: object) -> bool:
            previews.append(preview)
            return True

        result = await agent.edit_source(
            current.id,
            SQLiteSource(edited_path, name="Warehouse"),
            confirmation_handler=confirm,
        )

        assert result is not None and result.identity_changed
        assert result.previous_credential_deleted
        assert len(previews) == 1
        preview = previews[0]
        assert getattr(preview, "preserved_read_resource_count") == 1
        assert getattr(preview, "omitted_read_resources") == ()
        sources = await agent.list_sources()
        assert len(sources) == 2
        assert (
            next(source for source in sources if source.id == current.id).active
            is False
        )
        inspection = await agent.inspect_source_permissions(result.source.id)
        assert inspection.state.read_scope.mode.value == "selected"
        assert len(inspection.state.read_scope.resource_ids) == 1
        assert inspection.state.postgresql_update_scopes == ()
    finally:
        await agent.close()


async def test_source_edit_rejection_leaves_current_source_untouched(
    tmp_path: Path,
) -> None:
    current_path = tmp_path / "current.sqlite"
    edited_path = tmp_path / "edited.sqlite"
    _database(current_path, "current_records")
    _database(edited_path, "edited_records")
    agent = await Agent.create(
        "source-edit-rejected", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        current = await agent.attach(SQLiteSource(current_path, name="Current"))

        async def reject(_preview: object) -> bool:
            return False

        result = await agent.edit_source(
            current.id,
            SQLiteSource(edited_path, name="Edited"),
            confirmation_handler=reject,
        )

        assert result is None
        assert await agent.list_sources() == (current,)
        resources = await agent.list_catalog_resources(source_id=current.id)
        assert (
            len(resources) == 1
            and resources[0].native_identity == "main.current_records"
        )
    finally:
        await agent.close()


async def test_one_run_filter_narrows_sources_and_preserves_classified_history(
    tmp_path: Path,
):
    first_path = tmp_path / "first.sqlite"
    second_path = tmp_path / "second.sqlite"
    _database(first_path, "first_records")
    _database(second_path, "second_records")
    provider = MockModelProvider(
        (
            _stop("first answer"),
            _stop("override answer"),
            _stop("follow-up answer"),
        )
    )
    agent = await Agent.create(
        "source-override",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        workspace=workspace_for(tmp_path),
    )
    try:
        first_source = await agent.attach(SQLiteSource(first_path, name="First Source"))
        second_source = await agent.attach(
            SQLiteSource(second_path, name="Second Source")
        )
        first = await agent.run("Question about first_records")
        override = await agent.run(
            "Question about second_records",
            conversation_id=first.conversation_id,
            source_scope_ids=(second_source.id,),
        )
        follow_up = await agent.run(
            "Follow up about first_records",
            conversation_id=first.conversation_id,
        )

        override_run = (await agent.transcript(override.run_id)).run
        assert override_run.source_scope_ids == (second_source.id,)
        assert override_run.resolved_source_scope is not None
        assert override_run.resolved_source_scope.source_ids == frozenset(
            {second_source.id}
        )
        assert _user_texts(provider.requests[0]) == ("Question about first_records",)
        assert _user_texts(provider.requests[1]) == (
            "Question about first_records",
            "Question about second_records",
        )
        assert _user_texts(provider.requests[2]) == (
            "Question about first_records",
            "Question about second_records",
            "Follow up about first_records",
        )
        first_system = repr(provider.requests[0].messages[0])
        override_system = repr(provider.requests[1].messages[0])
        follow_up_system = repr(provider.requests[2].messages[0])
        assert first_source.id in first_system
        assert second_source.id in first_system
        assert second_source.id in override_system
        assert first_source.id not in override_system
        assert first_source.id in follow_up_system
        assert second_source.id in follow_up_system
        assert follow_up.conversation_id == first.conversation_id
    finally:
        await agent.close()


async def test_runtime_binds_catalog_scope_without_injection_and_rejects_outside_query(
    tmp_path: Path,
):
    first_path = tmp_path / "first.sqlite"
    second_path = tmp_path / "second.sqlite"
    _database(first_path, "shared_first")
    _database(second_path, "shared_second")
    provider = MockModelProvider(())
    agent = await Agent.create(
        "source-enforcement",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        workspace=workspace_for(tmp_path),
    )
    first_source = await agent.attach(SQLiteSource(first_path, name="First Source"))
    second_source = await agent.attach(SQLiteSource(second_path, name="Second Source"))
    (second_resource,) = await agent.list_catalog_resources(source_id=second_source.id)
    provider._script = (
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    id="catalog",
                    name="catalog_search",
                    arguments={"query": "shared"},
                ),
            ),
        ),
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    id="cross-source",
                    name="data_query",
                    arguments={
                        "source_id": second_source.id,
                        "resource_ids": (second_resource.id,),
                        "sql": "SELECT id FROM shared_second",
                    },
                ),
            ),
        ),
        _stop("The cross-source query was blocked."),
    )
    try:
        result = await agent.run(
            "Find shared records", source_scope_ids=(first_source.id,)
        )
        transcript = await agent.transcript(result.run_id)
        assert dict(transcript.messages[1].tool_calls[0].arguments) == {
            "query": "shared"
        }
        catalog_result = transcript.messages[2].content[0]
        blocked_result = transcript.messages[4].content[0]
        assert isinstance(catalog_result, ToolResultBlock)
        assert isinstance(blocked_result, ToolResultBlock)
        catalog_data = catalog_result.output["data"]
        assert isinstance(catalog_data, Mapping)
        hits = catalog_data["hits"]
        assert isinstance(hits, tuple)
        assert hits
        assert all(
            isinstance(hit, Mapping) and hit["source_id"] == first_source.id
            for hit in hits
        )
        error = blocked_result.output["error"]
        assert isinstance(error, Mapping)
        assert error["code"] == "source_scope_violation"
    finally:
        await agent.close()


async def test_source_filter_still_projects_source_independent_file_tools(
    tmp_path: Path,
):
    first_database = tmp_path / "first.sqlite"
    second_database = tmp_path / "second.sqlite"
    _database(first_database, "first_records")
    _database(second_database, "second_records")
    provider = MockModelProvider((_stop("second source selected"),))
    agent = await Agent.create(
        "source-tool-projection",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.attach(SQLiteSource(first_database, name="First Source"))
        await agent.attach(SQLiteSource(second_database, name="Second Source"))

        second = await agent.resolve_source("second-source")
        await agent.run(
            "What data and workspace files are available?",
            source_scope_ids=(second.id,),
        )

        tool_names = {tool.name for tool in provider.requests[0].tools}
        assert {"file_search", "file_read", "data_query"} <= tool_names
    finally:
        await agent.close()


class _ScopeCatalog:
    def __init__(self):
        self.resources = {"first": {"first-row"}, "second": {"second-row"}}
        self.calls = 0

    async def source_routing_facts(self, agent_id, source_ids=()):
        self.calls += 1
        return tuple(
            {"source_id": item, "adapter_id": "sqlite"}
            for item in self.resources
            if not source_ids or item in source_ids
        )

    async def readable_resource_ids(self, agent_id, source_ids=()):
        self.calls += 1
        return frozenset(
            resource for source in source_ids for resource in self.resources[source]
        )


async def test_prepared_scope_rejects_late_attachment_and_new_resource_permission():
    catalog = _ScopeCatalog()
    run = RunInput("prepared", "agent", "Discover data", datetime.now(UTC))
    scope = await resolve_effective_source_scope(run, catalog)
    prepared = replace(run, resolved_source_scope=scope)
    catalog.resources["third"] = {"third-row"}
    catalog.resources["first"].add("new-row")

    assert await resolve_effective_source_scope(prepared, catalog) == scope
    catalog.resources.pop("second")
    narrowed = await resolve_effective_source_scope(prepared, catalog)
    assert narrowed.source_ids == frozenset({"first"})
    assert narrowed.resource_ids == frozenset({"first-row"})


async def test_prepared_empty_and_files_only_scope_never_expand():
    catalog = _ScopeCatalog()
    run = RunInput("empty", "agent", "Files only", datetime.now(UTC))
    denied = await resolve_effective_source_scope(run, catalog, files_only=True)
    assert not denied.source_ids and not denied.resource_ids
    assert catalog.calls == 0
    assert (
        await resolve_effective_source_scope(
            replace(run, resolved_source_scope=denied), catalog
        )
        == denied
    )
    assert catalog.calls == 0


async def test_private_continuity_survives_detach_compression_and_restart(tmp_path):
    database = tmp_path / "private.sqlite"
    _database(database, "internal_accounts")
    provider = MockModelProvider((_stop("Private account conclusion. " * 900),))
    agent = await Agent.create(
        "history-classification",
        root=tmp_path,
        hosted=True,
        model=provider,
        model_profile=_profile(provider),
    )
    source = await agent.attach(SQLiteSource(database))
    first = await agent.run("Explain the internal accounts")
    assert first.sensitivity is ModelSensitivity.INTERNAL
    await agent.detach(source.id)
    await agent.close()

    public = MockModelProvider((_stop("Must not receive private history"),))
    router = ModelRouter(
        (
            ModelProviderRegistration(
                provider=public,
                profile=public.model_profile,
                allowed_sensitivities=frozenset({ModelSensitivity.PUBLIC}),
            ),
        ),
        retry_policy=RetryPolicy(attempts=1, backoff_seconds=0),
    )
    reopened = await Agent.open(
        "history-classification",
        root=tmp_path,
        hosted=True,
        model=router,
        model_profile=router.model_profile,
    )
    try:
        result = await reopened.run(
            "Summarize that answer", conversation_id=first.conversation_id
        )
        assert result.kind is LoopExitKind.FAILED
        assert result.reason == "model_route_ineligible"
        assert result.sensitivity is ModelSensitivity.INTERNAL
        assert public.requests == ()
        assert (
            await reopened.transcript(result.run_id)
        ).run.history_sensitivity is ModelSensitivity.INTERNAL
    finally:
        await reopened.close()


async def test_one_foreground_run_compares_two_exact_sources_without_selection(
    tmp_path,
):
    provider = MockModelProvider(())
    agent = await Agent.create(
        "compare-sources",
        root=tmp_path,
        hosted=True,
        model=provider,
        model_profile=_profile(provider),
    )
    sources = []
    resources = []
    for label, count in (("billing", 2), ("application", 3)):
        path = tmp_path / (label + ".sqlite")
        _database(path, "customers")
        with sqlite3.connect(path) as connection:
            connection.executemany(
                "INSERT INTO customers(id) VALUES (?)",
                ((index,) for index in range(count)),
            )
        source = await agent.attach(SQLiteSource(path, name=label))
        sources.append(source)
        resources.append((await agent.list_catalog_resources(source_id=source.id))[0])
    provider._script = (
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    "discover",
                    "catalog_search",
                    {"query": "customer counts billing application"},
                ),
            ),
        ),
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=tuple(
                ToolCall(
                    f"count-{index}",
                    "data_query",
                    {
                        "source_id": source.id,
                        "resource_ids": (resource.id,),
                        "sql": "SELECT COUNT(*) AS customers FROM customers",
                    },
                )
                for index, (source, resource) in enumerate(
                    zip(sources, resources, strict=True)
                )
            ),
        ),
        _stop(
            "Billing has 2 customer rows; application has 3. Matching table names alone do not establish equivalent customer definitions."
        ),
    )
    try:
        result = await agent.run(
            "Compare customer counts in our billing warehouse and application database."
        )
        assert result.kind is LoopExitKind.COMPLETED
        transcript = await agent.transcript(result.run_id)
        assert transcript.run.source_scope_ids == ()
        results = [
            block
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        ]
        assert all(not block.is_error for block in results)
        assert len(results) == 3
        catalog_data = results[0].output["data"]
        assert isinstance(catalog_data, Mapping)
        hits = catalog_data["hits"]
        assert isinstance(hits, tuple)
        assert {hit["source_id"] for hit in hits if isinstance(hit, Mapping)} == {
            source.id for source in sources
        }
        for block, count in zip(results[1:], (2, 3), strict=True):
            data = block.output["data"]
            assert isinstance(data, Mapping)
            rows = data["rows"]
            assert isinstance(rows, tuple) and len(rows) == 1
            assert isinstance(rows[0], Mapping)
            assert rows[0]["customers"] == count
        assert result.sensitivity is ModelSensitivity.INTERNAL
    finally:
        await agent.close()
