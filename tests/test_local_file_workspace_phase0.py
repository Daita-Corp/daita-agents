"""Phase 2 vertical locks for source-free local workspace reads."""

from __future__ import annotations

import inspect
import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path

import pytest

import daita
from daita import Agent, LocalWorkspace, SQLiteSource
from daita.adapters.models import SourceRegistration
from daita.hosting.embedded import EmbeddedAgent
from daita.domains.data import LOCAL_FILE_CAPABILITY_IDS, LOCAL_FILE_EXECUTOR_IDS
from daita.domains.data.export_capabilities import (
    ARTIFACT_SAVE_LOCAL_EXECUTOR_ID,
    ARTIFACT_SET_EXPORT_LOCATION_EXECUTOR_ID,
)
from daita.adapters.local_workspace import LocalWorkspaceBackend
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelResponse,
    ModelSensitivity,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=32_000,
        max_output_tokens=2_000,
        supports_tools=True,
        supports_parallel_tools=True,
    )


def _stop(text: str = "done") -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text)


def _call(call_id: str, name: str, arguments: dict[str, object]) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
    )


def _workspace(
    tmp_path: Path,
    *,
    sensitivity: ModelSensitivity = ModelSensitivity.INTERNAL,
) -> LocalWorkspace:
    root = tmp_path.parent / f"{tmp_path.name}-workspace"
    root.mkdir()
    return LocalWorkspace(root, sensitivity=sensitivity)


def _database(path: Path, table: str) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(f'CREATE TABLE "{table}" (id INTEGER PRIMARY KEY)')


def test_local_workspace_is_required_and_removed_source_surface_is_absent() -> None:
    create = inspect.signature(Agent.create)
    opened = inspect.signature(Agent.open)
    assert create.parameters["workspace"].default is inspect.Parameter.empty
    assert opened.parameters["workspace"].default is inspect.Parameter.empty
    assert not hasattr(daita, "LocalDirectorySource")
    assert not hasattr(Agent, "attach_local_directory")
    assert not hasattr(Agent, "edit_local_directory_source")


async def test_source_free_user_run_projects_and_executes_workspace_file_tools(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path, sensitivity=ModelSensitivity.CONFIDENTIAL)
    (workspace.root / "notes.txt").write_text("workspace sentinel", encoding="utf-8")
    provider = MockModelProvider(
        (
            _call(
                "search",
                "file_search",
                {"query": "notes", "mode": "paths"},
            ),
            _call("read", "file_read", {"path": "notes.txt"}),
            _stop("found it"),
        )
    )
    agent = await Agent.create(
        "workspace-read",
        workspace=workspace,
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
    )
    try:
        result = await agent.run("Find and read my notes")
        assert result.final_text == "found it"
        assert {tool.name for tool in provider.requests[0].tools} >= {
            "file_search",
            "file_read",
        }
        assert all(
            request.sensitivity is ModelSensitivity.CONFIDENTIAL
            for request in provider.requests
        )
        assert str(workspace.root) not in repr(provider.requests)
    finally:
        await agent.close()


async def test_files_only_omits_all_source_tools_with_multiple_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    first_database = tmp_path / "one.sqlite"
    second_database = tmp_path / "two.sqlite"
    _database(first_database, "one")
    _database(second_database, "two")
    provider = MockModelProvider((_stop(),))
    agent = await Agent.create(
        "files-only",
        workspace=workspace,
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
    )
    try:
        await agent.attach(SQLiteSource(first_database))
        await agent.attach(SQLiteSource(second_database))

        async def unexpected_source_read(*_args, **_kwargs):
            raise AssertionError("files-only projection must not read source state")

        for method_name in (
            "source_routing_facts",
            "postgresql_update_applicable_source_ids",
            "admitted_model_sensitivity",
            "catalog_context",
            "semantic_resource_facts",
        ):
            monkeypatch.setattr(
                agent._embedded._data_view,
                method_name,
                unexpected_source_read,
            )
        await agent.run("Use files only", files_only=True)
        names = {tool.name for tool in provider.requests[0].tools}
        assert {"file_search", "file_read"} <= names
        assert not any(
            name.startswith(("catalog_", "data_query_", "data_profile", "mcp_"))
            for name in names
        )
        prompt = "\n".join(
            block.text
            for message in provider.requests[0].messages
            for block in message.content
            if hasattr(block, "text")
        )
        assert '"sources":[]' in prompt
    finally:
        await agent.close()


async def test_file_read_mixed_continuation_shape_fails_before_workspace_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    provider = MockModelProvider(
        (
            _call(
                "invalid-continuation",
                "file_read",
                {"cursor": "opaque-cursor", "position": "start"},
            ),
            _stop(),
        )
    )
    agent = await Agent.create(
        "file-read-shape",
        workspace=workspace,
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
    )

    async def unexpected_workspace_read(*_args, **_kwargs):
        raise AssertionError("invalid file_read arguments reached workspace I/O")

    assert agent._embedded._workspace_backend is not None
    monkeypatch.setattr(
        agent._embedded._workspace_backend,
        "read",
        unexpected_workspace_read,
    )
    try:
        result = await agent.run("Continue reading the file")
        transcript = await agent.transcript(result.run_id)
        rejected = next(
            block
            for message in transcript.messages
            if message.role is MessageRole.TOOL
            for block in message.content
            if isinstance(block, ToolResultBlock)
            and block.call_id == "invalid-continuation"
        )
        assert rejected.is_error
        error = rejected.output["error"]
        assert isinstance(error, Mapping)
        assert error["code"] == "cursor_invalid"
    finally:
        await agent.close()


async def test_hosted_composition_has_no_local_file_authority(tmp_path: Path) -> None:
    provider = MockModelProvider(
        (_call("forged-file-read", "file_read", {"path": "secret.txt"}), _stop())
    )
    hosted = await EmbeddedAgent.create(
        "hosted",
        hosted=True,
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
    )
    try:
        assert "file_search" not in hosted._capabilities.tool_names
        assert "file_read" not in hosted._capabilities.tool_names
        assert "artifact_save_local" not in hosted._capabilities.tool_names
        assert hosted._workspace_backend is None
        assert hosted._artifact_delivery is None
        assert not (set(hosted._capabilities._capabilities) & LOCAL_FILE_CAPABILITY_IDS)
        assert not (set(hosted._capabilities._executors) & LOCAL_FILE_EXECUTOR_IDS)
        assert {
            ARTIFACT_SAVE_LOCAL_EXECUTOR_ID,
            ARTIFACT_SET_EXPORT_LOCATION_EXECUTOR_ID,
        }.isdisjoint(hosted._capabilities._executors)
        result = await hosted.run("Attempt a forged local file call")
        transcript = await hosted.transcript(result.run_id)
        rejected = next(
            block
            for message in transcript.messages
            if message.role is MessageRole.TOOL
            for block in message.content
            if isinstance(block, ToolResultBlock)
            and block.call_id == "forged-file-read"
        )
        assert rejected.is_error
        error = rejected.output["error"]
        assert isinstance(error, Mapping)
        assert error["code"] == "tool_not_available"
        with pytest.raises(ValueError, match="hosted"):
            await EmbeddedAgent.open(
                "hosted",
                hosted=True,
                workspace=LocalWorkspace(tmp_path.parent),
                root=tmp_path,
            )
    finally:
        await hosted.close()


async def test_files_only_cannot_be_combined_with_a_selected_source(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    agent = await Agent.create("exclusive", workspace=workspace, root=tmp_path)
    try:
        with pytest.raises(ValueError, match="mutually exclusive"):
            await agent.run("invalid", source_id="source-1", files_only=True)
    finally:
        await agent.close()


async def test_workspace_sensitivity_rejects_ineligible_route_before_model_or_file_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path, sensitivity=ModelSensitivity.CONFIDENTIAL)
    calls = 0

    class IneligibleProvider:
        provider_id = "mock:workspace-ineligible"

        def supports_request_policy(self, request) -> bool:
            assert request.sensitivity is ModelSensitivity.CONFIDENTIAL
            return False

        async def generate(self, request):
            nonlocal calls
            calls += 1
            raise AssertionError("ineligible provider must not be called")

    async def unexpected_io(*_args, **_kwargs):
        raise AssertionError("workspace I/O must not begin")

    monkeypatch.setattr(LocalWorkspaceBackend, "search", unexpected_io)
    monkeypatch.setattr(LocalWorkspaceBackend, "read", unexpected_io)
    provider = IneligibleProvider()
    agent = await Agent.create(
        "workspace-route",
        workspace=workspace,
        root=tmp_path,
        model=provider,
        model_profile=ModelProfile(
            id=provider.provider_id,
            context_window_tokens=32_000,
            max_output_tokens=2_000,
            supports_tools=True,
            supports_parallel_tools=True,
        ),
    )
    try:
        result = await agent.run("Read a confidential workspace file")
        assert result.reason == "model_route_ineligible"
        assert calls == 0
    finally:
        await agent.close()


async def test_removed_preproduction_file_source_home_is_explicitly_rejected(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    agent = await Agent.create("old-file-state", workspace=workspace, root=tmp_path)
    state_path = agent.home / "state.db"
    await agent._embedded._store.register_source(
        SourceRegistration.build(
            agent_id=agent.id,
            adapter_id="sqlite",
            native_identity="removed-development-record",
            display_name="Removed development record",
            configuration={"path": str(tmp_path / "removed.sqlite")},
            attached_at=agent._embedded.identity.created_at,
        )
    )
    await agent.close()

    with sqlite3.connect(state_path) as connection:
        source_id, encoded = connection.execute(
            "SELECT id, data FROM sources"
        ).fetchone()
        payload = json.loads(encoded)
        payload["fields"]["adapter_id"] = "local-directory"
        connection.execute(
            "UPDATE sources SET data = ? WHERE id = ?",
            (json.dumps(payload, sort_keys=True, separators=(",", ":")), source_id),
        )

    with pytest.raises(RuntimeError, match="delete and recreate"):
        await Agent.open("old-file-state", workspace=workspace, root=tmp_path)
