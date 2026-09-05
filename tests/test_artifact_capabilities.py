from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from _capability_runtime_support import (
    ContextToolProjectionAdapter,
    execute_projected,
)
from _toolbox_model_support import (
    ToolboxAwareMockModelProvider as MockModelProvider,
)
from _workspace_support import workspace_for

import daita.domains.data.controller as data_controller
from daita import Agent, ApprovalDecision, ApprovalRequest, ArtifactError
from daita._json import FrozenJsonObject
from daita.capabilities import AccessMode, AutomationEligibility, OperationalEffect
from daita.domains.data.context import DataContextBuilder
from daita.domains.data.export_capabilities import (
    ARTIFACT_CONVERT_TOOL_NAME,
    ARTIFACT_CREATE_TABULAR_CAPABILITY_ID,
    ARTIFACT_EDIT_TEXT_TOOL_NAME,
    ARTIFACT_LIST_TOOL_NAME,
    ARTIFACT_READ_TOOL_NAME,
    ARTIFACT_SAVE_LOCAL_TOOL_NAME,
    ARTIFACT_SET_EXPORT_LOCATION_TOOL_NAME,
    DOCUMENT_CREATE_CAPABILITY_ID,
    DOCUMENT_CREATE_TOOL_NAME,
    DATA_EXPORT_TABULAR_CAPABILITY_ID,
    RESULT_SNAPSHOT_CAPABILITY_ID,
    RESULT_SNAPSHOT_TOOL_NAME,
    artifact_capability_declarations,
    data_export_tabular_capability_declarations,
)
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
)
from daita.loop.models import RunInput, Transcript
from daita.storage.sqlite_codecs import decode_message, encode_message


async def _prepared_request(
    builder: DataContextBuilder,
    run: RunInput,
    messages: tuple[CanonicalMessage, ...],
    tools: tuple[ToolDefinition, ...],
    *,
    step: int,
) -> ModelRequest:
    projection = ContextToolProjectionAdapter(tools)
    catalog = await projection.prepare_run(run)
    snapshot = await builder.prepare(run, messages, catalog)
    return builder.project(
        snapshot,
        messages,
        step=step,
        tool_context=projection.project(catalog, messages),
    )


def _ids():
    counts: defaultdict[str, int] = defaultdict(int)

    def create(prefix: str) -> str:
        counts[prefix] += 1
        if prefix in {"run", "conversation", "artifact", "destination"}:
            return f"{prefix}-{counts[prefix]:032x}"
        return f"{prefix}-{counts[prefix]}"

    return create


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=32_000,
        max_output_tokens=2_000,
        supports_tools=True,
        supports_parallel_tools=True,
    )


def _call(call_id: str, name: str, arguments: dict[str, object]) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
    )


def _stop(text: str = "done") -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text)


def _document_call() -> ModelResponse:
    return _call(
        "create",
        "artifact_create_document",
        {"format": "txt", "filename": "result.txt", "content": "result\n"},
    )


def _result_error_code(result: ToolResultBlock) -> str:
    error = result.output.get("error")
    assert isinstance(error, Mapping)
    code = error.get("code")
    assert isinstance(code, str)
    return code


def _tool_result(transcript: Transcript, call_id: str) -> ToolResultBlock:
    matches = tuple(
        block
        for message in transcript.messages
        for block in message.content
        if isinstance(block, ToolResultBlock) and block.call_id == call_id
    )
    assert len(matches) == 1
    return matches[0]


def test_artifact_intent_classifiers_are_removed_and_model_tools_stay_narrow() -> None:
    assert not hasattr(data_controller, "_explicit_artifact_request")
    assert not hasattr(data_controller, "_explicit_default_location_request")
    declarations = artifact_capability_declarations()
    schemas = {
        view.name: next(
            item for item in declarations.capabilities if item.id == view.capability_id
        ).input_schema
        for view in declarations.tool_views
        if view.name
        in {
            ARTIFACT_LIST_TOOL_NAME,
            ARTIFACT_READ_TOOL_NAME,
            ARTIFACT_CONVERT_TOOL_NAME,
        }
    }
    properties = {name: schema.get("properties") for name, schema in schemas.items()}
    assert all(isinstance(value, Mapping) for value in properties.values())
    list_properties = cast(Mapping[str, object], properties[ARTIFACT_LIST_TOOL_NAME])
    read_properties = cast(Mapping[str, object], properties[ARTIFACT_READ_TOOL_NAME])
    convert_properties = cast(
        Mapping[str, object], properties[ARTIFACT_CONVERT_TOOL_NAME]
    )
    assert set(list_properties) == set()
    assert set(read_properties) == {"artifact_id"}
    assert set(convert_properties) == {
        "artifact_id",
        "format",
        "filename",
    }
    for schema in schemas.values():
        assert schema["additionalProperties"] is False


def test_d2_certifies_only_the_three_accepted_scheduled_artifact_capabilities() -> None:
    declarations = artifact_capability_declarations()
    data_declarations = data_export_tabular_capability_declarations()
    capabilities = (*declarations.capabilities, *data_declarations.capabilities)
    by_id = {capability.id: capability for capability in capabilities}
    scheduled = {
        capability.id
        for capability in capabilities
        if capability.automation_eligibility is AutomationEligibility.SCHEDULED_DIRECT
    }

    assert scheduled == {
        DOCUMENT_CREATE_CAPABILITY_ID,
        RESULT_SNAPSHOT_CAPABILITY_ID,
        DATA_EXPORT_TABULAR_CAPABILITY_ID,
    }
    assert by_id[DATA_EXPORT_TABULAR_CAPABILITY_ID].access_mode is AccessMode.READ
    assert (
        by_id[ARTIFACT_CREATE_TABULAR_CAPABILITY_ID].automation_eligibility
        is AutomationEligibility.INTERACTIVE_ONLY
    )
    assert by_id[ARTIFACT_CREATE_TABULAR_CAPABILITY_ID].access_mode is AccessMode.NONE
    assert by_id[DOCUMENT_CREATE_CAPABILITY_ID].access_mode is AccessMode.NONE
    assert by_id[RESULT_SNAPSHOT_CAPABILITY_ID].access_mode is AccessMode.NONE
    assert all(
        by_id[capability_id].operational_effect is OperationalEffect.NONE
        for capability_id in scheduled
    )


async def test_result_snapshot_commits_exact_current_run_result_with_lineage(
    tmp_path: Path,
) -> None:
    provider = MockModelProvider(
        (
            _document_call(),
            _call(
                "snapshot",
                RESULT_SNAPSHOT_TOOL_NAME,
                {"call_id": "create", "filename": "result.json"},
            ),
            _stop(),
        ),
        provider_id="mock:result-snapshot",
    )
    agent = await Agent.create(
        "result-snapshot",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        result = await agent.run("Create a document, then snapshot its tool result.")
        assert len(result.artifacts) == 2
        snapshot_ref = result.artifacts[1]
    finally:
        await agent.close()
    reopened = await Agent.open(
        "result-snapshot",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
    )
    try:
        payload = await reopened.read_artifact(snapshot_ref.artifact_id)
    finally:
        await reopened.close()

    assert payload.content == b'{"character_count":7,"format":"txt"}'
    assert snapshot_ref.capability_id == RESULT_SNAPSHOT_CAPABILITY_ID
    assert snapshot_ref.media_type == "application/json"
    binding = snapshot_ref.provenance.result_binding
    assert binding is not None
    assert binding.call_id == "create"
    assert binding.capability_id == DOCUMENT_CREATE_CAPABILITY_ID
    assert binding.executor_id == "artifact.create_document.executor"
    assert binding.result_sha256 == snapshot_ref.sha256
    assert binding.producer_provenance["run_id"] == snapshot_ref.run_id


async def test_result_snapshot_rejects_cross_run_evidence(tmp_path: Path) -> None:
    provider = MockModelProvider(
        (
            _document_call(),
            _stop("first run complete"),
            _call("snapshot", RESULT_SNAPSHOT_TOOL_NAME, {"call_id": "create"}),
            _stop("second run complete"),
        ),
        provider_id="mock:result-snapshot-cross-run",
    )
    agent = await Agent.create(
        "result-snapshot-cross-run",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        first = await agent.run("Create one artifact.")
        second = await agent.run(
            "Try to snapshot the result from the preceding run.",
            conversation_id=first.conversation_id,
        )
        snapshot = _tool_result(await agent.transcript(second.run_id), "snapshot")
    finally:
        await agent.close()

    assert second.artifacts == ()
    assert snapshot.is_error is True
    assert _result_error_code(snapshot) == "artifact_snapshot_evidence_invalid"


async def test_result_snapshot_rejects_cross_agent_evidence(tmp_path: Path) -> None:
    producer = MockModelProvider(
        (_document_call(), _stop()),
        provider_id="mock:result-snapshot-agent-one",
    )
    first = await Agent.create(
        "result-snapshot-agent-one",
        root=tmp_path,
        model=producer,
        model_profile=_profile(producer),
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        await first.run("Create one artifact in the first agent.")
    finally:
        await first.close()

    consumer = MockModelProvider(
        (
            _call("snapshot", RESULT_SNAPSHOT_TOOL_NAME, {"call_id": "create"}),
            _stop(),
        ),
        provider_id="mock:result-snapshot-agent-two",
    )
    second = await Agent.create(
        "result-snapshot-agent-two",
        root=tmp_path,
        model=consumer,
        model_profile=_profile(consumer),
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        result = await second.run("Try to snapshot another agent's result.")
        snapshot = _tool_result(await second.transcript(result.run_id), "snapshot")
    finally:
        await second.close()

    assert result.artifacts == ()
    assert snapshot.is_error is True
    assert _result_error_code(snapshot) == "artifact_snapshot_evidence_invalid"


async def test_result_snapshot_rejects_failed_call_evidence(tmp_path: Path) -> None:
    provider = MockModelProvider(
        (
            _call(
                "failed",
                DOCUMENT_CREATE_TOOL_NAME,
                {"format": "pdf", "content": "not admitted"},
            ),
            _call("snapshot", RESULT_SNAPSHOT_TOOL_NAME, {"call_id": "failed"}),
            _stop(),
        ),
        provider_id="mock:result-snapshot-failed-call",
    )
    agent = await Agent.create(
        "result-snapshot-failed-call",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        result = await agent.run("Try to snapshot a failed tool call.")
        transcript = await agent.transcript(result.run_id)
    finally:
        await agent.close()

    assert _tool_result(transcript, "failed").is_error is True
    snapshot = _tool_result(transcript, "snapshot")
    assert snapshot.is_error is True
    assert _result_error_code(snapshot) == "artifact_snapshot_evidence_invalid"
    assert result.artifacts == ()


class _TranscriptTamperingProvider(MockModelProvider):
    def __init__(
        self,
        tamper: Callable[[ToolResultBlock], ToolResultBlock],
        *,
        provider_id: str,
    ) -> None:
        super().__init__(
            (
                _document_call(),
                _call(
                    "snapshot",
                    RESULT_SNAPSHOT_TOOL_NAME,
                    {"call_id": "create"},
                ),
                _stop(),
            ),
            provider_id=provider_id,
        )
        self._tamper = tamper
        self.state_path: Path | None = None
        self.tampered = False

    async def generate(self, request: ModelRequest) -> ModelResponse:
        if not self.tampered and any(
            isinstance(block, ToolResultBlock) and block.call_id == "create"
            for message in request.messages
            for block in message.content
        ):
            assert self.state_path is not None
            with sqlite3.connect(self.state_path) as connection:
                rows = connection.execute(
                    "SELECT run_id, position, data FROM messages ORDER BY run_id, position"
                ).fetchall()
                for run_id, position, data in rows:
                    message = decode_message(data)
                    if not any(
                        isinstance(block, ToolResultBlock) and block.call_id == "create"
                        for block in message.content
                    ):
                        continue
                    tampered = replace(
                        message,
                        content=tuple(
                            (
                                self._tamper(block)
                                if isinstance(block, ToolResultBlock)
                                and block.call_id == "create"
                                else block
                            )
                            for block in message.content
                        ),
                    )
                    connection.execute(
                        "UPDATE messages SET data = ? WHERE run_id = ? AND position = ?",
                        (encode_message(tampered), run_id, position),
                    )
                    self.tampered = True
                    break
            assert self.tampered
        return await super().generate(request)


async def test_result_snapshot_rejects_tampered_execution_lineage(
    tmp_path: Path,
) -> None:
    provider = _TranscriptTamperingProvider(
        lambda block: replace(block, executor_id="tampered.executor"),
        provider_id="mock:result-snapshot-lineage-tampered",
    )
    agent = await Agent.create(
        "result-snapshot-tampered",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    provider.state_path = agent._embedded._store.path
    try:
        result = await agent.run("Create a document and snapshot its exact result.")
        snapshot = _tool_result(await agent.transcript(result.run_id), "snapshot")
    finally:
        await agent.close()

    assert provider.tampered is True
    assert len(result.artifacts) == 1
    assert snapshot.is_error is True
    assert _result_error_code(snapshot) == "artifact_snapshot_evidence_invalid"


def _tamper_result_data(block: ToolResultBlock) -> ToolResultBlock:
    output = dict(block.output)
    data = output.get("data")
    assert isinstance(data, Mapping)
    changed = dict(data)
    changed["character_count"] = 8
    output["data"] = changed
    return replace(block, output=output)


async def test_result_snapshot_rejects_schema_valid_result_data_tampering(
    tmp_path: Path,
) -> None:
    provider = _TranscriptTamperingProvider(
        _tamper_result_data,
        provider_id="mock:result-snapshot-data-tampered",
    )
    agent = await Agent.create(
        "result-snapshot-data-tampered",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    provider.state_path = agent._embedded._store.path
    try:
        result = await agent.run("Create a document and snapshot its exact result.")
        snapshot = _tool_result(await agent.transcript(result.run_id), "snapshot")
    finally:
        await agent.close()

    assert provider.tampered is True
    assert len(result.artifacts) == 1
    assert snapshot.is_error is True
    assert _result_error_code(snapshot) == "artifact_snapshot_evidence_invalid"


async def test_result_snapshot_obeys_the_existing_artifact_quota(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import daita.artifacts.store as artifact_store_module

    monkeypatch.setattr(artifact_store_module, "MAX_ARTIFACTS_PER_RUN", 1)
    provider = MockModelProvider(
        (
            _document_call(),
            _call("snapshot", RESULT_SNAPSHOT_TOOL_NAME, {"call_id": "create"}),
            _stop(),
        ),
        provider_id="mock:result-snapshot-quota",
    )
    agent = await Agent.create(
        "result-snapshot-quota",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        result = await agent.run("Create and snapshot one result.")
        snapshot = _tool_result(await agent.transcript(result.run_id), "snapshot")
    finally:
        await agent.close()

    assert len(result.artifacts) == 1
    assert snapshot.is_error is True
    assert _result_error_code(snapshot) == "artifact_quota_exceeded"


async def test_result_snapshot_cancellation_cannot_publish_a_partial_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MockModelProvider(
        (
            _document_call(),
            _call("snapshot", RESULT_SNAPSHOT_TOOL_NAME, {"call_id": "create"}),
            _stop(),
        ),
        provider_id="mock:result-snapshot-cancelled",
    )
    agent = await Agent.create(
        "result-snapshot-cancelled",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    entered = threading.Event()
    release = threading.Event()
    store = agent._embedded._artifact_store
    original_commit = store._commit_sync

    def blocked_snapshot_commit(*args: object, **kwargs: object):
        capability_id = args[5]
        if capability_id == RESULT_SNAPSHOT_CAPABILITY_ID:
            entered.set()
            release.wait(timeout=5)
        return original_commit(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(store, "_commit_sync", blocked_snapshot_commit)
    run = asyncio.create_task(agent.run("Create and snapshot one result."))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        run.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await run
        refs = await store.list_refs()
        assert len(refs) == 1
        assert refs[0].capability_id == DOCUMENT_CREATE_CAPABILITY_ID
        assert not tuple(store.staging.iterdir())
    finally:
        release.set()
        await agent.close()


def test_artifact_save_local_schema_rejects_bytes_paths_commands_and_overwrite() -> (
    None
):
    extension = artifact_capability_declarations()
    view = next(
        item
        for item in extension.tool_views
        if item.name == ARTIFACT_SAVE_LOCAL_TOOL_NAME
    )
    capability = next(
        item for item in extension.capabilities if item.id == view.capability_id
    )
    schema = FrozenJsonObject.from_mapping(capability.input_schema)
    assert schema.to_dict() == {
        "type": "object",
        "properties": {
            "artifact_id": {
                "type": "string",
                "pattern": "^artifact-[0-9a-f]{32}$",
            },
            "mode": {
                "type": "string",
                "enum": ["create_new", "replace_bound_file"],
            },
            "destination_id": {
                "type": "string",
                "minLength": 1,
                "maxLength": 64,
            },
            "filename": {"type": "string", "minLength": 1, "maxLength": 120},
        },
        "required": ["artifact_id", "mode"],
        "additionalProperties": False,
    }
    properties = schema["properties"]
    assert isinstance(properties, FrozenJsonObject)
    assert set(properties) == {
        "artifact_id",
        "mode",
        "destination_id",
        "filename",
    }


def test_artifact_edit_binding_contract_requires_verbatim_opaque_reuse() -> None:
    declarations = artifact_capability_declarations(include_local_edit=True)
    view = next(
        item
        for item in declarations.tool_views
        if item.name == ARTIFACT_EDIT_TEXT_TOOL_NAME
    )
    capability = next(
        item for item in declarations.capabilities if item.id == view.capability_id
    )
    properties = capability.input_schema.get("properties")
    assert isinstance(properties, Mapping)
    binding = properties.get("binding")
    assert isinstance(binding, Mapping)
    contract = f"{capability.description} {binding.get('description')}".casefold()
    assert "verbatim" in contract
    assert "opaque" in contract
    assert "never be decoded" in contract


def test_artifact_set_export_location_schema_accepts_only_one_destination_id() -> None:
    extension = artifact_capability_declarations()
    view = next(
        item
        for item in extension.tool_views
        if item.name == ARTIFACT_SET_EXPORT_LOCATION_TOOL_NAME
    )
    capability = next(
        item for item in extension.capabilities if item.id == view.capability_id
    )
    assert FrozenJsonObject.from_mapping(capability.input_schema).to_dict() == {
        "type": "object",
        "properties": {
            "destination_id": {
                "type": "string",
                "minLength": 1,
                "maxLength": 64,
            }
        },
        "required": ["destination_id"],
        "additionalProperties": False,
    }


async def test_artifact_save_local_uses_only_committed_current_agent_artifact(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    first_provider = MockModelProvider(
        (_document_call(), _stop()), provider_id="mock:artifact-owner-one"
    )
    first = await Agent.create(
        "artifact-owner-one",
        root=tmp_path,
        model=first_provider,
        model_profile=_profile(first_provider),
        id_factory=_ids(),
        downloads_directory=downloads,
        workspace=workspace_for(tmp_path),
    )
    try:
        foreign_id = (await first.run("Create a file.")).artifacts[0].artifact_id
    finally:
        await first.close()
    second_downloads = tmp_path / "downloads-two"
    second_downloads.mkdir()
    second = await Agent.create(
        "artifact-owner-two",
        root=tmp_path,
        downloads_directory=second_downloads,
        workspace=workspace_for(tmp_path),
    )
    try:
        with pytest.raises(ArtifactError) as failure:
            await second.save_artifact(foreign_id)
        assert failure.value.code == "artifact_missing"
        assert not tuple(second_downloads.iterdir())
    finally:
        await second.close()


async def test_system_and_persistent_save_preflight_are_preauthorized_for_explicit_file_request(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    persistent = tmp_path / "persistent"
    downloads.mkdir()
    persistent.mkdir()
    artifact_id = "artifact-00000000000000000000000000000001"
    destination_id = "destination-00000000000000000000000000000001"
    provider = MockModelProvider(
        (
            _document_call(),
            _call(
                "save-system",
                ARTIFACT_SAVE_LOCAL_TOOL_NAME,
                {
                    "artifact_id": artifact_id,
                    "mode": "create_new",
                    "destination_id": "default",
                },
            ),
            _stop("saved system"),
            _call(
                "save-persistent",
                ARTIFACT_SAVE_LOCAL_TOOL_NAME,
                {
                    "artifact_id": artifact_id,
                    "mode": "create_new",
                    "destination_id": destination_id,
                },
            ),
            _stop("saved persistent"),
        ),
        provider_id="mock:artifact-preauthorized",
    )
    approvals: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.APPROVE

    agent = await Agent.create(
        "artifact-preauthorized",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        approval_handler=approve,
        downloads_directory=downloads,
        workspace=workspace_for(tmp_path),
    )
    try:
        first = await agent.run("Create and download a file.")
        selected = await agent.set_export_destination(persistent)
        assert selected.destination_id == destination_id
        second = await agent.run(
            "Save the existing artifact to my persistent export location.",
            conversation_id=first.conversation_id,
        )
        assert len(first.artifact_deliveries) == 1
        assert len(second.artifact_deliveries) == 1
        assert approvals == []
    finally:
        await agent.close()


async def test_one_time_save_approval_is_bound_to_frozen_artifact_and_destination_fingerprint(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    one_time = tmp_path / "one-time"
    downloads.mkdir()
    one_time.mkdir()
    provider = MockModelProvider(
        (_document_call(), _stop()), provider_id="mock:one-time-fingerprint"
    )
    approvals: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.APPROVE

    agent = await Agent.create(
        "one-time-fingerprint",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        approval_handler=approve,
        downloads_directory=downloads,
        workspace=workspace_for(tmp_path),
    )
    try:
        created = await agent.run("Create a file.")
        artifact_id = created.artifacts[0].artifact_id
        run_id = "run-ffffffffffffffffffffffffffffffff"
        delivery = agent._embedded._artifact_delivery
        assert delivery is not None
        destination = await delivery.register_one_time(
            one_time,
            run_id=run_id,
        )
        run = RunInput(
            id=run_id,
            agent_id=agent.id,
            message="Save this file once to the selected folder.",
            created_at=agent._embedded.identity.created_at,
            conversation_id=created.conversation_id,
        )
        runtime = agent._embedded._capability_runtime
        result = (
            await execute_projected(
                runtime,
                run,
                (
                    ToolCall(
                        id="save-once",
                        name=ARTIFACT_SAVE_LOCAL_TOOL_NAME,
                        arguments={
                            "artifact_id": artifact_id,
                            "mode": "create_new",
                            "destination_id": destination.destination_id,
                        },
                    ),
                ),
                sensitivity=ModelSensitivity.INTERNAL,
            )
        )[0]
        assert not result.is_error
        assert len(approvals) == 1
        fingerprint = approvals[0].arguments
        assert fingerprint["artifact_id"] == artifact_id
        artifact_sha256 = fingerprint["artifact_sha256"]
        artifact_byte_size = fingerprint["artifact_byte_size"]
        grant_digest = fingerprint["grant_digest"]
        assert isinstance(artifact_sha256, str)
        assert artifact_sha256.startswith("sha256:")
        assert isinstance(artifact_byte_size, int)
        assert artifact_byte_size > 0
        assert fingerprint["destination_id"] == destination.destination_id
        assert isinstance(grant_digest, str)
        assert grant_digest.startswith("sha256:")
        assert fingerprint["requested_filename"] == "result.txt"
    finally:
        await agent.close()


async def test_export_location_change_always_requires_exact_once_only_model_approval(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    provider = MockModelProvider(
        (
            _call(
                "set-location",
                ARTIFACT_SET_EXPORT_LOCATION_TOOL_NAME,
                {"destination_id": "destination-system-downloads"},
            ),
            _stop(),
        ),
        provider_id="mock:set-location-approval",
    )
    approvals: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.APPROVE

    agent = await Agent.create(
        "set-location-approval",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        approval_handler=approve,
        downloads_directory=downloads,
        workspace=workspace_for(tmp_path),
    )
    try:
        result = await agent.run("Use Downloads for future exports.")
        assert result.final_text == "done"
        assert len(approvals) == 1
        assert approvals[0].reason == (
            "Make “Downloads” the default location for future exports?"
        )
        assert approvals[0].arguments["destination_id"] == (
            "destination-system-downloads"
        )
        config_sha256 = approvals[0].arguments["delivery_config_sha256"]
        assert isinstance(config_sha256, str)
        assert config_sha256.startswith("sha256:")
    finally:
        await agent.close()


async def test_export_location_rejects_one_time_grant_and_rechecks_config_under_mutation_lock(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    persistent = tmp_path / "persistent"
    one_time = tmp_path / "one-time"
    for directory in (downloads, persistent, one_time):
        directory.mkdir()
    provider = MockModelProvider(
        (
            _call(
                "set-system",
                ARTIFACT_SET_EXPORT_LOCATION_TOOL_NAME,
                {"destination_id": "destination-system-downloads"},
            ),
            _stop(),
        ),
        provider_id="mock:set-location-state-change",
    )
    agent: Agent | None = None

    async def change_then_approve(request: ApprovalRequest) -> ApprovalDecision:
        del request
        assert agent is not None
        await agent.reset_export_destination()
        return ApprovalDecision.APPROVE

    agent = await Agent.create(
        "set-location-state-change",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        approval_handler=change_then_approve,
        downloads_directory=downloads,
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.set_export_destination(persistent)
        run_id = "run-eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
        delivery = agent._embedded._artifact_delivery
        assert delivery is not None
        one_time_view = await delivery.register_one_time(one_time, run_id=run_id)
        with pytest.raises(ArtifactError) as rejected:
            await delivery.preflight_set_default(one_time_view.destination_id)
        assert rejected.value.code == "artifact_destination_unauthorized"
        result = await agent.run("Change the future export location to Downloads.")
        tool_results = tuple(
            block
            for message in (await agent.transcript(result.run_id)).messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        )
        assert _result_error_code(tool_results[-1]) == "state_changed"
    finally:
        await agent.close()


async def test_context_requires_default_delivery_before_final_text_for_explicit_file_request(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    provider = MockModelProvider((_stop(),), provider_id="mock:artifact-context")
    agent = await Agent.create(
        "artifact-context",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        downloads_directory=downloads,
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.run("Create and download a Markdown file.")
        system = "\n".join(
            block.text
            for message in provider.requests[0].messages
            if message.role is MessageRole.SYSTEM
            for block in message.content
            if isinstance(block, TextBlock)
        )
        assert "artifact_create_document" in system
        assert (
            'artifact_save_local with mode="create_new" and '
            'destination_id="default" before normal' in system
        )
        assert "Normal assistant text ends the run" in system
        assert "Ordinary user wording is not an exact stored value" in system
        assert "bounded validated value read" in system
        assert "A committed artifact reference proves only internal creation" in system
        assert (
            "Only a successful artifact delivery receipt proves a local file exists"
            in system
        )
    finally:
        await agent.close()


async def test_context_does_not_create_artifacts_for_ordinary_analysis_or_reads(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    provider = MockModelProvider((_stop("analysis"),), provider_id="mock:no-artifact")
    agent = await Agent.create(
        "no-artifact",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        downloads_directory=downloads,
        workspace=workspace_for(tmp_path),
    )
    try:
        result = await agent.run("Summarize the approach in chat.")
        assert result.artifacts == ()
        assert result.artifact_deliveries == ()
        assert not tuple(downloads.iterdir())
    finally:
        await agent.close()


async def test_model_artifact_tools_use_exact_pinned_surface_without_classification(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    provider = MockModelProvider((_stop("analysis"),), provider_id="mock:model-led")
    agent = await Agent.create(
        "artifact-model-led",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        downloads_directory=downloads,
        workspace=workspace_for(tmp_path),
    )
    try:
        result = await agent.run("Show my user profile.")
        assert result.artifacts == ()
        projected = {tool.name for tool in provider.requests[0].tools}
        assert {
            ARTIFACT_LIST_TOOL_NAME,
            ARTIFACT_READ_TOOL_NAME,
        } <= projected
        assert {"toolbox_search", "toolbox_load"} <= projected
        assert {
            DOCUMENT_CREATE_TOOL_NAME,
            ARTIFACT_SET_EXPORT_LOCATION_TOOL_NAME,
            ARTIFACT_CONVERT_TOOL_NAME,
            ARTIFACT_SAVE_LOCAL_TOOL_NAME,
        }.isdisjoint(projected)
        assert not tuple(downloads.iterdir())
    finally:
        await agent.close()


async def test_default_location_request_leaves_operation_choice_to_the_model(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    provider = MockModelProvider(
        (_stop(),),
        provider_id="mock:default-location-intent",
    )
    agent = await Agent.create(
        "default-location-intent",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        downloads_directory=downloads,
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.run("Make Downloads my default export location.")
        request = provider.requests[0]
        projected = {tool.name for tool in request.tools}
        assert {"toolbox_search", "toolbox_load"} <= projected
        assert {
            ARTIFACT_SET_EXPORT_LOCATION_TOOL_NAME,
            ARTIFACT_SAVE_LOCAL_TOOL_NAME,
            DOCUMENT_CREATE_TOOL_NAME,
        }.isdisjoint(projected)
        system = "\n".join(
            block.text
            for message in request.messages
            if message.role is MessageRole.SYSTEM
            for block in message.content
            if isinstance(block, TextBlock)
        )
        assert "Call artifact_set_export_location only" in system
        assert "call artifact_create_document and then call" not in system
    finally:
        await agent.close()


class _Catalog:
    async def admitted_model_sensitivity(
        self, agent_id: str, source_ids: tuple[str, ...] = ()
    ) -> ModelSensitivity:
        del agent_id, source_ids
        return ModelSensitivity.PUBLIC

    async def catalog_context(self, *args, **kwargs) -> FrozenJsonObject:
        del args, kwargs
        return FrozenJsonObject.from_mapping(
            {
                "resources": [],
                "sources": [],
                "total_matches": 0,
                "returned_count": 0,
                "truncated": False,
                "trust_classification": "untrusted_external_data",
            }
        )


async def test_hosted_composition_does_not_project_local_delivery_tools_or_paths() -> (
    None
):
    profile = ModelProfile(
        id="mock:hosted",
        context_window_tokens=16_000,
        max_output_tokens=1_000,
        supports_tools=True,
    )
    builder = DataContextBuilder(_Catalog(), profile=profile)
    request = await _prepared_request(
        builder,
        RunInput(
            id="run-hosted",
            agent_id="agent-hosted",
            message="Create a file.",
            created_at=datetime.now(UTC),
        ),
        (
            CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock("Create a file."),),
            ),
        ),
        (),
        step=1,
    )
    assert request.tools == ()
    assert "Available local artifact destinations" not in str(request)
    assert "artifact_save_local" not in str(request)
