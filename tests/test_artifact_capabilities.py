from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest

from daita import Agent, ApprovalDecision, ApprovalRequest, ArtifactError
from daita._json import FrozenJsonObject
import daita.domains.data.controller as data_controller
from daita.domains.data.context import DataContextBuilder
from daita.domains.data.export_capabilities import (
    ARTIFACT_CONVERT_TOOL_NAME,
    ARTIFACT_LIST_TOOL_NAME,
    ARTIFACT_READ_TOOL_NAME,
    ARTIFACT_SAVE_LOCAL_TOOL_NAME,
    ARTIFACT_SET_EXPORT_LOCATION_TOOL_NAME,
    DOCUMENT_CREATE_TOOL_NAME,
    artifact_extension_declarations,
)
from daita.llm.models import (
    FinishReason,
    CanonicalMessage,
    MessageRole,
    ModelProfile,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import RunInput


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


def test_artifact_intent_classifiers_are_removed_and_model_tools_stay_narrow() -> None:
    assert not hasattr(data_controller, "_explicit_artifact_request")
    assert not hasattr(data_controller, "_explicit_default_location_request")
    declarations = artifact_extension_declarations()
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


def test_artifact_save_local_schema_rejects_bytes_paths_commands_and_overwrite() -> (
    None
):
    extension = artifact_extension_declarations()
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
            "destination_id": {
                "type": "string",
                "minLength": 1,
                "maxLength": 64,
            },
            "filename": {"type": "string", "minLength": 1, "maxLength": 120},
        },
        "required": ["artifact_id", "destination_id"],
        "additionalProperties": False,
    }
    properties = schema["properties"]
    assert isinstance(properties, FrozenJsonObject)
    assert set(properties) == {
        "artifact_id",
        "destination_id",
        "filename",
    }


def test_artifact_set_export_location_schema_accepts_only_one_destination_id() -> None:
    extension = artifact_extension_declarations()
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
                {"artifact_id": artifact_id, "destination_id": "default"},
            ),
            _stop("saved system"),
            _call(
                "save-persistent",
                ARTIFACT_SAVE_LOCAL_TOOL_NAME,
                {"artifact_id": artifact_id, "destination_id": destination_id},
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
    )
    try:
        created = await agent.run("Create a file.")
        artifact_id = created.artifacts[0].artifact_id
        run_id = "run-ffffffffffffffffffffffffffffffff"
        destination = await agent._embedded._artifact_delivery.register_one_time(
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
        result = (
            await agent._embedded._data_tool_runtime.execute_all(
                run,
                (
                    ToolCall(
                        id="save-once",
                        name=ARTIFACT_SAVE_LOCAL_TOOL_NAME,
                        arguments={
                            "artifact_id": artifact_id,
                            "destination_id": destination.destination_id,
                        },
                    ),
                ),
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
    )
    try:
        await agent.set_export_destination(persistent)
        run_id = "run-eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
        one_time_view = await agent._embedded._artifact_delivery.register_one_time(
            one_time, run_id=run_id
        )
        with pytest.raises(ArtifactError) as rejected:
            await agent._embedded._artifact_delivery.preflight_set_default(
                one_time_view.destination_id
            )
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
            'artifact_save_local with destination_id="default" before normal' in system
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
    )
    try:
        result = await agent.run("Summarize the approach in chat.")
        assert result.artifacts == ()
        assert result.artifact_deliveries == ()
        assert not tuple(downloads.iterdir())
    finally:
        await agent.close()


async def test_model_artifact_tools_are_projected_without_prompt_classification(
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
    )
    try:
        result = await agent.run("Show my user profile.")
        assert result.artifacts == ()
        projected = {tool.name for tool in provider.requests[0].tools}
        assert {
            DOCUMENT_CREATE_TOOL_NAME,
            ARTIFACT_SET_EXPORT_LOCATION_TOOL_NAME,
        } <= projected
        assert {
            ARTIFACT_LIST_TOOL_NAME,
            ARTIFACT_READ_TOOL_NAME,
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
    )
    try:
        await agent.run("Make Downloads my default export location.")
        request = provider.requests[0]
        projected = {tool.name for tool in request.tools}
        assert ARTIFACT_SET_EXPORT_LOCATION_TOOL_NAME in projected
        assert ARTIFACT_SAVE_LOCAL_TOOL_NAME not in projected
        assert DOCUMENT_CREATE_TOOL_NAME in projected
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
    async def catalog_context(self, *args, **kwargs) -> FrozenJsonObject:
        del args, kwargs
        return FrozenJsonObject.from_mapping(
            {
                "resources": [],
                "total_matches": 0,
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
    request = await builder.build(
        RunInput(
            id="run-hosted",
            agent_id="agent-hosted",
            message="Create a file.",
            created_at=datetime.now(timezone.utc),
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
