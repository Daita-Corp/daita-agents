"""Phase 0 locks for the pre-workspace local-file architecture.

These tests intentionally assert only current behavior. The red/green cases for
later owning phases are recorded here without executable future assertions:

Phase 1 — toolbox replacement:

* replace the three exposure classes and numeric eager priority with one exact
  canonical toolbox and pinned/on-demand load mode per applicable tool;
* replace the public projection enum, domain manifest, three runtime controls,
  deferred references, and nested calls with bounded toolbox search/load and an
  ordinary next-step call;
* replace the persisted MCP discovery fields in codec-v1 in place and prove
  remote text cannot select a toolbox, pin itself, or change effects.

Phase 2 — local workspace read slice:

* require and admit one non-overlapping local workspace, with explicit/safe-cwd/
  fallback CLI resolution and explicit hosted omission;
* project source-independent file_search/file_read for source-free and selected-
  source user runs, while files-only with multiple sources omits source tools;
* apply workspace sensitivity before model or file I/O and reject containment,
  symlink, race, secret-path, timeout, cancellation, and output-bound failures;
* remove LocalDirectorySource, local-directory, data_read_file, data_export_file,
  and their public/CLI/TUI/capability surfaces atomically.

Phase 3 — targeted text edit and exact publication:

* create a revision-bound edit artifact from an authenticated current-run file
  binding without mutating the source;
* add a mutually exclusive replace_bound_file save branch whose target comes
  only from the committed binding, with approval, drift recheck, atomic replace,
  uncertainty handling, and prior/result revision receipt facts;
* retain the current create-new destination admission, no-overwrite, and
  collision-safe receipt behavior unchanged.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sqlite3
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import fields
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest

import daita
import daita.adapters
import daita.capabilities
import daita.cli as cli
import daita.loop
from daita import Agent, ArtifactError, LocalDirectorySource, SQLiteSource
from daita._json import FrozenJsonObject, canonical_json
from daita.adapters.mcp import (
    MCPAuthentication,
    MCPBindingState,
    MCPServerBinding,
    MCPToolBinding,
)
from daita.artifacts.models import (
    SYSTEM_DOWNLOADS_DESTINATION_ID,
    ArtifactDeliveryReceipt,
    artifact_delivery_receipt_to_mapping,
)
from daita.capabilities import (
    TOOLBOX_DEFINITIONS,
    ToolLoadMode,
    ToolPresentation,
    ToolboxDefinition,
    ToolboxId,
    ToolTextTrust,
    ToolView,
)
from daita.capability_runtime import (
    RunToolCatalogEntry,
    StepToolProjection,
    ToolboxManifestEntry,
    _control_definitions,
)
from daita.domains.data.export_capabilities import (
    ARTIFACT_SAVE_LOCAL_CAPABILITY_ID,
    LOCAL_FILE_COPY_CAPABILITY_ID,
    LOCAL_FILE_COPY_EXECUTOR_ID,
    LOCAL_FILE_COPY_TOOL_NAME,
)
from daita.domains.data.file_capabilities import (
    LOCAL_FILE_READ_CAPABILITY_ID,
    LOCAL_FILE_READ_EXECUTOR_ID,
    LOCAL_FILE_READ_TOOL_NAME,
)
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    ModelSensitivity,
    ToolCall,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import LoopLimits, RunInput
from daita.storage.sqlite_codecs import encode_mcp_binding
from daita.tui.models import SOURCE_TYPE_LABELS, SOURCE_TYPES

_NOW = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)
_HOSTED_SOURCE_FREE_TOOL_DEFINITION_DIGEST = (
    "sha256:3a5cd3b42aa92875002b76c4b07748fd67224af57b5a53cc37d6d77c3e03d058"
)


def _database(path: Path, table: str) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            f'CREATE TABLE "{table}" (id INTEGER PRIMARY KEY)'  # noqa: S608
        )


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=32_000,
        max_output_tokens=2_000,
        supports_tools=True,
        supports_parallel_tools=True,
    )


def _stop(text: str) -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text)


def _definition_digest(definitions: tuple[object, ...]) -> str:
    material = tuple(
        {
            "name": getattr(definition, "name"),
            "description": getattr(definition, "description"),
            "input_schema": getattr(definition, "input_schema"),
        }
        for definition in definitions
    )
    return "sha256:" + sha256(canonical_json(material).encode("utf-8")).hexdigest()


def _ids():
    counts: defaultdict[str, int] = defaultdict(int)

    def create(prefix: str) -> str:
        counts[prefix] += 1
        if prefix in {"run", "conversation", "artifact", "destination"}:
            return f"{prefix}-{counts[prefix]:032x}"
        return f"{prefix}-{counts[prefix]}"

    return create


def _attach_kind_choices(parser: argparse.ArgumentParser) -> tuple[str, ...]:
    command_action = next(
        action for action in parser._actions if action.dest == "command"
    )
    command_choices = getattr(command_action, "choices")
    attach_parser = command_choices["attach"]
    kind_action = next(
        action for action in attach_parser._actions if action.dest == "kind"
    )
    return tuple(kind_action.choices or ())


async def test_source_free_run_constructs_with_exact_current_hosted_tool_surface(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    provider = MockModelProvider(
        (_stop("source-free"),), provider_id="mock:phase0-source-free"
    )
    agent = await Agent.create(
        "phase0-source-free",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        downloads_directory=downloads,
    )
    try:
        result = await agent.run("Construct a source-free run.")
        run = (await agent.transcript(result.run_id)).run
        definitions = provider.requests[0].tools

        assert run.source_id is None
        assert run.conversation_source_id is None
        assert tuple(tool.name for tool in definitions) == (
            "artifact_list",
            "artifact_read",
            "job_list",
            "skill_view",
            "toolbox_load",
            "toolbox_search",
        )
        assert _definition_digest(definitions) == (
            _HOSTED_SOURCE_FREE_TOOL_DEFINITION_DIGEST
        )
        assert {
            "data_read_file",
            "file_search",
            "file_read",
            "file_query",
        }.isdisjoint(tool.name for tool in definitions)
    finally:
        await agent.close()


async def test_explicit_source_resolves_one_run_with_multiple_sources(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.sqlite"
    second_path = tmp_path / "second.sqlite"
    _database(first_path, "first_records")
    _database(second_path, "second_records")
    provider = MockModelProvider(
        (_stop("explicit"),), provider_id="mock:phase0-explicit"
    )
    agent = await Agent.create(
        "phase0-explicit",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
    )
    try:
        first = await agent.attach(SQLiteSource(first_path, name="First"))
        second = await agent.attach(SQLiteSource(second_path, name="Second"))

        result = await agent.run("Use the second source.", source_id=second.id)
        run = (await agent.transcript(result.run_id)).run
        request_text = repr(provider.requests[0].messages)

        assert run.source_id == second.id
        assert run.conversation_source_id == first.id
        assert second.id in request_text
        assert first.id not in request_text
    finally:
        await agent.close()


async def test_sticky_conversation_source_wins_after_another_source_is_connected(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "sticky.sqlite"
    second_path = tmp_path / "later.sqlite"
    _database(first_path, "sticky_records")
    _database(second_path, "later_records")
    provider = MockModelProvider(
        (_stop("first"), _stop("follow-up")),
        provider_id="mock:phase0-sticky",
    )
    agent = await Agent.create(
        "phase0-sticky",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
    )
    try:
        sticky = await agent.attach(SQLiteSource(first_path, name="Sticky"))
        first = await agent.run("Start with the only source.")
        later = await agent.attach(SQLiteSource(second_path, name="Later"))

        follow_up = await agent.run(
            "Continue with the sticky source.",
            conversation_id=first.conversation_id,
        )
        run = (await agent.transcript(follow_up.run_id)).run
        request_text = repr(provider.requests[1].messages)

        assert run.source_id == sticky.id
        assert run.conversation_source_id == sticky.id
        assert sticky.id in request_text
        assert later.id not in request_text
    finally:
        await agent.close()


async def test_multiple_connected_sources_keep_the_first_persisted_default(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "ambiguous-first.sqlite"
    second_path = tmp_path / "ambiguous-second.sqlite"
    _database(first_path, "first_records")
    _database(second_path, "second_records")
    provider = MockModelProvider(
        (_stop("first default"),), provider_id="mock:phase0-multiple"
    )
    agent = await Agent.create(
        "phase0-multiple",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
    )
    try:
        first = await agent.attach(SQLiteSource(first_path, name="First"))
        second = await agent.attach(SQLiteSource(second_path, name="Second"))

        result = await agent.run("Use the current default.")
        run = (await agent.transcript(result.run_id)).run

        assert await agent.active_source() == first
        assert run.source_id == first.id
        assert run.conversation_source_id == first.id
        assert second != first
    finally:
        await agent.close()


def test_phase1_public_cli_tui_and_local_file_identifiers_are_exact() -> None:
    assert daita.LocalDirectorySource is LocalDirectorySource
    assert "LocalDirectorySource" in daita.__all__
    assert "ToolProjectionMode" not in daita.__all__
    assert {
        "LocalDirectoryReadBackend",
        "LocalDirectoryResourceAdapter",
        "LocalDirectorySource",
        "LocalDirectorySourceError",
    } <= set(daita.adapters.__all__)
    assert {
        "TOOLBOX_DEFINITIONS",
        "ToolLoadMode",
        "ToolPresentation",
        "ToolboxDefinition",
        "ToolboxId",
        "ToolTextTrust",
    } <= set(daita.capabilities.__all__)
    assert {"ToolDiscoveryMetadata", "ToolExposureClass"}.isdisjoint(
        daita.capabilities.__all__
    )
    assert "ToolProjectionMode" not in daita.loop.__all__

    assert _attach_kind_choices(cli.build_parser()) == (
        "sqlite",
        "files",
        "postgresql",
    )
    assert SOURCE_TYPES == (
        ("sqlite", "SQLite file"),
        ("directory", "Local CSV/JSON directory"),
        ("postgresql", "PostgreSQL"),
    )
    assert SOURCE_TYPE_LABELS["local-directory"] == "CSV/JSON"

    assert (
        LOCAL_FILE_READ_CAPABILITY_ID,
        LOCAL_FILE_READ_EXECUTOR_ID,
        LOCAL_FILE_READ_TOOL_NAME,
    ) == ("data.file.read", "data.file.read.executor", "data_read_file")
    assert (
        LOCAL_FILE_COPY_CAPABILITY_ID,
        LOCAL_FILE_COPY_EXECUTOR_ID,
        LOCAL_FILE_COPY_TOOL_NAME,
    ) == (
        "data.file.export_copy",
        "data.file.export_copy.executor",
        "data_export_file",
    )


def test_toolbox_projection_control_and_manifest_shapes_are_exact() -> None:
    assert tuple(member.value for member in ToolboxId) == (
        "files",
        "sources",
        "artifacts",
        "knowledge",
        "jobs",
    )
    assert tuple(member.value for member in ToolLoadMode) == (
        "pinned",
        "on_demand",
    )
    assert tuple(member.value for member in ToolTextTrust) == (
        "code",
        "admitted_untrusted",
    )
    assert tuple(field.name for field in fields(ToolboxDefinition)) == (
        "id",
        "label",
        "summary",
    )
    assert tuple(definition.id for definition in TOOLBOX_DEFINITIONS) == tuple(
        ToolboxId
    )
    assert tuple(field.name for field in fields(ToolPresentation)) == (
        "toolbox_id",
        "load_mode",
        "text_trust",
        "summary",
        "when_to_use",
        "keywords",
    )
    assert tuple(field.name for field in fields(ToolView)) == (
        "name",
        "capability_id",
        "description",
        "presentation",
        "origin_revision_digest",
    )
    limits = LoopLimits()
    assert tuple(field.name for field in fields(ToolboxManifestEntry)) == (
        "toolbox_id",
        "label",
        "summary",
        "pinned_count",
        "on_demand_count",
    )
    assert tuple(field.name for field in fields(RunToolCatalogEntry)) == (
        "view",
        "capability",
        "domain_owner_id",
        "executor_id",
        "input_schema_digest",
        "origin_revision_digest",
        "toolbox_id",
        "load_mode",
    )
    assert tuple(field.name for field in fields(StepToolProjection)) == (
        "run_id",
        "registry_digest",
        "catalog_digest",
        "transcript_digest",
        "projection_digest",
        "activation_digest",
        "provider_definitions",
        "catalog_entries",
        "callable_entries",
        "loaded_entries",
        "loaded_definition_bytes",
    )
    controls = _control_definitions(limits)
    assert tuple(control.name for control in controls) == (
        "toolbox_search",
        "toolbox_load",
    )


async def test_toolbox_projection_uses_exact_pinned_and_on_demand_paths(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "rows.csv").write_text("id\n1\n", encoding="utf-8")
    agent = await Agent.create("phase0-invocation-mode", root=tmp_path)
    try:
        source = await agent.attach(LocalDirectorySource(source_root, name="Files"))
        runtime = agent._embedded._capability_runtime
        run = RunInput(
            id="run-phase0-invocation-mode",
            agent_id=agent.id,
            message="Characterize current invocation modes.",
            created_at=_NOW,
            conversation_id="conversation-phase0-invocation-mode",
            source_id=source.id,
            conversation_source_id=source.id,
        )
        catalog = await runtime.prepare_run(run)
        modes = {entry.view.name: entry.load_mode for entry in catalog.entries}
        projection = runtime.project(catalog, ())
        visible_names = {
            definition.name for definition in projection.provider_definitions
        }

        assert modes["data_read_file"] is ToolLoadMode.PINNED
        assert modes["artifact_create_document"] is ToolLoadMode.ON_DEMAND
        assert modes["start_data_profile"] is ToolLoadMode.ON_DEMAND
        assert "data_read_file" in visible_names
        assert "artifact_create_document" not in visible_names
        assert "start_data_profile" not in visible_names
        assert {"toolbox_search", "toolbox_load"} <= visible_names
    finally:
        await agent.close()


def test_mcp_codec_persists_the_exact_toolbox_presentation_fields() -> None:
    binding = MCPServerBinding(
        binding_id="mcp-binding-" + "a" * 32,
        agent_id="agent-phase0-mcp",
        endpoint="https://phase0.example.test/mcp",
        authentication=MCPAuthentication.no_auth(),
        protocol_version="2025-11-25",
        server_name="phase0-server",
        server_version="1.0.0",
        local_label="Phase 0",
        maximum_outbound_sensitivity=ModelSensitivity.INTERNAL,
        tools=(
            MCPToolBinding(
                capability_id="mcp.phase0.lookup",
                executor_id="mcp.phase0.executor",
                local_name="phase0_lookup",
                remote_name="lookup",
                description="Look up one Phase 0 value.",
                presentation=ToolPresentation(
                    toolbox_id=ToolboxId.SOURCES,
                    load_mode=ToolLoadMode.ON_DEMAND,
                    text_trust=ToolTextTrust.ADMITTED_UNTRUSTED,
                    summary="Look up one Phase 0 value.",
                    when_to_use="Use for one admitted remote lookup.",
                    keywords=("lookup", "remote"),
                ),
                input_schema=FrozenJsonObject.from_mapping(
                    {"type": "object", "properties": {}}
                ),
                input_schema_digest="sha256:" + "1" * 64,
                output_schema=None,
                output_schema_digest=None,
                result_sensitivity=ModelSensitivity.INTERNAL,
            ),
        ),
        state=MCPBindingState.ACTIVE,
        revision=1,
        admitted_at=_NOW,
        last_checked_at=_NOW,
    )

    encoded = json.loads(encode_mcp_binding(binding))
    assert encoded["__record__"] == "MCPServerBinding"
    assert set(encoded["fields"]) == {
        "admitted_at",
        "authentication_mode",
        "endpoint",
        "last_checked_at",
        "local_label",
        "maximum_outbound_sensitivity",
        "protocol_version",
        "revision",
        "revoked_at",
        "secret_reference",
        "server_name",
        "server_version",
        "stale_reason",
        "state",
        "tools",
        "version",
    }
    tool = encoded["fields"]["tools"][0]
    assert tool["__record__"] == "MCPToolBinding"
    assert set(tool["fields"]) == {
        "capability_id",
        "description",
        "executor_id",
        "input_schema",
        "input_schema_digest",
        "local_name",
        "output_schema",
        "output_schema_digest",
        "presentation_keywords",
        "presentation_load_mode",
        "presentation_summary",
        "presentation_text_trust",
        "presentation_toolbox_id",
        "presentation_when_to_use",
        "remote_name",
        "result_sensitivity",
    }
    assert tool["fields"]["presentation_toolbox_id"] == "sources"
    assert tool["fields"]["presentation_load_mode"] == "on_demand"
    assert tool["fields"]["presentation_text_trust"] == "admitted_untrusted"


async def test_artifact_create_and_save_local_remain_create_new_only(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    (downloads / "phase0.txt").write_text("existing\n", encoding="utf-8")
    artifact_id = "artifact-" + "0" * 31 + "1"
    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="load-artifact-tools",
                        name="toolbox_load",
                        arguments={
                            "tool_names": [
                                "artifact_create_document",
                                "artifact_save_local",
                            ]
                        },
                    ),
                ),
            ),
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="create",
                        name="artifact_create_document",
                        arguments={
                            "format": "txt",
                            "filename": "phase0.txt",
                            "content": "new artifact bytes\n",
                        },
                    ),
                ),
            ),
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="save",
                        name="artifact_save_local",
                        arguments={
                            "artifact_id": artifact_id,
                            "destination_id": "default",
                        },
                    ),
                ),
            ),
            _stop("created and saved"),
        ),
        provider_id="mock:phase0-artifact",
    )
    agent = await Agent.create(
        "phase0-artifact",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        downloads_directory=downloads,
    )
    try:
        result = await agent.run("Create and save a text artifact.")
        assert result.artifacts[0].artifact_id == artifact_id
        assert len(result.artifact_deliveries) == 1
        receipt = result.artifact_deliveries[0]
        payload = await agent.read_artifact(artifact_id)
        assert receipt.artifact_id == artifact_id
        assert receipt.destination_id == SYSTEM_DOWNLOADS_DESTINATION_ID
        assert receipt.filename == "phase0 (1).txt"
        assert receipt.byte_size == payload.ref.byte_size
        assert receipt.sha256 == payload.ref.sha256
        assert receipt.renamed_for_collision is True
        assert (downloads / "phase0.txt").read_text(encoding="utf-8") == ("existing\n")
        assert Path(receipt.saved_path).read_text(encoding="utf-8") == (
            "new artifact bytes\n"
        )
        assert set(artifact_delivery_receipt_to_mapping(receipt)) == {
            "artifact_id",
            "destination_id",
            "filename",
            "saved_path",
            "byte_size",
            "sha256",
            "renamed_for_collision",
            "delivered_at",
        }

        _, save_capability = agent._embedded._capabilities.resolve_tool(
            "artifact_save_local"
        )
        assert save_capability.id == ARTIFACT_SAVE_LOCAL_CAPABILITY_ID
        assert FrozenJsonObject.from_mapping(
            save_capability.input_schema
        ).to_dict() == {
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
                "filename": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 120,
                },
            },
            "required": ["artifact_id", "destination_id"],
            "additionalProperties": False,
        }
        properties = save_capability.input_schema["properties"]
        assert isinstance(properties, Mapping)
        assert "mode" not in properties
        assert tuple(field.name for field in fields(ArtifactDeliveryReceipt)) == (
            "artifact_id",
            "destination_id",
            "filename",
            "saved_path",
            "byte_size",
            "sha256",
            "renamed_for_collision",
            "delivered_at",
        )
        assert tuple(inspect.signature(Agent.save_artifact).parameters) == (
            "self",
            "artifact_id",
            "destination",
            "filename",
        )
    finally:
        await agent.close()


async def test_artifact_destination_admission_rejects_an_attached_source_root(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads-admission"
    source_root = tmp_path / "source-admission"
    downloads.mkdir()
    source_root.mkdir()
    (source_root / "rows.csv").write_text("id\n1\n", encoding="utf-8")
    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="load-artifact-create",
                        name="toolbox_load",
                        arguments={"tool_names": ["artifact_create_document"]},
                    ),
                ),
            ),
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="create",
                        name="artifact_create_document",
                        arguments={
                            "format": "txt",
                            "filename": "admission.txt",
                            "content": "admitted artifact\n",
                        },
                    ),
                ),
            ),
            _stop("created"),
        ),
        provider_id="mock:phase0-destination-admission",
    )
    agent = await Agent.create(
        "phase0-destination-admission",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        downloads_directory=downloads,
    )
    try:
        result = await agent.run("Create one artifact.")
        await agent.attach(LocalDirectorySource(source_root, name="Source"))

        with pytest.raises(ArtifactError) as failure:
            await agent.save_artifact(result.artifacts[0].artifact_id, source_root)
        assert failure.value.code == "artifact_destination_unauthorized"
        assert tuple(source_root.iterdir()) == (source_root / "rows.csv",)
    finally:
        await agent.close()
