from __future__ import annotations

import argparse
import asyncio
import threading
from collections import defaultdict
from collections.abc import Mapping
from copy import copy
from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest

import daita.adapters.local_files as local_files_module
from daita import Agent, ArtifactError, LocalDirectorySource, cli
from daita.artifacts.models import ArtifactAuthorship
from daita.capabilities import CapabilityInputError, CapabilityRegistry
from daita.domains.data.export_capabilities import (
    ARTIFACT_SAVE_LOCAL_TOOL_NAME,
    LOCAL_FILE_COPY_CAPABILITY_ID,
    LOCAL_FILE_COPY_TOOL_NAME,
    LocalFileCopyExecutor,
    artifact_extension_declarations,
)
from daita.hosting.embedded import EmbeddedAgent
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider

_ARTIFACT_ID = "artifact-00000000000000000000000000000001"


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


def _tools(*calls: ToolCall) -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.TOOL_CALLS, tool_calls=calls)


def _stop(text: str = "done") -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text)


async def _agent_with_local_files(
    tmp_path: Path,
    name: str,
    files: Mapping[str, bytes],
) -> tuple[Agent, MockModelProvider, str, dict[str, str], Path, Path]:
    source_root = tmp_path / f"{name}-source"
    downloads = tmp_path / f"{name}-downloads"
    source_root.mkdir()
    downloads.mkdir()
    for filename, content in files.items():
        (source_root / filename).write_bytes(content)
    provider = MockModelProvider((), provider_id=f"mock:{name}")
    agent = await Agent.create(
        name,
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        downloads_directory=downloads,
    )
    source = await agent.attach(LocalDirectorySource(source_root))
    resources = {
        resource.name: resource.id
        for resource in await agent.list_catalog_resources(source_id=source.id)
        if resource.name in files
    }
    assert set(resources) == set(files)
    return agent, provider, source.id, resources, source_root, downloads


def _copy_script(
    source_id: str,
    resource_id: str,
    *,
    deliver: bool,
    filename: str | None = None,
) -> tuple[ModelResponse, ...]:
    arguments = {"source_id": source_id, "resource_id": resource_id}
    if filename is not None:
        arguments["filename"] = filename
    responses = [
        _tools(
            ToolCall(
                id="copy",
                name=LOCAL_FILE_COPY_TOOL_NAME,
                arguments=arguments,
            )
        )
    ]
    if deliver:
        responses.append(
            _tools(
                ToolCall(
                    id="save",
                    name=ARTIFACT_SAVE_LOCAL_TOOL_NAME,
                    arguments={
                        "artifact_id": _ARTIFACT_ID,
                        "destination_id": "default",
                    },
                )
            )
        )
    responses.append(_stop())
    return tuple(responses)


def _set_script(provider: MockModelProvider, script: tuple[ModelResponse, ...]) -> None:
    provider._script = script
    provider._cursor = 0


async def _tool_result(agent: Agent, run_id: str, call_id: str) -> ToolResultBlock:
    transcript = await agent.transcript(run_id)
    return next(
        block
        for message in transcript.messages
        for block in message.content
        if isinstance(block, ToolResultBlock) and block.call_id == call_id
    )


def _error_code(block: ToolResultBlock) -> str:
    error = block.output.get("error")
    assert isinstance(error, Mapping)
    code = error.get("code")
    assert isinstance(code, str)
    return code


def _internal_run_directories(agent: Agent) -> tuple[Path, ...]:
    root = agent.home / "artifacts"
    return tuple(
        item
        for item in root.iterdir()
        if item.name not in {".staging", "delivery-config.json"}
    )


@pytest.mark.parametrize(
    ("filename", "content", "media_type"),
    (
        (
            "source.csv",
            b'\xef\xbb\xbf"id","note"\r\n1,"comma, quote ""and"" CRLF\r\n"\r\n',
            "text/csv",
        ),
        (
            "source.json",
            b'[\n  {"id": 1, "note": "spacing stays exact"}\n]\n',
            "application/json",
        ),
    ),
)
async def test_cataloged_csv_and_json_downloads_are_byte_identical(
    tmp_path: Path,
    filename: str,
    content: bytes,
    media_type: str,
) -> None:
    agent, provider, source_id, resources, _source_root, downloads = (
        await _agent_with_local_files(
            tmp_path,
            f"byte-copy-{filename.removesuffix(Path(filename).suffix)}",
            {filename: content},
        )
    )
    _set_script(
        provider,
        _copy_script(source_id, resources[filename], deliver=True),
    )
    try:
        result = await agent.run(
            f"Download the attached {Path(filename).suffix.removeprefix('.')} file."
        )
        copy_result = await _tool_result(agent, result.run_id, "copy")
        assert not copy_result.is_error, copy_result.output
        assert len(result.artifacts) == 1
        assert len(result.artifact_deliveries) == 1
        ref = result.artifacts[0]
        receipt = result.artifact_deliveries[0]
        assert ref.artifact_id == _ARTIFACT_ID
        assert ref.media_type == media_type
        assert ref.byte_size == len(content)
        assert ref.sha256 == "sha256:" + sha256(content).hexdigest()
        assert ref.provenance.authorship is ArtifactAuthorship.EXACT_SOURCE_DATA
        assert len(ref.provenance.resource_bindings) == 1
        assert ref.provenance.resource_bindings[0].source_id == source_id
        assert ref.provenance.resource_bindings[0].resource_id == resources[filename]
        assert ref.provenance.sql_fingerprint is None
        assert ref.provenance.parameters_sha256 is None
        assert (await agent.read_artifact(ref.artifact_id)).content == content
        assert Path(receipt.saved_path).read_bytes() == content
        assert Path(receipt.saved_path).parent == downloads
        assert "content" not in copy_result.output
        assert "rows" not in copy_result.output
    finally:
        await agent.close()


def test_local_file_copy_is_one_fixed_catalog_id_only_capability() -> None:
    declarations = artifact_extension_declarations()
    copies = tuple(
        capability
        for capability in declarations.capabilities
        if capability.id == LOCAL_FILE_COPY_CAPABILITY_ID
    )
    assert len(copies) == 1
    capability = copies[0]
    properties = capability.input_schema.get("properties")
    assert isinstance(properties, Mapping)
    assert set(properties) == {
        "source_id",
        "resource_id",
        "filename",
    }
    assert capability.input_schema["additionalProperties"] is False
    assert capability.artifact_policy is not None
    assert capability.artifact_policy.allowed_media_types == frozenset(
        {"text/csv", "application/json"}
    )

    class _UnusedExecutor:
        executor_id = capability.executor_id

        async def execute(self, request):
            raise AssertionError(request)

    view = next(
        item for item in declarations.tool_views if item.capability_id == capability.id
    )
    registry = CapabilityRegistry(
        capabilities=(capability,),
        executors=(_UnusedExecutor(),),
        tool_views=(view,),
    )
    for forbidden in (
        "path",
        "content",
        "bytes",
        "revision",
        "provenance",
        "sensitivity",
        "destination",
        "overwrite",
        "format",
    ):
        with pytest.raises(CapabilityInputError):
            registry.validate_arguments(
                LOCAL_FILE_COPY_CAPABILITY_ID,
                {
                    "source_id": "source",
                    "resource_id": "resource",
                    forbidden: "untrusted",
                },
            )


async def test_changed_source_is_rejected_without_artifact_or_copy(
    tmp_path: Path,
) -> None:
    original = b"id,name\n1,Ada\n"
    agent, provider, source_id, resources, source_root, downloads = (
        await _agent_with_local_files(
            tmp_path,
            "changed-copy",
            {"records.csv": original},
        )
    )
    (source_root / "records.csv").write_bytes(b"id,name\n1,Eve\n")
    _set_script(
        provider,
        _copy_script(source_id, resources["records.csv"], deliver=False),
    )
    try:
        result = await agent.run("Download the attached CSV file.")
        assert result.artifacts == ()
        assert result.artifact_deliveries == ()
        assert _error_code(await _tool_result(agent, result.run_id, "copy")) == (
            "artifact_incomplete_export"
        )
        assert not tuple(downloads.iterdir())
        assert _internal_run_directories(agent) == ()
    finally:
        await agent.close()


@pytest.mark.parametrize("attack", ("escape", "symlink"))
async def test_copy_rejects_path_escape_and_symlink_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    content = b"id,name\n1,Ada\n"
    agent, provider, source_id, resources, source_root, downloads = (
        await _agent_with_local_files(
            tmp_path,
            f"{attack}-copy",
            {"records.csv": content},
        )
    )
    resource_id = resources["records.csv"]
    outside = tmp_path / f"{attack}-outside.csv"
    outside.write_bytes(b"id,name\n9,Outside\n")
    if attack == "symlink":
        (source_root / "records.csv").unlink()
        (source_root / "records.csv").symlink_to(outside)
    else:
        store = agent._embedded._store
        original_load = store.load_resource

        async def escaped_load(agent_id: str, selected_resource_id: str):
            resource = await original_load(agent_id, selected_resource_id)
            if resource is not None and selected_resource_id == resource_id:
                escaped = copy(resource)
                object.__setattr__(
                    escaped,
                    "native_identity",
                    "../escape-outside.csv",
                )
                return escaped
            return resource

        monkeypatch.setattr(store, "load_resource", escaped_load)
    _set_script(provider, _copy_script(source_id, resource_id, deliver=False))
    try:
        result = await agent.run("Download this attached CSV file.")
        assert result.artifacts == ()
        assert result.artifact_deliveries == ()
        assert (await _tool_result(agent, result.run_id, "copy")).is_error
        assert outside.read_bytes() == b"id,name\n9,Outside\n"
        assert not tuple(downloads.iterdir())
        assert _internal_run_directories(agent) == ()
    finally:
        await agent.close()


@pytest.mark.parametrize("limit_kind", ("bytes", "time"))
async def test_copy_byte_and_time_limits_fail_without_partial_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_kind: str,
) -> None:
    content = b"id,name\n1,Ada\n"
    agent, provider, source_id, resources, _source_root, downloads = (
        await _agent_with_local_files(
            tmp_path,
            f"{limit_kind}-limit-copy",
            {"records.csv": content},
        )
    )
    _capability, executor = agent._embedded._capabilities.resolve_execution(
        LOCAL_FILE_COPY_CAPABILITY_ID
    )
    executor = cast(LocalFileCopyExecutor, executor)
    if limit_kind == "bytes":
        monkeypatch.setattr(executor, "_max_bytes", len(content) - 1)
    else:
        monkeypatch.setattr(executor, "_max_seconds", 0.001)

        async def never_finishes(**_kwargs):
            await asyncio.Event().wait()

        monkeypatch.setattr(executor._backend, "execute_copy", never_finishes)
    _set_script(
        provider,
        _copy_script(source_id, resources["records.csv"], deliver=False),
    )
    try:
        result = await agent.run("Download this attached CSV file.")
        assert result.artifacts == ()
        assert result.artifact_deliveries == ()
        block = await _tool_result(agent, result.run_id, "copy")
        assert _error_code(block) == "artifact_incomplete_export"
        assert not tuple(downloads.iterdir())
        assert _internal_run_directories(agent) == ()
        assert not tuple((agent.home / "artifacts" / ".staging").iterdir())
    finally:
        await agent.close()


async def test_copy_cancellation_cleans_worker_and_internal_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"id,name\n1,Ada\n"
    agent, provider, source_id, resources, _source_root, downloads = (
        await _agent_with_local_files(
            tmp_path,
            "cancel-copy",
            {"records.csv": content},
        )
    )
    started = threading.Event()
    release = threading.Event()
    original_read = local_files_module._read_registered_file

    def blocked_read(*args):
        cancellation = args[-1]
        started.set()
        while not release.wait(0.01):
            if cancellation.is_set():
                raise RuntimeError("cancelled")
        return original_read(*args)

    monkeypatch.setattr(local_files_module, "_read_registered_file", blocked_read)
    _set_script(
        provider,
        _copy_script(source_id, resources["records.csv"], deliver=False),
    )
    try:
        running = asyncio.create_task(agent.run("Download this attached CSV file."))
        assert await asyncio.to_thread(started.wait, 2)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert not tuple(downloads.iterdir())
        assert _internal_run_directories(agent) == ()
        assert not tuple((agent.home / "artifacts" / ".staging").iterdir())
    finally:
        release.set()
        await agent.close()


async def test_ordinary_file_preview_never_creates_or_delivers_an_artifact(
    tmp_path: Path,
) -> None:
    agent, provider, source_id, resources, _source_root, downloads = (
        await _agent_with_local_files(
            tmp_path,
            "ordinary-preview",
            {"records.csv": b"id,name\n1,Ada\n"},
        )
    )
    _set_script(
        provider,
        (
            _tools(
                ToolCall(
                    id="preview",
                    name="data_read_file",
                    arguments={
                        "source_id": source_id,
                        "resource_id": resources["records.csv"],
                    },
                )
            ),
            _stop(),
        ),
    )
    try:
        result = await agent.run("Read and summarize this attached CSV file.")
        assert result.artifacts == ()
        assert result.artifact_deliveries == ()
        assert not tuple(downloads.iterdir())
        assert _internal_run_directories(agent) == ()
    finally:
        await agent.close()


async def test_delivery_failure_is_truthful_and_retains_known_artifact(
    tmp_path: Path,
) -> None:
    content = b"id,name\n1,Ada\n"
    agent, provider, source_id, resources, _source_root, downloads = (
        await _agent_with_local_files(
            tmp_path,
            "truthful-copy-failure",
            {"records.csv": content},
        )
    )
    downloads.rmdir()
    _set_script(
        provider,
        _copy_script(source_id, resources["records.csv"], deliver=True),
    )
    try:
        result = await agent.run("Download this attached CSV file.")
        assert len(result.artifacts) == 1
        assert result.artifact_deliveries == ()
        assert (await agent.read_artifact(_ARTIFACT_ID)).content == content
        save_result = await _tool_result(agent, result.run_id, "save")
        assert save_result.is_error
        assert _error_code(save_result) == "artifact_downloads_unavailable"
        error = save_result.output.get("error")
        assert isinstance(error, Mapping)
        assert "saved_path" not in repr(error)
    finally:
        await agent.close()


async def test_known_file_copy_id_saves_after_restart_without_rerunning_source(
    tmp_path: Path,
) -> None:
    content = b'[ { "id": 1 } ]\n'
    agent, provider, source_id, resources, source_root, downloads = (
        await _agent_with_local_files(
            tmp_path,
            "restart-copy",
            {"records.json": content},
        )
    )
    _set_script(
        provider,
        _copy_script(source_id, resources["records.json"], deliver=True),
    )
    result = await agent.run("Download this attached JSON file.")
    artifact_id = result.artifacts[0].artifact_id
    await agent.close()
    (source_root / "records.json").write_bytes(b"changed after artifact creation")

    reopened = await Agent.open(
        "restart-copy",
        root=tmp_path,
        downloads_directory=downloads,
    )
    try:
        assert (await reopened.read_artifact(artifact_id)).content == content
        receipt = await reopened.save_artifact(artifact_id)
        assert receipt.filename == "records (1).json"
        assert Path(receipt.saved_path).read_bytes() == content
    finally:
        await reopened.close()


async def test_conversation_clear_invalidates_known_id_but_preserves_delivered_copy(
    tmp_path: Path,
) -> None:
    content = b"id,name\n1,Ada\n"
    agent, provider, source_id, resources, _source_root, _downloads = (
        await _agent_with_local_files(
            tmp_path,
            "clear-copy",
            {"records.csv": content},
        )
    )
    _set_script(
        provider,
        _copy_script(source_id, resources["records.csv"], deliver=True),
    )
    try:
        result = await agent.run("Download this attached CSV file.")
        artifact_id = result.artifacts[0].artifact_id
        delivered = Path(result.artifact_deliveries[0].saved_path)
        assert await agent.clear_conversations() == 1
        with pytest.raises(ArtifactError) as failure:
            await agent.read_artifact(artifact_id)
        assert failure.value.code == "artifact_missing"
        assert delivered.read_bytes() == content
    finally:
        await agent.close()


def test_no_public_artifact_listing_surface_exists() -> None:
    assert "list_artifacts" not in Agent.__dict__
    assert "list_artifacts" not in EmbeddedAgent.__dict__
    parser = cli.build_parser()
    commands = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    artifacts = commands.choices["artifacts"]
    artifact_commands = next(
        action
        for action in artifacts._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert set(artifact_commands.choices) == {"save"}
    assert "list or save run artifacts" not in parser.format_help()
    with pytest.raises(SystemExit):
        parser.parse_args(["artifacts", "list", "agent"])
