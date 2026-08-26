from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from _capability_runtime_support import ContextToolProjectionAdapter

import daita.artifacts.store as store_module
import daita.capabilities as capabilities_module
from daita import Agent
from daita._json import canonical_json
from daita.artifacts.models import (
    ArtifactAuthorship,
    ArtifactDraft,
    ArtifactError,
    ArtifactProvenance,
    ArtifactRef,
    artifact_ref_from_mapping,
    artifact_ref_to_mapping,
)
from daita.artifacts.renderers import DOCUMENT_ALLOWED_EXTENSIONS, render_model_document
from daita.artifacts.store import AgentHomeArtifactStore
from daita.capabilities import ArtifactPolicy, Capability, ToolOutput
from daita.capability_runtime import CapabilityRuntime
from daita.catalog.models import Sensitivity
from daita.domains.data.export_capabilities import _resolved_sensitivity
from daita.domains.data.file_capabilities import LocalFileReadExecutor
from daita.llm.errors import ModelProviderError, ProviderErrorCode
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
)
from _toolbox_model_support import (
    ToolboxAwareMockModelProvider as MockModelProvider,
)
from daita.loop import AgentLoop, InMemoryTranscriptStore, ToolBatchOutcome
from daita.loop.models import LoopExitKind, RunInput

NOW = datetime(2026, 8, 1, 12, tzinfo=UTC)
RUN_ID = "run-00000000000000000000000000000001"
CONVERSATION_ID = "conversation-00000000000000000000000000000001"


class _References:
    def __init__(self, refs: tuple[ArtifactRef, ...] = ()) -> None:
        self.refs = refs

    async def list_artifact_refs(
        self,
        agent_id: str,
        *,
        run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> tuple[ArtifactRef, ...]:
        del agent_id
        return tuple(
            item
            for item in self.refs
            if (run_id is None or item.run_id == run_id)
            and (conversation_id is None or item.conversation_id == conversation_id)
        )


class _Catalog:
    async def source_routing_facts(self, *args, **kwargs):
        del args, kwargs
        return ()


def _artifact_ids():
    counter = 0

    def create(prefix: str) -> str:
        nonlocal counter
        assert prefix == "artifact"
        counter += 1
        return f"artifact-{counter:032x}"

    return create


def _policy(*, maximum: int = 256 * 1024) -> ArtifactPolicy:
    return ArtifactPolicy(
        allowed_media_types=frozenset({"text/markdown", "text/plain"}),
        allowed_extensions=DOCUMENT_ALLOWED_EXTENSIONS,
        artifact_required=True,
        max_artifact_count=1,
        max_bytes_per_artifact=maximum,
        max_total_bytes_per_call=maximum,
    )


def _draft(
    content: bytes = b"# Result\n",
    *,
    filename: str = "result.md",
    media_type: str = "text/markdown",
) -> ArtifactDraft:
    return ArtifactDraft(
        content=content,
        suggested_filename=filename,
        media_type=media_type,
        sensitivity=Sensitivity.INTERNAL,
        provenance=ArtifactProvenance(
            authorship=ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS
        ),
    )


async def _store(
    tmp_path: Path,
    *,
    refs: _References | None = None,
    agent_id: str = "agent-one",
) -> tuple[AgentHomeArtifactStore, _References]:
    home = tmp_path / agent_id
    home.mkdir(parents=True)
    references = refs or _References()
    store = await AgentHomeArtifactStore.open(
        agent_id=agent_id,
        agent_home=home,
        references=references,
        clock=lambda: NOW,
        id_factory=_artifact_ids(),
    )
    return store, references


async def _commit(
    store: AgentHomeArtifactStore,
    draft: ArtifactDraft | None = None,
    *,
    run_id: str = RUN_ID,
    call_id: str = "create",
    policy: ArtifactPolicy | None = None,
) -> ArtifactRef:
    return await store.commit(
        draft or _draft(),
        policy or _policy(),
        run_id=run_id,
        conversation_id=CONVERSATION_ID,
        call_id=call_id,
        capability_id="artifact.create_document",
    )


async def test_artifact_store_open_cancellation_finishes_admission_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "cancelled-admission"
    home.mkdir()
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    original = AgentHomeArtifactStore._admit_and_cleanup

    def blocked_cleanup(
        store: AgentHomeArtifactStore,
        refs: tuple[ArtifactRef, ...],
        reservations: frozenset[tuple[str, str]],
    ) -> None:
        started.set()
        assert release.wait(2)
        try:
            original(store, refs, reservations)
        finally:
            finished.set()

    monkeypatch.setattr(AgentHomeArtifactStore, "_admit_and_cleanup", blocked_cleanup)
    opening = asyncio.create_task(
        AgentHomeArtifactStore.open(
            agent_id="agent-one",
            agent_home=home,
            references=_References(),
            clock=lambda: NOW,
            id_factory=_artifact_ids(),
        )
    )
    assert await asyncio.to_thread(started.wait, 2)
    opening.cancel()
    await asyncio.sleep(0)
    assert not opening.done()
    assert not finished.is_set()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await opening
    assert finished.is_set()


async def test_artifact_store_open_preserves_admission_failure_as_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "failed-admission"
    home.mkdir()

    def fail_cleanup(
        store: AgentHomeArtifactStore,
        refs: tuple[ArtifactRef, ...],
        reservations: frozenset[tuple[str, str]],
    ) -> None:
        del store, refs, reservations
        raise ArtifactError(
            "artifact_storage_failed",
            "Injected admission failure.",
            {"stage": "admission_cleanup"},
        )

    monkeypatch.setattr(AgentHomeArtifactStore, "_admit_and_cleanup", fail_cleanup)
    store = await AgentHomeArtifactStore.open(
        agent_id="agent-one",
        agent_home=home,
        references=_References(),
        clock=lambda: NOW,
        id_factory=_artifact_ids(),
    )
    assert not store.available
    with pytest.raises(ArtifactError) as failure:
        await _commit(store)
    assert failure.value.code == "artifact_storage_failed"
    assert failure.value.details.to_dict() == {"stage": "admission_cleanup"}


def test_tool_artifact_is_removed_without_a_compatibility_alias() -> None:
    assert not hasattr(capabilities_module, "ToolArtifact")
    assert "ToolArtifact" not in capabilities_module.__all__


async def test_ordinary_local_file_read_no_longer_produces_an_implicit_artifact() -> (
    None
):
    assert "artifact" not in LocalFileReadExecutor.__annotations__
    assert "artifact" not in capabilities_module.ToolOutput.__annotations__ or (
        capabilities_module.ToolOutput.__annotations__["artifact"]
        == "ArtifactDraft | None"
    )


async def test_capability_without_artifact_policy_rejects_a_draft(
    tmp_path: Path,
) -> None:
    store, _ = await _store(tmp_path)
    runtime = CapabilityRuntime(
        capabilities_module.CapabilityRegistry(declarations=(), executors=()),
        (),
        artifacts=store,
    )
    capability = Capability(
        id="test.no_policy",
        description="test",
        input_schema={"type": "object", "properties": {}},
        output_kind="test.output",
        output_schema={"type": "object", "properties": {}},
        executor_id="test.executor",
    )
    run = RunInput(
        id=RUN_ID,
        agent_id="agent-one",
        message="create a file",
        created_at=NOW,
        conversation_id=CONVERSATION_ID,
    )
    with pytest.raises(
        capabilities_module.ToolOutputValidationError,
        match="without artifact policy",
    ):
        await runtime._commit_artifact_output(
            run,
            ToolCall(id="create", name="test"),
            capability,
            ToolOutput(kind="test.output", artifact=_draft()),
        )


def test_artifact_policy_rejects_media_extension_count_and_byte_violations() -> None:
    with pytest.raises(ValueError, match="zero or one"):
        ArtifactPolicy(
            allowed_media_types=frozenset({"text/plain"}),
            allowed_extensions=(("text/plain", (".txt",)),),
            artifact_required=True,
            max_artifact_count=2,
            max_bytes_per_artifact=1,
            max_total_bytes_per_call=1,
        )
    with pytest.raises(ArtifactError, match="extension"):
        render_model_document(
            content="text",
            format="markdown",
            filename="report.txt",
            evidence_call_ids=(),
        )
    with pytest.raises(ArtifactError) as too_large:
        render_model_document(
            content="é" * 140_000,
            format="txt",
            filename="report.txt",
            evidence_call_ids=(),
        )
    assert too_large.value.code == "artifact_quota_exceeded"


async def test_artifact_draft_bytes_never_enter_tool_result_transcript_exit_or_json(
    tmp_path: Path,
) -> None:
    store, refs = await _store(tmp_path)
    secret = b"DRAFT_BYTE_SENTINEL"
    ref = await _commit(store, _draft(secret))
    refs.refs = (ref,)
    serialized_ref = canonical_json(artifact_ref_to_mapping(ref))
    assert "DRAFT_BYTE_SENTINEL" not in serialized_ref
    assert "content" not in artifact_ref_to_mapping(ref)
    assert (await store.read(ref.artifact_id)).content == secret


async def test_pre_lineage_artifact_reference_is_admitted_and_reserialized_canonically(
    tmp_path: Path,
) -> None:
    store, _ = await _store(tmp_path)
    ref = await _commit(store)
    serialized = artifact_ref_to_mapping(ref)
    provenance = cast(dict[str, object], serialized["provenance"])
    provenance.pop("derived_from_artifact_id")

    class _PreLineageReferences:
        async def list_artifact_refs(
            self,
            agent_id: str,
            *,
            run_id: str | None = None,
            conversation_id: str | None = None,
        ) -> tuple[ArtifactRef, ...]:
            del agent_id
            decoded = artifact_ref_from_mapping(serialized)
            if run_id is not None and decoded.run_id != run_id:
                return ()
            if (
                conversation_id is not None
                and decoded.conversation_id != conversation_id
            ):
                return ()
            return (decoded,)

    reopened = await AgentHomeArtifactStore.open(
        agent_id="agent-one",
        agent_home=store.agent_home,
        references=_PreLineageReferences(),
        clock=lambda: NOW,
        id_factory=_artifact_ids(),
    )
    assert reopened.available
    payload = await reopened.read(ref.artifact_id)
    assert payload.content == b"# Result\n"
    assert payload.ref.provenance.derived_from_artifact_id is None
    canonical = artifact_ref_to_mapping(payload.ref)
    canonical_provenance = cast(dict[str, object], canonical["provenance"])
    assert canonical_provenance["derived_from_artifact_id"] is None

    invalid = dict(serialized)
    invalid["provenance"] = {**provenance, "unexpected": True}
    with pytest.raises(ValueError, match="invalid shape"):
        artifact_ref_from_mapping(invalid)


async def test_artifact_commit_returns_ref_only_after_atomic_directory_publication(
    tmp_path: Path,
) -> None:
    store, _ = await _store(tmp_path)
    ref = await _commit(store)
    final = store.root / ref.run_id / ref.artifact_id
    assert final.is_dir()
    assert (final / "manifest.json").is_file()
    assert (final / "payload").read_bytes() == b"# Result\n"
    assert not tuple(store.staging.iterdir())


async def test_artifact_commit_failure_boundaries_leave_no_dangling_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, refs = await _store(tmp_path)

    def fail_write(path: Path, content: bytes) -> None:
        del path, content
        raise OSError("injected write failure")

    monkeypatch.setattr(store_module, "_write_exclusive", fail_write)
    with pytest.raises(ArtifactError) as failure:
        await _commit(store)
    assert failure.value.code == "artifact_storage_failed"
    assert refs.refs == ()
    assert not tuple(store.staging.iterdir())
    assert not tuple(store_module._run_entries(store.root))


async def test_artifact_commit_cancellation_boundaries_leave_only_clean_staging_or_complete_orphan(
    tmp_path: Path,
) -> None:
    store, _ = await _store(tmp_path)
    task = asyncio.create_task(_commit(store))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not tuple(store.staging.iterdir())
    for run in store_module._run_entries(store.root):
        for artifact in Path(run.path).iterdir():
            assert {item.name for item in artifact.iterdir()} == {
                "manifest.json",
                "payload",
            }


async def test_complete_unreferenced_artifact_is_removed_on_next_open(
    tmp_path: Path,
) -> None:
    store, references = await _store(tmp_path)
    ref = await _commit(store)
    final = store.root / ref.run_id / ref.artifact_id
    assert final.exists()
    reopened = await AgentHomeArtifactStore.open(
        agent_id="agent-one",
        agent_home=store.agent_home,
        references=references,
        clock=lambda: NOW,
        id_factory=_artifact_ids(),
    )
    assert reopened.available
    assert not final.exists()


async def test_referenced_artifact_survives_restart_and_read_verifies_bytes(
    tmp_path: Path,
) -> None:
    store, references = await _store(tmp_path)
    ref = await _commit(store)
    references.refs = (ref,)
    reopened = await AgentHomeArtifactStore.open(
        agent_id="agent-one",
        agent_home=store.agent_home,
        references=references,
        clock=lambda: NOW,
        id_factory=_artifact_ids(),
    )
    payload = await reopened.read(ref.artifact_id)
    assert payload.ref == ref
    assert payload.content == b"# Result\n"


@pytest.mark.parametrize(
    "corruption",
    ("missing", "manifest", "size", "digest", "type"),
)
async def test_artifact_read_rejects_missing_payload_and_manifest_size_digest_or_type_corruption(
    tmp_path: Path,
    corruption: str,
) -> None:
    store, references = await _store(tmp_path / corruption)
    ref = await _commit(store)
    references.refs = (ref,)
    directory = store.root / ref.run_id / ref.artifact_id
    payload = directory / "payload"
    if corruption == "missing":
        payload.unlink()
    elif corruption == "manifest":
        (directory / "manifest.json").write_text("{}", encoding="utf-8")
    elif corruption == "size":
        payload.write_bytes(b"shorter")
    elif corruption == "digest":
        payload.write_bytes(b"x" * ref.byte_size)
    else:
        payload.unlink()
        payload.mkdir()
    with pytest.raises(ArtifactError) as failure:
        await store.read(ref.artifact_id)
    assert failure.value.code == "artifact_corrupt"


async def test_unknown_cleared_and_cross_agent_artifact_ids_are_indistinguishable_missing_errors(
    tmp_path: Path,
) -> None:
    first, first_refs = await _store(tmp_path, agent_id="agent-one")
    ref = await _commit(first)
    first_refs.refs = (ref,)
    second, _ = await _store(tmp_path, agent_id="agent-two")
    errors = []
    for store, artifact_id in (
        (first, "artifact-ffffffffffffffffffffffffffffffff"),
        (second, ref.artifact_id),
    ):
        with pytest.raises(ArtifactError) as failure:
            await store.read(artifact_id)
        errors.append((failure.value.code, failure.value.message))
    assert (
        errors[0]
        == errors[1]
        == (
            "artifact_missing",
            "The requested artifact is not available.",
        )
    )


async def test_call_run_and_agent_quotas_fail_without_silent_eviction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, references = await _store(tmp_path)
    with pytest.raises(ArtifactError) as call_limit:
        await _commit(store, _draft(b"12"), policy=_policy(maximum=1))
    assert call_limit.value.code == "artifact_quota_exceeded"
    monkeypatch.setattr(store_module, "MAX_ARTIFACTS_PER_RUN", 1)
    first = await _commit(store)
    references.refs = (first,)
    with pytest.raises(ArtifactError) as run_limit:
        await _commit(store, call_id="second")
    assert run_limit.value.details["scope"] == "run"
    assert (store.root / first.run_id / first.artifact_id).exists()


async def test_parallel_commits_cannot_oversubscribe_run_or_agent_quota(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, _ = await _store(tmp_path)
    monkeypatch.setattr(store_module, "MAX_ARTIFACTS_PER_RUN", 1)
    results = await asyncio.gather(
        _commit(store, call_id="one"),
        _commit(store, call_id="two"),
        return_exceptions=True,
    )
    assert sum(isinstance(item, ArtifactRef) for item in results) == 1
    failures = [item for item in results if isinstance(item, ArtifactError)]
    assert len(failures) == 1
    assert failures[0].code == "artifact_quota_exceeded"


def _agent_ids():
    counts: dict[str, int] = {}

    def create(prefix: str) -> str:
        counts[prefix] = counts.get(prefix, 0) + 1
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


def _document_script() -> tuple[ModelResponse, ...]:
    return (
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    id="create",
                    name="artifact_create_document",
                    arguments={
                        "format": "txt",
                        "filename": "notes.txt",
                        "content": "retained notes",
                    },
                ),
            ),
        ),
        ModelResponse(finish_reason=FinishReason.STOP, text="created"),
    )


async def test_clear_conversations_removes_internal_artifacts_but_preserves_delivery_config(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    export = tmp_path / "export"
    downloads.mkdir()
    export.mkdir()
    provider = MockModelProvider(_document_script(), provider_id="mock:artifact-clear")
    agent = await Agent.create(
        "artifact-clear",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_agent_ids(),
        downloads_directory=downloads,
    )
    try:
        await agent.set_export_destination(export)
        result = await agent.run("Create a TXT file.")
        artifact_id = result.artifacts[0].artifact_id
        assert await agent.clear_conversations() == 1
        assert (await agent.export_destination()).display_name == "export"
        assert (agent.home / "artifacts" / "delivery-config.json").is_file()
        with pytest.raises(ArtifactError) as missing:
            await agent.read_artifact(artifact_id)
        assert missing.value.code == "artifact_missing"
    finally:
        await agent.close()


async def test_clear_conversations_cancellation_never_leaves_a_persisted_dangling_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloads = tmp_path / "downloads-cancel"
    downloads.mkdir()
    provider = MockModelProvider(
        _document_script(), provider_id="mock:artifact-clear-cancel"
    )
    agent = await Agent.create(
        "artifact-clear-cancel",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_agent_ids(),
        downloads_directory=downloads,
    )
    try:
        result = await agent.run("Create a TXT file.")
        artifact_id = result.artifacts[0].artifact_id
        entered = asyncio.Event()
        release = asyncio.Event()

        async def delayed_cleanup() -> None:
            entered.set()
            await release.wait()

        monkeypatch.setattr(
            agent._embedded._artifact_store,
            "remove_all_run_artifacts",
            delayed_cleanup,
        )
        clearing = asyncio.create_task(agent.clear_conversations())
        await entered.wait()
        clearing.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await clearing
        with pytest.raises(ArtifactError) as missing:
            await agent.read_artifact(artifact_id)
        assert missing.value.code == "artifact_missing"
    finally:
        await agent.close()


class _LoopContext:
    async def prepare(self, run, messages, tool_context):
        del run
        return messages[:-1], tool_context.initial_provider_definitions

    def project(
        self,
        snapshot,
        messages,
        *,
        step,
        tool_context,
        final=False,
        previous_request_input_tokens=None,
    ):
        del step, previous_request_input_tokens, tool_context
        static, tools = snapshot
        return ModelRequest(
            messages=(*static, *messages),
            tools=() if final else tools,
        )


class _LoopTools:
    def __init__(self, results: dict[str, ToolResultBlock]) -> None:
        self.results = results
        self._projection = ContextToolProjectionAdapter(
            (
                ToolDefinition(
                    name="artifact",
                    description="artifact test",
                    input_schema={"type": "object", "properties": {}},
                ),
            )
        )

    async def prepare_run(self, run):
        return await self._projection.prepare_run(run)

    def project(self, catalog, messages):
        return self._projection.project(catalog, messages)

    async def execute_all(self, run, calls, *, projection, messages, sensitivity):
        del run, projection, messages, sensitivity
        return ToolBatchOutcome(tuple(self.results[call.id] for call in calls))


def _ref(artifact_id: str, call_id: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        run_id=RUN_ID,
        conversation_id=CONVERSATION_ID,
        call_id=call_id,
        capability_id="artifact.create_document",
        filename="result.md",
        media_type="text/markdown",
        byte_size=9,
        sha256="sha256:" + "1" * 64,
        sensitivity=Sensitivity.INTERNAL,
        provenance=ArtifactProvenance(
            authorship=ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS
        ),
        created_at=NOW,
    )


async def test_loop_exit_collects_successful_refs_in_call_order_with_failed_sibling() -> (
    None
):
    first = _ref("artifact-00000000000000000000000000000001", "first")
    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(id="first", name="artifact"),
                    ToolCall(id="failed", name="artifact"),
                ),
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="done"),
        )
    )
    loop = AgentLoop(
        model=provider,
        context_builder=_LoopContext(),
        tools=_LoopTools(
            {
                "first": ToolResultBlock(
                    call_id="first",
                    output={"artifact": artifact_ref_to_mapping(first)},
                ),
                "failed": ToolResultBlock(
                    call_id="failed",
                    output={"error": {"code": "failed", "message": "failed"}},
                    is_error=True,
                ),
            }
        ),
        clock=lambda: NOW,
    )
    result = await loop.run(
        RunInput(
            id=RUN_ID,
            agent_id="agent-one",
            message="create",
            created_at=NOW,
            conversation_id=CONVERSATION_ID,
        )
    )
    assert result.artifacts == (first,)


async def test_failed_or_interrupted_run_retains_only_refs_already_appended_to_transcript() -> (
    None
):
    first = _ref("artifact-00000000000000000000000000000001", "first")
    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(ToolCall(id="first", name="artifact"),),
            ),
            ModelProviderError(ProviderErrorCode.INVALID_REQUEST),
        )
    )
    transcripts = InMemoryTranscriptStore()
    loop = AgentLoop(
        model=provider,
        context_builder=_LoopContext(),
        tools=_LoopTools(
            {
                "first": ToolResultBlock(
                    call_id="first",
                    output={"artifact": artifact_ref_to_mapping(first)},
                )
            }
        ),
        transcripts=transcripts,
        clock=lambda: NOW,
    )
    result = await loop.run(
        RunInput(
            id=RUN_ID,
            agent_id="agent-one",
            message="create",
            created_at=NOW,
            conversation_id=CONVERSATION_ID,
        )
    )
    assert result.kind is LoopExitKind.FAILED
    assert result.artifacts == (first,)
    persisted = await transcripts.result(RUN_ID)
    assert persisted is not None and persisted.artifacts == (first,)


def test_provenance_and_sensitivity_are_runtime_bound_and_cannot_be_lowered_by_model_arguments() -> (
    None
):
    assert (
        _resolved_sensitivity((Sensitivity.PUBLIC, Sensitivity.UNKNOWN))
        is Sensitivity.RESTRICTED
    )
    with pytest.raises(ValueError, match="cannot claim exact export facts"):
        ArtifactProvenance(
            authorship=ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS,
            row_count=0,
        )


async def test_artifact_store_rejects_path_components_symlinks_and_entries_outside_agent_home(
    tmp_path: Path,
) -> None:
    store, _ = await _store(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    store.staging.rmdir()
    store.staging.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ArtifactError) as failure:
        await _commit(store)
    assert failure.value.code == "artifact_storage_failed"
    assert not tuple(outside.iterdir())
