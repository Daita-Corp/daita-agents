from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import math
from pathlib import Path
import sqlite3
import time

import pytest

from daita import Agent
from daita._json import canonical_json
from daita.capabilities import (
    AccessMode,
    Capability,
    CapabilityRegistry,
    EvidenceCandidate,
    ExecutionRequest,
    RiskLevel,
    ToolView,
)
from daita.catalog import (
    CatalogResource,
    CatalogResourceRevision,
    CatalogSearchRequest,
    CatalogSync,
    CatalogSyncStatus,
    ResourceKind,
    Sensitivity,
    SourceCatalogSnapshot,
    catalog_resource_id,
)
from daita.hosting import AgentHost
from daita.identity import AgentIdentity
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
)
from daita.loop.models import LoopExitKind, Readiness, Turn
from daita.monitors import (
    IntervalSchedule,
    MonitorDefinition,
    MonitorRunStatus,
    MonitorScope,
)
from daita.operations.checkpoints import OperationSnapshot
from daita.operations.models import (
    ActionProposal,
    Evidence,
    Observation,
    OperationStatus,
)
from daita.storage.blobs import BlobPut, LocalBlobStore
from daita.storage.sqlite import SQLiteOperationStore

pytestmark = pytest.mark.performance

NOW = datetime(2026, 7, 19, 17, 0, tzinfo=timezone.utc)
LOOP_OPERATIONS = 8
LOOP_BATCH_CEILING_SECONDS = 8.0
LOOP_P95_CEILING_SECONDS = 2.0
OBSERVATION_CEILING_BYTES = 2_048
SQLITE_GROWTH_CEILING_BYTES = 16 * 1_024 * 1_024
BLOB_LOGICAL_WRITES = 32
BLOB_GROWTH_CEILING_BYTES = 128 * 1_024
CATALOG_RESOURCES = 1_000
CATALOG_COMMIT_CEILING_SECONDS = 8.0
CATALOG_SEARCH_P95_CEILING_SECONDS = 1.0
CONTENDED_WRITES = 32
CONTENTION_CEILING_SECONDS = 10.0
MONITOR_RUNS = 12
MONITOR_BATCH_CEILING_SECONDS = 10.0
MONITOR_P95_CEILING_SECONDS = 2.0


def _p95(samples: list[float]) -> float:
    assert samples
    return sorted(samples)[math.ceil(len(samples) * 0.95) - 1]


def _record_seconds(
    record_testsuite_property: Callable[[str, object], None],
    name: str,
    value: float,
) -> None:
    record_testsuite_property(name, round(value, 6))


def _state_file_bytes(home: Path) -> int:
    return sum(
        path.stat().st_size
        for path in (
            home / "state.db",
            home / "state.db-wal",
            home / "state.db-shm",
        )
        if path.is_file()
    )


def _baseline_capability() -> Capability:
    return Capability(
        id="baseline.read",
        owner="phase9-performance",
        description="Return one bounded deterministic baseline observation.",
        input_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
        output_evidence_kind="baseline.read.result",
        output_schema_version=1,
        output_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "padding": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["key", "padding", "value"],
            "additionalProperties": False,
        },
        executor_id="baseline.read.executor",
        access_mode=AccessMode.READ,
        risk=RiskLevel.LOW,
        side_effecting=False,
        idempotent=True,
        replay_safe=True,
    )


class _BaselineExecutor:
    executor_id = "baseline.read.executor"

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        key = request.arguments["key"]
        assert isinstance(key, str)
        return EvidenceCandidate(
            kind="baseline.read.result",
            schema_version=1,
            payload={
                "key": key,
                "padding": "x" * 512,
                "value": key.upper(),
            },
        )


class _BaselineContext:
    async def build(
        self,
        operation: OperationSnapshot,
        turn: Turn,
        tools: tuple[ToolDefinition, ...],
    ) -> ModelRequest:
        assert tuple(tool.name for tool in tools) == ("read_baseline_value",)
        messages: tuple[CanonicalMessage, ...]
        if not operation.model_calls:
            messages = (
                CanonicalMessage(
                    agent_id=operation.operation.agent_id,
                    operation_id=operation.operation.id,
                    turn_id=turn.id,
                    role=MessageRole.USER,
                    content=(TextBlock("Read the bounded baseline value."),),
                ),
            )
        else:
            first = operation.model_calls[0]
            assert first.response is not None
            projected = [
                *first.request.messages,
                CanonicalMessage(
                    agent_id=operation.operation.agent_id,
                    operation_id=operation.operation.id,
                    turn_id=first.turn_id,
                    role=MessageRole.ASSISTANT,
                    tool_calls=first.response.tool_calls,
                ),
            ]
            for observation in operation.observations:
                assert observation.task_id is not None
                task = next(
                    task for task in operation.tasks if task.id == observation.task_id
                )
                projected.append(
                    CanonicalMessage(
                        agent_id=operation.operation.agent_id,
                        operation_id=operation.operation.id,
                        turn_id=first.turn_id,
                        role=MessageRole.TOOL,
                        content=(
                            ToolResultBlock(
                                call_id=task.call_id,
                                output=observation.payload,
                            ),
                        ),
                    )
                )
            messages = tuple(projected)
        return ModelRequest(
            operation_id=operation.operation.id,
            turn_id=turn.id,
            messages=messages,
            tools=tools,
        )


class _BaselineDomain:
    def __init__(self, registry: CapabilityRegistry) -> None:
        self._registry = registry

    async def tool_views(
        self,
        operation: OperationSnapshot,
    ) -> tuple[ToolDefinition, ...]:
        assert operation.operation.status is OperationStatus.RUNNING
        return self._registry.tool_definitions()

    async def validate_action(
        self,
        call: ToolCall,
        operation: OperationSnapshot,
    ) -> ActionProposal:
        view, capability = self._registry.resolve_tool(call.name)
        return ActionProposal(
            operation_id=operation.operation.id,
            turn_id=operation.turns[-1].id,
            call_id=call.id,
            capability_id=view.capability_id,
            arguments=self._registry.validate_arguments(
                capability.id,
                call.arguments,
            ),
            proposed_at=operation.operation.updated_at,
        )

    async def project_observation(self, evidence: Evidence) -> Observation:
        return Observation(
            operation_id=evidence.operation_id,
            turn_id=evidence.turn_id,
            code="baseline.read.succeeded",
            message="The bounded baseline read completed.",
            payload=evidence.payload,
            success=True,
            task_id=evidence.task_id,
            evidence_id=evidence.id,
            created_at=evidence.created_at,
        )

    async def evaluate_final_answer(
        self,
        text: str,
        operation: OperationSnapshot,
    ) -> Readiness:
        assert text == "The bounded baseline evidence is ready."
        assert operation.evidence
        return Readiness(
            allowed=True,
            code="ready.baseline",
            message="The accepted baseline evidence is cited by the operation.",
            evaluated_at=operation.operation.updated_at,
        )


class _BaselineProvider:
    provider_id = "mock:phase9-baseline"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self._calls_by_operation: dict[str, int] = {}

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return True

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        position = self._calls_by_operation.get(request.operation_id, 0) + 1
        self._calls_by_operation[request.operation_id] = position
        if position == 1:
            return ModelResponse(
                tool_calls=(
                    ToolCall(
                        id=f"call-{len(self.requests)}",
                        name="read_baseline_value",
                        arguments={"key": request.operation_id[-12:]},
                    ),
                ),
                finish_reason=FinishReason.TOOL_CALLS,
                usage=ModelUsage(input_tokens=12, output_tokens=4),
            )
        if position == 2:
            return ModelResponse(
                text="The bounded baseline evidence is ready.",
                finish_reason=FinishReason.STOP,
                usage=ModelUsage(input_tokens=18, output_tokens=6),
            )
        raise AssertionError("a baseline operation exceeded two model calls")


async def test_loop_persistence_and_projection_baseline(
    tmp_path: Path,
    record_testsuite_property: Callable[[str, object], None],
) -> None:
    capability = _baseline_capability()
    registry = CapabilityRegistry(
        capabilities=(capability,),
        executors=(_BaselineExecutor(),),
        tool_views=(
            ToolView(
                name="read_baseline_value",
                capability_id=capability.id,
                description="Read one bounded deterministic value.",
            ),
        ),
    )
    provider = _BaselineProvider()
    agent = await Agent.create(
        "loop-baseline",
        root=tmp_path,
        model=provider,
        context_builder=_BaselineContext(),
        domain=_BaselineDomain(registry),
        capabilities=registry,
    )
    initial_bytes = _state_file_bytes(agent.home)
    latencies: list[float] = []
    observation_sizes: list[int] = []
    try:
        batch_started = time.perf_counter()
        for index in range(LOOP_OPERATIONS):
            started = time.perf_counter()
            result = await agent.run(f"Baseline operation {index}.")
            latencies.append(time.perf_counter() - started)
            assert result.kind is LoopExitKind.COMPLETED
            snapshot = await agent.inspect(result.operation_id)
            assert len(snapshot.model_calls) == 2
            assert snapshot.loop_state.input_tokens == 30
            assert snapshot.loop_state.output_tokens == 10
            assert len(snapshot.observations) == 1
            observation_sizes.append(
                len(canonical_json(snapshot.observations[0].payload).encode("utf-8"))
            )
        batch_seconds = time.perf_counter() - batch_started
        final_bytes = _state_file_bytes(agent.home)
        state_growth = max(0, final_bytes - initial_bytes)

        assert len(provider.requests) == LOOP_OPERATIONS * 2
        assert batch_seconds <= LOOP_BATCH_CEILING_SECONDS
        assert _p95(latencies) <= LOOP_P95_CEILING_SECONDS
        assert max(observation_sizes) <= OBSERVATION_CEILING_BYTES
        assert state_growth <= SQLITE_GROWTH_CEILING_BYTES
        with sqlite3.connect(agent.home / "state.db") as connection:
            assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)

        _record_seconds(
            record_testsuite_property,
            "phase9.loop.batch_seconds",
            batch_seconds,
        )
        _record_seconds(
            record_testsuite_property,
            "phase9.loop.p95_seconds",
            _p95(latencies),
        )
        record_testsuite_property(
            "phase9.loop.model_calls",
            len(provider.requests),
        )
        record_testsuite_property("phase9.loop.input_tokens_per_operation", 30)
        record_testsuite_property("phase9.loop.output_tokens_per_operation", 10)
        record_testsuite_property(
            "phase9.loop.max_observation_bytes",
            max(observation_sizes),
        )
        record_testsuite_property(
            "phase9.sqlite.incremental_bytes",
            state_growth,
        )
    finally:
        await agent.close()


async def test_content_addressed_blob_growth_baseline(
    tmp_path: Path,
    record_testsuite_property: Callable[[str, object], None],
) -> None:
    root = tmp_path / "blobs"
    store = LocalBlobStore(root)
    content = b"b" * 4_096
    digest = "sha256:" + sha256(content).hexdigest()
    started = time.perf_counter()
    metadata = await asyncio.gather(
        *(
            store.put(
                BlobPut(
                    blob_id=f"baseline-blob-{index:03d}",
                    media_type="application/octet-stream",
                    created_at=NOW,
                    sensitivity_class="internal",
                    retention_class="operation",
                    expected_digest=digest,
                ),
                content,
            )
            for index in range(BLOB_LOGICAL_WRITES)
        )
    )
    elapsed = time.perf_counter() - started

    objects = tuple(path for path in (root / "sha256").rglob("*") if path.is_file())
    stored_bytes = sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )
    assert len(metadata) == BLOB_LOGICAL_WRITES
    assert {item.digest for item in metadata} == {digest}
    assert len(objects) == 1
    assert objects[0].read_bytes() == content
    assert stored_bytes <= BLOB_GROWTH_CEILING_BYTES
    assert elapsed <= 4.0

    _record_seconds(
        record_testsuite_property,
        "phase9.blob.batch_seconds",
        elapsed,
    )
    record_testsuite_property("phase9.blob.logical_writes", len(metadata))
    record_testsuite_property("phase9.blob.physical_objects", len(objects))
    record_testsuite_property("phase9.blob.stored_bytes", stored_bytes)


def _catalog_snapshot(
    *,
    agent_id: str,
    source_id: str,
    sync_id: str,
    names: tuple[str, ...],
) -> SourceCatalogSnapshot:
    revisions: list[CatalogResourceRevision] = []
    resources: list[CatalogResource] = []
    for name in names:
        resource_id = catalog_resource_id(source_id, ResourceKind.TABLE, name)
        revision = CatalogResourceRevision.build(
            resource_id=resource_id,
            sync_id=sync_id,
            observed_at=NOW,
            source_revision=f"schema:{sync_id}",
        )
        revisions.append(revision)
        resources.append(
            CatalogResource.build(
                agent_id=agent_id,
                source_id=source_id,
                native_identity=name,
                external_uri=f"baseline://{source_id}/{name}",
                kind=ResourceKind.TABLE,
                name=name,
                sensitivity=Sensitivity.INTERNAL,
                revision=revision,
                first_observed_at=NOW,
                last_observed_at=NOW,
            )
        )
    return SourceCatalogSnapshot(
        sync=CatalogSync(
            id=sync_id,
            agent_id=agent_id,
            source_id=source_id,
            adapter_id="phase9-baseline",
            status=CatalogSyncStatus.SUCCEEDED,
            started_at=NOW,
            completed_at=NOW,
            source_revision=f"schema:{sync_id}",
            resource_count=len(resources),
        ),
        resources=tuple(resources),
        revisions=tuple(revisions),
    )


async def test_catalog_scale_and_multi_connection_contention_baseline(
    tmp_path: Path,
    record_testsuite_property: Callable[[str, object], None],
) -> None:
    path = tmp_path / "catalog.db"
    store = await SQLiteOperationStore.open(path)
    agent_id = "agent-phase9-baseline"
    await store.initialize_identity(
        AgentIdentity(
            id=agent_id,
            display_name="Phase 9 baseline",
            created_at=NOW,
        )
    )
    additional_stores: list[SQLiteOperationStore] = []
    try:
        names = tuple(f"orders{index:04d}" for index in range(CATALOG_RESOURCES))
        snapshot = _catalog_snapshot(
            agent_id=agent_id,
            source_id="source-scale",
            sync_id="sync-scale",
            names=names,
        )
        commit_started = time.perf_counter()
        await store.commit_snapshot(snapshot)
        commit_seconds = time.perf_counter() - commit_started
        assert commit_seconds <= CATALOG_COMMIT_CEILING_SECONDS

        search_latencies: list[float] = []
        for index in range(0, CATALOG_RESOURCES, 25):
            query = names[index]
            search_started = time.perf_counter()
            result = await store.search(
                CatalogSearchRequest(agent_id=agent_id, query=query, limit=5)
            )
            search_latencies.append(time.perf_counter() - search_started)
            assert result.total_matches == 1
            assert tuple(hit.name for hit in result.hits) == (query,)
        assert _p95(search_latencies) <= CATALOG_SEARCH_P95_CEILING_SECONDS
        assert len(await store.list_resources(agent_id, "source-scale")) == (
            CATALOG_RESOURCES
        )

        additional_stores = [await SQLiteOperationStore.open(path) for _ in range(3)]
        writers = (store, *additional_stores)
        contended_snapshots = tuple(
            _catalog_snapshot(
                agent_id=agent_id,
                source_id=f"source-contention-{index:03d}",
                sync_id=f"sync-contention-{index:03d}",
                names=(f"contended{index:03d}",),
            )
            for index in range(CONTENDED_WRITES)
        )
        contention_started = time.perf_counter()
        committed = await asyncio.gather(
            *(
                writers[index % len(writers)].commit_snapshot(item)
                for index, item in enumerate(contended_snapshots)
            )
        )
        contention_seconds = time.perf_counter() - contention_started
        assert len(committed) == CONTENDED_WRITES
        assert contention_seconds <= CONTENTION_CEILING_SECONDS
        assert len(await store.list_resources(agent_id)) == (
            CATALOG_RESOURCES + CONTENDED_WRITES
        )

        _record_seconds(
            record_testsuite_property,
            "phase9.catalog.commit_seconds",
            commit_seconds,
        )
        _record_seconds(
            record_testsuite_property,
            "phase9.catalog.search_p95_seconds",
            _p95(search_latencies),
        )
        record_testsuite_property("phase9.catalog.resources", CATALOG_RESOURCES)
        _record_seconds(
            record_testsuite_property,
            "phase9.sqlite.contention_seconds",
            contention_seconds,
        )
        record_testsuite_property(
            "phase9.sqlite.contended_writes",
            len(committed),
        )
    finally:
        for additional in additional_stores:
            await additional.close()
        await store.close()


class _TextContext:
    async def build(
        self,
        operation: OperationSnapshot,
        turn: Turn,
        tools: tuple[ToolDefinition, ...],
    ) -> ModelRequest:
        assert tools == ()
        message = operation.trigger.payload["message"]
        assert isinstance(message, str)
        return ModelRequest(
            operation_id=operation.operation.id,
            turn_id=turn.id,
            messages=(
                CanonicalMessage(
                    agent_id=operation.operation.agent_id,
                    operation_id=operation.operation.id,
                    turn_id=turn.id,
                    role=MessageRole.USER,
                    content=(TextBlock(message),),
                ),
            ),
        )


class _TextDomain:
    async def tool_views(
        self,
        operation: OperationSnapshot,
    ) -> tuple[ToolDefinition, ...]:
        return ()

    async def validate_action(
        self,
        call: ToolCall,
        operation: OperationSnapshot,
    ) -> ActionProposal:
        raise AssertionError("text-only monitor baseline has no actions")

    async def project_observation(self, evidence: Evidence) -> Observation:
        raise AssertionError("text-only monitor baseline has no observations")

    async def evaluate_final_answer(
        self,
        text: str,
        operation: OperationSnapshot,
    ) -> Readiness:
        assert text == "Monitor completed."
        return Readiness(
            allowed=True,
            code="ready.monitor_baseline",
            message="The deterministic monitor run completed.",
            evaluated_at=operation.operation.updated_at,
        )


class _MonitorProvider:
    provider_id = "mock:phase9-monitor-baseline"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return True

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            text="Monitor completed.",
            finish_reason=FinishReason.STOP,
            usage=ModelUsage(input_tokens=7, output_tokens=3),
        )


async def test_monitor_reliability_and_idempotent_replay_baseline(
    tmp_path: Path,
    record_testsuite_property: Callable[[str, object], None],
) -> None:
    provider = _MonitorProvider()
    host = await AgentHost.create(
        "monitor-baseline",
        root=tmp_path,
        model=provider,
        context_builder=_TextContext(),
        domain=_TextDomain(),
        cadence_seconds=3_600,
        clock=lambda: NOW,
    )
    await host.start()
    run_latencies: list[float] = []
    try:
        for index in range(MONITOR_RUNS):
            definition = MonitorDefinition(
                name=f"Baseline monitor {index:02d}",
                objective=f"Run deterministic baseline monitor {index:02d}.",
                scope=MonitorScope(),
                schedule=IntervalSchedule(
                    interval_seconds=300,
                    anchor_at=NOW + timedelta(days=1),
                ),
            )
            proposal = await host.propose_monitor(
                f"monitor-{index:02d}",
                definition,
                idempotency_key=f"create-{index:02d}",
            )
            await host.confirm_monitor(
                proposal.id,
                candidate_hash=proposal.candidate_hash,
                actor_id="phase9-baseline",
                reason="Enable the deterministic reliability baseline.",
            )

        batch_started = time.perf_counter()
        first_results = []
        for index in range(MONITOR_RUNS):
            started = time.perf_counter()
            result = await host.run_monitor_now(
                f"monitor-{index:02d}",
                idempotency_key=f"run-{index:02d}",
            )
            run_latencies.append(time.perf_counter() - started)
            first_results.append(result)
        batch_seconds = time.perf_counter() - batch_started

        calls_after_first_pass = len(provider.requests)
        replayed = [
            await host.run_monitor_now(
                f"monitor-{index:02d}",
                idempotency_key=f"run-{index:02d}",
            )
            for index in range(MONITOR_RUNS)
        ]

        assert all(
            result.run_status is MonitorRunStatus.SUCCEEDED for result in first_results
        )
        assert all(
            result.run_status is MonitorRunStatus.SUCCEEDED for result in replayed
        )
        assert all(not result.claimed for result in replayed)
        assert len(provider.requests) == calls_after_first_pass == MONITOR_RUNS
        assert batch_seconds <= MONITOR_BATCH_CEILING_SECONDS
        assert _p95(run_latencies) <= MONITOR_P95_CEILING_SECONDS
        for index in range(MONITOR_RUNS):
            inspection = await host.inspect_monitor(f"monitor-{index:02d}")
            assert len(inspection.runs) == 1
            assert inspection.runs[0].status is MonitorRunStatus.SUCCEEDED

        _record_seconds(
            record_testsuite_property,
            "phase9.monitor.batch_seconds",
            batch_seconds,
        )
        _record_seconds(
            record_testsuite_property,
            "phase9.monitor.p95_seconds",
            _p95(run_latencies),
        )
        record_testsuite_property("phase9.monitor.attempted", MONITOR_RUNS)
        record_testsuite_property(
            "phase9.monitor.succeeded",
            len(first_results),
        )
        record_testsuite_property("phase9.monitor.replay_model_calls", 0)
    finally:
        await host.stop()
