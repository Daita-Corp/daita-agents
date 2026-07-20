from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import errno
import hashlib
import os
from pathlib import Path
import sqlite3
import stat
import tempfile
from typing import cast

import pytest
from daita._json import canonical_json
from daita.capabilities import AccessMode, RiskLevel
from daita.hosting.host import AgentHost, AgentHostState
from daita.hosting.local_protocol import (
    LocalAgentClient,
    LocalErrorResponse,
    LocalRequest,
    LocalSocketSecurityError,
    LocalSuccessResponse,
    encode_frame,
    encode_request,
)
from daita.hosting.local_server import (
    LocalAgentServer,
    _approval_projection,
    _task_governance_projection,
)
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolDefinition,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import Readiness, Turn
from daita.operations.checkpoints import OperationSnapshot
from daita.operations.governance import ApprovalRequest
from daita.operations.models import (
    ActionProposal,
    ActionValidationFacts,
    Evidence,
    Observation,
    Task,
    TaskExecutionFacts,
    TaskStatus,
)

NOW = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc)


def test_local_governance_projection_exposes_safe_facts_and_approval_hashes() -> None:
    arguments = {"source_id": "source-sqlite", "value": "redacted"}
    validation = ActionValidationFacts(
        schema_version=1,
        source_id="source-sqlite",
        resource_ids=("resource-marker",),
        resource_revisions=(("resource-marker", "sha256:" + ("a" * 64)),),
        source_revision="sqlite:data-version:2",
        impact={"affected_rows": 1, "bounded": True},
        evidence_ids=("evidence-impact",),
    )
    task = Task(
        id="task-write",
        operation_id="operation-write",
        turn_id="turn-write",
        call_id="call-write",
        capability_id="data.sqlite.update",
        executor_id="data.sqlite.update.executor",
        status=TaskStatus.WAITING_FOR_APPROVAL,
        attempt=1,
        arguments=arguments,
        execution_facts=TaskExecutionFacts(
            capability_fingerprint="sha256:" + ("b" * 64),
            arguments_hash="sha256:"
            + hashlib.sha256(canonical_json(arguments).encode("utf-8")).hexdigest(),
            access_mode=AccessMode.WRITE,
            risk=RiskLevel.HIGH,
            side_effecting=True,
            idempotent=True,
            replay_safe=True,
            idempotency_key="operation-write:task-write",
            validation_facts=validation,
        ),
        created_at=NOW,
        updated_at=NOW,
    )
    governance = _task_governance_projection(task, actor_id="user:actor-local")

    assert governance["actor_id"] == "user:actor-local"
    assert governance["access_mode"] == "write"
    assert governance["risk"] == "high"
    projected_validation = governance["validation"]
    assert isinstance(projected_validation, dict)
    assert projected_validation["source_id"] == "source-sqlite"
    assert projected_validation["validation_fingerprint"] == validation.fingerprint
    assert "arguments" not in governance
    assert "redacted" not in str(governance)

    approval = ApprovalRequest(
        id="approval-write",
        operation_id=task.operation_id,
        task_id=task.id,
        task_fingerprint="sha256:" + ("c" * 64),
        policy_fingerprint="sha256:" + ("d" * 64),
        requested_at=NOW,
    )
    projected_approval = _approval_projection(approval)
    assert projected_approval["task_fingerprint"] == approval.task_fingerprint
    assert projected_approval["policy_fingerprint"] == approval.policy_fingerprint


@pytest.fixture
def short_root():
    project_root = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory(prefix=".p6-", dir=project_root) as value:
        yield Path(value)


class _Clock:
    def __call__(self) -> datetime:
        return NOW


class _MemoryWriter:
    def __init__(self) -> None:
        self.data = bytearray()
        self.closed = asyncio.Event()

    def write(self, value: bytes) -> None:
        self.data.extend(value)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed.set()

    def is_closing(self) -> bool:
        return self.closed.is_set()

    async def wait_closed(self) -> None:
        await self.closed.wait()


def _stream_writer(value: _MemoryWriter) -> asyncio.StreamWriter:
    return cast(asyncio.StreamWriter, value)


def _feed_request(reader: asyncio.StreamReader, request: LocalRequest) -> None:
    reader.feed_data(encode_frame(encode_request(request)))


class _Context:
    async def build(
        self,
        operation: OperationSnapshot,
        turn: Turn,
        tools: tuple[ToolDefinition, ...],
    ) -> ModelRequest:
        message = operation.trigger.payload["message"]
        assert isinstance(message, str)
        return ModelRequest(
            operation_id=operation.operation.id,
            turn_id=turn.id,
            messages=(
                CanonicalMessage(
                    agent_id=operation.operation.agent_id,
                    operation_id=operation.operation.id,
                    session_id=operation.operation.session_id,
                    turn_id=turn.id,
                    role=MessageRole.USER,
                    content=(TextBlock(message),),
                ),
            ),
            tools=tools,
        )


class _Domain:
    def tool_views(
        self,
        operation: OperationSnapshot,
    ) -> tuple[ToolDefinition, ...]:
        return ()

    async def validate_action(
        self,
        call: ToolCall,
        operation: OperationSnapshot,
    ) -> ActionProposal:
        raise AssertionError("text-only journey cannot validate an action")

    async def project_observation(self, evidence: Evidence) -> Observation:
        raise AssertionError("text-only journey cannot project evidence")

    async def evaluate_final_answer(
        self,
        text: str,
        operation: OperationSnapshot,
    ) -> Readiness:
        return Readiness(
            allowed=True,
            code="ready.text",
            message="Text response is ready.",
            evaluated_at=NOW,
        )


async def _request(
    server: LocalAgentServer,
    request_id: str,
    method: str,
    *,
    params: dict[str, object] | None = None,
    idempotency_key: str | None = None,
):
    return await server.dispatch(
        LocalRequest.create(
            request_id=request_id,
            method=method,
            params=params,
            idempotency_key=idempotency_key,
        )
    )


def _result_object(response: LocalSuccessResponse) -> dict[str, object]:
    result = response.to_wire()["result"]
    assert isinstance(result, dict)
    assert all(isinstance(key, str) for key in result)
    return result


def _object(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    assert all(isinstance(key, str) for key in value)
    return value


def _object_list(value: object) -> list[dict[str, object]]:
    assert isinstance(value, list)
    return [_object(item) for item in value]


async def test_local_dispatch_journey_is_durable_strict_and_bounded(
    short_root,
) -> None:
    provider = MockModelProvider(
        (
            ModelResponse(
                text="chat complete",
                finish_reason=FinishReason.STOP,
            ),
            ModelResponse(
                text="monitor complete",
                finish_reason=FinishReason.STOP,
            ),
        ),
        provider_id="mock:local-server",
    )
    host = await AgentHost.create(
        "atlas",
        root=short_root,
        model=provider,
        context_builder=_Context(),
        domain=_Domain(),
        cadence_seconds=3_600,
        clock=_Clock(),
    )
    server = LocalAgentServer(host)
    assert not server.started
    assert not server.socket_path.exists()

    await host.start()
    try:
        assert not server.started
        assert host.state is AgentHostState.RUNNING

        health = await _request(server, "health-1", "host.health")
        assert isinstance(health, LocalSuccessResponse)
        health_value = _result_object(health)
        assert health_value["healthy"] is True
        assert health_value["configured"] is True

        model = await _request(server, "model-1", "model.status")
        assert isinstance(model, LocalSuccessResponse)
        assert model.to_wire()["result"] == {
            "configured": True,
            "profile": None,
        }

        submitted = await _request(
            server,
            "chat-1",
            "chat.submit",
            params={"message": "Hello from the local client."},
            idempotency_key="chat-1",
        )
        assert isinstance(submitted, LocalSuccessResponse)
        inbox = _result_object(submitted)
        assert inbox["status"] == "completed"
        operation_id = inbox["operation_id"]
        assert isinstance(operation_id, str)

        replay = await _request(
            server,
            "chat-2",
            "chat.submit",
            params={"message": "Hello from the local client."},
            idempotency_key="chat-1",
        )
        assert isinstance(replay, LocalSuccessResponse)
        assert replay.to_wire()["result"] == inbox
        assert len(provider.requests) == 1

        inspected = await _request(
            server,
            "inspect-1",
            "operation.inspect",
            params={"operation_id": operation_id},
        )
        assert isinstance(inspected, LocalSuccessResponse)
        operation = _object(_result_object(inspected)["operation"])
        assert operation["status"] == "succeeded"
        assert operation["final_text"] == "chat complete"

        operations = await _request(server, "operations-1", "operation.list")
        assert isinstance(operations, LocalSuccessResponse)
        listed_operations = _object_list(_result_object(operations)["operations"])
        assert [value["id"] for value in listed_operations] == [operation_id]

        approvals = await _request(server, "approvals-1", "approval.list")
        assert isinstance(approvals, LocalSuccessResponse)
        assert _result_object(approvals)["approvals"] == []

        memories = await _request(server, "memories-1", "memory.list")
        assert isinstance(memories, LocalSuccessResponse)
        assert _result_object(memories)["items"] == []

        skills = await _request(server, "skills-1", "skill.list")
        assert isinstance(skills, LocalSuccessResponse)
        assert _result_object(skills)["skills"] == []

        definition = {
            "name": "Backlog check",
            "objective": "Inspect the current backlog.",
            "scope": {"source_ids": [], "resource_ids": []},
            "schedule": {
                "kind": "interval",
                "interval_seconds": 60,
                "anchor_at": NOW.isoformat(),
            },
        }
        proposed = await _request(
            server,
            "monitor-propose-1",
            "monitor.propose",
            params={"monitor_id": "monitor-backlog", "definition": definition},
            idempotency_key="monitor-proposal-1",
        )
        assert isinstance(proposed, LocalSuccessResponse)
        proposal = _result_object(proposed)

        confirmed = await _request(
            server,
            "monitor-confirm-1",
            "monitor.confirm",
            params={
                "proposal_id": proposal["id"],
                "candidate_hash": proposal["candidate_hash"],
                "actor_id": "reviewer-1",
                "reason": "Enable the reviewed definition.",
            },
            idempotency_key="monitor-confirmation-1",
        )
        assert isinstance(confirmed, LocalSuccessResponse)
        confirmed_monitor = _object(_result_object(confirmed)["monitor"])
        assert confirmed_monitor["status"] == "enabled"

        confirmed_replay = await _request(
            server,
            "monitor-confirm-2",
            "monitor.confirm",
            params={
                "proposal_id": proposal["id"],
                "candidate_hash": proposal["candidate_hash"],
                "actor_id": "reviewer-1",
                "reason": "Enable the reviewed definition.",
            },
            idempotency_key="monitor-confirmation-1",
        )
        assert isinstance(confirmed_replay, LocalSuccessResponse)
        assert confirmed_replay.result == confirmed.result

        reused_confirmation_key = await _request(
            server,
            "monitor-key-conflict",
            "monitor.pause",
            params={
                "monitor_id": "monitor-backlog",
                "actor_id": "reviewer-1",
                "reason": "Do not apply.",
            },
            idempotency_key="monitor-confirmation-1",
        )
        assert isinstance(reused_confirmation_key, LocalErrorResponse)
        assert reused_confirmation_key.error.code == "state_conflict"

        listed = await _request(server, "monitor-list-1", "monitor.list")
        assert isinstance(listed, LocalSuccessResponse)
        listed_monitors = _object_list(_result_object(listed)["monitors"])
        assert [item["id"] for item in listed_monitors] == ["monitor-backlog"]

        run_now = await _request(
            server,
            "monitor-run-1",
            "monitor.run_now",
            params={"monitor_id": "monitor-backlog", "lease_seconds": 30.0},
            idempotency_key="monitor-run-1",
        )
        assert isinstance(run_now, LocalSuccessResponse)
        assert _result_object(run_now)["run_status"] == "succeeded"
        assert len(provider.requests) == 2

        missing_key = await _request(
            server,
            "error-key",
            "chat.submit",
            params={"message": "Do not run."},
        )
        assert isinstance(missing_key, LocalErrorResponse)
        assert missing_key.error.code == "idempotency_required"

        unknown_param = await _request(
            server,
            "error-param",
            "host.health",
            params={"verbose": True},
        )
        assert isinstance(unknown_param, LocalErrorResponse)
        assert unknown_param.error.code == "invalid_params"

        unknown_method = await _request(server, "error-method", "host.secret")
        assert isinstance(unknown_method, LocalErrorResponse)
        assert unknown_method.error.code == "method_not_found"
        assert "traceback" not in str(unknown_method.to_wire()).lower()
    finally:
        await server.stop(drain=True)

    assert host.state is AgentHostState.STOPPED
    assert not server.socket_path.exists()


async def test_source_attach_uses_default_host_capability_owners(short_root) -> None:
    source_root = short_root / "source-data"
    source_root.mkdir()
    (source_root / "rows.csv").write_text(
        "id,value\n1,alpha\n",
        encoding="utf-8",
    )
    host = await AgentHost.create(
        "source-atlas",
        root=short_root,
        cadence_seconds=3_600,
        clock=_Clock(),
    )
    server = LocalAgentServer(host)
    await host.start()
    try:
        source = await _request(
            server,
            "source-1",
            "source.attach",
            params={"kind": "local_files", "path": str(source_root)},
            idempotency_key="source-1",
        )
        assert isinstance(source, LocalSuccessResponse)
        source_projection = _result_object(source)
        assert source_projection["adapter_id"] == "local-directory"
        source_id = source_projection["id"]
        assert isinstance(source_id, str)

        listed = await _request(server, "source-list", "source.list")
        assert isinstance(listed, LocalSuccessResponse)
        assert [
            item["id"] for item in _object_list(_result_object(listed)["sources"])
        ] == [source_id]

        health = await _request(
            server,
            "source-health",
            "source.health",
            params={"source_id": source_id},
        )
        assert isinstance(health, LocalSuccessResponse)
        health_items = _object_list(_result_object(health)["health"])
        assert health_items[0]["healthy"] is True
        catalog_resource_count = health_items[0]["catalog_resource_count"]
        assert isinstance(catalog_resource_count, int)
        assert catalog_resource_count >= 1

        searched = await _request(
            server,
            "catalog-search",
            "catalog.search",
            params={"query": "rows", "source_ids": [source_id]},
        )
        assert isinstance(searched, LocalSuccessResponse)
        hits = _object_list(_result_object(searched)["hits"])
        assert hits
        shown = await _request(
            server,
            "catalog-show",
            "catalog.show",
            params={"resource_id": hits[0]["resource_id"]},
        )
        assert isinstance(shown, LocalSuccessResponse)
        assert _object(_result_object(shown)["resource"])["source_id"] == source_id

        natural = await _request(
            server,
            "natural-monitor",
            "monitor.propose_natural",
            params={
                "monitor_id": "rows-check",
                "request": f"Every 5 minutes, Inspect rows for source {source_id}",
            },
            idempotency_key="natural-monitor-1",
        )
        assert isinstance(natural, LocalSuccessResponse)
        natural_projection = _result_object(natural)
        candidate = _object(natural_projection["candidate"])
        assert _object(candidate["scope"])["source_ids"] == [source_id]

        replay = await _request(
            server,
            "source-2",
            "source.attach",
            params={"kind": "local_files", "path": str(source_root)},
            idempotency_key="source-1",
        )
        assert isinstance(replay, LocalSuccessResponse)
        assert replay.result == source.result

        other_root = short_root / "other-source-data"
        other_root.mkdir()
        conflict = await _request(
            server,
            "source-3",
            "source.attach",
            params={"kind": "local_files", "path": str(other_root)},
            idempotency_key="source-1",
        )
        assert isinstance(conflict, LocalErrorResponse)
        assert conflict.error.code == "state_conflict"

        with sqlite3.connect(host.home / "state.db") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM catalog_syncs"
            ).fetchone() == (1,)

        detached = await _request(
            server,
            "source-detach",
            "source.detach",
            params={"source_id": source_id},
            idempotency_key="source-detach-1",
        )
        assert isinstance(detached, LocalSuccessResponse)
        assert _result_object(detached)["active"] is False
    finally:
        await server.stop(drain=True)


async def test_source_attach_persists_explicit_sqlite_write_admission(
    short_root,
) -> None:
    database = short_root / "controlled-write.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE records (id TEXT PRIMARY KEY, status TEXT NOT NULL)"
        )
    host = await AgentHost.create(
        "source-write-admission",
        root=short_root,
        cadence_seconds=3_600,
        clock=_Clock(),
    )
    server = LocalAgentServer(host)
    await host.start()
    try:
        attached = await _request(
            server,
            "source-write-1",
            "source.attach",
            params={
                "kind": "sqlite",
                "path": str(database),
                "write_access": True,
            },
            idempotency_key="source-write-1",
        )
        assert isinstance(attached, LocalSuccessResponse)
        projection = _result_object(attached)
        assert projection["adapter_id"] == "sqlite"
        assert projection["write_access"] is True

        source_id = projection["id"]
        assert isinstance(source_id, str)
        with sqlite3.connect(host.home / "state.db") as connection:
            row = connection.execute(
                "SELECT configuration_json FROM attached_sources WHERE id = ?",
                (source_id,),
            ).fetchone()
        assert row is not None
        assert '"write_access":true' in str(row[0])

        directory = short_root / "not-writable-source"
        directory.mkdir()
        rejected = await _request(
            server,
            "source-write-invalid",
            "source.attach",
            params={
                "kind": "local_files",
                "path": str(directory),
                "write_access": True,
            },
            idempotency_key="source-write-invalid",
        )
        assert isinstance(rejected, LocalErrorResponse)
        assert rejected.error.code == "invalid_params"
    finally:
        await server.stop(drain=True)


async def test_private_socket_lifecycle_and_health_when_sandbox_allows_bind(
    short_root,
) -> None:
    host = await AgentHost.create(
        "socket-smoke",
        root=short_root,
        cadence_seconds=3_600,
        clock=_Clock(),
    )
    server = LocalAgentServer(host)
    try:
        await server.start()
    except LocalSocketSecurityError as error:
        cause = error.__cause__
        await server.stop(drain=False)
        if isinstance(cause, PermissionError) and cause.errno == errno.EPERM:
            pytest.skip("sandbox forbids binding Unix-domain sockets")
        raise

    try:
        run_mode = stat.S_IMODE(os.lstat(server.socket_path.parent).st_mode)
        socket_mode = stat.S_IMODE(os.lstat(server.socket_path).st_mode)
        assert run_mode == 0o700
        assert socket_mode == 0o600

        client = LocalAgentClient(host.home)
        response = await client.request(
            LocalRequest.create(
                request_id="socket-health-1",
                method="host.health",
            )
        )
        assert isinstance(response, LocalSuccessResponse)
        assert _result_object(response)["healthy"] is True
    finally:
        await server.stop(drain=True)

    assert not server.socket_path.exists()
    assert host.state is AgentHostState.STOPPED


async def test_partial_frame_has_a_deadline_and_is_reaped(short_root) -> None:
    host = await AgentHost.create(
        "partial-frame",
        root=short_root,
        cadence_seconds=3_600,
        clock=_Clock(),
    )
    server = LocalAgentServer(host, request_read_timeout_seconds=0.01)
    reader = asyncio.StreamReader()
    reader.feed_data(b"\x00\x00")
    writer = _MemoryWriter()
    await host.start()
    server._accept_connection(reader, _stream_writer(writer))

    try:
        await asyncio.wait_for(writer.closed.wait(), timeout=0.5)
        assert b"request_timeout" in writer.data
        assert server.active_connections == 0
    finally:
        await server.stop(drain=False)


async def test_drain_waits_for_inflight_request_before_host_shutdown(
    short_root,
    monkeypatch,
) -> None:
    host = await AgentHost.create(
        "drain-request",
        root=short_root,
        cadence_seconds=3_600,
        clock=_Clock(),
    )
    server = LocalAgentServer(host)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow_dispatch(request: LocalRequest):
        entered.set()
        await release.wait()
        return LocalSuccessResponse.create(
            request_id=request.request_id,
            result={"drained": True},
        )

    monkeypatch.setattr(server, "dispatch", slow_dispatch)
    reader = asyncio.StreamReader()
    _feed_request(
        reader,
        LocalRequest.create(request_id="drain-1", method="host.health"),
    )
    writer = _MemoryWriter()
    await host.start()
    server._accept_connection(reader, _stream_writer(writer))
    await asyncio.wait_for(entered.wait(), timeout=0.5)

    stopping = asyncio.create_task(server.stop(drain=True))
    await asyncio.sleep(0)
    assert not stopping.done()
    assert host.state is AgentHostState.RUNNING
    release.set()
    await asyncio.wait_for(stopping, timeout=0.5)

    assert host.state is AgentHostState.STOPPED
    assert server.active_connections == 0


async def test_nondraining_stop_cancels_inflight_request_before_host_shutdown(
    short_root,
    monkeypatch,
) -> None:
    host = await AgentHost.create(
        "cancel-request",
        root=short_root,
        cadence_seconds=3_600,
        clock=_Clock(),
    )
    server = LocalAgentServer(host)
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow_dispatch(request: LocalRequest):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return LocalSuccessResponse.create(request_id=request.request_id)

    monkeypatch.setattr(server, "dispatch", slow_dispatch)
    reader = asyncio.StreamReader()
    _feed_request(
        reader,
        LocalRequest.create(request_id="cancel-1", method="host.health"),
    )
    writer = _MemoryWriter()
    await host.start()
    server._accept_connection(reader, _stream_writer(writer))
    await asyncio.wait_for(entered.wait(), timeout=0.5)

    await asyncio.wait_for(server.stop(drain=False), timeout=0.5)

    assert cancelled.is_set()
    assert host.state is AgentHostState.STOPPED
    assert server.active_connections == 0
