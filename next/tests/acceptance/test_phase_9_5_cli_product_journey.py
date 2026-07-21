from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import datetime, timezone
import errno
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from typing import cast

import pytest

from daita import Agent, AgentHost, SQLiteSource
from daita.catalog import ResourceKind, catalog_resource_id
from daita.hosting.local_protocol import LocalSocketSecurityError
from daita.hosting.local_server import LocalAgentServer
from daita.learning import LearningCandidateCategory
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolDefinition,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import Readiness, Turn
from daita.operations.checkpoints import OperationSnapshot
from daita.operations.models import ActionProposal, Evidence, Observation

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
PROFILE = ModelProfile(
    id="mock:cli-product",
    context_window_tokens=32_768,
    max_output_tokens=4_096,
    supports_tools=True,
    supports_streaming=True,
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
INSTALLED_CLI = os.environ.get("DAITA_TEST_INSTALLED_CLI")


def _cli_command() -> tuple[str, ...]:
    return (
        (INSTALLED_CLI,)
        if INSTALLED_CLI is not None
        else (sys.executable, "-m", "daita.cli")
    )


def _cli_process_environment() -> dict[str, str]:
    environment = dict(os.environ)
    if INSTALLED_CLI is None:
        environment["PYTHONPATH"] = str(SOURCE_ROOT)
    else:
        environment.pop("PYTHONPATH", None)
    return environment


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
    async def tool_views(
        self,
        operation: OperationSnapshot,
    ) -> tuple[ToolDefinition, ...]:
        del operation
        return ()

    async def validate_action(
        self,
        call: ToolCall,
        operation: OperationSnapshot,
    ) -> ActionProposal:
        del call, operation
        raise AssertionError("text-only CLI journey has no actions")

    async def project_observation(self, evidence: Evidence) -> Observation:
        del evidence
        raise AssertionError("text-only CLI journey has no observations")

    async def evaluate_final_answer(
        self,
        text: str,
        operation: OperationSnapshot,
    ) -> Readiness:
        del text, operation
        return Readiness(
            allowed=True,
            code="ready.text",
            message="The text response is ready.",
            evaluated_at=NOW,
        )


async def _cli(
    root: Path,
    *arguments: str,
    stdin: str | None = None,
    timeout: float = 10.0,
) -> tuple[int, list[dict[str, object]], str]:
    process = await asyncio.create_subprocess_exec(
        *_cli_command(),
        "--root",
        str(root),
        *arguments,
        stdin=(asyncio.subprocess.PIPE if stdin is not None else None),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_cli_process_environment(),
        cwd=(root if INSTALLED_CLI is not None else PROJECT_ROOT),
    )
    output, error = await asyncio.wait_for(
        process.communicate(None if stdin is None else stdin.encode("utf-8")),
        timeout=timeout,
    )
    lines = [
        json.loads(line) for line in output.decode("utf-8").splitlines() if line.strip()
    ]
    assert all(isinstance(line, dict) for line in lines)
    return process.returncode or 0, lines, error.decode("utf-8")


def _only(lines: list[dict[str, object]]) -> dict[str, object]:
    assert len(lines) == 1
    return lines[0]


def _objects(value: object) -> list[dict[str, object]]:
    assert isinstance(value, list)
    assert all(isinstance(item, dict) for item in value)
    return cast(list[dict[str, object]], value)


def _ids():
    counters: dict[str, int] = {}

    def factory(prefix: str) -> str:
        counters[prefix] = counters.get(prefix, 0) + 1
        return f"{prefix}-{counters[prefix]}"

    return factory


class _ScriptedProvider:
    provider_id = PROFILE.id

    def __init__(self) -> None:
        self.responses: list[ModelResponse] = []

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return True

    async def generate(self, request: ModelRequest) -> ModelResponse:
        del request
        if not self.responses:
            raise AssertionError("unexpected model call")
        return self.responses.pop(0)


async def test_real_socket_cli_covers_interactive_inspection_and_event_follow() -> None:
    with tempfile.TemporaryDirectory(prefix=".p95-cli-", dir=PROJECT_ROOT) as value:
        root = Path(value)
        id_factory = _ids()
        seed_provider = MockModelProvider(
            (
                ModelResponse(text="Remembered.", finish_reason=FinishReason.STOP),
                ModelResponse(text="Skill proposed.", finish_reason=FinishReason.STOP),
            ),
            provider_id=PROFILE.id,
        )
        agent = await Agent.create(
            "atlas",
            root=root,
            model=seed_provider,
            context_builder=_Context(),
            domain=_Domain(),
            clock=lambda: NOW,
            id_factory=id_factory,
        )
        try:
            await agent.run(
                "Remember that fiscal weeks start on Monday.",
                session_id="seed-memory",
            )
            skill_result = await agent.run(
                "Propose a skill named reconcile-status: "
                "Compare accepted totals and cite discrepancies.",
                session_id="seed-skill",
            )
            proposal = next(
                value
                for value in await agent.list_learning_proposals(
                    operation_id=skill_result.operation_id
                )
                if value.category is LearningCandidateCategory.SKILL_CHANGE
            )
            await agent.accept_skill_change(
                proposal.id,
                expected_active_version_id=None,
                actor_id="user:owner",
                reason="Reviewed for bounded read-only use.",
            )
        finally:
            await agent.close()

        database = root / "orders.db"
        with sqlite3.connect(database) as connection:
            connection.executescript("""
                CREATE TABLE orders (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL
                );
                INSERT INTO orders (id, status) VALUES ('order-1', 'pending');
                """)
        approval_provider = _ScriptedProvider()
        approval_agent = await Agent.open(
            "atlas",
            root=root,
            model=approval_provider,
            model_profile=PROFILE,
            clock=lambda: NOW,
            id_factory=id_factory,
        )
        try:
            registration = await approval_agent.attach(
                SQLiteSource(database, name="Orders", allow_writes=True)
            )
            recipe: dict[str, object] = {
                "source_id": registration.id,
                "resource_id": catalog_resource_id(
                    registration.id,
                    ResourceKind.TABLE,
                    "main.orders",
                ),
                "key_column": "id",
                "key_value": "order-1",
                "target_column": "status",
                "expected_value": "pending",
                "new_value": "complete",
            }
            approval_provider.responses.extend(
                (
                    ModelResponse(
                        finish_reason=FinishReason.TOOL_CALLS,
                        tool_calls=(
                            ToolCall(
                                id="call-preview",
                                name="data_preview_sqlite_update",
                                arguments=recipe,
                            ),
                        ),
                    ),
                    ModelResponse(
                        finish_reason=FinishReason.TOOL_CALLS,
                        tool_calls=(
                            ToolCall(
                                id="call-update",
                                name="data_update_sqlite",
                                arguments={
                                    **recipe,
                                    "impact_evidence_id": "evidence-1",
                                },
                            ),
                        ),
                    ),
                )
            )
            waiting = await approval_agent.run(
                "Change order-1 from pending to complete after preview and approval.",
                session_id="seed-approval",
            )
            approval_snapshot = await approval_agent.inspect(waiting.operation_id)
            approval_id = approval_snapshot.approvals[0].id
        finally:
            await approval_agent.close()

        runtime_provider = MockModelProvider(
            (
                ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(
                        ToolCall(
                            id="call-chat-query",
                            name="data_query_sqlite",
                            arguments={
                                "source_id": registration.id,
                                "sql": "SELECT COUNT(*) AS total FROM orders",
                            },
                        ),
                    ),
                ),
                ModelResponse(
                    text="There is one order. [evidence:evidence-2]",
                    finish_reason=FinishReason.STOP,
                ),
            ),
            provider_id=PROFILE.id,
        )
        host = await AgentHost.open(
            "atlas",
            root=root,
            model=runtime_provider,
            model_profile=PROFILE,
            cadence_seconds=3_600,
            clock=lambda: NOW,
            id_factory=id_factory,
        )
        server = LocalAgentServer(host)
        source_directory = tempfile.TemporaryDirectory(
            prefix=".p95-cli-source-", dir=PROJECT_ROOT
        )
        try:
            try:
                await server.start()
            except LocalSocketSecurityError as socket_error:
                cause = socket_error.__cause__
                if isinstance(cause, PermissionError) and cause.errno == errno.EPERM:
                    pytest.skip("sandbox forbids binding Unix-domain sockets")
                raise
            assert runtime_provider.requests == ()

            source_root = Path(source_directory.name)
            (source_root / "customers.csv").write_text(
                "id,status\n1,active\n",
                encoding="utf-8",
            )
            code, lines, error = await _cli(
                root,
                "source",
                "add",
                "atlas",
                "local_files",
                str(source_root),
                "--idempotency-key",
                "source-1",
            )
            assert code == 0, error
            source_id = _only(lines)["id"]
            assert isinstance(source_id, str)

            for arguments, result_key in (
                (("source", "list", "atlas"), "sources"),
                (("source", "health", "atlas", source_id), "health"),
                (("operation", "list", "atlas"), "operations"),
                (("approval", "list", "atlas"), "approvals"),
                (("memory", "list", "atlas"), "items"),
                (("skill", "list", "atlas"), "skills"),
                (("monitor", "list", "atlas"), "monitors"),
            ):
                code, lines, error = await _cli(root, *arguments)
                assert code == 0, error
                assert isinstance(_only(lines)[result_key], list)

            code, lines, error = await _cli(
                root, "approval", "inspect", "atlas", approval_id
            )
            assert code == 0, error
            assert _only(lines)["status"] == "pending"

            code, lines, error = await _cli(
                root,
                "catalog",
                "search",
                "atlas",
                "customers",
                "--source-id",
                source_id,
            )
            assert code == 0, error
            hit = _objects(_only(lines)["hits"])[0]
            code, lines, error = await _cli(
                root,
                "catalog",
                "show",
                "atlas",
                str(hit["resource_id"]),
            )
            assert code == 0, error
            assert isinstance(_only(lines)["resource"], Mapping)

            code, lines, error = await _cli(root, "memory", "list", "atlas")
            assert code == 0, error
            memory_id = _objects(_only(lines)["items"])[0]["id"]
            code, lines, error = await _cli(
                root, "memory", "inspect", "atlas", str(memory_id)
            )
            assert code == 0, error
            assert _only(lines)["qualification"] == "unbound"

            code, lines, error = await _cli(root, "skill", "list", "atlas")
            assert code == 0, error
            skill_id = _objects(_only(lines)["skills"])[0]["skill_id"]
            code, lines, error = await _cli(
                root, "skill", "inspect", "atlas", str(skill_id)
            )
            assert code == 0, error
            assert _only(lines)["current_instructions"]

            code, streamed, error = await _cli(
                root,
                "chat",
                "atlas",
                stdin="How many orders are there?\n",
                timeout=15,
            )
            assert code == 0, error
            assert any(value.get("kind") == "event" for value in streamed)
            result_line = next(
                value for value in streamed if value.get("kind") == "result"
            )
            result = result_line["result"]
            assert isinstance(result, Mapping)
            assert result.get("status") == "completed", result
            operation_id = result["operation_id"]
            code, lines, error = await _cli(
                root, "operation", "inspect", "atlas", str(operation_id)
            )
            assert code == 0, error
            operation = _only(lines)["operation"]
            assert isinstance(operation, Mapping)
            assert operation["status"] == "succeeded", (
                operation["terminal_reason"],
                len(runtime_provider.requests),
            )

            natural_request = (
                f"Every 5 minutes, Count customer rows for source {source_id} "
                "when rows.0.total > 0"
            )
            code, lines, error = await _cli(
                root,
                "monitor",
                "propose-natural",
                "atlas",
                "customer-count",
                natural_request,
                "--idempotency-key",
                "monitor-proposal-1",
            )
            assert code == 0, error
            monitor_proposal = _only(lines)
            code, lines, error = await _cli(
                root,
                "monitor",
                "confirm",
                "atlas",
                str(monitor_proposal["id"]),
                "--candidate-hash",
                str(monitor_proposal["candidate_hash"]),
                "--actor",
                "user:owner",
                "--reason",
                "Scope and threshold reviewed.",
                "--idempotency-key",
                "monitor-confirm-1",
            )
            assert code == 0, error
            assert isinstance(_only(lines)["monitor"], Mapping)
            code, lines, error = await _cli(
                root, "monitor", "inspect", "atlas", "customer-count"
            )
            assert code == 0, error
            assert _only(lines)["counts"]["versions"] == 1  # type: ignore[index]

            code, lines, error = await _cli(root, "events", "read", "atlas")
            assert code == 0, error
            after = _only(lines)["next_after"]
            assert isinstance(after, int)
            follower = await asyncio.create_subprocess_exec(
                *_cli_command(),
                "--root",
                str(root),
                "events",
                "follow",
                "atlas",
                "--after",
                str(after),
                "--max-events",
                "1",
                "--poll-seconds",
                "0.02",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=_cli_process_environment(),
                cwd=(root if INSTALLED_CLI is not None else PROJECT_ROOT),
            )
            await asyncio.sleep(0.1)
            code, _, error = await _cli(
                root,
                "monitor",
                "pause",
                "atlas",
                "customer-count",
                "--actor",
                "user:owner",
                "--reason",
                "Prove live event follow.",
                "--idempotency-key",
                "monitor-pause-1",
            )
            assert code == 0, error
            followed, follow_error = await asyncio.wait_for(
                follower.communicate(), timeout=10
            )
            assert follower.returncode == 0, follow_error.decode("utf-8")
            followed_event = json.loads(followed.decode("utf-8").strip())
            assert followed_event["sequence"] > after
            assert followed_event["type"] == "monitor.pause"

            code, lines, error = await _cli(
                root,
                "source",
                "detach",
                "atlas",
                source_id,
                "--idempotency-key",
                "source-detach-1",
            )
            assert code == 0, error
            assert _only(lines)["active"] is False
            code, lines, error = await _cli(
                root, "source", "list", "atlas", "--include-detached"
            )
            assert code == 0, error
            detached = next(
                item
                for item in _objects(_only(lines)["sources"])
                if item["id"] == source_id
            )
            assert detached["active"] is False
        finally:
            await server.stop(drain=True)
            source_directory.cleanup()
