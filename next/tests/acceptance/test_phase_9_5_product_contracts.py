from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from daita import Agent, AgentHost, SQLiteSource
from daita.capabilities import AccessMode, RiskLevel
from daita.extensions import (
    ConfiguredExtension,
    ExtensionKind,
    ExtensionManifest,
    ExtensionRegistration,
    tool,
)
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
from daita.llm.providers.openai import OpenAIResponsesProvider
from daita.loop.models import Readiness, Turn
from daita.memory.models import MemoryListRequest, MemoryScope
from daita.monitors import (
    IntervalSchedule,
    MonitorCondition,
    MonitorConditionKind,
    MonitorDefinition,
    MonitorScope,
)
from daita.operations.checkpoints import OperationSnapshot
from daita.operations.models import ActionProposal, Evidence, Observation
from daita.storage.sqlite import SQLiteCorruptionError

NOW = datetime(2026, 7, 19, 18, 0, tzinfo=timezone.utc)
PROFILE = ModelProfile(
    id="mock:phase-9-5",
    context_window_tokens=32_768,
    max_output_tokens=4_096,
    supports_tools=True,
    supports_streaming=True,
)


class ScriptedProvider:
    provider_id = PROFILE.id

    def __init__(self, *responses: ModelResponse) -> None:
        self._responses = list(responses)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        del request
        if not self._responses:
            raise AssertionError("unexpected model call")
        return self._responses.pop(0)


class TextContext:
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
                    session_id=operation.operation.session_id,
                    turn_id=turn.id,
                    role=MessageRole.USER,
                    content=(TextBlock(message),),
                ),
            ),
        )


class TextDomain:
    def tool_views(self, operation: OperationSnapshot) -> tuple[ToolDefinition, ...]:
        del operation
        return ()

    async def validate_action(
        self,
        call: ToolCall,
        operation: OperationSnapshot,
    ) -> ActionProposal:
        del call, operation
        raise AssertionError("text-only domain has no actions")

    async def project_observation(self, evidence: Evidence) -> Observation:
        del evidence
        raise AssertionError("text-only domain has no observations")

    async def evaluate_final_answer(
        self,
        text: str,
        operation: OperationSnapshot,
    ) -> Readiness:
        del text, operation
        return Readiness(
            allowed=True,
            code="ready",
            message="The response is ready.",
            evaluated_at=NOW,
        )


def _ids():
    counters: dict[str, int] = {}

    def factory(prefix: str) -> str:
        counters[prefix] = counters.get(prefix, 0) + 1
        return f"{prefix}-{counters[prefix]}"

    return factory


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                status TEXT NOT NULL
            );
            INSERT INTO customers (status) VALUES ('active'), ('inactive');
            """)


def _tool(
    call_id: str,
    name: str,
    arguments: Mapping[str, object],
) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
    )


async def test_sqlite_read_persists_exact_validator_owned_authority(
    tmp_path: Path,
) -> None:
    database = tmp_path / "customers.db"
    _database(database)
    provider = ScriptedProvider()
    agent = await Agent.create(
        "authority",
        root=tmp_path / "state",
        model=provider,
        model_profile=PROFILE,
        clock=lambda: NOW,
        id_factory=_ids(),
    )
    state_path = agent.home / "state.db"
    try:
        registration = await agent.attach(SQLiteSource(database))
        provider._responses.extend(
            (
                _tool(
                    "query",
                    "data_query_sqlite",
                    {
                        "source_id": registration.id,
                        "sql": "SELECT COUNT(*) AS total FROM customers",
                    },
                ),
                ModelResponse(
                    text="There are 2 customers. [evidence:evidence-1]",
                    finish_reason=FinishReason.STOP,
                ),
            )
        )
        result = await agent.run("Count the customers.")
        snapshot = await agent.inspect(result.operation_id)
        facts = snapshot.tasks[0].execution_facts.validation_facts

        assert facts.schema_version == 1
        assert facts.source_id == registration.id
        assert len(facts.resource_ids) == 1
        assert dict(facts.resource_revisions).keys() == set(facts.resource_ids)
        assert facts.source_revision is not None
        assert facts.source_ids == (registration.id,)
        assert facts.source_revisions == ((registration.id, facts.source_revision),)
        assert facts.freshness_state == "current"
        assert facts.sensitivity_class == "internal"
        evidence = snapshot.evidence[0]
        assert evidence.metadata_schema_version == 1
        assert evidence.accepted is evidence.applicable is True
        assert evidence.acceptance_reason == "schema_validated"
        assert evidence.applicability_reason == "current_operation"
        assert evidence.validation_facts == facts

    finally:
        await agent.close()

    reopened = await Agent.open(
        "authority",
        root=tmp_path / "state",
        model=provider,
        model_profile=PROFILE,
        clock=lambda: NOW,
    )
    try:
        durable = await reopened.inspect(result.operation_id)
        assert durable.tasks[0].execution_facts.validation_facts == facts
        assert durable.evidence[0] == evidence
    finally:
        await reopened.close()

    with sqlite3.connect(state_path) as connection:
        connection.execute(
            "UPDATE tasks SET validation_sensitivity_class = 'public' "
            "WHERE operation_id = ?",
            (result.operation_id,),
        )
    tampered = await Agent.open(
        "authority",
        root=tmp_path / "state",
        model=provider,
        model_profile=PROFILE,
        clock=lambda: NOW,
    )
    try:
        with pytest.raises(SQLiteCorruptionError):
            await tampered.inspect(result.operation_id)
    finally:
        await tampered.close()


async def test_configured_retained_model_reopens_without_caller_injection(
    tmp_path: Path,
) -> None:
    provider = OpenAIResponsesProvider("gpt-phase-9-5")
    profile = ModelProfile(
        id=provider.provider_id,
        context_window_tokens=8_192,
        max_output_tokens=1_024,
        supports_tools=True,
    )
    agent = await Agent.create(
        "configured",
        root=tmp_path,
        model=provider,
        model_profile=profile,
        clock=lambda: NOW,
    )
    await agent.close()

    reopened = await AgentHost.open("configured", root=tmp_path, clock=lambda: NOW)
    try:
        assert reopened.configured is True
        assert reopened.model_profile == profile
    finally:
        await reopened.stop(drain=False)


async def test_default_host_projects_always_monitor_finding_from_read_evidence(
    tmp_path: Path,
) -> None:
    database = tmp_path / "customers.db"
    _database(database)
    provider = ScriptedProvider()
    host = await AgentHost.create(
        "monitoring",
        root=tmp_path / "state",
        model=provider,
        model_profile=PROFILE,
        clock=lambda: NOW,
        id_factory=_ids(),
    )
    try:
        await host.start()
        registration = await host.attach(
            SQLiteSource(database),
            idempotency_key="source-1",
        )
        provider._responses.extend(
            (
                _tool(
                    "query",
                    "data_query_sqlite",
                    {
                        "source_id": registration.id,
                        "sql": "SELECT COUNT(*) AS total FROM customers",
                    },
                ),
                ModelResponse(
                    text="There are 2 customers. [evidence:evidence-1]",
                    finish_reason=FinishReason.STOP,
                ),
            )
        )
        definition = MonitorDefinition(
            name="Customer count",
            objective="Count the customers.",
            scope=MonitorScope(source_ids=(registration.id,)),
            schedule=IntervalSchedule(interval_seconds=300, anchor_at=NOW),
            condition=MonitorCondition(kind=MonitorConditionKind.ALWAYS),
        )
        proposal = await host.propose_monitor(
            "customer-count",
            definition,
            idempotency_key="proposal-1",
        )
        await host.confirm_monitor(
            proposal.id,
            candidate_hash=proposal.candidate_hash,
            actor_id="operator",
            reason="The scope and schedule are correct.",
        )
        result = await host.run_monitor_now(
            "customer-count",
            idempotency_key="run-1",
        )
        inspection = await host.inspect_monitor("customer-count")

        assert result.finding_id is not None
        assert len(inspection.findings) == 1
        assert inspection.findings[0].operation_id == result.operation_id
        operation = await host.inspect_operation(result.operation_id or "")
        assert inspection.findings[0].evidence_id in {
            evidence.id for evidence in operation.evidence if evidence.accepted
        }
    finally:
        await host.stop(drain=False)


async def test_ordinary_remember_interaction_enters_learning_service(
    tmp_path: Path,
) -> None:
    agent = await Agent.create(
        "learning",
        root=tmp_path,
        model=MockModelProvider(
            (
                ModelResponse(
                    text="I will remember that.", finish_reason=FinishReason.STOP
                ),
            ),
            provider_id="mock:phase-9-5",
        ),
        context_builder=TextContext(),
        domain=TextDomain(),
        clock=lambda: NOW,
    )
    try:
        result = await agent.run(
            "Remember that fiscal weeks start on Monday.",
            session_id="learning-session",
        )
        memories = await agent.list_memories(
            MemoryListRequest(scope=MemoryScope(agent_id=agent.id))
        )

        assert result.post_operation_notices == ()
        assert len(memories.items) == 1
        assert memories.items[0].snapshot.version.content == (
            "Fiscal weeks start on Monday."
        )
    finally:
        await agent.close()


async def test_configured_extension_composes_with_builtin_data_capability(
    tmp_path: Path,
) -> None:
    database = tmp_path / "customers.db"
    _database(database)

    @tool(
        id="example.lookup",
        owner="example",
        name="example_lookup",
        description="Look up one configured extension value.",
        input_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        access_mode=AccessMode.READ,
        risk=RiskLevel.LOW,
    )
    async def lookup(arguments: Mapping[str, object]) -> Mapping[str, object]:
        return {"value": f"configured:{arguments['key']}"}

    manifest = ExtensionManifest(
        id="example",
        version="1.0.0",
        kind=ExtensionKind.CAPABILITY_PROVIDER,
        declarations=lookup.declarations(),
    )
    configured = ConfiguredExtension(
        id="example",
        factory=lambda: ExtensionRegistration(
            manifest=manifest,
            executors=(lookup.executor,),
        ),
    )
    provider = ScriptedProvider()
    agent = await Agent.create(
        "extended",
        root=tmp_path / "state",
        model=provider,
        model_profile=PROFILE,
        extensions=(configured,),
        clock=lambda: NOW,
        id_factory=_ids(),
    )
    try:
        registration = await agent.attach(SQLiteSource(database))
        provider._responses.extend(
            (
                _tool(
                    "query",
                    "data_query_sqlite",
                    {
                        "source_id": registration.id,
                        "sql": "SELECT COUNT(*) AS total FROM customers",
                    },
                ),
                _tool("extension", "example_lookup", {"key": "alpha"}),
                ModelResponse(
                    text="There are 2 customers. [evidence:evidence-1]",
                    finish_reason=FinishReason.STOP,
                ),
            )
        )
        result = await agent.run("Count customers, then use the example lookup.")
        snapshot = await agent.inspect(result.operation_id)

        assert [task.capability_id for task in snapshot.tasks] == [
            "data.sqlite.query",
            "example.lookup",
        ]
        assert all(evidence.accepted for evidence in snapshot.evidence)
    finally:
        await agent.close()


def test_real_cli_subprocess_supports_create_and_model_set(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[2] / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root)

    created = subprocess.run(
        (
            sys.executable,
            "-m",
            "daita.cli",
            "--root",
            str(tmp_path),
            "agent",
            "create",
            "cli-agent",
            "--idempotency-key",
            "create-1",
        ),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        cwd=source_root.parent,
        timeout=10,
    )

    assert created.returncode == 0, created.stderr
    assert json.loads(created.stdout)["created"] is True

    configured = subprocess.run(
        (
            sys.executable,
            "-m",
            "daita.cli",
            "--root",
            str(tmp_path),
            "model",
            "set",
            "cli-agent",
            "--provider",
            "openai",
            "--model",
            "gpt-phase-9-5",
            "--secret",
            "env:OPENAI_API_KEY",
            "--context-window-tokens",
            "8192",
            "--max-output-tokens",
            "1024",
            "--idempotency-key",
            "model-1",
        ),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        cwd=source_root.parent,
        timeout=10,
    )

    assert configured.returncode == 0, configured.stderr
    assert json.loads(configured.stdout)["configured"] is True
