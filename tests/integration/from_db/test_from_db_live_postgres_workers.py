"""Live PostgreSQL + LLM specialist-worker integration for ``Agent.from_db``."""

from __future__ import annotations

import os

import pytest

from daita.agents.agent import Agent
from daita.plugins import PluginKind, PluginManifest
from daita.runtime import (
    AccessMode,
    Capability,
    Evidence,
    EvidenceSchema,
    Operation,
    PolicyDecision,
    PolicyEffect,
    RiskLevel,
    RuntimeEventType,
    Task,
    Worker,
)

from tests.integration.runtime.live_postgres_runtime_helpers import (
    require_live_postgres_runtime,
    start_seeded_postgres,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.requires_db,
]


@pytest.fixture(scope="module")
def live_openai_kwargs() -> dict:
    require_live_postgres_runtime()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return {
        "llm_provider": "openai",
        "model": os.environ.get("OPENAI_TEST_MODEL", "gpt-5.4-mini"),
        "api_key": api_key,
        "temperature": 0,
    }


@pytest.fixture(scope="module")
def seeded_postgres_url(live_openai_kwargs):
    container, url = start_seeded_postgres("daita-from-db-pg-workers")
    try:
        yield url
    finally:
        container.remove()


async def test_from_db_live_postgres_schema_specialist_worker_persists_evidence(
    seeded_postgres_url,
    live_openai_kwargs,
):
    specialist = LiveSchemaSpecialistPlugin()
    agent = await Agent.from_db(
        seeded_postgres_url,
        name="LiveFromDbPostgresSchemaWorker",
        cache_ttl=0,
        plugins=(specialist,),
        **live_openai_kwargs,
    )

    try:
        result = await agent.run_detailed(
            "Which columns are in the orders table? Use the schema specialist."
        )
        snapshot = await agent.runtime.inspect_operation(result.operation_id)
    finally:
        await agent.stop()

    assert specialist.executor.calls == 1
    assert any(
        task.capability_id == "specialist.schema.summarize"
        and task.metadata["reason"] == "worker:live.schema_specialist"
        for task in snapshot.tasks
    )
    assert any(
        evidence.kind == "specialist.schema.summary"
        and "orders" in evidence.payload["summary"]
        for evidence in snapshot.evidence
    )
    assert any(
        decision.effect is PolicyEffect.ALLOW for decision in snapshot.policy_decisions
    )
    assert RuntimeEventType.WORKER_DELEGATED in {
        event.type for event in snapshot.events
    }
    assert RuntimeEventType.WORKER_COMPLETED in {
        event.type for event in snapshot.events
    }
    assert result.answer
    assert "orders" in result.answer.lower()
    _assert_no_unvalidated_db_reads(snapshot.tasks)


class LiveSchemaSpecialistExecutor:
    id = "live_schema_specialist.executor"
    capability_ids = frozenset({"specialist.schema.summarize"})

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, task: Task, operation: Operation, context):
        del context
        self.calls += 1
        schema = task.input.get("schema") or {}
        orders = next(
            (
                table
                for table in schema.get("tables", [])
                if table.get("name") == "orders"
            ),
            {},
        )
        columns = [
            column.get("name") or column.get("column_name")
            for column in orders.get("columns", [])
        ]
        return [
            Evidence(
                kind="specialist.schema.summary",
                owner="live_schema_specialist",
                operation_id=operation.id,
                task_id=task.id,
                payload={
                    "summary": (
                        "orders has columns: "
                        + ", ".join(str(column) for column in columns if column)
                    ),
                    "table": "orders",
                    "columns": [str(column) for column in columns if column],
                },
            )
        ]


class LiveSchemaSpecialistPolicy:
    id = "live_schema_specialist_allowed"
    owner = "live_schema_specialist"

    def applies_to(self, request, operation_type: str) -> bool:
        del request
        return operation_type == "schema.query"

    def modify_contract(self, contract):
        return contract

    def evaluate_operation(self, operation: Operation):
        return PolicyDecision(
            policy_id=self.id,
            owner=self.owner,
            effect=PolicyEffect.ALLOW,
            reason="Live schema specialist work is allowed.",
            severity=RiskLevel.LOW,
            operation_id=operation.id,
        )


class LiveSchemaSpecialistPlugin:
    manifest = PluginManifest(
        id="live_schema_specialist",
        display_name="Live Schema Specialist",
        version="1.0.0",
        kind=PluginKind.WORKER_PROVIDER,
        domains=frozenset({"db"}),
    )

    def __init__(self) -> None:
        self.executor = LiveSchemaSpecialistExecutor()
        self.policy = LiveSchemaSpecialistPolicy()

    def declare_capabilities(self):
        return (
            Capability(
                id="specialist.schema.summarize",
                owner="live_schema_specialist",
                description="Summarize live schema evidence for a schema specialist.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"schema.query"}),
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"specialist.schema.summary"}),
                executor=self.executor.id,
                runtime_only=True,
                specialist_only=True,
                side_effecting=False,
            ),
        )

    def get_executors(self):
        return (self.executor,)

    def declare_evidence_schemas(self):
        return (
            EvidenceSchema(
                kind="specialist.schema.summary",
                owner="live_schema_specialist",
                json_schema={"type": "object"},
            ),
        )

    def declare_policies(self):
        return (self.policy,)

    def get_workers(self):
        return (
            Worker(
                id="live.schema_specialist",
                owner="live_schema_specialist",
                role="schema_specialist",
                capability_ids=frozenset({"specialist.schema.summarize"}),
                input_schema={"type": "object"},
                output_evidence=frozenset({"specialist.schema.summary"}),
            ),
        )


def _assert_no_unvalidated_db_reads(tasks) -> None:
    capabilities = [task.capability_id for task in tasks]
    for index, capability_id in enumerate(capabilities):
        if capability_id != "db.sql.execute_read":
            continue
        assert "db.sql.validate" in capabilities[:index]
