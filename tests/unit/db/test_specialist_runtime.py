from daita.db import DbRequest, DbRuntime
from daita.plugins import PluginKind, PluginManifest, WorkerProviderPlugin
from daita.runtime import (
    AccessMode,
    Capability,
    Evidence,
    EvidenceSchema,
    Operation,
    OperationStatus,
    PolicyDecision,
    PolicyEffect,
    RiskLevel,
    RuntimeEventType,
    Task,
    Worker,
)

SCHEMA_PAYLOAD = {
    "database_type": "probe",
    "tables": [
        {
            "name": "orders",
            "columns": [
                {"name": "id", "type": "integer"},
                {"name": "total", "type": "real"},
            ],
        }
    ],
    "table_count": 1,
}


SEARCH_PAYLOAD = {
    "tables": [{"name": "orders", "score": 1.0}],
    "columns": [{"table": "orders", "name": "total", "score": 1.0}],
}


INSPECT_PAYLOAD = {
    "asset": {"name": "orders", "columns": ["id", "total"]},
}


SPECIALIST_PAYLOAD = {
    "summary": "orders has id and total columns",
    "confidence": 0.95,
}


class SchemaProbeExecutor:
    def __init__(self, executor_id: str, evidence_kind: str, payload: dict):
        self.id = executor_id
        self.capability_ids = frozenset(
            {
                "db.schema.inspect",
                "catalog.schema.search",
                "catalog.asset.inspect",
                "specialist.schema.summarize",
            }
        )
        self.evidence_kind = evidence_kind
        self.payload = payload
        self.calls = 0

    async def execute(self, task: Task, operation: Operation, context):
        self.calls += 1
        return [
            Evidence(
                kind=self.evidence_kind,
                owner="schema_probe",
                operation_id=operation.id,
                task_id=task.id,
                payload=dict(self.payload),
            )
        ]


class SchemaGovernancePolicy:
    id = "schema_specialist_allowed"
    owner = "schema_probe"

    def applies_to(self, request, operation_type: str) -> bool:
        return operation_type == "schema.query"

    def modify_contract(self, contract):
        return contract

    def evaluate_operation(self, operation: Operation):
        return PolicyDecision(
            policy_id=self.id,
            owner=self.owner,
            effect=PolicyEffect.ALLOW,
            reason="Schema specialist delegation is allowed.",
            severity=RiskLevel.LOW,
            operation_id=operation.id,
        )


class DenySchemaSpecialistTaskPolicy:
    id = "deny_schema_specialist_task"
    owner = "schema_probe"

    def applies_to(self, request, operation_type: str) -> bool:
        capability = request.get("capability") if isinstance(request, dict) else None
        return (
            isinstance(capability, dict)
            and capability.get("id") == "specialist.schema.summarize"
        )

    def modify_contract(self, contract):
        return contract

    def evaluate_operation(self, operation: Operation):
        return PolicyDecision(
            policy_id=self.id,
            owner=self.owner,
            effect=PolicyEffect.DENY,
            reason="Schema specialist delegation is denied for this test.",
            severity=RiskLevel.HIGH,
            operation_id=operation.id,
        )


class SchemaSpecialistPlugin(WorkerProviderPlugin):
    manifest = PluginManifest(
        id="schema_probe",
        display_name="Schema Probe",
        version="1.0.0",
        kind=PluginKind.WORKER_PROVIDER,
        domains=frozenset({"db"}),
    )

    def __init__(self, *, deny_specialist: bool = False):
        self.schema_executor = SchemaProbeExecutor(
            "schema_probe.schema.inspect", "schema.asset_profile", SCHEMA_PAYLOAD
        )
        self.search_executor = SchemaProbeExecutor(
            "schema_probe.schema.search", "schema.search_result", SEARCH_PAYLOAD
        )
        self.inspect_executor = SchemaProbeExecutor(
            "schema_probe.asset.inspect", "schema.asset_profile", INSPECT_PAYLOAD
        )
        self.specialist_executor = SchemaProbeExecutor(
            "schema_probe.specialist.schema_summarize",
            "specialist.schema.summary",
            SPECIALIST_PAYLOAD,
        )
        self.policy = SchemaGovernancePolicy()
        self.deny_specialist = deny_specialist

    def declare_capabilities(self):
        return [
            Capability(
                id="db.schema.inspect",
                owner="schema_probe",
                description="Inspect probe schema.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"schema.query"}),
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"schema.asset_profile"}),
                executor="schema_probe.schema.inspect",
                runtime_only=True,
                side_effecting=False,
            ),
            Capability(
                id="catalog.schema.search",
                owner="schema_probe",
                description="Search probe schema.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"schema.query"}),
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"schema.search_result"}),
                executor="schema_probe.schema.search",
                runtime_only=True,
                side_effecting=False,
            ),
            Capability(
                id="catalog.asset.inspect",
                owner="schema_probe",
                description="Inspect one probe asset.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"schema.query"}),
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"schema.asset_profile"}),
                executor="schema_probe.asset.inspect",
                runtime_only=True,
                side_effecting=False,
            ),
            Capability(
                id="specialist.schema.summarize",
                owner="schema_probe",
                description="Summarize schema evidence for a schema specialist.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"schema.query"}),
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"specialist.schema.summary"}),
                executor="schema_probe.specialist.schema_summarize",
                runtime_only=True,
                specialist_only=True,
                side_effecting=False,
            ),
        ]

    def get_executors(self):
        return [
            self.schema_executor,
            self.search_executor,
            self.inspect_executor,
            self.specialist_executor,
        ]

    def declare_evidence_schemas(self):
        return [
            EvidenceSchema(
                kind="schema.asset_profile",
                owner="schema_probe",
                json_schema={"type": "object"},
            ),
            EvidenceSchema(
                kind="schema.search_result",
                owner="schema_probe",
                json_schema={"type": "object"},
            ),
            EvidenceSchema(
                kind="specialist.schema.summary",
                owner="schema_probe",
                json_schema={"type": "object"},
            ),
        ]

    def declare_policies(self):
        policies = [self.policy]
        if self.deny_specialist:
            policies.append(DenySchemaSpecialistTaskPolicy())
        return policies

    def get_workers(self):
        return [
            Worker(
                id="schema.specialist",
                owner="schema_probe",
                role="schema_specialist",
                capability_ids=frozenset({"specialist.schema.summarize"}),
                input_schema={"type": "object"},
                output_evidence=frozenset({"specialist.schema.summary"}),
            )
        ]


async def test_db_runtime_delegates_schema_specialist_worker_with_persisted_state():
    plugin = SchemaSpecialistPlugin()
    runtime = DbRuntime(plugins=(plugin,))

    result = await runtime.run(DbRequest("which columns are in orders table?"))
    snapshot = await runtime.inspect_operation(result.operation_id)

    assert result.status is OperationStatus.SUCCEEDED
    assert plugin.specialist_executor.calls == 1
    assert any(
        task.capability_id == "specialist.schema.summarize"
        and task.metadata["reason"] == "worker:schema.specialist"
        for task in snapshot.tasks
    )
    assert any(
        evidence.kind == "specialist.schema.summary"
        and evidence.payload == SPECIALIST_PAYLOAD
        for evidence in snapshot.evidence
    )
    assert PolicyEffect.ALLOW in {
        decision.effect for decision in snapshot.policy_decisions
    }
    assert RuntimeEventType.WORKER_DELEGATED in {
        event.type for event in snapshot.events
    }
    assert snapshot.operation.status is OperationStatus.SUCCEEDED


async def test_schema_specialist_delegation_is_blocked_by_task_governance():
    plugin = SchemaSpecialistPlugin(deny_specialist=True)
    runtime = DbRuntime(plugins=(plugin,))

    result = await runtime.run(DbRequest("which columns are in orders table?"))
    snapshot = await runtime.inspect_operation(result.operation_id)

    assert result.status is OperationStatus.BLOCKED
    assert result.warnings == ("db_runtime_governance_denied",)
    assert plugin.specialist_executor.calls == 0
    assert any(
        task.capability_id == "specialist.schema.summarize"
        and task.status.value == "blocked"
        for task in snapshot.tasks
    )
    assert PolicyEffect.DENY in {
        decision.effect for decision in snapshot.policy_decisions
    }
    assert RuntimeEventType.WORKER_DELEGATED in {
        event.type for event in snapshot.events
    }
