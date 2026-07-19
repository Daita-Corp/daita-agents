from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone

from daita.capabilities import (
    AccessMode,
    Capability,
    CapabilityRegistry,
    EvidenceCandidate,
    ExecutionRequest,
    RiskLevel,
    ToolView,
)
from daita.domains.data import (
    LOCAL_FILE_READ_CAPABILITY_ID,
    LOCAL_FILE_READ_EVIDENCE_KIND,
    SQLITE_QUERY_CAPABILITY_ID,
    SQLITE_QUERY_EVIDENCE_KIND,
    TABULAR_COMPARE_CAPABILITY_ID,
    TABULAR_COMPARE_EVIDENCE_KIND,
    DataDomainController,
    ResourceSchema,
)
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
)
from daita.loop.models import LoopPhase
from daita.operations.models import (
    ActionProposal,
    ActionRejection,
    AgentTrigger,
    Evidence,
    TriggerKind,
)
from daita.operations.runtime import OperationRuntime

NOW = datetime(2026, 7, 19, 1, 0, tzinfo=timezone.utc)


class CatalogReader:
    async def source_adapter_id(self, agent_id: str, source_id: str) -> str | None:
        return "sqlite"

    async def resource_schemas(
        self,
        agent_id: str,
        source_id: str,
    ) -> tuple[ResourceSchema, ...]:
        assert agent_id == "agent-atlas"
        if source_id == "source-files":
            return (
                ResourceSchema(
                    resource_id="resource-export",
                    source_id=source_id,
                    name="customers.csv",
                    aliases=("customers.csv",),
                    columns=("id", "name", "status"),
                ),
            )
        if source_id == "source-database":
            return (
                ResourceSchema(
                    resource_id="resource-customers",
                    source_id=source_id,
                    name="customers",
                    aliases=("main.customers",),
                    columns=("id", "name", "status"),
                ),
            )
        return ()

    async def is_current_tabular_file(
        self,
        agent_id: str,
        source_id: str,
        resource_id: str,
    ) -> bool:
        assert agent_id == "agent-atlas"
        return source_id == "source-files" and resource_id == "resource-export"

    async def is_writable_sqlite_source(
        self,
        agent_id: str,
        source_id: str,
    ) -> bool:
        return False


class CandidateExecutor:
    def __init__(
        self,
        executor_id: str,
        evidence_kind: str,
        payload: Mapping[str, object] | None = None,
    ) -> None:
        self.executor_id = executor_id
        self.evidence_kind = evidence_kind
        self.payload = payload
        self.requests: list[ExecutionRequest] = []

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        self.requests.append(request)
        payload = self.payload
        if payload is None:
            payload = {
                "left": {
                    "evidence_id": request.arguments["left_evidence_id"],
                    "source_id": "source-files",
                },
                "right": {
                    "evidence_id": request.arguments["right_evidence_id"],
                    "source_id": "source-database",
                },
                "truncated": bool(request.arguments.get("test_truncated", False)),
            }
        return EvidenceCandidate(
            kind=self.evidence_kind,
            schema_version=1,
            payload=payload,
        )


def _capability(
    capability_id: str,
    executor_id: str,
    evidence_kind: str,
    input_properties: Mapping[str, str],
    output_properties: Mapping[str, str],
) -> Capability:
    return Capability(
        id=capability_id,
        owner="test-data",
        description=f"Test contract for {capability_id}.",
        input_schema={
            "type": "object",
            "properties": {
                name: {"type": kind} for name, kind in input_properties.items()
            },
            "required": list(input_properties),
            "additionalProperties": False,
        },
        output_evidence_kind=evidence_kind,
        output_schema_version=1,
        output_schema={
            "type": "object",
            "properties": {
                name: {"type": kind} for name, kind in output_properties.items()
            },
            "required": list(output_properties),
            "additionalProperties": False,
        },
        executor_id=executor_id,
        access_mode=AccessMode.READ,
        risk=RiskLevel.LOW,
        side_effecting=False,
        idempotent=True,
        replay_safe=True,
    )


def _registry(
    *,
    file_source: str,
    query_source: str,
) -> tuple[CapabilityRegistry, tuple[CandidateExecutor, ...]]:
    file_executor = CandidateExecutor(
        "test.file.executor",
        LOCAL_FILE_READ_EVIDENCE_KIND,
        {
            "source_id": file_source,
            "resource_id": "resource-export",
            "truncated": False,
        },
    )
    query_executor = CandidateExecutor(
        "test.query.executor",
        SQLITE_QUERY_EVIDENCE_KIND,
        {
            "source_id": query_source,
            "resource_id": "resource-customers",
            "truncated": False,
        },
    )
    comparison_executor = CandidateExecutor(
        "test.compare.executor",
        TABULAR_COMPARE_EVIDENCE_KIND,
    )
    capabilities = (
        _capability(
            LOCAL_FILE_READ_CAPABILITY_ID,
            file_executor.executor_id,
            LOCAL_FILE_READ_EVIDENCE_KIND,
            {"source_id": "string", "resource_id": "string"},
            {
                "source_id": "string",
                "resource_id": "string",
                "truncated": "boolean",
            },
        ),
        _capability(
            SQLITE_QUERY_CAPABILITY_ID,
            query_executor.executor_id,
            SQLITE_QUERY_EVIDENCE_KIND,
            {"source_id": "string"},
            {
                "source_id": "string",
                "resource_id": "string",
                "truncated": "boolean",
            },
        ),
        _capability(
            TABULAR_COMPARE_CAPABILITY_ID,
            comparison_executor.executor_id,
            TABULAR_COMPARE_EVIDENCE_KIND,
            {
                "left_evidence_id": "string",
                "right_evidence_id": "string",
                "key_columns": "array",
                "compare_columns": "array",
                "test_truncated": "boolean",
            },
            {"left": "object", "right": "object", "truncated": "boolean"},
        ),
    )
    executors = (file_executor, query_executor, comparison_executor)
    return (
        CapabilityRegistry(
            capabilities=capabilities,
            executors=executors,
            tool_views=tuple(
                ToolView(
                    name=name,
                    capability_id=capability.id,
                    description=capability.description,
                )
                for name, capability in zip(
                    ("data_read_file", "data_query_sqlite", "data_compare_tabular"),
                    capabilities,
                    strict=True,
                )
            ),
        ),
        executors,
    )


def _ids(label: str):
    counters: dict[str, int] = {}

    def factory(prefix: str) -> str:
        counters[prefix] = counters.get(prefix, 0) + 1
        return f"{prefix}-{label}-{counters[prefix]}"

    return factory


async def _snapshot_with_reads(
    *,
    label: str,
    file_source: str = "source-files",
    query_source: str = "source-database",
    with_comparison: bool = False,
    comparison_truncated: bool = False,
    message: str = "Compare the newest export for discrepancies.",
) -> tuple[DataDomainController, OperationRuntime, tuple[Evidence, ...]]:
    registry, _ = _registry(file_source=file_source, query_source=query_source)
    controller = DataDomainController(registry, CatalogReader(), clock=lambda: NOW)
    runtime = OperationRuntime(
        clock=lambda: NOW,
        id_factory=_ids(label),
        capabilities=registry,
    )
    snapshot = await runtime.begin(
        AgentTrigger(
            id=f"trigger-{label}",
            agent_id="agent-atlas",
            kind=TriggerKind.USER,
            source_id="user:test",
            payload={"message": message},
            created_at=NOW,
        )
    )
    turn = await runtime.begin_turn(snapshot.operation.id)
    calls = [
        ToolCall(
            id=f"call-{label}-file",
            name="data_read_file",
            arguments={
                "source_id": file_source,
                "resource_id": "resource-export",
            },
        ),
        ToolCall(
            id=f"call-{label}-query",
            name="data_query_sqlite",
            arguments={"source_id": query_source},
        ),
    ]
    if with_comparison:
        calls.append(
            ToolCall(
                id=f"call-{label}-compare",
                name="data_compare_tabular",
                arguments={
                    "left_evidence_id": f"evidence-{label}-1",
                    "right_evidence_id": f"evidence-{label}-2",
                    "key_columns": ["id"],
                    "compare_columns": ["name", "status"],
                    "test_truncated": comparison_truncated,
                },
            )
        )
    request = ModelRequest(
        operation_id=snapshot.operation.id,
        turn_id=turn.id,
        messages=(
            CanonicalMessage(
                agent_id="agent-atlas",
                operation_id=snapshot.operation.id,
                turn_id=turn.id,
                role=MessageRole.USER,
                content=(TextBlock("Compare the newest export."),),
            ),
        ),
        tools=registry.tool_definitions(),
    )
    model_call = await runtime.begin_model_call(
        snapshot.operation.id,
        turn.id,
        "mock:phase4-controller",
        request,
    )
    await runtime.record_model_response(
        snapshot.operation.id,
        model_call.id,
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=tuple(calls),
        ),
        next_phase=LoopPhase.VALIDATING_ACTION,
    )
    evidence: list[Evidence] = []
    for call in calls:
        _, capability = registry.resolve_tool(call.name)
        proposal = ActionProposal(
            operation_id=snapshot.operation.id,
            turn_id=turn.id,
            call_id=call.id,
            capability_id=capability.id,
            arguments=call.arguments,
            proposed_at=NOW,
        )
        accepted = await runtime.submit(proposal)
        assert accepted is not None
        evidence.append(accepted)
        await runtime.append_observation(await controller.project_observation(accepted))
    return controller, runtime, tuple(evidence)


def _comparison_call(
    label: str,
    left_id: str,
    right_id: str,
) -> ToolCall:
    return ToolCall(
        id=f"call-{label}-comparison-proposal",
        name="data_compare_tabular",
        arguments={
            "left_evidence_id": left_id,
            "right_evidence_id": right_id,
            "key_columns": ["id"],
            "compare_columns": ["name", "status"],
            "test_truncated": False,
        },
    )


async def test_file_resource_scope_rejects_before_action_proposal() -> None:
    controller, runtime, _ = await _snapshot_with_reads(label="file-scope")
    snapshot = await runtime.inspect("operation-file-scope-1")
    task_count = len(snapshot.tasks)
    call = ToolCall(
        id="call-file-escape",
        name="data_read_file",
        arguments={
            "source_id": "source-files",
            "resource_id": "resource-outside-root",
        },
    )

    result = await controller.validate_action(call, snapshot)

    assert isinstance(result, ActionRejection)
    assert result.code == "data.file.catalog_resource_missing"
    assert len((await runtime.inspect(snapshot.operation.id)).tasks) == task_count


async def test_comparison_rejects_missing_foreign_and_same_source_evidence() -> None:
    controller, runtime, evidence = await _snapshot_with_reads(label="main")
    snapshot = await runtime.inspect("operation-main-1")
    missing = await controller.validate_action(
        _comparison_call("missing", "evidence-missing", evidence[1].id),
        snapshot,
    )

    _, _, foreign_evidence = await _snapshot_with_reads(label="foreign")
    foreign = await controller.validate_action(
        _comparison_call("foreign", foreign_evidence[0].id, evidence[1].id),
        snapshot,
    )

    same_controller, same_runtime, same_evidence = await _snapshot_with_reads(
        label="same",
        query_source="source-files",
    )
    same = await same_controller.validate_action(
        _comparison_call("same", same_evidence[0].id, same_evidence[1].id),
        await same_runtime.inspect("operation-same-1"),
    )

    assert isinstance(missing, ActionRejection)
    assert missing.code == "data.compare.evidence_unavailable"
    assert isinstance(foreign, ActionRejection)
    assert foreign.code == "data.compare.evidence_unavailable"
    assert isinstance(same, ActionRejection)
    assert same.code == "data.compare.sources_not_distinct"


async def test_valid_distinct_source_evidence_becomes_comparison_proposal() -> None:
    controller, runtime, evidence = await _snapshot_with_reads(label="valid")
    call = _comparison_call("valid", evidence[0].id, evidence[1].id)

    result = await controller.validate_action(
        call,
        await runtime.inspect("operation-valid-1"),
    )

    assert isinstance(result, ActionProposal)
    assert result.capability_id == TABULAR_COMPARE_CAPABILITY_ID
    assert result.arguments["left_evidence_id"] == evidence[0].id
    assert result.arguments["right_evidence_id"] == evidence[1].id


async def test_readiness_requires_comparison_and_both_input_citations() -> None:
    controller, runtime, evidence = await _snapshot_with_reads(
        label="ready",
        with_comparison=True,
    )
    snapshot = await runtime.inspect("operation-ready-1")
    left, right, comparison = evidence

    no_comparison = await controller.evaluate_final_answer(
        f"Different. [evidence:{left.id}] [evidence:{right.id}]",
        snapshot,
    )
    missing_input = await controller.evaluate_final_answer(
        f"Different. [evidence:{comparison.id}] [evidence:{left.id}]",
        snapshot,
    )
    ready = await controller.evaluate_final_answer(
        (
            f"Different. [evidence:{comparison.id}] "
            f"[evidence:{left.id}] [evidence:{right.id}]"
        ),
        snapshot,
    )

    assert no_comparison.allowed is False
    assert missing_input.allowed is False
    assert ready.allowed is True


async def test_truncated_comparison_requires_explicit_partial_disclosure() -> None:
    controller, runtime, evidence = await _snapshot_with_reads(
        label="partial",
        with_comparison=True,
        comparison_truncated=True,
    )
    snapshot = await runtime.inspect("operation-partial-1")
    left, right, comparison = evidence
    citations = f"[evidence:{comparison.id}] [evidence:{left.id}] [evidence:{right.id}]"

    undisclosed = await controller.evaluate_final_answer(
        f"The sources differ. {citations}",
        snapshot,
    )
    disclosed = await controller.evaluate_final_answer(
        f"This is a partial comparison with limited coverage. {citations}",
        snapshot,
    )
    negated = await controller.evaluate_final_answer(
        f"This comparison is not partial; it is complete. {citations}",
        snapshot,
    )

    assert undisclosed.allowed is False
    assert undisclosed.missing_facts == (
        "an explicit partial or truncation disclosure",
    )
    assert negated.allowed is False
    assert negated.missing_facts == ("an explicit partial or truncation disclosure",)
    assert disclosed.allowed is True


async def test_reconciliation_language_requires_comparison_evidence() -> None:
    controller, runtime, evidence = await _snapshot_with_reads(
        label="reconcile",
        message="Reconcile the export against the customer table.",
    )
    snapshot = await runtime.inspect("operation-reconcile-1")

    readiness = await controller.evaluate_final_answer(
        f"They reconcile. [evidence:{evidence[0].id}] [evidence:{evidence[1].id}]",
        snapshot,
    )

    assert readiness.allowed is False
    assert readiness.missing_facts == (
        "accepted current-operation comparison evidence",
    )
