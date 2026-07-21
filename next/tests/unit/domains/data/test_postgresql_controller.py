from __future__ import annotations

from datetime import datetime, timezone

from daita._json import FrozenJsonObject
from daita.domains.data import (
    DataDomainController,
    PostgreSQLReadResult,
    ResourceSchema,
    postgresql_query_declarations,
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
    TriggerKind,
)
from daita.operations.runtime import OperationRuntime

NOW = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc)
RESOURCE_REVISION = "sha256:" + ("a" * 64)


class Backend:
    async def execute_read(
        self,
        *,
        agent_id: str,
        source_id: str,
        sql: str,
        parameters: tuple[object, ...],
        max_rows: int,
        max_bytes: int,
    ) -> PostgreSQLReadResult:
        raise AssertionError("validation must not invoke source I/O")


class Catalog:
    def __init__(self, adapter_id: str = "postgresql") -> None:
        self.adapter_id = adapter_id

    async def source_routing_facts(
        self,
        agent_id: str,
        configuration_flags: tuple[str, ...],
    ) -> tuple[FrozenJsonObject, ...]:
        assert agent_id == "agent-1"
        return (
            FrozenJsonObject.from_mapping(
                {
                    "adapter_id": self.adapter_id,
                    "configuration_flags": {
                        flag: False for flag in configuration_flags
                    },
                    "source_id": "source-postgresql",
                }
            ),
        )

    async def source_adapter_id(
        self,
        agent_id: str,
        source_id: str,
    ) -> str | None:
        assert agent_id == "agent-1"
        return self.adapter_id

    async def resource_identity(
        self,
        agent_id: str,
        resource_id: str,
    ) -> tuple[str, str, str] | None:
        assert agent_id == "agent-1"
        if resource_id == "resource-orders":
            return ("source-postgresql", "table", RESOURCE_REVISION)
        return None

    async def resource_schemas(
        self,
        agent_id: str,
        source_id: str,
    ) -> tuple[ResourceSchema, ...]:
        assert agent_id == "agent-1"
        return (
            ResourceSchema(
                resource_id="resource-orders",
                source_id=source_id,
                name="orders",
                aliases=("public.orders",),
                columns=("id", "status"),
                revision=RESOURCE_REVISION,
                source_revision="postgresql-revision-1",
                resource_kind="table",
                sensitivity_class="confidential",
                column_declared_types=(("id", "integer"), ("status", "text")),
            ),
        )

    async def is_current_tabular_file(
        self,
        agent_id: str,
        source_id: str,
        resource_id: str,
    ) -> bool:
        return False

    async def is_writable_sqlite_source(
        self,
        agent_id: str,
        source_id: str,
    ) -> bool:
        return False


def _ids():
    counts: dict[str, int] = {}

    def factory(prefix: str) -> str:
        counts[prefix] = counts.get(prefix, 0) + 1
        return f"{prefix}-{counts[prefix]}"

    return factory


async def _snapshot(call: ToolCall):
    declarations = postgresql_query_declarations("agent-1", Backend())
    from daita.capabilities import CapabilityRegistry

    registry = CapabilityRegistry(
        capabilities=declarations.capabilities,
        executors=declarations.executors,
        tool_views=declarations.tool_views,
    )
    runtime = OperationRuntime(
        clock=lambda: NOW,
        id_factory=_ids(),
        capabilities=registry,
    )
    snapshot = await runtime.begin(
        AgentTrigger(
            id=f"trigger-{call.id}",
            agent_id="agent-1",
            kind=TriggerKind.USER,
            source_id="user:test",
            payload={"message": "Find open orders"},
            created_at=NOW,
        )
    )
    turn = await runtime.begin_turn(snapshot.operation.id)
    request = ModelRequest(
        operation_id=snapshot.operation.id,
        turn_id=turn.id,
        messages=(
            CanonicalMessage(
                agent_id="agent-1",
                operation_id=snapshot.operation.id,
                turn_id=turn.id,
                role=MessageRole.USER,
                content=(TextBlock("Find open orders"),),
            ),
        ),
        tools=registry.tool_definitions(),
    )
    model_call = await runtime.begin_model_call(
        snapshot.operation.id,
        turn.id,
        "mock:data",
        request,
    )
    await runtime.record_model_response(
        snapshot.operation.id,
        model_call.id,
        ModelResponse(finish_reason=FinishReason.TOOL_CALLS, tool_calls=(call,)),
        next_phase=LoopPhase.VALIDATING_ACTION,
    )
    return registry, await runtime.inspect(snapshot.operation.id)


async def test_controller_routes_postgresql_proposals_to_postgresql_validation() -> (
    None
):
    valid_call = ToolCall(
        id="valid",
        name="data_query_postgresql",
        arguments={
            "source_id": "source-postgres",
            "sql": "SELECT id FROM public.orders WHERE status = $1",
            "parameters": ["open"],
        },
    )
    registry, snapshot = await _snapshot(valid_call)
    controller = DataDomainController(registry, Catalog(), clock=lambda: NOW)

    valid = await controller.validate_action(valid_call, snapshot)

    assert isinstance(valid, ActionProposal)
    assert valid.validation_facts.source_ids == ("source-postgres",)
    assert valid.validation_facts.source_revisions == (
        ("source-postgres", "postgresql-revision-1"),
    )
    assert valid.validation_facts.resource_revisions == (
        ("resource-orders", RESOURCE_REVISION),
    )
    assert valid.validation_facts.sensitivity_class == "confidential"
    assert valid.validation_facts.freshness_state == "current"

    invalid_call = ToolCall(
        id="invalid",
        name="data_query_postgresql",
        arguments={
            "source_id": "source-postgres",
            "sql": "SELECT id FROM orders WHERE status = ?",
            "parameters": ["open"],
        },
    )
    registry, snapshot = await _snapshot(invalid_call)
    invalid = await DataDomainController(
        registry, Catalog(), clock=lambda: NOW
    ).validate_action(invalid_call, snapshot)

    assert isinstance(invalid, ActionRejection)
    assert invalid.code == "data.sql.parameter_style_invalid"
    assert invalid.details["anonymous_placeholders"] == 1
    issue_codes = invalid.details["issue_codes"]
    assert isinstance(issue_codes, tuple)
    assert "parameter_style_invalid" in issue_codes

    missing_column_call = ToolCall(
        id="missing-column",
        name="data_query_postgresql",
        arguments={
            "source_id": "source-postgres",
            "sql": "SELECT sttaus FROM public.orders",
            "parameters": [],
        },
    )
    registry, snapshot = await _snapshot(missing_column_call)
    missing_column = await DataDomainController(
        registry, Catalog(), clock=lambda: NOW
    ).validate_action(missing_column_call, snapshot)

    assert isinstance(missing_column, ActionRejection)
    assert missing_column.code == "data.sql.missing_column"
    assert missing_column.details["column"] == "sttaus"
    assert missing_column.details["resource_id"] == "resource-orders"
    assert missing_column.details["candidates"] == ("status",)


async def test_controller_rejects_postgresql_tool_for_sqlite_source_before_task() -> (
    None
):
    call = ToolCall(
        id="adapter-mismatch",
        name="data_query_postgresql",
        arguments={
            "source_id": "source-sqlite",
            "sql": "SELECT id FROM public.orders",
            "parameters": [],
        },
    )
    registry, snapshot = await _snapshot(call)

    result = await DataDomainController(
        registry,
        Catalog("sqlite"),
        clock=lambda: NOW,
    ).validate_action(call, snapshot)

    assert isinstance(result, ActionRejection)
    assert result.code == "data.sql.source_adapter_mismatch"
    assert result.details["expected_adapter_id"] == "postgresql"
    assert not snapshot.tasks
