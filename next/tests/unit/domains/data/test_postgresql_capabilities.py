from __future__ import annotations

from daita.capabilities import CapabilityRegistry, ExecutionRequest
from daita.domains.data import (
    POSTGRESQL_QUERY_CAPABILITY_ID,
    POSTGRESQL_QUERY_EVIDENCE_KIND,
    PostgreSQLQueryExecutor,
    PostgreSQLReadResult,
    SQLiteReadResult,
    postgresql_query_extension_declarations,
    project_result_rows,
)


def test_relational_result_types_keep_adapter_specific_public_names() -> None:
    assert SQLiteReadResult.__name__ == "SQLiteReadResult"
    assert PostgreSQLReadResult.__name__ == "PostgreSQLReadResult"


class Backend:
    def __init__(self, result: PostgreSQLReadResult) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def execute_read(self, **kwargs: object) -> PostgreSQLReadResult:
        self.calls.append(kwargs)
        return self.result


async def test_postgresql_query_uses_same_bounded_read_contract_as_sqlite() -> None:
    backend = Backend(
        PostgreSQLReadResult(
            source_id="source-postgres",
            canonical_sql="SELECT id FROM public.orders",
            sql_fingerprint="sha256:" + "a" * 64,
            resource_ids=("resource-orders",),
            resource_revisions=(("resource-orders", "sha256:" + "b" * 64),),
            source_revision="catalog:sha256:" + "c" * 64,
            columns=("id",),
            projection=project_result_rows(
                ({"id": 1},),
                max_rows=100,
                max_bytes=65_536,
            ),
        )
    )
    executor = PostgreSQLQueryExecutor("agent-1", backend)
    extension = postgresql_query_extension_declarations()
    registry = CapabilityRegistry(
        capabilities=extension.capabilities,
        executors=(executor,),
        tool_views=extension.tool_views,
    )
    request = ExecutionRequest(
        operation_id="operation-1",
        task_id="task-1",
        turn_id="turn-1",
        capability_id=POSTGRESQL_QUERY_CAPABILITY_ID,
        executor_id=executor.executor_id,
        attempt=1,
        fencing_token=1,
        arguments={
            "source_id": "source-postgres",
            "sql": "SELECT id FROM public.orders",
        },
    )

    candidate = await executor.execute(request)

    assert [tool.name for tool in registry.tool_definitions()] == [
        "data_query_postgresql"
    ]
    assert candidate.kind == POSTGRESQL_QUERY_EVIDENCE_KIND
    assert backend.calls == [
        {
            "agent_id": "agent-1",
            "source_id": "source-postgres",
            "sql": "SELECT id FROM public.orders",
            "parameters": (),
            "max_rows": 100,
            "max_bytes": 65_536,
        }
    ]
    registry.validate_evidence(POSTGRESQL_QUERY_CAPABILITY_ID, candidate)
