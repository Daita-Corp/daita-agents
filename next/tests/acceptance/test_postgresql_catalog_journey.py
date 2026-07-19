from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from daita import Agent, PostgreSQLSource
from daita._json import FrozenJsonObject
from daita.catalog import ResourceKind, catalog_resource_id
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolResultBlock,
)
from daita.loop.models import LoopExitKind

NOW = datetime(2026, 7, 19, 19, 0, tzinfo=timezone.utc)
PROFILE = ModelProfile(
    id="mock:postgresql-journey",
    context_window_tokens=32_768,
    max_output_tokens=4_096,
    supports_tools=True,
)


class JourneyProvider:
    provider_id = "mock:postgresql-journey"

    def __init__(self) -> None:
        self.script: list[ModelResponse] = []
        self.requests: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self.script:
            raise AssertionError("unexpected model call")
        return self.script.pop(0)


def _ids():
    counters: dict[str, int] = {}

    def factory(prefix: str) -> str:
        counters[prefix] = counters.get(prefix, 0) + 1
        return f"{prefix}-{counters[prefix]}"

    return factory


def _tool(call_id: str, name: str, arguments: dict[str, object]) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
    )


class Transaction:
    def __init__(self) -> None:
        self.started = False
        self.committed = False
        self.rolled_back = False

    async def start(self) -> None:
        self.started = True

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True


class Attribute:
    def __init__(self, name: str) -> None:
        self.name = name


class AsyncpgRecordLike:
    """Exercise asyncpg.Record's duck contract without importing asyncpg."""

    def __init__(self, *values: object) -> None:
        self._values = values

    def values(self) -> tuple[object, ...]:
        return self._values


class Cursor:
    def __aiter__(self):
        async def records():
            yield AsyncpgRecordLike(2, True)

        return records()


class Statement:
    def __init__(self, *, bounded: bool) -> None:
        self._bounded = bounded
        self.cursor_arguments: tuple[object, ...] | None = None

    def get_attributes(self) -> tuple[Attribute, ...]:
        return (Attribute("order_count"),)

    def cursor(self, *arguments: object, **kwargs: object) -> Cursor:
        assert self._bounded is True
        assert kwargs == {"prefetch": 101, "timeout": 5.0}
        self.cursor_arguments = arguments
        return Cursor()


class Connection:
    def __init__(self, *, query: bool) -> None:
        self.query = query
        self.transaction_record = Transaction()
        self.closed = False
        self.settings: list[tuple[str, tuple[object, ...]]] = []
        self.prepared_sql: list[str] = []
        self.bounded_statement: Statement | None = None

    def transaction(self, **kwargs: object) -> Transaction:
        assert kwargs == {"isolation": "repeatable_read", "readonly": True}
        return self.transaction_record

    async def execute(self, sql: str, *arguments: object) -> None:
        self.settings.append((sql, arguments))

    async def fetch(self, sql: str, *arguments: object):
        if "daita:postgresql.resources" in sql:
            return [
                {
                    "schema_name": "analytics",
                    "resource_name": "orders",
                    "resource_kind": "table",
                }
            ]
        if "daita:postgresql.columns" in sql:
            return [
                {
                    "column_name": "id",
                    "native_type": "bigint",
                    "type_schema": "pg_catalog",
                    "type_name": "int8",
                    "ordinal": 1,
                    "nullable": False,
                    "default_expression": None,
                    "primary_key_ordinal": 1,
                },
                {
                    "column_name": "status",
                    "native_type": "text",
                    "type_schema": "pg_catalog",
                    "type_name": "text",
                    "ordinal": 2,
                    "nullable": False,
                    "default_expression": None,
                    "primary_key_ordinal": None,
                },
            ]
        if "daita:postgresql.indexes" in sql:
            return []
        if "daita:postgresql.relationships" in sql:
            return []
        raise AssertionError(sql)

    async def prepare(self, sql: str, **kwargs: object) -> Statement:
        assert self.query is True
        assert kwargs == {"timeout": 5.0}
        self.prepared_sql.append(sql)
        bounded = "daita:postgresql.bounded_result" in sql
        if bounded:
            assert "pg_catalog.pg_column_size" in sql
            assert "pg_catalog.octet_length(pg_catalog.to_jsonb" in sql
            assert "LIMIT 101" in sql
        statement = Statement(bounded=bounded)
        if bounded:
            self.bounded_statement = statement
        return statement

    async def close(self) -> None:
        self.closed = True


class Asyncpg:
    def __init__(self, *connections: Connection) -> None:
        self.connections = list(connections)

    async def connect(self, **kwargs: object) -> Connection:
        assert kwargs["host"] == "db.internal"
        assert kwargs["database"] == "warehouse"
        assert kwargs["user"] == "reader"
        if not self.connections:
            raise AssertionError("unexpected PostgreSQL connection")
        return self.connections.pop(0)


def _inspect_observation(request: ModelRequest) -> str:
    for message in request.messages:
        for block in message.content:
            if isinstance(block, ToolResultBlock) and block.call_id == "call-inspect":
                observation = block.output["observation"]
                assert isinstance(observation, str)
                return observation
    raise AssertionError("catalog inspection was not projected to the model")


async def test_public_grounded_postgresql_journey_uses_qualified_catalog_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovery_connection = Connection(query=False)
    query_connection = Connection(query=True)
    asyncpg = Asyncpg(discovery_connection, query_connection)
    monkeypatch.setattr("daita.adapters.postgresql._load_asyncpg", lambda: asyncpg)

    provider = JourneyProvider()
    agent = await Agent.create(
        "atlas-postgresql",
        root=tmp_path / "state",
        model=provider,
        model_profile=PROFILE,
        clock=lambda: NOW,
        id_factory=_ids(),
    )
    registration = await agent.attach(
        PostgreSQLSource(
            host="db.internal",
            database="warehouse",
            username="reader",
            schemas=("analytics",),
            ssl_mode="require",
            name="Warehouse",
        )
    )
    orders_resource_id = catalog_resource_id(
        registration.id,
        ResourceKind.TABLE,
        "analytics.orders",
    )
    qualified_sql = (
        "SELECT COUNT(*) AS order_count FROM analytics.orders WHERE status = $1"
    )
    provider.script.extend(
        (
            _tool(
                "call-search",
                "catalog_search",
                {"query": "orders", "source_id": registration.id, "limit": 5},
            ),
            _tool(
                "call-inspect",
                "catalog_inspect",
                {"resource_id": orders_resource_id},
            ),
            _tool(
                "call-query",
                "data_query_postgresql",
                {
                    "source_id": registration.id,
                    "sql": qualified_sql,
                    "parameters": ["open"],
                },
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="There are 2 open orders. [evidence:evidence-3]",
            ),
        )
    )

    result = await agent.run(
        "How many open orders are in the warehouse?",
        session_id="session-postgresql",
    )
    snapshot = await agent.inspect(result.operation_id)

    assert result.kind is LoopExitKind.COMPLETED
    assert result.final_text == "There are 2 open orders. [evidence:evidence-3]"
    assert provider.script == []
    assert len(provider.requests) == 4
    assert [task.capability_id for task in snapshot.tasks] == [
        "catalog.search",
        "catalog.inspect",
        "data.postgresql.query",
    ]
    assert snapshot.loop_state.repair_count == 0
    assert [decision.allowed for decision in snapshot.readiness] == [True]
    assert not any(event.type == "action.rejected" for event in snapshot.events)

    inspect_evidence = snapshot.evidence[1]
    inspected_resource = inspect_evidence.payload["resource"]
    assert isinstance(inspected_resource, FrozenJsonObject)
    assert inspected_resource["native_identity"] == "analytics.orders"
    assert '"native_identity":"analytics.orders"' in _inspect_observation(
        provider.requests[2]
    )

    query_task = snapshot.tasks[-1]
    assert query_task.arguments["sql"] == qualified_sql
    query_evidence = snapshot.evidence[-1]
    assert query_evidence.id == "evidence-3"
    assert query_evidence.kind == "data.postgresql.query_result"
    assert query_evidence.accepted is True
    assert query_evidence.payload["resource_ids"] == (orders_resource_id,)
    assert query_evidence.payload["returned_rows"] == 1
    assert query_evidence.payload["truncated"] is False
    rows = query_evidence.payload["rows"]
    assert isinstance(rows, tuple)
    row = rows[0]
    assert isinstance(row, FrozenJsonObject)
    assert row["order_count"] == 2

    assert len(query_connection.prepared_sql) == 2
    assert "analytics.orders" in query_connection.prepared_sql[0]
    assert "daita:postgresql.bounded_result" in query_connection.prepared_sql[1]
    assert query_connection.bounded_statement is not None
    assert query_connection.bounded_statement.cursor_arguments == ("open",)
    assert discovery_connection.transaction_record.committed is True
    assert query_connection.transaction_record.committed is True
    assert discovery_connection.closed is True
    assert query_connection.closed is True
    assert asyncpg.connections == []

    await agent.close()
