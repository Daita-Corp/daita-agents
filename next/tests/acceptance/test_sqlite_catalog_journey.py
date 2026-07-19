from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from daita import Agent, SQLiteSource
from daita._json import FrozenJsonObject
from daita.catalog import ResourceKind, catalog_resource_id
from daita.llm.models import (
    FinishReason,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
)
from daita.loop.models import LoopExitKind

NOW = datetime(2026, 7, 18, 22, 0, tzinfo=timezone.utc)


class JourneyProvider:
    provider_id = "mock:sqlite-journey"

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


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL
            );
            INSERT INTO customers (name, status) VALUES
                ('Ada', 'active'),
                ('Grace', 'active'),
                ('Linus', 'inactive');
            """)


async def test_public_persistent_grounded_sqlite_journey(tmp_path: Path) -> None:
    database = tmp_path / "customers.db"
    _database(database)
    provider = JourneyProvider()
    agent = await Agent.create(
        "atlas",
        root=tmp_path / "state",
        model=provider,
        clock=lambda: NOW,
        id_factory=_ids(),
    )
    registration = await agent.attach(SQLiteSource(database, name="Customers"))
    customer_resource_id = catalog_resource_id(
        registration.id,
        ResourceKind.TABLE,
        "main.customers",
    )
    provider.script.extend(
        (
            _tool(
                "call-search",
                "catalog_search",
                {"query": "customers", "source_id": registration.id, "limit": 5},
            ),
            _tool(
                "call-inspect",
                "catalog_inspect",
                {"resource_id": customer_resource_id},
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="There are probably two active customers.",
            ),
            _tool(
                "call-invalid",
                "data_query_sqlite",
                {
                    "source_id": registration.id,
                    "sql": "DELETE FROM customers WHERE status = ?",
                    "parameters": ["inactive"],
                },
            ),
            _tool(
                "call-query",
                "data_query_sqlite",
                {
                    "source_id": registration.id,
                    "sql": (
                        "SELECT COUNT(*) AS customer_count FROM customers "
                        "WHERE status = ?"
                    ),
                    "parameters": ["active"],
                },
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="There are 2 active customers. [evidence:evidence-3]",
            ),
        )
    )

    result = await agent.run(
        "How many active customers are there?",
        session_id="session-atlas",
    )
    snapshot = await agent.inspect(result.operation_id)

    assert result.kind is LoopExitKind.COMPLETED
    assert result.final_text == "There are 2 active customers. [evidence:evidence-3]"
    assert provider.script == []
    assert [task.capability_id for task in snapshot.tasks] == [
        "catalog.search",
        "catalog.inspect",
        "data.sqlite.query",
    ]
    query_evidence = snapshot.evidence[-1]
    assert query_evidence.id == "evidence-3"
    assert query_evidence.accepted is True
    query_rows = query_evidence.payload["rows"]
    assert isinstance(query_rows, tuple)
    query_row = query_rows[0]
    assert isinstance(query_row, FrozenJsonObject)
    assert query_row["customer_count"] == 2
    assert query_evidence.payload["trust_classification"] == "untrusted_external_data"
    assert [decision.allowed for decision in snapshot.readiness] == [False, True]
    event_types = {event.type for event in snapshot.events}
    assert {
        "action.rejected",
        "evidence.accepted",
        "model_call.started",
        "readiness.recorded",
        "task.created",
    } <= event_types
    assert any(
        event.type == "action.rejected"
        and event.payload["code"] == "data.sql.mutation_not_allowed"
        for event in snapshot.events
    )
    for request in provider.requests:
        system_block = request.messages[0].content[0]
        assert isinstance(system_block, TextBlock)
        assert "UNTRUSTED_CATALOG_CONTEXT" in system_block.text
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 3

    await agent.close()
    reopened = await Agent.open("atlas", root=tmp_path / "state", clock=lambda: NOW)
    recovered = await reopened.inspect(result.operation_id)
    transcript = await reopened.transcript("session-atlas")
    await reopened.close()

    assert recovered == snapshot
    assert transcript.session.id == "session-atlas"
    assert transcript.operation_ids == (result.operation_id,)
