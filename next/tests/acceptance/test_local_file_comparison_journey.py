from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
from typing import cast

from daita import Agent, LocalDirectorySource, SQLiteSource
from daita._json import FrozenJsonObject
from daita.catalog import ResourceKind, catalog_resource_id
from daita.llm.models import (
    FinishReason,
    ModelRequest,
    ModelResponse,
    ToolCall,
)
from daita.loop.models import LoopExitKind

NOW = datetime(2026, 7, 19, 2, 0, tzinfo=timezone.utc)
OLD_MTIME = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
NEW_MTIME = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)


class JourneyProvider:
    provider_id = "mock:local-file-comparison-journey"

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
    return _tools(ToolCall(id=call_id, name=name, arguments=arguments))


def _tools(*calls: ToolCall) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=calls,
    )


def _write_export(
    path: Path, rows: tuple[tuple[object, ...], ...], mtime: datetime
) -> None:
    lines = ["id,name,status", *(",".join(str(value) for value in row) for row in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    nanoseconds = int(mtime.timestamp() * 1_000_000_000)
    os.utime(path, ns=(nanoseconds, nanoseconds))


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL
            );
            INSERT INTO customers (id, name, status) VALUES
                (1, 'Ada', 'active'),
                (2, 'Grace', 'active'),
                (3, 'Linus', 'inactive');
            """)


def _file_modified_at(inspect_payload: FrozenJsonObject) -> str:
    facets = inspect_payload["facets"]
    assert isinstance(facets, tuple)
    file_facet = next(
        facet
        for facet in facets
        if isinstance(facet, FrozenJsonObject) and facet["kind"] == "file"
    )
    payload = file_facet["payload"]
    assert isinstance(payload, FrozenJsonObject)
    modified_at = payload["modified_at"]
    assert isinstance(modified_at, str)
    return modified_at


def _frozen_object(value: object) -> FrozenJsonObject:
    assert isinstance(value, FrozenJsonObject)
    return value


def _frozen_objects(value: object) -> tuple[FrozenJsonObject, ...]:
    assert isinstance(value, tuple)
    assert all(isinstance(item, FrozenJsonObject) for item in value)
    return cast(tuple[FrozenJsonObject, ...], value)


async def test_public_cross_source_local_file_comparison_journey(
    tmp_path: Path,
) -> None:
    exports = tmp_path / "exports"
    exports.mkdir()
    old_export = exports / "customers-2026-06.csv"
    new_export = exports / "customers-2026-07.csv"
    _write_export(
        old_export,
        (
            (1, "Ada", "active"),
            (2, "Grace", "active"),
        ),
        OLD_MTIME,
    )
    _write_export(
        new_export,
        (
            (1, "Ada", "active"),
            (2, "Grace", "inactive"),
            (4, "Margaret", "active"),
        ),
        NEW_MTIME,
    )
    database = tmp_path / "customers.db"
    _database(database)

    provider = JourneyProvider()
    state_root = tmp_path / "state"
    agent = await Agent.create(
        "atlas",
        root=state_root,
        model=provider,
        clock=lambda: NOW,
        id_factory=_ids(),
    )
    file_source = await agent.attach(
        LocalDirectorySource(exports, name="Customer exports")
    )
    database_source = await agent.attach(SQLiteSource(database, name="Customers"))
    old_resource_id = catalog_resource_id(
        file_source.id,
        ResourceKind.FILE,
        old_export.name,
    )
    new_resource_id = catalog_resource_id(
        file_source.id,
        ResourceKind.FILE,
        new_export.name,
    )
    table_resource_id = catalog_resource_id(
        database_source.id,
        ResourceKind.TABLE,
        "main.customers",
    )
    final_text = (
        "Compared 3 export rows with 3 database rows: customer 2 is inactive in "
        "the export but active in SQLite; customer 4 exists only in the export; "
        "customer 3 exists only in SQLite. Reads were bounded to 100 rows and "
        "65,536 bytes per source. [evidence:evidence-5] [evidence:evidence-6] "
        "[evidence:evidence-7]"
    )
    provider.script.extend(
        (
            _tool(
                "call-search",
                "catalog_search",
                {"query": "customers", "limit": 10},
            ),
            _tools(
                ToolCall(
                    id="call-inspect-old",
                    name="catalog_inspect",
                    arguments={"resource_id": old_resource_id},
                ),
                ToolCall(
                    id="call-inspect-new",
                    name="catalog_inspect",
                    arguments={"resource_id": new_resource_id},
                ),
                ToolCall(
                    id="call-inspect-table",
                    name="catalog_inspect",
                    arguments={"resource_id": table_resource_id},
                ),
            ),
            _tool(
                "call-read-newest",
                "data_read_file",
                {
                    "source_id": file_source.id,
                    "resource_id": new_resource_id,
                },
            ),
            _tool(
                "call-query-customers",
                "data_query_sqlite",
                {
                    "source_id": database_source.id,
                    "sql": (
                        "SELECT CAST(id AS TEXT) AS id, name, status "
                        "FROM customers ORDER BY id"
                    ),
                },
            ),
            _tool(
                "call-compare",
                "data_compare_tabular",
                {
                    "left_evidence_id": "evidence-5",
                    "right_evidence_id": "evidence-6",
                    "key_columns": ["id"],
                    "compare_columns": ["name", "status"],
                },
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text=(
                    "Customer 2 differs, and customers 3 and 4 appear on only one "
                    "side."
                ),
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text=final_text),
        )
    )

    result = await agent.run(
        "Find the newest customer export, compare it with the customer table, "
        "and explain every discrepancy.",
        session_id="session-cross-source",
    )
    snapshot = await agent.inspect(result.operation_id)

    assert result.kind is LoopExitKind.COMPLETED
    assert result.final_text == final_text
    assert provider.script == []
    assert [task.capability_id for task in snapshot.tasks] == [
        "catalog.search",
        "catalog.inspect",
        "catalog.inspect",
        "catalog.inspect",
        "data.file.read",
        "data.sqlite.query",
        "data.tabular.compare",
    ]
    assert len(snapshot.evidence) == 7
    assert all(evidence.accepted for evidence in snapshot.evidence)

    search, old_inspect, new_inspect, table_inspect = snapshot.evidence[:4]
    hits = search.payload["hits"]
    assert isinstance(hits, tuple)
    assert {hit["source_id"] for hit in hits} == {
        file_source.id,
        database_source.id,
    }
    old_resource = _frozen_object(old_inspect.payload["resource"])
    new_resource = _frozen_object(new_inspect.payload["resource"])
    table_resource = _frozen_object(table_inspect.payload["resource"])
    assert old_resource["resource_id"] == old_resource_id
    assert new_resource["resource_id"] == new_resource_id
    assert table_resource["resource_id"] == table_resource_id
    assert (
        _file_modified_at(_frozen_object(old_inspect.payload)) == OLD_MTIME.isoformat()
    )
    assert (
        _file_modified_at(_frozen_object(new_inspect.payload)) == NEW_MTIME.isoformat()
    )

    file_evidence, query_evidence, comparison = snapshot.evidence[4:]
    assert file_evidence.id == "evidence-5"
    assert file_evidence.kind == "data.file.read_result"
    assert file_evidence.payload["source_id"] == file_source.id
    assert file_evidence.payload["resource_id"] == new_resource_id
    assert file_evidence.payload["returned_rows"] == 3
    assert file_evidence.payload["complete"] is True
    assert file_evidence.payload["truncated"] is False
    assert query_evidence.id == "evidence-6"
    assert query_evidence.payload["source_id"] == database_source.id
    assert query_evidence.payload["resource_ids"] == (table_resource_id,)
    assert query_evidence.payload["returned_rows"] == 3

    assert comparison.id == "evidence-7"
    assert comparison.kind == "data.tabular.comparison"
    assert comparison.blob_id is not None
    assert comparison.content_hash == comparison.payload["artifact_digest"]
    assert comparison.payload["complete"] is True
    assert comparison.payload["truncated"] is False
    assert comparison.payload["total_discrepancies"] == 3
    counts = _frozen_object(comparison.payload["counts"])
    assert counts["left_rows"] == 3
    assert counts["right_rows"] == 3
    assert counts["matched_keys"] == 2
    assert counts["equal_rows"] == 1
    assert counts["different_rows"] == 1
    assert counts["left_only"] == 1
    assert counts["right_only"] == 1
    assert counts["value_mismatches"] == 1
    discrepancy_sample = _frozen_objects(comparison.payload["discrepancy_sample"])
    assert tuple(item["kind"] for item in discrepancy_sample) == (
        "left_only",
        "right_only",
        "value_mismatch",
    )
    mismatch_key = _frozen_object(discrepancy_sample[-1]["key"])
    assert mismatch_key["id"] == "2"
    assert discrepancy_sample[-1]["column"] == "status"
    assert discrepancy_sample[-1]["left_value"] == "inactive"
    assert discrepancy_sample[-1]["right_value"] == "active"

    left_provenance = _frozen_object(comparison.payload["left"])
    right_provenance = _frozen_object(comparison.payload["right"])
    assert left_provenance["evidence_id"] == file_evidence.id
    assert left_provenance["source_id"] == file_source.id
    left_revisions = _frozen_objects(left_provenance["resource_revisions"])
    right_revisions = _frozen_objects(right_provenance["resource_revisions"])
    assert left_revisions[0]["resource_id"] == new_resource_id
    assert right_provenance["evidence_id"] == query_evidence.id
    assert right_provenance["source_id"] == database_source.id
    assert right_revisions[0]["resource_id"] == table_resource_id

    assert [decision.allowed for decision in snapshot.readiness] == [False, True]
    assert snapshot.readiness[0].missing_facts == (
        "citations to one comparison and both accepted source inputs",
    )
    file_tasks = [
        task for task in snapshot.tasks if task.capability_id == "data.file.read"
    ]
    assert len(file_tasks) == 1
    assert file_tasks[0].arguments["resource_id"] == new_resource_id
    assert all(
        task.arguments.get("resource_id") != old_resource_id for task in file_tasks
    )

    event_types = {event.type for event in snapshot.events}
    assert {
        "evidence.accepted",
        "executor.completed",
        "model_call.started",
        "readiness.recorded",
        "task.created",
    } <= event_types
    assert all(
        evidence.payload["trust_classification"] == "untrusted_external_data"
        for evidence in (file_evidence, query_evidence, comparison)
    )

    await agent.close()
    reopened = await Agent.open("atlas", root=state_root, clock=lambda: NOW)
    recovered = await reopened.inspect(result.operation_id)
    transcript = await reopened.transcript("session-cross-source")
    await reopened.close()

    assert recovered == snapshot
    assert recovered.evidence[-1].blob_id == comparison.blob_id
    assert recovered.evidence[-1].content_hash == comparison.content_hash
    assert transcript.session.id == "session-cross-source"
    assert transcript.operation_ids == (result.operation_id,)
