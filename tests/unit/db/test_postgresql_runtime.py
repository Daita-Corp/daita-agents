from daita.db import DbRequest, DbRuntime
from daita.db.runtime import DbRuntimeGovernanceBlocked
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.postgresql import PostgreSQLPlugin


def _postgres() -> PostgreSQLPlugin:
    return PostgreSQLPlugin(connection_string="postgresql://localhost/testdb")


async def test_postgresql_registers_provider_neutral_db_capabilities():
    runtime = DbRuntime(plugins=(_postgres(),))

    inspection = await runtime.inspect()

    assert inspection.plugin_ids == ("postgresql",)
    assert "postgresql:db.schema.inspect" in inspection.capability_ids
    assert "postgresql:db.sql.validate" in inspection.capability_ids
    assert "postgresql:db.sql.execute_read" in inspection.capability_ids
    assert "postgresql:db.sql.execute_write" in inspection.capability_ids
    assert "postgresql:db.sql.explain" in inspection.capability_ids
    assert "postgresql.sql.execute_read" in inspection.executor_ids
    assert "postgresql:query.result" in inspection.evidence_schema_kinds


def test_db_runtime_selects_postgresql_owned_capabilities_for_postgresql_source():
    runtime = DbRuntime(plugins=(CatalogPlugin(auto_persist=False), _postgres()))

    contract = runtime.build_contract(DbRequest("How many orders are there?"))

    selected = {
        item["id"]: item
        for item in contract.metadata["selected_capabilities"]
        if item["id"].startswith("db.")
    }
    assert selected["db.sql.validate"]["owner"] == "postgresql"
    assert selected["db.sql.execute_read"]["owner"] == "postgresql"
    assert selected["db.sql.execute_read"]["executor"] == (
        "postgresql.sql.execute_read"
    )


async def test_postgresql_schema_inspect_returns_typed_evidence_through_runtime():
    postgres = _postgres()

    async def fake_tables():
        return ["customers", "orders"]

    async def fake_describe(table):
        return [
            {
                "column_name": "id",
                "data_type": "integer",
                "is_nullable": "NO",
                "is_primary_key": True,
            },
            {
                "column_name": f"{table}_name",
                "data_type": "text",
                "is_nullable": "YES",
                "is_primary_key": False,
            },
        ]

    async def fake_foreign_keys():
        return [
            {
                "source_table": "orders",
                "source_column": "customer_id",
                "target_table": "customers",
                "target_column": "id",
            }
        ]

    postgres.tables = fake_tables
    postgres.describe = fake_describe
    postgres.foreign_keys = fake_foreign_keys
    postgres.connect = _noop_connect
    runtime = DbRuntime(plugins=(postgres,))

    evidence = await runtime.execute_capability(
        "db.schema.inspect",
        owner="postgresql",
        operation_type="schema.query",
    )

    assert evidence[0].kind == "schema.asset_profile"
    assert evidence[0].owner == "postgresql"
    assert evidence[0].payload["database_type"] == "postgresql"
    assert evidence[0].payload["table_count"] == 2
    assert evidence[0].payload["tables"][0]["columns"][0]["is_primary_key"] is True
    assert evidence[0].payload["foreign_keys"][0]["source_table"] == "orders"


async def test_postgresql_sql_executors_return_typed_evidence_without_live_db():
    postgres = _postgres()
    captured = {}

    async def fake_query(sql, params=None):
        captured.setdefault("queries", []).append((sql, params))
        if sql.startswith("EXPLAIN"):
            return [{"QUERY PLAN": "Seq Scan on orders"}]
        return [{"count": 2}]

    async def fake_execute(sql, params=None):
        captured["execute"] = (sql, params)
        return 3

    postgres.query = fake_query
    postgres.execute = fake_execute
    postgres.connect = _noop_connect
    runtime = DbRuntime(plugins=(postgres,))

    validation = await runtime.execute_capability(
        "db.sql.validate",
        owner="postgresql",
        operation_type="data.query",
        input={"sql": "SELECT COUNT(*) AS count FROM orders"},
    )
    query = await runtime.execute_capability(
        "db.sql.execute_read",
        owner="postgresql",
        operation_type="data.query",
        input={"sql": "SELECT COUNT(*) AS count FROM orders"},
    )
    plan = await runtime.execute_capability(
        "db.sql.explain",
        owner="postgresql",
        operation_type="data.query",
        input={"sql": "SELECT COUNT(*) AS count FROM orders"},
    )
    try:
        await runtime.execute_capability(
            "db.sql.execute_write",
            owner="postgresql",
            operation_type="write.execute",
            input={
                "sql": "UPDATE orders SET total = $1 WHERE id = $2",
                "params": [5, 1],
            },
        )
    except DbRuntimeGovernanceBlocked as exc:
        snapshot = await runtime.inspect_operation(exc.operation.id)
        await runtime.approval_channel.approve(
            snapshot.approval_requests[0].approval_id
        )
        resumed = await runtime.resume_operation(exc.operation.id)
        write = tuple(
            item for item in resumed.evidence if item.kind == "write.execution"
        )

    assert validation[0].payload["valid"] is True
    assert validation[0].payload["tables"] == ["orders"]
    query_result = next(item for item in query if item.kind == "query.result")
    assert query_result.payload["rows"] == [{"count": 2}]
    assert query_result.payload["sql"].endswith("LIMIT 50")
    assert plan[0].kind == "sql.explain.plan"
    assert plan[0].payload["plan"] == [{"QUERY PLAN": "Seq Scan on orders"}]
    assert write[0].kind == "write.execution"
    assert write[0].payload["affected_rows"] == 3


async def _noop_connect() -> None:
    return None
