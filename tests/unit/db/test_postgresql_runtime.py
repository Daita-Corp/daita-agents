import asyncio
from datetime import date, datetime, time, timezone
from decimal import Decimal
from uuid import UUID

import pytest

from daita.core.exceptions import ValidationError
from daita.db import DbRequest, DbRuntime
from daita.db.runtime import DbRuntimeGovernanceBlocked
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.postgresql import PostgreSQLPlugin
from daita.plugins.sql_params import SQLParameterCoercionError, coerce_sql_params


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
    assert "postgresql:db.column_values.profile" in inspection.capability_ids
    assert "postgresql.sql.execute_read" in inspection.executor_ids
    assert "postgresql:query.result" in inspection.evidence_schema_kinds
    assert "postgresql:column_values.profile" in inspection.evidence_schema_kinds
    assert {
        capability.id
        for capability in runtime.registry.capabilities
        if capability.owner == "postgresql" and capability.concurrent_safe
    } == {"db.sql.execute_read"}
    profile_capability = runtime.registry.get_capability(
        "db.column_values.profile",
        owner="postgresql",
    )
    policy = profile_capability.metadata["profile_policy"]
    assert policy["bounded_aggregate"] is True
    assert policy["fingerprint_only_supported"] is True
    assert policy["profile_only_readable_tables"] is True


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


async def test_postgresql_runtime_supports_two_safe_reads_through_one_pool():
    postgres = _postgres()
    both_started = asyncio.Event()
    release = asyncio.Event()
    active = 0

    class Connection:
        async def fetch(self, sql, *params):
            nonlocal active
            active += 1
            if active == 2:
                both_started.set()
            await release.wait()
            try:
                value = 1 if "1 AS value" in sql else 2
                return [{"value": value}]
            finally:
                active -= 1

    class Acquire:
        async def __aenter__(self):
            return Connection()

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    class Pool:
        def acquire(self):
            return Acquire()

        async def close(self):
            return None

    postgres._pool = Pool()
    runtime = DbRuntime(plugins=(postgres,))
    first = asyncio.create_task(
        runtime.execute_capability(
            "db.sql.execute_read",
            owner="postgresql",
            operation_type="data.query",
            input={"sql": "SELECT 1 AS value"},
        )
    )
    second = asyncio.create_task(
        runtime.execute_capability(
            "db.sql.execute_read",
            owner="postgresql",
            operation_type="data.query",
            input={"sql": "SELECT 2 AS value"},
        )
    )
    try:
        await both_started.wait()
        assert active == 2
        release.set()
        first_result, second_result = await asyncio.gather(first, second)
        with pytest.raises(ValidationError):
            await runtime.execute_capability(
                "db.sql.execute_read",
                owner="postgresql",
                operation_type="data.query",
                input={"sql": "DELETE FROM orders"},
            )
    finally:
        release.set()
        await runtime.teardown()

    assert next(item for item in first_result if item.kind == "query.result").payload[
        "rows"
    ] == [{"value": 1}]
    assert next(item for item in second_result if item.kind == "query.result").payload[
        "rows"
    ] == [{"value": 2}]


def test_postgresql_typed_parameter_coercion_covers_json_restored_values():
    params = coerce_sql_params(
        [
            "2026-06-24T18:22:26.382861+00:00",
            "2026-06-24",
            "18:22:26",
            "6f78f12b-6b6d-42f6-b5bf-f0d25e7cb5ba",
            "12.34",
            '{"ok": true}',
            {"nested": ["value"]},
            "42",
            "3.5",
            "true",
            123,
        ],
        [
            {"ref": "monitor.state.cursor.last_created_at", "db_type": "timestamptz"},
            {"db_type": "date"},
            {"db_type": "time"},
            {"db_type": "uuid"},
            {"db_type": "numeric"},
            {"db_type": "jsonb"},
            {"db_type": "jsonb"},
            {"db_type": "integer"},
            {"db_type": "double precision"},
            {"db_type": "boolean"},
            {"db_type": "text"},
        ],
        dialect="postgresql",
    )

    assert params == [
        datetime(2026, 6, 24, 18, 22, 26, 382861, tzinfo=timezone.utc),
        date(2026, 6, 24),
        time(18, 22, 26),
        UUID("6f78f12b-6b6d-42f6-b5bf-f0d25e7cb5ba"),
        Decimal("12.34"),
        '{"ok": true}',
        '{"nested":["value"]}',
        42,
        3.5,
        True,
        "123",
    ]


def test_postgresql_invalid_typed_parameter_error_is_actionable():
    with pytest.raises(SQLParameterCoercionError) as exc_info:
        coerce_sql_params(
            ["not-a-date"],
            [
                {
                    "ref": "monitor.state.cursor.last_created_at",
                    "db_type": "timestamptz",
                }
            ],
            dialect="postgresql",
        )

    message = str(exc_info.value)
    assert "monitor.state.cursor.last_created_at" in message
    assert "timestamptz" in message
    assert "raw type str" in message


async def test_postgresql_executor_coerces_monitor_param_before_driver_call():
    postgres = _postgres()
    captured = {}

    async def fake_query(sql, params=None):
        captured["query"] = (sql, params)
        return [{"id": 1}]

    postgres.query = fake_query
    postgres.connect = _noop_connect

    await postgres._execute_sql_read(
        {
            "sql": (
                "select * from runtime_operations "
                "where created_at > $1 order by created_at asc limit 100"
            ),
            "params": ["2026-06-24T18:22:26.382861+00:00"],
            "param_specs": [
                {
                    "ref": "monitor.state.cursor.last_created_at",
                    "source": "monitor_state",
                    "path": ["cursor", "last_created_at"],
                    "table": "runtime_operations",
                    "column": "created_at",
                    "db_type": "timestamp with time zone",
                    "native_type": "datetime",
                    "dialect": "postgresql",
                    "nullable": False,
                }
            ],
        }
    )

    bound = captured["query"][1][0]
    assert isinstance(bound, datetime)
    assert bound.tzinfo is not None


async def test_postgresql_column_value_profile_uses_bounded_aggregate_sql():
    postgres = _postgres()
    captured = []

    async def fake_query(sql, params=None):
        captured.append((sql, params))
        if "COUNT(DISTINCT" in sql:
            return [
                {
                    "row_count": 3,
                    "null_count": 0,
                    "distinct_count": 2,
                    "max_value_length": 8,
                }
            ]
        return [
            {"value": "complete", "count": 2},
            {"value": "pending", "count": 1},
        ]

    postgres.query = fake_query
    postgres.connect = _noop_connect
    runtime = DbRuntime(plugins=(CatalogPlugin(auto_persist=False), postgres))

    raw_profile = await runtime.execute_capability(
        "db.column_values.profile",
        owner="postgresql",
        operation_type="source.profile",
        input={"table": "orders", "column": "status", "max_values": 25},
    )
    registered = await runtime.execute_capability(
        "catalog.source.register",
        owner="catalog",
        operation_type="schema.register",
        input={
            "schema": {
                "database_type": "postgresql",
                "database_name": "shop",
                "tables": [
                    {
                        "name": "orders",
                        "columns": [
                            {"name": "id", "data_type": "integer"},
                            {"name": "status", "data_type": "text"},
                        ],
                    }
                ],
            },
            "store_type": "postgresql",
            "store_id": "store:pg",
            "persist": False,
        },
    )
    catalog_profile = await runtime.execute_capability(
        "catalog.column_values.register",
        owner="catalog",
        operation_type="source.profile",
        input={
            "store_id": "store:pg",
            "profiles": [raw_profile[0].payload],
            "source_evidence_id": raw_profile[0].id,
        },
    )

    assert raw_profile[0].kind == "column_values.profile"
    assert raw_profile[0].owner == "postgresql"
    assert raw_profile[0].payload["table"] == "orders"
    assert raw_profile[0].payload["schema"] == "public"
    assert raw_profile[0].payload["top_values"] == [
        {"value": "complete", "count": 2},
        {"value": "pending", "count": 1},
    ]
    assert raw_profile[0].payload["source_fingerprint"]
    assert raw_profile[0].payload["policy"]["policy_owner"] == "postgresql"
    assert raw_profile[0].payload["policy"]["bounded_aggregate"] is True
    assert raw_profile[0].payload["policy"]["profile_only_readable_tables"] is True
    assert "max_profile_rows" in raw_profile[0].payload["policy"]["eligibility_checks"]
    stats_sql = captured[0][0]
    values_sql = captured[1][0]
    assert 'FROM "public"."orders"' in stats_sql
    assert 'COUNT(DISTINCT "status")::bigint AS distinct_count' in stats_sql
    assert 'GROUP BY "status"' in values_sql
    assert values_sql.endswith("LIMIT 25")
    assert registered[0].kind == "catalog.source_registered"
    assert catalog_profile[0].kind == "schema.column_value_profile"
    assert (
        catalog_profile[0].payload["profiles"][0]["source_evidence_id"]
        == raw_profile[0].id
    )


async def test_postgresql_explicit_profile_registration_skips_schema_inspection():
    postgres = _postgres()
    captured = []

    async def fake_query(sql, params=None):
        captured.append((sql, params))
        if "COUNT(DISTINCT" in sql:
            return [
                {
                    "row_count": 1,
                    "null_count": 0,
                    "distinct_count": 1,
                    "max_value_length": 8,
                }
            ]
        return [{"value": "complete", "count": 1}]

    async def fail_schema_inspection():
        raise AssertionError("explicit registration must not inspect schema")

    postgres.query = fake_query
    postgres.tables = fail_schema_inspection
    postgres.connect = _noop_connect
    catalog = CatalogPlugin(auto_persist=False)
    await catalog.register_schema(
        {
            "database_type": "postgresql",
            "database_name": "shop",
            "tables": [
                {
                    "name": "orders",
                    "columns": [{"name": "status", "data_type": "text"}],
                }
            ],
        },
        store_type="postgresql",
        store_id="store:pg-explicit",
        persist=False,
    )
    runtime = DbRuntime(plugins=(catalog, postgres))

    raw_profile = await runtime.execute_capability(
        "db.column_values.profile",
        owner="postgresql",
        operation_type="source.profile",
        input={"table": "orders", "column": "status", "max_values": 25},
    )
    registered = await runtime.execute_capability(
        "catalog.column_values.register",
        owner="catalog",
        operation_type="source.profile",
        operation_id="postgresql-explicit-profile-registration",
        input={
            "store_id": "store:pg-explicit",
            "profiles": [raw_profile[0].payload],
            "source_evidence_id": raw_profile[0].id,
        },
    )
    tasks = await runtime.store.list_tasks(
        "postgresql-explicit-profile-registration"
    )

    assert len(captured) == 2
    assert registered[0].accepted is True
    assert [task.capability_id for task in tasks] == [
        "catalog.column_values.register"
    ]


async def test_postgresql_column_value_profile_fingerprint_only_uses_catalog_stats():
    postgres = _postgres()
    captured = []

    async def fake_query(sql, params=None):
        captured.append(sql)
        assert "COUNT(DISTINCT" not in sql
        return [
            {
                "table_oid": "123",
                "relfilenode": "456",
                "relpages": 2,
                "reltuples": 3,
                "n_tup_ins": 4,
                "n_tup_upd": 5,
                "n_tup_del": 6,
            }
        ]

    postgres.query = fake_query
    postgres.connect = _noop_connect
    runtime = DbRuntime(plugins=(postgres,))

    fingerprint = await runtime.execute_capability(
        "db.column_values.profile",
        owner="postgresql",
        operation_type="source.profile",
        input={
            "table": "orders",
            "column": "status",
            "fingerprint_only": True,
        },
    )

    assert fingerprint[0].kind == "column_values.profile"
    assert fingerprint[0].payload["profile_status"] == "fingerprint"
    assert fingerprint[0].payload["profile_kind"] == "source_fingerprint"
    assert fingerprint[0].payload["source_fingerprint"]
    assert "n_tup_ins:4" in fingerprint[0].payload["source_revision"]
    assert fingerprint[0].payload["source_fingerprint_status"] == "best_effort"
    assert (
        fingerprint[0].payload["source_fingerprint_reason"]
        == "postgresql_catalog_stats"
    )
    assert "FROM pg_class" in captured[0]


async def test_postgresql_column_value_profile_fingerprint_unavailable_has_no_fake_revision():
    postgres = _postgres()

    async def fake_query(sql, params=None):
        raise RuntimeError("stats unavailable")

    postgres.query = fake_query
    postgres.connect = _noop_connect
    runtime = DbRuntime(plugins=(postgres,))

    fingerprint = await runtime.execute_capability(
        "db.column_values.profile",
        owner="postgresql",
        operation_type="source.profile",
        input={
            "table": "orders",
            "column": "status",
            "fingerprint_only": True,
        },
    )

    assert fingerprint[0].payload["profile_status"] == "fingerprint"
    assert fingerprint[0].payload["source_fingerprint_status"] == "unavailable"
    assert (
        fingerprint[0].payload["source_fingerprint_reason"]
        == "postgresql_stats_unavailable"
    )
    assert "source_revision" not in fingerprint[0].payload
    assert "source_fingerprint" not in fingerprint[0].payload


async def test_postgresql_column_value_profile_fingerprint_only_respects_blocked_table():
    postgres = PostgreSQLPlugin(
        connection_string="postgresql://localhost/testdb",
        blocked_tables=["orders"],
    )

    async def fail_query(sql, params=None):
        raise AssertionError("blocked table should not be queried")

    postgres.query = fail_query
    postgres.connect = _noop_connect
    runtime = DbRuntime(plugins=(postgres,))

    full_profile = await runtime.execute_capability(
        "db.column_values.profile",
        owner="postgresql",
        operation_type="source.profile",
        input={"table": "orders", "column": "status"},
    )
    fingerprint = await runtime.execute_capability(
        "db.column_values.profile",
        owner="postgresql",
        operation_type="source.profile",
        input={
            "table": "orders",
            "column": "status",
            "fingerprint_only": True,
        },
    )

    for evidence in (full_profile[0], fingerprint[0]):
        assert evidence.payload["profile_status"] == "skipped"
        assert evidence.payload["skipped_reason"] == "blocked_table"
        assert evidence.payload["top_values"] == []
        assert "source_revision" not in evidence.payload
        assert "source_fingerprint" not in evidence.payload
        assert "row_count" not in evidence.payload


async def test_postgresql_column_value_profile_fingerprint_only_respects_sensitive_column():
    postgres = _postgres()

    async def fail_query(sql, params=None):
        raise AssertionError("sensitive column should not be queried")

    postgres.query = fail_query
    postgres.connect = _noop_connect
    runtime = DbRuntime(plugins=(postgres,))

    full_profile = await runtime.execute_capability(
        "db.column_values.profile",
        owner="postgresql",
        operation_type="source.profile",
        input={"table": "customers", "column": "email"},
    )
    fingerprint = await runtime.execute_capability(
        "db.column_values.profile",
        owner="postgresql",
        operation_type="source.profile",
        input={
            "table": "customers",
            "column": "email",
            "fingerprint_only": True,
        },
    )

    for evidence in (full_profile[0], fingerprint[0]):
        assert evidence.payload["profile_status"] == "skipped"
        assert evidence.payload["redacted"] is True
        assert evidence.payload["skipped_reason"] == "sensitive_or_blocked_column"
        assert evidence.payload["top_values"] == []
        assert "source_revision" not in evidence.payload
        assert "source_fingerprint" not in evidence.payload
        assert "row_count" not in evidence.payload


async def test_postgresql_column_value_profile_handles_schema_qualified_table():
    postgres = _postgres()
    captured = []

    async def fake_query(sql, params=None):
        captured.append(sql)
        if "COUNT(DISTINCT" in sql:
            return [
                {
                    "row_count": 1,
                    "null_count": 0,
                    "distinct_count": 1,
                    "max_value_length": 4,
                }
            ]
        return [{"value": "open", "count": 1}]

    postgres.query = fake_query
    postgres.connect = _noop_connect
    runtime = DbRuntime(plugins=(postgres,))

    evidence = await runtime.execute_capability(
        "db.column_values.profile",
        owner="postgresql",
        operation_type="source.profile",
        input={"table": "tenant.orders", "column": "state", "max_values": 5},
    )

    assert evidence[0].payload["table"] == "tenant.orders"
    assert evidence[0].payload["schema"] == "tenant"
    assert 'FROM "tenant"."orders"' in captured[0]
    assert captured[1].endswith("LIMIT 5")


async def test_postgresql_column_value_profile_skips_sensitive_columns_without_query():
    postgres = _postgres()

    async def fake_query(sql, params=None):
        raise AssertionError("sensitive column should not be queried")

    postgres.query = fake_query
    postgres.connect = _noop_connect
    runtime = DbRuntime(plugins=(postgres,))

    evidence = await runtime.execute_capability(
        "db.column_values.profile",
        owner="postgresql",
        operation_type="source.profile",
        input={"table": "customers", "column": "email"},
    )

    assert evidence[0].payload["profile_status"] == "skipped"
    assert evidence[0].payload["redacted"] is True
    assert evidence[0].payload["skipped_reason"] == "sensitive_or_blocked_column"


async def test_postgresql_column_value_profile_honors_source_value_policy():
    samples_disabled = PostgreSQLPlugin(
        connection_string="postgresql://localhost/testdb",
        include_sample_values=False,
    )

    async def fail_query(sql, params=None):
        raise AssertionError("disabled sample values should not be queried")

    samples_disabled.query = fail_query
    samples_disabled.connect = _noop_connect
    disabled_runtime = DbRuntime(plugins=(samples_disabled,))

    disabled = await disabled_runtime.execute_capability(
        "db.column_values.profile",
        owner="postgresql",
        operation_type="source.profile",
        input={"table": "orders", "column": "status"},
    )

    assert disabled[0].payload["profile_status"] == "skipped"
    assert disabled[0].payload["skipped_reason"] == "sample_values_disabled"
    assert disabled[0].payload["policy"]["include_sample_values"] is False

    pii_allowed = PostgreSQLPlugin(
        connection_string="postgresql://localhost/testdb",
        redact_pii_columns=False,
    )

    async def fake_query(sql, params=None):
        if "COUNT(DISTINCT" in sql:
            return [
                {
                    "row_count": 1,
                    "null_count": 0,
                    "distinct_count": 1,
                    "max_value_length": 17,
                }
            ]
        return [{"value": "user@example.com", "count": 1}]

    pii_allowed.query = fake_query
    pii_allowed.connect = _noop_connect
    pii_runtime = DbRuntime(plugins=(pii_allowed,))
    profiled = await pii_runtime.execute_capability(
        "db.column_values.profile",
        owner="postgresql",
        operation_type="source.profile",
        input={"table": "customers", "column": "email"},
    )

    assert profiled[0].payload["profile_status"] == "profiled"
    assert profiled[0].payload["redacted"] is False
    assert profiled[0].payload["top_values"] == [
        {"value": "user@example.com", "count": 1}
    ]
    assert profiled[0].payload["policy"]["redact_pii_columns"] is False


async def test_postgresql_column_value_profile_skips_when_row_count_exceeds_limit():
    postgres = _postgres()
    captured = []

    async def fake_query(sql, params=None):
        captured.append(sql)
        if "COUNT(DISTINCT" not in sql:
            raise AssertionError("value aggregation should not run after row limit")
        return [
            {
                "row_count": 10,
                "null_count": 0,
                "distinct_count": 2,
                "max_value_length": 8,
            }
        ]

    postgres.query = fake_query
    postgres.connect = _noop_connect
    runtime = DbRuntime(plugins=(postgres,))

    evidence = await runtime.execute_capability(
        "db.column_values.profile",
        owner="postgresql",
        operation_type="source.profile",
        input={
            "table": "orders",
            "column": "status",
            "max_profile_rows": 2,
        },
    )

    assert len(captured) == 1
    assert evidence[0].payload["profile_status"] == "skipped"
    assert evidence[0].payload["skipped_reason"] == "row_count_exceeds_profile_limit"
    assert evidence[0].payload["row_count"] == 10
    assert evidence[0].payload["top_values"] == []


async def _noop_connect() -> None:
    return None
