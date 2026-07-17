from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from daita.db.llm_service import DbLLMConfig, DbLLMService
from daita.db.models import DbRuntimeConfig, DbSourceOptions
from daita.db.runtime import DbRuntime
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.postgresql import PostgreSQLPlugin


class ScriptedProvider:
    provider_name = "scripted"
    model = "postgresql-slim-test"
    model_name = model

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self._usage = {}

    async def generate(self, messages, tools=None, stream=False, **kwargs):
        assert stream is False
        self.calls.append(
            {
                "messages": deepcopy(messages),
                "tools": tuple(tools or ()),
            }
        )
        self._usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cached_input_tokens": 0,
            "reasoning_tokens": 0,
        }
        return self.responses.pop(0)

    def _get_last_token_usage(self):
        return dict(self._usage)

    def _estimate_cost(self, usage):
        return float(usage["total_tokens"]) / 1_000_000


def _query_call(sql, *, params=None, param_specs=None, call_id="query-1"):
    arguments = {"sql": sql, "params": list(params or [])}
    if param_specs is not None:
        arguments["param_specs"] = list(param_specs)
    return {
        "tool_calls": [
            {
                "id": call_id,
                "name": "query",
                "arguments": arguments,
            }
        ]
    }


async def _runtime_for(responses):
    postgres = PostgreSQLPlugin(
        connection_string="postgresql://localhost/testdb",
        read_only=True,
        query_default_limit=50,
        query_max_rows=100,
        query_max_chars=10_000,
    )
    executed = []

    async def noop():
        return None

    async def tables():
        return ["customers", "orders", "support_tickets"]

    async def describe(table):
        columns = {
            "customers": [
                ("id", "integer", False, True),
                ("name", "text", False, False),
                ("region", "text", False, False),
            ],
            "orders": [
                ("id", "integer", False, True),
                ("customer_id", "integer", False, False),
                ("status", "text", False, False),
                ("total", "numeric", False, False),
            ],
            "support_tickets": [
                ("id", "integer", False, True),
                ("customer_id", "integer", False, False),
                ("severity", "text", False, False),
                ("status", "text", False, False),
                ("opened_at", "date", False, False),
            ],
        }
        return [
            {
                "column_name": name,
                "data_type": data_type,
                "is_nullable": nullable,
                "column_default": None,
                "is_primary_key": primary,
            }
            for name, data_type, nullable, primary in columns[table]
        ]

    async def foreign_keys():
        return [
            {
                "source_table": "orders",
                "source_column": "customer_id",
                "target_table": "customers",
                "target_column": "id",
            },
            {
                "source_table": "support_tickets",
                "source_column": "customer_id",
                "target_table": "customers",
                "target_column": "id",
            },
        ]

    async def source_revision():
        return {
            "revision": "postgresql-schema:test-v1",
            "status": "authoritative",
            "reason": "test_fixture",
        }

    async def query(sql, params=None):
        executed.append((sql, list(params or [])))
        normalized = " ".join(sql.lower().split())
        if "sum(o.total)" in normalized:
            return [{"name": "Ada", "total": 42}]
        if "support_tickets" in normalized:
            return [{"name": "Ada"}]
        if "where region = $1" in normalized:
            return [{"name": "Ada"}]
        if "count(*)" in normalized:
            return [{"customer_count": 3}]
        return []

    postgres.connect = noop
    postgres.disconnect = noop
    postgres.tables = tables
    postgres.describe = describe
    postgres.foreign_keys = foreign_keys
    postgres._execute_source_revision = lambda _payload: source_revision()
    postgres.query = query

    catalog = CatalogPlugin(auto_persist=False)
    provider = ScriptedProvider(responses)
    service = DbLLMService(
        DbLLMConfig(provider="openai", model=provider.model),
        agent_id="postgresql-slim-test",
    )
    service._provider = provider
    config = DbRuntimeConfig(
        source_options=DbSourceOptions(
            read_only=True,
            query_default_limit=50,
            query_max_rows=100,
            query_max_chars=10_000,
        ),
        plugins=(catalog, postgres),
        metadata={
            "from_db_options": {
                "catalog_store_id": "from_db:postgresql-slim-test",
                "catalog_profile_key": "postgresql-slim-test",
                "catalog_keys": ["postgresql-slim-test"],
                "source_options": {
                    "cache_ttl": None,
                    "include_sample_values": False,
                },
            }
        },
    )
    runtime = DbRuntime(
        source=postgres,
        config=config,
        db_llm_service=service,
        owns_db_llm_service=True,
    )
    await runtime.setup(agent_id="postgresql-slim-test")
    return runtime, catalog, postgres, provider, executed


async def _register_ticket_groundings(catalog):
    state = catalog.runtime_source_state()
    await catalog.register_column_value_profiles(
        "from_db:postgresql-slim-test",
        [
            {
                "table": "support_tickets",
                "column": "severity",
                "distinct_count": 3,
                "top_values": [
                    {"value": "high", "count": 2},
                    {"value": "medium", "count": 1},
                    {"value": "low", "count": 1},
                ],
                "profile_status": "profiled",
                "sampled": False,
                "redacted": False,
                "truncated": False,
                "profiled_at": datetime.now(timezone.utc).isoformat(),
                "source_fingerprint": "support-ticket-severity-v1",
                "source_fingerprint_status": "authoritative",
                "source_revision": state["source_revision"],
                "policy": {"policy_owner": "postgresql"},
            },
            {
                "table": "support_tickets",
                "column": "status",
                "distinct_count": 2,
                "top_values": [
                    {"value": "open", "count": 2},
                    {"value": "closed", "count": 1},
                ],
                "profile_status": "profiled",
                "sampled": False,
                "redacted": False,
                "truncated": False,
                "profiled_at": datetime.now(timezone.utc).isoformat(),
                "source_fingerprint": "support-ticket-status-v1",
                "source_fingerprint_status": "authoritative",
                "source_revision": state["source_revision"],
                "policy": {"policy_owner": "postgresql"},
            },
        ],
    )


async def _operation_state(runtime, operation_id):
    return (
        tuple(await runtime.store.list_tasks(operation_id)),
        tuple(await runtime.store.list_evidence(operation_id)),
    )


async def test_postgresql_simple_read_uses_shared_two_turn_fixed_recipe():
    runtime, _catalog, _postgres, provider, executed = await _runtime_for(
        [
            _query_call("SELECT COUNT(*) AS customer_count FROM customers"),
            "There are 3 customers.",
        ]
    )
    try:
        result = await runtime.run("How many customers?")
        tasks, evidence = await _operation_state(runtime, result.operation_id)

        assert result.status.value == "succeeded"
        assert result.answer == "There are 3 customers."
        assert [task.capability_id for task in tasks] == [
            "db.sql.validate",
            "db.sql.execute_read",
        ]
        assert [task.executor_id for task in tasks] == [
            "postgresql.sql.validate",
            "postgresql.sql.execute_read",
        ]
        assert {item.owner for item in evidence} == {"postgresql"}
        assert len(evidence) == 2
        assert len(provider.calls) == 2
        operation = await runtime.store.load_operation(result.operation_id)
        assert operation.metadata["loop_state"]["implementation"] == (
            "postgresql_provider_native"
        )
        assert [view.name for view in provider.calls[0]["tools"]] == [
            "search_schema",
            "inspect_asset",
            "find_relationships",
            "search_column_values",
            "query",
        ]
        assert len(executed) == 1
    finally:
        await runtime.teardown()


async def test_postgresql_uses_shared_typed_asset_inspection_contract():
    runtime, _catalog, _postgres, provider, executed = await _runtime_for(
        [
            {
                "tool_calls": [
                    {
                        "id": "inspect-orders-fields",
                        "name": "inspect_asset",
                        "arguments": {
                            "asset_ref": "orders",
                            "fields": ["status", "total", "missing_field"],
                            "limit": 10,
                        },
                    }
                ]
            },
            "SCHEMA: The requested order fields are present in the asset schema.",
        ]
    )
    try:
        result = await runtime.run("Describe selected order fields")
        tasks, evidence = await _operation_state(runtime, result.operation_id)
        payload = evidence[0].payload

        assert result.status.value == "succeeded"
        assert len(provider.calls) == 2
        assert [task.capability_id for task in tasks] == ["catalog.asset.inspect"]
        assert tasks[0].input["fields"] == ["status", "total", "missing_field"]
        assert [field["name"] for field in payload["fields"]] == ["status", "total"]
        assert payload["matched_field_count"] == 2
        assert payload["returned_field_count"] == 2
        assert payload["missing_fields"] == ["missing_field"]
        assert payload["truncated"] is False
        assert executed == []
    finally:
        await runtime.teardown()


async def test_postgresql_filter_uses_dialect_placeholders_and_typed_params():
    runtime, _catalog, _postgres, _provider, executed = await _runtime_for(
        [
            _query_call(
                "SELECT name FROM customers WHERE region = $1 ORDER BY name",
                params=["NA"],
                param_specs=[{"ref": "region", "native_type": "string"}],
            ),
            "Ada is in NA.",
        ]
    )
    try:
        result = await runtime.run("Which customers are in NA?")
        tasks, _evidence = await _operation_state(runtime, result.operation_id)

        assert result.status.value == "succeeded"
        assert tasks[1].input["params"] == ["NA"]
        assert tasks[1].input["param_specs"] == [
            {"ref": "region", "native_type": "string"}
        ]
        assert executed[0][1] == ["NA"]
    finally:
        await runtime.teardown()


@pytest.mark.parametrize(
    ("where_sql", "params"),
    (
        ("t.severity = 'high' AND t.status = 'open'", []),
        ("t.severity = $1 AND t.status = $2", ["high", "open"]),
    ),
)
async def test_postgresql_exact_grounding_literal_or_parameter_executes(
    where_sql, params
):
    sql = (
        "SELECT c.name FROM support_tickets t "
        "JOIN customers c ON c.id = t.customer_id "
        f"WHERE {where_sql} ORDER BY c.name"
    )
    runtime, catalog, _postgres, provider, executed = await _runtime_for(
        [_query_call(sql, params=params), "Ada has an open high-severity ticket."]
    )
    await _register_ticket_groundings(catalog)
    try:
        result = await runtime.run(
            "customer names for open high-severity support tickets"
        )
        tasks, evidence = await _operation_state(runtime, result.operation_id)
        validation = next(item for item in evidence if item.kind == "sql.validation")
        coverage = validation.payload["grounding_coverage"]

        assert result.status.value == "succeeded"
        assert len(provider.calls) == 2
        assert len(executed) == 1
        assert [task.capability_id for task in tasks] == [
            "db.sql.validate",
            "db.sql.execute_read",
        ]
        assert coverage["status"] == "covered"
        assert coverage["applicable_count"] == 2
        assert {item["target"] for item in coverage["covered"]} == {
            "support_tickets.severity",
            "support_tickets.status",
        }
    finally:
        await runtime.teardown()


async def test_postgresql_missing_status_rejects_opened_at_before_io():
    sql = (
        "SELECT c.name FROM support_tickets t "
        "JOIN customers c ON c.id = t.customer_id "
        "WHERE t.severity = 'high' AND t.opened_at IS NOT NULL"
    )
    runtime, catalog, _postgres, provider, executed = await _runtime_for(
        [
            _query_call(sql),
            "CLARIFY: I could not safely represent the open-ticket constraint.",
        ]
    )
    await _register_ticket_groundings(catalog)
    try:
        result = await runtime.run(
            "customer names for open high-severity support tickets"
        )
        tasks, evidence = await _operation_state(runtime, result.operation_id)
        validation = next(item for item in evidence if item.kind == "sql.validation")
        coverage = validation.payload["grounding_coverage"]

        assert result.status.value == "blocked"
        assert len(provider.calls) == 2
        assert executed == []
        assert [task.capability_id for task in tasks] == ["db.sql.validate"]
        assert coverage["status"] == "missing"
        assert coverage["missing"][0]["target"] == "support_tickets.status"
        assert not any(item.kind == "query.result" for item in evidence)
    finally:
        await runtime.teardown()


async def test_postgresql_conflicting_grounding_rejects_before_io():
    sql = (
        "SELECT c.name FROM support_tickets t "
        "JOIN customers c ON c.id = t.customer_id "
        "WHERE t.severity = 'high' AND t.status = 'closed'"
    )
    runtime, catalog, _postgres, _provider, executed = await _runtime_for(
        [_query_call(sql), "CLARIFY: The requested status conflicts."]
    )
    await _register_ticket_groundings(catalog)
    try:
        result = await runtime.run(
            "customer names for open high-severity support tickets"
        )
        tasks, evidence = await _operation_state(runtime, result.operation_id)
        coverage = next(
            item.payload["grounding_coverage"]
            for item in evidence
            if item.kind == "sql.validation"
        )

        assert result.status.value == "blocked"
        assert executed == []
        assert [task.capability_id for task in tasks] == ["db.sql.validate"]
        assert coverage["status"] == "conflicting"
        assert coverage["conflicting"][0]["target"] == "support_tickets.status"
    finally:
        await runtime.teardown()


async def test_postgresql_zero_row_query_returns_grounded_empty_answer():
    runtime, _catalog, _postgres, provider, executed = await _runtime_for(
        [
            _query_call(
                "SELECT name FROM customers WHERE name = $1",
                params=["Nobody"],
                param_specs=[{"ref": "name", "native_type": "string"}],
            ),
            "No customers match the requested name.",
        ]
    )
    try:
        result = await runtime.run("Find the customer named Nobody")
        tasks, evidence = await _operation_state(runtime, result.operation_id)
        query_result = next(item for item in evidence if item.kind == "query.result")

        assert result.status.value == "succeeded"
        assert result.answer == "No customers match the requested name."
        assert query_result.payload["rows"] == []
        assert query_result.payload["total_rows"] == 0
        assert [task.capability_id for task in tasks] == [
            "db.sql.validate",
            "db.sql.execute_read",
        ]
        assert len(provider.calls) == 2
        assert executed == [
            ("SELECT name FROM customers WHERE name = $1 LIMIT 50", ["Nobody"])
        ]
    finally:
        await runtime.teardown()


@pytest.mark.parametrize(
    "unsafe_sql",
    (
        "DELETE FROM customers",
        "DROP TABLE customers",
        (
            "WITH deleted AS (DELETE FROM customers RETURNING *) "
            "SELECT * FROM deleted"
        ),
        "/* harmless-looking prefix */ UPDATE customers SET name = 'changed'",
        "SELECT * FROM customers; -- hidden second statement\nDROP TABLE customers",
    ),
)
async def test_public_postgresql_read_refuses_unsafe_sql_before_io(unsafe_sql):
    call = _query_call(unsafe_sql)
    runtime, _catalog, _postgres, _provider, executed = await _runtime_for([call, call])
    try:
        result = await runtime.run("Run this as a read")
        tasks, evidence = await _operation_state(runtime, result.operation_id)

        assert result.status.value == "failed"
        assert executed == []
        assert tasks
        assert all(task.capability_id == "db.sql.validate" for task in tasks)
        assert not any(item.kind == "query.result" for item in evidence)
        assert "slim_sql_validation_failed" in result.warnings
    finally:
        await runtime.teardown()


async def test_public_postgresql_query_timeout_fails_without_false_answer(
    monkeypatch,
):
    runtime, _catalog, postgres, provider, _executed = await _runtime_for(
        [_query_call("SELECT name FROM customers ORDER BY name")]
    )
    attempted = []

    async def timed_out_query(sql, params=None):
        attempted.append((sql, params))
        raise TimeoutError("query timed out")

    monkeypatch.setattr(postgres, "query", timed_out_query)
    try:
        result = await runtime.run("List customer names")
        tasks, evidence = await _operation_state(runtime, result.operation_id)

        assert result.status.value == "failed"
        assert len(provider.calls) == 1
        assert len(attempted) == 1
        assert [task.status.value for task in tasks] == ["succeeded", "failed"]
        assert not any(item.kind == "query.result" for item in evidence)
        assert "slim_query_execution_failed" in result.warnings
        assert "timed out" not in result.answer.lower()
    finally:
        await runtime.teardown()


async def test_postgresql_relationship_and_literal_grounding_match_sqlite_semantics():
    runtime, catalog, _postgres, provider, executed = await _runtime_for(
        [
            {
                "tool_calls": [
                    {
                        "id": "relationship",
                        "name": "find_relationships",
                        "arguments": {
                            "from_assets": ["orders"],
                            "to_assets": ["customers"],
                        },
                    }
                ]
            },
            {
                "tool_calls": [
                    {
                        "id": "literal",
                        "name": "search_column_values",
                        "arguments": {
                            "query": "completed",
                            "tables": ["orders"],
                            "columns": ["status"],
                        },
                    }
                ]
            },
            _query_call(
                "SELECT c.name, SUM(o.total) AS total "
                "FROM orders o JOIN customers c ON c.id = o.customer_id "
                "WHERE o.status = $1 GROUP BY c.name",
                params=["complete"],
            ),
            "Ada has 42 in completed-order totals.",
        ]
    )
    await catalog.register_column_value_profiles(
        "from_db:postgresql-slim-test",
        [
            {
                "table": "orders",
                "column": "status",
                "distinct_count": 2,
                "top_values": [
                    {"value": "complete", "count": 1},
                    {"value": "pending", "count": 1},
                ],
                "profile_status": "profiled",
                "sampled": False,
                "redacted": False,
                "truncated": False,
                "profiled_at": datetime.now(timezone.utc).isoformat(),
            }
        ],
    )
    try:
        result = await runtime.run("What are completed order totals by customer?")
        tasks, _evidence = await _operation_state(runtime, result.operation_id)
        observations = [
            message["content"]
            for message in provider.calls[2]["messages"]
            if message.get("role") == "tool"
        ]

        assert result.status.value == "succeeded"
        assert [task.capability_id for task in tasks] == [
            "catalog.relationship_paths.find",
            "catalog.column_values.search",
            "db.sql.validate",
            "db.sql.execute_read",
        ]
        assert any("complete" in item for item in observations)
        assert executed[-1][1] == ["complete"]
        assert not {
            "db.schema.inspect",
            "catalog.source.register",
            "db.planning.context.build",
            "catalog.value_grounding.plan",
            "db.column_values.profile",
        } & {task.capability_id for task in tasks}
    finally:
        await runtime.teardown()


async def test_postgresql_invalid_sql_repairs_once_before_connector_io():
    runtime, _catalog, _postgres, provider, executed = await _runtime_for(
        [
            _query_call("SELECT missing_name FROM customers", call_id="bad"),
            _query_call("SELECT name FROM customers ORDER BY name", call_id="fixed"),
            "The query completed safely.",
        ]
    )
    try:
        result = await runtime.run("List customer names")
        tasks, evidence = await _operation_state(runtime, result.operation_id)

        assert result.status.value == "succeeded"
        assert len(provider.calls) == 3
        assert len(executed) == 1
        assert "missing_name" not in executed[0][0]
        assert [task.capability_id for task in tasks] == [
            "db.sql.validate",
            "db.sql.validate",
            "db.sql.execute_read",
        ]
        assert [item.kind for item in evidence].count("query.result") == 1
    finally:
        await runtime.teardown()


async def test_postgresql_grounding_repair_succeeds_once():
    missing = (
        "SELECT c.name FROM support_tickets t "
        "JOIN customers c ON c.id = t.customer_id "
        "WHERE t.severity = 'high' AND t.opened_at IS NOT NULL"
    )
    repaired = (
        "SELECT c.name FROM support_tickets t "
        "JOIN customers c ON c.id = t.customer_id "
        "WHERE t.severity = 'high' AND t.status = 'open'"
    )
    runtime, catalog, _postgres, provider, executed = await _runtime_for(
        [
            _query_call(missing, call_id="missing-grounding"),
            _query_call(repaired, call_id="repaired-grounding"),
            "Ada has an open high-severity ticket.",
        ]
    )
    await _register_ticket_groundings(catalog)
    try:
        result = await runtime.run(
            "customer names for open high-severity support tickets"
        )
        tasks, evidence = await _operation_state(runtime, result.operation_id)

        assert result.status.value == "succeeded"
        assert len(provider.calls) == 3
        assert result.telemetry["llm_calls"] == 3
        assert len(executed) == 1
        assert [task.capability_id for task in tasks] == [
            "db.sql.validate",
            "db.sql.validate",
            "db.sql.execute_read",
        ]
        assert [
            item.payload["grounding_coverage"]["status"]
            for item in evidence
            if item.kind == "sql.validation"
        ] == ["missing", "covered"]
    finally:
        await runtime.teardown()


async def test_postgresql_repeated_missing_grounding_stops_within_budget():
    first = (
        "SELECT c.name FROM support_tickets t "
        "JOIN customers c ON c.id = t.customer_id "
        "WHERE t.severity = 'high' AND t.opened_at IS NOT NULL"
    )
    second = f"{first} ORDER BY c.name"
    runtime, catalog, _postgres, provider, executed = await _runtime_for(
        [
            _query_call(first, call_id="missing-one"),
            _query_call(second, call_id="missing-two"),
        ]
    )
    await _register_ticket_groundings(catalog)
    try:
        result = await runtime.run(
            "customer names for open high-severity support tickets"
        )
        tasks, evidence = await _operation_state(runtime, result.operation_id)

        assert result.status.value == "failed"
        assert len(provider.calls) == 2
        assert executed == []
        assert [task.capability_id for task in tasks] == [
            "db.sql.validate",
            "db.sql.validate",
        ]
        assert [
            item.payload["grounding_coverage"]["status"]
            for item in evidence
            if item.kind == "sql.validation"
        ] == ["missing", "missing"]
        assert not any(item.kind == "query.result" for item in evidence)
    finally:
        await runtime.teardown()


async def test_postgresql_read_rejects_multi_statement_before_io():
    invalid = _query_call("SELECT * FROM customers; DELETE FROM customers")
    runtime, _catalog, _postgres, _provider, executed = await _runtime_for(
        [invalid, invalid]
    )
    try:
        result = await runtime.run("Show customers")

        assert result.status.value == "failed"
        assert executed == []
        assert "slim_sql_validation_failed" in result.warnings
    finally:
        await runtime.teardown()


async def test_postgresql_supported_run_does_not_enter_legacy_phase2_paths(
    monkeypatch,
):
    from daita.db.runtime.extensions.query import DbPlanningContextExecutor
    from daita.db.runtime.tasks.catalog import DbTaskCatalog

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy Phase 2 path was entered")

    monkeypatch.setattr(DbPlanningContextExecutor, "execute", forbidden)
    monkeypatch.setattr(DbTaskCatalog, "executable_input_for_task", forbidden)
    runtime, _catalog, _postgres, provider, executed = await _runtime_for(
        [
            _query_call("SELECT COUNT(*) AS customer_count FROM customers"),
            "There are 3 customers.",
        ]
    )
    try:
        result = await runtime.run("How many customers?")

        assert result.status.value == "succeeded"
        assert len(provider.calls) == 2
        assert len(executed) == 1
        assert not {
            "db.planning.context.build",
            "db.query.plan",
            "db.query.plan.validate",
            "db.query.repair",
            "db.answer.synthesize",
        } & {capability.id for capability in runtime.registry.capabilities}
    finally:
        await runtime.teardown()
