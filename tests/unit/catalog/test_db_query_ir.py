from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from daita.agents.db.query import (
    FieldRef,
    Filter,
    Metric,
    OrderBy,
    QueryPlan,
    compile_query_plan,
)
from daita.agents.db.query.ir_validator import validate_query_plan
from daita.agents.db.query.requirements import parse_answer_requirements
from daita.agents.db.query.sql_validator import validate_sql_against_schema
from daita.agents.db.runtime.completeness import (
    attach_db_completeness,
    summarize_db_completeness,
)
from daita.agents.db.runtime.state import DbRunState, set_db_run_state
from daita.agents.db.tools.query import create_db_query_tools
from daita.agents.runtime.contextvars import active_run_state
from daita.agents.runtime.guardrails import ToolCallGuardrails
from daita.agents.runtime.state import RunState
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.catalog.base_profiler import NormalizedSchema


def _schema(tables=None, fks=None, db_type="postgresql"):
    return {
        "database_type": db_type,
        "database_name": "analytics",
        "tables": tables or [],
        "foreign_keys": fks or [],
        "table_count": len(tables or []),
    }


def _table(name, columns):
    return {"name": name, "columns": columns, "row_count": None}


def _agent_with_catalog(schema, **attrs):
    store_id = schema.get("store_id") or "test-store"
    catalog_schema = dict(schema)
    catalog_schema["store_id"] = store_id
    catalog = CatalogPlugin()
    catalog._schemas[store_id] = NormalizedSchema.from_dict(catalog_schema)
    return SimpleNamespace(
        _db_catalog=catalog,
        _db_catalog_store_id=store_id,
        **attrs,
    )


async def _attach_plugin_catalog(plugin, schema):
    catalog = CatalogPlugin()
    registered = await catalog.register_schema(
        schema, store_type=schema["database_type"]
    )
    plugin._db_catalog = catalog
    plugin._db_catalog_store_id = registered["store_id"]
    return catalog, registered["store_id"]


def _activate_run_state(db_state=None):
    run_state = RunState(agent_id="test-agent")
    if db_state is not None:
        run_state.domains["db"] = db_state
    return run_state, active_run_state.set(run_state)


def _set_answer_requirements(state, values):
    state.record_answer_requirements(parse_answer_requirements(list(values)))


def test_query_ir_round_trips_to_plain_dicts():
    plan = QueryPlan(
        grain=[FieldRef("customers", "customer_name")],
        metrics=[Metric("total_orders", "count", "orders", "order_id")],
        filters=[
            Filter(
                FieldRef("orders", "created_at"),
                ">=",
                {"macro": "start_of_current_month"},
            )
        ],
        order_by=[OrderBy("total_orders", "desc")],
        limit=1,
    )

    assert QueryPlan.from_dict(plan.to_dict()).to_dict() == plan.to_dict()


def test_identity_column_modes_use_one_shared_policy():
    from daita.agents.db.query.metadata import identity_column

    assert (
        identity_column(
            _table(
                "orders",
                [
                    {"name": "id", "type": "integer"},
                    {"name": "order_id", "type": "integer", "is_primary_key": True},
                ],
            ),
            mode="declared_only",
        )
        == "order_id"
    )
    assert (
        identity_column(
            _table("orders", [{"name": "order_id", "type": "integer"}]),
            mode="declared_or_conventional",
        )
        == "order_id"
    )
    assert (
        identity_column(
            _table("users", [{"name": "uuid", "type": "text"}]),
            mode="count_stable_row",
        )
        == "uuid"
    )
    assert (
        identity_column(
            _table("events", [{"name": "key", "type": "text"}]),
            mode="declared_or_conventional",
        )
        == "key"
    )
    assert (
        identity_column(
            _table("audit_logs", [{"name": "account_id", "type": "integer"}]),
            mode="declared_or_conventional",
        )
        == "account_id"
    )
    assert (
        identity_column(
            _table("users", [{"name": "uuid", "type": "text"}]),
            mode="declared_only",
        )
        is None
    )
    assert (
        identity_column(
            _table("metrics", [{"name": "value", "type": "numeric"}]),
            mode="count_stable_row",
        )
        is None
    )


async def test_catalog_adapter_requires_catalog_for_search_and_catalog_for_paths():
    from daita.agents.db.query.catalog_adapter import (
        find_relationship_paths,
        search_tables,
    )

    schema = _schema(
        tables=[
            _table(
                "orders",
                [
                    {"name": "order_id", "type": "integer", "is_primary_key": True},
                    {"name": "customer_id", "type": "integer"},
                ],
            ),
            _table(
                "customers",
                [{"name": "customer_id", "type": "integer", "is_primary_key": True}],
            ),
        ],
        fks=[
            {
                "source_table": "orders",
                "source_column": "customer_id",
                "target_table": "customers",
                "target_column": "customer_id",
            }
        ],
    )

    search = search_tables(schema, query="orders")
    missing_paths = find_relationship_paths(
        schema,
        from_tables=["orders"],
        to_tables=["customers"],
    )
    catalog, store_id = await _attach_plugin_catalog(SimpleNamespace(), schema)
    paths = find_relationship_paths(
        schema,
        from_tables=["orders"],
        to_tables=["customers"],
        catalog=catalog,
        store_id=store_id,
    )

    assert search["source"] == "missing_catalog"
    assert search["tables"] == []
    assert missing_paths["source"] == "missing_catalog"
    assert missing_paths["reachable"] is False
    assert paths["source"] == "catalog"
    assert paths["reachable"] is True
    assert paths["paths"][0]["joins"] == [
        {
            "left_table": "orders",
            "left_column": "customer_id",
            "right_table": "customers",
            "right_column": "customer_id",
            "predicate": "orders.customer_id = customers.customer_id",
            "relationship_direction": "forward",
        }
    ]


async def test_query_ir_compiles_catalog_join_path_for_count_question():
    schema = _schema(
        tables=[
            _table(
                "orders",
                [
                    {"name": "order_id", "type": "integer", "is_primary_key": True},
                    {"name": "customer_id", "type": "integer"},
                    {"name": "created_at", "type": "timestamp"},
                ],
            ),
            _table(
                "customers",
                [
                    {"name": "customer_id", "type": "integer", "is_primary_key": True},
                    {"name": "customer_name", "type": "text"},
                ],
            ),
        ],
        fks=[
            {
                "source_table": "orders",
                "source_column": "customer_id",
                "target_table": "customers",
                "target_column": "customer_id",
            }
        ],
    )
    plugin = MagicMock()
    plugin.sql_dialect = "postgresql"
    plugin.read_only = True
    plugin._tool_query = AsyncMock(return_value={"rows": []})
    plugin._tool_count = AsyncMock(return_value={"count": 0})
    plugin._tool_sample = AsyncMock(return_value={"rows": []})
    await _attach_plugin_catalog(plugin, schema)

    tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}
    result = await tools["db_plan_query"].handler(
        {
            "goal": "Which customer has the most orders this month?",
            "required_fields": ["customer_id", "customer_name", "total_orders"],
            "candidate_tables": ["orders", "customers"],
            "grouping": ["customers.customer_id", "customers.customer_name"],
            "ordering": ["total_orders desc"],
            "limit": 1,
            "include_diagnostics": True,
        }
    )

    assert result["query_ir"]["metrics"] == [
        {
            "name": "total_orders",
            "kind": "count",
            "table": "orders",
            "column": "order_id",
        }
    ]
    assert result["validation"]["ok"] is True
    assert result["compiled_sql"] == (
        "SELECT\n"
        '  "customers"."customer_id",\n'
        '  "customers"."customer_name",\n'
        '  COUNT("orders"."order_id") AS "total_orders"\n'
        'FROM "orders"\n'
        'JOIN "customers" ON "orders"."customer_id" = "customers"."customer_id"\n'
        'WHERE "orders"."created_at" >= date_trunc(\'month\', current_date)\n'
        'GROUP BY "customers"."customer_id", "customers"."customer_name"\n'
        'ORDER BY "total_orders" DESC\n'
        "LIMIT 1"
    )
    assert result["next_steps"] == ["run db_query with compiled_sql"]


async def test_query_ir_compiles_join_for_cross_table_sum_and_label():
    schema = _schema(
        tables=[
            _table(
                "orders",
                [
                    {"name": "order_id", "type": "integer", "is_primary_key": True},
                    {"name": "customer_id", "type": "integer"},
                    {"name": "total_amount", "type": "numeric"},
                ],
            ),
            _table(
                "customers",
                [
                    {"name": "customer_id", "type": "integer", "is_primary_key": True},
                    {"name": "name", "type": "text"},
                ],
            ),
        ],
        fks=[
            {
                "source_table": "orders",
                "source_column": "customer_id",
                "target_table": "customers",
                "target_column": "customer_id",
            }
        ],
    )
    plugin = MagicMock()
    plugin.sql_dialect = "postgresql"
    plugin.read_only = True
    plugin._tool_query = AsyncMock(return_value={"rows": []})
    plugin._tool_count = AsyncMock(return_value={"count": 0})
    plugin._tool_sample = AsyncMock(return_value={"rows": []})
    await _attach_plugin_catalog(plugin, schema)

    tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}
    result = await tools["db_plan_query"].handler(
        {
            "goal": "Which customer has the highest total order revenue?",
            "required_fields": ["customers.name", "total_revenue"],
            "candidate_tables": ["orders", "customers"],
            "required_joins": [{"from_tables": ["customers"], "to_tables": ["orders"]}],
            "aggregations": ["SUM(orders.total_amount) AS total_revenue"],
            "grouping": ["customers.name"],
            "ordering": ["total_revenue desc"],
            "limit": 1,
            "include_diagnostics": True,
        }
    )

    assert result["validation"]["ok"] is True
    assert result["query_ir"]["joins"] == [
        {
            "left": {"table": "orders", "column": "customer_id"},
            "right": {"table": "customers", "column": "customer_id"},
        }
    ]
    assert (
        'JOIN "customers" ON "orders"."customer_id" = "customers"."customer_id"'
        in result["compiled_sql"]
    )


async def test_db_plan_query_compiles_join_from_recorded_catalog_path():
    schema = _schema(
        tables=[
            _table(
                "orders",
                [
                    {"name": "order_id", "type": "integer", "is_primary_key": True},
                    {"name": "customer_id", "type": "integer"},
                    {"name": "total_amount", "type": "numeric"},
                ],
            ),
            _table(
                "customers",
                [
                    {"name": "customer_id", "type": "integer", "is_primary_key": True},
                    {"name": "name", "type": "text"},
                ],
            ),
        ],
        fks=[],
    )
    plugin = MagicMock()
    plugin.sql_dialect = "postgresql"
    plugin.read_only = True
    plugin._tool_query = AsyncMock(return_value={"rows": []})
    plugin._tool_count = AsyncMock(return_value={"count": 0})
    plugin._tool_sample = AsyncMock(return_value={"rows": []})
    state = DbRunState()
    set_db_run_state(plugin, state)
    state.record_catalog_tool_result(
        "catalog_find_join_paths",
        {"from_tables": ["orders"], "to_tables": ["customers"]},
        {
            "success": True,
            "from_assets": ["orders"],
            "to_assets": ["customers"],
            "reachable": True,
            "path_count": 1,
            "paths": [
                {
                    "tables": ["orders", "customers"],
                    "joins": [
                        {
                            "left_table": "orders",
                            "left_column": "customer_id",
                            "right_table": "customers",
                            "right_column": "customer_id",
                            "predicate": ("orders.customer_id = customers.customer_id"),
                        }
                    ],
                }
            ],
        },
    )

    tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}
    result = await tools["db_plan_query"].handler(
        {
            "goal": "Which customer has the highest total order revenue?",
            "required_fields": ["customers.name", "total_revenue"],
            "aggregations": ["SUM(orders.total_amount) AS total_revenue"],
            "grouping": ["customers.name"],
            "ordering": ["total_revenue desc"],
            "limit": 1,
            "include_diagnostics": True,
        }
    )

    assert state.summary()["catalog_join_evidence_count"] == 1
    assert result["validation"]["ok"] is True
    assert result["evidence"]["joins"][0]["provenance"] == "catalog_find_join_paths"
    assert result["intent"]["required_joins"] == [
        {
            "from_tables": ["orders"],
            "to_tables": ["customers"],
            "paths": result["evidence"]["joins"][0]["paths"],
            "source": "catalog",
            "provenance": "catalog_find_join_paths",
        }
    ]
    assert (
        'JOIN "customers" ON "orders"."customer_id" = "customers"."customer_id"'
        in result["compiled_sql"]
    )


async def test_query_ir_preserves_simple_value_filter_and_metric_phrase():
    schema = _schema(
        tables=[
            _table(
                "orders",
                [
                    {"name": "order_id", "type": "integer", "is_primary_key": True},
                    {"name": "total_amount", "type": "numeric"},
                    {"name": "status", "type": "text"},
                ],
            )
        ],
        db_type="sqlite",
    )
    plugin = MagicMock()
    plugin.sql_dialect = "sqlite"
    plugin.read_only = True
    plugin._tool_query = AsyncMock(return_value={"rows": []})
    plugin._tool_count = AsyncMock(return_value={"count": 0})
    plugin._tool_sample = AsyncMock(return_value={"rows": []})

    tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}
    result = await tools["db_plan_query"].handler(
        {
            "goal": "What is the total shipped order revenue?",
            "required_fields": ["total shipped order revenue"],
            "candidate_tables": ["orders"],
            "filters": ["status = shipped"],
            "aggregations": ["SUM(orders.total_amount) AS total_revenue"],
        }
    )

    assert result["validation"]["ok"] is True
    assert 'SUM("orders"."total_amount") AS "total_revenue"' in result["compiled_sql"]
    assert 'WHERE "orders"."status" = \'shipped\'' in result["compiled_sql"]
    assert result["suggested_next_tool"] == "db_query"


def test_query_ir_validation_rejects_unjoined_cross_table_references():
    schema = _schema(
        tables=[
            _table(
                "orders",
                [
                    {"name": "total_amount", "type": "numeric"},
                ],
            ),
            _table("customers", [{"name": "name", "type": "text"}]),
        ]
    )
    plan = QueryPlan(
        grain=[FieldRef("customers", "name")],
        metrics=[
            Metric(
                name="total_revenue",
                kind="sum",
                table="orders",
                column="total_amount",
            )
        ],
    )

    result = validate_query_plan(plan, schema, dialect="postgresql")

    assert result["ok"] is False
    assert result["errors"][-1]["type"] == "unjoined_table_reference"
    assert set(result["errors"][-1]["missing_tables"]) == {"orders", "customers"}


async def test_db_query_executes_validated_plan_id_without_sql_preflight():
    schema = _schema(
        tables=[
            _table(
                "orders",
                [
                    {"name": "order_id", "type": "integer", "is_primary_key": True},
                    {"name": "created_at", "type": "timestamp"},
                ],
            )
        ]
    )
    plugin = MagicMock()
    plugin.sql_dialect = "postgresql"
    plugin.read_only = True
    plugin._tool_query = AsyncMock(return_value={"rows": [{"total_orders": 3}]})
    plugin._tool_count = AsyncMock(return_value={"count": 0})
    plugin._tool_sample = AsyncMock(return_value={"rows": []})
    db_state = DbRunState()
    set_db_run_state(plugin, db_state)

    tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}
    plan_result = await tools["db_plan_query"].handler(
        {
            "goal": "How many orders were created this month?",
            "required_fields": ["total_orders"],
            "candidate_tables": ["orders"],
            "aggregations": ["COUNT(orders.order_id) AS total_orders"],
        }
    )
    result = await tools["db_query"].handler({"plan_id": plan_result["plan_id"]})

    assert plan_result["next_steps"] == ["run db_query with plan_id=plan_1"]
    assert plan_result["suggested_next_tool"] == "db_query"
    assert plan_result["suggested_next_arguments"] == {"plan_id": "plan_1"}
    assert "query_ir" not in plan_result
    assert "intent" not in plan_result
    stored_plan = db_state.get_plan("plan_1")
    assert stored_plan["intent"]["goal"] == "How many orders were created this month?"
    assert stored_plan["query_ir"]["metrics"][0]["name"] == "total_orders"
    assert stored_plan["compiled_sql"] == plan_result["compiled_sql"]
    assert "result" not in stored_plan
    assert result == {"rows": [{"total_orders": 3}]}
    plugin._tool_query.assert_awaited_once()
    assert plugin._tool_query.await_args.args[0]["sql"] == plan_result["compiled_sql"]


async def test_db_query_revalidates_plan_id_before_execution():
    schema = _schema(
        tables=[
            _table(
                "orders",
                [
                    {"name": "order_id", "type": "integer"},
                ],
            )
        ]
    )
    plugin = MagicMock()
    plugin.sql_dialect = "postgresql"
    plugin.read_only = True
    plugin._tool_query = AsyncMock(return_value={"rows": []})
    plugin._tool_count = AsyncMock(return_value={"count": 0})
    plugin._tool_sample = AsyncMock(return_value={"rows": []})
    state = DbRunState()
    set_db_run_state(plugin, state)
    state.plans_by_id["plan_1"] = {
        "plan_id": "plan_1",
        "compiled_sql": "SELECT missing_column FROM orders",
        "validation": {"ok": True},
    }

    tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}
    result = await tools["db_query"].handler({"plan_id": "plan_1"})

    assert result["error_type"] == "schema_reference_error"
    assert result["repair_required"] is True
    assert result["suggested_next_tool"] == "db_plan_query"
    assert result["plan_id"] == "plan_1"
    plugin._tool_query.assert_not_awaited()


async def test_db_plan_preserves_structured_aggregate_alias_with_unrelated_revenue_column():
    schema = _schema(
        tables=[
            _table(
                "public.customers",
                [
                    {"name": "customer_id", "type": "integer", "is_primary_key": True},
                    {"name": "name", "type": "text"},
                ],
            ),
            _table(
                "public.orders",
                [
                    {"name": "order_id", "type": "integer", "is_primary_key": True},
                    {"name": "customer_id", "type": "integer"},
                    {"name": "total_amount", "type": "numeric"},
                ],
            ),
            _table("public.daily_metrics", [{"name": "revenue", "type": "numeric"}]),
        ],
        fks=[
            {
                "source_table": "public.orders",
                "source_column": "customer_id",
                "target_table": "public.customers",
                "target_column": "customer_id",
            }
        ],
    )
    plugin = MagicMock()
    plugin.sql_dialect = "postgresql"
    plugin.read_only = True
    plugin._tool_query = AsyncMock(
        return_value={"rows": [{"customer_id": 2, "name": "Bob", "total_revenue": 150}]}
    )
    plugin._tool_count = AsyncMock(return_value={"count": 0})
    plugin._tool_sample = AsyncMock(return_value={"rows": []})
    await _attach_plugin_catalog(plugin, schema)
    set_db_run_state(plugin, DbRunState())

    tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}
    plan_result = await tools["db_plan_query"].handler(
        {
            "goal": "Which customer has the highest total revenue?",
            "required_fields": [
                "customers.customer_id",
                "customers.name",
                "SUM(orders.total_amount) AS total_revenue",
            ],
            "candidate_tables": ["orders", "customers"],
            "aggregations": ["SUM(orders.total_amount)"],
            "grouping": ["customers.customer_id", "customers.name"],
            "ordering": ["total_revenue DESC"],
            "limit": 1,
        }
    )

    assert plan_result["validation"]["ok"] is True
    assert 'AS "total_revenue"' in plan_result["compiled_sql"]
    assert "sum_total_amount" not in plan_result["compiled_sql"]

    result = await tools["db_query"].handler({"plan_id": plan_result["plan_id"]})

    assert result["rows"] == [{"customer_id": 2, "name": "Bob", "total_revenue": 150}]
    plugin._tool_query.assert_awaited_once()


async def test_db_plan_treats_required_metric_source_as_aggregate_requirement():
    schema = _schema(
        tables=[
            _table(
                "orders",
                [
                    {"name": "order_id", "type": "integer", "is_primary_key": True},
                    {"name": "total_amount", "type": "numeric"},
                ],
            )
        ]
    )
    plugin = MagicMock()
    plugin.sql_dialect = "postgresql"
    plugin.read_only = True
    plugin._tool_query = AsyncMock(return_value={"rows": [{"total_revenue": 150}]})
    plugin._tool_count = AsyncMock(return_value={"count": 0})
    plugin._tool_sample = AsyncMock(return_value={"rows": []})
    set_db_run_state(plugin, DbRunState())

    tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}
    plan_result = await tools["db_plan_query"].handler(
        {
            "goal": "Find total order revenue.",
            "required_fields": ["orders.total_amount"],
            "candidate_tables": ["orders"],
            "aggregations": ["SUM(orders.total_amount) AS total_revenue"],
        }
    )
    result = await tools["db_query"].handler({"plan_id": plan_result["plan_id"]})

    assert plan_result["validation"]["ok"] is True
    assert result["rows"] == [{"total_revenue": 150}]
    plugin._tool_query.assert_awaited_once()


def test_db_guardrail_uses_sql_fingerprint_across_different_plan_ids():
    run_state = RunState(agent_id="test-agent")
    run_state.domains["db"] = DbRunState()
    guardrails = ToolCallGuardrails()
    raw = {
        "repair_required": True,
        "preflight_failed": True,
        "sql_fingerprint": "same-sql",
        "error_type": "required_field_error",
        "message": "same SQL failed",
    }

    first = guardrails.observe_tool_result(
        run_state,
        {"name": "db_query", "arguments": {"plan_id": "plan_1"}},
        {"result": raw},
    )
    second = guardrails.observe_tool_result(
        run_state,
        {"name": "db_query", "arguments": {"plan_id": "plan_2"}},
        {"result": raw},
    )

    assert first.guidance_result is None
    assert second.guidance_result["guardrail"] == "repeated_tool_error"


async def test_db_query_falls_back_to_sql_when_unknown_plan_id_is_paired_with_sql():
    schema = _schema(
        tables=[
            _table(
                "orders",
                [
                    {"name": "order_id", "type": "integer"},
                ],
            )
        ]
    )
    plugin = MagicMock()
    plugin.sql_dialect = "postgresql"
    plugin.read_only = True
    plugin._tool_query = AsyncMock(return_value={"rows": [{"order_id": 1}]})
    plugin._tool_count = AsyncMock(return_value={"count": 0})
    plugin._tool_sample = AsyncMock(return_value={"rows": []})
    set_db_run_state(plugin, DbRunState())

    tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}
    result = await tools["db_query"].handler(
        {"plan_id": "missing_plan", "sql": "SELECT order_id FROM orders"}
    )

    assert result == {"rows": [{"order_id": 1}]}
    plugin._tool_query.assert_awaited_once()
    assert plugin._tool_query.await_args.args[0] == {
        "sql": "SELECT order_id FROM orders"
    }


async def test_db_query_falls_back_to_sql_when_plan_id_is_not_executable():
    schema = _schema(
        tables=[
            _table(
                "orders",
                [
                    {"name": "order_id", "type": "integer"},
                ],
            )
        ]
    )
    plugin = MagicMock()
    plugin.sql_dialect = "postgresql"
    plugin.read_only = True
    plugin._tool_query = AsyncMock(return_value={"rows": []})
    plugin._tool_count = AsyncMock(return_value={"count": 0})
    plugin._tool_sample = AsyncMock(return_value={"rows": []})
    state = DbRunState()
    set_db_run_state(plugin, state)
    state.plans_by_id["plan_1"] = {
        "plan_id": "plan_1",
        "compiled_sql": "",
        "validation": {"ok": False},
    }

    tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}
    result = await tools["db_query"].handler(
        {"plan_id": "plan_1", "sql": "SELECT order_id FROM orders WHERE order_id > 10"}
    )

    assert result == {"rows": []}
    plugin._tool_query.assert_awaited_once()
    assert plugin._tool_query.await_args.args[0] == {
        "sql": "SELECT order_id FROM orders WHERE order_id > 10"
    }


async def test_db_plan_query_returns_compact_output_by_default_and_debug_details_on_request():
    schema = _schema(
        tables=[
            _table(
                "orders",
                [
                    {"name": "order_id", "type": "integer", "is_primary_key": True},
                    {"name": "customer_id", "type": "integer"},
                ],
            )
        ]
    )
    plugin = MagicMock()
    plugin.sql_dialect = "postgresql"
    plugin.read_only = True
    plugin._tool_query = AsyncMock(return_value={"rows": []})
    plugin._tool_count = AsyncMock(return_value={"count": 0})
    plugin._tool_sample = AsyncMock(return_value={"rows": []})
    await _attach_plugin_catalog(plugin, schema)

    tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}
    compact = await tools["db_plan_query"].handler(
        {
            "goal": "How many orders?",
            "required_fields": ["total_orders"],
            "candidate_tables": ["orders"],
            "aggregations": ["COUNT(orders.order_id) AS total_orders"],
        }
    )
    diagnostic = await tools["db_plan_query"].handler(
        {
            "goal": "How many orders?",
            "required_fields": ["total_orders"],
            "candidate_tables": ["orders"],
            "aggregations": ["COUNT(orders.order_id) AS total_orders"],
            "include_diagnostics": True,
        }
    )

    assert compact["ok"] is True
    assert compact["next_step"] == "run db_query with compiled_sql"
    assert "query_ir" not in compact
    assert "intent" not in compact
    assert "table_candidates" not in compact
    assert "field_candidates" not in compact
    assert diagnostic["intent"]["goal"] == "How many orders?"
    assert diagnostic["query_ir"]["metrics"][0]["name"] == "total_orders"
    assert diagnostic["table_candidates"]
    assert diagnostic["field_candidates"]


async def test_db_plan_query_records_shared_evidence():
    plugin = SimpleNamespace(
        sql_dialect="postgresql",
        read_only=True,
        _tool_query=AsyncMock(return_value={"rows": []}),
        _tool_count=AsyncMock(return_value={"count": 0}),
        _tool_sample=AsyncMock(return_value={"rows": []}),
    )
    state = DbRunState()
    set_db_run_state(plugin, state)
    run_state, token = _activate_run_state(state)
    schema = _schema(
        tables=[
            _table(
                "orders",
                [
                    {"name": "order_id", "type": "integer"},
                    {"name": "total_amount", "type": "numeric"},
                ],
            )
        ]
    )
    tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}

    try:
        result = await tools["db_plan_query"].handler(
            {
                "goal": "Revenue by order",
                "required_fields": ["total amount"],
                "candidate_tables": ["orders"],
            }
        )
    finally:
        active_run_state.reset(token)

    assert result["plan_id"] == "plan_1"
    assert state.summary()["planned_query_count"] == 1
    assert [record.kind for record in run_state.evidence] == ["query_plan"]
    evidence = run_state.evidence[0]
    assert evidence.domain == "db"
    assert evidence.source_tool == "db_plan_query"
    assert evidence.payload["plan_id"] == "plan_1"
    assert evidence.payload["intent"]["goal"] == "Revenue by order"
    assert "query_ir" in evidence.payload
    assert evidence.payload["resolved_tables"] == ["orders"]


async def test_db_plan_query_uses_catalog_evidence_for_candidate_tables():
    schema = _schema(
        tables=[
            _table(
                "orders",
                [
                    {"name": "order_id", "type": "integer", "is_primary_key": True},
                    {"name": "amount", "type": "numeric"},
                ],
            ),
            _table(
                "events",
                [
                    {"name": "event_id", "type": "integer", "is_primary_key": True},
                    {"name": "payload", "type": "text"},
                ],
            ),
        ]
    )
    plugin = MagicMock()
    plugin.sql_dialect = "postgresql"
    plugin.read_only = True
    plugin._tool_query = AsyncMock(return_value={"rows": []})
    plugin._tool_count = AsyncMock(return_value={"count": 0})
    plugin._tool_sample = AsyncMock(return_value={"rows": []})
    store_id = "analytics-store"
    catalog_schema = dict(schema)
    catalog_schema["store_id"] = store_id
    catalog = CatalogPlugin()
    catalog._schemas[store_id] = NormalizedSchema.from_dict(catalog_schema)
    plugin._db_catalog = catalog
    plugin._db_catalog_store_id = store_id

    tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}
    result = await tools["db_plan_query"].handler(
        {
            "goal": "What is total revenue?",
            "required_fields": ["total_revenue"],
            "aggregations": ["SUM(orders.amount) AS total_revenue"],
            "include_diagnostics": True,
        }
    )

    assert result["resolved_tables"] == ["orders"]
    assert result["knowledge_used"]["sources"] == ["catalog"]
    assert result["knowledge_used"]["tables"][0] == "orders"
    assert result["evidence"]["tables"][0]["source"] == "catalog"


async def test_db_plan_query_uses_attached_catalog_for_candidate_tables():
    schema = _schema(
        tables=[
            _table(
                "orders",
                [
                    {"name": "order_id", "type": "integer", "is_primary_key": True},
                    {"name": "amount", "type": "numeric"},
                ],
            ),
            _table(
                "events",
                [
                    {"name": "event_id", "type": "integer", "is_primary_key": True},
                    {"name": "payload", "type": "text"},
                ],
            ),
        ]
    )
    catalog = CatalogPlugin()
    registered = await catalog.register_schema(schema, store_type="postgresql")

    plugin = MagicMock()
    plugin.sql_dialect = "postgresql"
    plugin.read_only = True
    plugin._tool_query = AsyncMock(return_value={"rows": []})
    plugin._tool_count = AsyncMock(return_value={"count": 0})
    plugin._tool_sample = AsyncMock(return_value={"rows": []})
    plugin._db_catalog = catalog
    plugin._db_catalog_store_id = registered["store_id"]

    tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}
    result = await tools["db_plan_query"].handler(
        {
            "goal": "What is total revenue?",
            "required_fields": ["total_revenue"],
            "aggregations": ["SUM(orders.amount) AS total_revenue"],
            "include_diagnostics": True,
        }
    )

    assert result["resolved_tables"] == ["orders"]
    assert result["knowledge_used"]["sources"] == ["catalog"]
    assert result["evidence"]["tables"][0]["source"] == "catalog"


async def test_db_compile_and_query_plans_validates_and_executes_supported_intent():
    schema = _schema(
        tables=[
            _table(
                "orders",
                [
                    {"name": "order_id", "type": "integer", "is_primary_key": True},
                    {"name": "created_at", "type": "timestamp"},
                ],
            )
        ]
    )
    plugin = MagicMock()
    plugin.sql_dialect = "postgresql"
    plugin.read_only = True
    plugin._tool_query = AsyncMock(return_value={"rows": [{"total_orders": 7}]})
    plugin._tool_count = AsyncMock(return_value={"count": 0})
    plugin._tool_sample = AsyncMock(return_value={"rows": []})
    set_db_run_state(plugin, DbRunState())

    tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}
    result = await tools["db_compile_and_query"].handler(
        {
            "goal": "How many orders were created this month?",
            "required_fields": ["total_orders"],
            "candidate_tables": ["orders"],
            "aggregations": ["COUNT(orders.order_id) AS total_orders"],
        }
    )

    assert result["ok"] is True
    assert result["rows"] == [{"total_orders": 7}]
    assert result["route"] == "aggregation"
    assert result["resolved_tables"] == ["orders"]
    assert result["validation"]["ok"] is True
    assert "COUNT" in result["sql"]
    plugin._tool_query.assert_awaited_once()


async def test_db_query_records_validated_and_executed_query_evidence():
    plugin = SimpleNamespace(
        sql_dialect="postgresql",
        read_only=True,
        _tool_query=AsyncMock(
            return_value={
                "rows": [{"revenue": 125}],
                "total_rows": 1,
                "truncated": False,
            }
        ),
        _tool_count=AsyncMock(return_value={"count": 0}),
        _tool_sample=AsyncMock(return_value={"rows": []}),
    )
    state = DbRunState()
    set_db_run_state(plugin, state)
    run_state, token = _activate_run_state(state)
    schema = _schema(
        tables=[
            _table(
                "orders",
                [
                    {"name": "customer_id", "type": "integer"},
                    {"name": "total_amount", "type": "numeric"},
                ],
            )
        ]
    )
    tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}

    try:
        result = await tools["db_query"].handler(
            {
                "sql": (
                    "SELECT SUM(total_amount) AS revenue "
                    "FROM orders WHERE customer_id = 1"
                )
            }
        )
    finally:
        active_run_state.reset(token)

    assert result["rows"] == [{"revenue": 125}]
    assert state.summary()["validated_sql_count"] == 1
    assert state.summary()["executed_query_count"] == 1
    assert [record.kind for record in run_state.evidence] == [
        "validated_sql",
        "executed_query",
    ]
    validated = run_state.evidence[0]
    assert validated.payload["referenced_tables"] == ["orders"]
    assert validated.payload["referenced_columns"] == ["customer_id", "total_amount"]
    assert validated.payload["selected_columns"] == ["revenue"]
    executed = run_state.evidence[1]
    assert executed.domain == "db"
    assert executed.source_tool == "db_query"
    assert executed.payload["columns"] == ["revenue"]
    assert executed.payload["row_count"] == 1
    assert executed.payload["truncated"] is False

    completeness = attach_db_completeness(run_state)
    assert completeness["status"] == "answerable"
    assert completeness["can_answer"] is True
    assert completeness["queries_executed"] == 1
    assert completeness["total_rows_observed"] == 1
    assert run_state.diagnostics["db_completeness"] == completeness
    assert state.final_completeness_status == completeness


async def test_db_query_records_rejected_sql_evidence():
    plugin = SimpleNamespace(
        sql_dialect="postgresql",
        read_only=True,
        _tool_query=AsyncMock(return_value={"rows": []}),
        _tool_count=AsyncMock(return_value={"count": 0}),
        _tool_sample=AsyncMock(return_value={"rows": []}),
    )
    state = DbRunState()
    set_db_run_state(plugin, state)
    run_state, token = _activate_run_state(state)
    schema = _schema(
        tables=[
            _table(
                "customers",
                [
                    {"name": "customer_id", "type": "integer"},
                    {"name": "customer_name", "type": "text"},
                ],
            )
        ]
    )
    tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}

    try:
        result = await tools["db_query"].handler(
            {"sql": "SELECT c.email FROM customers c"}
        )
    finally:
        active_run_state.reset(token)

    assert result["repair_required"] is True
    assert state.summary()["failed_sql_count"] == 1
    assert [record.kind for record in run_state.evidence] == ["rejected_sql"]
    rejected = run_state.evidence[0]
    assert rejected.domain == "db"
    assert rejected.source_tool == "db_query"
    assert rejected.payload["error_type"] == "schema_reference_error"
    assert rejected.payload["missing_columns"] == [
        {"table": "customers", "column": "email", "reason": "column not found"}
    ]
    assert rejected.payload["attempt_count"] == 1

    completeness = summarize_db_completeness(run_state)
    assert completeness["status"] == "blocked"
    assert completeness["can_answer"] is False
    assert completeness["sql_rejected"] == 1
    assert completeness["unresolved_repair"] is True
    assert completeness["latest_db_evidence_kind"] == "rejected_sql"
    assert completeness["warnings"] == [
        "sql_rejections_present",
        "unresolved_sql_repair",
    ]


def test_db_completeness_reports_insufficient_evidence_for_empty_db_run_state():
    state = DbRunState()
    _set_answer_requirements(state, ["total revenue"])
    run_state = RunState(agent_id="test-agent")
    run_state.domains["db"] = state

    completeness = attach_db_completeness(run_state)

    assert completeness["status"] == "insufficient_evidence"
    assert completeness["can_answer"] is False
    assert completeness["missing_required_fields"] == ["total revenue"]
    assert completeness["warnings"] == [
        "required_fields_tracked",
        "missing_required_fields",
    ]
    assert state.final_completeness_status == completeness


def test_schema_only_completeness_accepts_catalog_evidence():
    state = DbRunState(intent_kind="schema_only")
    run_state = RunState(agent_id="test-agent")
    run_state.domains["db"] = state

    state.record_catalog_tool_result(
        "catalog_search_schema",
        {"query": "revenue"},
        {
            "tables": [
                {
                    "name": "payments",
                    "score": 0.95,
                    "matched_fields": [
                        {"name": "amount", "score": 0.9},
                        {"name": "currency", "score": 0.7},
                    ],
                }
            ]
        },
    )

    completeness = attach_db_completeness(run_state)

    assert completeness["status"] == "answerable_schema"
    assert completeness["can_answer"] is True
    assert completeness["queries_executed"] == 0
    assert completeness["catalog_table_evidence_count"] == 1
    assert completeness["catalog_column_evidence_count"] == 2


def test_db_completeness_requires_planned_fields_in_returned_columns():
    state = DbRunState()
    _set_answer_requirements(state, ["customer_name"])
    run_state = RunState(agent_id="test-agent")
    run_state.domains["db"] = state
    token = active_run_state.set(run_state)
    try:
        state.record_executed_query(
            {
                "sql": "SELECT customer_id FROM customers",
                "columns": ["customer_id"],
                "row_count": 1,
                "truncated": False,
            }
        )
    finally:
        active_run_state.reset(token)

    completeness = attach_db_completeness(run_state)

    assert completeness["status"] == "missing_required_fields"
    assert completeness["can_answer"] is False
    assert completeness["returned_columns"] == ["customer_id"]
    assert completeness["missing_required_fields"] == ["customer_name"]


def test_db_completeness_uses_selected_columns_for_empty_results():
    state = DbRunState()
    _set_answer_requirements(state, ["customer_id", "signup_date"])
    run_state = RunState(agent_id="test-agent")
    run_state.domains["db"] = state
    token = active_run_state.set(run_state)
    try:
        state.record_executed_query(
            {
                "sql": (
                    "SELECT customer_id, signup_date FROM customers "
                    "WHERE signup_date > '2030-01-01'"
                ),
                "columns": [],
                "selected_columns": ["customer_id", "signup_date"],
                "row_count": 0,
                "truncated": False,
            }
        )
    finally:
        active_run_state.reset(token)

    completeness = attach_db_completeness(run_state)

    assert completeness["status"] == "answerable_empty"
    assert completeness["can_answer"] is True
    assert completeness["returned_columns"] == []
    assert completeness["evidence_columns"] == ["customer_id", "signup_date"]
    assert completeness["missing_required_fields"] == []


def test_db_completeness_matches_qualified_required_fields_to_output_names():
    state = DbRunState()
    _set_answer_requirements(
        state,
        [
            "customers.customer_id",
            "customers.name",
            "SUM(orders.total_amount) AS total_revenue",
        ],
    )
    run_state = RunState(agent_id="test-agent")
    run_state.domains["db"] = state
    token = active_run_state.set(run_state)
    try:
        state.record_executed_query(
            {
                "sql": (
                    "SELECT customers.customer_id, customers.name, "
                    "SUM(orders.total_amount) AS total_revenue FROM orders"
                ),
                "columns": ["customer_id", "name", "total_revenue"],
                "row_count": 1,
                "truncated": False,
            }
        )
    finally:
        active_run_state.reset(token)

    completeness = attach_db_completeness(run_state)

    assert completeness["status"] == "answerable"
    assert completeness["can_answer"] is True
    assert completeness["missing_required_fields"] == []


def test_db_completeness_matches_semantic_returned_column_aliases():
    state = DbRunState()
    _set_answer_requirements(state, ["customer records", "order revenue"])
    run_state = RunState(agent_id="test-agent")
    run_state.domains["db"] = state
    token = active_run_state.set(run_state)
    try:
        state.record_executed_query(
            {
                "sql": "SELECT COUNT(*) AS total_customers, SUM(total_amount) AS total_revenue FROM orders",
                "columns": ["total_customers", "total_revenue"],
                "row_count": 1,
                "truncated": False,
            }
        )
    finally:
        active_run_state.reset(token)

    completeness = attach_db_completeness(run_state)

    assert completeness["status"] == "answerable"
    assert completeness["can_answer"] is True
    assert completeness["missing_required_fields"] == []


def test_db_completeness_blocks_when_latest_sql_repair_is_unresolved():
    state = DbRunState()
    run_state = RunState(agent_id="test-agent")
    run_state.domains["db"] = state
    token = active_run_state.set(run_state)
    try:
        state.record_executed_query(
            {
                "sql": "SELECT customer_name FROM customers",
                "columns": ["customer_name"],
                "row_count": 1,
                "truncated": False,
            }
        )
        state.record_failed_sql(
            "bad-sql",
            {"sql_fingerprint": "bad-sql", "repair_required": True},
        )
    finally:
        active_run_state.reset(token)

    blocked = attach_db_completeness(run_state)

    assert blocked["status"] == "blocked"
    assert blocked["can_answer"] is False
    assert blocked["unresolved_repair"] is True

    token = active_run_state.set(run_state)
    try:
        state.record_executed_query(
            {
                "sql": "SELECT customer_name FROM customers",
                "columns": ["customer_name"],
                "row_count": 1,
                "truncated": False,
            }
        )
    finally:
        active_run_state.reset(token)

    unblocked = attach_db_completeness(run_state)

    assert unblocked["status"] == "answerable"
    assert unblocked["can_answer"] is True
    assert unblocked["unresolved_repair"] is False


async def test_db_compile_and_query_returns_repair_payload_when_intent_cannot_compile():
    schema = _schema(
        tables=[_table("orders", [{"name": "order_id", "type": "integer"}])]
    )
    plugin = MagicMock()
    plugin.sql_dialect = "postgresql"
    plugin.read_only = True
    plugin._tool_query = AsyncMock(return_value={"rows": []})
    plugin._tool_count = AsyncMock(return_value={"count": 0})
    plugin._tool_sample = AsyncMock(return_value={"rows": []})

    tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}
    result = await tools["db_compile_and_query"].handler(
        {
            "goal": "Show customers by region",
            "required_fields": ["customer_name"],
            "candidate_tables": ["customers"],
        }
    )

    assert result["ok"] is False
    assert result["repair_required"] is True
    assert result["suggested_next_tool"] == "db_plan_query"
    plugin._tool_query.assert_not_awaited()


async def test_db_fast_path_executes_simple_count_when_metadata_is_confident():
    from daita.agents.db.runtime.fast_path import try_db_fast_path

    schema = _schema(
        tables=[
            _table(
                "orders",
                [
                    {"name": "order_id", "type": "integer", "is_primary_key": True},
                    {"name": "created_at", "type": "timestamp"},
                ],
            )
        ]
    )
    plugin = MagicMock()
    plugin.sql_dialect = "postgresql"
    plugin.read_only = True
    plugin._tool_query = AsyncMock(return_value={"rows": [{"total_orders": 4}]})
    plugin._tool_count = AsyncMock(return_value={"count": 0})
    plugin._tool_sample = AsyncMock(return_value={"rows": []})
    tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}
    agent = _agent_with_catalog(
        schema,
        _tool_call_history=[],
        tool_registry=SimpleNamespace(get=tools.get),
    )

    result = await try_db_fast_path(agent, "How many orders this month?", {})

    assert result["from_db_fast_path"]["used"] is True
    assert result["iterations"] == 0
    assert result["result"] == "total_orders: 4"
    assert result["tool_calls"][0]["tool"] == "db_compile_and_query"
    plugin._tool_query.assert_awaited_once()


async def test_db_fast_path_uses_catalog_metadata_evidence():
    from daita.agents.db.runtime.fast_path import try_db_fast_path

    schema = _schema(
        tables=[
            {
                "name": "users",
                "columns": [
                    {"name": "user_id", "type": "integer", "is_primary_key": True},
                    {"name": "created_at", "type": "timestamp"},
                ],
                "row_count": None,
                "metadata": {"business_name": "signups"},
            }
        ]
    )
    plugin = MagicMock()
    plugin.sql_dialect = "postgresql"
    plugin.read_only = True
    plugin._tool_query = AsyncMock(return_value={"rows": [{"total_users": 9}]})
    plugin._tool_count = AsyncMock(return_value={"count": 0})
    plugin._tool_sample = AsyncMock(return_value={"rows": []})
    tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}
    agent = _agent_with_catalog(
        schema,
        _tool_call_history=[],
        tool_registry=SimpleNamespace(get=tools.get),
    )

    result = await try_db_fast_path(agent, "How many signups?", {})

    assert result["from_db_fast_path"]["used"] is True
    assert result["result"] == "total_users: 9"


async def test_fast_path_and_planner_use_shared_conventional_identity():
    from daita.agents.db.runtime.fast_path import try_db_fast_path

    schema = _schema(
        tables=[
            _table(
                "users",
                [
                    {"name": "uuid", "type": "text"},
                    {"name": "created_at", "type": "timestamp"},
                ],
            )
        ]
    )
    plugin = MagicMock()
    plugin.sql_dialect = "postgresql"
    plugin.read_only = True
    plugin._tool_query = AsyncMock(return_value={"rows": [{"total_users": 5}]})
    plugin._tool_count = AsyncMock(return_value={"count": 0})
    plugin._tool_sample = AsyncMock(return_value={"rows": []})
    await _attach_plugin_catalog(plugin, schema)
    tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}

    plan = await tools["db_plan_query"].handler(
        {
            "goal": "How many users?",
            "required_fields": ["total_users"],
            "candidate_tables": ["users"],
            "include_diagnostics": True,
        }
    )
    agent = _agent_with_catalog(
        schema,
        _tool_call_history=[],
        tool_registry=SimpleNamespace(get=tools.get),
    )
    fast_path = await try_db_fast_path(agent, "How many users?", {})

    assert plan["query_ir"]["metrics"] == [
        {"name": "total_users", "kind": "count", "table": "users", "column": "uuid"}
    ]
    assert fast_path["from_db_fast_path"]["used"] is True
    assert (
        'COUNT("users"."uuid") AS "total_users"'
        in fast_path["tool_calls"][0]["result"]["sql"]
    )
    assert plugin._tool_query.await_args.args[0]["sql"].startswith("SELECT")


async def test_db_fast_path_skips_ambiguous_prompt_metadata():
    from daita.agents.db.runtime.fast_path import try_db_fast_path

    schema = _schema(tables=[_table("orders", []), _table("customers", [])])
    agent = _agent_with_catalog(
        schema,
        tool_registry=SimpleNamespace(get=lambda name: None),
    )

    assert await try_db_fast_path(agent, "How many records?", {}) is None


def test_query_ir_validation_rejects_non_numeric_sum_metric():
    schema = _schema(
        tables=[
            _table(
                "orders",
                [
                    {"name": "order_id", "type": "integer", "is_primary_key": True},
                    {"name": "status", "type": "text"},
                ],
            )
        ]
    )
    plan = QueryPlan(metrics=[Metric("total_status", "sum", "orders", "status")])

    result = validate_query_plan(plan, schema, dialect="postgresql")

    assert result["ok"] is False
    assert result["errors"][0]["type"] == "non_numeric_metric_column"


def test_query_ir_compiler_supports_sqlite_date_macros():
    plan = QueryPlan(
        metrics=[Metric("total_orders", "count", "orders", "order_id")],
        filters=[
            Filter(
                FieldRef("orders", "created_at"),
                ">=",
                {"macro": "last_30_days"},
            )
        ],
        order_by=[OrderBy("total_orders", "desc")],
    )

    sql = compile_query_plan(plan, "sqlite").sql

    assert "WHERE \"orders\".\"created_at\" >= date('now', '-30 days')" in sql


def test_ast_sql_preflight_allows_quoted_postgres_identifiers():
    schema = _schema(
        tables=[
            _table(
                "agents",
                [
                    {"name": "agent_id", "type": "text"},
                    {"name": "agent_name", "type": "text"},
                ],
            ),
            _table(
                "operations",
                [
                    {"name": "operation_id", "type": "uuid"},
                    {"name": "agent_id", "type": "text"},
                    {"name": "created_at", "type": "timestamp"},
                ],
            ),
        ]
    )
    sql = """
    SELECT
      "agents"."agent_id",
      "agents"."agent_name",
      COUNT("operations"."operation_id") AS "total_operations"
    FROM "agents"
    LEFT JOIN "operations" ON "agents"."agent_id" = "operations"."agent_id"
    WHERE "operations"."created_at" >= NOW() - INTERVAL '30 days'
    GROUP BY "agents"."agent_id", "agents"."agent_name"
    ORDER BY "total_operations" DESC
    LIMIT 10
    """

    assert validate_sql_against_schema(sql, schema, dialect="postgresql")["ok"] is True


def test_ast_sql_preflight_allows_mysql_backtick_identifiers():
    schema = _schema(
        db_type="mysql",
        tables=[
            _table(
                "agents",
                [
                    {"name": "agent_id", "type": "text"},
                    {"name": "agent_name", "type": "text"},
                ],
            )
        ],
    )
    sql = "SELECT `agents`.`agent_id`, `agents`.`agent_name` FROM `agents` LIMIT 10"

    assert validate_sql_against_schema(sql, schema, dialect="mysql")["ok"] is True


async def test_sql_facade_preflight_uses_query_validator_owner():
    plugin = SimpleNamespace(
        sql_dialect="postgresql",
        read_only=True,
        _tool_query=AsyncMock(return_value={"rows": []}),
        _tool_count=AsyncMock(return_value={"count": 0}),
        _tool_sample=AsyncMock(return_value={"rows": []}),
    )
    schema = _schema(
        tables=[
            _table(
                "customers",
                [
                    {"name": "customer_id", "type": "integer"},
                    {"name": "customer_name", "type": "text"},
                ],
            )
        ]
    )
    tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}

    result = await tools["db_query"].handler({"sql": "SELECT c.email FROM customers c"})

    assert result["error"] == "SQL preflight failed against known schema"
    assert result["missing_columns"] == [
        {"table": "customers", "column": "email", "reason": "column not found"}
    ]
    plugin._tool_query.assert_not_awaited()


async def test_sql_facade_delegates_guardrails_before_schema_preflight():
    def reject_mutating_cte(sql, *, operation):
        assert operation == "query"
        if "DELETE" in sql.upper():
            raise ValueError(
                "SQL guardrail rejected mutating statement in read-only mode: DELETE"
            )

    plugin = SimpleNamespace(
        sql_dialect="postgresql",
        read_only=True,
        _validate_sql_policy=reject_mutating_cte,
        _tool_query=AsyncMock(return_value={"rows": []}),
        _tool_count=AsyncMock(return_value={"count": 0}),
        _tool_sample=AsyncMock(return_value={"rows": []}),
    )
    schema = _schema(
        tables=[_table("orders", [{"name": "order_id", "type": "integer"}])]
    )
    tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}

    try:
        await tools["db_query"].handler(
            {
                "sql": (
                    "WITH deleted AS (DELETE FROM orders WHERE order_id = 1 "
                    "RETURNING *) SELECT * FROM deleted"
                )
            }
        )
    except ValueError as exc:
        assert "read-only mode" in str(exc)
    else:
        raise AssertionError("mutating CTE should be rejected before schema preflight")

    plugin._tool_query.assert_not_awaited()


async def test_sql_preflight_allows_schema_qualified_table_references():
    plugin = SimpleNamespace(
        sql_dialect="postgresql",
        read_only=True,
        _tool_query=AsyncMock(return_value={"rows": [{"revenue": 125}]}),
        _tool_count=AsyncMock(return_value={"count": 0}),
        _tool_sample=AsyncMock(return_value={"rows": []}),
    )
    schema = _schema(
        tables=[
            _table(
                "public.orders",
                [
                    {"name": "customer_id", "type": "integer"},
                    {"name": "total_amount", "type": "numeric"},
                ],
            )
        ]
    )
    tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}

    result = await tools["db_query"].handler(
        {
            "sql": (
                "SELECT SUM(total_amount) AS revenue "
                "FROM public.orders WHERE customer_id = 1"
            )
        }
    )

    assert result["rows"] == [{"revenue": 125}]
    plugin._tool_query.assert_awaited_once()
