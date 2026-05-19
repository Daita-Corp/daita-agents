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
from daita.agents.db.query.sql_validator import validate_sql_against_schema
from daita.agents.db.runtime.state import DbRunState, set_db_run_state
from daita.agents.db.tools.query import create_db_query_tools
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
    registered = await catalog.register_schema(schema, store_type=schema["database_type"])
    plugin._db_catalog = catalog
    plugin._db_catalog_store_id = registered["store_id"]
    return catalog, registered["store_id"]


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


def test_catalog_adapter_requires_catalog_for_search_and_paths():
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
    paths = find_relationship_paths(
        schema,
        from_tables=["orders"],
        to_tables=["customers"],
    )

    assert search["source"] == "missing_catalog"
    assert search["tables"] == []
    assert paths["source"] == "missing_catalog"
    assert paths["reachable"] is False


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
    set_db_run_state(plugin, DbRunState())

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

    assert plan_result["next_steps"] == ["run db_query with plan_id"]
    assert "query_ir" not in plan_result
    assert result == {"rows": [{"total_orders": 3}]}
    plugin._tool_query.assert_awaited_once()
    assert plugin._tool_query.await_args.args[0]["sql"] == plan_result["compiled_sql"]


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
    assert "table_candidates" not in compact
    assert "field_candidates" not in compact
    assert diagnostic["query_ir"]["metrics"][0]["name"] == "total_orders"
    assert diagnostic["table_candidates"]
    assert diagnostic["field_candidates"]


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
