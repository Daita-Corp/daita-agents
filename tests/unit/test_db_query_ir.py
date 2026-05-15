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
from daita.agents.db.tools.query_facade import create_db_query_tools


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

    tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}
    result = await tools["db_plan_query"].handler(
        {
            "goal": "Which customer has the most orders this month?",
            "required_fields": ["customer_id", "customer_name", "total_orders"],
            "candidate_tables": ["orders", "customers"],
            "grouping": ["customers.customer_id", "customers.customer_name"],
            "ordering": ["total_orders desc"],
            "limit": 1,
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
    assert result == {"rows": [{"total_orders": 3}]}
    plugin._tool_query.assert_awaited_once()
    assert plugin._tool_query.await_args.args[0]["sql"] == plan_result["compiled_sql"]


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
