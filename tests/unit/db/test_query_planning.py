from daita.db import DbIntent, DbIntentKind, DbQueryPlan, DbQueryPlanner, DbRequest
from daita.runtime import AccessMode, Operation


def _schema():
    return {
        "tables": [
            {
                "name": "customers",
                "columns": [
                    {"name": "id"},
                    {"name": "email"},
                    {"name": "status"},
                    {"name": "region_id"},
                ],
            },
            {
                "name": "regions",
                "columns": [
                    {"name": "id"},
                    {"name": "name"},
                ],
            },
            {
                "name": "orders",
                "columns": [
                    {"name": "id"},
                    {"name": "customer_id"},
                    {"name": "total"},
                ],
            },
            {
                "name": "support_tickets",
                "columns": [
                    {"name": "id"},
                    {"name": "customer_id"},
                    {"name": "status"},
                    {"name": "severity"},
                ],
            },
        ],
        "foreign_keys": [
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
            {
                "source_table": "customers",
                "source_column": "region_id",
                "target_table": "regions",
                "target_column": "id",
            },
        ],
    }


def _schema_without_customer_status():
    schema = _schema()
    return {
        **schema,
        "tables": [
            {
                **table,
                "columns": [
                    column
                    for column in table.get("columns", [])
                    if not (
                        table.get("name") == "customers"
                        and column.get("name") == "status"
                    )
                ],
            }
            for table in schema["tables"]
        ],
    }


def test_query_planner_builds_count_plan_without_executing_sql():
    planner = DbQueryPlanner()
    plan = planner.plan_read_query(
        DbRequest("How many orders are there?"),
        DbIntent(kind=DbIntentKind.DATA_QUERY, access=AccessMode.READ),
        Operation(id="op-1", operation_type="data.query"),
        _schema(),
    )

    assert plan.sql == 'SELECT COUNT(*) AS count FROM "orders"'
    assert plan.evidence.kind == "query.plan.proposal"
    assert plan.evidence.payload["strategy"] == "single_table"


def test_query_plan_normalizes_common_llm_operation_aliases():
    plan = DbQueryPlan.from_mapping(
        {
            "operation": "query_planning",
            "selected_sql": "SELECT COUNT(*) AS count FROM customers",
            "selected_tables": ["customers"],
            "confidence": 0.8,
        }
    )

    assert plan.operation == "read"


def test_query_planner_uses_catalog_metadata_business_name_for_table_match():
    schema = {
        "tables": [
            {
                "name": "events",
                "columns": [{"name": "id"}],
            },
            {
                "name": "users",
                "columns": [{"name": "user_id"}],
                "metadata": {"business_name": "signups"},
            },
        ]
    }
    planner = DbQueryPlanner()
    plan = planner.plan_read_query(
        DbRequest("How many signups?"),
        DbIntent(kind=DbIntentKind.DATA_QUERY, access=AccessMode.READ),
        Operation(id="op-1", operation_type="data.query"),
        schema,
    )

    assert plan.sql == 'SELECT COUNT(*) AS count FROM "users"'


def test_query_planner_builds_filtered_select_plan():
    planner = DbQueryPlanner()
    plan = planner.plan_read_query(
        DbRequest("List orders where total > 40"),
        DbIntent(kind=DbIntentKind.DATA_QUERY, access=AccessMode.READ),
        Operation(id="op-1", operation_type="data.query"),
        _schema(),
    )

    assert plan.sql == 'SELECT * FROM "orders" WHERE "total" > 40'
    structured = plan.evidence.payload["structured_plan"]
    assert structured["filters"] == [
        {"column": "orders.total", "operator": ">", "value": "40"}
    ]


def test_query_planner_preserves_multiple_prompt_filters():
    planner = DbQueryPlanner()
    plan = planner.plan_read_query(
        DbRequest("Show support_tickets where status = 'open' and severity = 'high'"),
        DbIntent(kind=DbIntentKind.DATA_QUERY, access=AccessMode.READ),
        Operation(id="op-1", operation_type="data.query"),
        _schema(),
    )

    assert plan.sql == (
        'SELECT * FROM "support_tickets" '
        "WHERE \"status\" = 'open' AND \"severity\" = 'high'"
    )
    structured = plan.evidence.payload["structured_plan"]
    assert structured["filters"] == [
        {"column": "support_tickets.status", "operator": "=", "value": "open"},
        {"column": "support_tickets.severity", "operator": "=", "value": "high"},
    ]


def test_query_planner_preserves_explicit_filters_in_count_plan():
    planner = DbQueryPlanner()
    plan = planner.plan_read_query(
        DbRequest("How many support_tickets have status='open' and severity='high'?"),
        DbIntent(kind=DbIntentKind.DATA_QUERY, access=AccessMode.READ),
        Operation(id="op-1", operation_type="data.query"),
        _schema(),
    )

    assert plan.sql == (
        'SELECT COUNT(*) AS count FROM "support_tickets" '
        "WHERE \"status\" = 'open' AND \"severity\" = 'high'"
    )
    structured = plan.evidence.payload["structured_plan"]
    assert structured["filters"] == [
        {"column": "support_tickets.status", "operator": "=", "value": "open"},
        {"column": "support_tickets.severity", "operator": "=", "value": "high"},
    ]


def test_query_planner_preserves_explicit_filters_for_spaced_table_phrase_count():
    planner = DbQueryPlanner()
    plan = planner.plan_read_query(
        DbRequest("How many support tickets have status='open' and severity='high'?"),
        DbIntent(kind=DbIntentKind.DATA_QUERY, access=AccessMode.READ),
        Operation(id="op-1", operation_type="data.query"),
        _schema(),
    )

    assert plan.sql == (
        'SELECT COUNT(*) AS count FROM "support_tickets" '
        "WHERE \"status\" = 'open' AND \"severity\" = 'high'"
    )


def test_query_planner_value_hint_context_gate_is_schema_derived():
    planner = DbQueryPlanner()

    assert (
        planner.needs_value_hint_context(
            "How many orders are there?",
            _schema(),
        )
        is False
    )
    assert (
        planner.needs_value_hint_context(
            "Count open high severity support tickets.",
            _schema(),
        )
        is True
    )
    assert (
        planner.needs_value_hint_context(
            "How many support tickets have status='open'?",
            _schema(),
        )
        is False
    )


def test_query_planner_builds_join_plan_from_relationship_payload():
    planner = DbQueryPlanner()
    plan = planner.plan_read_query(
        DbRequest("Join orders to customers using their relationship"),
        DbIntent(
            kind=DbIntentKind.CATALOG_ASSISTED_DATA_QUERY,
            access=AccessMode.READ,
        ),
        Operation(id="op-1", operation_type="data.query"),
        _schema(),
        relationship_payload={
            "paths": [
                {
                    "joins": [
                        {
                            "left_table": "orders",
                            "left_column": "customer_id",
                            "right_table": "customers",
                            "right_column": "id",
                        }
                    ]
                }
            ]
        },
    )

    assert 'JOIN "customers"' in plan.sql
    assert '"orders"."customer_id" = "customers"."id"' in plan.sql
    assert plan.evidence.payload["strategy"] == "catalog_relationship_join"


def test_query_planner_preserves_and_qualifies_join_filters():
    planner = DbQueryPlanner()
    plan = planner.plan_read_query(
        DbRequest(
            "Join customers to support_tickets where status='open' and severity='high'"
        ),
        DbIntent(
            kind=DbIntentKind.CATALOG_ASSISTED_DATA_QUERY,
            access=AccessMode.READ,
        ),
        Operation(id="op-1", operation_type="data.query"),
        _schema_without_customer_status(),
    )

    assert 'JOIN "customers"' in plan.sql
    assert '"support_tickets"."customer_id" = "customers"."id"' in plan.sql
    assert (
        'WHERE "support_tickets"."status" = \'open\' '
        'AND "support_tickets"."severity" = \'high\''
    ) in plan.sql
    structured = plan.evidence.payload["structured_plan"]
    assert structured["filters"] == [
        {"column": "support_tickets.status", "operator": "=", "value": "open"},
        {"column": "support_tickets.severity", "operator": "=", "value": "high"},
    ]


def test_query_planner_normalizes_join_endpoint_phrases():
    planner = DbQueryPlanner()
    plan = planner.plan_read_query(
        DbRequest(
            "Which customers have open high severity support tickets? Join customers to support tickets."
        ),
        DbIntent(
            kind=DbIntentKind.CATALOG_ASSISTED_DATA_QUERY,
            access=AccessMode.READ,
        ),
        Operation(id="op-1", operation_type="data.query"),
        _schema(),
    )

    assert plan.sql is not None
    assert '"support_tickets"."customer_id" = "customers"."id"' in plan.sql
    assert '"regions"' not in plan.sql


def test_query_planner_does_not_guess_ambiguous_join_filter_table():
    planner = DbQueryPlanner()
    plan = planner.plan_read_query(
        DbRequest("Join customers to support tickets where status='open'"),
        DbIntent(
            kind=DbIntentKind.CATALOG_ASSISTED_DATA_QUERY,
            access=AccessMode.READ,
        ),
        Operation(id="op-1", operation_type="data.query"),
        _schema(),
    )

    assert plan.sql is None
    assert "db_runtime_ambiguous_filter_column:status" in plan.warnings
    structured = plan.evidence.payload["structured_plan"]
    assert structured["filters"] == [
        {"column": "status", "operator": "=", "value": "open"}
    ]
