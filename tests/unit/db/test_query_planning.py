from daita.db import DbIntent, DbIntentKind, DbQueryPlanner, DbRequest
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
        ],
        "foreign_keys": [
            {
                "source_table": "orders",
                "source_column": "customer_id",
                "target_table": "customers",
                "target_column": "id",
            }
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
    assert plan.evidence.kind == "query.plan"
    assert plan.evidence.payload["strategy"] == "single_table"


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
