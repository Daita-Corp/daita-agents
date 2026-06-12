from daita.db.plan_validation import DbQueryPlanValidator
from daita.db.planning_context import _column_value_hints
from daita.db.query_plan import DbFilterSpec, DbQueryPlan
from daita.runtime import Evidence


def _planning_context():
    return {
        "dialect": "sqlite",
        "schema": {
            "database_type": "sqlite",
            "tables": [
                {
                    "name": "orders",
                    "columns": [
                        {"name": "id", "data_type": "INTEGER"},
                        {"name": "total", "data_type": "REAL"},
                        {"name": "status", "data_type": "TEXT"},
                        {"name": "severity", "data_type": "TEXT"},
                    ],
                }
            ],
        },
        "column_value_hints": [
            {
                "table": "orders",
                "column": "status",
                "profile_status": "profiled",
                "observed_values": [
                    {"value": "complete", "count": 4},
                    {"value": "pending", "count": 1},
                ],
            },
            {
                "table": "orders",
                "column": "severity",
                "profile_status": "profiled",
                "observed_values": [
                    {"value": "high", "count": 2},
                    {"value": "low", "count": 3},
                ],
            },
        ],
    }


def _relationship_planning_context():
    context = _planning_context()
    context["schema"]["tables"] = [
        {
            "name": "customers",
            "columns": [
                {"name": "id", "data_type": "INTEGER"},
                {"name": "name", "data_type": "TEXT"},
                {"name": "region_id", "data_type": "INTEGER"},
            ],
        },
        {
            "name": "regions",
            "columns": [
                {"name": "id", "data_type": "INTEGER"},
                {"name": "name", "data_type": "TEXT"},
            ],
        },
    ]
    return context


def _order_refund_planning_context():
    context = _planning_context()
    context["schema"]["tables"] = [
        {
            "name": "orders",
            "columns": [
                {"name": "id", "data_type": "INTEGER"},
                {"name": "customer_id", "data_type": "INTEGER"},
            ],
        },
        {
            "name": "refunds",
            "columns": [
                {"name": "id", "data_type": "INTEGER"},
                {"name": "order_id", "data_type": "INTEGER"},
            ],
        },
    ]
    return context


def test_validator_rejects_unobserved_filter_literal_from_known_values():
    plan = DbQueryPlan(
        operation="read",
        selected_sql="SELECT * FROM orders WHERE status = 'completed' LIMIT 10",
        selected_tables=("orders",),
        filters=(
            DbFilterSpec(
                column="orders.status",
                operator="=",
                value="completed",
            ),
        ),
        confidence=0.9,
    )

    validation = DbQueryPlanValidator().validate(plan, _planning_context())

    assert validation.valid is False
    assert validation.accepted_sql is None
    assert (
        "unobserved_filter_literal:orders.status=completed;"
        "candidates=complete,pending"
    ) in validation.errors


def test_validator_accepts_observed_filter_literal_from_known_values():
    plan = DbQueryPlan(
        operation="read",
        selected_sql="SELECT * FROM orders WHERE status = 'complete' LIMIT 10",
        selected_tables=("orders",),
        filters=(
            DbFilterSpec(
                column="orders.status",
                operator="=",
                value="complete",
            ),
        ),
        confidence=0.9,
    )

    validation = DbQueryPlanValidator().validate(plan, _planning_context())

    assert validation.valid is True
    assert validation.accepted_sql == plan.selected_sql


def test_validator_rejects_declared_filter_missing_from_sql():
    plan = DbQueryPlan(
        operation="read",
        selected_sql="SELECT * FROM orders WHERE status = 'complete' LIMIT 10",
        selected_tables=("orders",),
        filters=(
            DbFilterSpec(
                column="orders.status",
                operator="=",
                value="complete",
            ),
            DbFilterSpec(
                column="orders.severity",
                operator="=",
                value="high",
            ),
        ),
        confidence=0.9,
    )

    validation = DbQueryPlanValidator().validate(plan, _planning_context())

    assert validation.valid is False
    assert validation.accepted_sql is None
    assert "declared_filter_not_in_sql:orders.severity=high" in validation.errors


def test_validator_accepts_declared_filters_present_in_sql():
    plan = DbQueryPlan(
        operation="read",
        selected_sql=(
            "SELECT * FROM orders "
            "WHERE status = 'complete' AND severity = 'high' LIMIT 10"
        ),
        selected_tables=("orders",),
        filters=(
            DbFilterSpec(
                column="orders.status",
                operator="=",
                value="complete",
            ),
            DbFilterSpec(
                column="orders.severity",
                operator="=",
                value="high",
            ),
        ),
        confidence=0.9,
    )

    validation = DbQueryPlanValidator().validate(plan, _planning_context())

    assert validation.valid is True
    assert validation.accepted_sql == plan.selected_sql


def test_validator_matches_declared_filter_against_sql_alias():
    plan = DbQueryPlan(
        operation="read",
        selected_sql=(
            "SELECT * FROM orders AS o "
            "WHERE o.status = 'complete' AND o.severity = 'high' LIMIT 10"
        ),
        selected_tables=("orders",),
        filters=(
            DbFilterSpec(
                column="orders.status",
                operator="=",
                value="complete",
            ),
            DbFilterSpec(
                column="orders.severity",
                operator="=",
                value="high",
            ),
        ),
        confidence=0.9,
    )

    validation = DbQueryPlanValidator().validate(plan, _planning_context())

    assert validation.valid is True
    assert validation.accepted_sql == plan.selected_sql


def test_validator_matches_declared_comparison_filter():
    plan = DbQueryPlan(
        operation="read",
        selected_sql="SELECT * FROM orders WHERE total > 40 LIMIT 10",
        selected_tables=("orders",),
        filters=(
            DbFilterSpec(
                column="orders.total",
                operator=">",
                value="40",
            ),
        ),
        confidence=0.9,
    )

    validation = DbQueryPlanValidator().validate(plan, _planning_context())

    assert validation.valid is True
    assert validation.accepted_sql == plan.selected_sql


def test_plan_mapping_infers_join_columns_from_relationship_text():
    plan = DbQueryPlan.from_mapping(
        {
            "operation": "read",
            "selected_sql": (
                "SELECT customers.id AS customer_id, customers.name, regions.name "
                "AS region_name FROM customers JOIN regions "
                "ON customers.region_id = regions.id LIMIT 200"
            ),
            "selected_tables": ["customers", "regions"],
            "joins": [
                {
                    "left_table": "customers",
                    "right_table": "regions",
                    "condition": "customers.region_id = regions.id",
                    "relationship": "customers.region_id -> regions.id",
                }
            ],
            "limit": 200,
            "confidence": 0.99,
        }
    )

    validation = DbQueryPlanValidator().validate(
        plan,
        _relationship_planning_context(),
    )

    assert plan.joins[0].left_column == "region_id"
    assert plan.joins[0].right_column == "id"
    assert validation.valid is True
    assert validation.accepted_sql == plan.selected_sql


def test_plan_mapping_accepts_llm_join_key_aliases():
    sql = (
        "SELECT COUNT(DISTINCT r.order_id) AS refunded_orders, "
        "COUNT(DISTINCT o.id) AS total_orders "
        "FROM orders o LEFT JOIN refunds r ON r.order_id = o.id"
    )
    plan = DbQueryPlan.from_mapping(
        {
            "operation": "read",
            "selected_sql": sql,
            "selected_tables": ["orders", "refunds"],
            "joins": [
                {
                    "left_table": "orders",
                    "right_table": "refunds",
                    "left_key": "orders.id",
                    "right_key": "refunds.order_id",
                }
            ],
            "confidence": 0.96,
        }
    )

    validation = DbQueryPlanValidator().validate(
        plan,
        _order_refund_planning_context(),
    )

    assert plan.joins[0].left_column == "id"
    assert plan.joins[0].right_column == "order_id"
    assert validation.valid is True
    assert validation.accepted_sql == sql


def test_validator_rejects_unobserved_sql_in_literal_with_table_alias():
    plan = DbQueryPlan(
        operation="read",
        selected_sql=(
            "SELECT * FROM orders AS o "
            "WHERE o.status IN ('complete', 'cancelled') LIMIT 10"
        ),
        selected_tables=("orders",),
        confidence=0.9,
    )

    validation = DbQueryPlanValidator().validate(plan, _planning_context())

    assert validation.valid is False
    assert (
        "unobserved_filter_literal:orders.status=cancelled;"
        "candidates=complete,pending"
    ) in validation.errors


def test_validator_handles_reversed_sql_string_equality():
    plan = DbQueryPlan(
        operation="read",
        selected_sql="SELECT * FROM orders WHERE 'completed' = status LIMIT 10",
        selected_tables=("orders",),
        confidence=0.9,
    )

    validation = DbQueryPlanValidator().validate(plan, _planning_context())

    assert validation.valid is False
    assert (
        "unobserved_filter_literal:orders.status=completed;"
        "candidates=complete,pending"
    ) in validation.errors


def test_validator_does_not_exact_match_like_predicates():
    plan = DbQueryPlan(
        operation="read",
        selected_sql="SELECT * FROM orders WHERE status LIKE 'complete%' LIMIT 10",
        selected_tables=("orders",),
        confidence=0.9,
    )

    validation = DbQueryPlanValidator().validate(plan, _planning_context())

    assert validation.valid is True
    assert validation.accepted_sql == plan.selected_sql


def test_validator_does_not_enforce_stale_value_hints():
    context = _planning_context()
    context["column_value_hints"] = [
        {
            **context["column_value_hints"][0],
            "profile_status": "stale",
            "stale": True,
        }
    ]
    plan = DbQueryPlan(
        operation="read",
        selected_sql="SELECT * FROM orders WHERE status = 'completed' LIMIT 10",
        selected_tables=("orders",),
        confidence=0.9,
    )

    validation = DbQueryPlanValidator().validate(plan, context)

    assert validation.valid is True
    assert validation.accepted_sql == plan.selected_sql


def test_planning_context_and_validator_share_value_hint_eligibility():
    unsafe_profiles = [
        {"profile_status": "stale", "stale": True},
        {"profile_status": "profiled", "redacted": True},
        {"profile_status": "profiled", "sampled": True},
        {"profile_status": "profiled", "truncated": True},
    ]
    plan = DbQueryPlan(
        operation="read",
        selected_sql="SELECT * FROM orders WHERE status = 'completed' LIMIT 10",
        selected_tables=("orders",),
        confidence=0.9,
    )

    for overrides in unsafe_profiles:
        profile = {
            "table": "orders",
            "column": "status",
            "profile_status": "profiled",
            "top_values": [
                {"value": "complete", "count": 4},
                {"value": "pending", "count": 1},
            ],
            **overrides,
        }
        hints = _column_value_hints(
            (
                Evidence(
                    kind="schema.column_value_search_result",
                    owner="catalog",
                    payload={"profiles": [profile]},
                ),
            ),
            _planning_context()["schema"],
        )
        context = _planning_context()
        context["column_value_hints"] = [
            {
                "table": "orders",
                "column": "status",
                "profile_status": profile["profile_status"],
                "stale": bool(profile.get("stale", False)),
                "redacted": bool(profile.get("redacted", False)),
                "sampled": bool(profile.get("sampled", False)),
                "truncated": bool(profile.get("truncated", False)),
                "observed_values": profile["top_values"],
            }
        ]

        assert hints == ()
        validation = DbQueryPlanValidator().validate(plan, context)
        assert validation.valid is True

    eligible_hints = _column_value_hints(
        (
            Evidence(
                kind="schema.column_value_search_result",
                owner="catalog",
                payload={
                    "profiles": [
                        {
                            "table": "orders",
                            "column": "status",
                            "profile_status": "profiled",
                            "top_values": [
                                {"value": "complete", "count": 4},
                                {"value": "pending", "count": 1},
                            ],
                        }
                    ]
                },
            ),
        ),
        _planning_context()["schema"],
    )
    context = _planning_context()
    context["column_value_hints"] = list(eligible_hints)

    assert eligible_hints
    validation = DbQueryPlanValidator().validate(plan, context)
    assert validation.valid is False
