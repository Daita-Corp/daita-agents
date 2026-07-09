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


def _requires_grounding_fact(literal: str = "completed") -> dict[str, str]:
    return {
        "kind": "filter_literal_requires_grounding",
        "table": "orders",
        "column": "status",
        "operator": "=",
        "literal": literal,
        "source": "query.plan.validation",
        "reason": "proposed_sql_filter_literal_without_accepted_value_evidence",
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
    context["relationship_evidence_details"] = [
        _relationship_evidence_detail(
            "relationship-customers-regions",
            "customers",
            "region_id",
            "regions",
            "id",
        )
    ]
    return context


def _order_refund_planning_context():
    context = _planning_context()
    context["schema"]["tables"] = [
        {
            "name": "orders",
            "columns": [
                {"name": "id", "data_type": "INTEGER"},
                {"name": "total", "data_type": "REAL"},
                {"name": "status", "data_type": "TEXT"},
                {"name": "customer_id", "data_type": "INTEGER"},
            ],
        },
        {
            "name": "refunds",
            "columns": [
                {"name": "id", "data_type": "INTEGER"},
                {"name": "order_id", "data_type": "INTEGER"},
                {"name": "amount", "data_type": "REAL"},
            ],
        },
    ]
    context["relationship_evidence_details"] = [
        _relationship_evidence_detail(
            "relationship-orders-refunds",
            "refunds",
            "order_id",
            "orders",
            "id",
        )
    ]
    return context


def _relationship_evidence_detail(
    evidence_id,
    left_table,
    left_column,
    right_table,
    right_column,
):
    return {
        "id": evidence_id,
        "kind": "schema.relationship_path",
        "owner": "catalog",
        "accepted": True,
        "payload_fingerprint": f"fp-{evidence_id}",
        "reachable": True,
        "paths": [
            {
                "tables": [left_table, right_table],
                "joins": [
                    {
                        "left_table": left_table,
                        "left_column": left_column,
                        "right_table": right_table,
                        "right_column": right_column,
                    }
                ],
            }
        ],
    }


def _board_revenue_context():
    context = _order_refund_planning_context()
    context["prompt"] = "Calculate board revenue"
    context["db_memory_semantics"] = [
        {
            "key": "metric:board_revenue",
            "memory_key": "metric:board_revenue",
            "kind": "metric_definition",
            "contract_kind": "metric_definition",
            "subject_aliases": ["board revenue"],
            "required_refs": ["orders.total", "refunds.amount", "orders.status"],
            "required_relationships": ["refunds.order_id -> orders.id"],
            "required_filters": [
                {
                    "ref": "orders.status",
                    "operator": "semantic_equals",
                    "value": "complete",
                }
            ],
            "required_aggregations": [
                {"function": "sum", "ref": "orders.total"},
                {"function": "sum", "ref": "refunds.amount"},
            ],
            "result_shape": {"grain": "single_aggregate"},
            "evidence_refs": ["evidence-memory"],
            "confidence": 0.95,
            "enforcement_mode": "required_when_recalled",
            "enforceable": True,
        }
    ]
    return context


def _unit_convention_context():
    context = _planning_context()
    context["prompt"] = "Show total revenue in dollars"
    context["schema"]["tables"] = [
        {
            "name": "orders",
            "columns": [
                {"name": "id", "data_type": "INTEGER"},
                {"name": "total_cents", "data_type": "INTEGER"},
            ],
        }
    ]
    context["db_memory_semantics"] = [
        {
            "key": "orders.total_cents",
            "memory_key": "unit_convention:orders.total_cents",
            "kind": "unit_convention",
            "contract_kind": "unit_convention",
            "subject_aliases": ["total cents"],
            "required_refs": ["orders.total_cents"],
            "unit_conversion": {
                "stored_unit": "cents",
                "display_unit": "dollars",
                "operator": "divide",
                "factor": 100,
            },
            "confidence": 0.9,
            "enforcement_mode": "required_when_recalled",
            "enforceable": True,
        }
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
    assert validation.validation_facts == (
        {
            "kind": "unobserved_filter_literal",
            "table": "orders",
            "column": "status",
            "literal": "completed",
            "candidates": ["complete", "pending"],
        },
    )
    assert validation.to_dict()["validation_facts"] == [
        {
            "kind": "unobserved_filter_literal",
            "table": "orders",
            "column": "status",
            "literal": "completed",
            "candidates": ["complete", "pending"],
        }
    ]


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


def test_join_sql_without_relationship_evidence_fails_query_plan_validation():
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
    context = _order_refund_planning_context()
    context["relationship_evidence_details"] = []

    validation = DbQueryPlanValidator().validate(plan, context)

    assert validation.valid is False
    assert validation.accepted_sql is None
    assert (
        "missing_catalog_relationship_path:orders.id->refunds.order_id"
        in validation.errors
    )
    assert validation.validation_facts[-1] == {
        "kind": "missing_catalog_relationship_path",
        "left_table": "orders",
        "right_table": "refunds",
        "source": "query_plan_join",
        "reason": "joined_data_query_without_catalog_relationship_evidence",
        "left_column": "id",
        "right_column": "order_id",
    }


def test_accepted_relationship_evidence_allows_matching_join():
    sql = (
        "SELECT COUNT(DISTINCT r.order_id) AS refunded_orders, "
        "COUNT(DISTINCT o.id) AS total_orders "
        "FROM orders o LEFT JOIN refunds r ON r.order_id = o.id"
    )
    plan = DbQueryPlan(
        operation="read",
        selected_sql=sql,
        selected_tables=("orders", "refunds"),
        confidence=0.96,
    )

    validation = DbQueryPlanValidator().validate(
        plan,
        _order_refund_planning_context(),
    )

    assert validation.valid is True
    assert validation.accepted_sql == sql
    relationship_validation = validation.metadata["catalog_relationship_validation"]
    assert relationship_validation["checked"] is True
    assert relationship_validation["passed"] is True
    assert relationship_validation["relationship_evidence_refs"] == [
        {
            "id": "relationship-orders-refunds",
            "kind": "schema.relationship_path",
            "owner": "catalog",
            "payload_fingerprint": "fp-relationship-orders-refunds",
        }
    ]


def test_validator_rejects_board_revenue_missing_ref_filter_aggregation_and_shape():
    plan = DbQueryPlan(
        operation="read",
        selected_sql="SELECT total FROM orders LIMIT 10",
        selected_tables=("orders",),
        confidence=0.9,
    )

    validation = DbQueryPlanValidator().validate(plan, _board_revenue_context())

    assert validation.valid is False
    assert (
        "missing_db_memory_required_ref:metric:board_revenue:refunds.amount"
        in validation.errors
    )
    assert (
        "missing_db_memory_required_filter:metric:board_revenue:orders.status=complete"
        in validation.errors
    )
    assert (
        "missing_db_memory_required_aggregation:metric:board_revenue:sum:orders.total"
        in validation.errors
    )
    assert (
        "missing_db_memory_required_result_shape:metric:board_revenue:single_aggregate"
        in validation.errors
    )


def test_validator_rejects_board_revenue_missing_required_join():
    plan = DbQueryPlan(
        operation="read",
        selected_sql=(
            "SELECT SUM(o.total) - SUM(r.amount) AS board_revenue "
            "FROM orders o, refunds r WHERE o.status = 'complete'"
        ),
        selected_tables=("orders", "refunds"),
        confidence=0.9,
    )

    validation = DbQueryPlanValidator().validate(plan, _board_revenue_context())

    assert validation.valid is False
    assert (
        "missing_db_memory_required_join:"
        "metric:board_revenue:refunds.order_id -> orders.id"
    ) in validation.errors


def test_validator_accepts_contract_compliant_board_revenue_sql():
    sql = (
        "SELECT SUM(o.total) - COALESCE(SUM(r.amount), 0) AS board_revenue "
        "FROM orders o LEFT JOIN refunds r ON r.order_id = o.id "
        "WHERE o.status = 'complete'"
    )
    plan = DbQueryPlan(
        operation="read",
        selected_sql=sql,
        selected_tables=("orders", "refunds"),
        confidence=0.95,
    )

    validation = DbQueryPlanValidator().validate(plan, _board_revenue_context())

    assert validation.valid is True
    assert validation.accepted_sql == sql


def test_validator_allows_incomplete_declared_join_when_sql_join_is_valid():
    sql = (
        "SELECT SUM(o.total) - COALESCE(SUM(r.amount), 0) AS board_revenue "
        "FROM orders AS o LEFT JOIN refunds AS r ON r.order_id = o.id "
        "WHERE o.status = 'complete'"
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
                    "on": "r.order_id = o.id",
                }
            ],
            "confidence": 0.95,
        }
    )

    validation = DbQueryPlanValidator().validate(plan, _board_revenue_context())

    assert validation.valid is True
    assert validation.accepted_sql == sql
    assert "join_left_column_not_declared:orders" in validation.warnings
    assert "join_right_column_not_declared:refunds" in validation.warnings


def test_validator_enforces_unit_convention_when_selected_expression_uses_column():
    plan = DbQueryPlan(
        operation="read",
        selected_sql="SELECT SUM(total_cents) AS revenue FROM orders",
        selected_tables=("orders",),
        confidence=0.9,
    )

    validation = DbQueryPlanValidator().validate(plan, _unit_convention_context())

    assert validation.valid is False
    assert (
        "missing_db_memory_required_unit_conversion:"
        "unit_convention:orders.total_cents:orders.total_cents:divide:100"
    ) in validation.errors


def test_validator_accepts_unit_convention_conversion():
    sql = "SELECT SUM(total_cents) / 100 AS revenue_dollars FROM orders"
    plan = DbQueryPlan(
        operation="read",
        selected_sql=sql,
        selected_tables=("orders",),
        confidence=0.9,
    )

    validation = DbQueryPlanValidator().validate(plan, _unit_convention_context())

    assert validation.valid is True
    assert validation.accepted_sql == sql


def test_validator_does_not_enforce_unit_convention_for_same_column_on_other_table():
    context = _unit_convention_context()
    context["schema"]["tables"].append(
        {
            "name": "invoices",
            "columns": [
                {"name": "id", "data_type": "INTEGER"},
                {"name": "total_cents", "data_type": "INTEGER"},
            ],
        }
    )
    sql = "SELECT SUM(total_cents) AS invoice_total FROM invoices"
    plan = DbQueryPlan(
        operation="read",
        selected_sql=sql,
        selected_tables=("invoices",),
        confidence=0.9,
    )

    validation = DbQueryPlanValidator().validate(plan, context)

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

    assert validation.valid is False
    assert validation.accepted_sql is None
    assert validation.validation_facts == (_requires_grounding_fact(),)
    assert not any(
        str(error).startswith("unobserved_filter_literal:")
        for error in validation.errors
    )


def test_planning_context_column_value_hints_only_use_hint_evidence():
    profile = {
        "table": "orders",
        "column": "status",
        "profile_status": "profiled",
        "top_values": [
            {"value": "complete", "count": 4},
            {"value": "pending", "count": 1},
        ],
    }
    schema_with_metadata_profiles = {
        **_planning_context()["schema"],
        "metadata": {"column_value_profiles": {"orders.status": profile}},
    }
    profile_evidence = Evidence(
        kind="schema.column_value_profile",
        owner="catalog",
        payload={"profiles": [profile]},
    )
    search_evidence = Evidence(
        kind="schema.column_value_search_result",
        owner="catalog",
        payload={"profiles": [profile]},
    )
    hint_evidence = Evidence(
        kind="schema.column_value_hint",
        owner="catalog",
        payload={
            "hints": [
                {
                    "table": "orders",
                    "column": "status",
                    "profile_status": "profiled",
                    "observed_values": profile["top_values"],
                }
            ]
        },
    )
    plan = DbQueryPlan(
        operation="read",
        selected_sql="SELECT * FROM orders WHERE status = 'completed' LIMIT 10",
        selected_tables=("orders",),
        confidence=0.9,
    )

    assert _column_value_hints((profile_evidence,), _planning_context()["schema"]) == ()
    assert _column_value_hints((search_evidence,), _planning_context()["schema"]) == ()
    assert _column_value_hints((), schema_with_metadata_profiles) == ()

    hint_hints = _column_value_hints((hint_evidence,), _planning_context()["schema"])
    assert hint_hints == (
        {
            "table": "orders",
            "column": "status",
            "profile_ref": "orders.status",
            "distinct_count": None,
            "observed_values": [
                {"value": "complete", "count": 4},
                {"value": "pending", "count": 1},
            ],
            "profile_status": "profiled",
            "sampled": False,
            "truncated": False,
            "redacted": False,
            "stale": False,
        },
    )
    context = _planning_context()
    context["column_value_hints"] = list(hint_hints)
    validation = DbQueryPlanValidator().validate(plan, context)
    assert validation.valid is False
    assert (
        "unobserved_filter_literal:orders.status=completed;"
        "candidates=complete,pending"
    ) in validation.errors
