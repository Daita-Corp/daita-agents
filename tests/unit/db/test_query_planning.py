from daita.db.query_plan import DbQueryPlan


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
