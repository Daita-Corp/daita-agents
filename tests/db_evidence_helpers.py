def assert_no_invalid_accepted_query_plans(evidence):
    for item in evidence:
        if item.kind == "query.plan.proposal" and item.accepted:
            assert item.payload.get("valid") is True
            assert isinstance(item.payload.get("sql"), str)
            assert item.payload["sql"].strip()
