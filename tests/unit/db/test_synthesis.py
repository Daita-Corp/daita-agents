import pytest

from daita.db import (
    DbIntent,
    DbIntentKind,
    DbOperationContract,
    DbRequest,
)
from daita.db.synthesis import (
    DbAnswerCitation,
    DbAnswerSynthesisPayload,
    DbSynthesizer,
    _apply_schema_db_memory_annotation,
    _monitor_answer,
    _monitor_answer_from_evidence,
    build_synthesis_context,
    deterministic_synthesis_payload,
)
from daita.db.verification import DbVerificationResult
from daita.runtime import AccessMode, Evidence


def _verification(
    passed: bool = True, evidence_id: str | None = None
) -> DbVerificationResult:
    return DbVerificationResult(
        passed=passed,
        missing_evidence=(),
        warnings=(),
        diagnostics={},
        evidence_refs=(
            {
                "id": evidence_id,
                "kind": "query.result",
                "owner": None,
                "task_id": "task-2",
            },
        ),
    )


def _data_intent() -> DbIntent:
    return DbIntent(kind=DbIntentKind.DATA_QUERY, access=AccessMode.READ)


def _data_contract() -> DbOperationContract:
    return DbOperationContract(
        operation_type="data.query",
        required_evidence=("sql.validation", "query.result"),
    )


def _query_result(
    rows,
    *,
    sql: str = "SELECT COUNT(*) AS count FROM orders",
    total_rows: int | None = None,
    truncated: bool = False,
) -> Evidence:
    payload = {"rows": rows, "sql": sql, "truncated": truncated}
    if total_rows is not None:
        payload["total_rows"] = total_rows
    return Evidence(
        id="query-result-1",
        kind="query.result",
        accepted=True,
        task_id="task-2",
        payload=payload,
    )


def test_monitor_synthesis_prefers_resolution_over_prior_inbox_state():
    answer = _monitor_answer(
        (
            Evidence(
                kind="monitor.approval_state",
                payload={
                    "approvals": [
                        {
                            "approval_id": "approval-1",
                            "status": "pending",
                            "monitor_id": "pending_orders",
                        }
                    ]
                },
            ),
            Evidence(
                kind="monitor.approval_resolution",
                payload={
                    "status": "resolved",
                    "approval_action": "approve",
                    "approval_id": "approval-1",
                    "approval_status": "approved",
                },
            ),
        )
    )

    assert answer == "Approved monitor approval approval-1; approval is approved."


@pytest.mark.parametrize(
    ("status", "expected"),
    (
        (
            "inbox_required",
            "Read the pending monitor approval inbox before resolving an approval.",
        ),
        (
            "inbox_incomplete",
            "The monitor approval inbox result was incomplete; no approval was changed.",
        ),
        ("unexpected", "The monitor approval was not changed."),
    ),
)
def test_monitor_synthesis_does_not_report_unresolved_approval_as_success(
    status,
    expected,
):
    answer = _monitor_answer_from_evidence(
        Evidence(
            kind="monitor.approval_resolution",
            payload={"status": status, "approval_action": "approve"},
        )
    )

    assert answer == expected


def test_monitor_approval_state_synthesis_uses_bounded_monitor_id():
    answer = _monitor_answer_from_evidence(
        Evidence(
            kind="monitor.approval_state",
            payload={
                "approvals": [
                    {
                        "approval_id": "approval-1",
                        "status": "pending",
                        "monitor_id": "pending_orders",
                    }
                ]
            },
        )
    )

    assert (
        answer == "Pending monitor approvals:\n- approval-1: pending for pending_orders"
    )


def test_synthesizer_answers_count_from_query_result_evidence_only():
    result = DbSynthesizer().synthesize(
        request=DbRequest("How many orders are there?"),
        intent=_data_intent(),
        contract=_data_contract(),
        evidence=(
            Evidence(
                kind="query.result",
                accepted=True,
                task_id="task-2",
                payload={"rows": [{"count": 2}], "sql": "SELECT COUNT(*)"},
            ),
        ),
        verification=_verification(),
    )

    assert result.answer == "The count is 2."
    assert result.evidence_refs[0]["kind"] == "query.result"


@pytest.mark.parametrize(
    ("label", "expected"),
    (
        ("count", "The count is 4."),
        ("customer_count", "customer_count is 4."),
        ("total_orders", "total_orders is 4."),
        ("completed_orders", "completed_orders is 4."),
    ),
)
def test_synthesizer_answers_single_column_count_aliases(label, expected):
    result = DbSynthesizer().synthesize(
        request=DbRequest("How many customers are there?"),
        intent=_data_intent(),
        contract=_data_contract(),
        evidence=(
            _query_result(
                [{label: 4}],
                sql=f"SELECT COUNT(*) AS {label} FROM customers",
            ),
        ),
        verification=_verification(evidence_id="query-result-1"),
    )

    assert result.answer == expected
    assert result.diagnostics["answer_facts"]["primary_scalar"]["value"] == 4
    assert (
        result.diagnostics["answer_facts"]["primary_scalar"]["aggregate_kind"]
        == "count"
    )


def test_synthesizer_preserves_sensitive_named_count_alias_value():
    result = DbSynthesizer().synthesize(
        request=DbRequest("How many customer emails are there?"),
        intent=_data_intent(),
        contract=_data_contract(),
        evidence=(
            _query_result(
                [{"email_count": 4}],
                sql="SELECT COUNT(email) AS email_count FROM customers",
            ),
        ),
        verification=_verification(evidence_id="query-result-1"),
    )

    primary = result.diagnostics["answer_facts"]["primary_scalar"]
    assert result.answer == "email_count is 4."
    assert primary["value"] == 4
    assert primary["aggregate_kind"] == "count"
    assert primary["redacted"] is False


def test_synthesizer_refuses_to_answer_before_verification_passes():
    with pytest.raises(ValueError, match="verification passes"):
        DbSynthesizer().synthesize(
            request=DbRequest("How many orders?"),
            intent=DbIntent(kind=DbIntentKind.DATA_QUERY, access=AccessMode.READ),
            contract=DbOperationContract(operation_type="data.query"),
            evidence=(),
            verification=_verification(passed=False),
        )


def test_synthesis_context_derives_answer_facts_for_result_shapes():
    request = DbRequest("Summarize the query result.")
    record_context = build_synthesis_context(
        request=request,
        intent=_data_intent(),
        contract=_data_contract(),
        evidence=(
            _query_result(
                [
                    {
                        "refunded_orders": 1,
                        "total_orders": 5,
                        "refund_rate_percent": 20,
                    }
                ],
                sql=(
                    "SELECT COUNT(*) AS refunded_orders, "
                    "COUNT(*) AS total_orders, "
                    "20 AS refund_rate_percent FROM orders"
                ),
            ),
        ),
    )
    record_facts = record_context.metadata["answer_facts"]
    assert record_facts["result_shape"] == "record"
    assert [item["value"] for item in record_facts["scalars"]] == [1, 5, 20]
    assert [item["aggregate_kind"] for item in record_facts["scalars"][:2]] == [
        "count",
        "count",
    ]

    table_context = build_synthesis_context(
        request=request,
        intent=_data_intent(),
        contract=_data_contract(),
        evidence=(_query_result([{"id": 1}, {"id": 2}], total_rows=2),),
    )
    table_facts = table_context.metadata["answer_facts"]
    assert table_facts["result_shape"] == "table"
    assert table_facts["row_count"] == 2
    assert table_facts["scalars"] == []

    empty_context = build_synthesis_context(
        request=request,
        intent=_data_intent(),
        contract=_data_contract(),
        evidence=(_query_result([], total_rows=0),),
    )
    assert empty_context.metadata["answer_facts"]["result_shape"] == "empty"

    truncated_context = build_synthesis_context(
        request=request,
        intent=_data_intent(),
        contract=_data_contract(),
        evidence=(
            _query_result([{"id": 1}, {"id": 2}], total_rows=10, truncated=True),
        ),
    )
    truncated_facts = truncated_context.metadata["answer_facts"]
    assert truncated_facts["result_shape"] == "table"
    assert truncated_facts["row_count"] == 10
    assert truncated_facts["truncated"] is True


def test_deterministic_fallback_uses_answer_facts_when_context_is_truncated():
    query_result = _query_result(
        [{"customer_count": 4}],
        sql="SELECT COUNT(*) AS customer_count FROM customers",
    )
    context = build_synthesis_context(
        request=DbRequest("How many customers are there?"),
        intent=_data_intent(),
        contract=_data_contract(),
        evidence=(query_result,),
        char_budget=1,
    )

    payload = deterministic_synthesis_payload(
        request=DbRequest("How many customers are there?"),
        intent=_data_intent(),
        contract=_data_contract(),
        evidence=(query_result,),
        verification=_verification(evidence_id="query-result-1"),
        context_metadata=context.metadata,
        fallback_reason="test",
    )

    assert context.metadata["truncation"]["context_chars_truncated"] is True
    assert payload.answer == "customer_count is 4."
    assert payload.sufficiency == "answered"
    assert "synthesis_context_truncated" not in payload.limitations
    assert "synthesis_context_truncated" not in payload.warnings
    assert payload.answer_facts["primary_scalar"]["value"] == 4


def test_schema_synthesis_uses_database_scope_for_database_wide_prompts():
    result = DbSynthesizer().synthesize(
        request=DbRequest("Summarize the database schema."),
        intent=DbIntent(
            kind=DbIntentKind.SCHEMA_QUERY,
            access=AccessMode.METADATA_READ,
        ),
        contract=DbOperationContract(
            operation_type="schema.query",
            required_evidence=("schema.asset_profile",),
        ),
        evidence=(
            Evidence(
                kind="schema.asset_profile",
                payload={
                    "asset": {"name": "customers"},
                    "fields": [{"name": "email"}],
                    "metadata": {"scope": "asset"},
                },
                metadata={"scope": "asset"},
            ),
            Evidence(
                kind="schema.asset_profile",
                payload={
                    "tables": [
                        {"name": "customers", "columns": [{"name": "id"}]},
                        {"name": "orders", "columns": [{"name": "customer_id"}]},
                    ],
                    "metadata": {"scope": "database"},
                },
                metadata={"scope": "database"},
            ),
        ),
        verification=_verification(),
    )

    assert "Found 2 tables" in result.answer
    assert "customers: id" in result.answer
    assert "orders: customer_id" in result.answer
    assert result.diagnostics["schema_answer_scope"]["mode"] == "database"


def test_llm_schema_synthesis_appends_db_memory_annotation():
    payload = DbAnswerSynthesisPayload(
        answer="The operations table stores execution metadata.",
        reasoning_summary="Used accepted schema evidence.",
        cited_evidence_refs=(
            DbAnswerCitation(
                id="schema-1",
                kind="schema.asset_profile",
                purpose="schema",
            ),
        ),
        assumptions=(),
        limitations=(),
        warnings=(),
        follow_up_questions=(),
        sufficiency="answered",
        confidence=0.88,
        truncation={
            "rows_truncated": False,
            "fields_truncated": False,
            "context_chars_truncated": False,
        },
        grounding={"all_claims_from_evidence": True},
        diagnostics={"mode": "llm"},
    )

    annotated = _apply_schema_db_memory_annotation(
        payload,
        intent=DbIntent(
            kind=DbIntentKind.SCHEMA_QUERY,
            access=AccessMode.METADATA_READ,
        ),
        evidence=(
            Evidence(
                kind="planning.context",
                accepted=True,
                payload={
                    "db_memory_refs": [
                        {
                            "key": "business_rule:operations",
                            "text": "operations are agent runs",
                        }
                    ]
                },
            ),
        ),
    )

    assert annotated.answer == (
        "The operations table stores execution metadata. "
        "Semantic memory note: operations are agent runs"
    )
    assert (
        _apply_schema_db_memory_annotation(
            annotated,
            intent=DbIntent(
                kind=DbIntentKind.SCHEMA_QUERY,
                access=AccessMode.METADATA_READ,
            ),
            evidence=(
                Evidence(
                    kind="planning.context",
                    accepted=True,
                    payload={
                        "db_memory_refs": [
                            {
                                "key": "business_rule:operations",
                                "text": "operations are agent runs",
                            }
                        ]
                    },
                ),
            ),
        ).answer
        == annotated.answer
    )


def test_schema_synthesis_uses_exact_asset_scope_for_table_prompts():
    result = DbSynthesizer().synthesize(
        request=DbRequest("Tell me about the customers table."),
        intent=DbIntent(
            kind=DbIntentKind.SCHEMA_QUERY,
            access=AccessMode.METADATA_READ,
        ),
        contract=DbOperationContract(
            operation_type="schema.query",
            required_evidence=("schema.asset_profile",),
        ),
        evidence=(
            Evidence(
                kind="schema.asset_profile",
                id="db-schema",
                payload={
                    "tables": [
                        {"name": "customers", "columns": [{"name": "id"}]},
                        {"name": "orders", "columns": [{"name": "customer_id"}]},
                    ],
                    "metadata": {"scope": "database"},
                },
                metadata={"scope": "database"},
            ),
            Evidence(
                kind="schema.asset_profile",
                id="customers-profile",
                payload={
                    "asset": {"name": "customers"},
                    "fields": [{"name": "email"}],
                    "metadata": {"scope": "asset"},
                },
                metadata={"scope": "asset"},
            ),
        ),
        verification=_verification(),
    )

    assert "customers: email" in result.answer
    assert "orders" not in result.answer
    assert result.diagnostics["schema_answer_scope"]["mode"] == "asset"
    assert result.diagnostics["schema_answer_scope"]["selected_table_names"] == [
        "customers"
    ]


def test_schema_synthesis_missing_table_returns_closest_matches_not_full_schema():
    result = DbSynthesizer().synthesize(
        request=DbRequest("Tell me about the custmers table."),
        intent=DbIntent(
            kind=DbIntentKind.SCHEMA_QUERY,
            access=AccessMode.METADATA_READ,
        ),
        contract=DbOperationContract(operation_type="schema.query"),
        evidence=(
            Evidence(
                kind="schema.asset_profile",
                id="db-schema",
                payload={
                    "tables": [
                        {"name": "customers", "columns": [{"name": "id"}]},
                        {"name": "orders", "columns": [{"name": "customer_id"}]},
                    ],
                    "metadata": {"scope": "database"},
                },
                metadata={"scope": "database"},
            ),
            Evidence(
                kind="schema.search_result",
                id="search",
                payload={
                    "tables": [
                        {"name": "customers", "fields": [{"name": "email"}]},
                    ]
                },
            ),
        ),
        verification=_verification(),
    )

    assert "Closest matches: customers" in result.answer
    assert "orders: customer_id" not in result.answer
    assert result.diagnostics["schema_answer_scope"]["mode"] == "ambiguous"


def test_relationship_synthesis_uses_relationship_path_evidence():
    result = DbSynthesizer().synthesize(
        request=DbRequest("What relationships do I need to join customers to orders?"),
        intent=DbIntent(
            kind=DbIntentKind.SCHEMA_RELATIONSHIP_QUERY,
            access=AccessMode.METADATA_READ,
        ),
        contract=DbOperationContract(
            operation_type="schema.relationship_query",
            required_evidence=("schema.relationship_path",),
        ),
        evidence=(
            Evidence(
                kind="schema.relationship_path",
                payload={
                    "reachable": True,
                    "paths": [{"assets": ["customers", "orders"]}],
                },
            ),
        ),
        verification=_verification(),
    )

    assert result.answer == "Found relationship path: customers -> orders"
