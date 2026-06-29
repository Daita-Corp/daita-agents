import pytest

from daita.db import (
    DbAnswerCitation,
    DbAnswerSynthesisPayload,
    DbOperationContract,
    DbRequest,
    DbSynthesizer,
    DbVerificationResult,
)
from daita.db.synthesis import _apply_schema_db_memory_annotation
from daita.runtime import AccessMode, Evidence


def _verification(passed: bool = True) -> DbVerificationResult:
    return DbVerificationResult(
        passed=passed,
        missing_evidence=(),
        warnings=(),
        diagnostics={},
        evidence_refs=(
            {
                "id": None,
                "kind": "query.result",
                "owner": None,
                "task_id": "task-2",
            },
        ),
    )


def test_synthesizer_answers_count_from_query_result_evidence_only():
    result = DbSynthesizer().synthesize(
        request=DbRequest("How many orders are there?"),
        contract=DbOperationContract(
            operation_type="data.query",
            required_evidence=("sql.validation", "query.result"),
        ),
        evidence=(
            Evidence(
                kind="query.result",
                task_id="task-2",
                payload={"rows": [{"count": 2}], "sql": "SELECT COUNT(*)"},
            ),
        ),
        verification=_verification(),
    )

    assert result.answer == "The count is 2."
    assert result.evidence_refs[0]["kind"] == "query.result"


def test_synthesizer_refuses_to_answer_before_verification_passes():
    with pytest.raises(ValueError, match="verification passes"):
        DbSynthesizer().synthesize(
            request=DbRequest("How many orders?"),
            contract=DbOperationContract(operation_type="data.query"),
            evidence=(),
            verification=_verification(passed=False),
        )


def test_schema_synthesis_uses_database_scope_for_database_wide_prompts():
    result = DbSynthesizer().synthesize(
        request=DbRequest("Summarize the database schema."),
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


def test_schema_memory_annotation_prefers_answer_memory_context():
    payload = DbAnswerSynthesisPayload(
        answer="The operations table stores execution metadata.",
        reasoning_summary="Used accepted schema evidence.",
        cited_evidence_refs=(),
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
        evidence=(
            Evidence(
                kind="planning.context",
                accepted=True,
                payload={
                    "db_memory_refs": [
                        {
                            "key": "business_rule:operations",
                            "text": "planner-only memory should not be first",
                        }
                    ]
                },
            ),
            Evidence(
                kind="answer.memory.context",
                accepted=True,
                payload={
                    "refs": [
                        {
                            "key": "business_rule:operations",
                            "text": "answer memory is first",
                        }
                    ]
                },
            ),
        ),
    )

    assert annotated.answer == (
        "The operations table stores execution metadata. "
        "Semantic memory note: answer memory is first"
    )


def test_memory_recall_synthesizes_from_answer_memory_without_planner_refs():
    result = DbSynthesizer().synthesize(
        request=DbRequest("What do you remember about operations?"),
        contract=DbOperationContract(
            operation_type="memory.recall",
            required_evidence=("answer.memory.context",),
            access=AccessMode.METADATA_READ,
        ),
        evidence=(
            Evidence(
                kind="answer.memory.context",
                accepted=True,
                payload={
                    "refs": [
                        {
                            "kind": "business_rule",
                            "key": "operations",
                            "text": "operations are agent runs",
                        }
                    ],
                    "diagnostics": {"included_count": 1},
                },
            ),
        ),
        verification=_verification(),
    )

    assert "operations are agent runs" in result.answer
    assert "did not find" not in result.answer


def test_schema_synthesis_uses_exact_asset_scope_for_table_prompts():
    result = DbSynthesizer().synthesize(
        request=DbRequest("Tell me about the customers table."),
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
        contract=DbOperationContract(
            operation_type="schema.relationships",
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
