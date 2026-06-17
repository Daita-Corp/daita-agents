import pytest

from daita.db import (
    DbIntent,
    DbIntentKind,
    DbOperationContract,
    DbRequest,
    DbSynthesizer,
    DbVerificationResult,
)
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
        intent=DbIntent(kind=DbIntentKind.DATA_QUERY, access=AccessMode.READ),
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
            intent=DbIntent(kind=DbIntentKind.DATA_QUERY, access=AccessMode.READ),
            contract=DbOperationContract(operation_type="data.query"),
            evidence=(),
            verification=_verification(passed=False),
        )


def test_schema_synthesis_prefers_database_scoped_profile_over_asset_profile():
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
    assert "orders: customer_id" in result.answer


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
