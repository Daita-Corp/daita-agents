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
