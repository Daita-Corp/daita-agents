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
