from daita.db import (
    DbIntent,
    DbIntentKind,
    DbOperationContract,
    DbVerifier,
)
from daita.runtime import AccessMode, Evidence, Task


def _task(task_id: str, capability_id: str, executor_id: str) -> Task:
    return Task(
        id=task_id,
        operation_id="op-1",
        capability_id=capability_id,
        executor_id=executor_id,
    )


def _data_contract() -> DbOperationContract:
    return DbOperationContract(
        operation_type="data.query",
        required_evidence=("sql.validation", "query.result"),
        access=AccessMode.READ,
    )


def test_verifier_accepts_data_query_when_validation_precedes_result():
    result = DbVerifier().verify(
        _data_contract(),
        DbIntent(kind=DbIntentKind.DATA_QUERY, access=AccessMode.READ),
        (
            Evidence(
                kind="sql.validation",
                task_id="task-1",
                payload={"valid": True, "sql": "SELECT 1"},
            ),
            Evidence(
                kind="query.result",
                task_id="task-2",
                payload={"rows": [{"count": 1}], "sql": "SELECT 1"},
            ),
        ),
        (
            _task("task-1", "db.sql.validate", "sqlite.sql.validate"),
            _task("task-2", "db.sql.execute_read", "sqlite.sql.execute_read"),
        ),
    )

    assert result.passed is True
    assert result.missing_evidence == ()
    assert result.diagnostics["row_count"] == 1


def test_verifier_rejects_query_result_without_prior_validation():
    result = DbVerifier().verify(
        _data_contract(),
        DbIntent(kind=DbIntentKind.DATA_QUERY, access=AccessMode.READ),
        (
            Evidence(
                kind="query.result",
                task_id="task-2",
                payload={"rows": [{"count": 1}], "sql": "SELECT 1"},
            ),
        ),
        (_task("task-2", "db.sql.execute_read", "sqlite.sql.execute_read"),),
    )

    assert result.passed is False
    assert "sql.validation" in result.missing_evidence
    assert "sql_validation_missing_before_query_result" in result.warnings


def test_verifier_allows_schema_evidence_without_query_result():
    result = DbVerifier().verify(
        DbOperationContract(
            operation_type="schema.query",
            required_evidence=("schema.asset_profile",),
            access=AccessMode.METADATA_READ,
        ),
        DbIntent(kind=DbIntentKind.SCHEMA_QUERY, access=AccessMode.METADATA_READ),
        (
            Evidence(
                kind="schema.asset_profile",
                payload={"tables": [{"name": "customers"}]},
            ),
        ),
        (),
    )

    assert result.passed is True
    assert result.diagnostics["schema_answer_uses_query_result"] is False
