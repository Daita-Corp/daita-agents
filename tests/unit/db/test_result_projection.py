from __future__ import annotations

from dataclasses import asdict
import hashlib
import hmac
import json

import pytest

from daita.db.fingerprints import sensitive_fingerprint, stable_fingerprint
from daita.db.models import (
    DbIntent,
    DbIntentKind,
    DbOperationContract,
    DbOperationResult,
    DbRequest,
)
from daita.db.runtime import DbRuntime
from daita.db.runtime.results import _audit_entry_from_result
from daita.runtime import AccessMode, Evidence, OperationStatus, Task, TaskStatus

_AUDIT_KEY = b"audit-hmac-key-canary-0123456789abcdef"
_OTHER_AUDIT_KEY = b"other-hmac-key-canary-0123456789abcdef"
_EMAIL = "audit-email-canary@example.invalid"
_ACCOUNT_ID = "customer-acct-audit-canary-741"
_SQL_LITERAL = "quoted-sql-literal-audit-canary"
_BOUND_PARAMETER = "bound-parameter-audit-canary"
_QUERY_ROW_VALUE = "query-row-value-audit-canary"
_TASK_INPUT = "task-input-audit-canary"
_TASK_METADATA = "task-metadata-audit-canary"
_PLANNER_ACTION_INPUT = "planner-action-input-audit-canary"
_VALIDATION_ERROR = "Validation failed for validation-error-audit-canary!"
_MEMORY_TEXT = "Memory text for memory-audit-canary"
_RAW_EXCEPTION_MESSAGE = "Raw exception message for exception-audit-canary"
_RAW_CANARIES = (
    _EMAIL,
    _ACCOUNT_ID,
    _SQL_LITERAL,
    _BOUND_PARAMETER,
    _QUERY_ROW_VALUE,
    _TASK_INPUT,
    _TASK_METADATA,
    _PLANNER_ACTION_INPUT,
    _VALIDATION_ERROR,
    _MEMORY_TEXT,
    _RAW_EXCEPTION_MESSAGE,
)


def _raw_evidence(operation_id: str, task_id: str) -> tuple[Evidence, ...]:
    sql = f"SELECT email FROM customers WHERE account_id = '{_SQL_LITERAL}'"
    return (
        Evidence(
            id=f"{operation_id}.query",
            kind="query.result",
            owner="audit_test",
            operation_id=operation_id,
            task_id=task_id,
            payload={
                "sql": sql,
                "parameters": {"account_id": _BOUND_PARAMETER},
                "rows": [{"email": _QUERY_ROW_VALUE}],
                "row_count": 999,
                "total_rows": 12,
                "truncated": True,
                "success": False,
                "error": _RAW_EXCEPTION_MESSAGE,
                "error_type": "DatabaseError",
            },
        ),
        Evidence(
            id=f"{operation_id}.validation",
            kind="sql.validation",
            owner="audit_test",
            operation_id=operation_id,
            task_id=task_id,
            payload={
                "sql": sql,
                "valid": False,
                "statement_facts": {
                    "statement_type": "select",
                    "target_resources": [f"customers_{_ACCOUNT_ID}"],
                },
                "referenced_tables": [f"customers_{_ACCOUNT_ID}"],
                "referenced_columns": [f"customers.{_EMAIL}"],
                "validation_errors": [_VALIDATION_ERROR],
            },
        ),
        Evidence(
            id=f"{operation_id}.planner",
            kind="planner.decision",
            owner="db_runtime",
            operation_id=operation_id,
            payload={
                "actions": [{"input": {"value": _PLANNER_ACTION_INPUT}}],
                "task_plan": {
                    "tasks": [
                        {
                            "input": {"value": _TASK_INPUT},
                            "metadata": {"value": _TASK_METADATA},
                        }
                    ]
                },
            },
        ),
        Evidence(
            id=f"{operation_id}.memory",
            kind="planning.context",
            owner="db_runtime",
            operation_id=operation_id,
            payload={"memory_text": _MEMORY_TEXT},
        ),
    )


def _raw_result(
    operation_id: str = "operation-audit-v2",
    task_id: str = "task-audit-v2",
) -> DbOperationResult:
    return DbOperationResult(
        operation_id=operation_id,
        request=DbRequest(
            f"Find {_EMAIL} for customer {_ACCOUNT_ID}.",
            metadata={"account_id": _ACCOUNT_ID},
        ),
        intent=DbIntent(kind=DbIntentKind.DATA_QUERY, access=AccessMode.READ),
        contract=DbOperationContract(
            operation_type="db.run",
            access=AccessMode.READ,
        ),
        status=OperationStatus.FAILED,
        evidence=_raw_evidence(operation_id, task_id),
        warnings=(
            "db_runtime_known_warning",
            f"Warning details: {_RAW_EXCEPTION_MESSAGE}",
        ),
        diagnostics={
            "telemetry": {
                "provider": "audit-test-provider",
                "model": "audit-test-model",
                "input_tokens": 5,
                "output_tokens": 3,
                "total_tokens": 8,
                "llm_calls": 1,
                "estimated_cost_usd": 0.01,
                "latency_ms": 2.5,
                "mode": "llm",
            },
            "planner": {
                "actions": [{"input": {"value": _PLANNER_ACTION_INPUT}}],
                "task_plan": {"tasks": [{"input": {"value": _TASK_INPUT}}]},
            },
            "error": {
                "type": "RuntimeError",
                "message": _RAW_EXCEPTION_MESSAGE,
            },
        },
    )


def _serialized(value) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def _mapping_keys(value) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        keys.update(str(key) for key in value)
        for item in value.values():
            keys.update(_mapping_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.update(_mapping_keys(item))
    return keys


def _sql_fingerprint(entry: dict) -> str:
    return next(
        summary["sql_fingerprint"]
        for summary in entry["evidence_summaries"]
        if summary["kind"] == "query.result"
    )


def test_canonical_fingerprints_are_deterministic_and_keyed():
    first = {"z": [3, 2, 1], "a": {"value": 7}}
    second = {"a": {"value": 7}, "z": [3, 2, 1]}
    canonical = json.dumps(
        first,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")

    assert stable_fingerprint(first) == stable_fingerprint(second)
    assert stable_fingerprint(first) == hashlib.sha256(canonical).hexdigest()
    assert (
        sensitive_fingerprint(first, _AUDIT_KEY)
        == hmac.new(
            _AUDIT_KEY,
            canonical,
            hashlib.sha256,
        ).hexdigest()
    )
    assert sensitive_fingerprint(first, _AUDIT_KEY) != sensitive_fingerprint(
        first,
        _OTHER_AUDIT_KEY,
    )


@pytest.mark.parametrize(
    ("key", "exception"),
    [
        ("not-bytes", TypeError),
        (bytearray(32), TypeError),
        (b"too-short", ValueError),
    ],
)
def test_sensitive_fingerprint_rejects_invalid_keys(key, exception):
    with pytest.raises(exception):
        sensitive_fingerprint("sensitive", key)


def test_audit_schema_v2_is_safe_without_relying_on_public_projection():
    raw_result = _raw_result()
    entry = _audit_entry_from_result(
        raw_result,
        raw_prompt=raw_result.request.prompt,
        raw_evidence=raw_result.evidence,
        raw_warnings=raw_result.warnings,
        raw_error=raw_result.diagnostics["error"],
        audit_key=_AUDIT_KEY,
    )

    dumped = _serialized(entry)
    assert entry["schema_version"] == 2
    assert all(canary not in dumped for canary in _RAW_CANARIES)
    assert not {
        "prompt",
        "sql",
        "parameters",
        "params",
        "rows",
        "error",
        "message",
        "preview",
        "prompt_preview",
        "sql_preview",
        "error_message",
        "task_input",
        "task_metadata",
        "actions",
        "task_plan",
        "metadata",
    } & _mapping_keys(entry)
    assert set(entry) == {
        "schema_version",
        "timestamp",
        "operation_id",
        "status",
        "intent_kind",
        "operation_type",
        "prompt_fingerprint",
        "prompt_length",
        "warning_codes",
        "evidence_count",
        "accepted_evidence_count",
        "evidence_refs",
        "evidence_summaries",
        "telemetry",
        "error_type",
        "error_code",
    }
    assert entry["prompt_length"] == len(raw_result.request.prompt)
    assert entry["warning_codes"] == [
        "db_runtime_known_warning",
        "db_runtime_warning_redacted",
    ]
    assert entry["error_type"] == "RuntimeError"
    assert entry["error_code"] == "db_runtime_error_redacted"
    assert entry["evidence_count"] == 4
    assert entry["accepted_evidence_count"] == 4

    query_summary = next(
        item for item in entry["evidence_summaries"] if item["kind"] == "query.result"
    )
    assert query_summary["row_count"] == 1
    assert query_summary["total_rows"] == 12
    assert query_summary["truncated"] is True
    assert query_summary["success"] is False
    assert query_summary["error_type"] == "DatabaseError"
    assert query_summary["error_code"] == "db_runtime_error_redacted"
    assert len(query_summary["sql_fingerprint"]) == 64

    validation_summary = next(
        item for item in entry["evidence_summaries"] if item["kind"] == "sql.validation"
    )
    assert validation_summary["statement_type"] == "select"
    assert validation_summary["referenced_table_count"] == 1
    assert validation_summary["referenced_column_count"] == 1
    assert validation_summary["sql_fingerprint"] == query_summary["sql_fingerprint"]


async def test_prompt_and_sql_fingerprints_correlate_only_with_same_runtime_key():
    same_key_runtimes = (
        DbRuntime(host_services={"audit_fingerprint_key": _AUDIT_KEY}),
        DbRuntime(host_services={"audit_fingerprint_key": _AUDIT_KEY}),
    )
    different_key_runtime = DbRuntime(
        host_services={"audit_fingerprint_key": _OTHER_AUDIT_KEY}
    )
    for runtime in (*same_key_runtimes, different_key_runtime):
        await runtime._record_operation_result(_raw_result())

    first = same_key_runtimes[0].audit_log[0]
    second = same_key_runtimes[1].audit_log[0]
    different = different_key_runtime.audit_log[0]
    assert first["prompt_fingerprint"] == second["prompt_fingerprint"]
    assert _sql_fingerprint(first) == _sql_fingerprint(second)
    assert first["prompt_fingerprint"] != different["prompt_fingerprint"]
    assert _sql_fingerprint(first) != _sql_fingerprint(different)
    assert all(entry["schema_version"] == 2 for entry in (first, second, different))


@pytest.mark.parametrize(
    "key",
    [None, "not-bytes", bytearray(32), b"too-short"],
)
def test_runtime_rejects_invalid_supplied_audit_keys(key):
    with pytest.raises(
        ValueError,
        match=r"audit_fingerprint_key.*bytes.*at least 32",
    ):
        DbRuntime(host_services={"audit_fingerprint_key": key})


def test_runtime_without_supplied_key_gets_distinct_private_random_key():
    first = DbRuntime()
    second = DbRuntime()

    assert isinstance(first._audit_fingerprint_key, bytes)
    assert len(first._audit_fingerprint_key) == 32
    assert first._audit_fingerprint_key != second._audit_fingerprint_key
    assert "audit_fingerprint_key" not in first.host_services
    assert "audit_fingerprint_key" not in second.host_services


async def test_audit_key_is_private_while_raw_inspection_and_events_stay_intact():
    runtime = DbRuntime(host_services={"audit_fingerprint_key": _AUDIT_KEY})
    await runtime.setup()
    operation = await runtime.kernel.create_operation(
        operation_type="db.run",
        request={"prompt": f"Find {_EMAIL} for {_ACCOUNT_ID}."},
        metadata={"safety_frame": {}},
        evaluate_governance=False,
    )
    task = Task(
        id=f"{operation.id}.task",
        operation_id=operation.id,
        capability_id="db.sql.execute_read",
        executor_id="audit_test.sql.execute_read",
        input={
            "sql": f"SELECT '{_SQL_LITERAL}'",
            "parameters": [_BOUND_PARAMETER],
            "task_input": _TASK_INPUT,
        },
        status=TaskStatus.SUCCEEDED,
        metadata={"task_metadata": _TASK_METADATA},
    )
    evidence = _raw_evidence(operation.id, task.id)
    await runtime.store.save_task(task)
    for item in evidence:
        await runtime.store.save_evidence(item)
    raw_result = _raw_result(operation.id, task.id)

    public_result = await runtime._record_operation_result(
        raw_result,
        operation=operation,
    )
    snapshot = await runtime.inspect_operation(operation.id)
    runtime_inspection = await runtime.inspect()

    assert snapshot is not None
    audit_dumped = _serialized(runtime.audit_log)
    assert all(canary not in audit_dumped for canary in _RAW_CANARIES)
    assert runtime.audit_log[0]["schema_version"] == 2
    assert runtime.audit_log[0]["warning_codes"] == [
        "db_runtime_known_warning",
        "db_runtime_warning_redacted",
    ]

    projection_canaries = _RAW_CANARIES[2:]
    public_dumped = _serialized(asdict(public_result))
    assert all(canary not in public_dumped for canary in projection_canaries)
    assert public_result is runtime.operation_results[-1]
    assert all(
        item.metadata["projection_mode"] == "public_result"
        for item in public_result.evidence
    )

    snapshot_dumped = _serialized(
        {
            "operation": snapshot.operation.to_dict(),
            "tasks": [item.to_dict() for item in snapshot.tasks],
            "evidence": [item.to_dict() for item in snapshot.evidence],
            "events": [item.to_dict() for item in snapshot.events],
        }
    )
    assert all(canary in snapshot_dumped for canary in _RAW_CANARIES)
    assert snapshot.tasks[0].input["task_input"] == _TASK_INPUT
    assert snapshot.tasks[0].metadata["task_metadata"] == _TASK_METADATA
    assert snapshot.events[-1].payload["warnings"][-1].endswith(_RAW_EXCEPTION_MESSAGE)

    key_text = _AUDIT_KEY.decode("ascii")
    surfaces = (
        audit_dumped,
        public_dumped,
        snapshot_dumped,
        _serialized(runtime_inspection.to_dict()),
        _serialized(runtime.config.metadata),
        _serialized(runtime.host_services),
    )
    assert all(key_text not in surface for surface in surfaces)
    assert "audit_fingerprint_key" not in runtime.host_services
    assert "audit_fingerprint_key" not in runtime.setup_context.services.as_dict()
