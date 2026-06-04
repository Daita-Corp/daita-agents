"""
Evidence verification for DB runtime operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from daita.runtime import Evidence, Task

from .models import DbIntent, DbIntentKind, DbOperationContract


@dataclass(frozen=True)
class DbVerificationResult:
    """Outcome of verifying accepted evidence against a DB contract."""

    passed: bool
    missing_evidence: tuple[str, ...]
    warnings: tuple[str, ...]
    diagnostics: dict[str, Any]
    evidence_refs: tuple[dict[str, str | None], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "missing_evidence": list(self.missing_evidence),
            "warnings": list(self.warnings),
            "diagnostics": self.diagnostics,
            "evidence_refs": list(self.evidence_refs),
        }


class DbVerifier:
    """Verify that evidence satisfies the operation contract."""

    def verify(
        self,
        contract: DbOperationContract,
        intent: DbIntent,
        evidence: tuple[Evidence, ...],
        tasks: tuple[Task, ...],
    ) -> DbVerificationResult:
        """Return whether accepted evidence is sufficient for final synthesis."""
        kinds = {item.kind for item in evidence if item.accepted}
        missing = tuple(
            kind for kind in contract.required_evidence if kind not in kinds
        )
        warnings: list[str] = []
        diagnostics: dict[str, Any] = {
            "required_evidence": list(contract.required_evidence),
            "accepted_evidence_kinds": [item.kind for item in evidence],
        }

        if intent.kind in {
            DbIntentKind.DATA_QUERY,
            DbIntentKind.CATALOG_ASSISTED_DATA_QUERY,
        }:
            warnings.extend(_verify_data_query(evidence, tasks, diagnostics))
        elif intent.kind is DbIntentKind.SCHEMA_QUERY:
            diagnostics["schema_answer_uses_query_result"] = "query.result" in kinds
            if "query.result" in kinds:
                warnings.append("schema_answer_includes_unneeded_query_result")
        elif intent.kind is DbIntentKind.MEMORY_UPDATE:
            warnings.extend(_verify_memory_update(evidence))

        passed = not missing and not warnings
        return DbVerificationResult(
            passed=passed,
            missing_evidence=missing,
            warnings=tuple(warnings),
            diagnostics=diagnostics,
            evidence_refs=_evidence_refs(evidence),
        )


def _verify_data_query(
    evidence: tuple[Evidence, ...],
    tasks: tuple[Task, ...],
    diagnostics: dict[str, Any],
) -> tuple[str, ...]:
    warnings: list[str] = []
    validation = next(
        (item for item in evidence if item.kind == "sql.validation"), None
    )
    query_result = next(
        (item for item in evidence if item.kind == "query.result"), None
    )

    if validation is None:
        warnings.append("sql_validation_missing_before_query_result")
    elif validation.payload.get("valid") is not True:
        warnings.append("sql_validation_not_valid")

    if query_result is None:
        warnings.append("query_result_missing")
    else:
        rows = query_result.payload.get("rows")
        if not isinstance(rows, list):
            warnings.append("query_result_rows_not_list")
        else:
            diagnostics["row_count"] = len(rows)
            diagnostics["empty_result"] = len(rows) == 0
        diagnostics["query_result_truncated"] = bool(
            query_result.payload.get("truncated")
        )
        if "sql" not in query_result.payload:
            warnings.append("query_result_sql_missing")

    if validation is not None and query_result is not None:
        if not _validation_precedes_query_result(validation, query_result, tasks):
            warnings.append("sql_validation_did_not_precede_query_result")

    return tuple(warnings)


def _verify_memory_update(evidence: tuple[Evidence, ...]) -> tuple[str, ...]:
    memory_write = next(
        (item for item in evidence if item.kind == "memory.semantic.write"), None
    )
    if memory_write is None:
        return ()
    if memory_write.payload.get("success") is False:
        return ("memory_write_not_successful",)
    return ()


def _validation_precedes_query_result(
    validation: Evidence, query_result: Evidence, tasks: tuple[Task, ...]
) -> bool:
    order = {task.id: index for index, task in enumerate(tasks)}
    validation_index = order.get(str(validation.task_id))
    query_index = order.get(str(query_result.task_id))
    if validation_index is None or query_index is None:
        return False
    return validation_index < query_index


def _evidence_refs(evidence: tuple[Evidence, ...]) -> tuple[dict[str, str | None], ...]:
    return tuple(
        {
            "id": item.id,
            "kind": item.kind,
            "owner": item.owner,
            "task_id": item.task_id,
        }
        for item in evidence
    )
