"""
Evidence verification for DB runtime operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from daita.runtime import Evidence, Task

from .models import DbIntent, DbIntentKind, DbOperationContract

DB_FINALIZATION_CONTROL_EVIDENCE_KINDS = frozenset(
    {
        "planner.decision",
        "planner.compilation",
        "planner.observation",
        "verification.result",
        "answer.synthesis",
    }
)


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


@dataclass(frozen=True)
class DbFinalizationCheck:
    """Final DB run readiness using shared verification and support evidence."""

    finalizable: bool
    verification: DbVerificationResult
    query_result_required: bool
    query_result_present: bool
    supporting_evidence: tuple[Evidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "finalizable": self.finalizable,
            "verification": self.verification.to_dict(),
            "query_result_required": self.query_result_required,
            "query_result_present": self.query_result_present,
            "synthesis_supporting_evidence": tuple(
                _evidence_refs(self.supporting_evidence)
            ),
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
            "skill_verifier_metadata": contract.metadata.get(
                "skill_verifier_metadata", {}
            ),
        }

        if intent.kind in {
            DbIntentKind.DATA_QUERY,
            DbIntentKind.CATALOG_ASSISTED_DATA_QUERY,
        }:
            warnings.extend(_verify_data_query(evidence, tasks, diagnostics))
        elif intent.kind in {
            DbIntentKind.SCHEMA_QUERY,
            DbIntentKind.SCHEMA_RELATIONSHIP_QUERY,
        }:
            diagnostics["schema_answer_uses_query_result"] = "query.result" in kinds
            if "query.result" in kinds:
                warnings.append("metadata_operation_includes_query_result")
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


def db_accepted_synthesis_support_evidence(
    evidence: tuple[Evidence, ...],
) -> tuple[Evidence, ...]:
    """Return accepted evidence that can support final answer synthesis."""
    return tuple(
        item
        for item in evidence
        if item.accepted and item.kind not in DB_FINALIZATION_CONTROL_EVIDENCE_KINDS
    )


def db_operation_requires_query_result(operation: Any, intent: DbIntent) -> bool:
    """Return whether a DB operation needs data evidence before finalization."""
    mode = str(getattr(operation, "request", {}).get("mode") or "").lower()
    operation_type = str(getattr(operation, "operation_type", "") or "").lower()
    return (
        intent.kind
        in {
            DbIntentKind.DATA_QUERY,
            DbIntentKind.CATALOG_ASSISTED_DATA_QUERY,
            DbIntentKind.METRIC_QUERY,
        }
        or mode in {"data", "data.query", "query", "read"}
        or operation_type
        in {"data.query", "data.query.catalog_assisted", "metric.query"}
    )


def db_run_finalization_check(
    *,
    operation: Any,
    verifier: DbVerifier,
    contract: DbOperationContract,
    intent: DbIntent,
    evidence: tuple[Evidence, ...],
    tasks: tuple[Task, ...],
) -> DbFinalizationCheck:
    """Check whether accepted evidence is sufficient to finalize a DB run."""
    verification = verifier.verify(contract, intent, evidence, tasks)
    supporting_evidence = db_accepted_synthesis_support_evidence(evidence)
    query_result_required = db_operation_requires_query_result(operation, intent)
    query_result_present = any(
        item.accepted and item.kind == "query.result" for item in evidence
    )
    finalizable = (
        verification.passed
        and bool(supporting_evidence)
        and (not query_result_required or query_result_present)
    )
    return DbFinalizationCheck(
        finalizable=finalizable,
        verification=verification,
        query_result_required=query_result_required,
        query_result_present=query_result_present,
        supporting_evidence=supporting_evidence,
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
    proposal = next(
        (item for item in evidence if item.kind == "db.memory.proposal"), None
    )
    if proposal is not None and not proposal.accepted:
        reasons = (
            proposal.payload.get("validation", {}).get("reasons", [])
            if isinstance(proposal.payload.get("validation"), dict)
            else []
        )
        return tuple(["memory_proposal_not_accepted", *[str(item) for item in reasons]])
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
