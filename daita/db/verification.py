"""
Evidence verification for DB runtime operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from daita.runtime import Evidence, Task

from .models import DbIntent, DbIntentKind, DbOperationContract
from .sql_evidence import blocked_scope_resources, sql_validation_facts_from_evidence

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
class DbSlimReadiness:
    """Deterministic final-text readiness for the SQLite slim read slice."""

    ready: bool
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    query_result: Evidence | None = None
    validation: Evidence | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "query_result_ref": _evidence_ref(self.query_result),
            "validation_ref": _evidence_ref(self.validation),
            "truncated": bool(
                self.query_result is not None
                and self.query_result.payload.get("truncated") is True
            ),
        }


def db_sqlite_slim_readiness_check(
    *,
    operation: Any,
    evidence: tuple[Evidence, ...],
    tasks: tuple[Task, ...],
    answer: str | None = None,
    expected_owner: str = "sqlite",
) -> DbSlimReadiness:
    """Require an applicable accepted read result before accepting DB claims."""

    operation_id = str(getattr(operation, "id", "") or "")
    request = getattr(operation, "request", None)
    request = request if isinstance(request, Mapping) else {}
    metadata = getattr(operation, "metadata", None)
    metadata = metadata if isinstance(metadata, Mapping) else {}
    raw_scope = request.get("source_scope") or metadata.get("source_scope") or ()
    source_scope = (
        (str(raw_scope),)
        if isinstance(raw_scope, str)
        else tuple(str(item) for item in raw_scope)
    )
    task_order = {task.id: index for index, task in enumerate(tasks)}
    tasks_by_id = {task.id: task for task in tasks}
    accepted_validations = {
        item.task_id: item
        for item in evidence
        if item.accepted
        and item.kind == "sql.validation"
        and item.operation_id == operation_id
        and item.owner == expected_owner
        and item.task_id
    }
    applicable: list[tuple[int, Evidence, Evidence]] = []
    rejection_reasons: set[str] = set()
    for result in evidence:
        if result.kind != "query.result" or not result.accepted:
            continue
        if result.operation_id != operation_id:
            rejection_reasons.add("query_result_wrong_operation")
            continue
        if result.owner != expected_owner:
            rejection_reasons.add("query_result_wrong_source")
            continue
        task = tasks_by_id.get(str(result.task_id or ""))
        if task is None or task.capability_id != "db.sql.execute_read":
            rejection_reasons.add("query_result_missing_read_task")
            continue
        if str(task.metadata.get("owner") or "") != expected_owner:
            rejection_reasons.add("query_task_wrong_source")
            continue
        dependency = next(
            (
                item
                for item in task.dependencies
                if item.kind_value == "evidence"
                and item.evidence_kind == "sql.validation"
                and item.operation_id == operation_id
            ),
            None,
        )
        validation = (
            accepted_validations.get(dependency.producer_task_id)
            if dependency is not None
            else None
        )
        if validation is None:
            rejection_reasons.add("applicable_sql_validation_missing")
            continue
        facts = sql_validation_facts_from_evidence(validation)
        if facts.valid is not True or facts.is_read is not True:
            rejection_reasons.add("applicable_sql_validation_not_read")
            continue
        if blocked_scope_resources(facts.target_resources, source_scope):
            rejection_reasons.add("query_result_outside_source_scope")
            continue
        if task_order.get(str(validation.task_id), -1) >= task_order.get(task.id, -1):
            rejection_reasons.add("sql_validation_did_not_precede_query_result")
            continue
        if not isinstance(result.payload.get("rows"), list):
            rejection_reasons.add("query_result_rows_not_list")
            continue
        applicable.append((task_order.get(task.id, -1), result, validation))

    if not applicable:
        reasons = tuple(sorted(rejection_reasons)) or (
            "accepted_current_query_result_required",
        )
        return DbSlimReadiness(False, reasons, ())

    _order, query_result, validation = max(applicable, key=lambda item: item[0])
    warnings: list[str] = []
    reasons: list[str] = []
    if query_result.payload.get("truncated") is True:
        warnings.append("query_result_truncated")
        normalized_answer = str(answer or "").lower()
        if answer is not None and not any(
            marker in normalized_answer
            for marker in ("truncat", "limited", "first ", "partial")
        ):
            reasons.append("truncation_disclosure_required")
    return DbSlimReadiness(
        ready=not reasons,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
        query_result=query_result,
        validation=validation,
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
        elif intent.kind is DbIntentKind.WRITE_EXECUTE:
            warnings.extend(_verify_write_execute(evidence, tasks))
        elif intent.kind is DbIntentKind.WRITE_PROPOSE:
            warnings.extend(_verify_write_proposal(evidence))
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
    latest_plan = next(
        (
            item
            for item in reversed(evidence)
            if item.accepted
            and item.kind == "query.plan.proposal"
            and item.payload.get("valid") is True
            and isinstance(item.payload.get("sql"), str)
            and bool(str(item.payload.get("sql") or "").strip())
        ),
        None,
    )
    accepted_results = tuple(
        item for item in evidence if item.accepted and item.kind == "query.result"
    )
    applicable_results = accepted_results
    if latest_plan is not None:
        tasks_by_id = {task.id: task for task in tasks}
        matched_results: list[Evidence] = []
        for result in accepted_results:
            task = tasks_by_id.get(str(result.task_id or ""))
            task_plan_id = None
            if task is not None:
                task_plan_id = task.input.get("plan_evidence_id")
                if not task_plan_id:
                    provenance = task.metadata.get("sql_provenance")
                    if isinstance(provenance, Mapping):
                        task_plan_id = provenance.get("source_evidence_id")
            result_plan_id = (
                result.payload.get("plan_evidence_id")
                or result.metadata.get("plan_evidence_id")
                or task_plan_id
            )
            if str(result_plan_id or "") == latest_plan.id:
                matched_results.append(result)
        applicable_results = tuple(matched_results[-1:])
    elif accepted_results:
        applicable_results = accepted_results[-1:]

    verification_evidence = tuple(
        item
        for item in evidence
        if item.kind != "query.result" or item in applicable_results
    )
    verification = verifier.verify(contract, intent, verification_evidence, tasks)
    supporting_evidence = db_accepted_synthesis_support_evidence(verification_evidence)
    query_result_required = db_operation_requires_query_result(operation, intent)
    query_result_present = bool(applicable_results)
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
        (
            item
            for item in reversed(evidence)
            if item.accepted and item.kind == "sql.validation"
        ),
        None,
    )
    query_result = next(
        (
            item
            for item in reversed(evidence)
            if item.accepted and item.kind == "query.result"
        ),
        None,
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
        if not _validation_precedes_evidence(validation, query_result, tasks):
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
    if proposal is not None and proposal.accepted:
        definition = next(
            (item for item in evidence if item.kind == "db.memory.definition"),
            None,
        )
        memory_write = next(
            (item for item in evidence if item.kind == "memory.semantic.write"),
            None,
        )
        if (
            definition is None
            or not definition.accepted
            or memory_write is None
            or not memory_write.accepted
        ):
            return ("memory_update_not_committed",)
    memory_write = next(
        (item for item in evidence if item.kind == "memory.semantic.write"), None
    )
    if memory_write is None:
        return ()
    if memory_write.payload.get("success") is False:
        return ("memory_write_not_successful",)
    return ()


def _verify_write_proposal(evidence: tuple[Evidence, ...]) -> tuple[str, ...]:
    validation = next(
        (item for item in evidence if item.kind == "sql.validation" and item.accepted),
        None,
    )
    if validation is None:
        return ("sql_validation_missing_for_write_proposal",)
    if validation.payload.get("valid") is not True:
        return ("sql_validation_not_valid",)
    return ()


def _verify_write_execute(
    evidence: tuple[Evidence, ...],
    tasks: tuple[Task, ...],
) -> tuple[str, ...]:
    warnings = list(_verify_write_proposal(evidence))
    validation = next(
        (item for item in evidence if item.kind == "sql.validation" and item.accepted),
        None,
    )
    write_execution = next(
        (item for item in evidence if item.kind == "write.execution" and item.accepted),
        None,
    )
    if write_execution is None:
        warnings.append("write_execution_missing")
    elif validation is not None and not _validation_precedes_evidence(
        validation,
        write_execution,
        tasks,
    ):
        warnings.append("sql_validation_did_not_precede_write_execution")
    return tuple(warnings)


def _validation_precedes_evidence(
    validation: Evidence, evidence: Evidence, tasks: tuple[Task, ...]
) -> bool:
    order = {task.id: index for index, task in enumerate(tasks)}
    validation_index = order.get(str(validation.task_id))
    evidence_index = order.get(str(evidence.task_id))
    if validation_index is None or evidence_index is None:
        return False
    return validation_index < evidence_index


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


def _evidence_ref(evidence: Evidence | None) -> dict[str, str | None] | None:
    if evidence is None:
        return None
    return {
        "id": evidence.id,
        "kind": evidence.kind,
        "owner": evidence.owner,
        "task_id": evidence.task_id,
    }
