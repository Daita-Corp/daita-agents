"""Query repair and zero-row diagnosis helpers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from daita.runtime import Evidence, Operation, Task, TaskDependency

from ..capabilities import (
    PLANNING_CONTEXT_EVIDENCE,
    QUERY_PLAN_PROPOSAL_EVIDENCE,
    QUERY_PLAN_REPAIR_EVIDENCE,
    QUERY_PLAN_VALIDATION_EVIDENCE,
    QUERY_ZERO_ROW_DIAGNOSIS_EVIDENCE,
)
from ..evidence import DbEvidenceStore
from ..models import DbOperationContract, DbRequest
from .helpers import _short_table_name
from .planning import _accepted_sql


class _ExecutionRepairMixin:
    async def _repair_zero_row_result_if_available(
        self,
        request: DbRequest,
        contract: DbOperationContract,
        operation: Operation,
        schema: dict[str, Any],
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        *,
        planning_context: Evidence,
        plan_evidence: Evidence,
        sql_validation: Evidence,
        read_evidence: tuple[Evidence, ...],
    ) -> str | None:
        if not self.runtime.db_llm_service.available:
            return None
        if any(
            item.kind == QUERY_PLAN_REPAIR_EVIDENCE for item in evidence_store.list()
        ):
            return None
        query_result = _query_result_evidence(read_evidence)
        if query_result is None or not _query_result_is_zero_rows(query_result):
            return None
        sql = str(sql_validation.payload.get("sql") or "")
        if not sql or sql_validation.payload.get("is_read") is False:
            return None

        try:
            from ..sql_analysis import analyze_sql

            analysis = analyze_sql(
                sql,
                dialect=str(
                    planning_context.payload.get("dialect")
                    or schema.get("database_type")
                    or ""
                ),
            )
        except Exception:
            return None
        predicates = _zero_row_literal_predicates(analysis.literal_predicates)
        if not predicates:
            return None
        hints = _column_value_hints_for_predicates(
            planning_context.payload,
            predicates,
        )
        if not hints:
            return None

        diagnosis = await self._persist_runtime_evidence(
            operation,
            Evidence(
                kind=QUERY_ZERO_ROW_DIAGNOSIS_EVIDENCE,
                owner="db_runtime",
                operation_id=operation.id,
                accepted=False,
                payload={
                    "valid": False,
                    "failure": "zero_row_result",
                    "prompt": request.prompt,
                    "sql": sql,
                    "sql_validation_evidence_id": sql_validation.id,
                    "query_result_evidence_id": query_result.id,
                    "prior_plan_evidence_id": plan_evidence.id,
                    "predicates": predicates,
                    "column_value_hints": hints,
                    "repair_policy": {
                        "max_attempts": 1,
                        "only_read_queries": True,
                        "only_literal_predicates": True,
                    },
                },
            ),
        )
        repaired = await self._repair_query_plan(
            operation,
            tasks,
            evidence_store,
            planning_context=planning_context,
            prior_plan=plan_evidence,
            failure=diagnosis,
            repair_attempt=1,
        )
        if repaired is None:
            return None
        validation = await self._validate_query_plan(
            operation,
            tasks,
            evidence_store,
            plan_evidence=repaired,
            planning_context=planning_context,
            analysis_metadata={"zero_row_repair": True},
            extra_dependencies=(
                TaskDependency(
                    kind="evidence",
                    evidence_kind=QUERY_ZERO_ROW_DIAGNOSIS_EVIDENCE,
                    evidence_id=diagnosis.id,
                    evidence_accepted=diagnosis.accepted,
                    operation_id=operation.id,
                ),
            ),
        )
        repaired_sql = validation.payload.get("accepted_sql")
        if not validation.payload.get("valid") or not repaired_sql:
            return None
        repaired_sql_validation = await self._execute_sql_validation(
            contract,
            operation,
            tasks,
            evidence_store,
            str(repaired_sql),
            plan_validation=validation,
            analysis_metadata={"zero_row_repair": True},
            extra_dependencies=(
                TaskDependency(
                    kind="evidence",
                    evidence_kind=QUERY_ZERO_ROW_DIAGNOSIS_EVIDENCE,
                    evidence_id=diagnosis.id,
                    evidence_accepted=diagnosis.accepted,
                    operation_id=operation.id,
                ),
            ),
        )
        repaired_read = await self._execute_validated_read(
            contract,
            operation,
            tasks,
            evidence_store,
            repaired_sql_validation,
            analysis_metadata={"zero_row_repair": True},
            extra_dependencies=(
                TaskDependency(
                    kind="evidence",
                    evidence_kind=QUERY_ZERO_ROW_DIAGNOSIS_EVIDENCE,
                    evidence_id=diagnosis.id,
                    evidence_accepted=diagnosis.accepted,
                    operation_id=operation.id,
                ),
            ),
        )
        if _query_result_evidence(repaired_read) is not None:
            evidence_store.discard(query_result.id)
            if query_result.id is not None:
                await self.runtime.store.discard_evidence(query_result.id)
            return str(repaired_sql)
        return None

    async def _repair_query_plan(
        self,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        *,
        planning_context: Evidence,
        prior_plan: Evidence,
        failure: Evidence,
        repair_attempt: int,
    ) -> Evidence | None:
        if repair_attempt > 1:
            return None
        capability = self.runtime.registry.get_capability(
            "db.query.repair", owner="db_runtime"
        )
        evidence = await self._execute_direct_capability(
            capability,
            operation,
            tasks,
            evidence_store,
            {
                "planning_context_evidence_id": planning_context.id,
                "prior_plan_evidence_id": prior_plan.id,
                "failure_evidence_id": failure.id,
                "repair_attempt": repair_attempt,
            },
            reason="query_plan_repair",
            dependencies=(
                TaskDependency(
                    kind="evidence",
                    evidence_kind=PLANNING_CONTEXT_EVIDENCE,
                    evidence_id=planning_context.id,
                    operation_id=operation.id,
                ),
                TaskDependency(
                    kind="evidence",
                    evidence_kind=QUERY_PLAN_PROPOSAL_EVIDENCE,
                    evidence_id=prior_plan.id,
                    evidence_accepted=prior_plan.accepted,
                    operation_id=operation.id,
                ),
                TaskDependency(
                    kind="evidence",
                    evidence_kind=failure.kind,
                    evidence_id=failure.id,
                    evidence_accepted=failure.accepted,
                    operation_id=operation.id,
                ),
            ),
        )
        proposals = [
            item for item in evidence if item.kind == QUERY_PLAN_PROPOSAL_EVIDENCE
        ]
        return proposals[-1] if proposals else None

    async def _try_deterministic_repair_fallback(
        self,
        request: DbRequest,
        operation: Operation,
        schema: dict[str, Any],
        relationship_payload: dict[str, Any] | None,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        *,
        planning_context: Evidence,
        prior_plan: Evidence,
        failure: Evidence,
        diagnostics: dict[str, Any],
    ) -> tuple[Evidence, Evidence, tuple[str, ...], dict[str, Any]] | None:
        plan = self.query_planner.plan_read_query(
            request,
            operation,
            schema,
            relationship_payload=relationship_payload,
            planning_context=planning_context.payload,
        )
        plan_evidence = replace(
            plan.evidence,
            metadata={
                **plan.evidence.metadata,
                "repair_fallback": True,
                "fallback_after_failure_evidence_id": failure.id,
                "prior_plan_evidence_id": prior_plan.id,
            },
        )
        persisted = await self._persist_runtime_evidence(operation, plan_evidence)
        evidence_store.add(persisted)
        validation = await self._validate_query_plan(
            operation,
            tasks,
            evidence_store,
            plan_evidence=persisted,
            planning_context=planning_context,
            analysis_metadata={
                "repair_fallback": True,
                "fallback_after_failure_evidence_id": failure.id,
                "prior_plan_evidence_id": prior_plan.id,
            },
            extra_dependencies=(
                TaskDependency(
                    kind="evidence",
                    evidence_kind=failure.kind,
                    evidence_id=failure.id,
                    evidence_accepted=failure.accepted,
                    operation_id=operation.id,
                ),
            ),
        )
        fallback_diagnostics = {
            **plan.diagnostics,
            "repair_fallback": True,
            "fallback_after_failure_evidence_id": failure.id,
            "prior_plan_evidence_id": prior_plan.id,
        }
        if _accepted_sql(validation):
            diagnostics["repair_fallback_used"] = True
            return persisted, validation, plan.warnings, fallback_diagnostics

        await self._record_repair_exhaustion(
            operation,
            evidence_store,
            failure=failure,
            prior_plan=prior_plan,
            fallback_plan=persisted,
            fallback_validation=validation,
            diagnostics=diagnostics,
        )
        return None

    async def _record_repair_exhaustion(
        self,
        operation: Operation,
        evidence_store: DbEvidenceStore,
        *,
        failure: Evidence,
        prior_plan: Evidence,
        fallback_plan: Evidence,
        fallback_validation: Evidence,
        diagnostics: dict[str, Any],
    ) -> Evidence:
        terminal_payload = {
            "valid": False,
            "failure": "repair_exhausted",
            "repair_exhausted": True,
            "accepted_sql_missing": not bool(
                fallback_validation.payload.get("accepted_sql")
            ),
            "failure_evidence_id": failure.id,
            "failure_evidence_kind": failure.kind,
            "prior_plan_evidence_id": prior_plan.id,
            "fallback_plan_evidence_id": fallback_plan.id,
            "fallback_validation_evidence_id": fallback_validation.id,
            "fallback_validation_payload": dict(fallback_validation.payload),
        }
        diagnostics.update(
            {
                "repair_exhausted": True,
                "accepted_sql_missing": terminal_payload["accepted_sql_missing"],
                "repair_terminal_failure_evidence_kind": QUERY_PLAN_VALIDATION_EVIDENCE,
                "repair_terminal_failure": terminal_payload,
            }
        )
        return await self._record_failure_evidence(
            operation,
            evidence_store,
            QUERY_PLAN_VALIDATION_EVIDENCE,
            terminal_payload,
        )

    async def _record_failure_evidence(
        self,
        operation: Operation,
        evidence_store: DbEvidenceStore,
        kind: str,
        payload: dict[str, Any],
    ) -> Evidence:
        evidence = await self._persist_runtime_evidence(
            operation,
            Evidence(
                kind=kind,
                owner="db_runtime",
                operation_id=operation.id,
                accepted=False,
                payload=payload,
            ),
        )
        evidence_store.add(evidence)
        return evidence


def _query_result_evidence(evidence: tuple[Evidence, ...]) -> Evidence | None:
    return next((item for item in evidence if item.kind == "query.result"), None)


def _query_result_is_zero_rows(evidence: Evidence) -> bool:
    rows = evidence.payload.get("rows")
    return isinstance(rows, list) and len(rows) == 0


def _zero_row_literal_predicates(
    literal_predicates: tuple[Any, ...],
) -> list[dict[str, Any]]:
    predicates: list[dict[str, Any]] = []
    for predicate in literal_predicates:
        operator = str(getattr(predicate, "operator", "") or "").lower()
        if operator not in {"=", "in", "like"}:
            continue
        column_ref = getattr(predicate, "column", None)
        column = str(getattr(column_ref, "name", "") or "")
        if not column:
            continue
        values = [
            str(value)
            for value in getattr(predicate, "values", ()) or ()
            if value is not None
        ]
        if not values:
            continue
        predicates.append(
            {
                "table": _short_table_name(str(getattr(column_ref, "table", "") or "")),
                "column": column,
                "operator": operator,
                "values": values,
            }
        )
    return predicates


def _column_value_hints_for_predicates(
    planning_context: dict[str, Any],
    predicates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    wanted = [
        (
            _short_table_name(str(predicate.get("table") or "")),
            str(predicate.get("column") or "").lower(),
        )
        for predicate in predicates
        if predicate.get("column")
    ]
    hint_candidates = [
        hint
        for hint in planning_context.get("column_value_hints", []) or []
        if isinstance(hint, dict)
    ]
    hints: list[dict[str, Any]] = []
    for hint in hint_candidates:
        key = (
            _short_table_name(str(hint.get("table") or "")),
            str(hint.get("column") or "").lower(),
        )
        if not _hint_key_matches_predicates(key, wanted, hint_candidates):
            continue
        observed = [
            item
            for item in hint.get("observed_values", []) or []
            if isinstance(item, dict) and item.get("value") is not None
        ]
        if not observed:
            continue
        hints.append(
            {
                "table": hint.get("table"),
                "column": hint.get("column"),
                "profile_ref": hint.get("profile_ref"),
                "observed_values": observed[:25],
                **(
                    {"candidate_mapping": dict(hint["candidate_mapping"])}
                    if isinstance(hint.get("candidate_mapping"), dict)
                    else {}
                ),
            }
        )
    return hints


def _hint_key_matches_predicates(
    key: tuple[str, str],
    wanted: list[tuple[str, str]],
    hints: list[dict[str, Any]],
) -> bool:
    for wanted_table, wanted_column in wanted:
        if wanted_column != key[1]:
            continue
        if wanted_table:
            if wanted_table == key[0]:
                return True
            continue
        matching_tables = {
            _short_table_name(str(hint.get("table") or ""))
            for hint in hints
            if str(hint.get("column") or "").lower() == wanted_column
        }
        if len(matching_tables) == 1:
            return True
    return False
