"""Planner-facing summary helpers for the DB agent loop."""

from __future__ import annotations

from typing import Any, Mapping

from daita.runtime import Evidence, Task

from ..planner_protocol import DbLoopState
from ..planning_context import planner_eligible_column_value_hint
from .utils import (
    _dedupe_dicts,
    _dedupe_json_values,
    _optional_string,
    _safe_iterable,
    _string_list,
)


def _capability_summary(capability: Any) -> dict[str, Any]:
    return {
        "id": capability.id,
        "owner": capability.owner,
        "description": capability.description,
        "access": capability.access.value,
        "risk": capability.risk.value,
        "output_evidence": sorted(capability.output_evidence),
        "runtime_only": capability.runtime_only,
    }


def _task_summary(task: Task) -> dict[str, Any]:
    return {
        "task_id": task.id,
        "capability_id": task.capability_id,
        "executor_id": task.executor_id,
        "status": task.status.value,
        "metadata": {
            key: task.metadata.get(key)
            for key in (
                "owner",
                "reason",
                "sequence",
                "planner_action_id",
                "planner_action_kind",
            )
            if key in task.metadata
        },
    }


def _task_ref(task: Task) -> dict[str, Any]:
    return {
        "task_id": task.id,
        "capability_id": task.capability_id,
        "status": task.status.value,
    }


def _approval_task_id(approval: Any) -> str | None:
    direct = getattr(approval, "task_id", None)
    if direct:
        return str(direct)
    for source in (
        getattr(approval, "metadata", None),
        getattr(approval, "proposed_action", None),
    ):
        if isinstance(source, Mapping) and source.get("task_id"):
            return str(source["task_id"])
    return None


def _evidence_summary(evidence: Evidence) -> dict[str, Any]:
    summary = {
        "id": evidence.id,
        "kind": evidence.kind,
        "owner": evidence.owner,
        "accepted": evidence.accepted,
        "task_id": evidence.task_id,
    }
    sql = _sql_from_evidence_payload(evidence.payload)
    if sql:
        summary["sql"] = sql
    if isinstance(evidence.payload, dict):
        if evidence.kind in {
            "sql.validation",
            "query.plan.validation",
            "query.plan.proposal",
        }:
            if "valid" in evidence.payload:
                summary["valid"] = evidence.payload.get("valid") is True
            if evidence.kind == "query.plan.validation":
                plan_evidence_id = _optional_string(
                    evidence.payload.get("plan_evidence_id")
                )
                if plan_evidence_id is not None:
                    summary["plan_evidence_id"] = plan_evidence_id
            validation_facts = _safe_validation_items(
                evidence.payload.get("validation_facts")
            )
            validation_warnings = _safe_validation_items(
                evidence.payload.get("warnings")
            )
            validation_errors = _safe_validation_items(evidence.payload.get("errors"))
            if validation_facts:
                summary["validation_facts"] = validation_facts
            if validation_warnings:
                summary["warnings"] = validation_warnings
            if validation_errors:
                summary["validation_warnings"] = validation_errors
            if evidence.kind == "query.plan.proposal":
                structured_plan = evidence.payload.get("structured_plan")
                if isinstance(structured_plan, Mapping):
                    selected_tables = _string_list(
                        structured_plan.get("selected_tables")
                    )
                    if selected_tables:
                        summary["selected_tables"] = selected_tables
                    joins = _safe_join_summaries(structured_plan.get("joins"))
                    if joins:
                        summary["joins"] = joins
        if evidence.kind == "schema.column_value_hint":
            hints = _safe_column_value_hint_summaries(evidence.payload.get("hints"))
            if hints:
                summary["hints"] = hints
        if evidence.kind == "planning.context":
            for key in (
                "schema_evidence_refs",
                "catalog_evidence_refs",
                "relationship_evidence_refs",
            ):
                values = _string_list(evidence.payload.get(key))
                if values:
                    summary[key] = values
            relationship_joins = _planning_context_relationship_join_summaries(
                evidence.payload
            )
            if relationship_joins:
                summary["relationship_joins"] = relationship_joins
            diagnostics = evidence.payload.get("diagnostics")
            if isinstance(diagnostics, Mapping):
                structural_source = _optional_string(
                    diagnostics.get("structural_schema_source")
                )
                if structural_source is not None:
                    summary["structural_schema_source"] = structural_source
                catalog_structural_refs = _string_list(
                    diagnostics.get("catalog_structural_evidence_refs")
                )
                if catalog_structural_refs:
                    summary["catalog_structural_evidence_refs"] = (
                        catalog_structural_refs
                    )
                repair = diagnostics.get("validation_grounding_repair")
                if isinstance(repair, Mapping):
                    fingerprint = str(repair.get("fingerprint") or "").strip()
                    if fingerprint:
                        summary["validation_grounding_fingerprint"] = fingerprint
                    target_refs = [
                        str(item).strip()
                        for item in repair.get("target_refs", ())
                        if str(item).strip()
                    ]
                    if target_refs:
                        summary["validation_grounding_target_refs"] = target_refs
        if evidence.kind == "db.memory.proposal":
            proposal_fingerprint = evidence.payload.get("proposal_fingerprint")
            if isinstance(proposal_fingerprint, str) and proposal_fingerprint.strip():
                summary["proposal_fingerprint"] = proposal_fingerprint.strip()
        if evidence.kind == "db.memory.definition":
            proposal_evidence_id = evidence.payload.get(
                "proposal_evidence_id"
            ) or evidence.metadata.get("proposal_evidence_id")
            if proposal_evidence_id:
                summary["proposal_evidence_id"] = str(proposal_evidence_id)
            proposal_fingerprint = evidence.payload.get(
                "proposal_fingerprint"
            ) or evidence.metadata.get("proposal_fingerprint")
            if isinstance(proposal_fingerprint, str) and proposal_fingerprint.strip():
                summary["proposal_fingerprint"] = proposal_fingerprint.strip()
        operation = _optional_string(evidence.payload.get("operation"))
        if operation is not None:
            summary["operation"] = operation
        if evidence.kind == "schema.relationship_path":
            summary["reachable"] = evidence.payload.get("reachable")
            joins = _relationship_path_join_summaries(evidence.payload)
            if joins:
                summary["joins"] = joins
    payload_fingerprint = evidence.metadata.get("payload_fingerprint")
    if payload_fingerprint:
        summary["payload_fingerprint"] = str(payload_fingerprint)
    task_input_hash = evidence.metadata.get("task_input_hash")
    if task_input_hash:
        summary["task_input_hash"] = str(task_input_hash)
    return summary


def _safe_validation_items(value: Any) -> list[Any]:
    items: list[Any] = []
    for item in _safe_iterable(value):
        if isinstance(item, Mapping):
            safe = {
                key: item[key]
                for key in (
                    "kind",
                    "table",
                    "table_name",
                    "column",
                    "column_name",
                    "operator",
                    "literal",
                    "value",
                    "filter_literal",
                    "candidates",
                    "source",
                    "reason",
                )
                if key in item
            }
            if safe:
                items.append(safe)
        elif isinstance(item, str) and item.strip():
            items.append(item.strip())
    return _dedupe_json_values(items)


def _safe_column_value_hint_summaries(value: Any) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    for item in _safe_iterable(value):
        if not isinstance(item, Mapping):
            continue
        if not planner_eligible_column_value_hint(dict(item)):
            continue
        table = str(item.get("table") or "").strip()
        column = str(item.get("column") or "").strip()
        if not table or not column:
            continue
        hint: dict[str, Any] = {
            "table": table,
            "column": column,
        }
        observed_values = []
        for observed in _safe_iterable(item.get("observed_values")):
            if isinstance(observed, Mapping):
                if observed.get("value") is not None:
                    observed_values.append({"value": observed.get("value")})
            elif observed is not None:
                observed_values.append({"value": observed})
        if observed_values:
            hint["observed_values"] = observed_values[:25]
        candidate_mapping = item.get("candidate_mapping")
        if isinstance(candidate_mapping, Mapping):
            hint["candidate_mapping"] = dict(candidate_mapping)
        hints.append(hint)
    return _dedupe_dicts(hints, keys=("table", "column"))


def _safe_join_summaries(value: Any) -> list[dict[str, Any]]:
    joins: list[dict[str, Any]] = []
    for item in _safe_iterable(value):
        if not isinstance(item, Mapping):
            continue
        left_table = _optional_string(item.get("left_table") or item.get("left_asset"))
        right_table = _optional_string(
            item.get("right_table") or item.get("right_asset")
        )
        if not left_table or not right_table:
            continue
        join = {
            "left_table": left_table,
            "right_table": right_table,
        }
        left_column = _optional_string(
            item.get("left_column") or item.get("left_field") or item.get("left_key")
        )
        right_column = _optional_string(
            item.get("right_column") or item.get("right_field") or item.get("right_key")
        )
        if left_column:
            join["left_column"] = left_column.split(".")[-1]
        if right_column:
            join["right_column"] = right_column.split(".")[-1]
        joins.append(join)
    return _dedupe_dicts(
        joins, keys=("left_table", "left_column", "right_table", "right_column")
    )


def _relationship_path_join_summaries(
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    joins: list[dict[str, Any]] = []
    for path in payload.get("paths", []) or []:
        if not isinstance(path, Mapping):
            continue
        joins.extend(_safe_join_summaries(path.get("joins")))
    return _dedupe_dicts(
        joins,
        keys=("left_table", "left_column", "right_table", "right_column"),
    )


def _planning_context_relationship_join_summaries(
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    joins: list[dict[str, Any]] = []
    for detail in payload.get("relationship_evidence_details", []) or []:
        if not isinstance(detail, Mapping):
            continue
        joins.extend(_relationship_path_join_summaries(detail))
    return _dedupe_dicts(
        joins,
        keys=("left_table", "left_column", "right_table", "right_column"),
    )


def _evidence_ref(evidence: Evidence) -> dict[str, Any]:
    return {
        "id": evidence.id,
        "kind": evidence.kind,
        "accepted": evidence.accepted,
    }


def _sql_from_evidence_payload(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("sql", "selected_sql"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    structured_plan = payload.get("structured_plan")
    if isinstance(structured_plan, dict):
        selected_sql = structured_plan.get("selected_sql")
        if isinstance(selected_sql, str) and selected_sql.strip():
            return selected_sql.strip()
    return None


def _latest_accepted_evidence_summary(
    state: DbLoopState,
    kind: str,
) -> dict[str, Any] | None:
    for item in reversed(state.accepted_evidence_summaries):
        if item.get("kind") == kind and item.get("accepted", True) is True:
            return dict(item)
    return None


def _state_has_accepted_evidence(state: DbLoopState, kind: str) -> bool:
    return any(
        item.get("kind") == kind and item.get("accepted", True) is True
        for item in state.accepted_evidence_summaries
    )
