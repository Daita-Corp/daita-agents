"""Monitor runtime behavior for ``DbRuntime``."""

from __future__ import annotations

from typing import Any

from daita.runtime import (
    ApprovalRequest,
    ApprovalStatus,
    Evidence,
    Operation,
    OperationStatus,
)

from ..analysis import (
    DbAnalysisPlan,
    capability_contract_for_step_kind,
)


def _normalize_monitor_action_plan(
    action_plan: dict[str, Any],
    *,
    operation_id: str,
) -> dict[str, Any]:
    raw = dict(action_plan or {})
    if raw.get("valid") is not None and isinstance(raw.get("analysis_plan"), dict):
        if raw.get("kind") == "scheduled_report" and "delivery_intent" not in raw:
            return {
                **raw,
                "delivery_intent": dict(raw.get("delivery") or {}),
                "delivery_status": raw.get("delivery_status") or "deferred",
            }
        return raw
    kind = str(raw.get("kind") or raw.get("type") or "").strip()
    steps = [dict(item) for item in raw.get("steps") or () if isinstance(item, dict)]
    if not kind:
        if any(str(step.get("kind") or "") == "report_generate" for step in steps):
            kind = "scheduled_report"
        elif steps:
            kind = "investigation"
    if kind in {"report", "scheduled-report", "scheduled_report"}:
        kind = "scheduled_report"
    if kind in {"investigate", "investigation"}:
        kind = "investigation"
    if kind in {
        "write",
        "write-proposal",
        "write_proposal",
        "remediation_sql",
        "propose_write",
    }:
        kind = "write_proposal"
    if kind in {"notify", "notification", "deliver", "delivery"}:
        kind = "notification"

    if kind == "notification":
        delivery_intent = dict(
            raw.get("delivery_intent")
            or raw.get("delivery")
            or raw.get("notification")
            or {}
        )
        if not delivery_intent:
            return _invalid_monitor_action(
                raw,
                kind=kind,
                reason="missing_delivery_intent",
            )
        if not delivery_intent.get("delivery_kind") and delivery_intent.get("mode"):
            delivery_intent["delivery_kind"] = delivery_intent.get("mode")
        if not delivery_intent.get("delivery_kind"):
            return _invalid_monitor_action(
                raw,
                kind=kind,
                reason="missing_delivery_kind",
            )
        return {
            "valid": True,
            "kind": "notification",
            "title": raw.get("title") or raw.get("goal") or "Monitor notification",
            "template": raw.get("template") or delivery_intent.get("template"),
            "output": dict(
                raw.get("output") or {"kind": "notification", "format": "markdown"}
            ),
            "delivery_status": "deferred",
            "delivery_phase": 3,
            "delivery_intent": delivery_intent,
            "original_action_plan": raw,
        }

    if kind == "investigation":
        analysis_steps = [
            _normalize_monitor_analysis_step(step)
            for step in steps
            if str(step.get("kind") or "") != "report_generate"
        ]
        if not analysis_steps:
            return _invalid_monitor_action(
                raw,
                kind=kind,
                reason="missing_executable_investigation_steps",
            )
        analysis_payload = {
            "analysis_id": str(
                raw.get("analysis_id") or f"monitor-action-{operation_id}"
            ),
            "goal": str(
                raw.get("goal") or raw.get("purpose") or "Investigate monitor trigger"
            ),
            "steps": analysis_steps,
            "budgets": dict(raw.get("budgets") or {}),
            "diagnostics": {
                **dict(raw.get("diagnostics") or {}),
                "source": "monitor.action_plan",
                "monitor_action_kind": "investigation",
            },
        }
        try:
            DbAnalysisPlan.from_mapping(analysis_payload)
        except Exception as exc:
            return _invalid_monitor_action(raw, kind=kind, reason=str(exc))
        return {
            "valid": True,
            "kind": "investigation",
            "goal": analysis_payload["goal"],
            "analysis_plan": analysis_payload,
            "original_action_plan": raw,
        }

    if kind == "scheduled_report":
        report_steps = _normalize_monitor_report_steps(steps)
        if not report_steps:
            return _invalid_monitor_action(
                raw,
                kind=kind,
                reason="missing_deterministic_report_steps",
            )
        analysis_steps = [
            _normalize_monitor_analysis_step(step)
            for step in steps
            if _is_monitor_report_analysis_step(step)
        ]
        if not any(step["kind"] == "synthesis" for step in analysis_steps):
            depends_on = [
                str(step.get("id"))
                for step in analysis_steps
                if step.get("id") and step.get("kind") != "checkpoint"
            ]
            analysis_steps.append(
                {
                    "id": "report_summary",
                    "kind": "synthesis",
                    "purpose": str(
                        raw.get("summary_purpose")
                        or "Generate the durable monitor report narrative"
                    ),
                    "depends_on": [],
                    "expected_evidence": ["analysis.synthesis"],
                    "input": {"report_step_ids": depends_on},
                }
            )
        analysis_payload = {
            "analysis_id": str(
                raw.get("analysis_id") or f"monitor-report-{operation_id}"
            ),
            "goal": str(
                raw.get("title")
                or raw.get("goal")
                or "Generate scheduled monitor report"
            ),
            "steps": analysis_steps,
            "budgets": dict(raw.get("budgets") or {}),
            "diagnostics": {
                **dict(raw.get("diagnostics") or {}),
                "source": "monitor.action_plan",
                "monitor_action_kind": "scheduled_report",
            },
        }
        try:
            DbAnalysisPlan.from_mapping(analysis_payload)
        except Exception as exc:
            return _invalid_monitor_action(raw, kind=kind, reason=str(exc))
        return {
            "valid": True,
            "kind": "scheduled_report",
            "title": raw.get("title") or raw.get("goal"),
            "steps": report_steps,
            "analysis_plan": analysis_payload,
            "output": dict(
                raw.get("output") or {"kind": "report", "format": "markdown"}
            ),
            "delivery_status": "deferred",
            "delivery_phase": 6,
            "delivery_intent": dict(
                raw.get("delivery_intent") or raw.get("delivery") or {}
            ),
            "original_action_plan": raw,
        }

    if kind == "write_proposal":
        proposal = raw.get("proposal")
        proposal = proposal if isinstance(proposal, dict) else {}
        sql = str(raw.get("sql") or proposal.get("sql") or "").strip()
        if not sql:
            return _invalid_monitor_action(
                raw,
                kind=kind,
                reason="missing_write_sql",
            )
        normalized = {
            "valid": True,
            "kind": "write_proposal",
            "sql": sql,
            "params": list(raw.get("params") or proposal.get("params") or ()),
            "source_scope": list(raw.get("source_scope") or ()),
            "purpose": str(
                raw.get("purpose")
                or proposal.get("purpose")
                or "Monitor write proposal"
            ),
            "original_action_plan": raw,
        }
        for key in ("capability_id", "capability_owner"):
            if raw.get(key) or proposal.get(key):
                normalized[key] = str(raw.get(key) or proposal.get(key))
        return normalized

    return _invalid_monitor_action(
        raw,
        kind=kind or "unknown",
        reason="unsupported_action_kind",
    )


def _normalize_monitor_analysis_step(step: dict[str, Any]) -> dict[str, Any]:
    kind = str(step.get("kind") or "").strip()
    normalized = {
        "id": str(step.get("id") or f"{kind}_step").strip(),
        "kind": kind,
        "purpose": str(step.get("purpose") or step.get("metric") or kind).strip(),
        "depends_on": [str(item) for item in step.get("depends_on") or ()],
        "input_refs": [
            dict(item)
            for item in step.get("input_refs") or ()
            if isinstance(item, dict)
        ],
        "expected_evidence": [
            str(item) for item in step.get("expected_evidence") or ()
        ],
        "input": dict(step.get("input") or {}),
        "context_evidence_refs": [
            dict(item)
            for item in step.get("context_evidence_refs") or ()
            if isinstance(item, dict)
        ],
        "budgets": dict(step.get("budgets") or {}),
    }
    for key in ("capability_id", "capability_owner"):
        if step.get(key):
            normalized[key] = str(step[key])
    return normalized


def _normalize_monitor_report_steps(
    steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, step in enumerate(steps, start=1):
        kind = str(step.get("kind") or "").strip()
        if kind == "report_generate":
            continue
        if kind == "query" and step.get("sql"):
            kind = "metric_sql"
        if kind not in {"metric_sql", "freshness_sql", "planned_read"}:
            continue
        sql = str(step.get("sql") or "").strip()
        if not sql:
            continue
        normalized.append(
            {
                "id": str(step.get("id") or f"report_step_{index}"),
                "kind": kind,
                "metric": step.get("metric"),
                "purpose": str(step.get("purpose") or step.get("metric") or kind),
                "sql": sql,
                "value_path": step.get("value_path"),
                "source_scope": list(step.get("source_scope") or ()),
                "parameters": list(step.get("parameters") or step.get("params") or ()),
                **(
                    {"capability_owner": str(step["capability_owner"])}
                    if step.get("capability_owner")
                    else {}
                ),
            }
        )
    return normalized


def _is_monitor_report_analysis_step(step: dict[str, Any]) -> bool:
    kind = str(step.get("kind") or "").strip()
    if kind in {"metric_sql", "freshness_sql", "planned_read", "report_generate"}:
        return False
    if kind == "query" and step.get("sql"):
        return False
    return kind in {"query", "checkpoint", "synthesis"} or (
        capability_contract_for_step_kind(kind) is not None
    )


def _invalid_monitor_action(
    raw: dict[str, Any],
    *,
    kind: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "valid": False,
        "kind": kind,
        "block_reason": reason,
        "original_action_plan": raw,
    }


def _monitor_action_budget_usage(evidence: tuple[Evidence, ...]) -> dict[str, Any]:
    total_rows = 0
    for item in evidence:
        if item.kind != "query.result":
            continue
        rows = item.payload.get("rows")
        if isinstance(rows, list):
            total_rows += len(rows)
    return {
        "evidence_count": len(evidence),
        "query_result_rows": total_rows,
    }


def _monitor_report_has_analysis_work(plan: DbAnalysisPlan) -> bool:
    return any(step.kind not in {"checkpoint", "synthesis"} for step in plan.steps)


def _terminal_monitor_approval_reason(
    approvals: tuple[ApprovalRequest, ...],
) -> str | None:
    statuses = {approval.status for approval in approvals}
    if ApprovalStatus.REJECTED in statuses:
        return "approval_rejected"
    if ApprovalStatus.CANCELLED in statuses:
        return "approval_cancelled"
    if ApprovalStatus.EXPIRED in statuses:
        return "approval_expired"
    return None


def _monitor_action_status_from_operation(operation: Operation) -> str:
    if operation.status is OperationStatus.BLOCKED:
        return "blocked"
    if operation.status is OperationStatus.FAILED:
        return "failed"
    if operation.status is OperationStatus.SUCCEEDED:
        return "succeeded"
    return operation.status.value
