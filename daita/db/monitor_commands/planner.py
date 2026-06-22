"""Planning and validation for prompt-created DB monitor proposals."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any

from daita.plugins import ExtensionRegistry

from ..monitor_plugin_planning import (
    MonitorPluginPlanner,
    MonitorPluginPlanningBlocked,
)
from ..models import DbRequest
from ..monitors import DbMonitor
from ..query_metadata import column_name, is_numeric_type, matching_tables
from .extractor import DeterministicMonitorIntentExtractor
from .intent import MonitorConditionIntent, MonitorCreateIntent
from .naming import monitor_display_name, monitor_id as monitor_id_from_intent
from .types import DbMonitorCommand, DbMonitorValidation


class DbMonitorPlanner:
    """Parse a monitor-management prompt into a narrow `DbMonitor` spec."""

    def __init__(
        self,
        *,
        registry: ExtensionRegistry | None = None,
        limits: dict[str, Any] | None = None,
        delivery_default: str | None = None,
    ) -> None:
        self.registry = registry
        self.limits = dict(limits or {})
        self.delivery_default = delivery_default

    def create_monitor(
        self,
        command: DbMonitorCommand,
        *,
        source_scope: tuple[str, ...] = (),
        owner: dict[str, Any] | None = None,
    ) -> DbMonitor:
        if command.kind != "create":
            raise ValueError("DbMonitorPlanner can only create monitor specs")
        proposal, validation = self.create_proposal(
            command,
            source_scope=source_scope,
            owner=owner,
        )
        if not validation.accepted:
            raise ValueError(
                "monitor proposal is incomplete or unsupported: "
                + ", ".join(validation.errors)
            )
        return _monitor_from_proposal(proposal, validation=validation)

    def create_proposal(
        self,
        command: DbMonitorCommand,
        *,
        source_scope: tuple[str, ...] = (),
        owner: dict[str, Any] | None = None,
        schema: dict[str, Any] | None = None,
        schema_evidence_id: str | None = None,
        intent: MonitorCreateIntent | None = None,
    ) -> tuple[dict[str, Any], DbMonitorValidation]:
        """Plan a prompt-created monitor as an executable proposal."""
        if command.kind != "create":
            raise ValueError("DbMonitorPlanner can only create monitor proposals")
        prompt = command.prompt
        if intent is None:
            intent = DeterministicMonitorIntentExtractor().extract(
                command,
                DbRequest(prompt=prompt, source_scope=source_scope),
                schema=schema,
                host_defaults={"delivery_default": self.delivery_default},
            )
        target = intent.target.name or ""
        schedule = (
            intent.schedule.to_schedule_dict()
            if intent.schedule is not None
            else {"interval_seconds": 300}
        )
        trigger = _trigger_from_condition(intent.condition)
        actions = intent.action.actions
        budgets = intent.budget.to_dict()
        policy = intent.policy.to_dict()
        action_steps = intent.action.steps
        name = monitor_display_name(intent)
        monitor_id = monitor_id_from_intent(intent, explicit_id=command.monitor_id)
        effective_scope = tuple(intent.target.source_scope or source_scope or ())
        if not effective_scope and target:
            effective_scope = (target,)
        observation_plan, initial_cursor = _planned_read_observation_plan(
            target,
            budgets=budgets,
            schema=schema,
        )
        action_plan = _notification_action_plan(intent)
        validation = self.validate(
            action_steps=action_steps,
            actions=actions,
            source_scope=effective_scope,
            policy=policy,
            budgets=budgets,
            observation_plan=observation_plan,
            schedule=schedule,
            stream=None,
            trigger=trigger,
            action_plan=action_plan,
        )
        proposal: dict[str, Any] = {
            "kind": "monitor.proposal",
            "monitor_id": monitor_id,
            "name": name,
            "description": intent.display.description or prompt,
            "status": "active",
            "target_type": intent.target.target_type,
            "target_name": target,
            "source_scope": list(effective_scope),
            "schedule": schedule,
            "stream": None,
            "trigger": trigger,
            "observation_plan": observation_plan,
            "action_plan": action_plan,
            "initial_state": {
                "cursor": initial_cursor,
            },
            "policy": policy,
            "budgets": {
                "max_rows_per_tick": observation_plan.get("max_rows", 100),
                **budgets,
            },
            "owner": dict(owner or {}),
            "runtime_limits": dict(self.limits),
            "governance": {
                "approval_required": command.diagnostics.get("approval_required")
                is True,
                "risk": "medium",
                "side_effect_summary": (
                    "Creates runtime monitor configuration; observation is read-only."
                ),
            },
            "metadata": {
                "created_from_prompt": True,
                "prompt": prompt,
                "command": command.diagnostics,
                "intent": intent.to_dict(),
                "extraction": intent.diagnostics,
                "validation": validation.to_dict(),
                "schema_evidence_id": schema_evidence_id,
            },
        }
        proposal["proposal_fingerprint"] = _stable_monitor_payload_hash(proposal)
        proposal["metadata"]["proposal_fingerprint"] = proposal["proposal_fingerprint"]
        return proposal, validation

    def validate(
        self,
        *,
        action_steps: tuple[dict[str, Any], ...],
        actions: tuple[str, ...],
        source_scope: tuple[str, ...],
        policy: dict[str, Any],
        budgets: dict[str, Any],
        observation_plan: dict[str, Any] | None = None,
        schedule: dict[str, Any] | None = None,
        stream: dict[str, Any] | None = None,
        trigger: dict[str, Any] | None = None,
        action_plan: dict[str, Any] | None = None,
    ) -> DbMonitorValidation:
        capability_ids = (
            {capability.id for capability in self.registry.capabilities}
            if self.registry is not None
            else set()
        )
        required: list[str] = []
        unsupported: list[str] = []
        for step in action_steps:
            capability_id = step.get("required_capability")
            if isinstance(capability_id, str) and capability_id:
                required.append(capability_id)
                if capability_id not in capability_ids:
                    unsupported.append(str(step.get("instruction") or step["kind"]))
        if policy.get("requires_approval") and not _has_any(
            capability_ids, ("approval", "write", "sql")
        ):
            required.append("approval_or_write_action")
            unsupported.append("approval_or_write_action")
        max_rows = budgets.get("max_rows_per_tick")
        limit_max_rows = self.limits.get("max_rows")
        budget_warnings = []
        if isinstance(max_rows, int) and isinstance(limit_max_rows, int):
            if max_rows > limit_max_rows:
                budget_warnings.append("max_rows_per_tick_exceeds_runtime_limit")
        policy_warnings = []
        if policy.get("access") not in {None, "read"}:
            policy_warnings.append("monitor_policy_access_requires_runtime_review")
        plan_errors = _proposal_validation_errors(
            observation_plan=observation_plan or {},
            schedule=schedule,
            stream=stream,
            trigger=trigger or {},
            action_plan=action_plan or {},
        )
        delivery_errors: list[str] = []
        delivery_diagnostics: dict[str, Any] = {}
        if (action_plan or {}).get("kind") == "notification" and not any(
            error.startswith("monitor.proposal_incomplete:delivery")
            for error in plan_errors
        ):
            delivery_intent = (action_plan or {}).get("delivery_intent")
            if isinstance(delivery_intent, dict):
                try:
                    capability = MonitorPluginPlanner(
                        tuple(self.registry.capabilities)
                        if self.registry is not None
                        else ()
                    ).select_delivery_capability(delivery_intent)
                except MonitorPluginPlanningBlocked as exc:
                    missing_label = _delivery_missing_capability_label(
                        exc,
                        delivery_intent,
                    )
                    if missing_label is not None:
                        required.append(missing_label)
                    if (
                        exc.reason == "missing_capability"
                        and missing_label is not None
                        and missing_label not in capability_ids
                    ):
                        unsupported.append(
                            f"delivery:{delivery_intent.get('delivery_kind')}"
                        )
                    delivery_errors.append(f"monitor.delivery_unsupported:{exc.reason}")
                    delivery_diagnostics = {
                        "accepted": False,
                        "reason": exc.reason,
                        "details": dict(exc.details or {}),
                        "delivery_intent": dict(delivery_intent),
                    }
                else:
                    required.append(capability.id)
                    delivery_diagnostics = {
                        "accepted": True,
                        "capability_id": capability.id,
                        "capability_owner": capability.owner,
                        "delivery_kind": delivery_intent.get("delivery_kind"),
                    }
        missing = tuple(
            capability_id
            for capability_id in dict.fromkeys(required)
            if capability_id not in capability_ids
        )
        errors = tuple(
            (
                *(f"missing_capability:{capability_id}" for capability_id in missing),
                *plan_errors,
                *delivery_errors,
            )
        )
        warnings = tuple((*budget_warnings, *policy_warnings))
        return DbMonitorValidation(
            accepted=not errors,
            warnings=warnings,
            errors=errors,
            required_capabilities=tuple(dict.fromkeys(required)),
            missing_capabilities=missing,
            unsupported_actions=tuple(unsupported),
            diagnostics={
                "capability_ids": sorted(capability_ids),
                "source_scope": list(source_scope),
                "connector_guardrails": {
                    "sql_planning": "deferred_to_db_runtime",
                    "execution": "deferred_to_runtime_capabilities",
                },
                "runtime_limits": dict(self.limits),
                "delivery_validation": delivery_diagnostics,
            },
        )


def _planned_read_observation_plan(
    target: str,
    *,
    budgets: dict[str, Any],
    schema: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not _valid_sql_identifier(target):
        return (
            {
                "kind": "planned_read",
                "target": target,
                "planning_status": "blocked",
                "block_reason": "invalid_target_identifier",
            },
            {},
        )
    cursor = _cursor_strategy_from_schema(target, schema)
    if cursor is None:
        return (
            {
                "kind": "planned_read",
                "target_type": "table",
                "target_name": target,
                "planning_status": "blocked",
                "block_reason": "missing_reliable_cursor",
                "schema_available": bool(schema),
            },
            {},
        )
    max_rows = int(budgets.get("max_rows_per_tick") or 100)
    max_rows = max(1, min(max_rows, 1000))
    field = cursor["field"]
    cursor_key = cursor["cursor_key"]
    return (
        {
            "kind": "planned_read",
            "target_type": "table",
            "target_name": target,
            "sql": (
                f"select * from {target} "
                f"where {field} > ? order by {field} asc limit "
                f"{max_rows}"
            ),
            "parameters": [f"monitor.state.cursor.{cursor_key}"],
            "cursor": {
                "field": field,
                "initialization": cursor["initialization"],
                "strategy": cursor["strategy"],
                "source": cursor["source"],
            },
            "cursor_update": {
                cursor_key: f"max(rows.{field})",
            },
            "max_rows": max_rows,
            "value_path": "rows",
            "validation_owner": "db_runtime",
            "capability_id": "db.sql.execute_read",
        },
        {
            cursor_key: cursor["initial_value"],
            "initialized_from": "monitor_create",
        },
    )


def _cursor_strategy_from_schema(
    target: str,
    schema: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not schema:
        return None
    matches = matching_tables(schema, target)
    if len(matches) != 1:
        return None
    table = matches[0]
    columns = [
        column for column in table.get("columns") or [] if isinstance(column, dict)
    ]
    timestamp = _timestamp_cursor_column(columns)
    if timestamp is not None:
        field = column_name(timestamp)
        return {
            "field": field,
            "cursor_key": f"last_{field}",
            "initialization": "monitor_created_at",
            "initial_value": _now_iso(),
            "strategy": "insert_timestamp",
            "source": "schema_column",
        }
    primary_key = _auto_increment_primary_key(columns)
    if primary_key is not None:
        field = column_name(primary_key)
        return {
            "field": field,
            "cursor_key": f"last_{field}",
            "initialization": "zero",
            "initial_value": 0,
            "strategy": "auto_increment_primary_key",
            "source": "schema_primary_key",
        }
    return None


def _timestamp_cursor_column(columns: list[dict[str, Any]]) -> dict[str, Any] | None:
    preferred_names = (
        "created_at",
        "inserted_at",
        "created_on",
        "inserted_on",
        "created_ts",
        "inserted_ts",
    )
    by_name = {column_name(column).lower(): column for column in columns}
    for name in preferred_names:
        column = by_name.get(name)
        if column is not None and _looks_timestamp_type(column.get("data_type")):
            return column
    for column in columns:
        name = column_name(column).lower()
        if ("created" in name or "inserted" in name) and _looks_timestamp_type(
            column.get("data_type")
        ):
            return column
    return None


def _auto_increment_primary_key(columns: list[dict[str, Any]]) -> dict[str, Any] | None:
    for column in columns:
        if not column.get("is_primary_key"):
            continue
        if not is_numeric_type(str(column.get("data_type") or "")):
            continue
        if _looks_auto_incrementing(column):
            return column
    return None


def _looks_timestamp_type(type_name: Any) -> bool:
    lowered = str(type_name or "").lower()
    return any(token in lowered for token in ("timestamp", "datetime", "date"))


def _looks_auto_incrementing(column: dict[str, Any]) -> bool:
    data_type = str(column.get("data_type") or "").lower()
    default = str(
        column.get("default_value")
        or column.get("column_default")
        or column.get("default")
        or ""
    ).lower()
    extra = str(
        column.get("extra") or column.get("generation_expression") or ""
    ).lower()
    identity = str(column.get("identity") or column.get("is_identity") or "").lower()
    return any(
        token in f"{data_type} {default} {extra} {identity}"
        for token in (
            "serial",
            "identity",
            "nextval",
            "auto_increment",
            "autoincrement",
        )
    )


def _trigger_from_condition(condition: MonitorConditionIntent) -> dict[str, Any]:
    trigger: dict[str, Any] = {
        "type": condition.kind,
    }
    if condition.path is not None:
        trigger["path"] = condition.path
    if condition.operator is not None:
        trigger["operator"] = condition.operator
    if condition.value is not None:
        trigger["value"] = condition.value
    if condition.expression is not None:
        trigger["expression"] = condition.expression
    return trigger


def _notification_action_plan(intent: MonitorCreateIntent) -> dict[str, Any]:
    if intent.delivery is None:
        return {"kind": "none", "steps": []}
    return {
        "kind": "notification",
        "delivery_intent": intent.delivery.to_action_plan_intent(),
    }


def _delivery_missing_capability_label(
    exc: MonitorPluginPlanningBlocked,
    delivery_intent: dict[str, Any],
) -> str | None:
    details = dict(exc.details or {})
    capability_id = details.get("capability_id")
    if isinstance(capability_id, str) and capability_id:
        return capability_id
    delivery_kind = str(delivery_intent.get("delivery_kind") or "")
    if exc.reason == "missing_capability":
        if delivery_kind in {"in_app", "local"}:
            return f"monitor.delivery.{delivery_kind}"
    return None


def _proposal_validation_errors(
    *,
    observation_plan: dict[str, Any],
    schedule: dict[str, Any] | None,
    stream: dict[str, Any] | None,
    trigger: dict[str, Any],
    action_plan: dict[str, Any],
) -> tuple[str, ...]:
    errors: list[str] = []
    kind = observation_plan.get("kind")
    if kind not in {"planned_read", "metric_sql", "freshness_sql", "plugin_source"}:
        errors.append("monitor.proposal_incomplete:observation_plan")
    if kind in {"planned_read", "metric_sql", "freshness_sql"}:
        sql = observation_plan.get("sql")
        target = observation_plan.get("target_name") or observation_plan.get("target")
        if not isinstance(sql, str) or not sql.strip():
            errors.append("monitor.proposal_incomplete:observation_sql")
        if not isinstance(target, str) or not _valid_sql_identifier(target):
            errors.append("monitor.proposal_incomplete:source")
        if not observation_plan.get("cursor") or not observation_plan.get(
            "cursor_update"
        ):
            errors.append("monitor.proposal_incomplete:cursor")
        if not observation_plan.get("value_path"):
            errors.append("monitor.proposal_incomplete:value_path")
    if schedule is None and stream is None:
        errors.append("monitor.proposal_incomplete:schedule_or_stream")
    if not trigger.get("operator") and not any(
        key in trigger for key in ("gt", "gte", "lt", "lte", "equals", "truthy")
    ):
        errors.append("monitor.proposal_incomplete:trigger")
    if action_plan.get("kind") == "notification":
        delivery = action_plan.get("delivery_intent")
        if not isinstance(delivery, dict) or not delivery.get("delivery_kind"):
            errors.append("monitor.proposal_incomplete:delivery")
    return tuple(dict.fromkeys(errors))


def _monitor_from_proposal(
    proposal: dict[str, Any],
    *,
    validation: DbMonitorValidation | None = None,
) -> DbMonitor:
    metadata = dict(proposal.get("metadata") or {})
    if validation is not None:
        metadata["validation"] = validation.to_dict()
    metadata["proposal_fingerprint"] = proposal.get("proposal_fingerprint")
    return DbMonitor(
        id=str(proposal["monitor_id"]),
        name=str(proposal["name"]),
        description=str(proposal.get("description") or ""),
        status=str(proposal.get("status") or "active"),  # type: ignore[arg-type]
        source_scope=tuple(str(item) for item in proposal.get("source_scope") or ()),
        schedule=proposal.get("schedule"),
        stream=proposal.get("stream"),
        trigger=dict(proposal.get("trigger") or {}),
        observation_plan=dict(proposal.get("observation_plan") or {}),
        action_plan=dict(proposal.get("action_plan") or {}),
        policy=dict(proposal.get("policy") or {}),
        budgets=dict(proposal.get("budgets") or {}),
        owner=dict(proposal.get("owner") or {}),
        metadata=metadata,
    )


def _valid_sql_identifier(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", value or ""))


def _stable_monitor_payload_hash(payload: dict[str, Any]) -> str:
    cleaned = {
        key: value
        for key, value in payload.items()
        if key not in {"proposal_fingerprint"}
    }
    encoded = json.dumps(cleaned, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _has_any(capability_ids: set[str], tokens: tuple[str, ...]) -> bool:
    return any(
        token in capability_id for capability_id in capability_ids for token in tokens
    )
