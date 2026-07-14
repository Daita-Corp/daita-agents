"""Planning and validation for prompt-created DB monitor proposals."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import re
from typing import Any, Mapping

from daita.plugins import ExtensionRegistry

from ..fingerprints import persisted_fingerprint
from ..monitor_plugin_planning import (
    MonitorPluginPlanner,
    MonitorPluginPlanningBlocked,
)
from ..monitors import DbMonitor
from ..planning_context import planner_eligible_column_value_hint
from ..query_metadata import column_name, is_numeric_type, matching_tables, table_name
from ...core.db_type_metadata import column_type_metadata
from .intent import (
    MonitorActionIntent,
    MonitorBudgetIntent,
    MonitorConditionIntent,
    MonitorCreateIntent,
    MonitorDeliveryRequest,
    MonitorDisplayIntent,
    MonitorPolicyIntent,
    MonitorScheduleIntent,
    MonitorTargetIntent,
)
from .naming import monitor_display_name, monitor_id as monitor_id_from_intent
from .types import DbMonitorCommand, DbMonitorValidation


class DbMonitorPlanner:
    """Build and validate monitor proposals from structured planner input."""

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
        grounding_evidence_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], DbMonitorValidation]:
        """Plan a prompt-created monitor as an executable proposal."""
        if command.kind != "create":
            raise ValueError("DbMonitorPlanner can only create monitor proposals")
        prompt = command.prompt
        if intent is None:
            raw_intent = command.diagnostics.get("intent")
            if not isinstance(raw_intent, dict):
                raise ValueError("structured monitor create intent is required")
            intent = monitor_create_intent_from_dict(raw_intent)
        target = intent.target.name or ""
        schedule = (
            intent.schedule.to_schedule_dict()
            if intent.schedule is not None
            else {"interval_seconds": 300}
        )
        budgets = intent.budget.to_dict()
        policy = intent.policy.to_dict()
        action_steps = intent.action.steps
        name = monitor_display_name(intent)
        monitor_id = monitor_id_from_intent(intent, explicit_id=command.monitor_id)
        effective_scope = tuple(intent.target.source_scope or source_scope or ())
        if not effective_scope and target:
            effective_scope = (target,)
        observation_value_path = str(intent.observation.get("value_path") or "rows")
        trigger, trigger_errors = _trigger_from_condition(
            intent.condition,
            observation_value_path=observation_value_path,
        )
        observation_plan, initial_cursor, observation_errors = (
            _planned_read_observation_plan(
                target,
                budgets=budgets,
                schema=schema,
                registry=self.registry,
                observation=intent.observation,
                grounding_evidence_by_id=grounding_evidence_by_id or {},
                runtime_limits=self.limits,
                required_row_count=_required_row_count(trigger),
            )
        )
        action_plan = _notification_action_plan(intent)
        action_plan, validation = self._normalize_and_validate(
            action_steps=action_steps,
            source_scope=effective_scope,
            policy=policy,
            budgets=budgets,
            observation_plan=observation_plan,
            schedule=schedule,
            stream=None,
            trigger=trigger,
            action_plan=action_plan,
            additional_errors=(*observation_errors, *trigger_errors),
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

    def create_structured_proposal(
        self,
        data: dict[str, Any],
        *,
        source_scope: tuple[str, ...] = (),
        owner: dict[str, Any] | None = None,
        schema_evidence_id: str | None = None,
    ) -> tuple[dict[str, Any], DbMonitorValidation]:
        """Validate a fully structured monitor proposal payload."""
        proposal_input = dict(data)
        monitor_id = str(
            proposal_input.get("monitor_id") or proposal_input.get("id") or ""
        ).strip()
        name = str(
            proposal_input.get("name")
            or proposal_input.get("display_name")
            or monitor_id
            or "DB Monitor"
        ).strip()
        target_name = str(
            proposal_input.get("target_name") or proposal_input.get("target") or ""
        ).strip()
        effective_scope = tuple(
            str(item)
            for item in (
                proposal_input.get("source_scope")
                or source_scope
                or ((target_name,) if target_name else ())
            )
        )
        if not monitor_id:
            monitor_id = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            monitor_id = monitor_id or "db_monitor"
        schedule = _dict_or_none(proposal_input.get("schedule"))
        stream = _dict_or_none(proposal_input.get("stream"))
        trigger = dict(proposal_input.get("trigger") or {})
        observation_plan = dict(proposal_input.get("observation_plan") or {})
        action_plan = dict(
            proposal_input.get("action_plan") or {"kind": "none", "steps": []}
        )
        policy = dict(proposal_input.get("policy") or {})
        budgets = dict(proposal_input.get("budgets") or {})
        action_steps = tuple(
            dict(step)
            for step in action_plan.get("steps") or ()
            if isinstance(step, dict)
        )
        action_plan, validation = self._normalize_and_validate(
            action_steps=action_steps,
            source_scope=effective_scope,
            policy=policy,
            budgets=budgets,
            observation_plan=observation_plan,
            schedule=schedule,
            stream=stream,
            trigger=trigger,
            action_plan=action_plan,
        )
        proposal: dict[str, Any] = {
            "kind": "monitor.proposal",
            "monitor_id": monitor_id,
            "name": name,
            "description": str(proposal_input.get("description") or ""),
            "status": str(proposal_input.get("status") or "active"),
            "target_type": str(proposal_input.get("target_type") or "table"),
            "target_name": target_name,
            "source_scope": list(effective_scope),
            "schedule": schedule,
            "stream": stream,
            "trigger": trigger,
            "observation_plan": observation_plan,
            "action_plan": action_plan,
            "initial_state": dict(proposal_input.get("initial_state") or {}),
            "policy": policy,
            "budgets": budgets,
            "owner": dict(owner or proposal_input.get("owner") or {}),
            "runtime_limits": dict(self.limits),
            "governance": dict(
                proposal_input.get("governance")
                or {
                    "approval_required": False,
                    "risk": "medium",
                    "side_effect_summary": (
                        "Creates runtime monitor configuration; observation is read-only."
                    ),
                }
            ),
            "metadata": {
                **dict(proposal_input.get("metadata") or {}),
                "created_from_structured_planner_action": True,
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
        _, validation = self._normalize_and_validate(
            action_steps=action_steps,
            source_scope=source_scope,
            policy=policy,
            budgets=budgets,
            observation_plan=observation_plan,
            schedule=schedule,
            stream=stream,
            trigger=trigger,
            action_plan=action_plan,
        )
        return validation

    def _normalize_and_validate(
        self,
        *,
        action_steps: tuple[dict[str, Any], ...],
        source_scope: tuple[str, ...],
        policy: dict[str, Any],
        budgets: dict[str, Any],
        observation_plan: dict[str, Any] | None = None,
        schedule: dict[str, Any] | None = None,
        stream: dict[str, Any] | None = None,
        trigger: dict[str, Any] | None = None,
        action_plan: dict[str, Any] | None = None,
        additional_errors: tuple[str, ...] = (),
    ) -> tuple[dict[str, Any], DbMonitorValidation]:
        normalized_action_plan = _action_plan_with_delivery_kind_default(
            deepcopy(action_plan or {}),
            delivery_default=self.delivery_default,
        )
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
            action_plan=normalized_action_plan,
        )
        plan_errors = tuple(dict.fromkeys((*plan_errors, *additional_errors)))
        delivery_errors: list[str] = []
        delivery_diagnostics: dict[str, Any] = {}
        delivery_intent = normalized_action_plan.get("delivery_intent")
        if isinstance(delivery_intent, dict) and not any(
            error.startswith("monitor.proposal_incomplete:delivery")
            for error in plan_errors
        ):
            try:
                normalized_intent, capability = MonitorPluginPlanner(
                    tuple(self.registry.capabilities)
                    if self.registry is not None
                    else ()
                ).normalize_delivery_intent(delivery_intent)
            except MonitorPluginPlanningBlocked as exc:
                details = dict(exc.details or {})
                capability_id = details.get("capability_id")
                missing_label = (
                    capability_id
                    if isinstance(capability_id, str) and capability_id
                    else None
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
                    "details": details,
                    "capability_id": capability_id,
                    "capability_owner": details.get("owner"),
                    "delivery_kind": details.get("delivery_kind")
                    or delivery_intent.get("delivery_kind")
                    or delivery_intent.get("mode"),
                    "delivery_intent": dict(delivery_intent),
                }
            else:
                normalized_action_plan["delivery_intent"] = normalized_intent.to_dict()
                required.append(capability.id)
                delivery_diagnostics = {
                    "accepted": True,
                    "capability_id": capability.id,
                    "capability_owner": capability.owner,
                    "delivery_kind": normalized_intent.delivery_kind,
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
        return (
            normalized_action_plan,
            DbMonitorValidation(
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
            ),
        )


def monitor_create_intent_from_dict(data: dict[str, Any]) -> MonitorCreateIntent:
    """Rehydrate structured planner JSON into a monitor create intent."""
    target = dict(data.get("target") or {})
    condition = dict(data.get("condition") or {})
    schedule = data.get("schedule")
    delivery = data.get("delivery")
    delivery_data = dict(delivery) if isinstance(delivery, dict) else {}
    raw_delivery_target = delivery_data.get("target")
    delivery_target = (
        dict(raw_delivery_target) if isinstance(raw_delivery_target, Mapping) else {}
    )
    action = dict(data.get("action") or {})
    display = dict(data.get("display") or {})
    raw_observation = data.get("observation")
    observation = (
        dict(raw_observation)
        if isinstance(raw_observation, Mapping)
        else (
            {"_invalid_shape": type(raw_observation).__name__}
            if raw_observation
            else {}
        )
    )
    return MonitorCreateIntent(
        target=MonitorTargetIntent(
            target_type=str(target.get("target_type") or "table"),
            name=_optional_string(target.get("name")),
            source_scope=tuple(str(item) for item in target.get("source_scope") or ()),
            confidence=_confidence(target.get("confidence")),
            evidence=tuple(str(item) for item in target.get("evidence") or ()),
        ),
        condition=MonitorConditionIntent(
            kind=str(condition.get("kind") or "rows_present"),
            expression=_optional_string(condition.get("expression")),
            operator=_optional_string(condition.get("operator")),
            value=condition.get("value"),
            path=_optional_string(condition.get("path")),
        ),
        schedule=(
            None
            if schedule is None
            else MonitorScheduleIntent(
                kind=str(dict(schedule).get("kind") or "interval"),
                interval_seconds=_optional_int(dict(schedule).get("interval_seconds")),
                expression=_optional_string(dict(schedule).get("expression")),
                timezone=_optional_string(dict(schedule).get("timezone")),
            )
        ),
        delivery=(
            None
            if delivery is None
            else MonitorDeliveryRequest(
                delivery_kind=str(
                    delivery_data.get("delivery_kind")
                    or delivery_data.get("mode")
                    or ""
                ),
                target=delivery_target,
                target_explicit="target" in delivery_data,
                explicit=bool(delivery_data.get("explicit", True)),
                payload_source=dict(
                    delivery_data.get("payload_source") or {"type": "monitor.report"}
                ),
                capability_id=_optional_string(delivery_data.get("capability_id")),
                capability_owner=_optional_string(
                    delivery_data.get("capability_owner") or delivery_data.get("owner")
                ),
                format=_optional_string(delivery_data.get("format")),
                subject=_optional_string(
                    delivery_data.get("subject") or delivery_data.get("title")
                ),
                template=_optional_string(delivery_data.get("template")),
                include_observed_rows=bool(
                    delivery_data.get("include_observed_rows", True)
                ),
            )
        ),
        observation=observation,
        action=MonitorActionIntent(
            actions=tuple(str(item) for item in action.get("actions") or ()),
            steps=tuple(
                dict(step)
                for step in action.get("steps") or ()
                if isinstance(step, dict)
            ),
        ),
        display=MonitorDisplayIntent(
            explicit_name=_optional_string(display.get("explicit_name")),
            suggested_name=_optional_string(display.get("suggested_name")),
            description=_optional_string(display.get("description")),
        ),
        policy=MonitorPolicyIntent(dict(data.get("policy") or {})),
        budget=MonitorBudgetIntent(
            dict(data.get("budget") or data.get("budgets") or {})
        ),
        confidence=_confidence(data.get("confidence")),
        diagnostics=dict(data.get("diagnostics") or {}),
    )


def _dict_or_none(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return dict(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _planned_read_observation_plan(
    target: str,
    *,
    budgets: dict[str, Any],
    schema: dict[str, Any] | None,
    registry: ExtensionRegistry | None = None,
    observation: Mapping[str, Any] | None = None,
    grounding_evidence_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    runtime_limits: Mapping[str, Any] | None = None,
    required_row_count: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
    observation = dict(observation or {})
    grounding_evidence_by_id = grounding_evidence_by_id or {}
    errors: list[str] = []
    unsupported_keys = sorted(set(observation) - {"filters", "value_path"})
    if unsupported_keys:
        errors.append("monitor.observation_invalid:unsupported_fields")
    if not _valid_sql_identifier(target):
        return (
            {
                "kind": "planned_read",
                "target": target,
                "planning_status": "blocked",
                "block_reason": "invalid_target_identifier",
            },
            {},
            (*errors, "monitor.observation_invalid:unsafe_target_identifier"),
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
            (*errors, "monitor.observation_invalid:missing_reliable_cursor"),
        )
    max_rows, row_limit_error = _observation_row_limit(
        budgets,
        runtime_limits=runtime_limits or {},
        required_row_count=required_row_count,
    )
    if row_limit_error is not None:
        errors.append(row_limit_error)
    field = cursor["field"]
    cursor_key = cursor["cursor_key"]
    sql_contract = _planned_read_sql_contract(schema, registry)
    filters, filter_parameters, filter_errors = _planned_observation_filters(
        target,
        schema=schema or {},
        observation=observation,
        sql_dialect=sql_contract["sql_dialect"],
        grounding_evidence_by_id=grounding_evidence_by_id,
    )
    errors.extend(filter_errors)
    cursor_position = len(filter_parameters) + 1
    placeholder = _sql_placeholder(sql_contract["sql_dialect"], cursor_position)
    parameter = _cursor_observation_parameter(
        cursor,
        target=target,
        cursor_key=cursor_key,
        dialect=sql_contract["sql_dialect"],
    )
    filter_predicates = [
        f"{item['column']} = " f"{_sql_placeholder(sql_contract['sql_dialect'], index)}"
        for index, item in enumerate(filters, start=1)
    ]
    where_predicates = [*filter_predicates, f"{field} > {placeholder}"]
    value_path = str(observation.get("value_path") or "rows")
    if not _valid_observation_value_path(value_path):
        errors.append("monitor.observation_invalid:value_path")
    return (
        {
            "kind": "planned_read",
            "target_type": "table",
            "target_name": target,
            "sql": (
                f"select * from {target} "
                f"where {' and '.join(where_predicates)} "
                f"order by {field} asc limit "
                f"{max_rows}"
            ),
            "parameters": [*filter_parameters, parameter],
            "filters": filters,
            "cursor": {
                "field": field,
                "initialization": cursor["initialization"],
                "strategy": cursor["strategy"],
                "source": cursor["source"],
                **(
                    {"column_type": cursor["column_type"]}
                    if cursor.get("column_type")
                    else {}
                ),
            },
            "cursor_update": {
                cursor_key: f"max(rows.{field})",
            },
            "max_rows": max_rows,
            "value_path": value_path,
            "capability_id": "db.sql.execute_read",
            **sql_contract,
        },
        {
            cursor_key: cursor["initial_value"],
            "initialized_from": "monitor_create",
        },
        tuple(dict.fromkeys(errors)),
    )


def _observation_row_limit(
    budgets: Mapping[str, Any],
    *,
    runtime_limits: Mapping[str, Any],
    required_row_count: int | None,
) -> tuple[int, str | None]:
    explicit_limit = budgets.get("max_rows_per_tick")
    try:
        requested = int(explicit_limit) if explicit_limit is not None else 100
    except (TypeError, ValueError):
        return 1, "monitor.observation_invalid:max_rows_per_tick"
    try:
        runtime_limit = int(runtime_limits.get("max_rows") or 1000)
    except (TypeError, ValueError):
        runtime_limit = 1000
    ceiling = max(1, min(runtime_limit, 1000))
    required = max(1, int(required_row_count or 1))
    if explicit_limit is None:
        requested = max(requested, required)
    max_rows = max(1, min(requested, ceiling))
    if required > max_rows:
        return max_rows, "monitor.observation_invalid:threshold_exceeds_row_limit"
    return max_rows, None


def _planned_observation_filters(
    target: str,
    *,
    schema: dict[str, Any],
    observation: Mapping[str, Any],
    sql_dialect: str,
    grounding_evidence_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[str, ...]]:
    raw_filters = observation.get("filters") or ()
    if not isinstance(raw_filters, (list, tuple)):
        return [], [], ("monitor.observation_invalid:filters",)
    tables = matching_tables(schema, target)
    if len(tables) != 1:
        return [], [], ("monitor.observation_invalid:target_profile",)
    table = tables[0]
    columns = {
        column_name(item): item
        for item in table.get("columns") or ()
        if isinstance(item, dict) and column_name(item)
    }
    filters: list[dict[str, Any]] = []
    parameters: list[dict[str, Any]] = []
    errors: list[str] = []
    for raw_filter in raw_filters:
        if not isinstance(raw_filter, Mapping):
            errors.append("monitor.observation_invalid:filter_shape")
            continue
        item = dict(raw_filter)
        if set(item) - {"column", "operator", "value", "evidence_ids"}:
            errors.append("monitor.observation_invalid:filter_fields")
            continue
        column = str(item.get("column") or "")
        if not _valid_sql_identifier(column):
            errors.append("monitor.observation_invalid:unsafe_filter_identifier")
            continue
        column_schema = columns.get(column)
        if column_schema is None:
            errors.append("monitor.observation_invalid:unknown_filter_column")
            continue
        operator = str(item.get("operator") or "").lower()
        if operator != "eq":
            errors.append("monitor.observation_invalid:unsupported_filter_operator")
            continue
        value = item.get("value")
        if (
            "value" not in item
            or value is None
            or isinstance(value, (Mapping, list, tuple, set))
        ):
            errors.append("monitor.observation_invalid:filter_value")
            continue
        raw_evidence_ids = item.get("evidence_ids") or ()
        if not isinstance(raw_evidence_ids, (list, tuple)):
            errors.append("monitor.observation_invalid:filter_evidence_ids")
            continue
        evidence_ids = tuple(
            str(evidence_id) for evidence_id in raw_evidence_ids if str(evidence_id)
        )
        grounded_value, grounding_error = _grounded_filter_value(
            value,
            target=target,
            column=column,
            evidence_ids=evidence_ids,
            grounding_evidence_by_id=grounding_evidence_by_id,
        )
        if grounding_error is not None:
            errors.append(grounding_error)
            continue
        normalized_filter = {
            "column": column,
            "operator": "eq",
            "parameter_index": len(parameters) + 1,
            "evidence_ids": list(evidence_ids),
        }
        parameter = {
            "value": grounded_value,
            "source": "catalog_value_hint" if evidence_ids else "typed_literal",
            "table": table_name(table) or target,
            "column": column,
            "evidence_ids": list(evidence_ids),
        }
        type_metadata = column_type_metadata(table, column_schema, schema)
        if type_metadata is not None:
            parameter.update(type_metadata)
        else:
            parameter["dialect"] = sql_dialect
        filters.append(normalized_filter)
        parameters.append(parameter)
    return filters, parameters, tuple(dict.fromkeys(errors))


def _grounded_filter_value(
    value: Any,
    *,
    target: str,
    column: str,
    evidence_ids: tuple[str, ...],
    grounding_evidence_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[Any, str | None]:
    if not evidence_ids:
        if isinstance(value, str):
            return value, "monitor.observation_invalid:ambiguous_ungrounded_value"
        return value, None
    candidates: list[Any] = []
    for evidence_id in evidence_ids:
        evidence = grounding_evidence_by_id.get(evidence_id)
        if not isinstance(evidence, Mapping):
            return value, "monitor.observation_invalid:unrelated_grounding_evidence"
        if (
            evidence.get("accepted", True) is not True
            or evidence.get("owner") != "catalog"
        ):
            return value, "monitor.observation_invalid:unrelated_grounding_evidence"
        supported, relevant = _grounding_candidates_from_evidence(
            evidence,
            target=target,
            column=column,
            requested=value,
        )
        if not relevant:
            return value, "monitor.observation_invalid:unrelated_grounding_evidence"
        candidates.extend(supported)
    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    if len(unique) != 1:
        return value, "monitor.observation_invalid:ambiguous_ungrounded_value"
    return unique[0], None


def _grounding_candidates_from_evidence(
    evidence: Mapping[str, Any],
    *,
    target: str,
    column: str,
    requested: Any,
) -> tuple[list[Any], bool]:
    payload = evidence.get("payload")
    payload = payload if isinstance(payload, Mapping) else evidence
    kind = str(evidence.get("kind") or "")
    hints: list[Mapping[str, Any]] = []
    if kind == "schema.column_value_hint":
        for hint in payload.get("hints") or ():
            if not isinstance(hint, Mapping):
                continue
            if hint.get("table") != target or hint.get("column") != column:
                continue
            if planner_eligible_column_value_hint(dict(hint)):
                hints.append(hint)
    elif kind == "schema.asset_profile":
        asset = payload.get("asset")
        asset = asset if isinstance(asset, Mapping) else {}
        if target not in {asset.get("name"), asset.get("asset_ref")}:
            return [], False
        for field in payload.get("fields") or ():
            if not isinstance(field, Mapping) or field.get("name") != column:
                continue
            inline_hint = field.get("column_value_hint")
            if isinstance(inline_hint, Mapping):
                hints.append(inline_hint)
    else:
        return [], False
    if not hints:
        return [], False
    candidates: list[Any] = []
    for hint in hints:
        observed = hint.get("observed_values") or hint.get("top_values") or ()
        observed_values = [
            item.get("value") if isinstance(item, Mapping) else item
            for item in observed
        ]
        for observed_value in observed_values:
            if observed_value == requested or (
                isinstance(observed_value, str)
                and isinstance(requested, str)
                and observed_value.casefold() == requested.casefold()
            ):
                candidates.append(observed_value)
        mapping = hint.get("candidate_mapping")
        if isinstance(mapping, Mapping):
            prompt_term = mapping.get("prompt_term")
            closest = mapping.get("closest_value")
            if (
                isinstance(prompt_term, str)
                and isinstance(requested, str)
                and prompt_term.casefold() == requested.casefold()
                and closest in observed_values
            ):
                candidates.append(closest)
    return candidates, True


def _valid_observation_value_path(value: str) -> bool:
    return bool(
        re.fullmatch(
            r"rows(?:\.\d+\.[A-Za-z_][A-Za-z0-9_]*)?",
            value or "",
        )
    )


def _cursor_observation_parameter(
    cursor: dict[str, Any],
    *,
    target: str,
    cursor_key: str,
    dialect: str,
) -> dict[str, Any]:
    ref = f"monitor.state.cursor.{cursor_key}"
    base = {
        "ref": ref,
        "source": "monitor_state",
        "path": ["cursor", cursor_key],
        "table": target,
        "column": str(cursor["field"]),
    }
    column_type = cursor.get("column_type")
    if not isinstance(column_type, dict):
        return base
    db_type = column_type.get("db_type")
    native_type = column_type.get("native_type")
    param_dialect = column_type.get("dialect") or dialect
    return {
        **base,
        "table": str(column_type.get("table") or target),
        "column": str(column_type.get("column") or cursor["field"]),
        **({"db_type": str(db_type)} if db_type else {}),
        **({"native_type": str(native_type)} if native_type else {}),
        **({"dialect": str(param_dialect)} if param_dialect else {}),
        **(
            {"nullable": column_type["nullable"]}
            if column_type.get("nullable") is not None
            else {}
        ),
    }


def _planned_read_sql_contract(
    schema: dict[str, Any] | None,
    registry: ExtensionRegistry | None,
) -> dict[str, str]:
    schema_spec = _sql_dialect_spec(
        (schema or {}).get("sql_dialect") or (schema or {}).get("database_type")
    )
    read_owners = _capability_owners(registry, "db.sql.execute_read")
    validate_owners = _capability_owners(registry, "db.sql.validate")
    compatible_owners = read_owners & validate_owners
    known_owner = (
        schema_spec.dialect.value
        if schema_spec is not None and schema_spec.dialect is not _SqlDialect.STANDARD
        else None
    )
    if known_owner and (not compatible_owners or known_owner in compatible_owners):
        owner = known_owner
    elif len(compatible_owners) == 1:
        owner = next(iter(compatible_owners))
    elif len(read_owners) == 1:
        owner = next(iter(read_owners))
    else:
        owner = "db_runtime"
    dialect_spec = (
        schema_spec
        or _sql_dialect_spec(owner)
        or _SQL_DIALECT_SPECS[_SqlDialect.STANDARD]
    )
    return {
        "validation_owner": owner,
        "execution_owner": owner,
        "sql_dialect": dialect_spec.dialect.value,
    }


def _capability_owners(
    registry: ExtensionRegistry | None,
    capability_id: str,
) -> set[str]:
    if registry is None:
        return set()
    return {
        str(capability.owner)
        for capability in registry.capabilities
        if capability.id == capability_id
    }


class _SqlDialect(str, Enum):
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    STANDARD = "standard"


class _PlaceholderStyle(Enum):
    NUMBERED_DOLLAR = "numbered_dollar"
    PERCENT_S = "percent_s"
    QUESTION_MARK = "question_mark"


@dataclass(frozen=True)
class _SqlDialectSpec:
    dialect: _SqlDialect
    aliases: frozenset[str]
    placeholder_style: _PlaceholderStyle

    def placeholder(self, position: int) -> str:
        if self.placeholder_style is _PlaceholderStyle.NUMBERED_DOLLAR:
            return f"${position}"
        if self.placeholder_style is _PlaceholderStyle.PERCENT_S:
            return "%s"
        return "?"

    def placeholder_errors(self, sql: str, parameter_count: int) -> tuple[str, ...]:
        if self.placeholder_style is _PlaceholderStyle.NUMBERED_DOLLAR:
            if "?" in sql or "%s" in sql:
                return ("monitor.proposal_invalid:sql_placeholder_dialect",)
            found = {int(item) for item in re.findall(r"\$(\d+)", sql)}
            expected = set(range(1, parameter_count + 1))
            if expected <= found:
                return ()
            return ("monitor.proposal_invalid:sql_placeholder_count",)
        if self.placeholder_style is _PlaceholderStyle.PERCENT_S:
            if "?" in sql or re.search(r"\$\d+", sql):
                return ("monitor.proposal_invalid:sql_placeholder_dialect",)
            if sql.count("%s") >= parameter_count:
                return ()
            return ("monitor.proposal_invalid:sql_placeholder_count",)
        if "%s" in sql or re.search(r"\$\d+", sql):
            return ("monitor.proposal_invalid:sql_placeholder_dialect",)
        if sql.count("?") >= parameter_count:
            return ()
        return ("monitor.proposal_invalid:sql_placeholder_count",)


_SQL_DIALECT_SPECS = {
    _SqlDialect.POSTGRESQL: _SqlDialectSpec(
        dialect=_SqlDialect.POSTGRESQL,
        aliases=frozenset({"postgresql", "postgres", "asyncpg"}),
        placeholder_style=_PlaceholderStyle.NUMBERED_DOLLAR,
    ),
    _SqlDialect.MYSQL: _SqlDialectSpec(
        dialect=_SqlDialect.MYSQL,
        aliases=frozenset({"mysql", "aiomysql"}),
        placeholder_style=_PlaceholderStyle.PERCENT_S,
    ),
    _SqlDialect.SQLITE: _SqlDialectSpec(
        dialect=_SqlDialect.SQLITE,
        aliases=frozenset({"sqlite", "aiosqlite"}),
        placeholder_style=_PlaceholderStyle.QUESTION_MARK,
    ),
    _SqlDialect.STANDARD: _SqlDialectSpec(
        dialect=_SqlDialect.STANDARD,
        aliases=frozenset({"standard"}),
        placeholder_style=_PlaceholderStyle.QUESTION_MARK,
    ),
}

_SQL_DIALECT_ALIASES = {
    alias: dialect
    for dialect, spec in _SQL_DIALECT_SPECS.items()
    for alias in spec.aliases
}


def _sql_dialect_spec(value: Any) -> _SqlDialectSpec | None:
    dialect = _SQL_DIALECT_ALIASES.get(str(value or "").strip().lower())
    if dialect is None:
        return None
    return _SQL_DIALECT_SPECS[dialect]


def _sql_placeholder(dialect: str, position: int) -> str:
    spec = _sql_dialect_spec(dialect) or _SQL_DIALECT_SPECS[_SqlDialect.STANDARD]
    return spec.placeholder(position)


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
            "column_type": column_type_metadata(table, timestamp, schema),
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
            "column_type": column_type_metadata(table, primary_key, schema),
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


def _trigger_from_condition(
    condition: MonitorConditionIntent,
    *,
    observation_value_path: str,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    if condition.path is not None and condition.path != observation_value_path:
        return (
            {
                "type": condition.kind,
                "path": condition.path,
            },
            ("monitor.trigger_invalid:path_mismatch",),
        )
    path = condition.path or observation_value_path
    raw_operator = str(condition.operator or "").strip().lower()
    if not raw_operator and condition.kind in {"new_rows", "rows_present"}:
        raw_operator = ">"
        value = 0
    else:
        value = condition.value
    normalized_operator = {
        ">": "gt",
        "gt": "gt",
        ">=": "gte",
        "gte": "gte",
        "<": "lt",
        "lt": "lt",
        "<=": "lte",
        "lte": "lte",
        "=": "equals",
        "==": "equals",
        "eq": "equals",
        "equals": "equals",
        "!=": "not_equals",
        "ne": "not_equals",
        "not_equals": "not_equals",
        "count_gt": "count_gt",
        "count_gte": "count_gte",
    }.get(raw_operator)
    if normalized_operator is None or value is None:
        return (
            {"type": condition.kind, "path": path},
            ("monitor.trigger_invalid:unsupported_comparison",),
        )
    if path == "rows":
        count_operator = {
            "gt": "count_gt",
            "gte": "count_gte",
            "count_gt": "count_gt",
            "count_gte": "count_gte",
        }.get(normalized_operator)
        if count_operator is None:
            return (
                {"type": condition.kind, "path": path},
                ("monitor.trigger_invalid:unsupported_row_comparison",),
            )
        try:
            threshold = int(value)
        except (TypeError, ValueError):
            return (
                {"type": condition.kind, "path": path},
                ("monitor.trigger_invalid:row_threshold",),
            )
        if threshold < 0:
            return (
                {"type": condition.kind, "path": path},
                ("monitor.trigger_invalid:row_threshold",),
            )
        return (
            {
                "type": "threshold",
                "operator": count_operator,
                "path": "rows",
                "value": threshold,
            },
            (),
        )
    if normalized_operator in {"count_gt", "count_gte"}:
        return (
            {"type": condition.kind, "path": path},
            ("monitor.trigger_invalid:count_operator_requires_rows",),
        )
    return (
        {
            "type": condition.kind,
            normalized_operator: value,
        },
        (),
    )


def _required_row_count(trigger: Mapping[str, Any]) -> int | None:
    operator = trigger.get("operator")
    if operator not in {"count_gt", "count_gte"}:
        return None
    value = trigger.get("value")
    if value is None:
        return None
    try:
        threshold = int(value)
    except (TypeError, ValueError):
        return None
    return threshold + 1 if operator == "count_gt" else threshold


def _notification_action_plan(intent: MonitorCreateIntent) -> dict[str, Any]:
    if intent.delivery is None:
        return {"kind": "none", "steps": []}
    return {
        "kind": "notification",
        "delivery_intent": intent.delivery.to_action_plan_intent(),
    }


def _action_plan_with_delivery_kind_default(
    action_plan: dict[str, Any],
    *,
    delivery_default: str | None,
) -> dict[str, Any]:
    delivery = action_plan.get("delivery_intent")
    if not isinstance(delivery, dict):
        return action_plan
    normalized = dict(delivery)
    explicit_kind = normalized.get("delivery_kind") or normalized.get("mode")
    if explicit_kind:
        normalized["delivery_kind"] = str(explicit_kind)
    elif delivery_default:
        normalized["delivery_kind"] = delivery_default
    action_plan["delivery_intent"] = normalized
    return action_plan


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
    if observation_plan.get("planning_status") == "blocked":
        block_reason = str(
            observation_plan.get("block_reason") or "observation_planning_blocked"
        )
        errors.append(f"monitor.proposal_invalid:{block_reason}")
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
        errors.extend(_placeholder_validation_errors(observation_plan))
    if schedule is None and stream is None:
        errors.append("monitor.proposal_incomplete:schedule_or_stream")
    if not trigger.get("operator") and not any(
        key in trigger for key in ("gt", "gte", "lt", "lte", "equals", "truthy")
    ):
        errors.append("monitor.proposal_incomplete:trigger")
    delivery_present = "delivery_intent" in action_plan
    if action_plan.get("kind") == "notification" or delivery_present:
        delivery = action_plan.get("delivery_intent")
        if not isinstance(delivery, dict) or not delivery.get("delivery_kind"):
            errors.append("monitor.proposal_incomplete:delivery")
    return tuple(dict.fromkeys(errors))


def _placeholder_validation_errors(
    observation_plan: dict[str, Any],
) -> tuple[str, ...]:
    sql = observation_plan.get("sql")
    if not isinstance(sql, str) or not sql.strip():
        return ()
    spec = _sql_dialect_spec(observation_plan.get("sql_dialect"))
    if spec is None or spec.dialect is _SqlDialect.STANDARD:
        return ()
    validation_owner = observation_plan.get("validation_owner")
    execution_owner = observation_plan.get("execution_owner")
    errors: list[str] = []
    if validation_owner and execution_owner and validation_owner != execution_owner:
        errors.append("monitor.proposal_invalid:sql_owner_mismatch")
    errors.extend(
        spec.placeholder_errors(
            sql,
            len(list(observation_plan.get("parameters") or ())),
        )
    )
    return tuple(errors)


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
    return persisted_fingerprint(cleaned)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _has_any(capability_ids: set[str], tokens: tuple[str, ...]) -> bool:
    return any(
        token in capability_id for capability_id in capability_ids for token in tokens
    )
