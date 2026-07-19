"""Bounded Unix-socket dispatcher for one foreground :class:`AgentHost`.

The server is deliberately a transport adapter.  It creates no socket, task,
or host work until ``start``/``serve_forever`` is called, and every mutation is
delegated to the foreground host that already owns serialization and durable
state.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
import os
from pathlib import Path
import stat
from typing import NoReturn, Self

from .._json import FrozenJsonObject, thaw_json
from ..adapters.local_files import LocalDirectorySource
from ..adapters.models import SourceRegistration
from ..adapters.protocols import ResourceSource
from ..adapters.sqlite import SQLiteSource
from ..events.models import CommittedEvent, EventCursor
from ..loop.models import LoopExit
from ..monitors.models import (
    CatchUpPolicy,
    CronSchedule,
    IntervalSchedule,
    Monitor,
    MonitorBudgetOverrides,
    MonitorCondition,
    MonitorConditionKind,
    MonitorConfirmation,
    MonitorDefinition,
    MonitorInspection,
    MonitorProposal,
    MonitorScheduleState,
    MonitorScope,
    MonitorStatus,
    MonitorTimingPolicy,
)
from ..monitors.scheduler import MonitorSchedulerResult
from ..monitors.store import (
    MonitorConflictError,
    MonitorNotFoundError,
    MonitorProposalConflictError,
    MonitorProposalNotFoundError,
    MonitorTickClaimConflictError,
)
from ..operations.checkpoints import OperationSnapshot
from ..operations.governance import ApprovalRequest
from ..operations.runtime import OperationStateError
from ..operations.store import OperationNotFoundError
from .embedded import AgentNotConfiguredError
from .host import AgentHost, AgentHostState, AgentHostStateError, AgentHostStatus
from .inbox import (
    HostInboxEnqueueConflictError,
    HostInboxItem,
    HostMutationConflictError,
)
from .local_protocol import (
    LocalErrorResponse,
    LocalProtocolError,
    LocalRequest,
    LocalResponse,
    LocalSocketSecurityError,
    LocalSuccessResponse,
    encode_response,
    local_socket_path,
    prepare_local_socket_path,
    read_request,
    secure_bound_local_socket,
    write_frame,
    write_response,
)

_MAX_LIST_LIMIT = 1_000
_MAX_PROJECTED_RECORDS = 200
_DEFAULT_EVENT_LIMIT = 100
_DEFAULT_REQUEST_READ_TIMEOUT_SECONDS = 30.0
_MAX_REQUEST_READ_TIMEOUT_SECONDS = 300.0
_MUTATING_METHODS = frozenset(
    {
        "chat.submit",
        "operation.cancel",
        "approval.approve",
        "approval.reject",
        "source.attach",
        "monitor.propose",
        "monitor.confirm",
        "monitor.pause",
        "monitor.resume",
        "monitor.run_now",
        "monitor.delete",
    }
)


class LocalAgentServerStateError(RuntimeError):
    """Raised when a local server lifecycle command is invalid."""


class _RequestError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = {} if details is None else dict(details)


def _timestamp(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(timezone.utc).isoformat()


def _enum(value: Enum | None) -> str | None:
    return None if value is None else str(value.value)


def _thaw_object(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, FrozenJsonObject):
        raise TypeError("runtime JSON object must be frozen")
    thawed = thaw_json(value)
    if not isinstance(thawed, dict):
        raise TypeError("runtime JSON object must project to an object")
    return thawed


def _strict_object(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise _RequestError("invalid_params", f"{field_name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise _RequestError(
            "invalid_params",
            f"{field_name} must use string keys",
        )
    return dict(value)


def _shape(
    value: Mapping[str, object],
    *,
    required: frozenset[str] = frozenset(),
    optional: frozenset[str] = frozenset(),
    field_name: str = "params",
) -> dict[str, object]:
    resolved = dict(value)
    missing = required - resolved.keys()
    extra = resolved.keys() - required - optional
    if missing:
        raise _RequestError(
            "invalid_params",
            f"{field_name} is missing required fields",
            details={"fields": sorted(missing)},
        )
    if extra:
        raise _RequestError(
            "invalid_params",
            f"{field_name} contains unknown fields",
            details={"fields": sorted(extra)},
        )
    return resolved


def _text(
    value: object,
    field_name: str,
    *,
    maximum_bytes: int = 16_000,
) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise _RequestError(
            "invalid_params",
            f"{field_name} must be non-empty trimmed text",
        )
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise _RequestError(
            "invalid_params",
            f"{field_name} must be valid UTF-8 text",
        ) from error
    if len(encoded) > maximum_bytes:
        raise _RequestError("invalid_params", f"{field_name} is too long")
    return value


def _optional_text(
    value: object,
    field_name: str,
    *,
    maximum_bytes: int = 16_000,
) -> str | None:
    if value is None:
        return None
    return _text(value, field_name, maximum_bytes=maximum_bytes)


def _boolean(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise _RequestError("invalid_params", f"{field_name} must be a boolean")
    return value


def _integer(
    value: object,
    field_name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise _RequestError(
            "invalid_params",
            f"{field_name} must be an integer from {minimum} through {maximum}",
        )
    return value


def _optional_positive_integer(
    value: object,
    field_name: str,
    *,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    return _integer(value, field_name, minimum=1, maximum=maximum)


def _positive_number(
    value: object,
    field_name: str,
    *,
    maximum: float,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not 0 < float(value) <= maximum
    ):
        raise _RequestError(
            "invalid_params",
            f"{field_name} must be a positive number no greater than {maximum:g}",
        )
    return float(value)


def _utc_datetime(value: object, field_name: str) -> datetime:
    text = _text(value, field_name, maximum_bytes=64)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise _RequestError(
            "invalid_params",
            f"{field_name} must be an ISO-8601 UTC timestamp",
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _RequestError(
            "invalid_params",
            f"{field_name} must include a UTC offset",
        )
    return parsed.astimezone(timezone.utc)


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise _RequestError("invalid_params", f"{field_name} must be an array")
    if len(value) > 128:
        raise _RequestError("invalid_params", f"{field_name} is too large")
    return tuple(_text(item, f"{field_name} item", maximum_bytes=256) for item in value)


def _parse_schedule(value: object) -> IntervalSchedule | CronSchedule:
    schedule = _strict_object(value, "definition.schedule")
    kind_value = schedule.get("kind")
    if kind_value is None:
        if "interval_seconds" in schedule:
            kind = "interval"
        elif "expression" in schedule:
            kind = "cron"
        else:
            raise _RequestError(
                "invalid_params",
                "definition.schedule must select interval or cron",
            )
    else:
        kind = _text(kind_value, "definition.schedule.kind", maximum_bytes=16)
    if kind == "interval":
        fields = _shape(
            schedule,
            required=frozenset({"interval_seconds", "anchor_at"}),
            optional=frozenset({"kind"}),
            field_name="definition.schedule",
        )
        return IntervalSchedule(
            interval_seconds=_integer(
                fields["interval_seconds"],
                "definition.schedule.interval_seconds",
                minimum=1,
                maximum=366 * 24 * 60 * 60,
            ),
            anchor_at=_utc_datetime(
                fields["anchor_at"],
                "definition.schedule.anchor_at",
            ),
        )
    if kind == "cron":
        fields = _shape(
            schedule,
            required=frozenset({"expression"}),
            optional=frozenset({"kind", "timezone", "timezone_name"}),
            field_name="definition.schedule",
        )
        if "timezone" in fields and "timezone_name" in fields:
            raise _RequestError(
                "invalid_params",
                "definition.schedule must use one timezone field",
            )
        timezone_name = fields.get("timezone", fields.get("timezone_name", "UTC"))
        return CronSchedule(
            expression=_text(
                fields["expression"],
                "definition.schedule.expression",
                maximum_bytes=320,
            ),
            timezone_name=_text(
                timezone_name,
                "definition.schedule.timezone",
                maximum_bytes=64,
            ),
        )
    raise _RequestError(
        "invalid_params",
        "definition.schedule.kind must be interval or cron",
    )


def _parse_definition(value: object) -> MonitorDefinition:
    raw = _strict_object(value, "definition")
    definition = _shape(
        raw,
        required=frozenset({"name", "objective", "scope", "schedule"}),
        optional=frozenset(
            {
                "condition",
                "budget_overrides",
                "timing",
                "policy_overrides",
                "operation_template",
            }
        ),
        field_name="definition",
    )

    scope = _shape(
        _strict_object(definition["scope"], "definition.scope"),
        optional=frozenset({"source_ids", "resource_ids"}),
        field_name="definition.scope",
    )
    source_ids = _string_tuple(
        scope.get("source_ids", []),
        "definition.scope.source_ids",
    )
    resource_ids = _string_tuple(
        scope.get("resource_ids", []),
        "definition.scope.resource_ids",
    )

    condition_raw = _shape(
        _strict_object(definition.get("condition", {}), "definition.condition"),
        optional=frozenset({"kind", "expression", "configuration"}),
        field_name="definition.condition",
    )
    try:
        condition_kind = MonitorConditionKind(
            condition_raw.get("kind", MonitorConditionKind.ALWAYS.value)
        )
    except (TypeError, ValueError) as error:
        raise _RequestError(
            "invalid_params",
            "definition.condition.kind is unsupported",
        ) from error
    expression = _optional_text(
        condition_raw.get("expression"),
        "definition.condition.expression",
        maximum_bytes=4_000,
    )
    configuration = _strict_object(
        condition_raw.get("configuration", {}),
        "definition.condition.configuration",
    )

    budgets = _shape(
        _strict_object(
            definition.get("budget_overrides", {}),
            "definition.budget_overrides",
        ),
        optional=frozenset(
            {"max_turns", "max_capability_calls", "max_wall_time_seconds"}
        ),
        field_name="definition.budget_overrides",
    )

    timing = _shape(
        _strict_object(definition.get("timing", {}), "definition.timing"),
        optional=frozenset(
            {
                "catch_up",
                "cooldown_seconds",
                "initial_backoff_seconds",
                "max_backoff_seconds",
                "backoff_multiplier",
            }
        ),
        field_name="definition.timing",
    )
    try:
        catch_up = CatchUpPolicy(timing.get("catch_up", CatchUpPolicy.ONCE.value))
    except (TypeError, ValueError) as error:
        raise _RequestError(
            "invalid_params",
            "definition.timing.catch_up is unsupported",
        ) from error
    multiplier = timing.get("backoff_multiplier", 2.0)
    if not isinstance(multiplier, (int, float)) or isinstance(multiplier, bool):
        raise _RequestError(
            "invalid_params",
            "definition.timing.backoff_multiplier must be a number",
        )

    policy_overrides = _strict_object(
        definition.get("policy_overrides", {}),
        "definition.policy_overrides",
    )
    operation_template = _strict_object(
        definition.get("operation_template", {}),
        "definition.operation_template",
    )
    try:
        return MonitorDefinition(
            name=_text(definition["name"], "definition.name", maximum_bytes=128),
            objective=_text(
                definition["objective"],
                "definition.objective",
                maximum_bytes=16_000,
            ),
            scope=MonitorScope(
                source_ids=source_ids,
                resource_ids=resource_ids,
            ),
            schedule=_parse_schedule(definition["schedule"]),
            condition=MonitorCondition(
                kind=condition_kind,
                expression=expression,
                configuration=configuration,
            ),
            budget_overrides=MonitorBudgetOverrides(
                max_turns=_optional_positive_integer(
                    budgets.get("max_turns"),
                    "definition.budget_overrides.max_turns",
                    maximum=1_000,
                ),
                max_capability_calls=_optional_positive_integer(
                    budgets.get("max_capability_calls"),
                    "definition.budget_overrides.max_capability_calls",
                    maximum=10_000,
                ),
                max_wall_time_seconds=_optional_positive_integer(
                    budgets.get("max_wall_time_seconds"),
                    "definition.budget_overrides.max_wall_time_seconds",
                    maximum=86_400,
                ),
            ),
            timing=MonitorTimingPolicy(
                catch_up=catch_up,
                cooldown_seconds=_integer(
                    timing.get("cooldown_seconds", 0),
                    "definition.timing.cooldown_seconds",
                    minimum=0,
                    maximum=366 * 24 * 60 * 60,
                ),
                initial_backoff_seconds=_integer(
                    timing.get("initial_backoff_seconds", 1),
                    "definition.timing.initial_backoff_seconds",
                    minimum=1,
                    maximum=86_400,
                ),
                max_backoff_seconds=_integer(
                    timing.get("max_backoff_seconds", 300),
                    "definition.timing.max_backoff_seconds",
                    minimum=1,
                    maximum=366 * 24 * 60 * 60,
                ),
                backoff_multiplier=float(multiplier),
            ),
            policy_overrides=policy_overrides,
            operation_template=operation_template,
        )
    except _RequestError:
        raise
    except (TypeError, ValueError) as error:
        raise _RequestError(
            "invalid_params",
            "definition violates the monitor contract",
        ) from error


def _definition_projection(definition: MonitorDefinition) -> dict[str, object]:
    if isinstance(definition.schedule, IntervalSchedule):
        schedule: dict[str, object] = {
            "kind": "interval",
            "interval_seconds": definition.schedule.interval_seconds,
            "anchor_at": _timestamp(definition.schedule.anchor_at),
        }
    else:
        schedule = {
            "kind": "cron",
            "expression": definition.schedule.expression,
            "timezone": definition.schedule.timezone_name,
        }
    return {
        "name": definition.name,
        "objective": definition.objective,
        "scope": {
            "source_ids": list(definition.scope.source_ids),
            "resource_ids": list(definition.scope.resource_ids),
        },
        "schedule": schedule,
        "condition": {
            "kind": definition.condition.kind.value,
            "expression": definition.condition.expression,
            "configuration": _thaw_object(definition.condition.configuration),
        },
        "budget_overrides": {
            "max_turns": definition.budget_overrides.max_turns,
            "max_capability_calls": (definition.budget_overrides.max_capability_calls),
            "max_wall_time_seconds": (
                definition.budget_overrides.max_wall_time_seconds
            ),
        },
        "timing": {
            "catch_up": definition.timing.catch_up.value,
            "cooldown_seconds": definition.timing.cooldown_seconds,
            "initial_backoff_seconds": definition.timing.initial_backoff_seconds,
            "max_backoff_seconds": definition.timing.max_backoff_seconds,
            "backoff_multiplier": definition.timing.backoff_multiplier,
        },
        "policy_overrides": _thaw_object(definition.policy_overrides),
        "operation_template": _thaw_object(definition.operation_template),
        "content_hash": definition.content_hash,
    }


def _monitor_projection(monitor: Monitor) -> dict[str, object]:
    return {
        "id": monitor.id,
        "status": monitor.status.value,
        "current_version": monitor.current_version,
        "revision": monitor.revision,
        "created_at": _timestamp(monitor.created_at),
        "updated_at": _timestamp(monitor.updated_at),
        "paused_at": _timestamp(monitor.paused_at),
        "deleted_at": _timestamp(monitor.deleted_at),
    }


def _proposal_projection(proposal: MonitorProposal) -> dict[str, object]:
    return {
        "id": proposal.id,
        "monitor_id": proposal.intended_monitor_id,
        "candidate_hash": proposal.candidate_hash,
        "candidate": _definition_projection(proposal.candidate),
        "source_operation_id": proposal.source_operation_id,
        "created_at": _timestamp(proposal.created_at),
    }


def _confirmation_projection(
    confirmation: MonitorConfirmation,
) -> dict[str, object]:
    return {
        "id": confirmation.id,
        "proposal_id": confirmation.proposal_id,
        "decision": confirmation.decision.value,
        "candidate_hash": confirmation.candidate_hash,
        "actor_id": confirmation.actor_id,
        "reason": confirmation.reason,
        "decided_at": _timestamp(confirmation.decided_at),
        "monitor_id": confirmation.resulting_monitor_id,
        "version_id": confirmation.resulting_version_id,
    }


def _schedule_state_projection(
    schedule: MonitorScheduleState,
) -> dict[str, object]:
    return {
        "revision": schedule.revision,
        "next_scheduled_at": _timestamp(schedule.next_scheduled_at),
        "last_scheduled_at": _timestamp(schedule.last_scheduled_at),
        "cooldown_until": _timestamp(schedule.cooldown_until),
        "backoff_until": _timestamp(schedule.backoff_until),
        "consecutive_failures": schedule.consecutive_failures,
        "consecutive_matches": schedule.consecutive_matches,
        "checkpoint_version": schedule.checkpoint_version,
        "last_occurrence_id": schedule.last_occurrence_id,
        "last_run_id": schedule.last_run_id,
        "last_operation_id": schedule.last_operation_id,
        "updated_at": _timestamp(schedule.updated_at),
    }


def _inspection_projection(inspection: MonitorInspection) -> dict[str, object]:
    latest = inspection.versions[-1]
    limit = _MAX_PROJECTED_RECORDS
    return {
        "monitor": _monitor_projection(inspection.monitor),
        "current_version": {
            "id": latest.id,
            "version": latest.version,
            "content_hash": latest.content_hash,
            "proposal_id": latest.proposal_id,
            "source_operation_id": latest.source_operation_id,
            "created_at": _timestamp(latest.created_at),
            "definition": _definition_projection(latest.definition),
        },
        "schedule_state": _schedule_state_projection(inspection.schedule_state),
        "proposals": [
            _proposal_projection(value) for value in inspection.proposals[-limit:]
        ],
        "confirmations": [
            _confirmation_projection(value)
            for value in inspection.confirmations[-limit:]
        ],
        "lifecycle": [
            {
                "id": value.id,
                "action": value.action.value,
                "from_status": _enum(value.from_status),
                "to_status": value.to_status.value,
                "from_revision": value.from_revision,
                "to_revision": value.to_revision,
                "actor_id": value.actor_id,
                "reason": value.reason,
                "occurred_at": _timestamp(value.occurred_at),
                "operation_id": value.operation_id,
            }
            for value in inspection.lifecycle[-limit:]
        ],
        "runs": [
            {
                "id": value.id,
                "occurrence_id": value.occurrence_id,
                "trigger_id": value.trigger_id,
                "attempt": value.attempt,
                "status": value.status.value,
                "operation_id": value.operation_id,
                "started_at": _timestamp(value.started_at),
                "completed_at": _timestamp(value.completed_at),
                "failure_reason": value.failure_reason,
            }
            for value in inspection.runs[-limit:]
        ],
        "findings": [
            {
                "id": value.id,
                "run_id": value.run_id,
                "operation_id": value.operation_id,
                "evidence_id": value.evidence_id,
                "severity": value.severity.value,
                "summary": value.summary,
                "dedupe_key": value.dedupe_key,
                "created_at": _timestamp(value.created_at),
            }
            for value in inspection.findings[-limit:]
        ],
        "counts": {
            "versions": len(inspection.versions),
            "lifecycle": len(inspection.lifecycle),
            "proposals": len(inspection.proposals),
            "confirmations": len(inspection.confirmations),
            "occurrences": len(inspection.occurrences),
            "runs": len(inspection.runs),
            "findings": len(inspection.findings),
            "checkpoints": len(inspection.checkpoints),
        },
        "history_truncated": any(
            len(values) > limit
            for values in (
                inspection.lifecycle,
                inspection.proposals,
                inspection.confirmations,
                inspection.runs,
                inspection.findings,
            )
        ),
    }


def _approval_projection(approval: ApprovalRequest) -> dict[str, object]:
    return {
        "id": approval.id,
        "operation_id": approval.operation_id,
        "task_id": approval.task_id,
        "status": approval.status.value,
        "requested_at": _timestamp(approval.requested_at),
        "decided_at": _timestamp(approval.decided_at),
        "decided_by": approval.decided_by,
        "decision_reason": approval.decision_reason,
    }


def _operation_projection(snapshot: OperationSnapshot) -> dict[str, object]:
    operation = snapshot.operation
    task_limit = _MAX_PROJECTED_RECORDS
    evidence_limit = _MAX_PROJECTED_RECORDS
    return {
        "operation": {
            "id": operation.id,
            "agent_id": operation.agent_id,
            "trigger_id": operation.trigger_id,
            "session_id": operation.session_id,
            "status": operation.status.value,
            "created_at": _timestamp(operation.created_at),
            "updated_at": _timestamp(operation.updated_at),
            "final_text": operation.final_text,
            "terminal_reason": operation.terminal_reason,
        },
        "trigger": {
            "id": snapshot.trigger.id,
            "kind": snapshot.trigger.kind.value,
            "source_id": snapshot.trigger.source_id,
            "session_id": snapshot.trigger.session_id,
            "created_at": _timestamp(snapshot.trigger.created_at),
        },
        "loop": {
            "phase": snapshot.loop_state.phase.value,
            "turn_count": snapshot.loop_state.turn_count,
            "action_count": snapshot.loop_state.action_count,
            "repair_count": snapshot.loop_state.repair_count,
            "waiting_approval_id": snapshot.loop_state.waiting_approval_id,
            "interruption_reason": snapshot.loop_state.interruption_reason,
        },
        "tasks": [
            {
                "id": value.id,
                "status": value.status.value,
                "capability_id": value.capability_id,
                "executor_id": value.executor_id,
                "attempt": value.attempt,
                "evidence_ids": list(value.evidence_ids),
                "error_code": value.error_code,
                "cancellation_requested": value.cancellation_requested,
                "created_at": _timestamp(value.created_at),
                "updated_at": _timestamp(value.updated_at),
            }
            for value in snapshot.tasks[-task_limit:]
        ],
        "evidence": [
            {
                "id": value.id,
                "task_id": value.task_id,
                "kind": value.kind,
                "accepted": value.accepted,
                "content_hash": value.content_hash,
                "blob_id": value.blob_id,
                "created_at": _timestamp(value.created_at),
            }
            for value in snapshot.evidence[-evidence_limit:]
        ],
        "approvals": [
            _approval_projection(value) for value in snapshot.approvals[-task_limit:]
        ],
        "counts": {
            "turns": len(snapshot.turns),
            "model_calls": len(snapshot.model_calls),
            "tasks": len(snapshot.tasks),
            "evidence": len(snapshot.evidence),
            "approvals": len(snapshot.approvals),
            "events": len(snapshot.events),
        },
        "history_truncated": any(
            len(values) > task_limit
            for values in (snapshot.tasks, snapshot.evidence, snapshot.approvals)
        ),
    }


def _host_status_projection(status: AgentHostStatus) -> dict[str, object]:
    return {
        "agent_id": status.agent_id,
        "state": status.state.value,
        "configured": status.configured,
        "pending_inbox": status.pending_inbox,
        "nonterminal_operation_ids": list(status.nonterminal_operation_ids),
        "started_at": _timestamp(status.started_at),
        "last_pass_at": _timestamp(status.last_pass_at),
        "last_error": status.last_error,
    }


def _inbox_projection(item: HostInboxItem) -> dict[str, object]:
    return {
        "id": item.id,
        "kind": item.kind.value,
        "status": item.status.value,
        "revision": item.revision,
        "trigger_id": item.trigger_id,
        "operation_id": item.operation_id,
        "created_at": _timestamp(item.created_at),
        "updated_at": _timestamp(item.updated_at),
    }


def _source_projection(value: SourceRegistration) -> dict[str, object]:
    return {
        "id": value.id,
        "adapter_id": value.adapter_id,
        "native_identity": value.native_identity,
        "display_name": value.display_name,
        "attached_at": _timestamp(value.attached_at),
        "detached_at": _timestamp(value.detached_at),
        "active": value.active,
    }


def _model_projection(host: AgentHost) -> dict[str, object]:
    profile = host.model_profile
    if profile is None:
        return {
            "configured": host.configured,
            "profile": None,
        }
    return {
        "configured": host.configured,
        "profile": {
            "id": profile.id,
            "context_window_tokens": profile.context_window_tokens,
            "max_output_tokens": profile.max_output_tokens,
            "maximum_input_tokens": profile.maximum_input_tokens,
            "supports_tools": profile.supports_tools,
            "supports_parallel_tools": profile.supports_parallel_tools,
            "supports_structured_output": profile.supports_structured_output,
            "supports_streaming": profile.supports_streaming,
            "supports_reasoning": profile.supports_reasoning,
            "supports_vision": profile.supports_vision,
            "supports_documents": profile.supports_documents,
            "supports_prompt_caching": profile.supports_prompt_caching,
            "supports_native_continuation": profile.supports_native_continuation,
            "data_routing_classification": profile.data_routing_classification,
            "available": profile.available,
            "healthy": profile.healthy,
        },
    }


def _loop_exit_projection(value: LoopExit) -> dict[str, object]:
    return {
        "operation_id": value.operation_id,
        "kind": value.kind.value,
        "reason": value.reason,
        "created_at": _timestamp(value.created_at),
        "final_text": value.final_text,
        "post_operation_notices": list(value.post_operation_notices),
    }


def _scheduler_projection(value: MonitorSchedulerResult) -> dict[str, object]:
    return {
        "monitor_id": value.monitor_id,
        "occurrence_id": value.occurrence_id,
        "claimed": value.claimed,
        "reason": value.reason,
        "run_status": _enum(value.run_status),
        "operation_id": value.operation_id,
        "finding_id": value.finding_id,
    }


def _event_projection(value: CommittedEvent) -> dict[str, object]:
    event = value.event
    return {
        "sequence": value.cursor.sequence,
        "id": event.id,
        "type": event.type,
        "created_at": _timestamp(event.created_at),
        "operation_id": event.operation_id,
        "session_id": event.session_id,
        "turn_id": event.turn_id,
        "model_call_id": event.model_call_id,
        "call_id": event.call_id,
        "task_id": event.task_id,
        "evidence_id": event.evidence_id,
        "approval_id": event.approval_id,
        "monitor_id": event.monitor_id,
        "capability_id": event.capability_id,
        "executor_id": event.executor_id,
        "payload": _thaw_object(event.payload),
    }


def _verify_removable_socket(
    path: Path,
    *,
    expected_identity: tuple[int, int] | None,
) -> tuple[int, int]:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise LocalSocketSecurityError("could not inspect local socket") from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISSOCK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
    ):
        raise LocalSocketSecurityError("refusing to remove an unowned endpoint")
    identity = (metadata.st_dev, metadata.st_ino)
    if expected_identity is not None and identity != expected_identity:
        raise LocalSocketSecurityError("local socket identity changed")
    return identity


class LocalAgentServer:
    """Expose one already-admitted foreground host on a private local socket."""

    def __init__(
        self,
        host: AgentHost,
        *,
        request_read_timeout_seconds: float = _DEFAULT_REQUEST_READ_TIMEOUT_SECONDS,
    ) -> None:
        if not isinstance(host, AgentHost):
            raise TypeError("host must be an AgentHost")
        if (
            not isinstance(request_read_timeout_seconds, (int, float))
            or isinstance(request_read_timeout_seconds, bool)
            or not 0
            < float(request_read_timeout_seconds)
            <= _MAX_REQUEST_READ_TIMEOUT_SECONDS
        ):
            raise ValueError(
                "request_read_timeout_seconds must be finite, positive, and bounded"
            )
        self._host = host
        self._socket_path = local_socket_path(host.home)
        self._request_read_timeout_seconds = float(request_read_timeout_seconds)
        self._server: asyncio.AbstractServer | None = None
        self._socket_identity: tuple[int, int] | None = None
        self._connection_tasks: set[asyncio.Task[None]] = set()
        self._connection_writers: dict[asyncio.Task[None], asyncio.StreamWriter] = {}
        self._reading_tasks: set[asyncio.Task[None]] = set()
        self._starting = False
        self._stopping = False

    @property
    def socket_path(self) -> Path:
        return self._socket_path

    @property
    def started(self) -> bool:
        return self._server is not None and self._server.is_serving()

    @property
    def active_connections(self) -> int:
        return sum(not handler.done() for handler in self._connection_tasks)

    async def start(self) -> None:
        if self.started:
            return
        if self._starting or self._stopping:
            raise LocalAgentServerStateError("local server lifecycle is busy")
        if self._server is not None:
            raise LocalAgentServerStateError("local server cannot be restarted")
        self._starting = True
        host_started = False
        try:
            path = prepare_local_socket_path(self._host.home)
            self._socket_path = path
            try:
                _verify_removable_socket(path, expected_identity=None)
            except FileNotFoundError:
                pass
            else:
                os.unlink(path)
            await self._host.start()
            host_started = True
            try:
                self._server = await asyncio.start_unix_server(
                    self._accept_connection,
                    path=str(path),
                )
            except OSError as error:
                raise LocalSocketSecurityError(
                    "could not bind the local agent socket"
                ) from error
            metadata = os.lstat(path)
            self._socket_identity = _verify_removable_socket(
                path,
                expected_identity=(metadata.st_dev, metadata.st_ino),
            )
            secure_bound_local_socket(path)
        except BaseException:
            server = self._server
            self._server = None
            if server is not None:
                server.close()
                await server.wait_closed()
            try:
                await self._remove_owned_socket()
            finally:
                if host_started:
                    await self._host.stop(drain=False)
            raise
        finally:
            self._starting = False

    async def serve_forever(self) -> NoReturn:
        await self.start()
        server = self._server
        assert server is not None
        try:
            await server.serve_forever()
        finally:
            await self.stop(drain=True)
        raise AssertionError("Unix server returned without cancellation")

    async def stop(self, *, drain: bool = True) -> None:
        if self._stopping:
            return
        self._stopping = True
        try:
            server = self._server
            self._server = None
            if server is not None:
                server.close()
                await server.wait_closed()
            try:
                await self._settle_connections(drain=drain)
            finally:
                try:
                    await self._remove_owned_socket()
                finally:
                    await self._host.stop(drain=drain)
        finally:
            self._stopping = False

    async def dispatch(self, request: LocalRequest) -> LocalResponse:
        if not isinstance(request, LocalRequest):
            raise TypeError("request must be a LocalRequest")
        try:
            if request.method in _MUTATING_METHODS:
                if request.idempotency_key is None:
                    raise _RequestError(
                        "idempotency_required",
                        "mutation requires idempotency_key",
                    )
                await self._host.admit_mutation(
                    request.method,
                    request.params.to_dict(),
                    idempotency_key=request.idempotency_key,
                )
            result = await self._dispatch_result(request)
            return LocalSuccessResponse.create(
                request_id=request.request_id,
                result=result,
            )
        except _RequestError as error:
            return LocalErrorResponse.create(
                request_id=request.request_id,
                code=error.code,
                message=error.message,
                retryable=error.retryable,
                details=error.details,
            )
        except (OperationNotFoundError, MonitorNotFoundError):
            return self._error(request, "not_found", "requested record was not found")
        except MonitorProposalNotFoundError:
            return self._error(
                request,
                "not_found",
                "requested monitor proposal was not found",
            )
        except KeyError:
            return self._error(request, "not_found", "requested record was not found")
        except (
            HostInboxEnqueueConflictError,
            HostMutationConflictError,
            MonitorConflictError,
            MonitorProposalConflictError,
            MonitorTickClaimConflictError,
            OperationStateError,
        ):
            return self._error(
                request,
                "state_conflict",
                "request conflicts with durable state",
            )
        except (AgentHostStateError, AgentNotConfiguredError):
            return self._error(
                request,
                "host_unavailable",
                "agent host is not available for this request",
                retryable=True,
            )
        except (TypeError, ValueError):
            return self._error(
                request,
                "invalid_params",
                "request parameters are invalid",
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return self._error(
                request,
                "internal_error",
                "request could not be completed",
                retryable=True,
            )

    async def _dispatch_result(self, request: LocalRequest) -> object:
        method = request.method
        params = request.params.to_dict()
        key = request.idempotency_key

        if method in {"host.status", "host.health"}:
            _shape(params)
            status = await self._host.status()
            projected = _host_status_projection(status)
            if method == "host.health":
                return {
                    "healthy": status.state is AgentHostState.RUNNING,
                    "agent_id": status.agent_id,
                    "state": status.state.value,
                    "configured": status.configured,
                    "last_error": status.last_error,
                }
            return projected

        if method == "chat.submit":
            values = _shape(
                params,
                required=frozenset({"message"}),
                optional=frozenset({"session_id"}),
            )
            assert key is not None
            item = await self._host.submit(
                _text(values["message"], "message"),
                idempotency_key=key,
                session_id=_optional_text(
                    values.get("session_id"),
                    "session_id",
                    maximum_bytes=256,
                ),
            )
            return _inbox_projection(item)

        if method == "operation.inspect":
            values = _shape(params, required=frozenset({"operation_id"}))
            snapshot = await self._host.inspect_operation(
                _text(values["operation_id"], "operation_id", maximum_bytes=256)
            )
            return _operation_projection(snapshot)

        if method == "operation.cancel":
            values = _shape(
                params,
                required=frozenset({"operation_id"}),
                optional=frozenset({"reason"}),
            )
            cancelled_exit = await self._host.cancel(
                _text(values["operation_id"], "operation_id", maximum_bytes=256),
                reason=_text(
                    values.get("reason", "user_cancelled"),
                    "reason",
                    maximum_bytes=512,
                ),
            )
            return _loop_exit_projection(cancelled_exit)

        if method in {"approval.approve", "approval.reject"}:
            values = _shape(
                params,
                required=frozenset({"approval_id", "actor_id", "reason"}),
            )
            assert key is not None
            approval_action = (
                self._host.approve
                if method == "approval.approve"
                else self._host.reject
            )
            decision = await approval_action(
                _text(values["approval_id"], "approval_id", maximum_bytes=256),
                decided_by=_text(
                    values["actor_id"],
                    "actor_id",
                    maximum_bytes=256,
                ),
                reason=_text(values["reason"], "reason", maximum_bytes=2_000),
                idempotency_key=key,
            )
            return _approval_projection(decision)

        if method == "source.attach":
            values = _shape(
                params,
                required=frozenset({"kind", "path"}),
            )
            kind = _text(values["kind"], "kind", maximum_bytes=32)
            path = _text(values["path"], "path", maximum_bytes=2_048)
            if kind == "sqlite":
                source: ResourceSource = SQLiteSource(path)
            elif kind == "local_files":
                source = LocalDirectorySource(path)
            else:
                raise _RequestError(
                    "invalid_params",
                    "kind must be sqlite or local_files",
                )
            assert key is not None
            registration = await self._host.attach(
                source,
                idempotency_key=key,
            )
            return _source_projection(registration)

        if method == "model.status":
            _shape(params)
            return _model_projection(self._host)

        if method in {"events.read", "events.follow"}:
            values = _shape(
                params,
                optional=frozenset({"after", "limit"}),
            )
            after = values.get("after")
            cursor = None
            if after is not None:
                cursor = EventCursor(
                    self._host.id,
                    _integer(
                        after,
                        "after",
                        minimum=1,
                        maximum=9_223_372_036_854_775_807,
                    ),
                )
            limit = _integer(
                values.get("limit", _DEFAULT_EVENT_LIMIT),
                "limit",
                minimum=1,
                maximum=_MAX_LIST_LIMIT,
            )
            events = await self._host.read_events(cursor, limit=limit)
            return {
                "events": [_event_projection(value) for value in events],
                "next_after": (None if not events else events[-1].cursor.sequence),
                "follow": method == "events.follow",
                "streaming": False,
            }

        if method == "monitor.propose":
            values = _shape(
                params,
                required=frozenset({"monitor_id", "definition"}),
                optional=frozenset({"source_operation_id"}),
            )
            assert key is not None
            proposal = await self._host.propose_monitor(
                _text(values["monitor_id"], "monitor_id", maximum_bytes=256),
                _parse_definition(values["definition"]),
                idempotency_key=key,
                source_operation_id=_optional_text(
                    values.get("source_operation_id"),
                    "source_operation_id",
                    maximum_bytes=256,
                ),
            )
            return _proposal_projection(proposal)

        if method == "monitor.list":
            values = _shape(
                params,
                optional=frozenset({"statuses", "include_deleted", "limit"}),
            )
            statuses_value = values.get("statuses")
            statuses: tuple[MonitorStatus, ...] | None = None
            if statuses_value is not None:
                raw_statuses = _string_tuple(statuses_value, "statuses")
                if not raw_statuses:
                    raise _RequestError(
                        "invalid_params",
                        "statuses must not be empty",
                    )
                try:
                    statuses = tuple(MonitorStatus(value) for value in raw_statuses)
                except ValueError as error:
                    raise _RequestError(
                        "invalid_params",
                        "statuses contains an unsupported value",
                    ) from error
            monitors = await self._host.list_monitors(
                statuses=statuses,
                include_deleted=_boolean(
                    values.get("include_deleted", False),
                    "include_deleted",
                ),
                limit=_integer(
                    values.get("limit", 100),
                    "limit",
                    minimum=1,
                    maximum=_MAX_LIST_LIMIT,
                ),
            )
            return {"monitors": [_monitor_projection(value) for value in monitors]}

        if method == "monitor.inspect":
            values = _shape(params, required=frozenset({"monitor_id"}))
            inspection = await self._host.inspect_monitor(
                _text(values["monitor_id"], "monitor_id", maximum_bytes=256)
            )
            return _inspection_projection(inspection)

        if method == "monitor.confirm":
            values = _shape(
                params,
                required=frozenset(
                    {"proposal_id", "candidate_hash", "actor_id", "reason"}
                ),
            )
            inspection = await self._host.confirm_monitor(
                _text(values["proposal_id"], "proposal_id", maximum_bytes=256),
                candidate_hash=_text(
                    values["candidate_hash"],
                    "candidate_hash",
                    maximum_bytes=128,
                ),
                actor_id=_text(values["actor_id"], "actor_id", maximum_bytes=256),
                reason=_text(values["reason"], "reason", maximum_bytes=2_000),
            )
            return _inspection_projection(inspection)

        if method in {"monitor.pause", "monitor.resume", "monitor.delete"}:
            values = _shape(
                params,
                required=frozenset({"monitor_id", "actor_id", "reason"}),
                optional=frozenset({"operation_id"}),
            )
            assert key is not None
            monitor_action = {
                "monitor.pause": self._host.pause_monitor,
                "monitor.resume": self._host.resume_monitor,
                "monitor.delete": self._host.delete_monitor,
            }[method]
            inspection = await monitor_action(
                _text(values["monitor_id"], "monitor_id", maximum_bytes=256),
                actor_id=_text(values["actor_id"], "actor_id", maximum_bytes=256),
                reason=_text(values["reason"], "reason", maximum_bytes=2_000),
                idempotency_key=key,
                operation_id=_optional_text(
                    values.get("operation_id"),
                    "operation_id",
                    maximum_bytes=256,
                ),
            )
            return _inspection_projection(inspection)

        if method == "monitor.run_now":
            values = _shape(
                params,
                required=frozenset({"monitor_id"}),
                optional=frozenset({"lease_seconds"}),
            )
            lease_seconds = None
            if "lease_seconds" in values:
                lease_seconds = _positive_number(
                    values["lease_seconds"],
                    "lease_seconds",
                    maximum=300.0,
                )
            assert key is not None
            scheduler_result = await self._host.run_monitor_now(
                _text(values["monitor_id"], "monitor_id", maximum_bytes=256),
                idempotency_key=key,
                lease_seconds=lease_seconds,
            )
            return _scheduler_projection(scheduler_result)

        raise _RequestError("method_not_found", "request method is not supported")

    def _accept_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        if self._stopping:
            writer.close()
            return
        handler: asyncio.Task[None] = asyncio.ensure_future(
            self._handle_connection(reader, writer)
        )
        handler.set_name(f"daita-local-connection:{self._host.id}")
        self._connection_tasks.add(handler)
        self._connection_writers[handler] = writer
        self._reading_tasks.add(handler)
        handler.add_done_callback(self._connection_finished)

    def _connection_finished(self, handler: asyncio.Task[None]) -> None:
        self._connection_tasks.discard(handler)
        self._connection_writers.pop(handler, None)
        self._reading_tasks.discard(handler)
        if not handler.cancelled():
            handler.exception()

    async def _settle_connections(self, *, drain: bool) -> None:
        handlers = tuple(self._connection_tasks)
        writers = tuple(
            writer
            for handler in handlers
            if (writer := self._connection_writers.get(handler)) is not None
        )
        for handler in handlers:
            if not drain or handler in self._reading_tasks:
                writer = self._connection_writers.get(handler)
                if writer is not None:
                    writer.close()
                handler.cancel()
        if handlers:
            await asyncio.gather(*handlers, return_exceptions=True)
        for writer in writers:
            if not writer.is_closing():
                writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        request_id = "unknown"
        try:
            current = asyncio.current_task()
            try:
                async with asyncio.timeout(self._request_read_timeout_seconds):
                    request = await read_request(reader)
            finally:
                if current is not None:
                    self._reading_tasks.discard(current)
            request_id = request.request_id
            response = await self.dispatch(request)
            try:
                await write_response(writer, response)
            except LocalProtocolError:
                fallback = LocalErrorResponse.create(
                    request_id=request_id,
                    code="response_too_large",
                    message="response exceeds the local transport limit",
                )
                await write_frame(writer, encode_response(fallback))
        except TimeoutError:
            response = LocalErrorResponse.create(
                request_id=request_id,
                code="request_timeout",
                message="request frame was not received before the deadline",
                retryable=True,
            )
            try:
                await write_response(writer, response)
            except (ConnectionError, LocalProtocolError, OSError):
                pass
        except LocalProtocolError as error:
            response = LocalErrorResponse.create(
                request_id=request_id,
                code=error.code,
                message=error.message,
            )
            try:
                await write_response(writer, response)
            except (ConnectionError, LocalProtocolError, OSError):
                pass
        except asyncio.CancelledError:
            raise
        except Exception:
            response = LocalErrorResponse.create(
                request_id=request_id,
                code="internal_error",
                message="request could not be completed",
                retryable=True,
            )
            try:
                await write_response(writer, response)
            except (ConnectionError, LocalProtocolError, OSError):
                pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    def _error(
        self,
        request: LocalRequest,
        code: str,
        message: str,
        *,
        retryable: bool = False,
    ) -> LocalErrorResponse:
        return LocalErrorResponse.create(
            request_id=request.request_id,
            code=code,
            message=message,
            retryable=retryable,
        )

    async def _remove_owned_socket(self) -> None:
        identity = self._socket_identity
        if identity is None:
            return
        try:
            _verify_removable_socket(
                self._socket_path,
                expected_identity=identity,
            )
        except FileNotFoundError:
            pass
        else:
            os.unlink(self._socket_path)
        finally:
            self._socket_identity = None

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.stop(drain=True)


__all__ = [
    "LocalAgentServer",
    "LocalAgentServerStateError",
]
