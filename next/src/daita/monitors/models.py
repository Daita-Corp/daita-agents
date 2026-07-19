"""Canonical provider- and scheduler-neutral monitor records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from hashlib import sha256
import math
import re
from typing import TypeAlias

from .._json import FrozenJsonObject, canonical_json

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,127}\Z")
_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_OBJECTIVE_CHARS = 16_000
_MAX_SCOPE_IDS = 128
_MAX_CRON_SEARCH_MINUTES = 5 * 366 * 24 * 60


def _required_id(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a bounded stable identifier")
    return value


def _required_text(value: str, field_name: str, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized or len(normalized) > maximum:
        raise ValueError(f"{field_name} must contain 1 through {maximum} characters")
    return normalized


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{field_name} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must use UTC")
    return value.astimezone(timezone.utc)


def _optional_aware_utc(value: datetime | None, field_name: str) -> datetime | None:
    if value is None:
        return None
    return _aware_utc(value, field_name)


def _positive_int(value: int, field_name: str, *, maximum: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
        or value > maximum
    ):
        raise ValueError(f"{field_name} must be an integer from 1 through {maximum}")
    return value


def _nonnegative_int(value: int, field_name: str, *, maximum: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > maximum
    ):
        raise ValueError(f"{field_name} must be an integer from 0 through {maximum}")
    return value


def _stable_ids(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    resolved = tuple(values)
    if len(resolved) > _MAX_SCOPE_IDS:
        raise ValueError(f"{field_name} exceeds {_MAX_SCOPE_IDS} entries")
    if any(
        not isinstance(value, str) or _ID.fullmatch(value) is None for value in resolved
    ):
        raise ValueError(f"{field_name} must contain stable identifiers")
    if len(resolved) != len(set(resolved)) or resolved != tuple(sorted(resolved)):
        raise ValueError(f"{field_name} must be unique and sorted")
    return resolved


def _hash_json(value: Mapping[str, object]) -> str:
    return "sha256:" + sha256(canonical_json(value).encode("utf-8")).hexdigest()


class MonitorStatus(str, Enum):
    ENABLED = "enabled"
    PAUSED = "paused"
    DELETED = "deleted"


class CatchUpPolicy(str, Enum):
    ONCE = "once"


class MonitorConditionKind(str, Enum):
    ALWAYS = "always"
    EXPRESSION = "expression"
    THRESHOLD = "threshold"


class MonitorOccurrenceKind(str, Enum):
    SCHEDULED = "scheduled"
    RUN_NOW = "run_now"


class MonitorRunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class MonitorFindingSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class MonitorConfirmationDecision(str, Enum):
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class MonitorLifecycleAction(str, Enum):
    ACTIVATE = "activate"
    UPDATE = "update"
    PAUSE = "pause"
    RESUME = "resume"
    DELETE = "delete"
    RUN_NOW = "run_now"


@dataclass(frozen=True, slots=True)
class IntervalSchedule:
    interval_seconds: int
    anchor_at: datetime

    def __post_init__(self) -> None:
        _positive_int(
            self.interval_seconds,
            "interval_seconds",
            maximum=366 * 24 * 60 * 60,
        )
        object.__setattr__(self, "anchor_at", _aware_utc(self.anchor_at, "anchor_at"))

    def next_due_at(self, after: datetime) -> datetime:
        after = _aware_utc(after, "after")
        if after < self.anchor_at:
            return self.anchor_at
        elapsed = (after - self.anchor_at).total_seconds()
        steps = math.floor(elapsed / self.interval_seconds) + 1
        return self.anchor_at + timedelta(seconds=steps * self.interval_seconds)


def _parse_cron_atom(value: str, *, minimum: int, maximum: int) -> int:
    if not value.isascii() or not value.isdigit():
        raise ValueError("cron fields must contain only bounded numeric syntax")
    parsed = int(value)
    if maximum == 6 and parsed == 7:
        return 0
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"cron value must be from {minimum} through {maximum}")
    return parsed


def _parse_cron_field(
    source: str,
    *,
    minimum: int,
    maximum: int,
) -> tuple[frozenset[int], bool]:
    if not source or len(source) > 64:
        raise ValueError("cron field is empty or too long")
    wildcard = source == "*"
    values: set[int] = set()
    for component in source.split(","):
        if not component:
            raise ValueError("cron list cannot contain an empty component")
        base, separator, step_text = component.partition("/")
        step = 1
        if separator:
            step = _parse_cron_atom(step_text, minimum=1, maximum=maximum + 1)
        if base == "*":
            start, stop = minimum, maximum
        elif "-" in base:
            start_text, dash, stop_text = base.partition("-")
            if not dash or "-" in stop_text:
                raise ValueError("cron range is malformed")
            start = _parse_cron_atom(start_text, minimum=minimum, maximum=maximum)
            stop = _parse_cron_atom(stop_text, minimum=minimum, maximum=maximum)
            if start > stop:
                raise ValueError("cron ranges must be ascending")
        else:
            if separator:
                raise ValueError("cron step requires '*' or a range")
            start = stop = _parse_cron_atom(
                base,
                minimum=minimum,
                maximum=maximum,
            )
        values.update(range(start, stop + 1, step))
    if not values:
        raise ValueError("cron field selects no values")
    return frozenset(values), wildcard


@dataclass(frozen=True, slots=True)
class CronSchedule:
    expression: str
    timezone_name: str = "UTC"
    _minutes: frozenset[int] = field(init=False, repr=False, compare=False)
    _hours: frozenset[int] = field(init=False, repr=False, compare=False)
    _days: frozenset[int] = field(init=False, repr=False, compare=False)
    _months: frozenset[int] = field(init=False, repr=False, compare=False)
    _weekdays: frozenset[int] = field(init=False, repr=False, compare=False)
    _day_wildcard: bool = field(init=False, repr=False, compare=False)
    _weekday_wildcard: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        expression = _required_text(self.expression, "cron expression", maximum=320)
        if expression != " ".join(expression.split()):
            raise ValueError("cron expression must use canonical single spacing")
        if self.timezone_name != "UTC":
            raise ValueError("the local MVP supports UTC cron schedules only")
        fields = expression.split(" ")
        if len(fields) != 5:
            raise ValueError("cron expression must contain exactly five fields")
        minutes, _ = _parse_cron_field(fields[0], minimum=0, maximum=59)
        hours, _ = _parse_cron_field(fields[1], minimum=0, maximum=23)
        days, day_wildcard = _parse_cron_field(fields[2], minimum=1, maximum=31)
        months, _ = _parse_cron_field(fields[3], minimum=1, maximum=12)
        weekdays, weekday_wildcard = _parse_cron_field(fields[4], minimum=0, maximum=6)
        object.__setattr__(self, "expression", expression)
        object.__setattr__(self, "_minutes", minutes)
        object.__setattr__(self, "_hours", hours)
        object.__setattr__(self, "_days", days)
        object.__setattr__(self, "_months", months)
        object.__setattr__(self, "_weekdays", weekdays)
        object.__setattr__(self, "_day_wildcard", day_wildcard)
        object.__setattr__(self, "_weekday_wildcard", weekday_wildcard)

    def _matches(self, candidate: datetime) -> bool:
        if candidate.minute not in self._minutes or candidate.hour not in self._hours:
            return False
        if candidate.month not in self._months:
            return False
        day_match = candidate.day in self._days
        # Python Monday is 0; cron Sunday is 0.
        weekday_match = ((candidate.weekday() + 1) % 7) in self._weekdays
        if self._day_wildcard:
            date_match = weekday_match
        elif self._weekday_wildcard:
            date_match = day_match
        else:
            date_match = day_match or weekday_match
        return date_match

    def next_due_at(self, after: datetime) -> datetime:
        after = _aware_utc(after, "after")
        candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
        for _ in range(_MAX_CRON_SEARCH_MINUTES):
            if self._matches(candidate):
                return candidate
            candidate += timedelta(minutes=1)
        raise ValueError("cron schedule has no occurrence within the bounded horizon")


MonitorSchedule: TypeAlias = IntervalSchedule | CronSchedule


def advance_next_due_at(
    schedule: MonitorSchedule,
    *,
    completed_at: datetime,
    catch_up: CatchUpPolicy,
) -> datetime:
    if not isinstance(schedule, (IntervalSchedule, CronSchedule)):
        raise TypeError("schedule must be an IntervalSchedule or CronSchedule")
    if catch_up is not CatchUpPolicy.ONCE:
        raise ValueError("only catch_up=once is supported")
    return schedule.next_due_at(completed_at)


@dataclass(frozen=True, slots=True)
class MonitorScope:
    source_ids: tuple[str, ...] = ()
    resource_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_ids",
            _stable_ids(self.source_ids, "monitor source_ids"),
        )
        object.__setattr__(
            self,
            "resource_ids",
            _stable_ids(self.resource_ids, "monitor resource_ids"),
        )
        if self.resource_ids and not self.source_ids:
            raise ValueError("resource scope requires at least one source_id")


@dataclass(frozen=True, slots=True)
class MonitorCondition:
    kind: MonitorConditionKind = MonitorConditionKind.ALWAYS
    expression: str | None = None
    configuration: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, MonitorConditionKind):
            raise TypeError("monitor condition kind must be MonitorConditionKind")
        expression = self.expression
        if self.kind is MonitorConditionKind.ALWAYS:
            if expression is not None or self.configuration:
                raise ValueError("always conditions cannot contain expression data")
        else:
            if expression is None:
                raise ValueError("expression and threshold conditions require text")
            expression = _required_text(
                expression,
                "monitor condition expression",
                maximum=4_000,
            )
        object.__setattr__(self, "expression", expression)
        object.__setattr__(
            self,
            "configuration",
            FrozenJsonObject.from_mapping(self.configuration),
        )


@dataclass(frozen=True, slots=True)
class MonitorBudgetOverrides:
    max_turns: int | None = None
    max_capability_calls: int | None = None
    max_wall_time_seconds: int | None = None

    def __post_init__(self) -> None:
        for field_name, value, maximum in (
            ("max_turns", self.max_turns, 1_000),
            ("max_capability_calls", self.max_capability_calls, 10_000),
            ("max_wall_time_seconds", self.max_wall_time_seconds, 86_400),
        ):
            if value is not None:
                _positive_int(value, field_name, maximum=maximum)


@dataclass(frozen=True, slots=True)
class MonitorTimingPolicy:
    catch_up: CatchUpPolicy = CatchUpPolicy.ONCE
    cooldown_seconds: int = 0
    initial_backoff_seconds: int = 1
    max_backoff_seconds: int = 300
    backoff_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.catch_up is not CatchUpPolicy.ONCE:
            raise ValueError("only catch_up=once is supported")
        _nonnegative_int(
            self.cooldown_seconds,
            "cooldown_seconds",
            maximum=366 * 24 * 60 * 60,
        )
        _positive_int(
            self.initial_backoff_seconds,
            "initial_backoff_seconds",
            maximum=86_400,
        )
        _positive_int(
            self.max_backoff_seconds,
            "max_backoff_seconds",
            maximum=366 * 24 * 60 * 60,
        )
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("max_backoff_seconds cannot be below initial backoff")
        if (
            not isinstance(self.backoff_multiplier, (int, float))
            or isinstance(self.backoff_multiplier, bool)
            or not math.isfinite(float(self.backoff_multiplier))
            or float(self.backoff_multiplier) < 1.0
            or float(self.backoff_multiplier) > 100.0
        ):
            raise ValueError("backoff_multiplier must be finite from 1 through 100")
        object.__setattr__(self, "backoff_multiplier", float(self.backoff_multiplier))


@dataclass(frozen=True, slots=True)
class MonitorDefinition:
    name: str
    objective: str
    scope: MonitorScope
    schedule: MonitorSchedule
    condition: MonitorCondition = MonitorCondition()
    budget_overrides: MonitorBudgetOverrides = MonitorBudgetOverrides()
    timing: MonitorTimingPolicy = MonitorTimingPolicy()
    policy_overrides: Mapping[str, object] = field(default_factory=dict)
    operation_template: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name.strip()) is None:
            raise ValueError("monitor name must be bounded and human-readable")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(
            self,
            "objective",
            _required_text(
                self.objective,
                "monitor objective",
                maximum=_MAX_OBJECTIVE_CHARS,
            ),
        )
        if not isinstance(self.scope, MonitorScope):
            raise TypeError("monitor scope must be MonitorScope")
        if not isinstance(self.schedule, (IntervalSchedule, CronSchedule)):
            raise TypeError("monitor schedule must be interval or cron")
        if not isinstance(self.condition, MonitorCondition):
            raise TypeError("monitor condition must be MonitorCondition")
        if not isinstance(self.budget_overrides, MonitorBudgetOverrides):
            raise TypeError("monitor budget_overrides must be MonitorBudgetOverrides")
        if not isinstance(self.timing, MonitorTimingPolicy):
            raise TypeError("monitor timing must be MonitorTimingPolicy")
        object.__setattr__(
            self,
            "policy_overrides",
            FrozenJsonObject.from_mapping(self.policy_overrides),
        )
        object.__setattr__(
            self,
            "operation_template",
            FrozenJsonObject.from_mapping(self.operation_template),
        )

    @property
    def content_hash(self) -> str:
        if isinstance(self.schedule, IntervalSchedule):
            schedule: dict[str, object] = {
                "kind": "interval",
                "interval_seconds": self.schedule.interval_seconds,
                "anchor_at": self.schedule.anchor_at.isoformat(),
            }
        else:
            schedule = {
                "kind": "cron",
                "expression": self.schedule.expression,
                "timezone": self.schedule.timezone_name,
            }
        return _hash_json(
            {
                "budget_overrides": {
                    "max_capability_calls": self.budget_overrides.max_capability_calls,
                    "max_turns": self.budget_overrides.max_turns,
                    "max_wall_time_seconds": self.budget_overrides.max_wall_time_seconds,
                },
                "condition": {
                    "configuration": self.condition.configuration,
                    "expression": self.condition.expression,
                    "kind": self.condition.kind.value,
                },
                "name": self.name,
                "objective": self.objective,
                "operation_template": self.operation_template,
                "policy_overrides": self.policy_overrides,
                "schedule": schedule,
                "scope": {
                    "resource_ids": self.scope.resource_ids,
                    "source_ids": self.scope.source_ids,
                },
                "timing": {
                    "backoff_multiplier": self.timing.backoff_multiplier,
                    "catch_up": self.timing.catch_up.value,
                    "cooldown_seconds": self.timing.cooldown_seconds,
                    "initial_backoff_seconds": self.timing.initial_backoff_seconds,
                    "max_backoff_seconds": self.timing.max_backoff_seconds,
                },
            }
        )


@dataclass(frozen=True, slots=True)
class MonitorProposal:
    id: str
    agent_id: str
    intended_monitor_id: str
    idempotency_key: str
    candidate: MonitorDefinition
    candidate_hash: str
    created_at: datetime
    source_operation_id: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("proposal id", self.id),
            ("proposal agent_id", self.agent_id),
            ("intended monitor_id", self.intended_monitor_id),
            ("proposal idempotency_key", self.idempotency_key),
        ):
            _required_id(value, field_name)
        if not isinstance(self.candidate, MonitorDefinition):
            raise TypeError("monitor proposal candidate must be MonitorDefinition")
        if self.candidate_hash != self.candidate.content_hash:
            raise ValueError("monitor proposal candidate_hash does not match candidate")
        if self.source_operation_id is not None:
            _required_id(self.source_operation_id, "proposal source_operation_id")
        object.__setattr__(
            self,
            "created_at",
            _aware_utc(self.created_at, "proposal created_at"),
        )


@dataclass(frozen=True, slots=True)
class MonitorConfirmation:
    id: str
    agent_id: str
    proposal_id: str
    decision: MonitorConfirmationDecision
    candidate_hash: str
    actor_id: str
    reason: str
    decided_at: datetime
    resulting_monitor_id: str | None = None
    resulting_version_id: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("confirmation id", self.id),
            ("confirmation agent_id", self.agent_id),
            ("confirmation proposal_id", self.proposal_id),
            ("confirmation actor_id", self.actor_id),
        ):
            _required_id(value, field_name)
        if not isinstance(self.decision, MonitorConfirmationDecision):
            raise TypeError("confirmation decision must be MonitorConfirmationDecision")
        if (
            not isinstance(self.candidate_hash, str)
            or _HASH.fullmatch(self.candidate_hash) is None
        ):
            raise ValueError("confirmation candidate_hash must be canonical SHA-256")
        object.__setattr__(
            self,
            "reason",
            _required_text(self.reason, "confirmation reason", maximum=2_000),
        )
        object.__setattr__(
            self,
            "decided_at",
            _aware_utc(self.decided_at, "confirmation decided_at"),
        )
        results = (self.resulting_monitor_id, self.resulting_version_id)
        if self.decision is MonitorConfirmationDecision.CONFIRMED:
            if any(value is None for value in results):
                raise ValueError("confirmed proposal requires monitor and version IDs")
        elif any(value is not None for value in results):
            raise ValueError("rejected proposal cannot identify an activated monitor")
        for result_field_name, result_value in (
            ("resulting_monitor_id", self.resulting_monitor_id),
            ("resulting_version_id", self.resulting_version_id),
        ):
            if result_value is not None:
                _required_id(result_value, result_field_name)


@dataclass(frozen=True, slots=True)
class Monitor:
    id: str
    agent_id: str
    status: MonitorStatus
    current_version: int
    revision: int
    created_at: datetime
    updated_at: datetime
    paused_at: datetime | None = None
    deleted_at: datetime | None = None

    def __post_init__(self) -> None:
        _required_id(self.id, "monitor id")
        _required_id(self.agent_id, "monitor agent_id")
        if not isinstance(self.status, MonitorStatus):
            raise TypeError("monitor status must be MonitorStatus")
        _positive_int(self.current_version, "current_version", maximum=1_000_000)
        _positive_int(self.revision, "monitor revision", maximum=2_000_000_000)
        created = _aware_utc(self.created_at, "monitor created_at")
        updated = _aware_utc(self.updated_at, "monitor updated_at")
        paused = _optional_aware_utc(self.paused_at, "monitor paused_at")
        deleted = _optional_aware_utc(self.deleted_at, "monitor deleted_at")
        if updated < created:
            raise ValueError("monitor updated_at cannot precede created_at")
        if self.status is MonitorStatus.ENABLED and (
            paused is not None or deleted is not None
        ):
            raise ValueError("enabled monitor cannot have pause/delete timestamps")
        if self.status is MonitorStatus.PAUSED and (
            paused is None or deleted is not None
        ):
            raise ValueError("paused monitor requires only paused_at")
        if self.status is MonitorStatus.DELETED and deleted is None:
            raise ValueError("deleted monitor requires deleted_at")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)
        object.__setattr__(self, "paused_at", paused)
        object.__setattr__(self, "deleted_at", deleted)


@dataclass(frozen=True, slots=True)
class MonitorVersion:
    id: str
    agent_id: str
    monitor_id: str
    version: int
    definition: MonitorDefinition
    content_hash: str
    proposal_id: str
    created_at: datetime
    source_operation_id: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("monitor version id", self.id),
            ("monitor version agent_id", self.agent_id),
            ("monitor version monitor_id", self.monitor_id),
            ("monitor version proposal_id", self.proposal_id),
        ):
            _required_id(value, field_name)
        _positive_int(self.version, "monitor version", maximum=1_000_000)
        if not isinstance(self.definition, MonitorDefinition):
            raise TypeError("monitor version definition must be MonitorDefinition")
        if self.content_hash != self.definition.content_hash:
            raise ValueError("monitor version content_hash does not match definition")
        if self.source_operation_id is not None:
            _required_id(self.source_operation_id, "version source_operation_id")
        object.__setattr__(
            self,
            "created_at",
            _aware_utc(self.created_at, "monitor version created_at"),
        )


@dataclass(frozen=True, slots=True)
class MonitorLifecycleRecord:
    id: str
    agent_id: str
    monitor_id: str
    action: MonitorLifecycleAction
    from_status: MonitorStatus | None
    to_status: MonitorStatus
    from_revision: int
    to_revision: int
    monitor_version: int
    actor_id: str
    reason: str
    idempotency_key: str
    occurred_at: datetime
    operation_id: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("lifecycle id", self.id),
            ("lifecycle agent_id", self.agent_id),
            ("lifecycle monitor_id", self.monitor_id),
            ("lifecycle actor_id", self.actor_id),
            ("lifecycle idempotency_key", self.idempotency_key),
        ):
            _required_id(value, field_name)
        if not isinstance(self.action, MonitorLifecycleAction):
            raise TypeError("lifecycle action must be MonitorLifecycleAction")
        if self.from_status is not None and not isinstance(
            self.from_status, MonitorStatus
        ):
            raise TypeError("lifecycle from_status must be MonitorStatus or None")
        if not isinstance(self.to_status, MonitorStatus):
            raise TypeError("lifecycle to_status must be MonitorStatus")
        _nonnegative_int(self.from_revision, "from_revision", maximum=2_000_000_000)
        _positive_int(self.to_revision, "to_revision", maximum=2_000_000_000)
        if self.to_revision != self.from_revision + 1:
            raise ValueError("lifecycle revision must advance exactly once")
        _positive_int(self.monitor_version, "monitor_version", maximum=1_000_000)
        object.__setattr__(
            self,
            "reason",
            _required_text(self.reason, "lifecycle reason", maximum=2_000),
        )
        if self.operation_id is not None:
            _required_id(self.operation_id, "lifecycle operation_id")
        object.__setattr__(
            self,
            "occurred_at",
            _aware_utc(self.occurred_at, "lifecycle occurred_at"),
        )


@dataclass(frozen=True, slots=True)
class MonitorScheduleState:
    agent_id: str
    monitor_id: str
    revision: int
    next_scheduled_at: datetime | None
    updated_at: datetime
    last_scheduled_at: datetime | None = None
    cooldown_until: datetime | None = None
    backoff_until: datetime | None = None
    consecutive_failures: int = 0
    consecutive_matches: int = 0
    checkpoint_version: int = 0
    last_occurrence_id: str | None = None
    last_run_id: str | None = None
    last_operation_id: str | None = None

    def __post_init__(self) -> None:
        _required_id(self.agent_id, "schedule state agent_id")
        _required_id(self.monitor_id, "schedule state monitor_id")
        _positive_int(self.revision, "schedule state revision", maximum=2_000_000_000)
        for timestamp_field_name, timestamp_value in (
            ("next_scheduled_at", self.next_scheduled_at),
            ("last_scheduled_at", self.last_scheduled_at),
            ("cooldown_until", self.cooldown_until),
            ("backoff_until", self.backoff_until),
        ):
            object.__setattr__(
                self,
                timestamp_field_name,
                _optional_aware_utc(timestamp_value, timestamp_field_name),
            )
        object.__setattr__(
            self,
            "updated_at",
            _aware_utc(self.updated_at, "updated_at"),
        )
        _nonnegative_int(
            self.consecutive_failures,
            "consecutive_failures",
            maximum=1_000_000,
        )
        _nonnegative_int(
            self.consecutive_matches,
            "consecutive_matches",
            maximum=1_000_000,
        )
        _nonnegative_int(
            self.checkpoint_version,
            "checkpoint_version",
            maximum=1_000_000,
        )
        for identity_field_name, identity_value in (
            ("last_occurrence_id", self.last_occurrence_id),
            ("last_run_id", self.last_run_id),
            ("last_operation_id", self.last_operation_id),
        ):
            if identity_value is not None:
                _required_id(identity_value, identity_field_name)


def monitor_occurrence_key(
    *,
    agent_id: str,
    monitor_id: str,
    monitor_version: int,
    kind: MonitorOccurrenceKind,
    scheduled_for: datetime,
    manual_key: str | None = None,
) -> str:
    _required_id(agent_id, "occurrence agent_id")
    _required_id(monitor_id, "occurrence monitor_id")
    _positive_int(monitor_version, "occurrence monitor_version", maximum=1_000_000)
    if not isinstance(kind, MonitorOccurrenceKind):
        raise TypeError("occurrence kind must be MonitorOccurrenceKind")
    scheduled_for = _aware_utc(scheduled_for, "occurrence scheduled_for")
    if kind is MonitorOccurrenceKind.RUN_NOW:
        if manual_key is None:
            raise ValueError("run-now occurrence requires manual_key")
        _required_id(manual_key, "occurrence manual_key")
    elif manual_key is not None:
        raise ValueError("scheduled occurrence cannot contain manual_key")
    return _hash_json(
        {
            "agent_id": agent_id,
            "kind": kind.value,
            "manual_key": manual_key,
            "monitor_id": monitor_id,
            "monitor_version": monitor_version,
            "scheduled_for": (
                scheduled_for.isoformat()
                if kind is MonitorOccurrenceKind.SCHEDULED
                else None
            ),
        }
    )


def monitor_occurrence_id(occurrence_key: str) -> str:
    if not isinstance(occurrence_key, str) or _HASH.fullmatch(occurrence_key) is None:
        raise ValueError("occurrence_key must be a canonical SHA-256 hash")
    return "monitor-occurrence-" + occurrence_key.removeprefix("sha256:")


def monitor_trigger_id(occurrence_key: str) -> str:
    if not isinstance(occurrence_key, str) or _HASH.fullmatch(occurrence_key) is None:
        raise ValueError("occurrence_key must be a canonical SHA-256 hash")
    return "monitor-trigger-" + occurrence_key.removeprefix("sha256:")


def monitor_run_id(occurrence_key: str) -> str:
    if not isinstance(occurrence_key, str) or _HASH.fullmatch(occurrence_key) is None:
        raise ValueError("occurrence_key must be a canonical SHA-256 hash")
    return "monitor-run-" + occurrence_key.removeprefix("sha256:")


@dataclass(frozen=True, slots=True)
class MonitorOccurrence:
    id: str
    agent_id: str
    monitor_id: str
    monitor_version: int
    kind: MonitorOccurrenceKind
    scheduled_for: datetime
    occurrence_key: str
    trigger_id: str
    run_id: str
    created_at: datetime
    manual_key: str | None = None

    def __post_init__(self) -> None:
        expected_key = monitor_occurrence_key(
            agent_id=self.agent_id,
            monitor_id=self.monitor_id,
            monitor_version=self.monitor_version,
            kind=self.kind,
            scheduled_for=self.scheduled_for,
            manual_key=self.manual_key,
        )
        if self.occurrence_key != expected_key:
            raise ValueError("occurrence_key does not match occurrence identity")
        if self.id != monitor_occurrence_id(expected_key):
            raise ValueError("occurrence id does not match occurrence_key")
        if self.trigger_id != monitor_trigger_id(expected_key):
            raise ValueError("trigger_id does not match occurrence_key")
        if self.run_id != monitor_run_id(expected_key):
            raise ValueError("run_id does not match occurrence_key")
        object.__setattr__(
            self,
            "scheduled_for",
            _aware_utc(self.scheduled_for, "occurrence scheduled_for"),
        )
        object.__setattr__(
            self,
            "created_at",
            _aware_utc(self.created_at, "occurrence created_at"),
        )


@dataclass(frozen=True, slots=True)
class MonitorTickLease:
    id: str
    agent_id: str
    monitor_id: str
    occurrence_id: str
    holder_id: str
    fencing_token: int
    claimed_at: datetime
    expires_at: datetime
    released_at: datetime | None = None
    release_reason: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("lease id", self.id),
            ("lease agent_id", self.agent_id),
            ("lease monitor_id", self.monitor_id),
            ("lease occurrence_id", self.occurrence_id),
            ("lease holder_id", self.holder_id),
        ):
            _required_id(value, field_name)
        _positive_int(self.fencing_token, "lease fencing_token", maximum=2_000_000_000)
        claimed = _aware_utc(self.claimed_at, "lease claimed_at")
        expires = _aware_utc(self.expires_at, "lease expires_at")
        released = _optional_aware_utc(self.released_at, "lease released_at")
        if expires <= claimed:
            raise ValueError("lease expires_at must follow claimed_at")
        if (released is None) != (self.release_reason is None):
            raise ValueError("lease release time and reason must be set together")
        if released is not None:
            if released < claimed:
                raise ValueError("lease released_at cannot precede claimed_at")
            object.__setattr__(
                self,
                "release_reason",
                _required_text(
                    self.release_reason or "", "release_reason", maximum=512
                ),
            )
        object.__setattr__(self, "claimed_at", claimed)
        object.__setattr__(self, "expires_at", expires)
        object.__setattr__(self, "released_at", released)


@dataclass(frozen=True, slots=True)
class MonitorTickLeaseGuard:
    agent_id: str
    monitor_id: str
    occurrence_id: str
    holder_id: str
    fencing_token: int

    def __post_init__(self) -> None:
        for field_name, value in (
            ("guard agent_id", self.agent_id),
            ("guard monitor_id", self.monitor_id),
            ("guard occurrence_id", self.occurrence_id),
            ("guard holder_id", self.holder_id),
        ):
            _required_id(value, field_name)
        _positive_int(self.fencing_token, "guard fencing_token", maximum=2_000_000_000)


@dataclass(frozen=True, slots=True)
class MonitorRun:
    id: str
    agent_id: str
    monitor_id: str
    occurrence_id: str
    trigger_id: str
    attempt: int
    fencing_token: int
    status: MonitorRunStatus
    started_at: datetime
    operation_id: str | None = None
    completed_at: datetime | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("run id", self.id),
            ("run agent_id", self.agent_id),
            ("run monitor_id", self.monitor_id),
            ("run occurrence_id", self.occurrence_id),
            ("run trigger_id", self.trigger_id),
        ):
            _required_id(value, field_name)
        _positive_int(self.attempt, "run attempt", maximum=1_000_000)
        _positive_int(self.fencing_token, "run fencing_token", maximum=2_000_000_000)
        if not isinstance(self.status, MonitorRunStatus):
            raise TypeError("run status must be MonitorRunStatus")
        started = _aware_utc(self.started_at, "run started_at")
        completed = _optional_aware_utc(self.completed_at, "run completed_at")
        terminal = self.status in {
            MonitorRunStatus.SUCCEEDED,
            MonitorRunStatus.FAILED,
            MonitorRunStatus.CANCELLED,
            MonitorRunStatus.SKIPPED,
        }
        if terminal != (completed is not None):
            raise ValueError("terminal monitor run status must match completed_at")
        if completed is not None and completed < started:
            raise ValueError("run completed_at cannot precede started_at")
        if self.failure_reason is not None:
            object.__setattr__(
                self,
                "failure_reason",
                _required_text(
                    self.failure_reason, "run failure_reason", maximum=2_000
                ),
            )
        if self.operation_id is not None:
            _required_id(self.operation_id, "run operation_id")
        object.__setattr__(self, "started_at", started)
        object.__setattr__(self, "completed_at", completed)


@dataclass(frozen=True, slots=True)
class MonitorFinding:
    id: str
    agent_id: str
    monitor_id: str
    occurrence_id: str
    run_id: str
    operation_id: str
    evidence_id: str
    severity: MonitorFindingSeverity
    summary: str
    details: Mapping[str, object]
    dedupe_key: str
    created_at: datetime

    def __post_init__(self) -> None:
        for field_name, value in (
            ("finding id", self.id),
            ("finding agent_id", self.agent_id),
            ("finding monitor_id", self.monitor_id),
            ("finding occurrence_id", self.occurrence_id),
            ("finding run_id", self.run_id),
            ("finding operation_id", self.operation_id),
            ("finding evidence_id", self.evidence_id),
            ("finding dedupe_key", self.dedupe_key),
        ):
            _required_id(value, field_name)
        if not isinstance(self.severity, MonitorFindingSeverity):
            raise TypeError("finding severity must be MonitorFindingSeverity")
        object.__setattr__(
            self,
            "summary",
            _required_text(self.summary, "finding summary", maximum=4_000),
        )
        object.__setattr__(self, "details", FrozenJsonObject.from_mapping(self.details))
        object.__setattr__(
            self,
            "created_at",
            _aware_utc(self.created_at, "finding created_at"),
        )


@dataclass(frozen=True, slots=True)
class MonitorCheckpoint:
    id: str
    agent_id: str
    monitor_id: str
    version: int
    run_id: str
    cursor: Mapping[str, object]
    cursor_hash: str
    created_at: datetime
    previous_version: int | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("checkpoint id", self.id),
            ("checkpoint agent_id", self.agent_id),
            ("checkpoint monitor_id", self.monitor_id),
            ("checkpoint run_id", self.run_id),
        ):
            _required_id(value, field_name)
        _positive_int(self.version, "checkpoint version", maximum=1_000_000)
        cursor = FrozenJsonObject.from_mapping(self.cursor)
        expected_hash = _hash_json(cursor)
        if self.cursor_hash != expected_hash:
            raise ValueError("checkpoint cursor_hash does not match cursor")
        if self.version == 1:
            if self.previous_version is not None:
                raise ValueError("first checkpoint cannot have previous_version")
        elif self.previous_version != self.version - 1:
            raise ValueError("checkpoint must link the immediately previous version")
        object.__setattr__(self, "cursor", cursor)
        object.__setattr__(
            self,
            "created_at",
            _aware_utc(self.created_at, "checkpoint created_at"),
        )


@dataclass(frozen=True, slots=True)
class MonitorInspection:
    monitor: Monitor
    versions: tuple[MonitorVersion, ...]
    lifecycle: tuple[MonitorLifecycleRecord, ...]
    schedule_state: MonitorScheduleState
    proposals: tuple[MonitorProposal, ...] = ()
    confirmations: tuple[MonitorConfirmation, ...] = ()
    occurrences: tuple[MonitorOccurrence, ...] = ()
    leases: tuple[MonitorTickLease, ...] = ()
    runs: tuple[MonitorRun, ...] = ()
    findings: tuple[MonitorFinding, ...] = ()
    checkpoints: tuple[MonitorCheckpoint, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.monitor, Monitor):
            raise TypeError("inspection monitor must be Monitor")
        if not isinstance(self.schedule_state, MonitorScheduleState):
            raise TypeError("inspection schedule_state must be MonitorScheduleState")
        if (
            self.schedule_state.agent_id != self.monitor.agent_id
            or self.schedule_state.monitor_id != self.monitor.id
        ):
            raise ValueError("inspection schedule state does not match monitor")
        collections: tuple[tuple[str, tuple[object, ...], type[object]], ...] = (
            ("versions", tuple(self.versions), MonitorVersion),
            ("lifecycle", tuple(self.lifecycle), MonitorLifecycleRecord),
            ("proposals", tuple(self.proposals), MonitorProposal),
            ("confirmations", tuple(self.confirmations), MonitorConfirmation),
            ("occurrences", tuple(self.occurrences), MonitorOccurrence),
            ("leases", tuple(self.leases), MonitorTickLease),
            ("runs", tuple(self.runs), MonitorRun),
            ("findings", tuple(self.findings), MonitorFinding),
            ("checkpoints", tuple(self.checkpoints), MonitorCheckpoint),
        )
        for field_name, values, expected_type in collections:
            if any(not isinstance(value, expected_type) for value in values):
                raise TypeError(f"inspection {field_name} contain invalid records")
            object.__setattr__(self, field_name, values)
        if (
            not self.versions
            or self.versions[-1].version != self.monitor.current_version
        ):
            raise ValueError("inspection must contain the current monitor version")
