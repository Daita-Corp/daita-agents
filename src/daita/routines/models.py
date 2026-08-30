"""Strict bounded values for the accepted D1 scheduled-routine slice."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from hashlib import sha256
from typing import TypeAlias

from .._json import canonical_json
from ..capabilities import AccessMode, ExecutionScope, OperationalEffect
from ..distribution.models import (
    MAX_DISTRIBUTION_TARGETS,
    DistributionPlan,
    OutcomeContract,
)
from ..llm.models import ModelSensitivity

SCHEDULE_INTERPRETER_REVISION = 1

MAX_SCHEDULED_ROUTINES_PER_AGENT = 128
MAX_ACTIVE_ROUTINES_PER_AGENT = 32
MAX_ROUTINE_OCCURRENCES = 10_000
MAX_ROUTINE_ATTEMPTS = 3
MAX_ROUTINE_CONSECUTIVE_FAILURES = 10
MAX_ROUTINE_LIST_PAGE_SIZE = 50
MAX_ROUTINE_HISTORY_PAGE_SIZE = 100
MAX_ROUTINE_INSTRUCTION_BYTES = 16 * 1024
MAX_ROUTINE_TITLE_CHARACTERS = 256
MAX_ROUTINE_IDENTITY_ITEMS = 64
MAX_ROUTINE_IDENTITY_CHARACTERS = 1_024
MAX_ROUTINE_SKILL_BINDINGS = 8
MAX_ROUTINE_LIFETIME = timedelta(days=366 * 5)
MIN_ROUTINE_INTERVAL_SECONDS = 60
MAX_ROUTINE_INTERVAL_SECONDS = int(MAX_ROUTINE_LIFETIME.total_seconds())
MAX_SCHEDULE_LOOKAHEAD_DAYS = 366 * 8
MAX_ROUTINE_PER_RUN_TOKENS = 1_000_000
MAX_ROUTINE_CUMULATIVE_TOKENS = 100_000_000
MAX_ROUTINE_PER_RUN_COST_USD = Decimal("1000")
MAX_ROUTINE_CUMULATIVE_COST_USD = Decimal("100000")
ROUTINE_CLAIM_LEASE_SECONDS = 30.0

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_FAILURE_CODE = re.compile(r"[a-z][a-z0-9_]{0,127}\Z")


def _text(
    value: str,
    name: str,
    *,
    maximum: int = MAX_ROUTINE_IDENTITY_CHARACTERS,
    single_line: bool = True,
) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or "\x00" in value
        or (single_line and any(character in "\r\n" for character in value))
    ):
        raise ValueError(f"{name} must be bounded non-empty text")


def _digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical sha256 digest")


def _utc(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware UTC")
    offset = value.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError(f"{name} must be timezone-aware UTC")


def _optional_utc(value: datetime | None, name: str) -> None:
    if value is not None:
        _utc(value, name)


def _positive_revision(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _bounded_count(value: int, name: str, *, maximum: int) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 0 <= value <= maximum
    ):
        raise ValueError(f"{name} is outside its bound")


def _money(value: Decimal, name: str, *, maximum: Decimal) -> None:
    if (
        not isinstance(value, Decimal)
        or not value.is_finite()
        or not Decimal("0") <= value <= maximum
    ):
        raise ValueError(f"{name} must be a bounded non-negative Decimal")


def _identities(
    values: Iterable[str],
    name: str,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    items = tuple(values)
    if (not allow_empty and not items) or len(items) > MAX_ROUTINE_IDENTITY_ITEMS:
        raise ValueError(f"{name} is empty or exceeds its bound")
    for item in items:
        _text(item, name)
    if len(items) != len(set(items)):
        raise ValueError(f"{name} cannot contain duplicates")
    return tuple(sorted(items))


def _delivery_ids(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    if len(values) > MAX_DISTRIBUTION_TARGETS:
        raise ValueError(f"{name} exceeds its bound")
    for item in values:
        _text(item, name)
    if len(values) != len(set(values)):
        raise ValueError(f"{name} cannot contain duplicates")
    return values


def text_digest(value: str) -> str:
    """Return the canonical digest used for authorized routine instructions."""

    return "sha256:" + sha256(value.encode("utf-8")).hexdigest()


class ScheduleKind(str, Enum):
    ONCE = "once"
    INTERVAL = "interval"
    CALENDAR = "calendar"


class CalendarDaySelector(str, Enum):
    EVERY_DAY = "every_day"
    WEEKDAYS = "weekdays"
    MONTH_DAYS = "month_days"


class NonexistentTimePolicy(str, Enum):
    SKIP = "skip"
    NEXT_VALID = "next_valid"


class AmbiguousTimePolicy(str, Enum):
    FIRST = "first"
    SECOND = "second"


class MisfirePolicy(str, Enum):
    SKIP = "skip"
    LATEST_ONLY = "latest_only"


class ReportingMode(str, Enum):
    ALWAYS = "always"
    CHANGES_ONLY = "changes_only"


class RoutineState(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    EXPIRED = "expired"
    DISABLED = "disabled"
    NEEDS_ATTENTION = "needs_attention"


class RoutineSlotKind(str, Enum):
    SCHEDULED = "scheduled"
    MANUAL = "manual"


class RoutineOccurrenceDisposition(str, Enum):
    CLAIMED = "claimed"
    PRECHECKING = "prechecking"
    RUNNING = "running"
    RUN_TERMINAL_PENDING_FINALIZATION = "run_terminal_pending_finalization"
    SKIPPED_NO_CHANGE = "skipped_no_change"
    COMPLETED = "completed"
    RETRYABLE = "retryable"
    TERMINAL_FAILED = "terminal_failed"


class RoutineControlAction(str, Enum):
    PAUSE = "pause"
    RESUME = "resume"
    RUN_NOW = "run_now"
    DISABLE = "disable"


@dataclass(frozen=True, slots=True)
class OnceSchedule:
    exact_at: datetime

    def __post_init__(self) -> None:
        _utc(self.exact_at, "once schedule exact_at")

    @property
    def kind(self) -> ScheduleKind:
        return ScheduleKind.ONCE


@dataclass(frozen=True, slots=True)
class IntervalSchedule:
    interval_seconds: int
    anchor_at: datetime

    def __post_init__(self) -> None:
        _utc(self.anchor_at, "interval schedule anchor_at")
        if (
            not isinstance(self.interval_seconds, int)
            or isinstance(self.interval_seconds, bool)
            or not MIN_ROUTINE_INTERVAL_SECONDS
            <= self.interval_seconds
            <= MAX_ROUTINE_INTERVAL_SECONDS
        ):
            raise ValueError("interval schedule duration is outside its bound")

    @property
    def kind(self) -> ScheduleKind:
        return ScheduleKind.INTERVAL


@dataclass(frozen=True, slots=True)
class CalendarSchedule:
    timezone: str
    hour: int
    minute: int
    day_selector: CalendarDaySelector
    weekdays: tuple[int, ...] = ()
    month_days: tuple[int, ...] = ()
    months: tuple[int, ...] = ()
    nonexistent_time_policy: NonexistentTimePolicy = NonexistentTimePolicy.SKIP
    ambiguous_time_policy: AmbiguousTimePolicy = AmbiguousTimePolicy.FIRST

    def __post_init__(self) -> None:
        _text(self.timezone, "calendar timezone", maximum=128)
        if "/" not in self.timezone or self.timezone.upper() in {
            "EST",
            "EDT",
            "CST",
            "CDT",
            "MST",
            "MDT",
            "PST",
            "PDT",
        }:
            raise ValueError("calendar timezone must be an exact IANA zone")
        if (
            not isinstance(self.hour, int)
            or isinstance(self.hour, bool)
            or not 0 <= self.hour <= 23
        ):
            raise ValueError("calendar hour is outside its bound")
        if (
            not isinstance(self.minute, int)
            or isinstance(self.minute, bool)
            or not 0 <= self.minute <= 59
        ):
            raise ValueError("calendar minute is outside its bound")
        if not isinstance(self.day_selector, CalendarDaySelector):
            raise TypeError("calendar day_selector is invalid")
        weekdays = tuple(self.weekdays)
        month_days = tuple(self.month_days)
        months = tuple(self.months)
        for values, name, lower, upper in (
            (weekdays, "calendar weekdays", 1, 7),
            (month_days, "calendar month_days", 1, 31),
            (months, "calendar months", 1, 12),
        ):
            if len(values) != len(set(values)) or values != tuple(sorted(values)):
                raise ValueError(f"{name} must be sorted and distinct")
            if any(
                not isinstance(item, int)
                or isinstance(item, bool)
                or not lower <= item <= upper
                for item in values
            ):
                raise ValueError(f"{name} contains an invalid value")
        if self.day_selector is CalendarDaySelector.EVERY_DAY:
            if weekdays or month_days:
                raise ValueError("every_day calendar cannot contain day filters")
        elif self.day_selector is CalendarDaySelector.WEEKDAYS:
            if not weekdays or month_days:
                raise ValueError("weekday calendar requires only exact weekdays")
        elif not month_days or weekdays:
            raise ValueError("month-day calendar requires only exact month days")
        if not isinstance(self.nonexistent_time_policy, NonexistentTimePolicy):
            raise TypeError("calendar nonexistent-time policy is invalid")
        if not isinstance(self.ambiguous_time_policy, AmbiguousTimePolicy):
            raise TypeError("calendar ambiguous-time policy is invalid")
        object.__setattr__(self, "weekdays", weekdays)
        object.__setattr__(self, "month_days", month_days)
        object.__setattr__(self, "months", months)

    @property
    def kind(self) -> ScheduleKind:
        return ScheduleKind.CALENDAR


RoutineSchedule: TypeAlias = OnceSchedule | IntervalSchedule | CalendarSchedule


@dataclass(frozen=True, slots=True)
class ResourceRevisionPrecheck:
    capability_id: str
    contract_digest: str
    source_id: str
    resource_id: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.capability_id, "precheck capability_id"),
            (self.source_id, "precheck source_id"),
            (self.resource_id, "precheck resource_id"),
        ):
            _text(value, name)
        _digest(self.contract_digest, "precheck contract_digest")


@dataclass(frozen=True, slots=True)
class ScheduledRoutineDraft:
    """Bounded user-facing input for proposing or revising one routine."""

    origin_run_id: str
    title: str
    authorized_instruction: str
    schedule: RoutineSchedule
    misfire_policy: MisfirePolicy
    reporting_mode: ReportingMode
    precheck: ResourceRevisionPrecheck | None
    allowed_source_ids: tuple[str, ...]
    allowed_connector_binding_ids: tuple[str, ...]
    allowed_resource_ids: tuple[str, ...]
    allowed_capability_ids: tuple[str, ...]
    sensitivity_ceiling: ModelSensitivity
    outcome_contract: OutcomeContract
    distribution_destination_id: str
    eligible_model_routes: tuple[str, ...]
    per_run_max_tokens: int
    per_run_max_cost_usd: Decimal
    cumulative_max_tokens: int
    cumulative_max_cost_usd: Decimal
    cumulative_max_attempts: int
    cumulative_max_occurrences: int
    maximum_consecutive_failures: int
    expires_at: datetime
    skill_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.origin_run_id, "routine draft origin_run_id")
        _text(self.title, "routine draft title", maximum=MAX_ROUTINE_TITLE_CHARACTERS)
        _text(
            self.authorized_instruction,
            "routine draft authorized_instruction",
            maximum=MAX_ROUTINE_INSTRUCTION_BYTES,
            single_line=False,
        )
        if (
            len(self.authorized_instruction.encode("utf-8"))
            > MAX_ROUTINE_INSTRUCTION_BYTES
        ):
            raise ValueError("routine draft authorized_instruction exceeds its bound")
        if not isinstance(
            self.schedule, (OnceSchedule, IntervalSchedule, CalendarSchedule)
        ):
            raise TypeError("routine draft schedule is invalid")
        if not isinstance(self.misfire_policy, MisfirePolicy):
            raise TypeError("routine draft misfire policy is invalid")
        if not isinstance(self.reporting_mode, ReportingMode):
            raise TypeError("routine draft reporting mode is invalid")
        if self.reporting_mode is ReportingMode.ALWAYS and self.precheck is not None:
            raise ValueError("always-reporting draft cannot contain a precheck")
        if self.reporting_mode is ReportingMode.CHANGES_ONLY and not isinstance(
            self.precheck, ResourceRevisionPrecheck
        ):
            raise ValueError("changes-only draft requires an exact precheck")
        sources = _identities(self.allowed_source_ids, "draft allowed_source_ids")
        bindings = _identities(
            self.allowed_connector_binding_ids,
            "draft allowed_connector_binding_ids",
        )
        resources = _identities(self.allowed_resource_ids, "draft allowed_resource_ids")
        capabilities = _identities(
            self.allowed_capability_ids,
            "draft allowed_capability_ids",
            allow_empty=False,
        )
        routes = _identities(
            self.eligible_model_routes,
            "draft eligible_model_routes",
            allow_empty=False,
        )
        skills = _identities(self.skill_names, "draft skill_names")
        if len(skills) > MAX_ROUTINE_SKILL_BINDINGS:
            raise ValueError("routine draft skill bindings exceed their bound")
        if not sources and not bindings:
            raise ValueError("routine draft requires a source or connector binding")
        if sources and not resources:
            raise ValueError("source-scoped draft requires exact resources")
        if not isinstance(self.sensitivity_ceiling, ModelSensitivity):
            raise TypeError("routine draft sensitivity ceiling is invalid")
        if not isinstance(self.outcome_contract, OutcomeContract):
            raise TypeError("routine draft outcome contract is invalid")
        _text(
            self.distribution_destination_id,
            "routine draft distribution_destination_id",
        )
        if (
            self.outcome_contract.maximum_effective_sensitivity.routing_rank
            > self.sensitivity_ceiling.routing_rank
        ):
            raise ValueError("routine draft outcome sensitivity exceeds its ceiling")
        for value, name, maximum in (
            (
                self.per_run_max_tokens,
                "draft per-run tokens",
                MAX_ROUTINE_PER_RUN_TOKENS,
            ),
            (
                self.cumulative_max_tokens,
                "draft cumulative tokens",
                MAX_ROUTINE_CUMULATIVE_TOKENS,
            ),
            (
                self.cumulative_max_attempts,
                "draft cumulative attempts",
                MAX_ROUTINE_OCCURRENCES * MAX_ROUTINE_ATTEMPTS,
            ),
            (
                self.cumulative_max_occurrences,
                "draft cumulative occurrences",
                MAX_ROUTINE_OCCURRENCES,
            ),
            (
                self.maximum_consecutive_failures,
                "draft maximum consecutive failures",
                MAX_ROUTINE_CONSECUTIVE_FAILURES,
            ),
        ):
            _bounded_count(value, name, maximum=maximum)
            if value == 0:
                raise ValueError(f"{name} must be positive")
        _money(
            self.per_run_max_cost_usd,
            "draft per-run cost",
            maximum=MAX_ROUTINE_PER_RUN_COST_USD,
        )
        _money(
            self.cumulative_max_cost_usd,
            "draft cumulative cost",
            maximum=MAX_ROUTINE_CUMULATIVE_COST_USD,
        )
        if self.per_run_max_tokens > self.cumulative_max_tokens:
            raise ValueError("draft per-run token ceiling exceeds cumulative ceiling")
        if self.per_run_max_cost_usd > self.cumulative_max_cost_usd:
            raise ValueError("draft per-run cost ceiling exceeds cumulative ceiling")
        _utc(self.expires_at, "routine draft expires_at")
        object.__setattr__(self, "allowed_source_ids", sources)
        object.__setattr__(self, "allowed_connector_binding_ids", bindings)
        object.__setattr__(self, "allowed_resource_ids", resources)
        object.__setattr__(self, "allowed_capability_ids", capabilities)
        object.__setattr__(self, "eligible_model_routes", routes)
        object.__setattr__(self, "skill_names", skills)


@dataclass(frozen=True, slots=True)
class ResourceRevisionObservation:
    source_id: str
    resource_id: str
    resource_revision: str
    catalog_revision: str
    observed_at: datetime

    def __post_init__(self) -> None:
        for value, name in (
            (self.source_id, "observation source_id"),
            (self.resource_id, "observation resource_id"),
        ):
            _text(value, name)
        _digest(self.resource_revision, "observation resource_revision")
        _digest(self.catalog_revision, "observation catalog_revision")
        _utc(self.observed_at, "observation observed_at")

    @property
    def digest(self) -> str:
        return (
            "sha256:"
            + sha256(
                canonical_json(
                    {
                        "source_id": self.source_id,
                        "resource_id": self.resource_id,
                        "resource_revision": self.resource_revision,
                        "catalog_revision": self.catalog_revision,
                    }
                ).encode("utf-8")
            ).hexdigest()
        )


@dataclass(frozen=True, slots=True)
class RoutineSkillBinding:
    skill_name: str
    skill_revision: int
    content_digest: str
    attached_by_principal: str
    attached_at: datetime

    def __post_init__(self) -> None:
        _text(self.skill_name, "routine skill_name", maximum=128)
        _positive_revision(self.skill_revision, "routine skill_revision")
        _digest(self.content_digest, "routine skill content_digest")
        _text(self.attached_by_principal, "routine skill principal")
        _utc(self.attached_at, "routine skill attached_at")


@dataclass(frozen=True, slots=True)
class RoutinePromotionEvidence:
    basis_run_id: str
    terminal_result_digest: str
    executed_capability_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.basis_run_id, "routine basis_run_id")
        _digest(self.terminal_result_digest, "routine terminal result digest")
        capabilities = _identities(
            self.executed_capability_ids,
            "routine executed capability_ids",
            allow_empty=False,
        )
        object.__setattr__(self, "executed_capability_ids", capabilities)


@dataclass(frozen=True, slots=True)
class ScheduledRoutine:
    routine_id: str
    agent_id: str
    conversation_id: str
    owner_principal_id: str
    title: str
    authorized_instruction: str
    instruction_digest: str
    schedule: RoutineSchedule
    schedule_interpreter_revision: int
    misfire_policy: MisfirePolicy
    reporting_mode: ReportingMode
    precheck: ResourceRevisionPrecheck | None
    last_acknowledged_precheck_observation: ResourceRevisionObservation | None
    allowed_source_ids: tuple[str, ...]
    allowed_connector_binding_ids: tuple[str, ...]
    allowed_resource_ids: tuple[str, ...]
    allowed_capability_ids: tuple[str, ...]
    allowed_access_modes: frozenset[AccessMode]
    allowed_operational_effects: frozenset[OperationalEffect]
    sensitivity_ceiling: ModelSensitivity
    eligible_model_routes: tuple[str, ...]
    skill_bindings: tuple[RoutineSkillBinding, ...]
    outcome_contract: OutcomeContract
    distribution_plan: DistributionPlan
    per_run_max_tokens: int
    per_run_max_cost_usd: Decimal
    cumulative_max_tokens: int
    cumulative_max_cost_usd: Decimal
    cumulative_max_attempts: int
    cumulative_max_occurrences: int
    reserved_tokens: int
    reserved_cost_usd: Decimal
    charged_tokens: int
    charged_cost_usd: Decimal
    attempt_count: int
    occurrence_count: int
    maximum_consecutive_failures: int
    consecutive_failures: int
    expires_at: datetime
    next_due_at: datetime | None
    active_occurrence_id: str | None
    last_occurrence_id: str | None
    last_delivery_ids: tuple[str, ...]
    promotion_evidence: RoutinePromotionEvidence | None
    state: RoutineState
    revision: int
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        for value, name in (
            (self.routine_id, "routine_id"),
            (self.agent_id, "routine agent_id"),
            (self.conversation_id, "routine conversation_id"),
            (self.owner_principal_id, "routine owner_principal_id"),
        ):
            _text(value, name)
        _text(self.title, "routine title", maximum=MAX_ROUTINE_TITLE_CHARACTERS)
        _text(
            self.authorized_instruction,
            "routine authorized_instruction",
            maximum=MAX_ROUTINE_INSTRUCTION_BYTES,
            single_line=False,
        )
        if (
            len(self.authorized_instruction.encode("utf-8"))
            > MAX_ROUTINE_INSTRUCTION_BYTES
        ):
            raise ValueError("routine authorized_instruction exceeds its byte bound")
        _digest(self.instruction_digest, "routine instruction_digest")
        if self.instruction_digest != text_digest(self.authorized_instruction):
            raise ValueError("routine instruction digest does not match its content")
        if not isinstance(
            self.schedule, (OnceSchedule, IntervalSchedule, CalendarSchedule)
        ):
            raise TypeError("routine schedule is invalid")
        if self.schedule_interpreter_revision != SCHEDULE_INTERPRETER_REVISION:
            raise ValueError("routine schedule interpreter revision is unsupported")
        if not isinstance(self.misfire_policy, MisfirePolicy):
            raise TypeError("routine misfire policy is invalid")
        if not isinstance(self.reporting_mode, ReportingMode):
            raise TypeError("routine reporting mode is invalid")
        if self.reporting_mode is ReportingMode.ALWAYS and self.precheck is not None:
            raise ValueError("always-reporting routine cannot contain a precheck")
        if self.reporting_mode is ReportingMode.CHANGES_ONLY and not isinstance(
            self.precheck, ResourceRevisionPrecheck
        ):
            raise ValueError("changes-only routine requires its exact precheck")
        if (
            self.last_acknowledged_precheck_observation is not None
            and self.precheck is None
        ):
            raise ValueError("routine observation requires a precheck")
        sources = _identities(self.allowed_source_ids, "routine allowed_source_ids")
        bindings = _identities(
            self.allowed_connector_binding_ids,
            "routine allowed_connector_binding_ids",
        )
        resources = _identities(
            self.allowed_resource_ids, "routine allowed_resource_ids"
        )
        capabilities = _identities(
            self.allowed_capability_ids,
            "routine allowed_capability_ids",
            allow_empty=False,
        )
        routes = _identities(
            self.eligible_model_routes,
            "routine eligible_model_routes",
            allow_empty=False,
        )
        if not sources and not bindings:
            raise ValueError(
                "routine requires an exact source or connector binding ceiling"
            )
        if sources and not resources:
            raise ValueError("source-scoped routine requires exact resource identities")
        access_modes = frozenset(self.allowed_access_modes)
        if not access_modes or not access_modes <= {AccessMode.NONE, AccessMode.READ}:
            raise ValueError("D1 routine permits only none/read access modes")
        effects = frozenset(self.allowed_operational_effects)
        if effects != {OperationalEffect.NONE}:
            raise ValueError("D1 routine permits only no operational effect")
        if not isinstance(self.sensitivity_ceiling, ModelSensitivity):
            raise TypeError("routine sensitivity ceiling is invalid")
        skill_bindings = tuple(self.skill_bindings)
        if len(skill_bindings) > MAX_ROUTINE_SKILL_BINDINGS:
            raise ValueError("routine skill bindings exceed their bound")
        if any(not isinstance(item, RoutineSkillBinding) for item in skill_bindings):
            raise TypeError("routine skill bindings are invalid")
        if (
            tuple(sorted(skill_bindings, key=lambda item: item.skill_name))
            != skill_bindings
        ):
            raise ValueError("routine skill bindings must be sorted")
        if len({item.skill_name for item in skill_bindings}) != len(skill_bindings):
            raise ValueError("routine skill bindings cannot duplicate a skill")
        if not isinstance(self.outcome_contract, OutcomeContract):
            raise TypeError("routine outcome contract is invalid")
        if not isinstance(self.distribution_plan, DistributionPlan):
            raise TypeError("routine distribution plan is invalid")
        if any(
            target.conversation_id != self.conversation_id
            for target in self.distribution_plan.targets
        ):
            raise ValueError(
                "routine distribution plan belongs to another conversation"
            )
        if (
            self.outcome_contract.maximum_effective_sensitivity.routing_rank
            > self.sensitivity_ceiling.routing_rank
        ):
            raise ValueError("routine outcome sensitivity exceeds its ceiling")
        for count_value, count_name, maximum, require_positive in (
            (
                self.per_run_max_tokens,
                "per-run token ceiling",
                MAX_ROUTINE_PER_RUN_TOKENS,
                True,
            ),
            (
                self.cumulative_max_tokens,
                "cumulative token ceiling",
                MAX_ROUTINE_CUMULATIVE_TOKENS,
                True,
            ),
            (
                self.reserved_tokens,
                "reserved token amount",
                MAX_ROUTINE_CUMULATIVE_TOKENS,
                False,
            ),
            (
                self.charged_tokens,
                "charged token amount",
                MAX_ROUTINE_CUMULATIVE_TOKENS,
                False,
            ),
            (
                self.attempt_count,
                "routine attempt count",
                MAX_ROUTINE_OCCURRENCES * MAX_ROUTINE_ATTEMPTS,
                False,
            ),
            (
                self.occurrence_count,
                "routine occurrence count",
                MAX_ROUTINE_OCCURRENCES,
                False,
            ),
        ):
            _bounded_count(count_value, count_name, maximum=maximum)
            if require_positive and count_value == 0:
                raise ValueError(f"{count_name} must be positive")
        _money(
            self.per_run_max_cost_usd,
            "per-run cost ceiling",
            maximum=MAX_ROUTINE_PER_RUN_COST_USD,
        )
        _money(
            self.cumulative_max_cost_usd,
            "cumulative cost ceiling",
            maximum=MAX_ROUTINE_CUMULATIVE_COST_USD,
        )
        _money(
            self.reserved_cost_usd,
            "reserved cost amount",
            maximum=MAX_ROUTINE_CUMULATIVE_COST_USD,
        )
        _money(
            self.charged_cost_usd,
            "charged cost amount",
            maximum=MAX_ROUTINE_CUMULATIVE_COST_USD,
        )
        if self.per_run_max_tokens > self.cumulative_max_tokens:
            raise ValueError("per-run token ceiling exceeds cumulative ceiling")
        if self.per_run_max_cost_usd > self.cumulative_max_cost_usd:
            raise ValueError("per-run cost ceiling exceeds cumulative ceiling")
        if self.reserved_tokens + self.charged_tokens > self.cumulative_max_tokens:
            raise ValueError("routine token budget is oversubscribed")
        if (
            self.reserved_cost_usd + self.charged_cost_usd
            > self.cumulative_max_cost_usd
        ):
            raise ValueError("routine cost budget is oversubscribed")
        for ceiling_value, ceiling_name, maximum in (
            (
                self.cumulative_max_attempts,
                "cumulative attempt ceiling",
                MAX_ROUTINE_OCCURRENCES * MAX_ROUTINE_ATTEMPTS,
            ),
            (
                self.cumulative_max_occurrences,
                "cumulative occurrence ceiling",
                MAX_ROUTINE_OCCURRENCES,
            ),
            (
                self.maximum_consecutive_failures,
                "maximum consecutive failures",
                MAX_ROUTINE_CONSECUTIVE_FAILURES,
            ),
        ):
            _bounded_count(ceiling_value, ceiling_name, maximum=maximum)
            if ceiling_value == 0:
                raise ValueError(f"{ceiling_name} must be positive")
        _bounded_count(
            self.consecutive_failures,
            "consecutive failures",
            maximum=self.maximum_consecutive_failures,
        )
        if self.attempt_count > self.cumulative_max_attempts:
            raise ValueError("routine attempt ceiling is exhausted inconsistently")
        if self.occurrence_count > self.cumulative_max_occurrences:
            raise ValueError("routine occurrence ceiling is exhausted inconsistently")
        _utc(self.created_at, "routine created_at")
        _utc(self.updated_at, "routine updated_at")
        _utc(self.expires_at, "routine expires_at")
        _optional_utc(self.next_due_at, "routine next_due_at")
        if self.updated_at < self.created_at:
            raise ValueError("routine updated_at precedes created_at")
        if (
            not self.created_at
            < self.expires_at
            <= self.created_at + MAX_ROUTINE_LIFETIME
        ):
            raise ValueError("routine expiration is outside its finite lifetime")
        if self.next_due_at is not None and self.next_due_at > self.expires_at:
            raise ValueError("routine next due instant exceeds expiration")
        for optional_identity, identity_name in (
            (self.active_occurrence_id, "active occurrence_id"),
            (self.last_occurrence_id, "last occurrence_id"),
        ):
            if optional_identity is not None:
                _text(optional_identity, identity_name)
        last_delivery_ids = _delivery_ids(
            self.last_delivery_ids,
            "routine last_delivery_ids",
        )
        if not isinstance(self.state, RoutineState):
            raise TypeError("routine state is invalid")
        _positive_revision(self.revision, "routine revision")
        object.__setattr__(self, "allowed_source_ids", sources)
        object.__setattr__(self, "allowed_connector_binding_ids", bindings)
        object.__setattr__(self, "allowed_resource_ids", resources)
        object.__setattr__(self, "allowed_capability_ids", capabilities)
        object.__setattr__(self, "eligible_model_routes", routes)
        object.__setattr__(self, "allowed_access_modes", access_modes)
        object.__setattr__(self, "allowed_operational_effects", effects)
        object.__setattr__(self, "skill_bindings", skill_bindings)
        object.__setattr__(self, "last_delivery_ids", last_delivery_ids)


@dataclass(frozen=True, slots=True)
class RoutineOccurrence:
    occurrence_id: str
    agent_id: str
    routine_id: str
    routine_revision: int
    slot_kind: RoutineSlotKind
    slot_key: str
    scheduled_for: datetime
    claimed_at: datetime | None
    claim_token: str | None
    lease_expires_at: datetime | None
    precheck_observation: ResourceRevisionObservation | None
    execution_scope: ExecutionScope | None
    execution_scope_digest: str | None
    reserved_run_id: str | None
    reserved_tokens: int
    reserved_cost_usd: Decimal
    charged_tokens: int
    charged_cost_usd: Decimal
    run_bound_at: datetime | None
    run_terminal_at: datetime | None
    conclusion_digest: str | None
    terminal_run_id: str | None
    delivery_ids: tuple[str, ...]
    attempt_count: int
    failure_code: str | None
    retry_at: datetime | None
    disposition: RoutineOccurrenceDisposition
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        for identity_value, identity_name in (
            (self.occurrence_id, "occurrence_id"),
            (self.agent_id, "occurrence agent_id"),
            (self.routine_id, "occurrence routine_id"),
            (self.slot_key, "occurrence slot_key"),
        ):
            _text(identity_value, identity_name)
        _positive_revision(self.routine_revision, "occurrence routine_revision")
        if not isinstance(self.slot_kind, RoutineSlotKind):
            raise TypeError("occurrence slot kind is invalid")
        if not self.slot_key.startswith(f"{self.slot_kind.value}:"):
            raise ValueError("occurrence slot key does not match its kind")
        _utc(self.scheduled_for, "occurrence scheduled_for")
        for optional_instant, instant_name in (
            (self.claimed_at, "occurrence claimed_at"),
            (self.lease_expires_at, "occurrence lease_expires_at"),
            (self.run_bound_at, "occurrence run_bound_at"),
            (self.run_terminal_at, "occurrence run_terminal_at"),
            (self.retry_at, "occurrence retry_at"),
        ):
            _optional_utc(optional_instant, instant_name)
        if (self.claimed_at is None) != (self.claim_token is None):
            raise ValueError("occurrence claim time and token must be present together")
        if self.claim_token is not None:
            _text(self.claim_token, "occurrence claim_token")
        if self.lease_expires_at is not None and self.claimed_at is None:
            raise ValueError("occurrence lease requires a claim")
        if (
            self.lease_expires_at is not None
            and self.claimed_at is not None
            and self.lease_expires_at <= self.claimed_at
        ):
            raise ValueError("occurrence lease must expire after claim")
        if (self.execution_scope is None) != (self.execution_scope_digest is None):
            raise ValueError(
                "occurrence execution scope and digest must be present together"
            )
        if (
            self.execution_scope is not None
            and self.execution_scope.digest != self.execution_scope_digest
        ):
            raise ValueError("occurrence execution scope digest does not match")
        for optional_identity, identity_name in (
            (self.reserved_run_id, "occurrence reserved_run_id"),
            (self.terminal_run_id, "occurrence terminal_run_id"),
        ):
            if optional_identity is not None:
                _text(optional_identity, identity_name)
        delivery_ids = _delivery_ids(self.delivery_ids, "occurrence delivery_ids")
        if self.run_bound_at is not None and self.reserved_run_id is None:
            raise ValueError("occurrence run binding requires a reserved run")
        if self.run_terminal_at is not None and self.terminal_run_id is None:
            raise ValueError("occurrence terminal time requires terminal run identity")
        if (
            self.terminal_run_id is not None
            and self.terminal_run_id != self.reserved_run_id
        ):
            raise ValueError("occurrence terminal run must be its reserved run")
        if self.conclusion_digest is not None:
            _digest(self.conclusion_digest, "occurrence conclusion_digest")
        for count_value, count_name, maximum in (
            (
                self.reserved_tokens,
                "occurrence reserved tokens",
                MAX_ROUTINE_PER_RUN_TOKENS,
            ),
            (
                self.charged_tokens,
                "occurrence charged tokens",
                MAX_ROUTINE_PER_RUN_TOKENS,
            ),
            (self.attempt_count, "occurrence attempt count", MAX_ROUTINE_ATTEMPTS),
        ):
            _bounded_count(count_value, count_name, maximum=maximum)
        _money(
            self.reserved_cost_usd,
            "occurrence reserved cost",
            maximum=MAX_ROUTINE_PER_RUN_COST_USD,
        )
        _money(
            self.charged_cost_usd,
            "occurrence charged cost",
            maximum=MAX_ROUTINE_PER_RUN_COST_USD,
        )
        if (
            self.failure_code is not None
            and _FAILURE_CODE.fullmatch(self.failure_code) is None
        ):
            raise ValueError("occurrence failure_code is invalid")
        if not isinstance(self.disposition, RoutineOccurrenceDisposition):
            raise TypeError("occurrence disposition is invalid")
        _utc(self.created_at, "occurrence created_at")
        _utc(self.updated_at, "occurrence updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("occurrence updated_at precedes created_at")
        object.__setattr__(self, "delivery_ids", delivery_ids)


@dataclass(frozen=True, slots=True)
class ScheduledRoutineSummary:
    routine_id: str
    title: str
    state: RoutineState
    schedule_kind: ScheduleKind
    next_due_at: datetime | None
    revision: int
    occurrence_count: int
    consecutive_failures: int

    def __post_init__(self) -> None:
        _text(self.routine_id, "routine summary routine_id")
        _text(self.title, "routine summary title", maximum=MAX_ROUTINE_TITLE_CHARACTERS)
        if not isinstance(self.state, RoutineState):
            raise TypeError("routine summary state is invalid")
        if not isinstance(self.schedule_kind, ScheduleKind):
            raise TypeError("routine summary schedule kind is invalid")
        _optional_utc(self.next_due_at, "routine summary next_due_at")
        _positive_revision(self.revision, "routine summary revision")
        _bounded_count(
            self.occurrence_count,
            "routine summary occurrence count",
            maximum=MAX_ROUTINE_OCCURRENCES,
        )
        _bounded_count(
            self.consecutive_failures,
            "routine summary failure count",
            maximum=MAX_ROUTINE_CONSECUTIVE_FAILURES,
        )


@dataclass(frozen=True, slots=True)
class ScheduledRoutineInspection:
    routine: ScheduledRoutine
    recent_occurrences: tuple[RoutineOccurrence, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.routine, ScheduledRoutine):
            raise TypeError("routine inspection requires a routine")
        occurrences = tuple(self.recent_occurrences)
        if len(occurrences) > MAX_ROUTINE_HISTORY_PAGE_SIZE:
            raise ValueError("routine inspection history exceeds its bound")
        if any(
            not isinstance(item, RoutineOccurrence)
            or item.agent_id != self.routine.agent_id
            or item.routine_id != self.routine.routine_id
            for item in occurrences
        ):
            raise ValueError("routine inspection occurrence identity is invalid")
        object.__setattr__(self, "recent_occurrences", occurrences)
