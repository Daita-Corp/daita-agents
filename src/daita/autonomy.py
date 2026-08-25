"""Own the bounded Stage C terminal-job follow-up and conversation inbox values."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from hashlib import sha256

from ._json import FrozenJsonObject, canonical_json
from .capabilities import AccessMode, ExecutionScope, OperationalEffect
from .jobs.models import JobExecutionMode, JobRun
from .llm.models import (
    MessageRole,
    ModelSensitivity,
    ToolCall,
    ToolResultBlock,
)
from .loop.models import LoopExit, LoopExitKind, LoopLimits, RunOrigin, Transcript

MAX_AUTONOMOUS_FOLLOWUPS_PER_AGENT = 256
MAX_CONVERSATION_INBOX_ITEMS_PER_AGENT = 256
MAX_INBOX_PAGE_SIZE = 50
MAX_FOLLOWUP_ATTEMPTS = 3
MAX_FOLLOWUP_EVENT_BYTES = 16 * 1_024
MAX_FOLLOWUP_AUDIT_BYTES = 1024 * 1_024
MAX_INBOX_PAYLOAD_BYTES = 64 * 1_024
MAX_FOLLOWUP_RESULT_PREVIEW_BYTES = 4 * 1_024
MAX_INBOX_REPORT_PREVIEW_BYTES = 48 * 1_024
FOLLOWUP_LEASE_SECONDS = 30.0
FOLLOWUP_EXPIRY_SECONDS = 3_600.0
FOLLOWUP_INSTRUCTION_ID = "stage_c.terminal_job_report.v1"
FOLLOWUP_INSTRUCTION = (
    "Inspect the exact bound terminal durable job. Before producing final text, "
    "call job_inspect and job_read_results for the exact bound job_id and wait for "
    "both results. Both successful calls are required; job_read_results does not "
    "replace job_inspect. Use only the projected inspection and read capabilities "
    "inside the immutable scope. Verify current frozen resource facts when useful, "
    "then produce one concise report of the actual outcome, evidence, and any safe "
    "next action. If either required job call fails, report that failure without "
    "claiming a verified conclusion. "
    "Do not start or cancel jobs, mutate data or advisory context, expand scope, "
    "or request another autonomous continuation."
)
FOLLOWUP_INSTRUCTION_DIGEST = (
    "sha256:" + sha256(FOLLOWUP_INSTRUCTION.encode("utf-8")).hexdigest()
)

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _text(value: str, name: str, *, maximum: int = 512) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or any(character in "\r\n\x00" for character in value)
    ):
        raise ValueError(f"{name} must be bounded non-empty single-line text")


def _utc(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware UTC")
    offset = value.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError(f"{name} must be timezone-aware UTC")


def _optional_utc(value: datetime | None, name: str) -> None:
    if value is not None:
        _utc(value, name)


def _money(value: Decimal, name: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise ValueError(f"{name} must be a finite non-negative Decimal")


def _identities(
    values: Iterable[str],
    name: str,
    *,
    maximum_items: int = 256,
    maximum_characters: int = 2_048,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    items = tuple(values)
    if not items or len(items) > maximum_items:
        raise ValueError(f"{name} is empty or exceeds its bound")
    for item in items:
        _text(item, name, maximum=maximum_characters)
    if len(items) != len(set(items)):
        raise ValueError(f"{name} cannot contain duplicates")
    return tuple(sorted(items))


def _bounded_mapping(
    value: Mapping[str, object],
    name: str,
    maximum_bytes: int,
) -> FrozenJsonObject:
    frozen = FrozenJsonObject.from_mapping(value)
    if len(canonical_json(frozen).encode("utf-8")) > maximum_bytes:
        raise ValueError(f"{name} exceeds its byte bound")
    return frozen


def _utf8_preview(value: str, maximum_bytes: int) -> tuple[str, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= maximum_bytes:
        return value, False
    preview = encoded[:maximum_bytes].decode("utf-8", errors="ignore")
    return preview, True


def _mapping_digest(value: Mapping[str, object]) -> str:
    return "sha256:" + sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _text_digest(value: str) -> str:
    return "sha256:" + sha256(value.encode("utf-8")).hexdigest()


class FollowupObservationSource(str, Enum):
    DAITA_JOB = "daita_job"


class FollowupDisposition(str, Enum):
    AVAILABLE = "available"
    CLAIMED = "claimed"
    RUNNING = "running"
    RUN_TERMINAL_PENDING_FINALIZATION = "run_terminal_pending_finalization"
    COMPLETED = "completed"
    RETRYABLE_FAILED = "retryable_failed"
    TERMINAL_FAILED = "terminal_failed"
    EXPIRED = "expired"


class DeliveryState(str, Enum):
    AVAILABLE = "available"
    BLOCKED = "blocked"
    ACKNOWLEDGED = "acknowledged"


class DeliverySubjectKind(str, Enum):
    STANDALONE_FOLLOWUP = "standalone_followup"


@dataclass(frozen=True, slots=True)
class DeliverySubject:
    """The typed owner of one committed conclusion and logical delivery."""

    kind: DeliverySubjectKind
    subject_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, DeliverySubjectKind):
            raise TypeError("delivery subject kind is invalid")
        _text(self.subject_id, "delivery subject_id", maximum=256)


@dataclass(frozen=True, slots=True)
class FollowupConclusionEvidence:
    """Code-derived proof that the bounded run inspected its exact job outcome."""

    run_id: str
    job_id: str
    job_revision: int
    inspection_call_id: str
    inspection_result_digest: str
    result_call_id: str
    result_result_digest: str
    job_result_id: str | None
    report_digest: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.run_id, "follow-up evidence run_id"),
            (self.job_id, "follow-up evidence job_id"),
            (self.inspection_call_id, "follow-up evidence inspection_call_id"),
            (self.result_call_id, "follow-up evidence result_call_id"),
        ):
            _text(value, name, maximum=256)
        if (
            not isinstance(self.job_revision, int)
            or isinstance(self.job_revision, bool)
            or self.job_revision < 1
        ):
            raise ValueError("follow-up evidence job_revision must be positive")
        for value, name in (
            (self.inspection_result_digest, "inspection result digest"),
            (self.result_result_digest, "job result digest"),
            (self.report_digest, "report digest"),
        ):
            if _DIGEST.fullmatch(value) is None:
                raise ValueError(f"follow-up evidence {name} is invalid")
        if self.job_result_id is not None:
            _text(self.job_result_id, "follow-up evidence job_result_id", maximum=256)

    @property
    def digest(self) -> str:
        return _mapping_digest(
            {
                "run_id": self.run_id,
                "job_id": self.job_id,
                "job_revision": self.job_revision,
                "inspection_call_id": self.inspection_call_id,
                "inspection_result_digest": self.inspection_result_digest,
                "result_call_id": self.result_call_id,
                "result_result_digest": self.result_result_digest,
                "job_result_id": self.job_result_id,
                "report_digest": self.report_digest,
            }
        )


@dataclass(frozen=True, slots=True)
class FollowupGrant:
    grant_id: str
    job_id: str
    agent_id: str
    conversation_id: str
    authorizing_principal: str
    allowed_terminal_job_observation: str
    allowed_source_ids: tuple[str, ...]
    allowed_resource_ids: tuple[str, ...]
    allowed_capability_ids: tuple[str, ...]
    allowed_access_modes: frozenset[AccessMode]
    allowed_operational_effects: frozenset[OperationalEffect]
    instruction_id: str
    instruction_digest: str
    sensitivity_ceiling: ModelSensitivity
    delivery_sensitivity_ceiling: ModelSensitivity
    eligible_model_routes: tuple[str, ...]
    delivery_destination: str
    max_successful_runs: int
    max_attempts: int
    per_run_max_cost_usd: Decimal
    per_run_max_tokens: int
    cumulative_max_cost_usd: Decimal
    cumulative_max_tokens: int
    expires_at: datetime

    def __post_init__(self) -> None:
        for identity_value, identity_name in (
            (self.grant_id, "follow-up grant_id"),
            (self.job_id, "follow-up grant job_id"),
            (self.agent_id, "follow-up grant agent_id"),
            (self.conversation_id, "follow-up grant conversation_id"),
            (self.authorizing_principal, "follow-up authorizing principal"),
            (
                self.allowed_terminal_job_observation,
                "follow-up allowed terminal observation",
            ),
            (self.instruction_id, "follow-up instruction_id"),
            (self.delivery_destination, "follow-up delivery destination"),
        ):
            _text(identity_value, identity_name)
        if _DIGEST.fullmatch(self.instruction_digest) is None:
            raise ValueError("follow-up instruction digest is invalid")
        sources = _identities(self.allowed_source_ids, "follow-up source IDs")
        resources = _identities(self.allowed_resource_ids, "follow-up resource IDs")
        capabilities = _identities(
            self.allowed_capability_ids,
            "follow-up capability IDs",
        )
        routes = _identities(self.eligible_model_routes, "follow-up model routes")
        access_modes = frozenset(self.allowed_access_modes)
        effects = frozenset(self.allowed_operational_effects)
        if not access_modes or any(
            not isinstance(item, AccessMode) for item in access_modes
        ):
            raise ValueError("follow-up grant requires allowed access modes")
        if not effects or any(
            not isinstance(item, OperationalEffect) for item in effects
        ):
            raise ValueError("follow-up grant requires allowed operational effects")
        if not isinstance(self.sensitivity_ceiling, ModelSensitivity) or not isinstance(
            self.delivery_sensitivity_ceiling,
            ModelSensitivity,
        ):
            raise TypeError("follow-up sensitivity ceilings are invalid")
        if self.max_successful_runs != 1:
            raise ValueError("Stage C permits exactly one successful run")
        if (
            not isinstance(self.max_attempts, int)
            or isinstance(self.max_attempts, bool)
            or not 1 <= self.max_attempts <= MAX_FOLLOWUP_ATTEMPTS
        ):
            raise ValueError("follow-up max_attempts is outside its bound")
        for cost_value, cost_name in (
            (self.per_run_max_cost_usd, "follow-up per-run cost"),
            (self.cumulative_max_cost_usd, "follow-up cumulative cost"),
        ):
            _money(cost_value, cost_name)
        for token_value, token_name in (
            (self.per_run_max_tokens, "follow-up per-run tokens"),
            (self.cumulative_max_tokens, "follow-up cumulative tokens"),
        ):
            if (
                not isinstance(token_value, int)
                or isinstance(token_value, bool)
                or token_value < 1
            ):
                raise ValueError(f"{token_name} must be positive")
        if (
            self.cumulative_max_cost_usd < self.per_run_max_cost_usd
            or self.cumulative_max_tokens < self.per_run_max_tokens
        ):
            raise ValueError("follow-up cumulative budget cannot trail one run")
        _utc(self.expires_at, "follow-up grant expires_at")
        object.__setattr__(self, "allowed_source_ids", sources)
        object.__setattr__(self, "allowed_resource_ids", resources)
        object.__setattr__(self, "allowed_capability_ids", capabilities)
        object.__setattr__(self, "eligible_model_routes", routes)
        object.__setattr__(self, "allowed_access_modes", access_modes)
        object.__setattr__(self, "allowed_operational_effects", effects)

    def contains_scope(self, scope: ExecutionScope) -> bool:
        return (
            scope.grant_id == self.grant_id
            and scope.agent_id == self.agent_id
            and scope.principal_id == self.authorizing_principal
            and scope.job_id == self.job_id
            and set(scope.allowed_source_ids) <= set(self.allowed_source_ids)
            and set(scope.allowed_resource_ids) <= set(self.allowed_resource_ids)
            and set(scope.allowed_capability_ids) <= set(self.allowed_capability_ids)
            and scope.allowed_access_modes <= self.allowed_access_modes
            and scope.allowed_operational_effects <= self.allowed_operational_effects
            and scope.sensitivity_ceiling.routing_rank
            <= self.sensitivity_ceiling.routing_rank
            and set(scope.eligible_model_routes) <= set(self.eligible_model_routes)
            and scope.per_run_max_cost_usd <= self.per_run_max_cost_usd
            and scope.per_run_max_tokens <= self.per_run_max_tokens
            and scope.delivery_destination == self.delivery_destination
        )


@dataclass(frozen=True, slots=True)
class AutonomousFollowup:
    followup_id: str
    agent_id: str
    conversation_id: str
    event_id: str
    observation_source: FollowupObservationSource
    job_id: str
    job_terminal_revision: int
    event_type: str
    event_payload: Mapping[str, object]
    payload_digest: str
    received_at: datetime
    grant: FollowupGrant
    execution_scope: ExecutionScope
    disposition: FollowupDisposition
    created_at: datetime
    updated_at: datetime
    revision: int = 1
    attempt_count: int = 0
    claim_token: str | None = None
    lease_expires_at: datetime | None = None
    reserved_run_id: str | None = None
    reserved_cost_usd: Decimal = Decimal("0")
    reserved_tokens: int = 0
    charged_cost_usd: Decimal = Decimal("0")
    charged_tokens: int = 0
    run_bound_at: datetime | None = None
    run_terminal_at: datetime | None = None
    audit_context: Mapping[str, object] = field(default_factory=dict)
    grant_consumed_at: datetime | None = None
    conclusion_evidence: FollowupConclusionEvidence | None = None
    delivery_id: str | None = None
    failure_code: str | None = None

    def __post_init__(self) -> None:
        for identity_value, identity_name in (
            (self.followup_id, "follow-up id"),
            (self.agent_id, "follow-up agent_id"),
            (self.conversation_id, "follow-up conversation_id"),
            (self.event_id, "follow-up event_id"),
            (self.job_id, "follow-up job_id"),
            (self.event_type, "follow-up event_type"),
        ):
            _text(identity_value, identity_name)
        if not isinstance(self.observation_source, FollowupObservationSource):
            raise TypeError("follow-up observation source is invalid")
        if (
            not isinstance(self.job_terminal_revision, int)
            or isinstance(self.job_terminal_revision, bool)
            or self.job_terminal_revision < 1
        ):
            raise ValueError("follow-up terminal job revision must be positive")
        payload = _bounded_mapping(
            self.event_payload,
            "follow-up event payload",
            MAX_FOLLOWUP_EVENT_BYTES,
        )
        computed_digest = (
            "sha256:" + sha256(canonical_json(payload).encode("utf-8")).hexdigest()
        )
        if self.payload_digest != computed_digest:
            raise ValueError("follow-up event payload digest does not match")
        _utc(self.received_at, "follow-up received_at")
        _utc(self.created_at, "follow-up created_at")
        _utc(self.updated_at, "follow-up updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("follow-up updated_at precedes creation")
        if not isinstance(self.grant, FollowupGrant):
            raise TypeError("follow-up grant is invalid")
        if not isinstance(self.execution_scope, ExecutionScope):
            raise TypeError("follow-up execution scope is invalid")
        if (
            self.agent_id != self.grant.agent_id
            or self.job_id != self.grant.job_id
            or self.conversation_id != self.grant.conversation_id
            or not self.grant.contains_scope(self.execution_scope)
        ):
            raise ValueError("follow-up grant, scope, and identity differ")
        if not isinstance(self.disposition, FollowupDisposition):
            raise TypeError("follow-up disposition is invalid")
        if (
            not isinstance(self.revision, int)
            or isinstance(self.revision, bool)
            or self.revision < 1
        ):
            raise ValueError("follow-up revision must be positive")
        if (
            not isinstance(self.attempt_count, int)
            or isinstance(self.attempt_count, bool)
            or not 0 <= self.attempt_count <= self.grant.max_attempts
        ):
            raise ValueError("follow-up attempt count is outside its grant")
        _optional_utc(self.lease_expires_at, "follow-up lease_expires_at")
        _optional_utc(self.run_bound_at, "follow-up run_bound_at")
        _optional_utc(self.run_terminal_at, "follow-up run_terminal_at")
        _optional_utc(self.grant_consumed_at, "follow-up grant_consumed_at")
        for cost_value, cost_name in (
            (self.reserved_cost_usd, "follow-up reserved cost"),
            (self.charged_cost_usd, "follow-up charged cost"),
        ):
            _money(cost_value, cost_name)
        for token_value, token_name in (
            (self.reserved_tokens, "follow-up reserved tokens"),
            (self.charged_tokens, "follow-up charged tokens"),
        ):
            if (
                not isinstance(token_value, int)
                or isinstance(token_value, bool)
                or token_value < 0
            ):
                raise ValueError(f"{token_name} must be non-negative")
        if (
            self.charged_cost_usd + self.reserved_cost_usd
            > self.grant.cumulative_max_cost_usd
            or self.charged_tokens + self.reserved_tokens
            > self.grant.cumulative_max_tokens
        ):
            raise ValueError("follow-up budget exceeds its cumulative ceiling")
        audit = _bounded_mapping(
            self.audit_context,
            "follow-up audit context",
            MAX_FOLLOWUP_AUDIT_BYTES,
        )
        live_claim = self.disposition in {
            FollowupDisposition.CLAIMED,
            FollowupDisposition.RUNNING,
            FollowupDisposition.RUN_TERMINAL_PENDING_FINALIZATION,
        }
        if live_claim and (
            self.claim_token is None
            or self.lease_expires_at is None
            or self.reserved_run_id is None
        ):
            raise ValueError("live follow-up disposition requires its exact claim")
        if self.disposition is FollowupDisposition.CLAIMED and self.run_bound_at:
            raise ValueError("claimed follow-up cannot already bind a run")
        if self.disposition in {
            FollowupDisposition.RUNNING,
            FollowupDisposition.RUN_TERMINAL_PENDING_FINALIZATION,
        } and (self.run_bound_at is None or not audit):
            raise ValueError("running follow-up requires reconstructible audit context")
        if (
            self.disposition is FollowupDisposition.RUN_TERMINAL_PENDING_FINALIZATION
            and self.run_terminal_at is None
        ):
            raise ValueError("pending finalization requires terminal run time")
        if self.disposition is FollowupDisposition.COMPLETED and (
            self.grant_consumed_at is None
            or self.conclusion_evidence is None
            or self.delivery_id is None
        ):
            raise ValueError("completed follow-up requires grant and delivery")
        if self.conclusion_evidence is not None:
            if not isinstance(self.conclusion_evidence, FollowupConclusionEvidence):
                raise TypeError("follow-up conclusion evidence is invalid")
            if (
                self.conclusion_evidence.run_id != self.reserved_run_id
                or self.conclusion_evidence.job_id != self.job_id
            ):
                raise ValueError("follow-up conclusion evidence identity differs")
        if self.failure_code is not None:
            _text(self.failure_code, "follow-up failure_code", maximum=128)
        object.__setattr__(self, "event_payload", payload)
        object.__setattr__(self, "audit_context", audit)


@dataclass(frozen=True, slots=True)
class InboxItem:
    delivery_id: str
    agent_id: str
    conversation_id: str
    subject: DeliverySubject
    resulting_run_id: str
    grant_id: str
    logical_key: str
    conclusion_digest: str
    payload: Mapping[str, object]
    sensitivity: ModelSensitivity
    destination: str
    destination_sensitivity_ceiling: ModelSensitivity
    state: DeliveryState
    created_at: datetime
    updated_at: datetime
    attempt_count: int = 1
    acknowledged_at: datetime | None = None
    terminal_error: str | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.delivery_id, "inbox delivery_id"),
            (self.agent_id, "inbox agent_id"),
            (self.conversation_id, "inbox conversation_id"),
            (self.resulting_run_id, "inbox resulting_run_id"),
            (self.grant_id, "inbox grant_id"),
            (self.logical_key, "inbox logical_key"),
            (self.destination, "inbox destination"),
        ):
            _text(value, name)
        if not isinstance(self.subject, DeliverySubject):
            raise TypeError("inbox subject is invalid")
        if _DIGEST.fullmatch(self.conclusion_digest) is None:
            raise ValueError("inbox conclusion digest is invalid")
        payload = _bounded_mapping(
            self.payload,
            "inbox payload",
            MAX_INBOX_PAYLOAD_BYTES,
        )
        if not isinstance(self.sensitivity, ModelSensitivity) or not isinstance(
            self.destination_sensitivity_ceiling,
            ModelSensitivity,
        ):
            raise TypeError("inbox sensitivity values are invalid")
        if not isinstance(self.state, DeliveryState):
            raise TypeError("inbox state is invalid")
        _utc(self.created_at, "inbox created_at")
        _utc(self.updated_at, "inbox updated_at")
        _optional_utc(self.acknowledged_at, "inbox acknowledged_at")
        if (
            not isinstance(self.attempt_count, int)
            or isinstance(self.attempt_count, bool)
            or self.attempt_count < 1
        ):
            raise ValueError("inbox attempt_count must be positive")
        if (self.state is DeliveryState.ACKNOWLEDGED) != (
            self.acknowledged_at is not None
        ):
            raise ValueError("inbox acknowledgment state is inconsistent")
        if self.state is DeliveryState.AVAILABLE and (
            self.sensitivity.routing_rank
            > self.destination_sensitivity_ceiling.routing_rank
        ):
            raise ValueError("inbox available state exceeds destination eligibility")
        if self.terminal_error is not None:
            _text(self.terminal_error, "inbox terminal_error", maximum=128)
        object.__setattr__(self, "payload", payload)


class FollowupIdentityConflictError(ValueError):
    """One event identity was reused with different bounded content."""


class FollowupCompletionConflictError(ValueError):
    """One terminal job already belongs to another completion owner."""


def terminal_job_event_payload(job: JobRun) -> FrozenJsonObject:
    """Project the bounded untrusted machine payload frozen at observation."""

    if not isinstance(job, JobRun) or not job.terminal:
        raise ValueError("terminal follow-up requires a terminal JobRun")
    result_payload: Mapping[str, object] | None = None
    if job.result is not None:
        rendered_summary = canonical_json(job.result.summary)
        summary_preview, summary_truncated = _utf8_preview(
            rendered_summary,
            MAX_FOLLOWUP_RESULT_PREVIEW_BYTES,
        )
        result_payload = {
            "result_id": job.result.result_id,
            "summary_digest": _mapping_digest(job.result.summary),
            "summary_preview": summary_preview,
            "summary_truncated": summary_truncated,
            "sensitivity": job.result.sensitivity.value,
            "artifact_ids": [ref.artifact_id for ref in job.result.artifact_refs],
        }
    return FrozenJsonObject.from_mapping(
        {
            "job_id": job.job_id,
            "job_revision": job.revision,
            "job_status": job.status.value,
            "terminal_at": job.terminal_at.isoformat() if job.terminal_at else None,
            "source_scope": {
                "count": len(job.source_ids),
                "digest": _mapping_digest({"source_ids": job.source_ids}),
            },
            "resource_scope": {
                "count": len(job.resource_ids),
                "digest": _mapping_digest({"resource_ids": job.resource_ids}),
            },
            "result": result_payload,
            "failure_code": job.failure_code,
        }
    )


def assess_followup_conclusion(
    followup: AutonomousFollowup,
    job: JobRun,
    transcript: Transcript,
    result: LoopExit,
) -> tuple[FollowupConclusionEvidence | None, str | None]:
    """Return exact evidence or one bounded owner-local failure attribution."""

    if (
        result.kind is not LoopExitKind.COMPLETED
        or result.final_text is None
        or result.run_id != followup.reserved_run_id
        or transcript.run.id != result.run_id
        or transcript.run.origin is not RunOrigin.JOB_EVENT
        or transcript.run.execution_scope != followup.execution_scope
        or job.job_id != followup.job_id
        or job.completion_binding is None
        or job.completion_binding.owner_id != followup.followup_id
    ):
        return None, "followup_conclusion_evidence_invalid"
    calls: dict[str, ToolCall] = {}
    results: dict[str, ToolResultBlock] = {}
    for message in transcript.messages:
        if message.role is MessageRole.ASSISTANT:
            calls.update((call.id, call) for call in message.tool_calls)
        elif message.role is MessageRole.TOOL:
            results.update(
                (block.call_id, block)
                for block in message.content
                if isinstance(block, ToolResultBlock)
            )

    inspection: tuple[ToolCall, ToolResultBlock] | None = None
    result_read: tuple[ToolCall, ToolResultBlock] | None = None
    for call_id, call in calls.items():
        block = results.get(call_id)
        if block is None or call.arguments.get("job_id") != job.job_id:
            continue
        capability_id = _result_capability_id(block)
        if capability_id == "jobs.inspect" and _valid_job_inspection(block, job):
            inspection = (call, block)
        elif capability_id == "jobs.read_results" and _valid_job_result_read(
            block,
            job,
        ):
            result_read = (call, block)
    if inspection is None and result_read is None:
        return None, "followup_inspection_and_result_evidence_missing"
    if inspection is None:
        return None, "followup_inspection_evidence_missing"
    if result_read is None:
        return None, "followup_result_evidence_missing"
    inspection_call, inspection_block = inspection
    result_call, result_block = result_read
    return (
        FollowupConclusionEvidence(
            run_id=result.run_id,
            job_id=job.job_id,
            job_revision=job.revision,
            inspection_call_id=inspection_call.id,
            inspection_result_digest=_mapping_digest(inspection_block.output),
            result_call_id=result_call.id,
            result_result_digest=_mapping_digest(result_block.output),
            job_result_id=None if job.result is None else job.result.result_id,
            report_digest=_text_digest(result.final_text),
        ),
        None,
    )


def inbox_report_projection(report: str) -> tuple[str, str, bool]:
    """Return stable report identity and one bounded user-visible preview."""

    preview, truncated = _utf8_preview(report, MAX_INBOX_REPORT_PREVIEW_BYTES)
    return _text_digest(report), preview, truncated


def _result_capability_id(block: ToolResultBlock) -> str | None:
    return block.capability_id


def _valid_job_inspection(block: ToolResultBlock, job: JobRun) -> bool:
    if block.is_error or block.output.get("kind") != "job.inspection":
        return False
    data = block.output.get("data")
    return (
        isinstance(data, Mapping)
        and data.get("job_id") == job.job_id
        and data.get("status") == job.status.value
        and data.get("execution_mode") == job.specification.execution_mode.value
        and tuple(data.get("source_ids", ())) == job.source_ids
        and tuple(data.get("resource_ids", ())) == job.resource_ids
        and data.get("result_available") is (job.result is not None)
        and data.get("specification_digest") == job.specification_digest
        and data.get("terminal_at")
        == (None if job.terminal_at is None else job.terminal_at.isoformat())
        and data.get("failure_code") == job.failure_code
    )


def _valid_job_result_read(block: ToolResultBlock, job: JobRun) -> bool:
    if job.result is not None:
        if block.is_error or block.output.get("kind") != "job.result":
            return False
        data = block.output.get("data")
        return (
            isinstance(data, Mapping)
            and data.get("job_id") == job.job_id
            and data.get("result_id") == job.result.result_id
            and data.get("summary") == job.result.summary
            and data.get("sensitivity") == job.result.sensitivity.value
            and data.get("completed_at") == job.result.completed_at.isoformat()
        )
    if not block.is_error:
        return False
    error = block.output.get("error")
    if not isinstance(error, Mapping) or error.get("code") != "job_result_not_ready":
        return False
    details = error.get("details")
    return isinstance(details, Mapping) and details.get("status") == job.status.value


def create_terminal_job_followup(
    job: JobRun,
    *,
    followup_id: str,
    grant_id: str,
    scope_id: str,
    received_at: datetime,
    allowed_capability_ids: tuple[str, ...],
    eligible_model_routes: tuple[str, ...],
    limits: LoopLimits,
) -> AutonomousFollowup:
    """Create the sole code-authored Stage C grant for one terminal Daita job."""

    if (
        not isinstance(job, JobRun)
        or not job.terminal
        or job.specification.execution_mode is not JobExecutionMode.DAITA
        or job.completion_binding is not None
    ):
        raise ValueError("only one unbound terminal Daita JobRun is eligible")
    if limits.max_estimated_cost_usd is None:
        raise ValueError("autonomous follow-up requires a finite cost budget")
    payload = terminal_job_event_payload(job)
    payload_digest = (
        "sha256:" + sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    )
    event_id = f"stage-c:{job.job_id}:{job.revision}"
    sensitivity = (
        job.specification.sensitivity if job.result is None else job.result.sensitivity
    )
    expires_at = received_at + timedelta(seconds=FOLLOWUP_EXPIRY_SECONDS)
    cumulative_cost = limits.max_estimated_cost_usd * MAX_FOLLOWUP_ATTEMPTS
    cumulative_tokens = limits.max_total_tokens * MAX_FOLLOWUP_ATTEMPTS
    grant = FollowupGrant(
        grant_id=grant_id,
        job_id=job.job_id,
        agent_id=job.agent_id,
        conversation_id=job.conversation_id,
        authorizing_principal=job.agent_id,
        allowed_terminal_job_observation=event_id,
        allowed_source_ids=job.source_ids,
        allowed_resource_ids=job.resource_ids,
        allowed_capability_ids=allowed_capability_ids,
        allowed_access_modes=frozenset({AccessMode.NONE, AccessMode.READ}),
        allowed_operational_effects=frozenset({OperationalEffect.NONE}),
        instruction_id=FOLLOWUP_INSTRUCTION_ID,
        instruction_digest=FOLLOWUP_INSTRUCTION_DIGEST,
        sensitivity_ceiling=sensitivity,
        delivery_sensitivity_ceiling=sensitivity,
        eligible_model_routes=eligible_model_routes,
        delivery_destination=f"conversation_inbox:{job.conversation_id}",
        max_successful_runs=1,
        max_attempts=MAX_FOLLOWUP_ATTEMPTS,
        per_run_max_cost_usd=limits.max_estimated_cost_usd,
        per_run_max_tokens=limits.max_total_tokens,
        cumulative_max_cost_usd=cumulative_cost,
        cumulative_max_tokens=cumulative_tokens,
        expires_at=expires_at,
    )
    scope = ExecutionScope(
        scope_id=scope_id,
        revision=1,
        agent_id=job.agent_id,
        principal_id=job.agent_id,
        grant_id=grant_id,
        job_id=job.job_id,
        job_revision=job.revision,
        allowed_source_ids=job.source_ids,
        allowed_resource_ids=job.resource_ids,
        allowed_capability_ids=allowed_capability_ids,
        allowed_access_modes=grant.allowed_access_modes,
        allowed_operational_effects=grant.allowed_operational_effects,
        sensitivity_ceiling=sensitivity,
        eligible_model_routes=eligible_model_routes,
        per_run_max_cost_usd=limits.max_estimated_cost_usd,
        per_run_max_tokens=limits.max_total_tokens,
        delivery_destination=grant.delivery_destination,
    )
    return AutonomousFollowup(
        followup_id=followup_id,
        agent_id=job.agent_id,
        conversation_id=job.conversation_id,
        event_id=event_id,
        observation_source=FollowupObservationSource.DAITA_JOB,
        job_id=job.job_id,
        job_terminal_revision=job.revision,
        event_type="job_run.terminal",
        event_payload=payload,
        payload_digest=payload_digest,
        received_at=received_at,
        grant=grant,
        execution_scope=scope,
        disposition=FollowupDisposition.AVAILABLE,
        created_at=received_at,
        updated_at=received_at,
    )


__all__ = [
    "AutonomousFollowup",
    "DeliverySubject",
    "DeliverySubjectKind",
    "DeliveryState",
    "FOLLOWUP_EXPIRY_SECONDS",
    "FOLLOWUP_INSTRUCTION",
    "FOLLOWUP_INSTRUCTION_DIGEST",
    "FOLLOWUP_INSTRUCTION_ID",
    "FOLLOWUP_LEASE_SECONDS",
    "FollowupCompletionConflictError",
    "FollowupConclusionEvidence",
    "FollowupDisposition",
    "FollowupGrant",
    "FollowupIdentityConflictError",
    "FollowupObservationSource",
    "InboxItem",
    "MAX_AUTONOMOUS_FOLLOWUPS_PER_AGENT",
    "MAX_CONVERSATION_INBOX_ITEMS_PER_AGENT",
    "MAX_FOLLOWUP_ATTEMPTS",
    "MAX_INBOX_PAGE_SIZE",
    "create_terminal_job_followup",
    "assess_followup_conclusion",
    "inbox_report_projection",
    "terminal_job_event_payload",
]
