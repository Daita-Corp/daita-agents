"""Own foreground admission and lifecycle views for scheduled routines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
from typing import Protocol

from .._json import FrozenJsonObject, canonical_json
from ..adapters.mcp import MCPBindingState, MCPServerBinding
from ..capabilities import (
    AccessMode,
    AutomationEligibility,
    CapabilityRegistry,
    OperationalEffect,
    RESERVED_TOOL_NAMES,
)
from ..catalog.models import CatalogResource, Sensitivity
from ..errors import DaitaError, ErrorRetryability
from ..llm.models import ModelSensitivity
from ..loop.models import (
    LoopExit,
    LoopExitKind,
    RunOrigin,
    Transcript,
    validate_completed_transcript,
)
from ..skills import Skill
from ..skills.capabilities import SKILL_VIEW_CAPABILITY_ID
from .models import (
    MAX_ROUTINE_HISTORY_PAGE_SIZE,
    MAX_ROUTINE_LIST_PAGE_SIZE,
    SCHEDULE_INTERPRETER_REVISION,
    MisfirePolicy,
    CalendarSchedule,
    IntervalSchedule,
    OnceSchedule,
    ReportingMode,
    ResourceRevisionPrecheck,
    RoutineControlAction,
    RoutineOccurrenceV1,
    RoutinePromotionEvidence,
    RoutineSchedule,
    RoutineSkillBinding,
    RoutineState,
    ScheduledRoutineInspection,
    ScheduledRoutineSummary,
    ScheduledRoutineV1,
    text_digest,
)
from .schedule import validate_schedule


class RoutineError(DaitaError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(
            message,
            error_code=code,
            retryability=ErrorRetryability.PERMANENT,
        )


class RoutineStore(Protocol):
    async def admit_scheduled_routine(
        self, routine: ScheduledRoutineV1
    ) -> ScheduledRoutineV1: ...

    async def load_scheduled_routine(
        self, agent_id: str, routine_id: str
    ) -> ScheduledRoutineV1 | None: ...

    async def list_scheduled_routines(
        self,
        agent_id: str,
        *,
        states: frozenset[RoutineState] = frozenset(),
        limit: int = MAX_ROUTINE_LIST_PAGE_SIZE,
    ) -> tuple[ScheduledRoutineV1, ...]: ...

    async def list_routine_occurrences(
        self,
        agent_id: str,
        routine_id: str,
        *,
        limit: int = MAX_ROUTINE_HISTORY_PAGE_SIZE,
    ) -> tuple[RoutineOccurrenceV1, ...]: ...

    async def revise_scheduled_routine(
        self,
        routine: ScheduledRoutineV1,
        *,
        expected_revision: int,
    ) -> ScheduledRoutineV1 | None: ...

    async def transition_scheduled_routine(
        self,
        agent_id: str,
        routine_id: str,
        *,
        expected_revision: int,
        state: RoutineState,
        transitioned_at: datetime,
    ) -> ScheduledRoutineV1 | None: ...

    async def claim_manual_routine_occurrence(
        self,
        agent_id: str,
        routine_id: str,
        *,
        expected_revision: int,
        authorized_control_call_id: str,
        claimed_at: datetime,
        claim_token: str,
    ) -> object | None: ...

    async def conversation_exists(
        self, agent_id: str, conversation_id: str
    ) -> bool: ...

    async def load(self, run_id: str) -> Transcript: ...

    async def result(self, run_id: str) -> LoopExit | None: ...

    async def load_resource(
        self, agent_id: str, resource_id: str
    ) -> CatalogResource | None: ...

    async def load_mcp_binding(
        self, agent_id: str, binding_id: str
    ) -> MCPServerBinding | None: ...


class RoutineCatalog(Protocol):
    async def readable_resource_ids(
        self,
        agent_id: str,
        source_ids: tuple[str, ...] = (),
    ) -> frozenset[str]: ...


class RoutineSkillStore(Protocol):
    async def read_skill_with_digest(self, name: str) -> tuple[Skill | None, str]: ...

    async def retain_current_skill(self, name: str, content_digest: str) -> Skill: ...

    async def read_retained_skill(
        self, name: str, content_digest: str
    ) -> Skill | None: ...


class RoutineOwner:
    """Validate exact routine authority and own agent-scoped transitions."""

    def __init__(
        self,
        *,
        agent_id: str,
        store: RoutineStore,
        catalog: RoutineCatalog,
        skills: RoutineSkillStore | None,
        eligible_model_routes: tuple[str, ...],
        maximum_per_run_tokens: int,
        maximum_per_run_cost_usd: Decimal | None,
        clock: Callable[[], datetime],
    ) -> None:
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("routine owner agent_id must be non-empty text")
        if not callable(clock):
            raise TypeError("routine owner clock must be callable")
        routes = tuple(sorted(set(eligible_model_routes)))
        if any(not isinstance(item, str) or not item for item in routes):
            raise ValueError("routine owner model routes are invalid")
        if maximum_per_run_tokens < 1:
            raise ValueError("routine owner token limit must be positive")
        if maximum_per_run_cost_usd is not None and (
            not maximum_per_run_cost_usd.is_finite() or maximum_per_run_cost_usd < 0
        ):
            raise ValueError("routine owner cost limit is invalid")
        self.agent_id = agent_id
        self._store = store
        self._catalog = catalog
        self._capabilities: CapabilityRegistry | None = None
        self._skills = skills
        self._eligible_model_routes = routes
        self._maximum_per_run_tokens = maximum_per_run_tokens
        self._maximum_per_run_cost_usd = maximum_per_run_cost_usd
        self._clock = clock
        self._wake: Callable[[], None] | None = None

    def bind_capability_registry(self, capabilities: CapabilityRegistry) -> None:
        """Bind the one complete immutable registry during composition."""

        if self._capabilities is not None:
            raise RuntimeError("routine owner capability registry is already bound")
        if not isinstance(capabilities, CapabilityRegistry):
            raise TypeError("routine owner requires CapabilityRegistry")
        self._capabilities = capabilities

    def bind_wake(self, wake: Callable[[], None]) -> None:
        if self._wake is not None:
            raise RuntimeError("routine owner wake callback is already bound")
        if not callable(wake):
            raise TypeError("routine wake callback must be callable")
        self._wake = wake

    async def prepare_create(
        self,
        *,
        run_id: str,
        conversation_id: str,
        call_id: str,
        title: str,
        authorized_instruction: str,
        schedule: RoutineSchedule,
        misfire_policy: MisfirePolicy,
        reporting_mode: ReportingMode,
        precheck: ResourceRevisionPrecheck | None,
        allowed_source_ids: tuple[str, ...],
        allowed_connector_binding_ids: tuple[str, ...],
        allowed_resource_ids: tuple[str, ...],
        allowed_capability_ids: tuple[str, ...],
        sensitivity_ceiling: ModelSensitivity,
        eligible_model_routes: tuple[str, ...],
        per_run_max_tokens: int,
        per_run_max_cost_usd: Decimal,
        cumulative_max_tokens: int,
        cumulative_max_cost_usd: Decimal,
        cumulative_max_attempts: int,
        cumulative_max_occurrences: int,
        maximum_consecutive_failures: int,
        expires_at: datetime,
        skill_names: tuple[str, ...],
        basis_run_id: str | None,
    ) -> ScheduledRoutineV1:
        origin = await self._owned_run(run_id, conversation_id)
        promotion = (
            None
            if basis_run_id is None
            else await self._promotion_evidence(
                basis_run_id,
                conversation_id=conversation_id,
                instruction=authorized_instruction,
                allowed_capability_ids=allowed_capability_ids,
            )
        )
        routine_hash = sha256(
            f"{self.agent_id}\x00{run_id}\x00{call_id}".encode("utf-8")
        ).hexdigest()[:32]
        access_modes = self._access_modes(allowed_capability_ids)
        skill_bindings = await self._prepare_skill_bindings(
            skill_names,
            attached_at=origin.run.created_at,
        )
        record = ScheduledRoutineV1(
            routine_id=f"routine-{routine_hash}",
            agent_id=self.agent_id,
            conversation_id=conversation_id,
            owner_principal_id=f"agent:{self.agent_id}",
            title=title,
            authorized_instruction=authorized_instruction,
            instruction_digest=text_digest(authorized_instruction),
            schedule=schedule,
            schedule_interpreter_revision=SCHEDULE_INTERPRETER_REVISION,
            misfire_policy=misfire_policy,
            reporting_mode=reporting_mode,
            precheck=precheck,
            last_acknowledged_precheck_observation=None,
            allowed_source_ids=allowed_source_ids,
            allowed_connector_binding_ids=allowed_connector_binding_ids,
            allowed_resource_ids=allowed_resource_ids,
            allowed_capability_ids=allowed_capability_ids,
            allowed_access_modes=access_modes,
            allowed_operational_effects=frozenset({OperationalEffect.NONE}),
            sensitivity_ceiling=sensitivity_ceiling,
            eligible_model_routes=eligible_model_routes,
            skill_bindings=skill_bindings,
            delivery_destination=f"conversation_inbox:{conversation_id}",
            per_run_max_tokens=per_run_max_tokens,
            per_run_max_cost_usd=per_run_max_cost_usd,
            cumulative_max_tokens=cumulative_max_tokens,
            cumulative_max_cost_usd=cumulative_max_cost_usd,
            cumulative_max_attempts=cumulative_max_attempts,
            cumulative_max_occurrences=cumulative_max_occurrences,
            reserved_tokens=0,
            reserved_cost_usd=Decimal("0"),
            charged_tokens=0,
            charged_cost_usd=Decimal("0"),
            attempt_count=0,
            occurrence_count=0,
            maximum_consecutive_failures=maximum_consecutive_failures,
            consecutive_failures=0,
            expires_at=expires_at,
            next_due_at=None,
            active_occurrence_id=None,
            last_occurrence_id=None,
            last_delivery_id=None,
            promotion_evidence=promotion,
            state=RoutineState.ACTIVE,
            revision=1,
            created_at=origin.run.created_at,
            updated_at=origin.run.created_at,
        )
        await self.proposal_authority_snapshot(record)
        return record

    async def admit(self, routine: ScheduledRoutineV1) -> ScheduledRoutineV1:
        await self._retain_bindings(routine.skill_bindings)
        await self.authority_snapshot(routine)
        try:
            stored = await self._store.admit_scheduled_routine(routine)
        except ValueError as error:
            raise self._store_error(error, "routine_admission_failed") from error
        self._notify()
        return stored

    async def prepare_revision(
        self,
        current: ScheduledRoutineV1,
        *,
        run_id: str,
        conversation_id: str,
        title: str,
        authorized_instruction: str,
        schedule: RoutineSchedule,
        misfire_policy: MisfirePolicy,
        reporting_mode: ReportingMode,
        precheck: ResourceRevisionPrecheck | None,
        allowed_source_ids: tuple[str, ...],
        allowed_connector_binding_ids: tuple[str, ...],
        allowed_resource_ids: tuple[str, ...],
        allowed_capability_ids: tuple[str, ...],
        sensitivity_ceiling: ModelSensitivity,
        eligible_model_routes: tuple[str, ...],
        per_run_max_tokens: int,
        per_run_max_cost_usd: Decimal,
        cumulative_max_tokens: int,
        cumulative_max_cost_usd: Decimal,
        cumulative_max_attempts: int,
        cumulative_max_occurrences: int,
        maximum_consecutive_failures: int,
        expires_at: datetime,
        skill_names: tuple[str, ...],
        basis_run_id: str | None,
    ) -> ScheduledRoutineV1:
        origin = await self._owned_run(run_id, conversation_id)
        if current.conversation_id != conversation_id:
            raise RoutineError(
                "routine_conversation_mismatch",
                "The routine belongs to another conversation.",
            )
        if origin.run.created_at < current.updated_at:
            raise RoutineError(
                "routine_revision_changed",
                "The routine changed after this foreground run began.",
            )
        promotion = (
            current.promotion_evidence
            if basis_run_id is None
            else await self._promotion_evidence(
                basis_run_id,
                conversation_id=conversation_id,
                instruction=authorized_instruction,
                allowed_capability_ids=allowed_capability_ids,
            )
        )
        skill_bindings = await self._prepare_skill_bindings(
            skill_names,
            attached_at=origin.run.created_at,
        )
        revised = replace(
            current,
            title=title,
            authorized_instruction=authorized_instruction,
            instruction_digest=text_digest(authorized_instruction),
            schedule=schedule,
            misfire_policy=misfire_policy,
            reporting_mode=reporting_mode,
            precheck=precheck,
            last_acknowledged_precheck_observation=None,
            allowed_source_ids=allowed_source_ids,
            allowed_connector_binding_ids=allowed_connector_binding_ids,
            allowed_resource_ids=allowed_resource_ids,
            allowed_capability_ids=allowed_capability_ids,
            allowed_access_modes=self._access_modes(allowed_capability_ids),
            sensitivity_ceiling=sensitivity_ceiling,
            eligible_model_routes=eligible_model_routes,
            skill_bindings=skill_bindings,
            per_run_max_tokens=per_run_max_tokens,
            per_run_max_cost_usd=per_run_max_cost_usd,
            cumulative_max_tokens=cumulative_max_tokens,
            cumulative_max_cost_usd=cumulative_max_cost_usd,
            cumulative_max_attempts=cumulative_max_attempts,
            cumulative_max_occurrences=cumulative_max_occurrences,
            maximum_consecutive_failures=maximum_consecutive_failures,
            expires_at=expires_at,
            promotion_evidence=promotion,
            revision=current.revision + 1,
            updated_at=origin.run.created_at,
        )
        await self.proposal_authority_snapshot(revised)
        return revised

    async def revise(
        self,
        routine: ScheduledRoutineV1,
        *,
        expected_revision: int,
    ) -> ScheduledRoutineV1:
        await self._retain_bindings(routine.skill_bindings)
        await self.authority_snapshot(routine)
        try:
            stored = await self._store.revise_scheduled_routine(
                routine,
                expected_revision=expected_revision,
            )
        except ValueError as error:
            raise self._store_error(error, "routine_revision_failed") from error
        if stored is None:
            raise RoutineError(
                "routine_revision_changed",
                "The routine changed or is no longer owned by this agent.",
            )
        self._notify()
        return stored

    async def list(
        self,
        *,
        states: frozenset[RoutineState] = frozenset(),
        limit: int = MAX_ROUTINE_LIST_PAGE_SIZE,
    ) -> tuple[ScheduledRoutineSummary, ...]:
        records = await self._store.list_scheduled_routines(
            self.agent_id,
            states=states,
            limit=limit,
        )
        return tuple(_summary(item) for item in records)

    async def inspect(self, routine_id: str) -> ScheduledRoutineInspection | None:
        routine = await self._load_owned(routine_id)
        if routine is None:
            return None
        occurrences = await self._store.list_routine_occurrences(
            self.agent_id,
            routine_id,
            limit=MAX_ROUTINE_HISTORY_PAGE_SIZE,
        )
        return ScheduledRoutineInspection(routine, tuple(occurrences))

    async def control(
        self,
        routine_id: str,
        *,
        expected_revision: int,
        action: RoutineControlAction,
        authorized_control_call_id: str,
    ) -> ScheduledRoutineV1:
        current = await self._load_owned(routine_id)
        if current is None or current.revision != expected_revision:
            raise RoutineError(
                "routine_revision_changed",
                "The routine changed or is no longer owned by this agent.",
            )
        now = self._clock()
        if action is RoutineControlAction.RUN_NOW:
            if current.state is not RoutineState.ACTIVE:
                raise RoutineError(
                    "routine_not_active", "Only an active routine can run now."
                )
            await self.authority_snapshot(current)
            claim_hash = sha256(
                f"{self.agent_id}\x00{routine_id}\x00{authorized_control_call_id}".encode(
                    "utf-8"
                )
            ).hexdigest()[:32]
            try:
                occurrence = await self._store.claim_manual_routine_occurrence(
                    self.agent_id,
                    routine_id,
                    expected_revision=expected_revision,
                    authorized_control_call_id=authorized_control_call_id,
                    claimed_at=now,
                    claim_token=f"routine-claim-{claim_hash}",
                )
            except ValueError as error:
                raise self._store_error(error, "routine_run_now_failed") from error
            if occurrence is None:
                raise RoutineError(
                    "routine_run_now_conflict",
                    "The routine cannot start a manual occurrence in its current state.",
                )
            self._notify()
            reloaded = await self._load_owned(routine_id)
            if reloaded is None:
                raise RoutineError("routine_not_found", "The routine no longer exists.")
            return reloaded
        target = {
            RoutineControlAction.PAUSE: RoutineState.PAUSED,
            RoutineControlAction.RESUME: RoutineState.ACTIVE,
            RoutineControlAction.DISABLE: RoutineState.DISABLED,
        }[action]
        if (
            action is RoutineControlAction.PAUSE
            and current.state is not RoutineState.ACTIVE
        ):
            raise RoutineError(
                "routine_not_active", "Only an active routine can be paused."
            )
        if action is RoutineControlAction.RESUME and current.state not in {
            RoutineState.PAUSED,
            RoutineState.NEEDS_ATTENTION,
        }:
            raise RoutineError("routine_not_paused", "This routine cannot be resumed.")
        if action is RoutineControlAction.DISABLE and current.state in {
            RoutineState.DISABLED,
            RoutineState.COMPLETED,
            RoutineState.EXPIRED,
        }:
            raise RoutineError("routine_terminal", "This routine is already terminal.")
        if target is RoutineState.ACTIVE:
            await self.authority_snapshot(current)
        try:
            updated = await self._store.transition_scheduled_routine(
                self.agent_id,
                routine_id,
                expected_revision=expected_revision,
                state=target,
                transitioned_at=now,
            )
        except ValueError as error:
            raise self._store_error(error, "routine_control_failed") from error
        if updated is None:
            raise RoutineError(
                "routine_revision_changed", "The routine changed during control."
            )
        self._notify()
        return updated

    async def authority_snapshot(self, routine: ScheduledRoutineV1) -> FrozenJsonObject:
        """Revalidate the complete retained routine contract."""

        return await self._authority_snapshot(
            routine,
            allow_unretained_skills=False,
        )

    async def proposal_authority_snapshot(
        self, routine: ScheduledRoutineV1
    ) -> FrozenJsonObject:
        """Revalidate a proposal before its exact skill bytes are retained."""

        return await self._authority_snapshot(
            routine,
            allow_unretained_skills=True,
        )

    async def _authority_snapshot(
        self,
        routine: ScheduledRoutineV1,
        *,
        allow_unretained_skills: bool,
    ) -> FrozenJsonObject:
        """Revalidate current authority and return its bounded fingerprint."""

        if routine.agent_id != self.agent_id:
            raise RoutineError("routine_owner_mismatch", "The routine owner changed.")
        validate_schedule(routine.schedule)
        if self._clock() >= routine.expires_at:
            raise RoutineError("routine_expired", "The routine has expired.")
        if not await self._store.conversation_exists(
            self.agent_id, routine.conversation_id
        ):
            raise RoutineError(
                "routine_conversation_missing",
                "The exact destination conversation is unavailable.",
            )
        if not set(routine.eligible_model_routes) <= set(self._eligible_model_routes):
            raise RoutineError(
                "routine_model_route_revoked",
                "One or more exact model routes are no longer eligible.",
            )
        if routine.per_run_max_tokens > self._maximum_per_run_tokens:
            raise RoutineError(
                "routine_token_budget_exceeded",
                "The routine per-run token ceiling exceeds the host ceiling.",
            )
        if (
            self._maximum_per_run_cost_usd is not None
            and routine.per_run_max_cost_usd > self._maximum_per_run_cost_usd
        ):
            raise RoutineError(
                "routine_cost_budget_exceeded",
                "The routine per-run cost ceiling exceeds the host ceiling.",
            )
        capabilities = self._require_capability_registry()
        try:
            capability_ids = capabilities.validate_execution_scope_grant(
                routine.allowed_capability_ids,
                allowed_access_modes=routine.allowed_access_modes,
                allowed_operational_effects=frozenset({OperationalEffect.NONE}),
            )
        except (KeyError, ValueError) as error:
            raise RoutineError(
                "routine_capability_invalid",
                "The exact capability ceiling is no longer admitted.",
            ) from error
        capability_facts: list[dict[str, object]] = []
        mcp_capability_ids: set[str] = set()
        for capability_id in capability_ids:
            capability = capabilities.capability(capability_id)
            if (
                capability.automation_eligibility
                is not AutomationEligibility.SCHEDULED_DIRECT
                or capability.operational_effect is not OperationalEffect.NONE
                or capability.access_mode not in {AccessMode.NONE, AccessMode.READ}
            ):
                raise RoutineError(
                    "routine_capability_interactive_only",
                    "A requested capability is not eligible for scheduled execution.",
                )
            owner_id = capabilities.resolve_domain_owner(capability_id)
            if owner_id == "mcp":
                mcp_capability_ids.add(capability_id)
            capability_facts.append(
                {
                    "capability_id": capability_id,
                    "contract_digest": capabilities.contract_digest(capability_id),
                }
            )
        if bool(routine.skill_bindings) != (SKILL_VIEW_CAPABILITY_ID in capability_ids):
            raise RoutineError(
                "routine_skill_capability_invalid",
                "Pinned skills and the exact skill-view capability must be present together.",
            )
        readable = await self._catalog.readable_resource_ids(
            self.agent_id, routine.allowed_source_ids
        )
        if not set(routine.allowed_resource_ids) <= readable:
            raise RoutineError(
                "routine_resource_revoked",
                "One or more exact resources are no longer readable.",
            )
        resource_facts: list[dict[str, object]] = []
        for resource_id in routine.allowed_resource_ids:
            resource = await self._store.load_resource(self.agent_id, resource_id)
            if (
                resource is None
                or resource.agent_id != self.agent_id
                or resource.source_id not in routine.allowed_source_ids
            ):
                raise RoutineError(
                    "routine_resource_identity_changed",
                    "An exact resource identity or source binding changed.",
                )
            sensitivity = _model_sensitivity(resource.sensitivity)
            if sensitivity.routing_rank > routine.sensitivity_ceiling.routing_rank:
                raise RoutineError(
                    "routine_sensitivity_exceeded",
                    "The resource sensitivity exceeds the routine ceiling.",
                )
            resource_facts.append(
                {
                    "resource_id": resource.id,
                    "source_id": resource.source_id,
                    "revision": resource.current_revision,
                    "sensitivity": resource.sensitivity.value,
                }
            )
        binding_facts: list[dict[str, object]] = []
        admitted_mcp_capabilities: set[str] = set()
        for binding_id in routine.allowed_connector_binding_ids:
            binding = await self._store.load_mcp_binding(self.agent_id, binding_id)
            if binding is None or binding.state is not MCPBindingState.ACTIVE:
                raise RoutineError(
                    "routine_mcp_binding_revoked",
                    "An exact MCP binding is not currently active.",
                )
            selected_tools = tuple(
                tool
                for tool in binding.tools
                if tool.capability_id in routine.allowed_capability_ids
            )
            admitted_mcp_capabilities.update(
                tool.capability_id for tool in selected_tools
            )
            for tool in selected_tools:
                if (
                    tool.result_sensitivity.routing_rank
                    > routine.sensitivity_ceiling.routing_rank
                ):
                    raise RoutineError(
                        "routine_sensitivity_exceeded",
                        "An MCP result sensitivity exceeds the routine ceiling.",
                    )
            binding_facts.append(
                {
                    "binding_id": binding.binding_id,
                    "revision": binding.revision,
                    "tools": tuple(
                        {
                            "capability_id": tool.capability_id,
                            "input_schema_digest": tool.input_schema_digest,
                            "output_schema_digest": tool.output_schema_digest,
                        }
                        for tool in selected_tools
                    ),
                }
            )
        if mcp_capability_ids != admitted_mcp_capabilities:
            raise RoutineError(
                "routine_mcp_capability_unbound",
                "The exact MCP capability ceiling is not bound by the retained servers.",
            )
        if routine.precheck is not None:
            if (
                routine.precheck.source_id not in routine.allowed_source_ids
                or routine.precheck.resource_id not in routine.allowed_resource_ids
            ):
                raise RoutineError(
                    "routine_precheck_scope_invalid",
                    "The precheck is outside the exact resource ceiling.",
                )
            try:
                expected = capabilities.contract_digest(routine.precheck.capability_id)
            except KeyError as error:
                raise RoutineError(
                    "routine_precheck_unavailable",
                    "The exact precheck capability is unavailable.",
                ) from error
            if expected != routine.precheck.contract_digest:
                raise RoutineError(
                    "routine_precheck_contract_changed",
                    "The exact precheck contract changed.",
                )
        skill_facts: list[dict[str, object]] = []
        for skill_binding in routine.skill_bindings:
            if self._skills is None:
                raise RoutineError(
                    "routine_skill_store_unavailable",
                    "Pinned skill content is unavailable.",
                )
            retained = await self._skills.read_retained_skill(
                skill_binding.skill_name,
                skill_binding.content_digest,
            )
            if retained is None and allow_unretained_skills:
                current, digest = await self._skills.read_skill_with_digest(
                    skill_binding.skill_name
                )
                if (
                    current is not None
                    and f"sha256:{digest}" == skill_binding.content_digest
                ):
                    retained = current
            if retained is None:
                raise RoutineError(
                    "routine_skill_content_missing",
                    "Exact retained skill content is missing or changed.",
                )
            skill_facts.append(
                {
                    "skill_name": skill_binding.skill_name,
                    "content_digest": skill_binding.content_digest,
                }
            )
        return FrozenJsonObject.from_mapping(
            {
                "routine": _routine_proposal_payload(routine),
                "authority": {
                    "capabilities": tuple(capability_facts),
                    "resources": tuple(resource_facts),
                    "bindings": tuple(binding_facts),
                    "skills": tuple(skill_facts),
                    "eligible_model_routes": self._eligible_model_routes,
                },
            }
        )

    async def _owned_run(self, run_id: str, conversation_id: str) -> Transcript:
        try:
            transcript = await self._store.load(run_id)
        except KeyError as error:
            raise RoutineError(
                "routine_origin_run_missing", "The origin run is unknown."
            ) from error
        if (
            transcript.run.agent_id != self.agent_id
            or transcript.run.conversation_id != conversation_id
            or transcript.run.start is None
            or transcript.run.start.origin is not RunOrigin.USER
        ):
            raise RoutineError(
                "routine_origin_run_mismatch",
                "Routine management requires this exact foreground conversation run.",
            )
        return transcript

    async def _promotion_evidence(
        self,
        basis_run_id: str,
        *,
        conversation_id: str,
        instruction: str,
        allowed_capability_ids: tuple[str, ...],
    ) -> RoutinePromotionEvidence:
        if _contains_reference_marker(instruction):
            raise RoutineError(
                "routine_instruction_not_self_contained",
                "The authorized instruction contains a run-bound reference marker.",
            )
        try:
            transcript = await self._store.load(basis_run_id)
            result = await self._store.result(basis_run_id)
        except KeyError as error:
            raise RoutineError(
                "routine_basis_run_missing", "The basis run is unknown."
            ) from error
        if (
            transcript.run.agent_id != self.agent_id
            or transcript.run.conversation_id != conversation_id
            or result is None
            or result.kind is not LoopExitKind.COMPLETED
            or result.conversation_id != conversation_id
        ):
            raise RoutineError(
                "routine_basis_run_invalid",
                "The basis run is not a completed run in this exact conversation.",
            )
        try:
            validate_completed_transcript(transcript, result)
        except ValueError as error:
            raise RoutineError(
                "routine_basis_run_invalid",
                "The basis run transcript is not a valid completed run.",
            ) from error
        capabilities = self._require_capability_registry()
        executed: set[str] = set()
        for message in transcript.messages:
            for call in message.tool_calls:
                if _contains_reference_marker(call.arguments):
                    raise RoutineError(
                        "routine_basis_run_has_tool_ref",
                        "The basis run used a run-bound tool reference.",
                    )
                if call.name in RESERVED_TOOL_NAMES:
                    continue
                try:
                    capability = capabilities.tool_capability(call.name)
                except KeyError as error:
                    raise RoutineError(
                        "routine_basis_lineage_unknown",
                        "The basis run contains unknown capability lineage.",
                    ) from error
                executed.add(capability.id)
        if not executed or not executed <= set(allowed_capability_ids):
            raise RoutineError(
                "routine_basis_lineage_outside_scope",
                "The basis run capability lineage is not contained by the routine.",
            )
        terminal_material = canonical_json(
            {
                "run_id": result.run_id,
                "conversation_id": result.conversation_id,
                "kind": result.kind.value,
                "reason": result.reason,
                "created_at": result.created_at.isoformat(),
                "final_text": result.final_text,
                "steps": result.steps,
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
            }
        )
        return RoutinePromotionEvidence(
            basis_run_id=basis_run_id,
            terminal_result_digest=(
                "sha256:" + sha256(terminal_material.encode("utf-8")).hexdigest()
            ),
            executed_capability_ids=tuple(sorted(executed)),
        )

    async def _prepare_skill_bindings(
        self,
        skill_names: tuple[str, ...],
        *,
        attached_at: datetime,
    ) -> tuple[RoutineSkillBinding, ...]:
        names = tuple(sorted(skill_names))
        if len(names) != len(set(names)):
            raise RoutineError(
                "routine_skill_binding_invalid",
                "Routine skill names cannot be duplicated.",
            )
        if names and self._skills is None:
            raise RoutineError(
                "routine_skill_store_unavailable",
                "Pinned skill content is unavailable.",
            )
        bindings: list[RoutineSkillBinding] = []
        for name in names:
            assert self._skills is not None
            skill, digest = await self._skills.read_skill_with_digest(name)
            if skill is None:
                raise RoutineError(
                    "routine_skill_not_found",
                    "A requested skill is not currently available.",
                )
            bindings.append(
                RoutineSkillBinding(
                    skill_name=name,
                    skill_revision=1,
                    content_digest=f"sha256:{digest}",
                    attached_by_principal=f"agent:{self.agent_id}",
                    attached_at=attached_at,
                )
            )
        return tuple(bindings)

    async def _retain_bindings(
        self,
        bindings: tuple[RoutineSkillBinding, ...],
    ) -> None:
        if bindings and self._skills is None:
            raise RoutineError(
                "routine_skill_store_unavailable",
                "Pinned skill content is unavailable.",
            )
        for binding in bindings:
            assert self._skills is not None
            try:
                await self._skills.retain_current_skill(
                    binding.skill_name,
                    binding.content_digest,
                )
            except Exception as error:
                raise RoutineError(
                    "routine_skill_retention_failed",
                    "Exact skill content could not be retained.",
                ) from error

    def _access_modes(self, capability_ids: tuple[str, ...]) -> frozenset[AccessMode]:
        capabilities = self._require_capability_registry()
        try:
            modes = frozenset(
                capabilities.capability(item).access_mode for item in capability_ids
            )
        except KeyError as error:
            raise RoutineError(
                "routine_capability_invalid", "A requested capability is unknown."
            ) from error
        return modes

    def _require_capability_registry(self) -> CapabilityRegistry:
        capabilities = self._capabilities
        if capabilities is None:
            raise RuntimeError("routine owner capability registry is not bound")
        return capabilities

    async def _load_owned(self, routine_id: str) -> ScheduledRoutineV1 | None:
        if not isinstance(routine_id, str) or not routine_id:
            raise ValueError("routine_id must be non-empty text")
        return await self._store.load_scheduled_routine(self.agent_id, routine_id)

    @staticmethod
    def _store_error(error: ValueError, fallback: str) -> RoutineError:
        code = str(error)
        if not code.startswith("routine_"):
            code = fallback
        return RoutineError(code, "The routine transition failed its current contract.")

    def _notify(self) -> None:
        if self._wake is not None:
            self._wake()


def _summary(routine: ScheduledRoutineV1) -> ScheduledRoutineSummary:
    return ScheduledRoutineSummary(
        routine_id=routine.routine_id,
        title=routine.title,
        state=routine.state,
        schedule_kind=routine.schedule.kind,
        next_due_at=routine.next_due_at,
        revision=routine.revision,
        occurrence_count=routine.occurrence_count,
        consecutive_failures=routine.consecutive_failures,
    )


def _model_sensitivity(value: Sensitivity) -> ModelSensitivity:
    if value is Sensitivity.UNKNOWN:
        raise RoutineError(
            "routine_sensitivity_unknown",
            "A resource has unknown sensitivity and cannot be scheduled.",
        )
    return ModelSensitivity(value.value)


def _contains_reference_marker(value: object) -> bool:
    if isinstance(value, str):
        folded = value.casefold()
        return (
            "tool_ref" in folded or "conversation_ref" in folded or "run_ref" in folded
        )
    if isinstance(value, dict) or isinstance(value, FrozenJsonObject):
        return any(
            _contains_reference_marker(key) or _contains_reference_marker(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return any(_contains_reference_marker(item) for item in value)
    return False


def _schedule_payload(schedule: RoutineSchedule) -> dict[str, object]:
    payload: dict[str, object] = {"kind": schedule.kind.value}
    if isinstance(schedule, OnceSchedule):
        payload["exact_at"] = schedule.exact_at.isoformat()
    elif isinstance(schedule, IntervalSchedule):
        payload.update(
            {
                "interval_seconds": schedule.interval_seconds,
                "anchor_at": schedule.anchor_at.isoformat(),
            }
        )
    else:
        assert isinstance(schedule, CalendarSchedule)
        payload.update(
            {
                "timezone": schedule.timezone,
                "hour": schedule.hour,
                "minute": schedule.minute,
                "day_selector": schedule.day_selector.value,
                "weekdays": schedule.weekdays,
                "month_days": schedule.month_days,
                "months": schedule.months,
                "nonexistent_time_policy": schedule.nonexistent_time_policy.value,
                "ambiguous_time_policy": schedule.ambiguous_time_policy.value,
            }
        )
    return payload


def _routine_proposal_payload(routine: ScheduledRoutineV1) -> dict[str, object]:
    return {
        "routine_id": routine.routine_id,
        "conversation_id": routine.conversation_id,
        "title": routine.title,
        "authorized_instruction": routine.authorized_instruction,
        "instruction_digest": routine.instruction_digest,
        "schedule": _schedule_payload(routine.schedule),
        "misfire_policy": routine.misfire_policy.value,
        "reporting_mode": routine.reporting_mode.value,
        "precheck": (
            None
            if routine.precheck is None
            else {
                "capability_id": routine.precheck.capability_id,
                "contract_digest": routine.precheck.contract_digest,
                "source_id": routine.precheck.source_id,
                "resource_id": routine.precheck.resource_id,
            }
        ),
        "allowed_source_ids": routine.allowed_source_ids,
        "allowed_connector_binding_ids": routine.allowed_connector_binding_ids,
        "allowed_resource_ids": routine.allowed_resource_ids,
        "allowed_capability_ids": routine.allowed_capability_ids,
        "allowed_access_modes": tuple(
            sorted(item.value for item in routine.allowed_access_modes)
        ),
        "sensitivity_ceiling": routine.sensitivity_ceiling.value,
        "eligible_model_routes": routine.eligible_model_routes,
        "skill_bindings": tuple(
            {
                "skill_name": binding.skill_name,
                "skill_revision": binding.skill_revision,
                "content_digest": binding.content_digest,
            }
            for binding in routine.skill_bindings
        ),
        "delivery_destination": routine.delivery_destination,
        "per_run_max_tokens": routine.per_run_max_tokens,
        "per_run_max_cost_usd": str(routine.per_run_max_cost_usd),
        "cumulative_max_tokens": routine.cumulative_max_tokens,
        "cumulative_max_cost_usd": str(routine.cumulative_max_cost_usd),
        "cumulative_max_attempts": routine.cumulative_max_attempts,
        "cumulative_max_occurrences": routine.cumulative_max_occurrences,
        "maximum_consecutive_failures": routine.maximum_consecutive_failures,
        "expires_at": routine.expires_at.isoformat(),
        "basis_run_id": (
            None
            if routine.promotion_evidence is None
            else routine.promotion_evidence.basis_run_id
        ),
        "revision": routine.revision,
    }


__all__ = [
    "RoutineCatalog",
    "RoutineError",
    "RoutineOwner",
    "RoutineStore",
]
