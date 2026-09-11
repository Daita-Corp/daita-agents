"""Shared helpers extracted from ``test_owner.py``."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest

from daita.adapters.mcp import MCPServerBinding
from daita.capabilities import (
    AccessMode,
    AutomationEligibility,
    Capability,
    CapabilityDeclarations,
    CapabilityRegistry,
    ExecutionContractBindings,
    Executor,
    OperationalEffect,
    ToolboxId,
    ToolExecution,
    ToolLoadMode,
    ToolOutput,
    ToolPresentation,
    ToolTextTrust,
    ToolView,
)
from daita.catalog.models import (
    CatalogResource,
    ResourceKind,
    Sensitivity,
    catalog_resource_id,
)
from daita.distribution import DistributionOwner, conversation_inbox_destination_id
from daita.distribution.owner import DistributionStore
from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelSensitivity,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.loop.models import LoopExit, LoopExitKind, RunInput, Transcript
from daita.routines.capabilities import (
    ROUTINE_CREATE_CAPABILITY_ID,
    ROUTINE_LIST_TOOL_NAME,
    routine_capability_declarations,
)
from daita.routines.models import (
    IntervalSchedule,
    MisfirePolicy,
    ReportingMode,
    RoutineControlAction,
    RoutineOccurrence,
    RoutineState,
    ScheduledRoutine,
)
from daita.routines.owner import RoutineError, RoutineOwner
from daita.routines.schedule import first_slot
from daita.skills import SkillStore
from daita.skills.capabilities import (
    SKILL_DOMAIN_OWNER_ID,
    SKILL_VIEW_CAPABILITY_ID,
    skill_declarations,
)
from tests.support.distribution import (
    inbox_distribution_plan,
    no_artifact_outcome_contract,
)

NOW = datetime(2026, 8, 28, 12, tzinfo=UTC)


class _ReadExecutor:
    executor_id = "test.read.executor"

    async def execute(self, request: ToolExecution) -> ToolOutput:
        del request
        return ToolOutput(kind="test.read", data={"ok": True})


class _InteractiveExecutor:
    executor_id = "test.interactive.executor"

    async def execute(self, request: ToolExecution) -> ToolOutput:
        del request
        return ToolOutput(kind="test.interactive", data={"ok": True})


class _Catalog:
    async def readable_resource_ids(
        self,
        agent_id: str,
        source_ids: tuple[str, ...] = (),
    ) -> frozenset[str]:
        if agent_id == "agent-1" and source_ids == ("source-1",):
            return frozenset({RESOURCE.id})
        return frozenset()


RESOURCE = CatalogResource(
    id=catalog_resource_id("source-1", ResourceKind.TABLE, "main.current"),
    agent_id="agent-1",
    source_id="source-1",
    native_identity="main.current",
    external_uri="sqlite:///test.db#main.current",
    kind=ResourceKind.TABLE,
    name="current",
    sensitivity=Sensitivity.INTERNAL,
    current_revision="sha256:" + "1" * 64,
    current_sync_id="sync-1",
    first_observed_at=NOW,
    last_observed_at=NOW,
)


def _registry(skills: SkillStore | None = None) -> CapabilityRegistry:
    read = Capability(
        id="test.read",
        description="Read the exact test resource.",
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        output_kind="test.read",
        output_schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        executor_id="test.read.executor",
        access_mode=AccessMode.READ,
        automation_eligibility=AutomationEligibility.AUTOMATION_DIRECT,
    )
    interactive = Capability(
        id="test.interactive",
        description="An interactive-only read.",
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        output_kind="test.interactive",
        output_schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        executor_id="test.interactive.executor",
        access_mode=AccessMode.READ,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
    )
    views = tuple(
        ToolView(
            name=name,
            capability_id=capability.id,
            description=capability.description,
            presentation=ToolPresentation(
                toolbox_id=ToolboxId.SOURCES,
                load_mode=ToolLoadMode.PINNED,
                text_trust=ToolTextTrust.CODE,
                summary=capability.description,
                when_to_use="Use in tests.",
                keywords=("test",),
            ),
        )
        for name, capability in (
            ("read_current", read),
            ("interactive_read", interactive),
        )
    )
    declaration = CapabilityDeclarations(
        domain_owner_id="data",
        capabilities=(read, interactive),
        executor_ids=(read.executor_id, interactive.executor_id),
        tool_views=views,
    )
    declarations = [declaration]
    executors: list[Executor] = [_ReadExecutor(), _InteractiveExecutor()]
    if skills is not None:
        skill_bundle = skill_declarations(skills)
        declarations.append(
            CapabilityDeclarations(
                domain_owner_id=SKILL_DOMAIN_OWNER_ID,
                capabilities=skill_bundle.capabilities,
                executor_ids=tuple(
                    item.executor_id for item in skill_bundle.capabilities
                ),
                tool_views=skill_bundle.tool_views,
            )
        )
        executors.extend(skill_bundle.executors)
    return CapabilityRegistry(declarations=tuple(declarations), executors=executors)


class _Store:
    async def require_effects_unblocked(
        self, agent_id: str, *, run_id: str | None = None, routine_id: str | None = None
    ) -> None:
        assert agent_id == "agent-1"

    def __init__(self) -> None:
        self.routines: dict[str, ScheduledRoutine] = {}
        self.transcripts: dict[str, Transcript] = {
            "run-origin": Transcript(
                RunInput(
                    id="run-origin",
                    agent_id="agent-1",
                    conversation_id="conversation-1",
                    message="Schedule the current value report.",
                    created_at=NOW,
                )
            )
        }
        self.results: dict[str, LoopExit] = {}

    async def admit_scheduled_routine(
        self, routine: ScheduledRoutine
    ) -> ScheduledRoutine:
        if routine.routine_id in self.routines:
            raise ValueError("routine_identity_already_exists")
        stored = replace(
            routine,
            next_due_at=first_slot(
                routine.schedule,
                not_before=routine.created_at,
                expires_at=routine.expires_at,
            ),
        )
        self.routines[stored.routine_id] = stored
        return stored

    async def load_scheduled_routine(
        self, agent_id: str, routine_id: str
    ) -> ScheduledRoutine | None:
        routine = self.routines.get(routine_id)
        return routine if routine is not None and routine.agent_id == agent_id else None

    async def list_scheduled_routines(
        self,
        agent_id: str,
        *,
        states: frozenset[RoutineState] = frozenset(),
        limit: int = 50,
    ) -> tuple[ScheduledRoutine, ...]:
        return tuple(
            item
            for item in self.routines.values()
            if item.agent_id == agent_id and (not states or item.state in states)
        )[:limit]

    async def list_routine_occurrences(
        self,
        agent_id: str,
        routine_id: str,
        *,
        limit: int = 100,
    ) -> tuple[RoutineOccurrence, ...]:
        del agent_id, routine_id, limit
        return ()

    async def revise_scheduled_routine(
        self,
        routine: ScheduledRoutine,
        *,
        expected_revision: int,
    ) -> ScheduledRoutine | None:
        current = self.routines.get(routine.routine_id)
        if current is None or current.revision != expected_revision:
            return None
        self.routines[routine.routine_id] = routine
        return routine

    async def transition_scheduled_routine(
        self,
        agent_id: str,
        routine_id: str,
        *,
        expected_revision: int,
        state: RoutineState,
        transitioned_at: datetime,
    ) -> ScheduledRoutine | None:
        current = await self.load_scheduled_routine(agent_id, routine_id)
        if current is None or current.revision != expected_revision:
            return None
        updated = replace(
            current,
            state=state,
            revision=current.revision + 1,
            updated_at=transitioned_at,
            next_due_at=None,
        )
        self.routines[routine_id] = updated
        return updated

    async def claim_manual_routine_occurrence(
        self,
        agent_id: str,
        routine_id: str,
        *,
        expected_revision: int,
        authorized_control_call_id: str,
        claimed_at: datetime,
        claim_token: str,
    ) -> object | None:
        del authorized_control_call_id, claimed_at, claim_token
        current = await self.load_scheduled_routine(agent_id, routine_id)
        if current is None or current.revision != expected_revision:
            return None
        self.routines[routine_id] = replace(
            current, active_occurrence_id="occurrence-manual"
        )
        return object()

    async def conversation_exists(self, agent_id: str, conversation_id: str) -> bool:
        return agent_id == "agent-1" and conversation_id == "conversation-1"

    async def load(self, run_id: str) -> Transcript:
        try:
            return self.transcripts[run_id]
        except KeyError:
            raise KeyError(f"unknown run: {run_id}") from None

    async def result(self, run_id: str) -> LoopExit | None:
        return self.results.get(run_id)

    async def load_resource(
        self, agent_id: str, resource_id: str
    ) -> CatalogResource | None:
        return (
            RESOURCE if agent_id == "agent-1" and resource_id == RESOURCE.id else None
        )

    async def load_mcp_binding(
        self, agent_id: str, binding_id: str
    ) -> MCPServerBinding | None:
        del agent_id, binding_id
        return None


def _unbound_owner(store: _Store, skills: SkillStore | None = None) -> RoutineOwner:
    async def read_contracts(**kwargs):
        registry = owner._require_capability_registry()
        return ExecutionContractBindings(
            capability_contracts={
                key: registry.contract_digest(key) for key in kwargs["capability_ids"]
            },
            resource_revisions={
                key: RESOURCE.current_revision for key in kwargs["resource_ids"]
            },
            model_routes={
                key: "sha256:" + "c" * 64 for key in kwargs["model_route_ids"]
            },
        )

    owner = RoutineOwner(
        execution_contract_reader=read_contracts,
        agent_id="agent-1",
        store=store,
        catalog=_Catalog(),
        distribution=DistributionOwner(
            agent_id="agent-1", store=cast(DistributionStore, store)
        ),
        skills=skills,
        eligible_model_routes=("mock",),
        maximum_per_run_tokens=10_000,
        maximum_per_run_cost_usd=Decimal("1"),
        clock=lambda: NOW,
    )
    return owner


def _owner(store: _Store, skills: SkillStore | None = None) -> RoutineOwner:
    owner = _unbound_owner(store, skills)
    owner.bind_capability_registry(_registry(skills))
    return owner


async def _proposal(
    owner: RoutineOwner,
    *,
    capability_ids: tuple[str, ...] = ("test.read",),
    skill_names: tuple[str, ...] = (),
    basis_run_id: str | None = None,
) -> ScheduledRoutine:
    return await owner.prepare_create(
        run_id="run-origin",
        conversation_id="conversation-1",
        call_id="call-create",
        title="Current value",
        authorized_instruction="Read the exact current resource and report its value.",
        schedule=IntervalSchedule(3600, NOW),
        misfire_policy=MisfirePolicy.LATEST_ONLY,
        reporting_mode=ReportingMode.ALWAYS,
        precheck=None,
        allowed_source_ids=("source-1",),
        allowed_connector_binding_ids=(),
        allowed_resource_ids=(RESOURCE.id,),
        allowed_capability_ids=capability_ids,
        sensitivity_ceiling=ModelSensitivity.INTERNAL,
        outcome_contract=no_artifact_outcome_contract(),
        distribution_destination_id=conversation_inbox_destination_id("conversation-1"),
        eligible_model_routes=("mock",),
        per_run_max_tokens=1_000,
        per_run_max_cost_usd=Decimal("0.10"),
        cumulative_max_tokens=10_000,
        cumulative_max_cost_usd=Decimal("1"),
        cumulative_max_attempts=10,
        cumulative_max_occurrences=10,
        maximum_consecutive_failures=3,
        expires_at=NOW + timedelta(days=30),
        skill_names=skill_names,
        basis_run_id=basis_run_id,
    )
