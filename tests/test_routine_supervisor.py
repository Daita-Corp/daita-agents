from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import cast

from daita._json import FrozenJsonObject
from daita.capabilities import AccessMode, OperationalEffect, ToolOutput
from daita.capability_runtime import (
    CapabilityRuntime,
    InternalCapabilityOutcome,
    InternalCapabilityRequest,
)
from daita.llm.models import CanonicalMessage, MessageRole, ModelSensitivity, TextBlock
from daita.loop.models import LoopExit, LoopExitKind, RunInput
from daita.routines.models import (
    IntervalSchedule,
    MisfirePolicy,
    ReportingMode,
    ResourceRevisionObservation,
    ResourceRevisionPrecheck,
    RoutineOccurrenceDisposition,
    RoutineOccurrenceV1,
    RoutineState,
    ScheduledRoutineV1,
    text_digest,
)
from daita.routines.owner import RoutineOwner
from daita.routines.supervisor import RoutineSupervisor
from daita.storage.sqlite import SQLiteStateStore

NOW = datetime(2026, 8, 28, 12, tzinfo=UTC)


class _Owner:
    async def authority_snapshot(self, routine: ScheduledRoutineV1) -> FrozenJsonObject:
        return FrozenJsonObject.from_mapping({"routine_id": routine.routine_id})


class _UnusedRuntime:
    async def execute_internal(
        self, request: InternalCapabilityRequest
    ) -> InternalCapabilityOutcome:
        del request
        raise AssertionError("always-reporting routine cannot execute a precheck")


class _ObservationRuntime:
    def __init__(self, observation: ResourceRevisionObservation) -> None:
        self.observation = observation
        self.calls = 0

    async def execute_internal(
        self, request: InternalCapabilityRequest
    ) -> InternalCapabilityOutcome:
        self.calls += 1
        value = self.observation
        return InternalCapabilityOutcome(
            ToolOutput(
                kind="data.resource_revision_observation",
                data={
                    "source_id": value.source_id,
                    "resource_id": value.resource_id,
                    "resource_revision": value.resource_revision,
                    "catalog_revision": value.catalog_revision,
                    "observed_at": value.observed_at.isoformat(),
                },
                sensitivity=ModelSensitivity.INTERNAL,
                sensitivity_provenance={"authority": "test"},
            )
        )


def _routine(
    *,
    precheck: ResourceRevisionPrecheck | None = None,
    observation: ResourceRevisionObservation | None = None,
) -> ScheduledRoutineV1:
    instruction = "Read the exact resource and report its current value."
    return ScheduledRoutineV1(
        routine_id="routine-supervisor",
        agent_id="agent-1",
        conversation_id="conversation-1",
        owner_principal_id="agent:agent-1",
        title="Current value",
        authorized_instruction=instruction,
        instruction_digest=text_digest(instruction),
        schedule=IntervalSchedule(3600, NOW),
        schedule_interpreter_revision=1,
        misfire_policy=MisfirePolicy.LATEST_ONLY,
        reporting_mode=(
            ReportingMode.ALWAYS if precheck is None else ReportingMode.CHANGES_ONLY
        ),
        precheck=precheck,
        last_acknowledged_precheck_observation=observation,
        allowed_source_ids=("source-1",),
        allowed_connector_binding_ids=(),
        allowed_resource_ids=("resource-1",),
        allowed_capability_ids=("data.sqlite.query",),
        allowed_access_modes=frozenset({AccessMode.READ}),
        allowed_operational_effects=frozenset({OperationalEffect.NONE}),
        sensitivity_ceiling=ModelSensitivity.INTERNAL,
        eligible_model_routes=("mock",),
        skill_bindings=(),
        delivery_destination="conversation_inbox:conversation-1",
        per_run_max_tokens=1_000,
        per_run_max_cost_usd=Decimal("0.10"),
        cumulative_max_tokens=10_000,
        cumulative_max_cost_usd=Decimal("1"),
        cumulative_max_attempts=10,
        cumulative_max_occurrences=10,
        reserved_tokens=0,
        reserved_cost_usd=Decimal("0"),
        charged_tokens=0,
        charged_cost_usd=Decimal("0"),
        attempt_count=0,
        occurrence_count=0,
        maximum_consecutive_failures=3,
        consecutive_failures=0,
        expires_at=NOW + timedelta(days=30),
        next_due_at=None,
        active_occurrence_id=None,
        last_occurrence_id=None,
        last_delivery_id=None,
        promotion_evidence=None,
        state=RoutineState.ACTIVE,
        revision=1,
        created_at=NOW,
        updated_at=NOW,
    )


async def _seed_conversation(store: SQLiteStateStore) -> None:
    run = RunInput(
        id="run-foreground",
        agent_id="agent-1",
        conversation_id="conversation-1",
        message="Create the current value report.",
        created_at=NOW - timedelta(minutes=1),
    )
    await store.start(run)
    await store.append(run.id, run.start_message())
    await store.complete(
        LoopExit(
            run_id=run.id,
            conversation_id="conversation-1",
            kind=LoopExitKind.COMPLETED,
            reason="assistant_text",
            created_at=NOW - timedelta(seconds=30),
            final_text="The report is ready.",
            steps=1,
        ),
        CanonicalMessage(
            role=MessageRole.ASSISTANT,
            content=(TextBlock("The report is ready."),),
        ),
    )


def _ids() -> Callable[[str], str]:
    counters: dict[str, int] = {}

    def create(prefix: str) -> str:
        counters[prefix] = counters.get(prefix, 0) + 1
        if prefix == "run":
            return f"run-{counters[prefix]:032x}"
        return f"{prefix}-{counters[prefix]}"

    return create


async def _wait_for_terminal(
    store: SQLiteStateStore, occurrence_id: str
) -> RoutineOccurrenceV1:
    for _ in range(200):
        occurrence = await store.load_routine_occurrence("agent-1", occurrence_id)
        if occurrence is not None and occurrence.disposition in {
            RoutineOccurrenceDisposition.COMPLETED,
            RoutineOccurrenceDisposition.SKIPPED_NO_CHANGE,
            RoutineOccurrenceDisposition.TERMINAL_FAILED,
        }:
            return occurrence
        await asyncio.sleep(0.01)
    raise AssertionError("routine occurrence did not become terminal")


async def _wait_for_occurrence_id(store: SQLiteStateStore, routine_id: str) -> str:
    for _ in range(100):
        current = await store.load_scheduled_routine("agent-1", routine_id)
        if current is not None:
            occurrence_id = current.active_occurrence_id or current.last_occurrence_id
            if occurrence_id is not None:
                return occurrence_id
        await asyncio.sleep(0.01)
    raise AssertionError("routine occurrence was not claimed")


async def test_supervisor_runs_one_due_slot_and_delivers_once(tmp_path) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    await _seed_conversation(store)
    routine = await store.admit_scheduled_routine(_routine())
    executed = 0

    async def execute(
        occurrence: RoutineOccurrenceV1,
        run: RunInput,
        observation: ResourceRevisionObservation | None,
    ) -> LoopExit | None:
        nonlocal executed
        executed += 1
        assert observation is None
        assert run.execution_scope is not None
        bound = await store.bind_routine_occurrence_run(
            "agent-1",
            occurrence.occurrence_id,
            claim_token=cast(str, occurrence.claim_token),
            run_id=run.id,
            execution_scope=run.execution_scope,
            bound_at=NOW,
        )
        assert bound is not None
        await store.start(run)
        await store.append(run.id, run.start_message())
        result = LoopExit(
            run_id=run.id,
            conversation_id="conversation-1",
            kind=LoopExitKind.COMPLETED,
            reason="assistant_text",
            created_at=NOW,
            final_text="The current value is 42.",
            steps=1,
        )
        await store.complete(
            result,
            CanonicalMessage(
                role=MessageRole.ASSISTANT,
                content=(TextBlock("The current value is 42."),),
            ),
        )
        return result

    supervisor = RoutineSupervisor(
        agent_id="agent-1",
        store=store,
        owner=cast(RoutineOwner, _Owner()),
        runtime=cast(CapabilityRuntime, _UnusedRuntime()),
        execute_run=execute,
        clock=lambda: NOW,
        id_factory=_ids(),
        poll_seconds=0.02,
    )
    try:
        await supervisor.start()
        occurrence_id = await _wait_for_occurrence_id(store, routine.routine_id)
        terminal = await _wait_for_terminal(store, occurrence_id)
        supervisor.wake()
        await asyncio.sleep(0.05)
        assert terminal.disposition is RoutineOccurrenceDisposition.COMPLETED
        assert executed == 1
        assert (
            len(await store.list_routine_occurrences("agent-1", routine.routine_id))
            == 1
        )
        assert len(await store.list_inbox("agent-1")) == 1
    finally:
        await supervisor.close()
        await store.close()


async def test_unchanged_precheck_advances_with_zero_model_runs(tmp_path) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    await _seed_conversation(store)
    observation = ResourceRevisionObservation(
        source_id="source-1",
        resource_id="resource-1",
        resource_revision="sha256:" + "1" * 64,
        catalog_revision="sha256:" + "2" * 64,
        observed_at=NOW - timedelta(hours=1),
    )
    precheck = ResourceRevisionPrecheck(
        capability_id="data.resource_revision_observation",
        contract_digest="sha256:" + "3" * 64,
        source_id="source-1",
        resource_id="resource-1",
    )
    routine = await store.admit_scheduled_routine(
        _routine(precheck=precheck, observation=observation)
    )
    runtime = _ObservationRuntime(replace(observation, observed_at=NOW))
    model_calls = 0

    async def execute(
        occurrence: RoutineOccurrenceV1,
        run: RunInput,
        observed: ResourceRevisionObservation | None,
    ) -> LoopExit | None:
        del occurrence, run, observed
        nonlocal model_calls
        model_calls += 1
        raise AssertionError("unchanged precheck must not start a model run")

    supervisor = RoutineSupervisor(
        agent_id="agent-1",
        store=store,
        owner=cast(RoutineOwner, _Owner()),
        runtime=cast(CapabilityRuntime, runtime),
        execute_run=execute,
        clock=lambda: NOW,
        id_factory=_ids(),
        poll_seconds=0.02,
    )
    try:
        await supervisor.start()
        occurrence_id = await _wait_for_occurrence_id(store, routine.routine_id)
        terminal = await _wait_for_terminal(store, occurrence_id)
        assert terminal.disposition is RoutineOccurrenceDisposition.SKIPPED_NO_CHANGE
        assert runtime.calls == 1
        assert model_calls == 0
        assert await store.list_inbox("agent-1") == ()
    finally:
        await supervisor.close()
        await store.close()
