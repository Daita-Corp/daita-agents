"""Claim, execute, recover, and converge scheduled-read occurrences."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from datetime import datetime
from hashlib import sha256

from .._json import FrozenJsonObject, canonical_json
from ..artifacts.models import ArtifactError
from ..artifacts.store import AgentHomeArtifactStore
from ..capabilities import ExecutionScope
from ..capability_runtime import CapabilityRuntime, InternalCapabilityRequest
from ..distribution import DistributionOwner, OutcomeArtifactReference
from ..llm.models import ModelSensitivity
from ..loop.models import (
    InstructionAuthority,
    LoopExit,
    RunInput,
    RunOrigin,
    RunStartEnvelope,
)
from ..storage.sqlite import SQLiteStateStore
from .models import (
    ResourceRevisionObservation,
    RoutineOccurrence,
    RoutineOccurrenceDisposition,
    RoutineState,
    ScheduledRoutine,
)
from .owner import RoutineOwner

_DEFAULT_POLL_SECONDS = 1.0
_RUN_ID = re.compile(r"run-[0-9a-f]{32}\Z")

RoutineRunExecutor = Callable[
    [RoutineOccurrence, RunInput, ResourceRevisionObservation | None],
    Awaitable[LoopExit | None],
]


class RoutineSupervisor:
    """Drive one bounded scheduled-routine owner under an admitted host."""

    def __init__(
        self,
        *,
        agent_id: str,
        store: SQLiteStateStore,
        owner: RoutineOwner,
        runtime: CapabilityRuntime,
        distribution: DistributionOwner,
        artifacts: AgentHomeArtifactStore,
        execute_run: RoutineRunExecutor | None,
        clock: Callable[[], datetime],
        id_factory: Callable[[str], str],
        poll_seconds: float = _DEFAULT_POLL_SECONDS,
    ) -> None:
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("routine supervisor agent_id must be non-empty text")
        if execute_run is not None and not callable(execute_run):
            raise TypeError("routine run executor must be callable or None")
        if not isinstance(poll_seconds, (int, float)) or not 0 < poll_seconds <= 30:
            raise ValueError("routine supervisor poll interval is outside its bound")
        self._agent_id = agent_id
        self._store = store
        self._owner = owner
        self._runtime = runtime
        self._distribution = distribution
        self._artifacts = artifacts
        self._execute_run = execute_run
        self._clock = clock
        self._id_factory = id_factory
        self._poll_seconds = float(poll_seconds)
        self._wake = asyncio.Event()
        self._driver: asyncio.Task[None] | None = None
        self._worker: asyncio.Task[None] | None = None
        self._closing = False

    async def start(self) -> None:
        if self._driver is not None:
            raise RuntimeError("routine supervisor is already started")
        await self._recover()
        self._driver = asyncio.create_task(
            self._drive(),
            name=f"daita-routine-supervisor:{self._agent_id}",
        )

    def wake(self) -> None:
        if not self._closing:
            self._wake.set()

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        current_loop = asyncio.get_running_loop()
        tasks = tuple(
            item
            for item in (self._driver, self._worker)
            if item is not None and item.get_loop() is current_loop
        )
        if tasks:
            self._wake.set()
        for task in tasks:
            task.cancel("host_closing")
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._driver = None
        self._worker = None

    async def _drive(self) -> None:
        try:
            while not self._closing:
                self._wake.clear()
                if self._execute_run is not None and (
                    self._worker is None or self._worker.done()
                ):
                    await self._recover()
                    if self._worker is None or self._worker.done():
                        await self._claim_one_due()
                timeout = self._poll_seconds
                try:
                    deadline = await self._store.next_routine_deadline(self._agent_id)
                    if deadline is not None:
                        timeout = min(
                            timeout,
                            max(0.05, (deadline - self._clock()).total_seconds()),
                        )
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=timeout)
                except TimeoutError:
                    pass
        except asyncio.CancelledError:
            return

    async def _claim_one_due(self) -> None:
        routines = await self._store.list_scheduled_routines(
            self._agent_id,
            states=frozenset({RoutineState.ACTIVE}),
        )
        for routine in sorted(routines, key=lambda item: item.routine_id):
            if routine.active_occurrence_id is None:
                continue
            occurrence = await self._store.load_routine_occurrence(
                self._agent_id,
                routine.active_occurrence_id,
            )
            if (
                occurrence is not None
                and occurrence.reserved_run_id is None
                and occurrence.disposition
                in {
                    RoutineOccurrenceDisposition.CLAIMED,
                    RoutineOccurrenceDisposition.PRECHECKING,
                    RoutineOccurrenceDisposition.RETRYABLE,
                }
            ):
                self._launch(occurrence)
                return
        now = self._clock()
        due = sorted(
            (
                routine
                for routine in routines
                if routine.next_due_at is not None
                and routine.next_due_at <= now
                and routine.active_occurrence_id is None
            ),
            key=lambda item: (item.next_due_at, item.routine_id),
        )
        for routine in due:
            assert routine.next_due_at is not None
            try:
                occurrence = await self._store.claim_due_routine_occurrence(
                    self._agent_id,
                    routine.routine_id,
                    expected_revision=routine.revision,
                    expected_due_at=routine.next_due_at,
                    claimed_at=now,
                    claim_token=self._id_factory("routine-claim"),
                )
            except Exception:
                continue
            if occurrence is None:
                continue
            self._launch(occurrence)
            return

    def _launch(self, occurrence: RoutineOccurrence) -> None:
        task = asyncio.create_task(
            self._run_claimed(occurrence),
            name=f"daita-routine:{occurrence.occurrence_id}",
        )
        self._worker = task

        def done(completed: asyncio.Task[None]) -> None:
            if self._worker is completed:
                self._worker = None
            if not completed.cancelled():
                completed.exception()
            self._wake.set()

        task.add_done_callback(done)

    async def _run_claimed(self, occurrence: RoutineOccurrence) -> None:
        try:
            routine = await self._store.load_scheduled_routine(
                self._agent_id,
                occurrence.routine_id,
            )
            if (
                routine is None
                or routine.revision != occurrence.routine_revision
                or routine.active_occurrence_id != occurrence.occurrence_id
            ):
                return
            await self._owner.authority_snapshot(routine)
            scope = _execution_scope(routine, occurrence)
            observation = await self._observe_precheck(routine, occurrence, scope)
            if (
                observation is not None
                and routine.last_acknowledged_precheck_observation is not None
                and observation.digest
                == routine.last_acknowledged_precheck_observation.digest
            ):
                await self._store.finalize_routine_occurrence(
                    self._agent_id,
                    occurrence.occurrence_id,
                    delivery_id=self._id_factory("delivery"),
                    finalized_at=self._clock(),
                    skipped_no_change_observation=observation,
                )
                return
            run_id = self._id_factory("run")
            if _RUN_ID.fullmatch(run_id) is None:
                raise ValueError("routine run identity is invalid")
            run_input = _run_input(routine, occurrence, scope, run_id, observation)
            assert self._execute_run is not None
            result = await self._execute_run(occurrence, run_input, observation)
            if result is None:
                return
            terminal = await self._store.mark_routine_occurrence_run_terminal(
                self._agent_id,
                occurrence.occurrence_id,
                run_id=run_id,
                terminal_at=result.created_at,
            )
            if terminal is None:
                return
            artifact_references: tuple[OutcomeArtifactReference, ...] = ()
            outcome_contract_failure_code = None
            try:
                artifact_references = (
                    await self._distribution.validate_outcome_artifacts(
                        self._artifacts,
                        result.artifacts,
                        contract=routine.outcome_contract,
                        resulting_run_id=run_id,
                    )
                )
            except (ArtifactError, TypeError, ValueError):
                outcome_contract_failure_code = "outcome_artifact_contract_failed"
            await self._store.finalize_routine_occurrence(
                self._agent_id,
                occurrence.occurrence_id,
                delivery_id=self._id_factory("delivery"),
                finalized_at=self._clock(),
                artifact_references=artifact_references,
                outcome_contract_failure_code=outcome_contract_failure_code,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self._fail_unbound(occurrence, error)

    async def _observe_precheck(
        self,
        routine: ScheduledRoutine,
        occurrence: RoutineOccurrence,
        scope: ExecutionScope,
    ) -> ResourceRevisionObservation | None:
        precheck = routine.precheck
        if precheck is None:
            return None
        run = _run_input(
            routine,
            occurrence,
            scope,
            f"run-{'0' * 32}",
            None,
        )
        outcome = await self._runtime.execute_internal(
            InternalCapabilityRequest(
                run=run,
                call_id=f"precheck:{occurrence.occurrence_id}",
                capability_id=precheck.capability_id,
                contract_digest=precheck.contract_digest,
                arguments={
                    "source_id": precheck.source_id,
                    "resource_id": precheck.resource_id,
                },
                sensitivity=routine.sensitivity_ceiling,
            )
        )
        data = outcome.output.data
        try:
            observed_at = datetime.fromisoformat(str(data["observed_at"]))
            return ResourceRevisionObservation(
                source_id=str(data["source_id"]),
                resource_id=str(data["resource_id"]),
                resource_revision=str(data["resource_revision"]),
                catalog_revision=str(data["catalog_revision"]),
                observed_at=observed_at,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("routine_precheck_output_invalid") from error

    async def _fail_unbound(
        self,
        occurrence: RoutineOccurrence,
        error: BaseException,
    ) -> None:
        current = await self._store.load_routine_occurrence(
            self._agent_id,
            occurrence.occurrence_id,
        )
        if current is None or current.reserved_run_id is not None:
            return
        await self._store.finalize_routine_occurrence(
            self._agent_id,
            occurrence.occurrence_id,
            delivery_id=self._id_factory("delivery"),
            finalized_at=self._clock(),
            failure_code=_failure_code(error),
        )

    async def _recover(self) -> None:
        recovered = await self._store.recover_stale_routine_occurrences(
            self._agent_id,
            recovered_at=self._clock(),
            claim_token_factory=lambda occurrence_id: (
                f"routine-recovery-{sha256(occurrence_id.encode('utf-8')).hexdigest()[:32]}"
            ),
        )
        for occurrence in recovered:
            try:
                if (
                    occurrence.disposition
                    is RoutineOccurrenceDisposition.RUN_TERMINAL_PENDING_FINALIZATION
                ):
                    (
                        artifact_references,
                        outcome_contract_failure_code,
                    ) = await self._recovered_artifact_references(occurrence)
                    await self._store.finalize_routine_occurrence(
                        self._agent_id,
                        occurrence.occurrence_id,
                        delivery_id=self._id_factory("delivery"),
                        finalized_at=self._clock(),
                        artifact_references=artifact_references,
                        outcome_contract_failure_code=(outcome_contract_failure_code),
                    )
                elif (
                    occurrence.reserved_run_id is None and self._execute_run is not None
                ):
                    self._launch(occurrence)
                    break
            except Exception:
                continue

    async def _recovered_artifact_references(
        self,
        occurrence: RoutineOccurrence,
    ) -> tuple[tuple[OutcomeArtifactReference, ...], str | None]:
        if occurrence.terminal_run_id is None:
            return (), "outcome_artifact_contract_failed"
        routine = await self._store.load_scheduled_routine(
            self._agent_id,
            occurrence.routine_id,
        )
        result = await self._store.result(occurrence.terminal_run_id)
        if routine is None or result is None:
            return (), "outcome_artifact_contract_failed"
        try:
            references = await self._distribution.validate_outcome_artifacts(
                self._artifacts,
                result.artifacts,
                contract=routine.outcome_contract,
                resulting_run_id=occurrence.terminal_run_id,
            )
        except (ArtifactError, TypeError, ValueError):
            return (), "outcome_artifact_contract_failed"
        return references, None


def _execution_scope(
    routine: ScheduledRoutine,
    occurrence: RoutineOccurrence,
) -> ExecutionScope:
    return ExecutionScope(
        scope_id=f"scope:{occurrence.occurrence_id}",
        revision=1,
        agent_id=routine.agent_id,
        principal_id=routine.owner_principal_id,
        grant_id=f"routine:{routine.routine_id}:revision:{routine.revision}",
        job_id=None,
        job_revision=None,
        routine_id=routine.routine_id,
        routine_revision=routine.revision,
        occurrence_id=occurrence.occurrence_id,
        allowed_source_ids=routine.allowed_source_ids,
        allowed_connector_binding_ids=routine.allowed_connector_binding_ids,
        allowed_resource_ids=routine.allowed_resource_ids,
        allowed_capability_ids=routine.allowed_capability_ids,
        allowed_access_modes=routine.allowed_access_modes,
        allowed_operational_effects=routine.allowed_operational_effects,
        sensitivity_ceiling=routine.sensitivity_ceiling,
        eligible_model_routes=routine.eligible_model_routes,
        per_run_max_cost_usd=routine.per_run_max_cost_usd,
        per_run_max_tokens=routine.per_run_max_tokens,
        distribution_plan_digest=routine.distribution_plan.plan_digest,
    )


def _run_input(
    routine: ScheduledRoutine,
    occurrence: RoutineOccurrence,
    scope: ExecutionScope,
    run_id: str,
    observation: ResourceRevisionObservation | None,
) -> RunInput:
    payload = {
        "routine_id": routine.routine_id,
        "routine_revision": routine.revision,
        "occurrence_id": occurrence.occurrence_id,
        "scheduled_for": occurrence.scheduled_for.isoformat(),
        "slot_kind": occurrence.slot_kind.value,
        "pinned_skills": tuple(
            {
                "name": binding.skill_name,
                "content_digest": binding.content_digest,
            }
            for binding in routine.skill_bindings
        ),
        "precheck_observation": (
            None
            if observation is None
            else {
                "source_id": observation.source_id,
                "resource_id": observation.resource_id,
                "resource_revision": observation.resource_revision,
                "catalog_revision": observation.catalog_revision,
                "observed_at": observation.observed_at.isoformat(),
            }
        ),
    }
    return RunInput(
        id=run_id,
        agent_id=routine.agent_id,
        message=routine.authorized_instruction,
        created_at=occurrence.claimed_at or occurrence.created_at,
        conversation_id=routine.conversation_id,
        source_id=(
            routine.allowed_source_ids[0]
            if len(routine.allowed_source_ids) == 1
            else None
        ),
        start=RunStartEnvelope(
            origin=RunOrigin.SCHEDULED_ROUTINE,
            instruction_authority=InstructionAuthority.FOREGROUND_AUTHORIZED,
            trusted_instruction_id=(
                f"routine:{routine.routine_id}:revision:{routine.revision}"
            ),
            trusted_instruction=routine.authorized_instruction,
            instruction_digest=routine.instruction_digest,
            untrusted_payload=payload,
            payload_digest=(
                "sha256:" + sha256(canonical_json(payload).encode("utf-8")).hexdigest()
            ),
            execution_scope=scope,
        ),
    )


def _failure_code(error: BaseException) -> str:
    raw = getattr(error, "code", type(error).__name__)
    normalized = re.sub(r"[^a-z0-9_]+", "_", str(raw).casefold()).strip("_")
    return f"routine_{normalized[:96] or 'execution_failed'}"


__all__ = ["RoutineRunExecutor", "RoutineSupervisor"]
