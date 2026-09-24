"""Claim, execute, fence, recover, and reconcile durable job attempts within limits."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime
from functools import partial
from hashlib import sha256
from typing import TypeVar

from ..artifacts.models import (
    ArtifactAuthorship,
    ArtifactError,
    ArtifactRef,
    artifact_ref_to_mapping,
)
from ..artifacts.store import AgentHomeArtifactStore
from ..capabilities import CapabilityInputError
from ..capability_runtime import CapabilityRuntime, InternalCapabilityRequest
from ..distribution.models import (
    ArtifactRequirement,
    OutcomeContract,
    outcome_artifact_reference,
    validate_outcome_artifact_references,
)
from ..distribution.owner import DistributionOwner, construct_graph_job_delivery
from ..errors import DaitaError, ErrorRetryability
from ..hosting.execution_governor import RunAdmissionCoordinator, WorkloadClass
from ..llm.models import ModelSensitivity
from ..loop.driver import AgentLoop
from ..loop.models import (
    InstructionAuthority,
    LoopExitKind,
    RunInput,
    RunOrigin,
    RunStartEnvelope,
)
from ..loop.session import RunCancellationToken, RunSession, RunSessionOptions
from ..loop.transcripts import RunSessionWriter
from ..storage.sqlite import SQLiteStateStore
from .graph.execution import (
    GRAPH_RESULT_FINALIZER_KIND,
    PROFILE_WORK_KIND,
    effective_task_ids,
    graph_task_binding,
    inspection_task,
    internal_task_contract,
    model_task_contract,
    model_task_execution_scope,
    prepare_graph_attempt,
    task_context_bundle,
    task_conversation_id,
)
from .graph.guard import SQLiteTaskAttemptGuard
from .graph.models import (
    AttemptState,
    BudgetAmount,
    GraphInspection,
    GraphTask,
    TaskAttempt,
    TaskCheckpoint,
    TaskExecutionKind,
    TaskResult,
    TaskRole,
    TaskState,
    canonical_digest,
    reserved_artifact_id,
)
from .graph.reduction import fair_graph_dispatch_order
from .owner import JobOwner

_ARTIFACT_ID = re.compile(r"artifact-[0-9a-f]{32}\Z")
_RUN_ID = re.compile(r"run-[0-9a-f]{32}\Z")
_DEFAULT_POLL_SECONDS = 0.05
_T = TypeVar("_T")

# Temporary private compatibility name for the existing deterministic tests.
_fair_graph_dispatch_order = fair_graph_dispatch_order


class JobSupervisor:
    """Claim, execute, fence, recover, and finalize current graph tasks."""

    def __init__(
        self,
        *,
        agent_id: str,
        store: SQLiteStateStore,
        owner: JobOwner,
        runtime: CapabilityRuntime,
        artifacts: AgentHomeArtifactStore,
        clock: Callable[[], datetime],
        id_factory: Callable[[str], str],
        admission_coordinator: RunAdmissionCoordinator,
        distribution: DistributionOwner,
        graph_parallelism: int = 1,
        graph_heartbeat_seconds: float = 10.0,
        graph_model_loop: AgentLoop | None = None,
        poll_seconds: float = _DEFAULT_POLL_SECONDS,
    ) -> None:
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("job supervisor agent_id must be non-empty text")
        if not isinstance(poll_seconds, (int, float)) or not 0 < poll_seconds <= 1:
            raise ValueError("job supervisor poll interval is outside its bound")
        self._agent_id = agent_id
        self._store = store
        self._owner = owner
        self._runtime = runtime
        self._artifacts = artifacts
        self._clock = clock
        self._id_factory = id_factory
        if not isinstance(admission_coordinator, RunAdmissionCoordinator):
            raise TypeError("graph admission coordinator is invalid")
        if not isinstance(distribution, DistributionOwner):
            raise TypeError("graph distribution owner is invalid")
        if (
            not isinstance(graph_parallelism, int)
            or isinstance(graph_parallelism, bool)
            or not 1 <= graph_parallelism <= 4
        ):
            raise ValueError("graph parallelism must be between one and four")
        if (
            not isinstance(graph_heartbeat_seconds, (int, float))
            or isinstance(graph_heartbeat_seconds, bool)
            or not 10 <= float(graph_heartbeat_seconds) <= 30
        ):
            raise ValueError("graph heartbeat interval is outside its bound")
        self._admission_coordinator = admission_coordinator
        self._distribution = distribution
        self._graph_parallelism = graph_parallelism
        if graph_model_loop is not None and not isinstance(graph_model_loop, AgentLoop):
            raise TypeError("graph model loop must be AgentLoop or None")
        self._graph_model_loop = graph_model_loop
        self._graph_model_gate = asyncio.Semaphore(2)
        self._graph_heartbeat_seconds = float(graph_heartbeat_seconds)
        self._poll_seconds = float(poll_seconds)
        self._driver: asyncio.Task[None] | None = None
        self._graph_workers: dict[tuple[str, str], asyncio.Task[None]] = {}
        self._wake = asyncio.Event()
        self._last_graph_dispatch_job_id: str | None = None
        self._consecutive_graph_dispatches = 0
        self._closing = False

    async def start(self) -> None:
        if self._driver is not None:
            raise RuntimeError("job supervisor is already started")
        await self._recover_graph_attempts()
        self._driver = asyncio.create_task(
            self._drive_graph(),
            name=f"daita-job-supervisor:{self._agent_id}",
        )

    def wake(self, job_id: str | None = None) -> None:
        del job_id
        if self._closing:
            return
        self._wake.set()

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        current_loop = asyncio.get_running_loop()
        all_tasks = tuple(
            item
            for item in (
                self._driver,
                *self._graph_workers.values(),
            )
            if item is not None
        )
        tasks = tuple(item for item in all_tasks if item.get_loop() is current_loop)
        if tasks:
            self._wake.set()
        for task in tasks:
            task.cancel("host_closing")
        await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )
        self._driver = None
        self._graph_workers.clear()

    async def _graph_store_call(
        self,
        operation: Callable[[], Awaitable[_T]],
    ) -> _T:
        coordinator = self._required_graph_coordinator()
        permit = await coordinator.sqlite_pressure_permit()
        async with permit:
            return await operation()

    def _required_graph_coordinator(self) -> RunAdmissionCoordinator:
        return self._admission_coordinator

    def _required_graph_distribution(self) -> DistributionOwner:
        return self._distribution

    async def _recover_graph_attempts(self) -> None:
        attempts = await self._graph_store_call(
            lambda: self._store.list_active_graph_attempts(self._agent_id)
        )
        for attempt in attempts:
            has_effect_reservation = await self._graph_store_call(
                partial(
                    self._store.reconcile_graph_effect_attempt,
                    attempt.agent_id,
                    attempt.job_id,
                    attempt.task_id,
                    attempt.attempt_id,
                )
            )
            if has_effect_reservation:
                continue
            inspection = await self._graph_store_call(
                partial(self._store.inspect_graph, self._agent_id, attempt.job_id)
            )
            if inspection is None:
                continue
            task = inspection_task(inspection, attempt.task_id)
            if task.execution_kind is TaskExecutionKind.MODEL:
                await self._graph_store_call(
                    partial(
                        self._store.fence_graph_attempt,
                        attempt.agent_id,
                        attempt.job_id,
                        attempt.task_id,
                        attempt.attempt_id,
                        fencing_epoch=attempt.fencing_epoch,
                        fenced_at=self._clock(),
                        requeue=True,
                        reason_code="host_restarted",
                    )
                )
                continue
            try:
                internal_task_contract(task)
                payload = await self._artifacts.read_reserved(
                    attempt.run_id,
                    reserved_artifact_id(attempt.attempt_id),
                )
            except (
                ArtifactError,
                CapabilityInputError,
                TypeError,
                ValueError,
            ) as error:
                await self._graph_store_call(
                    partial(
                        self._store.fail_graph_attempt,
                        attempt.agent_id,
                        attempt.job_id,
                        attempt.task_id,
                        attempt.attempt_id,
                        claim_token=attempt.claim_token,
                        fencing_epoch=attempt.fencing_epoch,
                        failed_at=self._clock(),
                        retryable=False,
                        reason_code=_safe_graph_failure_code(error),
                    )
                )
                continue
            if payload is not None:
                started = await self._graph_store_call(
                    partial(
                        self._store.start_graph_attempt,
                        attempt.agent_id,
                        attempt.job_id,
                        attempt.task_id,
                        attempt.attempt_id,
                        claim_token=attempt.claim_token,
                        fencing_epoch=attempt.fencing_epoch,
                        started_at=attempt.started_at or payload.ref.created_at,
                    )
                )
                if started is not None:
                    try:
                        await self._promote_graph_artifact(
                            inspection,
                            task,
                            started,
                            payload.ref,
                            payload.content,
                        )
                    except (
                        ArtifactError,
                        CapabilityInputError,
                        TypeError,
                        ValueError,
                    ) as error:
                        await self._graph_store_call(
                            partial(
                                self._store.fail_graph_attempt,
                                started.agent_id,
                                started.job_id,
                                started.task_id,
                                started.attempt_id,
                                claim_token=started.claim_token,
                                fencing_epoch=started.fencing_epoch,
                                failed_at=self._clock(),
                                retryable=False,
                                reason_code=_safe_graph_failure_code(error),
                            )
                        )
                    continue
            await self._graph_store_call(
                partial(
                    self._store.fence_graph_attempt,
                    attempt.agent_id,
                    attempt.job_id,
                    attempt.task_id,
                    attempt.attempt_id,
                    fencing_epoch=attempt.fencing_epoch,
                    fenced_at=self._clock(),
                    requeue=True,
                    reason_code="host_restarted",
                )
            )

    async def _drive_graph(self) -> None:
        try:
            while not self._closing:
                self._wake.clear()
                await self._recover_stale_graph_attempts()
                await self._graph_store_call(
                    lambda: self._store.expire_due_graphs(
                        self._agent_id,
                        expired_at=self._clock(),
                    )
                )
                await self._launch_ready_graph_tasks()
                try:
                    await asyncio.wait_for(
                        self._wake.wait(),
                        timeout=self._poll_seconds,
                    )
                except TimeoutError:
                    pass
        except asyncio.CancelledError:
            return

    async def _recover_stale_graph_attempts(self) -> None:
        stale = await self._graph_store_call(
            lambda: self._store.list_stale_graph_attempts(
                self._agent_id,
                now=self._clock(),
            )
        )
        for attempt in stale:
            key = (attempt.job_id, attempt.task_id)
            fenced = await self._graph_store_call(
                partial(
                    self._store.fence_graph_attempt,
                    attempt.agent_id,
                    attempt.job_id,
                    attempt.task_id,
                    attempt.attempt_id,
                    fencing_epoch=attempt.fencing_epoch,
                    fenced_at=self._clock(),
                    requeue=True,
                    reason_code="lease_expired",
                )
            )
            if fenced is None:
                continue
            worker = self._graph_workers.get(key)
            if worker is not None and not worker.done():
                worker.cancel("graph_attempt_lease_expired")
                await asyncio.gather(worker, return_exceptions=True)

    async def _launch_ready_graph_tasks(self) -> None:
        available = self._graph_parallelism - len(self._graph_workers)
        if available <= 0:
            return
        ready = await self._graph_store_call(
            lambda: self._store.list_ready_graph_tasks(
                self._agent_id,
                now=self._clock(),
                limit=64,
            )
        )
        ready = fair_graph_dispatch_order(
            ready,
            last_job_id=self._last_graph_dispatch_job_id,
            consecutive=self._consecutive_graph_dispatches,
        )
        for task in ready:
            key = (task.job_id, task.task_id)
            if key in self._graph_workers:
                continue
            worker = asyncio.create_task(
                self._run_graph_task(task),
                name=f"daita-graph-task:{task.job_id}:{task.task_id}",
            )
            self._graph_workers[key] = worker
            if task.job_id == self._last_graph_dispatch_job_id:
                self._consecutive_graph_dispatches += 1
            else:
                self._last_graph_dispatch_job_id = task.job_id
                self._consecutive_graph_dispatches = 1

            def done(completed: asyncio.Task[None], *, key=key) -> None:
                self._graph_workers.pop(key, None)
                if not completed.cancelled():
                    completed.exception()
                self._wake.set()

            worker.add_done_callback(done)
            available -= 1
            if available == 0:
                return

    async def _run_graph_task(self, selected: GraphTask) -> None:
        coordinator = self._required_graph_coordinator()
        lease = await coordinator.admit_execution(
            WorkloadClass.GRAPH,
            f"{selected.job_id}:{selected.task_id}",
        )
        async with lease:
            inspection = await self._graph_store_call(
                lambda: self._store.inspect_graph(self._agent_id, selected.job_id)
            )
            if inspection is None:
                return
            current_task = inspection_task(inspection, selected.task_id)
            attempt_id = self._id_factory("attempt")
            claim_token = self._id_factory("claim")
            run_id = _required_generated_id(
                self._id_factory("run"),
                _RUN_ID,
                "graph execution run",
            )
            claimed_at = self._clock()
            preparation = prepare_graph_attempt(
                inspection, current_task, claimed_at=claimed_at
            )
            try:
                attempt = await self._graph_store_call(
                    lambda: self._store.claim_graph_task(
                        self._agent_id,
                        current_task.job_id,
                        current_task.task_id,
                        attempt_id=attempt_id,
                        claim_token=claim_token,
                        run_id=run_id,
                        executor_id=preparation.executor_id,
                        claimed_at=claimed_at,
                        lease_seconds=30,
                        absolute_deadline_at=preparation.absolute_deadline_at,
                        budget_reservations=preparation.budget_reservations,
                    )
                )
            except Exception:
                attempt = await self._find_graph_attempt(
                    current_task.job_id,
                    current_task.task_id,
                    attempt_id,
                )
                if attempt is None:
                    raise
            if attempt is None:
                return
            started = await self._graph_store_call(
                lambda: self._store.start_graph_attempt(
                    attempt.agent_id,
                    attempt.job_id,
                    attempt.task_id,
                    attempt.attempt_id,
                    claim_token=attempt.claim_token,
                    fencing_epoch=attempt.fencing_epoch,
                    started_at=self._clock(),
                )
            )
            if started is None:
                return
            await self._checkpoint_graph_start(started)
            heartbeat = asyncio.create_task(
                self._heartbeat_graph_attempt(started),
                name=f"daita-graph-heartbeat:{started.attempt_id}",
            )
            try:
                await self._execute_graph_attempt(inspection, current_task, started)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                failure = error
                current = await self._graph_store_call(
                    lambda: self._store.inspect_graph(
                        self._agent_id, current_task.job_id
                    )
                )
                if (
                    current is not None
                    and inspection_task(current, current_task.task_id).state
                    is TaskState.SUCCEEDED
                ):
                    return
                await self._graph_store_call(
                    lambda: self._store.fail_graph_attempt(
                        started.agent_id,
                        started.job_id,
                        started.task_id,
                        started.attempt_id,
                        claim_token=started.claim_token,
                        fencing_epoch=started.fencing_epoch,
                        failed_at=self._clock(),
                        retryable=(
                            getattr(failure, "code", None) == "protocol_violation"
                            or _graph_failure_is_retryable(failure)
                        ),
                        reason_code=_safe_graph_failure_code(failure),
                        attempt_state=(
                            AttemptState.PROTOCOL_VIOLATION
                            if getattr(failure, "code", None) == "protocol_violation"
                            else AttemptState.FAILED
                        ),
                    )
                )
            finally:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)

    async def _find_graph_attempt(
        self,
        job_id: str,
        task_id: str,
        attempt_id: str,
    ) -> TaskAttempt | None:
        inspection = await self._graph_store_call(
            lambda: self._store.inspect_graph(self._agent_id, job_id)
        )
        if inspection is None:
            return None
        return next(
            (
                item
                for item in inspection.attempts
                if item.task_id == task_id and item.attempt_id == attempt_id
            ),
            None,
        )

    async def _checkpoint_graph_start(self, attempt: TaskAttempt) -> None:
        payload = {
            "state": "running",
            "executor_id": attempt.executor_id,
            "reserved_budgets": {
                item.dimension: item.amount for item in attempt.reserved_budgets
            },
        }
        checkpoint = TaskCheckpoint(
            agent_id=attempt.agent_id,
            job_id=attempt.job_id,
            task_id=attempt.task_id,
            attempt_id=attempt.attempt_id,
            checkpoint_id=f"{attempt.attempt_id}:started",
            fencing_epoch=attempt.fencing_epoch,
            ordinal=1,
            milestone="execution_started",
            payload=payload,
            created_at=self._clock(),
            payload_digest=canonical_digest(payload),
        )
        try:
            await self._graph_store_call(
                lambda: self._store.checkpoint_graph_attempt(
                    checkpoint,
                    claim_token=attempt.claim_token,
                )
            )
        except Exception:
            inspection = await self._graph_store_call(
                lambda: self._store.inspect_graph(self._agent_id, attempt.job_id)
            )
            current = (
                None
                if inspection is None
                else next(
                    (
                        item
                        for item in inspection.attempts
                        if item.attempt_id == attempt.attempt_id
                    ),
                    None,
                )
            )
            if (
                current is None
                or checkpoint.checkpoint_id not in current.checkpoint_ids
            ):
                raise

    async def _heartbeat_graph_attempt(self, attempt: TaskAttempt) -> None:
        try:
            while True:
                await asyncio.sleep(self._graph_heartbeat_seconds)
                renewed = await self._graph_store_call(
                    lambda: self._store.heartbeat_graph_attempt(
                        attempt.agent_id,
                        attempt.job_id,
                        attempt.task_id,
                        attempt.attempt_id,
                        claim_token=attempt.claim_token,
                        fencing_epoch=attempt.fencing_epoch,
                        heartbeat_at=self._clock(),
                    )
                )
                if renewed is None:
                    return
        except asyncio.CancelledError:
            return

    async def _execute_graph_attempt(
        self,
        inspection: GraphInspection,
        task: GraphTask,
        attempt: TaskAttempt,
    ) -> None:
        if task.execution_kind is TaskExecutionKind.MODEL:
            await self._execute_model_graph_attempt(inspection, task, attempt)
            return
        contract = internal_task_contract(task)
        reservation = reserved_artifact_id(attempt.attempt_id)
        recovered = await self._artifacts.read_reserved(attempt.run_id, reservation)
        if recovered is not None:
            await self._promote_graph_artifact(
                inspection,
                task,
                attempt,
                recovered.ref,
                recovered.content,
            )
            return
        arguments = self._graph_execution_arguments(inspection, task, contract)
        run = RunInput(
            id=attempt.run_id,
            agent_id=attempt.agent_id,
            message="Execute the exact frozen internal graph task.",
            created_at=attempt.started_at or self._clock(),
            conversation_id=inspection.job.conversation_id,
            source_scope_ids=task.specification.authority.source_ids,
        )
        request = InternalCapabilityRequest(
            run=run,
            observation_origin=RunOrigin.JOB_TASK,
            call_id=f"graph-call-{attempt.attempt_id}",
            capability_id=str(contract["capability_id"]),
            contract_digest=str(contract["contract_digest"]),
            arguments=arguments,
            sensitivity=task.specification.authority.sensitivity,
            reserved_artifact_id=reservation,
            task_attempt_guard=_graph_attempt_guard(
                self._store,
                inspection,
                task,
                attempt,
                clock=self._clock,
            ),
        )
        permit_key = contract.get("source_permit_key")
        try:
            if permit_key is None:
                outcome = await self._runtime.execute_internal(request)
            else:
                if not isinstance(permit_key, str) or not permit_key:
                    raise ValueError("graph source permit key is invalid")
                permit = (
                    await self._required_graph_coordinator().source_resource_permit(
                        permit_key
                    )
                )
                async with permit:
                    outcome = await self._runtime.execute_internal(request)
        except BaseException:
            recovered = await self._artifacts.read_reserved(attempt.run_id, reservation)
            if recovered is None:
                raise
            await self._promote_graph_artifact(
                inspection,
                task,
                attempt,
                recovered.ref,
                recovered.content,
            )
            return
        if outcome.artifact_ref is None:
            raise ValueError("internal graph capability returned no required artifact")
        if outcome.output.kind != contract["output_kind"]:
            raise ValueError("internal graph capability returned the wrong output kind")
        payload = await self._artifacts.read_reserved(attempt.run_id, reservation)
        if payload is None or payload.ref != outcome.artifact_ref:
            raise ValueError("internal graph artifact cannot be authenticated")
        await self._promote_graph_artifact(
            inspection,
            task,
            attempt,
            payload.ref,
            payload.content,
        )

    async def _execute_model_graph_attempt(
        self,
        inspection: GraphInspection,
        task: GraphTask,
        attempt: TaskAttempt,
    ) -> None:
        if self._graph_model_loop is None:
            raise CapabilityInputError(
                "graph_model_executor_unavailable",
                "The explicit graph integration has no model-task executor.",
            )
        current = await self._graph_store_call(
            lambda: self._store.inspect_graph(self._agent_id, task.job_id)
        )
        if current is None:
            raise CapabilityInputError(
                "stale_task_attempt", "The graph task disappeared before execution."
            )
        current_task = inspection_task(current, task.task_id)
        current_attempt = next(
            (
                item
                for item in current.attempts
                if item.task_id == task.task_id
                and item.attempt_id == attempt.attempt_id
            ),
            None,
        )
        if current_attempt is None:
            raise CapabilityInputError(
                "stale_task_attempt", "The graph task attempt is unavailable."
            )
        contract = model_task_contract(current_task)
        guard = _graph_attempt_guard(
            self._store,
            current,
            current_task,
            current_attempt,
            clock=self._clock,
        )
        scope = model_task_execution_scope(
            current,
            current_task,
            current_attempt,
            guard.binding,
            contract,
        )
        context = task_context_bundle(
            current,
            current_task,
            current_attempt,
            guard.binding,
            created_at=self._clock(),
        )
        instruction = (
            "Execute the exact frozen graph task from the code-owned task context. "
            "Use one exclusive lifecycle terminator to finish."
        )
        initial_call = current_task.specification.expected_result_contract.get(
            "initial_call"
        )
        if isinstance(initial_call, Mapping) and isinstance(
            initial_call.get("effect_call"), Mapping
        ):
            effect_call = initial_call["effect_call"]
            assert isinstance(effect_call, Mapping)
            instruction += (
                " This is one exact grant-backed effect task."
                + (
                    " Call its frozen preview first, then apply only that authenticated preview once."
                    if effect_call.get("preview_capability_id") is not None
                    else " Call the exact frozen action at most once."
                )
                + (
                    " The runtime automatically binds the sole successful grant-backed "
                    "effect call when the task completes. Never copy an effect-receipt "
                    "ID into task_complete. Use evidence_call_ids only for additional "
                    "current-run ToolCall.id values and artifact_ids only for actual "
                    "artifacts; omit both when empty."
                )
            )
        payload = {"task_context_digest": context.digest}
        start = RunStartEnvelope(
            origin=RunOrigin.JOB_TASK,
            instruction_authority=InstructionAuthority.CODE_OWNED,
            trusted_instruction_id=f"graph-task:{current_attempt.attempt_id}",
            trusted_instruction=instruction,
            instruction_digest="sha256:"
            + sha256(instruction.encode("utf-8")).hexdigest(),
            untrusted_payload=payload,
            payload_digest=canonical_digest(payload),
            execution_scope=scope,
        )
        run = RunInput(
            id=current_attempt.run_id,
            agent_id=current_attempt.agent_id,
            message=instruction,
            created_at=current_attempt.started_at or self._clock(),
            conversation_id=task_conversation_id(
                current_attempt.job_id, current_attempt.attempt_id
            ),
            source_scope_ids=current_task.specification.authority.source_ids,
            start=start,
            history_sensitivity=current_task.specification.authority.sensitivity,
        )
        remaining = max(
            0.001,
            (current_attempt.absolute_deadline_at - self._clock()).total_seconds(),
        )
        session = RunSession(
            run=run,
            writer=RunSessionWriter(self._store, run, bind_predecessor=False),
            absolute_deadline=asyncio.get_running_loop().time() + remaining,
            cancellation=RunCancellationToken(),
            options=RunSessionOptions(
                task_context=context,
                task_attempt_guard=guard,
            ),
            admission_lease=None,
            budget_reservations=current_attempt.reserved_budgets,
        )
        async with self._graph_model_gate:
            exit = await self._graph_model_loop.run(session, prior_messages=())
        if exit.kind is LoopExitKind.MACHINE_TERMINATED:
            settled = await self._graph_store_call(
                lambda: self._store.inspect_graph(
                    self._agent_id, current_attempt.job_id
                )
            )
            if settled is None:
                raise CapabilityInputError(
                    "stale_task_attempt", "The terminated task cannot be authenticated."
                )
            stored_attempt = next(
                (
                    item
                    for item in settled.attempts
                    if item.attempt_id == current_attempt.attempt_id
                ),
                None,
            )
            if stored_attempt is None or stored_attempt.state not in {
                AttemptState.SUCCEEDED,
                AttemptState.BLOCKED,
                AttemptState.REVIEW_REQUESTED,
            }:
                raise CapabilityInputError(
                    "task_termination_not_committed",
                    "The machine lifecycle termination has no matching durable transition.",
                )
            await self._reauthenticate_machine_termination(
                settled,
                current_task,
                current_attempt,
                exit.reason,
            )
            return
        if exit.reason == "protocol_violation":
            error = CapabilityInputError(
                "protocol_violation",
                "A graph task returned normal text instead of a lifecycle terminator.",
            )
            raise error
        raise CapabilityInputError(
            exit.reason,
            "The graph model task did not reach an authenticated lifecycle transition.",
        )

    async def _reauthenticate_machine_termination(
        self,
        inspection: GraphInspection,
        original_task: GraphTask,
        attempt: TaskAttempt,
        directive_kind: str,
    ) -> None:
        """Bind the terminal transcript result to the exact durable transition."""

        expected_states = {
            "task_completed": AttemptState.SUCCEEDED,
            "task_blocked": AttemptState.BLOCKED,
            "task_review_requested": AttemptState.REVIEW_REQUESTED,
        }
        expected_state = expected_states.get(directive_kind)
        if expected_state is None:
            raise CapabilityInputError(
                "task_termination_not_committed",
                "The machine termination kind is not a graph lifecycle directive.",
            )
        stored_attempt = next(
            (
                item
                for item in inspection.attempts
                if item.attempt_id == attempt.attempt_id
            ),
            None,
        )
        stored_task = inspection_task(inspection, original_task.task_id)
        transcript = await self._store.load(attempt.run_id)
        pairs = transcript.tool_pairs
        if not pairs or pairs[-1][1] is None or pairs[-1][1].is_error:
            raise CapabilityInputError(
                "task_termination_not_committed",
                "The machine termination lacks its authenticated tool result.",
            )
        terminal = pairs[-1][1]
        assert terminal is not None
        data = terminal.output.get("data")
        if not isinstance(data, Mapping):
            raise CapabilityInputError(
                "task_termination_not_committed",
                "The machine termination output is malformed.",
            )
        record_id = data.get("record_id")
        result_matches = any(
            item.result_id == record_id
            and item.task_id == stored_task.task_id
            and item.attempt_id == attempt.attempt_id
            and item.run_id == attempt.run_id
            for item in inspection.results
        )
        control_matches = any(
            item.control_id == record_id
            and item.task_id == stored_task.task_id
            and item.requesting_attempt_id == attempt.attempt_id
            for item in inspection.controls
        )
        record_matches = (
            result_matches
            if expected_state is AttemptState.SUCCEEDED
            else control_matches
        )
        if (
            stored_attempt is None
            or stored_attempt.state is not expected_state
            or data.get("job_id") != attempt.job_id
            or data.get("task_id") != attempt.task_id
            or data.get("attempt_id") != attempt.attempt_id
            or data.get("fencing_epoch") != attempt.fencing_epoch
            or data.get("committed_task_revision") != stored_task.task_revision
            or not record_matches
        ):
            raise CapabilityInputError(
                "task_termination_not_committed",
                "The machine directive differs from its exact durable lifecycle record.",
            )

    def _graph_execution_arguments(
        self,
        inspection: GraphInspection,
        task: GraphTask,
        contract: Mapping[str, object],
    ) -> Mapping[str, object]:
        if contract["kind"] == PROFILE_WORK_KIND:
            arguments = contract.get("arguments")
            if not isinstance(arguments, Mapping):
                raise ValueError("graph work arguments are malformed")
            return arguments
        if task.role is not TaskRole.FINALIZER:
            raise ValueError("only the reserved finalizer can aggregate graph results")
        required_task_ids = tuple(
            sorted(
                edge.upstream_task_id
                for edge in inspection.dependencies
                if edge.downstream_task_id == task.task_id
            )
        )
        if contract["kind"] == GRAPH_RESULT_FINALIZER_KIND:
            required_task_ids = tuple(
                sorted(
                    item.task_id
                    for item in inspection.tasks
                    if item.role is not TaskRole.FINALIZER
                    and item.state is TaskState.SUCCEEDED
                )
            )
        else:
            required_task_ids = tuple(
                sorted(effective_task_ids(inspection, set(required_task_ids)))
            )
        accepted = {item.task_id: item for item in inspection.results}
        if set(accepted) - {item.task_id for item in inspection.tasks}:
            raise ValueError("graph accepted results reference unknown tasks")
        work_results: list[dict[str, object]] = []
        for task_id in required_task_ids:
            result = accepted.get(task_id)
            if result is None:
                raise ValueError("the finalizer barrier has an incomplete result")
            if contract["kind"] == GRAPH_RESULT_FINALIZER_KIND:
                work_results.append(
                    {
                        "task_id": task_id,
                        "result_id": result.result_id,
                        "result_digest": result.result_digest,
                        "result_kind": result.result_kind,
                        "summary": result.summary,
                        "payload": result.payload,
                        "artifact_ids": result.artifact_ids,
                        "effect_receipt_ids": result.effect_receipt_ids,
                        "sensitivity": result.sensitivity.value,
                    }
                )
                continue
            if len(result.artifact_ids) != 1:
                raise ValueError("the finalizer barrier has an incomplete result")
            raw_refs = result.provenance.get("artifact_refs")
            if not isinstance(raw_refs, tuple) or len(raw_refs) != 1:
                raise ValueError("a required accepted artifact is unauthenticated")
            work_results.append(
                {
                    "task_id": task_id,
                    "result_id": result.result_id,
                    "result_digest": result.result_digest,
                    "artifact_ref": raw_refs[0],
                }
            )
        if contract["kind"] == GRAPH_RESULT_FINALIZER_KIND:
            return {
                "job_id": inspection.job.job_id,
                "work_results": tuple(work_results),
            }
        resource_ids = contract.get("resource_ids")
        sample_rows = contract.get("sample_rows")
        if not isinstance(resource_ids, tuple) or not isinstance(sample_rows, int):
            raise ValueError("graph finalizer contract is malformed")
        return {
            "job_id": inspection.job.job_id,
            "resource_ids": resource_ids,
            "sample_rows": sample_rows,
            "work_results": tuple(work_results),
        }

    async def _promote_graph_artifact(
        self,
        inspection: GraphInspection,
        task: GraphTask,
        attempt: TaskAttempt,
        artifact: ArtifactRef,
        content: bytes,
    ) -> TaskResult:
        contract = internal_task_contract(task)
        _validate_graph_artifact(
            inspection,
            task,
            attempt,
            artifact,
            content,
            capability_id=str(contract["capability_id"]),
        )
        if contract["kind"] == GRAPH_RESULT_FINALIZER_KIND:
            payload = _generic_graph_result_payload(
                content,
                job_id=inspection.job.job_id,
            )
            summary = (
                f"Aggregated {payload['accepted_result_count']} authenticated "
                "graph task result(s)."
            )
        else:
            payload = _profile_result_payload(
                content,
                job_id=inspection.job.job_id,
                expected_resource_ids=(
                    task.specification.authority.resource_ids
                    if contract["kind"] == PROFILE_WORK_KIND
                    else _contract_resource_ids(contract)
                ),
            )
            summary = (
                f"Profiled {payload['profiled_resources']} exact resource(s) with "
                f"{payload['sampled_rows']} sampled row(s)."
            )
        completed_at = self._clock()
        result = _graph_task_result(
            task=task,
            attempt=attempt,
            result_id=self._id_factory("result"),
            result_kind=str(contract["output_kind"]),
            payload=payload,
            artifact=artifact,
            completed_at=completed_at,
            summary=summary,
        )
        if task.role is TaskRole.FINALIZER:
            await self._finalize_graph_result(inspection, attempt, result, artifact)
        else:
            try:
                await self._graph_store_call(
                    lambda: self._store.complete_graph_attempt(
                        result,
                        claim_token=attempt.claim_token,
                        fencing_epoch=attempt.fencing_epoch,
                        usage=(BudgetAmount("work_units", 1),),
                    )
                )
            except Exception:
                current = await self._graph_store_call(
                    lambda: self._store.inspect_graph(result.agent_id, result.job_id)
                )
                accepted = (
                    None
                    if current is None
                    else next(
                        (
                            item
                            for item in current.results
                            if item.task_id == result.task_id
                        ),
                        None,
                    )
                )
                if accepted != result:
                    raise
        return result

    async def _finalize_graph_result(
        self,
        inspection: GraphInspection,
        attempt: TaskAttempt,
        result: TaskResult,
        artifact: ArtifactRef,
    ) -> None:
        contract = internal_task_contract(inspection_task(inspection, attempt.task_id))
        generic_result = contract["kind"] == GRAPH_RESULT_FINALIZER_KIND
        requirement = ArtifactRequirement(
            required=True,
            minimum_count=1,
            maximum_count=1,
            allowed_media_types=("application/json",),
            allowed_authorships=(
                (
                    ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS
                    if generic_result
                    else ArtifactAuthorship.EXACT_SOURCE_DATA
                ),
            ),
            allowed_producer_capability_ids=(str(contract["capability_id"]),),
            maximum_artifact_bytes=1 * 1024 * 1024,
            maximum_total_bytes=1 * 1024 * 1024,
            maximum_sensitivity=inspection.job.specification.authority.sensitivity,
        )
        outcome_contract = OutcomeContract(
            require_terminal_conclusion=True,
            artifact_requirements=(requirement,),
            maximum_total_artifact_bytes=1 * 1024 * 1024,
            maximum_effective_sensitivity=(
                inspection.job.specification.authority.sensitivity
            ),
            require_current_run_provenance=True,
            require_exact_source_bindings=not generic_result,
        )
        artifact_references = validate_outcome_artifact_references(
            (outcome_artifact_reference(artifact),),
            contract=outcome_contract,
            resulting_run_id=attempt.run_id,
        )
        distribution = self._required_graph_distribution()
        target = distribution.resolve_conversation_inbox(
            inspection.job.conversation_id,
            sensitivity_ceiling=inspection.job.specification.authority.sensitivity,
        )
        plan = distribution.resolve_plan(
            inspection.job.conversation_id,
            destination_id=target.destination_id,
            sensitivity_ceiling=inspection.job.specification.authority.sensitivity,
        )
        if plan.plan_digest != inspection.job.specification.distribution_plan_digest:
            raise ValueError("graph distribution plan changed before finalization")
        delivery = construct_graph_job_delivery(
            delivery_id=self._id_factory("delivery"),
            agent_id=result.agent_id,
            conversation_id=inspection.job.conversation_id,
            job_id=result.job_id,
            target=target,
            result_id=result.result_id,
            result_digest=result.result_digest,
            result_summary=result.summary,
            artifact_references=artifact_references,
            effective_sensitivity=result.sensitivity,
            provenance_digest=canonical_digest(
                {
                    "job_id": result.job_id,
                    "result_digest": result.result_digest,
                    "artifact_ids": result.artifact_ids,
                }
            ),
            observed_at=result.completed_at,
        )
        try:
            await self._graph_store_call(
                lambda: self._store.finalize_graph_attempt(
                    result,
                    delivery,
                    claim_token=attempt.claim_token,
                    fencing_epoch=attempt.fencing_epoch,
                    usage=(BudgetAmount("work_units", 1),),
                )
            )
        except Exception:
            current, deliveries = await asyncio.gather(
                self._graph_store_call(
                    lambda: self._store.inspect_graph(result.agent_id, result.job_id)
                ),
                self._graph_store_call(
                    lambda: self._store.list_graph_deliveries(
                        result.agent_id,
                        job_id=result.job_id,
                        limit=2,
                    )
                ),
            )
            accepted = (
                None
                if current is None
                else next(
                    (
                        item
                        for item in current.results
                        if item.task_id == result.task_id
                    ),
                    None,
                )
            )
            if accepted != result or deliveries != (delivery,):
                raise


def _graph_attempt_guard(
    store: SQLiteStateStore,
    inspection: GraphInspection,
    task: GraphTask,
    attempt: TaskAttempt,
    *,
    clock: Callable[[], datetime],
) -> SQLiteTaskAttemptGuard:
    binding = graph_task_binding(inspection, task, attempt)
    return SQLiteTaskAttemptGuard(
        store=store,
        binding=binding,
        claim_token=attempt.claim_token,
        run_id=attempt.run_id,
        clock=clock,
    )


def _contract_resource_ids(contract: Mapping[str, object]) -> tuple[str, ...]:
    raw = contract.get("resource_ids")
    if not isinstance(raw, tuple) or any(not isinstance(item, str) for item in raw):
        raise ValueError("the graph finalizer resource contract is malformed")
    return raw


def _validate_graph_artifact(
    inspection: GraphInspection,
    task: GraphTask,
    attempt: TaskAttempt,
    artifact: ArtifactRef,
    content: bytes,
    *,
    capability_id: str,
) -> None:
    generic_finalizer = capability_id == "jobs.graph.result_finalize"
    if (
        artifact.run_id != attempt.run_id
        or artifact.conversation_id != inspection.job.conversation_id
        or artifact.call_id != f"graph-call-{attempt.attempt_id}"
        or artifact.capability_id != capability_id
        or artifact.artifact_id != reserved_artifact_id(attempt.attempt_id)
        or artifact.media_type != "application/json"
        or artifact.provenance.authorship
        is not (
            ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS
            if generic_finalizer
            else ArtifactAuthorship.EXACT_SOURCE_DATA
        )
        or artifact.byte_size != len(content)
    ):
        raise ValueError("the internal graph artifact differs from its attempt")
    expected_resources = set(task.specification.authority.resource_ids)
    actual_resources = {
        item.resource_id for item in artifact.provenance.resource_bindings
    }
    if expected_resources and actual_resources != expected_resources:
        raise ValueError("the internal graph artifact differs from its task scope")
    if (
        task.role is TaskRole.FINALIZER
        and not generic_finalizer
        and actual_resources != set(inspection.job.specification.authority.resource_ids)
    ):
        raise ValueError("the finalizer artifact lacks exact root resource bindings")


def _profile_result_payload(
    content: bytes,
    *,
    job_id: str,
    expected_resource_ids: tuple[str, ...],
) -> Mapping[str, object]:
    try:
        document = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("the profile artifact is not valid JSON") from error
    if (
        not isinstance(document, dict)
        or document.get("kind") != "data_profile"
        or document.get("job_id") != job_id
        or not isinstance(document.get("resources"), list)
    ):
        raise ValueError("the profile artifact has the wrong result contract")
    resources = document["resources"]
    resource_ids: list[str] = []
    sampled_rows = 0
    truncated_resources = 0
    for resource in resources:
        if not isinstance(resource, dict):
            raise ValueError("the profile artifact contains a malformed resource")
        resource_id = resource.get("resource_id")
        sampled = resource.get("sampled_rows")
        if (
            not isinstance(resource_id, str)
            or not isinstance(sampled, int)
            or isinstance(sampled, bool)
        ):
            raise ValueError("the profile artifact contains invalid result evidence")
        resource_ids.append(resource_id)
        sampled_rows += sampled
        if resource.get("truncated") is True:
            truncated_resources += 1
    if (
        document.get("resource_count") != len(resources)
        or tuple(sorted(resource_ids)) != tuple(sorted(expected_resource_ids))
        or len(resource_ids) != len(set(resource_ids))
    ):
        raise ValueError("the profile artifact does not cover its exact resources")
    return {
        "job_id": job_id,
        "profiled_resources": len(resources),
        "sampled_rows": sampled_rows,
        "truncated_resources": truncated_resources,
    }


def _generic_graph_result_payload(
    content: bytes,
    *,
    job_id: str,
) -> Mapping[str, object]:
    try:
        document = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("the graph finalizer artifact is not valid JSON") from error
    accepted = document.get("accepted_results") if isinstance(document, dict) else None
    if (
        not isinstance(document, dict)
        or document.get("kind") != "graph_result"
        or document.get("job_id") != job_id
        or not isinstance(accepted, list)
        or any(not isinstance(item, dict) for item in accepted)
    ):
        raise ValueError("the graph finalizer artifact has the wrong result contract")
    result_ids = tuple(item.get("result_id") for item in accepted)
    if any(not isinstance(item, str) for item in result_ids) or len(result_ids) != len(
        set(result_ids)
    ):
        raise ValueError("the graph finalizer accepted-result set is invalid")
    return {
        "job_id": job_id,
        "accepted_result_count": len(accepted),
        "accepted_result_ids": result_ids,
    }


def _graph_task_result(
    *,
    task: GraphTask,
    attempt: TaskAttempt,
    result_id: str,
    result_kind: str,
    payload: Mapping[str, object],
    artifact: ArtifactRef,
    completed_at: datetime,
    summary: str,
) -> TaskResult:
    schema_digest = canonical_digest(
        {
            "result_kind": result_kind,
            "task_contract": task.specification.expected_result_contract,
        }
    )
    sensitivity = ModelSensitivity(artifact.sensitivity.value)
    provenance = {
        "authority": "authenticated_internal_capability_artifact",
        "task_spec_digest": task.task_spec_digest,
        "artifact_refs": (artifact_ref_to_mapping(artifact),),
    }
    verification = {
        "artifact_sha256": artifact.sha256,
        "capability_id": artifact.capability_id,
        "artifact_authenticated": True,
    }
    digest_material = {
        "agent_id": task.agent_id,
        "job_id": task.job_id,
        "task_id": task.task_id,
        "result_id": result_id,
        "attempt_id": attempt.attempt_id,
        "run_id": attempt.run_id,
        "result_kind": result_kind,
        "schema_digest": schema_digest,
        "payload": payload,
        "summary": summary,
        "sensitivity": sensitivity.value,
        "provenance": provenance,
        "artifact_ids": (artifact.artifact_id,),
        "effect_receipt_ids": (),
        "verification": verification,
        "residual_risk": None,
        "downstream_constraints": {},
        "completed_at": completed_at.isoformat(),
    }
    return TaskResult(
        agent_id=task.agent_id,
        job_id=task.job_id,
        task_id=task.task_id,
        result_id=result_id,
        attempt_id=attempt.attempt_id,
        run_id=attempt.run_id,
        result_kind=result_kind,
        schema_digest=schema_digest,
        payload=payload,
        summary=str(digest_material["summary"]),
        sensitivity=sensitivity,
        provenance=provenance,
        artifact_ids=(artifact.artifact_id,),
        effect_receipt_ids=(),
        verification=verification,
        residual_risk=None,
        downstream_constraints={},
        completed_at=completed_at,
        result_digest=canonical_digest(digest_material),
    )


def _safe_graph_failure_code(error: BaseException) -> str:
    code = getattr(error, "code", None)
    if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9._:/-]{1,256}", code):
        return code
    if isinstance(error, ArtifactError):
        return error.code
    if isinstance(error, CapabilityInputError):
        return error.code
    return "graph_internal_execution_failed"


def _graph_failure_is_retryable(error: BaseException) -> bool:
    if isinstance(error, DaitaError):
        if error.retryability is ErrorRetryability.PERMANENT:
            return False
        if error.retryability in (
            ErrorRetryability.TRANSIENT,
            ErrorRetryability.RETRYABLE,
        ):
            return True
    return not isinstance(
        error,
        (ArtifactError, CapabilityInputError, TypeError, ValueError),
    )


def _required_generated_id(value: str, pattern: re.Pattern[str], name: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"job {name} identity factory returned an invalid identity")
    return value


__all__ = ["JobSupervisor"]
