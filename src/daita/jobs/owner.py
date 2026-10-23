"""Own checklist-job records: start, answer, cancel and the model's job tools."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from datetime import datetime
from hashlib import sha256
from typing import Protocol

from .._json import canonical_json
from ..adapters.models import SourceRegistration
from ..capabilities import RecoveryKind, ToolRecovery
from ..distribution.models import (
    Delivery,
    DeliverySubjectKind,
    OutcomeConclusionKind,
    OutcomeState,
    conclusion_preview_projection,
)
from ..distribution.owner import construct_logical_delivery, conversation_inbox_target
from ..llm.models import ModelSensitivity
from ..routines.owner import RoutineError, _resolve_sources
from ..scope import ScopeRefs
from ..storage.errors import JobStoreConflictError
from ..storage.sqlite_records import EffectReceipt
from .models import (
    JOB_LIMIT_FAILURES,
    JOB_RUN_LIMIT,
    JOB_TOKEN_LIMIT_RUNS,
    LIVE_JOB_STATES,
    MAX_ANSWER_CHARACTERS,
    MAX_ANSWERS,
    MAX_DELEGATIONS,
    MAX_FINAL_SUMMARY_CHARACTERS,
    MAX_GOAL_CHARACTERS,
    MAX_JOB_LIST_PAGE_SIZE,
    MAX_OBJECTIVE_CHARACTERS,
    MAX_QUESTION_CHARACTERS,
    MAX_SUMMARY_CHARACTERS,
    UNCERTAIN_WRITE_QUESTION,
    ChecklistStatus,
    Delegation,
    DelegationState,
    Job,
    JobError,
    JobState,
    UserAnswer,
    clean_text,
    require_startable,
    with_item_status,
    written_checklist,
)

# Every job result goes to its conversation's Inbox, with no ceiling of its own.
JOB_INBOX_SENSITIVITY = ModelSensitivity.RESTRICTED
_LIMIT_QUESTION = "The job reached its limits. Raise them and continue?"
_CONFLICT_RETRIES = 8


class JobStore(Protocol):
    async def list_sources(self, agent_id: str) -> tuple[SourceRegistration, ...]: ...

    async def admit_job(self, job: Job) -> Job: ...

    async def load_job(self, agent_id: str, job_id: str) -> Job | None: ...

    async def list_jobs(
        self, agent_id: str, *, states: frozenset[JobState], limit: int
    ) -> tuple[Job, ...]: ...

    async def replace_job(self, job: Job) -> Job: ...

    async def finish_job(self, job: Job, delivery: Delivery | None) -> Job: ...

    async def job_effect_receipts(
        self, agent_id: str, job_id: str
    ) -> tuple[EffectReceipt, ...]: ...


class JobOwner:
    """The one writer of job records in this process.

    Record changes are serialized here and compare-and-set in the store, so a
    receipt that holds a job for a person is never overwritten.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        store: JobStore,
        clock: Callable[[], datetime],
        id_factory: Callable[[str], str],
        per_run_max_tokens: int,
    ) -> None:
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("job owner agent_id must be non-empty text")
        if per_run_max_tokens < 1:
            raise ValueError("job owner needs the agent's run limits")
        self.agent_id = agent_id
        self._store = store
        self._clock = clock
        self._id_factory = id_factory
        self._token_allowance = JOB_TOKEN_LIMIT_RUNS * per_run_max_tokens
        self._lock = asyncio.Lock()
        self._wake: Callable[[], None] | None = None
        self._cancel: Callable[[str], None] | None = None

    def bind_supervisor(
        self, *, wake: Callable[[], None], cancel: Callable[[str], None]
    ) -> None:
        if self._wake is not None:
            raise RuntimeError("job owner is already bound to a supervisor")
        self._wake = wake
        self._cancel = cancel

    # People and foreground runs.

    async def start(
        self,
        objective: str,
        *,
        conversation_id: str,
        sources: Iterable[str] = (),
        refs: ScopeRefs | None = None,
    ) -> Job:
        text = clean_text(objective, MAX_OBJECTIVE_CHARACTERS, lines=True)
        if not text:
            raise JobError("job_objective_required", "A job needs an objective.")
        source_ids = await self.resolve_sources(tuple(sources), refs)
        now = self._clock()
        job = Job(
            job_id=self._id_factory("job"),
            agent_id=self.agent_id,
            conversation_id=conversation_id,
            objective=text,
            source_ids=source_ids,
            state=JobState.RUNNING,
            created_at=now,
            updated_at=now,
            run_limit=JOB_RUN_LIMIT,
            token_limit=self._token_allowance,
        )
        admitted = await self._store.admit_job(job)
        self._notify()
        return admitted

    async def resolve_sources(
        self, requested: tuple[str, ...], refs: ScopeRefs | None = None
    ) -> tuple[str, ...]:
        """Resolve source handles, names or ids against current sources."""

        if not requested:
            return ()
        registrations = tuple(
            item
            for item in await self._store.list_sources(self.agent_id)
            if item.active
        )
        try:
            return _resolve_sources(requested, registrations, refs)
        except RoutineError as error:
            raise JobError(
                "job_source_unknown", str(error), _recovery(error.details)
            ) from None

    async def list(
        self, *, states: frozenset[JobState] = frozenset(), limit: int = 50
    ) -> tuple[Job, ...]:
        if not 1 <= limit <= MAX_JOB_LIST_PAGE_SIZE:
            raise ValueError("job list limit is outside its bound")
        return await self._store.list_jobs(self.agent_id, states=states, limit=limit)

    async def inspect(self, job_id: str) -> Job | None:
        return await self._store.load_job(self.agent_id, job_id)

    async def committed_writes(self, job_id: str) -> tuple[EffectReceipt, ...]:
        return await self._store.job_effect_receipts(self.agent_id, job_id)

    async def answer(self, job_id: str, text: str) -> Job:
        """Answer a job's question, or raise a limit it stopped at, and resume."""

        answer = clean_text(text, MAX_ANSWER_CHARACTERS, lines=True)
        if not answer:
            raise JobError("job_answer_required", "An answer needs some text.")

        def change(job: Job) -> Job:
            now = max(self._clock(), job.updated_at)
            if job.state is JobState.WAITING_FOR_YOU:
                assert job.question is not None
                if job.question == UNCERTAIN_WRITE_QUESTION:
                    raise JobError(
                        "job_write_uncertain",
                        "Resolve the uncertain write under Effects first; the job "
                        "continues once nothing is uncertain.",
                    )
                return replace(
                    job,
                    state=JobState.RUNNING,
                    question=None,
                    answers=_answers(job, job.question, answer, now),
                    updated_at=now,
                )
            if job.state is JobState.FAILED and job.failure_code in JOB_LIMIT_FAILURES:
                return replace(
                    job,
                    state=JobState.RUNNING,
                    failure_code=None,
                    run_limit=job.run_limit + JOB_RUN_LIMIT,
                    token_limit=job.token_limit + self._token_allowance,
                    answers=_answers(job, _LIMIT_QUESTION, answer, now),
                    updated_at=now,
                )
            raise JobError(
                "job_not_waiting",
                "This job is not waiting for you.",
                ToolRecovery(RecoveryKind.TERMINAL, f"The job is {job.state.value}."),
            )

        answered = await self._required_update(job_id, change)
        self._notify()
        return answered

    async def cancel(self, job_id: str) -> Job | None:
        def change(job: Job) -> Job | None:
            if job.state not in LIVE_JOB_STATES:
                return None
            return replace(
                job,
                state=JobState.CANCELLED,
                question=None,
                delegations=tuple(
                    (
                        replace(item, state=DelegationState.CANCELLED)
                        if item.state is DelegationState.RUNNING
                        else item
                    )
                    for item in job.delegations
                ),
                updated_at=max(self._clock(), job.updated_at),
            )

        current = await self._update(job_id, change)
        if current is not None and self._cancel is not None:
            self._cancel(job_id)
        return current

    # The job's own runs.

    async def write_checklist(
        self, job_id: str, items: Iterable[Mapping[str, object]], *, merge: bool
    ) -> Job:
        raw = tuple(items)
        return await self._required_update(
            job_id,
            lambda job: replace(
                _running(job),
                checklist=written_checklist(job.checklist, raw, merge=merge),
                updated_at=max(self._clock(), job.updated_at),
            ),
        )

    async def ask(self, job_id: str, question: str) -> Job:
        text = clean_text(question, MAX_QUESTION_CHARACTERS, lines=True)
        if not text:
            raise JobError("job_question_required", "A question needs some text.")
        asked = await self._required_update(
            job_id,
            lambda job: replace(
                _running(job),
                state=JobState.WAITING_FOR_YOU,
                question=text,
                updated_at=max(self._clock(), job.updated_at),
            ),
        )
        self._notify()
        return asked

    async def begin_delegation(
        self, job_id: str, *, run_id: str, goal: str, item: str | None
    ) -> tuple[Job, Delegation]:
        delegation = Delegation(
            id=self._id_factory("delegation"),
            goal=clean_text(goal, MAX_GOAL_CHARACTERS, lines=True) or "Sub-run",
            state=DelegationState.RUNNING,
            run_id=run_id,
            item=item,
        )

        def change(job: Job) -> Job:
            _running(job)
            if len(job.delegations) >= MAX_DELEGATIONS:
                raise JobError(
                    "job_delegation_limit",
                    f"A job delegates at most {MAX_DELEGATIONS} times.",
                )
            checklist = job.checklist
            if item is not None:
                if item not in {entry.id for entry in checklist}:
                    raise JobError(
                        "checklist_item_unknown", f"No checklist item {item}."
                    )
                current = next(entry for entry in checklist if entry.id == item)
                if current.status in {
                    ChecklistStatus.DONE,
                    ChecklistStatus.CANCELLED,
                }:
                    raise JobError(
                        "checklist_item_finished",
                        f"Item {item} is already {current.status.value}.",
                    )
                require_startable(checklist, item)
                checklist = with_item_status(
                    checklist, item, ChecklistStatus.IN_PROGRESS
                )
            return replace(
                job,
                checklist=checklist,
                delegations=(*job.delegations, delegation),
                updated_at=max(self._clock(), job.updated_at),
            )

        return await self._required_update(job_id, change), delegation

    async def end_delegation(
        self,
        job_id: str,
        delegation_id: str,
        *,
        state: DelegationState,
        summary: str,
        tokens: int,
        sensitivity: ModelSensitivity = ModelSensitivity.PUBLIC,
    ) -> Job | None:
        """Record a sub-run's end; its item is done, or pending again."""

        text = clean_text(summary, MAX_SUMMARY_CHARACTERS, lines=True) or None

        def change(job: Job) -> Job | None:
            current = next(
                (item for item in job.delegations if item.id == delegation_id), None
            )
            if current is None:
                return None
            checklist = job.checklist
            if (
                current.item is not None
                and current.state is DelegationState.RUNNING
                and current.item in {entry.id for entry in checklist}
            ):
                checklist = with_item_status(
                    checklist,
                    current.item,
                    (
                        ChecklistStatus.DONE
                        if state is DelegationState.DONE
                        else ChecklistStatus.PENDING
                    ),
                )
            ended = replace(
                current,
                state=(
                    current.state
                    if current.state is not DelegationState.RUNNING
                    else state
                ),
                summary=text,
            )
            return replace(
                job,
                checklist=checklist,
                delegations=tuple(
                    ended if item.id == delegation_id else item
                    for item in job.delegations
                ),
                tokens_used=job.tokens_used + tokens,
                sensitivity=_higher(job.sensitivity, sensitivity),
                updated_at=max(self._clock(), job.updated_at),
            )

        return await self._update(job_id, change)

    # The supervisor.

    async def begin_run(self, job_id: str, run_id: str) -> Job | None:
        """Count one more main run, or stop the job at its limits."""

        def change(job: Job) -> Job | None:
            if job.state is not JobState.RUNNING:
                return None
            now = max(self._clock(), job.updated_at)
            if job.run_count >= job.run_limit or job.tokens_used >= job.token_limit:
                return replace(
                    job,
                    state=JobState.FAILED,
                    failure_code=(
                        "job_run_limit_reached"
                        if job.run_count >= job.run_limit
                        else "job_token_limit_reached"
                    ),
                    updated_at=now,
                )
            return replace(
                job,
                current_run_id=run_id,
                run_count=job.run_count + 1,
                updated_at=now,
            )

        return await self._update(job_id, change)

    async def end_run(
        self,
        job_id: str,
        run_id: str,
        *,
        tokens: int,
        state: JobState | None = None,
        question: str | None = None,
        failure_code: str | None = None,
        final_text: str | None = None,
        sensitivity: ModelSensitivity = ModelSensitivity.PUBLIC,
    ) -> Job | None:
        """Charge one main run and apply its outcome; ``done`` delivers it."""

        def change(job: Job) -> Job | None:
            if job.current_run_id != run_id:
                return None
            charged = replace(
                job,
                tokens_used=job.tokens_used + tokens,
                sensitivity=_higher(job.sensitivity, sensitivity),
                updated_at=max(self._clock(), job.updated_at),
            )
            if job.state is not JobState.RUNNING or state is None:
                return charged
            if state is JobState.DONE:
                return replace(
                    charged,
                    state=JobState.DONE,
                    final_summary=clean_text(
                        final_text or "", MAX_FINAL_SUMMARY_CHARACTERS, lines=True
                    ),
                )
            return replace(
                charged, state=state, question=question, failure_code=failure_code
            )

        def delivery(job: Job) -> Delivery | None:
            if job.state is not JobState.DONE or final_text is None:
                return None
            return _done_delivery(
                job,
                run_id,
                final_text,
                job.sensitivity,
                delivery_id=self._id_factory("delivery"),
            )

        ended = await self._update(job_id, change, delivery=delivery)
        if ended is not None and ended.state is not JobState.RUNNING:
            self._notify()
        return ended

    async def interrupt_delegations(self, reason: str) -> None:
        """Sub-runs left running by a restart failed; the model sees why."""

        for job in await self.list(
            states=LIVE_JOB_STATES, limit=MAX_JOB_LIST_PAGE_SIZE
        ):
            for item in job.running_delegations:
                await self.end_delegation(
                    job.job_id,
                    item.id,
                    state=DelegationState.FAILED,
                    summary=reason,
                    tokens=0,
                )

    # Record mechanics.

    async def _required_update(
        self, job_id: str, change: Callable[[Job], Job | None]
    ) -> Job:
        updated = await self._update(job_id, change)
        if updated is None:
            raise JobError("job_not_found", "The job is not owned by this agent.")
        return updated

    async def _update(
        self,
        job_id: str,
        change: Callable[[Job], Job | None],
        *,
        delivery: Callable[[Job], Delivery | None] | None = None,
    ) -> Job | None:
        async with self._lock:
            for _ in range(_CONFLICT_RETRIES):
                current = await self._store.load_job(self.agent_id, job_id)
                if current is None:
                    return None
                changed = change(current)
                if changed is None:
                    return current
                if changed == current:
                    return current
                written = replace(changed, revision=current.revision + 1)
                try:
                    if delivery is None:
                        return await self._store.replace_job(written)
                    return await self._store.finish_job(written, delivery(written))
                except JobStoreConflictError:
                    continue
            raise JobError("job_busy", "The job kept changing; try again.")

    def _notify(self) -> None:
        if self._wake is not None:
            self._wake()


def _higher(left: ModelSensitivity, right: ModelSensitivity) -> ModelSensitivity:
    return max(left, right, key=lambda item: item.routing_rank)


def _running(job: Job) -> Job:
    if job.state is not JobState.RUNNING:
        raise JobError(
            "job_not_running",
            f"The job is {job.state.value}; it takes no more changes.",
        )
    return job


def _answers(
    job: Job, question: str, text: str, at: datetime
) -> tuple[UserAnswer, ...]:
    answer = UserAnswer(question=question, text=text, answered_at=at)
    return (*job.answers, answer)[-MAX_ANSWERS:]


def _recovery(details: Mapping[str, object]) -> ToolRecovery | None:
    recovery = details.get("recovery")
    if not isinstance(recovery, Mapping):
        return None
    constraint = recovery.get("constraint")
    return (
        ToolRecovery(RecoveryKind.CORRECT_ARGUMENTS, constraint)
        if isinstance(constraint, str) and constraint
        else None
    )


def _done_delivery(
    job: Job,
    run_id: str,
    final_text: str,
    sensitivity: ModelSensitivity,
    *,
    delivery_id: str,
) -> Delivery:
    # shortcut: the Done card carries the final answer, not artifacts; add the
    # final run's artifact references if people need them in the Inbox.
    digest, preview, truncated = conclusion_preview_projection(final_text)
    observed_at = job.updated_at
    return construct_logical_delivery(
        delivery_id=delivery_id,
        agent_id=job.agent_id,
        conversation_id=job.conversation_id,
        subject_kind=DeliverySubjectKind.JOB,
        subject_id=job.job_id,
        target=conversation_inbox_target(
            job.conversation_id, sensitivity_ceiling=JOB_INBOX_SENSITIVITY
        ),
        conclusion_kind=OutcomeConclusionKind.TERMINAL_RUN,
        conclusion_state=OutcomeState.SUCCEEDED,
        conclusion_id=run_id,
        conclusion_digest=digest,
        conclusion_preview=preview,
        conclusion_preview_truncated=truncated,
        resulting_run_id=run_id,
        artifact_references=(),
        effective_sensitivity=sensitivity,
        provenance_digest="sha256:"
        + sha256(
            canonical_json(
                {"job_id": job.job_id, "run_id": run_id, "result": digest}
            ).encode("utf-8")
        ).hexdigest(),
        failure_code=None,
        observed_at=observed_at,
    )


__all__ = ["JOB_INBOX_SENSITIVITY", "JobOwner", "JobStore"]
