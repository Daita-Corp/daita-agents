"""Admit durable jobs and own their lifecycle transitions and public projections."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Protocol, cast

from ..capabilities import ExecutionContractBindings
from ..errors import DaitaError, ErrorRetryability
from .graph.models import (
    BudgetAmount,
    ControlKind,
    ControlState,
    GraphAdmission,
    GraphAuthority,
    GraphEventPage,
    GraphInspection,
    GraphJob,
    GraphMutation,
    GraphMutationRequest,
    GraphState,
    GraphTask,
    GraphTaskSpecification,
    TaskAttempt,
    TaskCheckpoint,
    TaskComment,
    TaskControl,
    TaskDependency,
    TaskExecutionKind,
    TaskResult,
    TaskRole,
    TaskState,
    canonical_digest,
)
from .graph.planning import planner_task_from_template
from .projections import (
    GraphBoardProjection,
    GraphTimelinePage,
    bounded_dependencies,
    bounded_tasks,
    graph_board,
    graph_diagnostics,
)


class JobError(DaitaError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(
            message,
            error_code=code,
            retryability=ErrorRetryability.PERMANENT,
        )


class GraphJobStore(Protocol):
    async def admit_graph(self, admission: GraphAdmission) -> GraphJob: ...

    async def admit_replacement_graph(
        self,
        admission: GraphAdmission,
        *,
        replaced_job_id: str,
        replaced_task_id: str,
        control_id: str,
        principal_id: str,
        idempotency_key: str,
        resolved_at: datetime,
        expected_control_digest: str,
        expected_task_revision: int,
    ) -> GraphJob: ...

    async def inspect_graph(
        self, agent_id: str, job_id: str
    ) -> GraphInspection | None: ...

    async def list_graph_jobs(
        self,
        agent_id: str,
        *,
        states: frozenset[GraphState] = frozenset(),
        limit: int = 50,
    ) -> tuple[GraphJob, ...]: ...

    async def apply_graph_mutation(
        self, request: GraphMutationRequest
    ) -> GraphMutation: ...

    async def list_graph_events(
        self,
        agent_id: str,
        job_id: str,
        *,
        after_event_id: int = 0,
        limit: int = 100,
        task_id: str | None = None,
    ) -> GraphEventPage: ...

    async def checkpoint_graph_attempt(
        self, checkpoint: TaskCheckpoint, *, claim_token: str
    ) -> TaskCheckpoint: ...

    async def add_graph_comment(
        self,
        comment: TaskComment,
        *,
        attempt_id: str | None = None,
        claim_token: str | None = None,
        fencing_epoch: int | None = None,
    ) -> TaskComment: ...

    async def complete_graph_attempt(
        self,
        result: TaskResult,
        *,
        claim_token: str,
        fencing_epoch: int,
        usage: tuple[BudgetAmount, ...] | None,
    ) -> TaskResult: ...

    async def open_graph_control(
        self,
        control: TaskControl,
        *,
        claim_token: str,
        fencing_epoch: int,
        replan_task: GraphTask | None = None,
        reviewer_task: GraphTask | None = None,
    ) -> TaskControl: ...

    async def accept_graph_review(
        self,
        *,
        agent_id: str,
        job_id: str,
        subject_task_id: str,
        control_id: str,
        reviewer_task_id: str,
        resolved_at: datetime,
        resolved_by_kind: str,
        resolved_by_id: str,
        rationale: str,
        idempotency_key: str,
        expected_control_digest: str,
        expected_subject_revision: int,
        reviewer_result: TaskResult | None = None,
        claim_token: str | None = None,
        fencing_epoch: int | None = None,
    ) -> TaskResult: ...

    async def request_graph_review_changes(
        self,
        *,
        changes_control: TaskControl,
        review_control_id: str,
        reviewer_task_id: str,
        resolved_at: datetime,
        resolved_by_kind: str,
        resolved_by_id: str,
        rationale: str,
        idempotency_key: str,
        expected_control_digest: str,
        expected_subject_revision: int,
        reviewer_result: TaskResult | None = None,
        claim_token: str | None = None,
        fencing_epoch: int | None = None,
    ) -> TaskControl: ...

    async def resolve_graph_control(
        self,
        agent_id: str,
        job_id: str,
        task_id: str,
        control_id: str,
        *,
        state: ControlState,
        resolved_at: datetime,
        resolved_by_kind: str,
        resolved_by_id: str,
        resolution: dict[str, object],
        make_ready: bool,
        expected_control_digest: str | None = None,
        expected_task_revision: int | None = None,
    ) -> TaskControl | None: ...

    async def request_graph_cancel(
        self,
        agent_id: str,
        job_id: str,
        *,
        requested_at: datetime,
        requested_by_id: str,
    ) -> GraphJob | None: ...


@dataclass(frozen=True, slots=True)
class GraphBlockerProjection:
    job_id: str
    graph_state: GraphState
    blockers: tuple[Mapping[str, object], ...]


class JobOwner:
    """Admit frozen jobs and expose only bounded owner-scoped lifecycle views."""

    def __init__(
        self,
        *,
        agent_id: str,
        store: GraphJobStore,
        clock: Callable[[], datetime],
        id_factory: Callable[[str], str],
    ) -> None:
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("job owner agent_id must be non-empty text")
        for method in (
            "admit_graph",
            "admit_replacement_graph",
            "inspect_graph",
            "list_graph_jobs",
            "request_graph_cancel",
        ):
            if not callable(getattr(store, method, None)):
                raise TypeError(f"job store must provide {method}")
        if not callable(clock) or not callable(id_factory):
            raise TypeError("job owner clock and id_factory must be callable")
        self.agent_id = agent_id
        self._store = store
        self._clock = clock
        self._id_factory = id_factory
        self._wake: Callable[[str | None], None] | None = None

    def bind_wake(self, wake: Callable[[str | None], None]) -> None:
        if self._wake is not None:
            raise RuntimeError("job owner wake callback is already bound")
        if not callable(wake):
            raise TypeError("job wake callback must be callable")
        self._wake = wake

    async def admit(self, admission: GraphAdmission) -> GraphJob:
        """Admit one current graph after its code-owned builder freezes authority."""

        if not isinstance(admission, GraphAdmission):
            raise TypeError("graph admission must be GraphAdmission")
        if admission.job.agent_id != self.agent_id:
            raise JobError("job_owner_mismatch", "The graph owner identity changed.")
        if admission.job.deadline_at <= self._clock():
            raise JobError(
                "job_deadline_expired", "The requested graph deadline has expired."
            )
        stored = await self._graph_store().admit_graph(admission)
        self._notify(stored.job_id)
        return stored

    async def list(
        self,
        *,
        states: frozenset[GraphState] = frozenset(),
        limit: int = 50,
    ) -> tuple[GraphJob, ...]:
        if not 1 <= limit <= 100:
            raise ValueError("job list limit must be between one and one hundred")
        return await self._graph_store().list_graph_jobs(
            self.agent_id,
            states=states,
            limit=limit,
        )

    async def inspect(self, job_id: str) -> GraphInspection | None:
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("job_id must be non-empty text")
        return await self._graph_store().inspect_graph(self.agent_id, job_id)

    async def read_result(self, job_id: str) -> TaskResult | None:
        inspection = await self.inspect(job_id)
        if inspection is None:
            return None
        return next(
            (
                result
                for result in inspection.results
                if result.task_id == inspection.job.finalizer_task_id
            ),
            None,
        )

    async def cancel(self, job_id: str) -> GraphJob | None:
        inspection = await self.inspect(job_id)
        if inspection is None:
            return None
        updated = await self._graph_store().request_graph_cancel(
            self.agent_id,
            job_id,
            requested_at=self._clock(),
            requested_by_id=inspection.job.specification.principal_id,
        )
        if updated is not None:
            self._notify(job_id)
        return updated

    async def admit_authorized_replacement_graph(
        self,
        admission: GraphAdmission,
        *,
        replaces_job_id: str,
        principal_id: str,
        replaces_task_id: str | None = None,
        control_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> GraphJob:
        """Start a separately authorized job; never widen an existing root."""

        prior = await self.inspect_graph(replaces_job_id)
        if prior is None:
            raise JobError("unknown_graph", "The replaced graph job is unavailable.")
        if (
            prior.job.specification.principal_id != principal_id
            or admission.job.specification.principal_id != principal_id
        ):
            raise JobError(
                "replacement_principal_mismatch",
                "The replacement job principal differs from its authorization.",
            )
        job = replace(
            admission.job,
            migration_provenance={
                **dict(admission.job.migration_provenance),
                "replaces_job_id": replaces_job_id,
                "authorized_by_principal": principal_id,
                **(
                    {}
                    if control_id is None
                    else {
                        "replacement_control_id": control_id,
                        "replacement_idempotency_key": idempotency_key,
                    }
                ),
            },
        )
        prepared = replace(admission, job=job)
        supplied = (
            replaces_task_id is not None,
            control_id is not None,
            idempotency_key is not None,
        )
        if not any(supplied):
            return await self.admit(prepared)
        if not all(supplied):
            raise ValueError(
                "replacement control task, control and idempotency must be supplied together"
            )
        assert replaces_task_id is not None
        assert control_id is not None
        assert idempotency_key is not None
        if not idempotency_key:
            raise ValueError("replacement job idempotency key must be non-empty")
        control = next(
            (
                item
                for item in prior.controls
                if item.task_id == replaces_task_id and item.control_id == control_id
            ),
            None,
        )
        task = next(
            (item for item in prior.tasks if item.task_id == replaces_task_id), None
        )
        if control is None or task is None:
            raise JobError(
                "replacement_control_unavailable",
                "The exact replacement authorization control is unavailable.",
            )
        replacement = await self._graph_store().admit_replacement_graph(
            prepared,
            replaced_job_id=replaces_job_id,
            replaced_task_id=replaces_task_id,
            control_id=control_id,
            principal_id=principal_id,
            idempotency_key=idempotency_key,
            resolved_at=self._clock(),
            expected_control_digest=control.payload_digest,
            expected_task_revision=task.task_revision,
        )
        self._notify(replaces_job_id)
        self._notify(replacement.job_id)
        return replacement

    async def inspect_graph(self, job_id: str) -> GraphInspection | None:
        return await self.inspect(job_id)

    async def list_graph_tasks(
        self,
        job_id: str,
        *,
        states: frozenset[TaskState] = frozenset(),
        limit: int = 64,
    ) -> tuple[GraphTask, ...]:
        inspection = await self.inspect_graph(job_id)
        if inspection is None:
            return ()
        return bounded_tasks(inspection, states=states, limit=limit)

    async def inspect_graph_task(self, job_id: str, task_id: str) -> GraphTask | None:
        inspection = await self.inspect_graph(job_id)
        if inspection is None:
            return None
        return next(
            (item for item in inspection.tasks if item.task_id == task_id), None
        )

    async def list_graph_dependencies(
        self,
        job_id: str,
        *,
        task_id: str | None = None,
        limit: int = 100,
    ) -> tuple[TaskDependency, ...]:
        inspection = await self.inspect_graph(job_id)
        if inspection is None:
            return ()
        return bounded_dependencies(inspection, task_id=task_id, limit=limit)

    async def list_graph_task_attempts(
        self, job_id: str, task_id: str, *, limit: int = 3
    ) -> tuple[TaskAttempt, ...]:
        if not 1 <= limit <= 3:
            raise ValueError("task attempt list limit must be between one and three")
        inspection = await self.inspect_graph(job_id)
        if inspection is None:
            return ()
        return tuple(item for item in inspection.attempts if item.task_id == task_id)[
            -limit:
        ]

    async def read_graph_task_result(
        self, job_id: str, task_id: str
    ) -> TaskResult | None:
        inspection = await self.inspect_graph(job_id)
        if inspection is None:
            return None
        return next(
            (item for item in inspection.results if item.task_id == task_id), None
        )

    async def list_graph_task_checkpoints(
        self, job_id: str, task_id: str, *, limit: int = 8
    ) -> tuple[TaskCheckpoint, ...]:
        if not 1 <= limit <= 8:
            raise ValueError("checkpoint list limit must be between one and eight")
        inspection = await self.inspect_graph(job_id)
        if inspection is None:
            return ()
        return tuple(
            item for item in inspection.checkpoints if item.task_id == task_id
        )[-limit:]

    async def list_graph_task_artifacts(
        self, job_id: str, *, task_id: str | None = None, limit: int = 64
    ) -> tuple[str, ...]:
        if not 1 <= limit <= 64:
            raise ValueError("artifact list limit must be between one and sixty-four")
        inspection = await self.inspect_graph(job_id)
        if inspection is None:
            return ()
        return tuple(
            sorted(
                {
                    artifact_id
                    for result in inspection.results
                    if task_id is None or result.task_id == task_id
                    for artifact_id in result.artifact_ids
                }
            )
        )[:limit]

    async def list_graph_task_controls(
        self, job_id: str, task_id: str, *, limit: int = 8
    ) -> tuple[TaskControl, ...]:
        if not 1 <= limit <= 8:
            raise ValueError("task control list limit must be between one and eight")
        inspection = await self.inspect_graph(job_id)
        if inspection is None:
            return ()
        return tuple(item for item in inspection.controls if item.task_id == task_id)[
            -limit:
        ]

    async def graph_board(self, job_id: str) -> GraphBoardProjection | None:
        inspection = await self.inspect_graph(job_id)
        return None if inspection is None else graph_board(inspection)

    async def graph_timeline(
        self,
        job_id: str,
        *,
        after_event_id: int = 0,
        limit: int = 100,
        task_id: str | None = None,
    ) -> GraphTimelinePage | None:
        inspection = await self.inspect_graph(job_id)
        if inspection is None:
            return None
        page = await self._graph_store().list_graph_events(
            self.agent_id,
            job_id,
            after_event_id=after_event_id,
            limit=limit,
            task_id=task_id,
        )
        return GraphTimelinePage(
            job_id=job_id,
            graph_state=inspection.job.state,
            graph_revision=inspection.graph.revision,
            events=page.events,
            next_cursor=page.next_cursor,
            diagnostics=graph_diagnostics(inspection),
        )

    async def mutate_graph(self, request: GraphMutationRequest) -> GraphMutation:
        self._require_graph_record_owner(request.agent_id)
        mutation = await self._graph_store().apply_graph_mutation(request)
        self._notify(request.job_id)
        return mutation

    async def graph_blockers(self, job_id: str) -> GraphBlockerProjection | None:
        inspection = await self.inspect_graph(job_id)
        if inspection is None:
            return None
        diagnostics = graph_diagnostics(inspection)
        return GraphBlockerProjection(
            job_id=job_id,
            graph_state=inspection.job.state,
            blockers=diagnostics.blockers[:32],
        )

    async def answer_graph_task_input(
        self,
        job_id: str,
        task_id: str,
        control_id: str,
        *,
        principal_id: str,
        answer: Mapping[str, object],
        idempotency_key: str | None = None,
    ) -> TaskControl | None:
        inspection = await self.inspect_graph(job_id)
        control = (
            None
            if inspection is None
            else next(
                (
                    item
                    for item in inspection.controls
                    if item.task_id == task_id and item.control_id == control_id
                ),
                None,
            )
        )
        if control is None or control.kind is not ControlKind.NEEDS_INPUT:
            raise JobError(
                "task_input_unavailable",
                "The exact open human-input control is unavailable.",
            )
        assert inspection is not None
        self._require_graph_principal(inspection, principal_id)
        task = next(item for item in inspection.tasks if item.task_id == task_id)
        response_schema = control.payload.get("response_schema")
        if isinstance(response_schema, Mapping):
            from ..capabilities import (
                ToolOutputValidationError,
                validate_tool_schema_value,
            )

            try:
                validate_tool_schema_value(response_schema, answer)
            except (TypeError, ValueError, ToolOutputValidationError) as error:
                raise JobError(
                    "task_input_invalid",
                    "The answer does not match the control's bounded response schema.",
                ) from error
        key = idempotency_key or canonical_digest(
            {
                "action": "answer_input",
                "control_id": control_id,
                "principal_id": principal_id,
                "answer": answer,
            }
        )
        resolution: dict[str, object] = {
            "action": "answer_input",
            "answer": dict(answer),
            "idempotency_key": key,
        }
        resolved = await self._graph_store().resolve_graph_control(
            self.agent_id,
            job_id,
            task_id,
            control_id,
            state=ControlState.RESOLVED,
            resolved_at=self._clock(),
            resolved_by_kind="principal",
            resolved_by_id=principal_id,
            resolution=resolution,
            make_ready=True,
            expected_control_digest=control.payload_digest,
            expected_task_revision=task.task_revision,
        )
        if resolved is not None:
            self._notify(job_id)
        return resolved

    async def reject_graph_task_control(
        self,
        job_id: str,
        task_id: str,
        control_id: str,
        *,
        principal_id: str,
        reason: str,
        idempotency_key: str | None = None,
    ) -> TaskControl | None:
        inspection = await self.inspect_graph(job_id)
        if inspection is None:
            raise JobError("unknown_graph", "The graph job is unavailable.")
        self._require_graph_principal(inspection, principal_id)
        control = next(
            (
                item
                for item in inspection.controls
                if item.task_id == task_id and item.control_id == control_id
            ),
            None,
        )
        task = next(
            (item for item in inspection.tasks if item.task_id == task_id), None
        )
        if control is None or task is None:
            raise JobError(
                "task_control_unavailable", "The task control is unavailable."
            )
        if control.kind is ControlKind.REVIEW_REQUESTED:
            raise JobError(
                "typed_review_required", "Use a typed review decision for this control."
            )
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or len(reason.encode("utf-8")) > 4096
        ):
            raise ValueError("control rejection reason is outside its bound")
        key = idempotency_key or canonical_digest(
            {
                "action": "reject",
                "control_id": control_id,
                "principal_id": principal_id,
                "reason": reason,
            }
        )
        resolution: dict[str, object] = {
            "action": "reject",
            "reason": reason,
            "idempotency_key": key,
        }
        resolved = await self._graph_store().resolve_graph_control(
            self.agent_id,
            job_id,
            task_id,
            control_id,
            state=ControlState.REJECTED,
            resolved_at=self._clock(),
            resolved_by_kind="principal",
            resolved_by_id=principal_id,
            resolution=resolution,
            make_ready=False,
            expected_control_digest=control.payload_digest,
            expected_task_revision=task.task_revision,
        )
        if resolved is not None:
            self._notify(job_id)
        return resolved

    async def retry_graph_task_control(
        self,
        job_id: str,
        task_id: str,
        control_id: str,
        *,
        principal_id: str,
        advisory_note: str,
        idempotency_key: str,
    ) -> TaskControl | None:
        inspection = await self.inspect_graph(job_id)
        if inspection is None:
            raise JobError("unknown_graph", "The graph job is unavailable.")
        self._require_graph_principal(inspection, principal_id)
        control = next(
            (
                item
                for item in inspection.controls
                if item.task_id == task_id and item.control_id == control_id
            ),
            None,
        )
        task = next(
            (item for item in inspection.tasks if item.task_id == task_id), None
        )
        if control is None or task is None:
            raise JobError(
                "task_control_unavailable", "The task control is unavailable."
            )
        if control.kind in {
            ControlKind.REVIEW_REQUESTED,
            ControlKind.CHANGES_REQUESTED,
            ControlKind.NEEDS_AUTHORIZATION,
            ControlKind.EFFECT_UNCERTAIN,
        }:
            raise JobError(
                "control_retry_forbidden",
                "This control requires its dedicated typed outcome.",
            )
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError("attention idempotency key must be non-empty text")
        if not advisory_note.strip() or len(advisory_note.encode("utf-8")) > 4096:
            raise ValueError("attention advisory note is outside its bound")
        resolution: dict[str, object] = {
            "action": "retry",
            "advisory_note": advisory_note,
            "idempotency_key": idempotency_key,
        }
        resolved = await self._graph_store().resolve_graph_control(
            self.agent_id,
            job_id,
            task_id,
            control_id,
            state=ControlState.RESOLVED,
            resolved_at=self._clock(),
            resolved_by_kind="principal",
            resolved_by_id=principal_id,
            resolution=resolution,
            make_ready=True,
            expected_control_digest=control.payload_digest,
            expected_task_revision=task.task_revision,
        )
        if resolved is not None:
            self._notify(job_id)
        return resolved

    async def cancel_graph_job(
        self, job_id: str, *, principal_id: str
    ) -> GraphJob | None:
        inspection = await self.inspect_graph(job_id)
        if inspection is None:
            return None
        self._require_graph_principal(inspection, principal_id)
        cancelled = await self._graph_store().request_graph_cancel(
            self.agent_id,
            job_id,
            requested_at=self._clock(),
            requested_by_id=principal_id,
        )
        if cancelled is not None:
            self._notify(job_id)
        return cancelled

    async def replace_graph_task_by_policy(
        self,
        job_id: str,
        task_id: str,
        *,
        principal_id: str,
        advisory_note: str,
        idempotency_key: str,
        expected_revision: int,
    ) -> GraphMutation:
        inspection = await self.inspect_graph(job_id)
        if inspection is None:
            raise JobError("unknown_graph", "The graph job is unavailable.")
        self._require_graph_principal(inspection, principal_id)
        if (
            not isinstance(expected_revision, int)
            or isinstance(expected_revision, bool)
            or expected_revision < 0
        ):
            raise ValueError("replacement expected_revision must be non-negative")
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError("replacement idempotency key must be non-empty text")
        if (
            not isinstance(advisory_note, str)
            or not advisory_note.strip()
            or len(advisory_note.encode("utf-8")) > 4096
        ):
            raise ValueError("replacement advisory note is outside its bound")
        task = next(
            (item for item in inspection.tasks if item.task_id == task_id), None
        )
        actor_key = principal_id
        replacement_id = _stable_graph_id(
            "task",
            job_id=job_id,
            actor_key=actor_key,
            idempotency_key=idempotency_key,
            discriminator=f"policy-replacement:{task_id}",
        )
        retrying_committed = (
            task is not None
            and task.state is TaskState.SUPERSEDED
            and task.superseded_by_task_id == replacement_id
        )
        if task is None or (
            task.state is not TaskState.BLOCKED and not retrying_committed
        ):
            raise JobError(
                "replacement_not_allowed",
                "Only blocked work has a policy replacement; review-waiting work "
                "requires a typed review decision.",
            )
        now = self._clock()
        replacement_spec = replace(
            task.specification,
            description=(
                task.specification.description
                + "\n\nUntrusted human advisory: "
                + advisory_note
            ),
            created_by=actor_key,
        )
        replacement = replace(
            task,
            task_id=replacement_id,
            state=TaskState.READY,
            not_before=None,
            current_attempt_id=None,
            task_revision=1,
            specification=replacement_spec,
            task_spec_digest=replacement_spec.digest,
            task_scope_digest=replacement_spec.authority.digest,
            attempt_count=0,
            fencing_epoch=0,
            created_at=now,
            updated_at=now,
            terminal_at=None,
            supersedes_task_id=task_id,
            superseded_by_task_id=None,
            latest_result_id=None,
            latest_control_id=None,
            latest_checkpoint_id=None,
        )
        mutation_id = _stable_graph_id(
            "mutation",
            job_id=job_id,
            actor_key=actor_key,
            idempotency_key=idempotency_key,
            discriminator="policy-replacement",
        )
        dependencies = tuple(
            replace(
                edge,
                downstream_task_id=replacement_id,
                created_at=now,
                creator_key=actor_key,
                mutation_id=mutation_id,
            )
            for edge in inspection.dependencies
            if edge.downstream_task_id == task_id
        )
        request = GraphMutationRequest(
            agent_id=self.agent_id,
            job_id=job_id,
            mutation_id=mutation_id,
            actor_kind="human_policy",
            actor_key=actor_key,
            idempotency_key=idempotency_key,
            expected_revision=expected_revision,
            created_at=now,
            tasks=(replacement,),
            dependencies=dependencies,
            supersessions=((task_id, replacement_id),),
        )
        return await self.mutate_graph(request)

    async def graph_events(
        self,
        job_id: str,
        *,
        after_event_id: int = 0,
        limit: int = 100,
        task_id: str | None = None,
    ) -> GraphEventPage:
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("job_id must be non-empty text")
        return await self._graph_store().list_graph_events(
            self.agent_id,
            job_id,
            after_event_id=after_event_id,
            limit=limit,
            task_id=task_id,
        )

    async def checkpoint_graph_task(
        self, checkpoint: TaskCheckpoint, *, claim_token: str
    ) -> TaskCheckpoint:
        self._require_graph_record_owner(checkpoint.agent_id)
        return await self._graph_store().checkpoint_graph_attempt(
            checkpoint, claim_token=claim_token
        )

    async def comment_graph_task(
        self,
        comment: TaskComment,
        *,
        attempt_id: str,
        claim_token: str,
        fencing_epoch: int,
    ) -> TaskComment:
        self._require_graph_record_owner(comment.agent_id)
        return await self._graph_store().add_graph_comment(
            comment,
            attempt_id=attempt_id,
            claim_token=claim_token,
            fencing_epoch=fencing_epoch,
        )

    async def complete_graph_task(
        self,
        result: TaskResult,
        *,
        claim_token: str,
        fencing_epoch: int,
        usage: tuple[BudgetAmount, ...] | None,
    ) -> TaskResult:
        self._require_graph_record_owner(result.agent_id)
        stored = await self._graph_store().complete_graph_attempt(
            result,
            claim_token=claim_token,
            fencing_epoch=fencing_epoch,
            usage=usage,
        )
        self._notify(result.job_id)
        return stored

    async def open_graph_task_control(
        self,
        control: TaskControl,
        *,
        claim_token: str,
        fencing_epoch: int,
    ) -> TaskControl:
        self._require_graph_record_owner(control.agent_id)
        replan_task = None
        reviewer_task = None
        inspection: GraphInspection | None = None
        if control.kind.value == "needs_replan":
            inspection = await self.inspect_graph(control.job_id)
            if inspection is None:
                raise JobError("unknown_graph", "The graph job is unavailable.")
            replan_task = planner_task_from_template(
                inspection,
                task_id=self._id_factory("task"),
                created_at=control.created_at,
            )
        elif control.kind is ControlKind.REVIEW_REQUESTED:
            inspection = await self.inspect_graph(control.job_id)
            if inspection is None:
                raise JobError("unknown_graph", "The graph job is unavailable.")
            subject = next(
                (item for item in inspection.tasks if item.task_id == control.task_id),
                None,
            )
            if subject is None:
                raise JobError(
                    "unknown_graph_task", "The review subject is unavailable."
                )
            reviewer_task = _reviewer_task_from_control(
                inspection,
                subject,
                control,
                task_id=_stable_graph_id(
                    "task",
                    job_id=control.job_id,
                    actor_key="review_owner",
                    idempotency_key=control.control_id,
                    discriminator="reviewer",
                ),
            )
        stored = await self._graph_store().open_graph_control(
            control,
            claim_token=claim_token,
            fencing_epoch=fencing_epoch,
            replan_task=replan_task,
            reviewer_task=reviewer_task,
        )
        self._notify(control.job_id)
        return stored

    async def accept_graph_task_review(
        self,
        job_id: str,
        subject_task_id: str,
        control_id: str,
        *,
        reviewer_task_id: str,
        rationale: str,
        idempotency_key: str,
        resolved_at: datetime | None = None,
        resolved_by_kind: str,
        resolved_by_id: str,
        reviewer_result: TaskResult | None = None,
        claim_token: str | None = None,
        fencing_epoch: int | None = None,
    ) -> TaskResult:
        _inspection, control, subject = await self._review_control(
            job_id, subject_task_id, control_id
        )
        if not rationale.strip() or len(rationale.encode("utf-8")) > 4096:
            raise ValueError("review rationale is outside its bound")
        if not idempotency_key:
            raise ValueError("review idempotency key must be non-empty")
        stored = await self._graph_store().accept_graph_review(
            agent_id=self.agent_id,
            job_id=job_id,
            subject_task_id=subject_task_id,
            control_id=control_id,
            reviewer_task_id=reviewer_task_id,
            resolved_at=resolved_at or self._clock(),
            resolved_by_kind=resolved_by_kind,
            resolved_by_id=resolved_by_id,
            rationale=rationale,
            idempotency_key=idempotency_key,
            expected_control_digest=control.payload_digest,
            expected_subject_revision=subject.task_revision,
            reviewer_result=reviewer_result,
            claim_token=claim_token,
            fencing_epoch=fencing_epoch,
        )
        self._notify(job_id)
        return stored

    async def request_graph_task_review_changes(
        self,
        job_id: str,
        subject_task_id: str,
        control_id: str,
        *,
        reviewer_task_id: str,
        rationale: str,
        replacement_guidance: str,
        idempotency_key: str,
        changes_control_id: str,
        resolved_at: datetime | None = None,
        resolved_by_kind: str,
        resolved_by_id: str,
        reviewer_result: TaskResult | None = None,
        claim_token: str | None = None,
        fencing_epoch: int | None = None,
    ) -> TaskControl:
        inspection, control, subject = await self._review_control(
            job_id, subject_task_id, control_id
        )
        for value, label in (
            (rationale, "review rationale"),
            (replacement_guidance, "replacement guidance"),
            (idempotency_key, "review idempotency key"),
            (changes_control_id, "changes control id"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be non-empty text")
        if (
            len(rationale.encode("utf-8")) > 4096
            or len(replacement_guidance.encode("utf-8")) > 4096
        ):
            raise ValueError("review change guidance is outside its bound")
        changed_at = resolved_at or self._clock()
        payload = {
            "message": "A validated policy replacement is required.",
            "review_control_id": control.control_id,
            "candidate_digest": control.payload["candidate_digest"],
            "rationale": rationale,
            "replacement_guidance": replacement_guidance,
            "expires_at": inspection.job.deadline_at.isoformat(),
            "default_behavior": "remain_blocked",
        }
        changes = TaskControl(
            agent_id=self.agent_id,
            job_id=job_id,
            task_id=subject_task_id,
            control_id=changes_control_id,
            kind=ControlKind.CHANGES_REQUESTED,
            state=ControlState.OPEN,
            requesting_attempt_id=control.requesting_attempt_id,
            payload=payload,
            created_at=changed_at,
            payload_digest=canonical_digest(payload),
        )
        stored = await self._graph_store().request_graph_review_changes(
            changes_control=changes,
            review_control_id=control_id,
            reviewer_task_id=reviewer_task_id,
            resolved_at=changed_at,
            resolved_by_kind=resolved_by_kind,
            resolved_by_id=resolved_by_id,
            rationale=rationale,
            idempotency_key=idempotency_key,
            expected_control_digest=control.payload_digest,
            expected_subject_revision=subject.task_revision,
            reviewer_result=reviewer_result,
            claim_token=claim_token,
            fencing_epoch=fencing_epoch,
        )
        self._notify(job_id)
        return stored

    async def _review_control(
        self, job_id: str, task_id: str, control_id: str
    ) -> tuple[GraphInspection, TaskControl, GraphTask]:
        inspection = await self.inspect_graph(job_id)
        if inspection is None:
            raise JobError("unknown_graph", "The graph job is unavailable.")
        control = next(
            (
                item
                for item in inspection.controls
                if item.task_id == task_id and item.control_id == control_id
            ),
            None,
        )
        task = next(
            (item for item in inspection.tasks if item.task_id == task_id), None
        )
        if (
            control is None
            or task is None
            or control.kind is not ControlKind.REVIEW_REQUESTED
        ):
            raise JobError(
                "review_control_unavailable", "The exact review control is unavailable."
            )
        return inspection, control, task

    async def accept_graph_task_review_by_principal(
        self,
        job_id: str,
        task_id: str,
        control_id: str,
        *,
        principal_id: str,
        rationale: str,
        idempotency_key: str,
    ) -> TaskResult:
        inspection, _control, _subject = await self._review_control(
            job_id, task_id, control_id
        )
        self._require_graph_principal(inspection, principal_id)
        reviewer = _reviewer_for_control(inspection, control_id)
        return await self.accept_graph_task_review(
            job_id,
            task_id,
            control_id,
            reviewer_task_id=reviewer.task_id,
            rationale=rationale,
            idempotency_key=idempotency_key,
            resolved_by_kind="principal",
            resolved_by_id=principal_id,
        )

    async def request_graph_task_review_changes_by_principal(
        self,
        job_id: str,
        task_id: str,
        control_id: str,
        *,
        principal_id: str,
        rationale: str,
        replacement_guidance: str,
        idempotency_key: str,
    ) -> TaskControl:
        inspection, control, _subject = await self._review_control(
            job_id, task_id, control_id
        )
        self._require_graph_principal(inspection, principal_id)
        reviewer = _reviewer_for_control(inspection, control_id)
        changes_control_id = _stable_graph_id(
            "control",
            job_id=job_id,
            actor_key=principal_id,
            idempotency_key=idempotency_key,
            discriminator=f"review-changes:{control.control_id}",
        )
        return await self.request_graph_task_review_changes(
            job_id,
            task_id,
            control_id,
            reviewer_task_id=reviewer.task_id,
            rationale=rationale,
            replacement_guidance=replacement_guidance,
            idempotency_key=idempotency_key,
            changes_control_id=changes_control_id,
            resolved_by_kind="principal",
            resolved_by_id=principal_id,
        )

    def _require_graph_record_owner(self, agent_id: str) -> None:
        if agent_id != self.agent_id:
            raise JobError("job_owner_mismatch", "The graph owner identity changed.")

    @staticmethod
    def _require_graph_principal(
        inspection: GraphInspection, principal_id: str
    ) -> None:
        if (
            not isinstance(principal_id, str)
            or not principal_id
            or inspection.job.specification.principal_id != principal_id
        ):
            raise JobError(
                "graph_principal_mismatch",
                "The graph principal identity differs from this command.",
            )

    def _notify(self, job_id: str | None) -> None:
        if self._wake is not None:
            self._wake(job_id)

    def _graph_store(self) -> GraphJobStore:
        for method in (
            "admit_graph",
            "admit_replacement_graph",
            "inspect_graph",
            "apply_graph_mutation",
            "list_graph_events",
            "checkpoint_graph_attempt",
            "add_graph_comment",
            "complete_graph_attempt",
            "open_graph_control",
            "accept_graph_review",
            "request_graph_review_changes",
            "resolve_graph_control",
            "request_graph_cancel",
        ):
            if not callable(getattr(self._store, method, None)):
                raise TypeError("current graph store is unavailable")
        return cast(GraphJobStore, self._store)


def _stable_graph_id(
    prefix: str,
    *,
    job_id: str,
    actor_key: str,
    idempotency_key: str,
    discriminator: str,
) -> str:
    digest = canonical_digest(
        {
            "job_id": job_id,
            "actor_key": actor_key,
            "idempotency_key": idempotency_key,
            "discriminator": discriminator,
        }
    )
    return f"{prefix}-{digest.removeprefix('sha256:')[:32]}"


def _reviewer_task_from_control(
    inspection: GraphInspection,
    subject: GraphTask,
    control: TaskControl,
    *,
    task_id: str,
) -> GraphTask:
    """Derive one review-only task from immutable root and candidate contracts."""

    from .graph.capabilities import REVIEW_CAPABILITY_IDS

    candidate = control.payload.get("candidate")
    candidate_digest = control.payload.get("candidate_digest")
    if not isinstance(candidate, Mapping) or not isinstance(candidate_digest, str):
        raise JobError("review_candidate_invalid", "The review candidate is malformed.")
    try:
        parsed = TaskResult.from_candidate_material(candidate)
    except (TypeError, ValueError) as error:
        raise JobError(
            "review_candidate_invalid", "The review candidate is malformed."
        ) from error
    if (
        parsed.result_digest != candidate_digest
        or parsed.task_id != subject.task_id
        or parsed.attempt_id != control.requesting_attempt_id
        or parsed.job_id != subject.job_id
        or parsed.agent_id != subject.agent_id
    ):
        raise JobError(
            "review_candidate_invalid", "The review candidate binding differs."
        )
    root = inspection.job.specification.authority
    material = root.contract_bindings
    raw_capabilities = material.get("capability_contracts")
    raw_routes = material.get("model_routes")
    if not isinstance(raw_capabilities, Mapping) or not isinstance(raw_routes, Mapping):
        raise JobError(
            "review_contract_unavailable", "The root review contracts are unavailable."
        )
    if any(
        not isinstance(raw_capabilities.get(item), str)
        for item in REVIEW_CAPABILITY_IDS
    ):
        raise JobError(
            "review_contract_unavailable", "The root review contracts are unavailable."
        )
    if not subject.specification.authority.model_route_ids:
        raise JobError(
            "review_route_unavailable", "The review model route is unavailable."
        )
    route_id = subject.specification.authority.model_route_ids[0]
    route_digest = raw_routes.get(route_id)
    if not isinstance(route_digest, str):
        raise JobError(
            "review_route_unavailable", "The review model route is unavailable."
        )
    bindings = ExecutionContractBindings(
        capability_contracts={
            item: str(raw_capabilities[item]) for item in REVIEW_CAPABILITY_IDS
        },
        model_routes={route_id: route_digest},
    )
    authority = GraphAuthority(
        capability_ids=tuple(sorted(REVIEW_CAPABILITY_IDS)),
        access_modes=("none",),
        operational_effects=("none",),
        model_route_ids=(route_id,),
        sensitivity=max(
            subject.specification.authority.sensitivity,
            parsed.sensitivity,
            key=lambda item: item.routing_rank,
        ),
        contract_bindings=bindings.material(),
    )
    contract = subject.specification.expected_result_contract
    spec = GraphTaskSpecification(
        title=f"Review candidate for {subject.specification.title}"[:512],
        description=(
            "Inspect only the immutable candidate bound to this review control. "
            "Accept it, request a policy replacement, or block with a typed control."
        ),
        expected_result_contract={
            "kind": "model_task",
            "result_kind": "graph.review_decision",
            "review_control_id": control.control_id,
            "subject_task_id": subject.task_id,
            "candidate_digest": candidate_digest,
            "model_route_id": route_id,
            "per_run_max_tokens": contract["per_run_max_tokens"],
            "per_run_max_cost_usd": contract["per_run_max_cost_usd"],
            "attempt_budgets": {"work_units": 1},
        },
        authority=authority,
        budgets=(BudgetAmount("work_units", 3),),
        max_steps=min(subject.specification.max_steps, 4),
        max_wall_time_seconds=subject.specification.max_wall_time_seconds,
        created_by="review_owner",
    )
    return GraphTask(
        agent_id=subject.agent_id,
        job_id=subject.job_id,
        task_id=task_id,
        state=TaskState.READY,
        role=TaskRole.REVIEWER,
        execution_kind=TaskExecutionKind.MODEL,
        priority=min(1_000, subject.priority + 1),
        not_before=None,
        current_attempt_id=None,
        task_revision=1,
        specification=spec,
        task_spec_digest=spec.digest,
        task_scope_digest=authority.digest,
        attempt_count=0,
        failure_streak=0,
        fencing_epoch=0,
        created_at=control.created_at,
        updated_at=control.created_at,
    )


def _reviewer_for_control(inspection: GraphInspection, control_id: str) -> GraphTask:
    matches = tuple(
        task
        for task in inspection.tasks
        if task.role is TaskRole.REVIEWER
        and task.specification.expected_result_contract.get("review_control_id")
        == control_id
    )
    if len(matches) != 1:
        raise JobError(
            "reviewer_task_unavailable",
            "The exact separate reviewer task is unavailable.",
        )
    return matches[0]


__all__ = [
    "GraphBlockerProjection",
    "GraphJobStore",
    "JobError",
    "JobOwner",
]
