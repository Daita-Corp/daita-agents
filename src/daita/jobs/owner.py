"""Admit durable jobs and own their lifecycle transitions and public projections."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Protocol, cast

from ..errors import DaitaError, ErrorRetryability
from .graph.models import (
    BudgetAmount,
    ControlState,
    GraphAdmission,
    GraphEventPage,
    GraphInspection,
    GraphJob,
    GraphMutation,
    GraphMutationRequest,
    GraphState,
    GraphTask,
    TaskCheckpoint,
    TaskComment,
    TaskControl,
    TaskResult,
    TaskState,
    canonical_digest,
)
from .graph.planning import planner_task_from_template


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
            "inspect_graph",
            "list_graph_jobs",
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
            },
        )
        return await self.admit(replace(admission, job=job))

    async def inspect_graph(self, job_id: str) -> GraphInspection | None:
        return await self.inspect(job_id)

    async def mutate_graph(self, request: GraphMutationRequest) -> GraphMutation:
        self._require_graph_record_owner(request.agent_id)
        mutation = await self._graph_store().apply_graph_mutation(request)
        self._notify(request.job_id)
        return mutation

    async def graph_blockers(self, job_id: str) -> GraphBlockerProjection | None:
        inspection = await self.inspect_graph(job_id)
        if inspection is None:
            return None
        open_controls: tuple[Mapping[str, object], ...] = tuple(
            {
                "control_id": control.control_id,
                "task_id": control.task_id,
                "kind": control.kind.value,
                "created_at": control.created_at.isoformat(),
                "payload_digest": control.payload_digest,
                "payload": control.payload,
            }
            for control in inspection.controls
            if control.state.value == "open"
        )[:32]
        failed: tuple[Mapping[str, object], ...] = tuple(
            {
                "task_id": task.task_id,
                "kind": "failed_task",
                "failure_streak": task.failure_streak,
            }
            for task in inspection.tasks
            if task.state is TaskState.FAILED
        )[:32]
        return GraphBlockerProjection(
            job_id=job_id,
            graph_state=inspection.job.state,
            blockers=(*open_controls, *failed)[:32],
        )

    async def answer_graph_task_input(
        self,
        job_id: str,
        task_id: str,
        control_id: str,
        *,
        principal_id: str,
        answer: Mapping[str, object],
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
        if (
            control is None
            or control.state is not ControlState.OPEN
            or control.kind.value != "needs_input"
        ):
            raise JobError(
                "task_input_unavailable",
                "The exact open human-input control is unavailable.",
            )
        assert inspection is not None
        self._require_graph_principal(inspection, principal_id)
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
        resolved = await self._graph_store().resolve_graph_control(
            self.agent_id,
            job_id,
            task_id,
            control_id,
            state=ControlState.RESOLVED,
            resolved_at=self._clock(),
            resolved_by_kind="principal",
            resolved_by_id=principal_id,
            resolution={"answer": dict(answer)},
            make_ready=True,
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
    ) -> TaskControl | None:
        inspection = await self.inspect_graph(job_id)
        if inspection is None:
            raise JobError("unknown_graph", "The graph job is unavailable.")
        self._require_graph_principal(inspection, principal_id)
        resolved = await self._graph_store().resolve_graph_control(
            self.agent_id,
            job_id,
            task_id,
            control_id,
            state=ControlState.REJECTED,
            resolved_at=self._clock(),
            resolved_by_kind="principal",
            resolved_by_id=principal_id,
            resolution={"reason": reason},
            make_ready=False,
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
            task.state not in {TaskState.BLOCKED, TaskState.REVIEW}
            and not retrying_committed
        ):
            raise JobError(
                "replacement_not_allowed",
                "Only blocked or review-waiting work has a policy replacement.",
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
    ) -> GraphEventPage:
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("job_id must be non-empty text")
        return await self._graph_store().list_graph_events(
            self.agent_id,
            job_id,
            after_event_id=after_event_id,
            limit=limit,
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
        if control.kind.value == "needs_replan":
            inspection = await self.inspect_graph(control.job_id)
            if inspection is None:
                raise JobError("unknown_graph", "The graph job is unavailable.")
            replan_task = planner_task_from_template(
                inspection,
                task_id=self._id_factory("task"),
                created_at=control.created_at,
            )
        stored = await self._graph_store().open_graph_control(
            control,
            claim_token=claim_token,
            fencing_epoch=fencing_epoch,
            replan_task=replan_task,
        )
        self._notify(control.job_id)
        return stored

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
            "inspect_graph",
            "apply_graph_mutation",
            "list_graph_events",
            "checkpoint_graph_attempt",
            "add_graph_comment",
            "complete_graph_attempt",
            "open_graph_control",
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


__all__ = [
    "GraphJobStore",
    "GraphBlockerProjection",
    "JobError",
    "JobOwner",
]
