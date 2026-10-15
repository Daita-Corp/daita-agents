"""Fenced revalidation for unreleased graph task attempts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import NoReturn, Protocol, cast

from ..._json import FrozenJsonObject
from ...capabilities import CapabilityInputError, GraphTaskBinding
from .models import (
    ACTIVE_ATTEMPT_STATES,
    GraphDesiredState,
    GraphInspection,
    GraphJob,
    GraphState,
    GraphTask,
    TaskAttempt,
    TaskExecutionKind,
    TaskState,
)


@dataclass(frozen=True, slots=True)
class GraphAttemptAuthority:
    """One consistent authority snapshot, without graph history or payloads."""

    job: GraphJob
    task: GraphTask | None
    attempt: TaskAttempt | None


class GraphAttemptStateReader(Protocol):
    async def read_graph_attempt_authority(
        self, agent_id: str, job_id: str, task_id: str, attempt_id: str
    ) -> GraphAttemptAuthority | None: ...
    async def inspect_graph(
        self, agent_id: str, job_id: str
    ) -> GraphInspection | None: ...


@dataclass(frozen=True, slots=True)
class TaskAttemptStateGuard:
    """Read current durable authority before each meaningful attempt operation."""

    store: GraphAttemptStateReader
    binding: GraphTaskBinding
    claim_token: str
    run_id: str
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)

    def __post_init__(self) -> None:
        if not isinstance(self.binding, GraphTaskBinding):
            raise TypeError("attempt guard requires GraphTaskBinding")
        if not isinstance(self.claim_token, str) or not self.claim_token:
            raise ValueError("attempt guard claim token must be non-empty")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("attempt guard run_id must be non-empty")
        digest = "sha256:" + sha256(self.claim_token.encode("utf-8")).hexdigest()
        if digest != self.binding.claim_token_digest:
            raise ValueError("attempt guard claim token differs from its binding")

    async def revalidate(
        self,
        *,
        capability_id: str,
        point: str,
        arguments: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(capability_id, str) or not capability_id:
            raise ValueError("attempt guard capability_id must be non-empty")
        if not isinstance(point, str) or not point:
            raise ValueError("attempt guard point must be non-empty")
        inspection = await self.store.read_graph_attempt_authority(
            self.binding.agent_id,
            self.binding.job_id,
            self.binding.task_id,
            self.binding.attempt_id,
        )
        now = self.clock()
        if inspection is None:
            self._stale(point, "graph_missing")
        assert inspection is not None
        job = inspection.job
        if (
            job.specification.authority.digest != self.binding.root_authority_digest
            or job.state not in {GraphState.QUEUED, GraphState.ACTIVE}
            or job.desired_state is not GraphDesiredState.RUN
            or job.deadline_at <= now
        ):
            self._stale(point, "job_inactive")
        task, attempt = inspection.task, inspection.attempt
        if task is None or attempt is None:
            self._stale(point, "attempt_missing")
        assert task is not None and attempt is not None
        claim_digest = (
            "sha256:" + sha256(attempt.claim_token.encode("utf-8")).hexdigest()
        )
        if (
            task.state is not TaskState.RUNNING
            or task.current_attempt_id != self.binding.attempt_id
            or task.task_revision < self.binding.task_revision
            or task.task_spec_digest != self.binding.task_spec_digest
            or task.task_scope_digest != self.binding.task_scope_digest
            or task.role.value != self.binding.task_role
            or task.fencing_epoch != self.binding.fencing_epoch
            or attempt.state not in ACTIVE_ATTEMPT_STATES
            or attempt.run_id != self.run_id
            or attempt.fencing_epoch != self.binding.fencing_epoch
            or claim_digest != self.binding.claim_token_digest
            or attempt.absolute_deadline_at != self.binding.task_deadline_at
            or attempt.absolute_deadline_at <= now
            or (
                attempt.lease_expires_at is not None and attempt.lease_expires_at <= now
            )
        ):
            self._stale(point, "attempt_stale")
        if capability_id not in task.specification.authority.capability_ids:
            self._stale(point, "capability_revoked")
        if capability_id in {"data.count_batch", "data.finalize_count_comparison"}:
            from ...domains.data.row_counts import CountComparisonPlan
            from .execution import COUNT_BATCH_KIND, internal_task_contract

            contract = internal_task_contract(task)
            outcome = job.specification.outcome_contract
            plan = CountComparisonPlan.from_material(
                cast(Mapping[str, object], outcome["plan"])
            )
            if (
                arguments is None
                or task.execution_kind is not TaskExecutionKind.INTERNAL_CAPABILITY
                or contract["capability_id"] != capability_id
                or contract["plan_digest"] != plan.digest
            ):
                self._stale(point, "count_contract_changed")
            assert arguments is not None
            if contract["kind"] == COUNT_BATCH_KIND:
                mapping = next(
                    (
                        m
                        for m in cast(
                            tuple[Mapping[str, object], ...], outcome["batch_tasks"]
                        )
                        if m["task_id"] == task.task_id
                    ),
                    None,
                )
                batch = next(
                    (b for b in plan.batches if b["batch_id"] == contract["batch_id"]),
                    None,
                )
                if (
                    mapping is None
                    or batch is None
                    or mapping["task_spec_digest"] != task.task_spec_digest
                    or arguments != contract["arguments"]
                    or arguments["source_id"] != batch["source_id"]
                    or arguments["resource_ids"] != batch["resource_ids"]
                    or arguments["selection_binding"]
                    != next(
                        s
                        for s in plan.selections
                        if s["source_id"] == batch["source_id"]
                    )
                ):
                    self._stale(point, "count_arguments_changed")
            elif (
                arguments["plan"] != plan.material
                or arguments["batch_tasks"] != outcome["batch_tasks"]
                or arguments["job_id"] != job.job_id
                or arguments["agent_id"] != job.agent_id
            ):
                self._stale(point, "count_finalizer_changed")
            else:
                current = await self.store.inspect_graph(job.agent_id, job.job_id)
                if current is None:
                    self._stale(point, "graph_missing")
                by_task = {r.task_id: r for r in current.results}
                required_mapping = cast(
                    tuple[Mapping[str, object], ...], outcome["batch_tasks"]
                )
                if any(
                    m["task_id"] not in by_task for m in required_mapping
                ) or arguments["work_results"] != tuple(
                    FrozenJsonObject.from_mapping(
                        by_task[cast(str, m["task_id"])].candidate_material()
                    )
                    for m in required_mapping
                ):
                    self._stale(point, "count_results_changed")

    @staticmethod
    def _stale(point: str, cause: str) -> NoReturn:
        raise CapabilityInputError(
            "stale_task_attempt",
            "The graph task attempt is no longer current; no operation was performed.",
            {"guard_point": point, "cause": cause},
        )


SQLiteTaskAttemptGuard = TaskAttemptStateGuard

__all__ = ["TaskAttemptStateGuard", "SQLiteTaskAttemptGuard"]
