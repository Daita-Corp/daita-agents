"""Fenced revalidation for unreleased graph task attempts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Protocol

from ...capabilities import CapabilityInputError, GraphTaskBinding
from .models import (
    ACTIVE_ATTEMPT_STATES,
    GraphDesiredState,
    GraphInspection,
    GraphState,
    TaskState,
)


class GraphAttemptStateReader(Protocol):
    async def inspect_graph(
        self, agent_id: str, job_id: str
    ) -> GraphInspection | None: ...


@dataclass(frozen=True, slots=True)
class SQLiteTaskAttemptGuard:
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

    async def revalidate(self, *, capability_id: str, point: str) -> None:
        if not isinstance(capability_id, str) or not capability_id:
            raise ValueError("attempt guard capability_id must be non-empty")
        if not isinstance(point, str) or not point:
            raise ValueError("attempt guard point must be non-empty")
        inspection = await self.store.inspect_graph(
            self.binding.agent_id, self.binding.job_id
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
        task = next(
            (item for item in inspection.tasks if item.task_id == self.binding.task_id),
            None,
        )
        attempt = next(
            (
                item
                for item in inspection.attempts
                if item.task_id == self.binding.task_id
                and item.attempt_id == self.binding.attempt_id
            ),
            None,
        )
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

    @staticmethod
    def _stale(point: str, cause: str) -> None:
        raise CapabilityInputError(
            "stale_task_attempt",
            "The graph task attempt is no longer current; no operation was performed.",
            {"guard_point": point, "cause": cause},
        )


__all__ = ["SQLiteTaskAttemptGuard"]
