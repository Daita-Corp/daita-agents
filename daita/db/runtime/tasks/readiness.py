"""Task readiness helpers for ``DbRuntime``."""

from __future__ import annotations

from typing import Any

from daita.runtime import Operation, Task, TaskDependency

from .common import _evidence_payload_fingerprint, _payload_contains


class DbRuntimeTaskReadinessMixin:
    async def task_readiness(
        self,
        task: Task,
        operation: Operation,
    ) -> dict[str, Any]:
        """Return DB-owned dependency readiness for kernel task execution."""
        return await self._task_readiness(task, operation)

    async def _task_readiness(
        self,
        task: Task,
        operation: Operation,
    ) -> dict[str, Any]:
        unsatisfied: list[dict[str, Any]] = []
        for dependency in task.dependencies:
            if dependency.kind.value == "evidence":
                if not await self._evidence_dependency_satisfied(dependency, operation):
                    unsatisfied.append(dependency.to_dict())
            elif dependency.kind.value == "approval":
                if not await self._approval_dependency_satisfied(dependency, operation):
                    unsatisfied.append(dependency.to_dict())
        return {
            "ready": not unsatisfied,
            "unsatisfied_dependencies": unsatisfied,
            "dependency_count": len(task.dependencies),
        }

    async def _evidence_dependency_satisfied(
        self,
        dependency: TaskDependency,
        operation: Operation,
    ) -> bool:
        operation_id = dependency.operation_id or operation.id
        for evidence in await self.store.list_evidence(operation_id):
            if evidence.kind != dependency.evidence_kind:
                continue
            if (
                dependency.evidence_id is not None
                and evidence.id != dependency.evidence_id
            ):
                continue
            if (
                dependency.evidence_owner is not None
                and evidence.owner != dependency.evidence_owner
            ):
                continue
            if (
                dependency.producer_task_id is not None
                and evidence.task_id != dependency.producer_task_id
            ):
                continue
            if evidence.accepted is not dependency.evidence_accepted:
                continue
            if (
                dependency.input_hash is not None
                and evidence.metadata.get("task_input_hash") != dependency.input_hash
            ):
                continue
            if _payload_contains(evidence.payload, dependency.evidence_payload):
                if (
                    dependency.payload_fingerprint is not None
                    and dependency.payload_fingerprint
                    != _evidence_payload_fingerprint(evidence)
                ):
                    continue
                return True
        return False

    async def _approval_dependency_satisfied(
        self,
        dependency: TaskDependency,
        operation: Operation,
    ) -> bool:
        operation_id = dependency.operation_id or operation.id
        for approval in await self.store.list_approval_requests(operation_id):
            if (
                dependency.approval_id is not None
                and approval.approval_id != dependency.approval_id
            ):
                continue
            if (
                dependency.approval_policy_id is not None
                and approval.requested_by_policy_id != dependency.approval_policy_id
            ):
                continue
            if (
                dependency.approval_name is not None
                and approval.proposed_action.get("approval") != dependency.approval_name
            ):
                continue
            if (
                dependency.approval_version is not None
                and approval.metadata.get("version") != dependency.approval_version
            ):
                continue
            if approval.status is dependency.approval_status:
                return True
        return False
