"""Task construction and capability execution helpers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from uuid import uuid4

from daita.runtime import Capability, Evidence, Operation, Task, TaskDependency

from ..analysis import with_analysis_evidence_trace
from ..evidence import DbEvidenceStore
from ..models import DbOperationContract
from .helpers import _stable_hash


class _ExecutionTaskMixin:
    async def _persist_runtime_evidence(
        self,
        operation: Operation,
        evidence: Evidence,
    ) -> Evidence:
        evidence_id = evidence.id or f"evidence-{uuid4()}"
        persisted = replace(
            evidence,
            id=evidence_id,
            operation_id=evidence.operation_id or operation.id,
            metadata={
                **evidence.metadata,
                "payload_fingerprint": _stable_hash(evidence.payload),
            },
        )
        await self.runtime.store.save_evidence(persisted)
        return persisted

    async def _execute_capability(
        self,
        capability_id: str,
        contract: DbOperationContract,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        input: dict[str, Any],
        *,
        dependencies: tuple[TaskDependency, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Evidence, ...]:
        capability = _selected_capability(contract, capability_id)
        if capability is None:
            return ()
        resolved = self.runtime.registry.get_capability(
            capability["id"], owner=capability["owner"]
        )
        return await self._execute_direct_capability(
            resolved,
            operation,
            tasks,
            evidence_store,
            input,
            dependencies=dependencies,
            metadata=metadata,
        )

    async def _execute_direct_capability(
        self,
        capability: Capability,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        input: dict[str, Any],
        *,
        reason: str | None = None,
        dependencies: tuple[TaskDependency, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Evidence, ...]:
        planned = await self.runtime._planned_task_for_capability(
            operation.id,
            capability,
            metadata_match=metadata,
        )
        task_metadata = with_analysis_evidence_trace(
            {
                "owner": capability.owner,
                "reason": reason or "contract",
                **dict(metadata or {}),
            }
        )
        task = (
            Task(
                id=f"db-task-{uuid4()}",
                operation_id=operation.id,
                capability_id=capability.id,
                executor_id=capability.executor,
                input=input,
                required_evidence=capability.output_evidence,
                metadata=task_metadata,
                dependencies=dependencies,
            )
            if planned is None
            else replace(
                planned,
                input=input,
                dependencies=dependencies or planned.dependencies,
                metadata={
                    **with_analysis_evidence_trace(
                        {
                            **planned.metadata,
                            "owner": capability.owner,
                            "reason": reason
                            or planned.metadata.get("reason")
                            or "contract",
                            **dict(metadata or {}),
                        }
                    ),
                },
            )
        )
        tasks.append(task)
        evidence = await self.runtime.execute_task(
            task,
            operation,
            context={"capability_owner": capability.owner},
        )
        evidence_store.add_many(evidence)
        return evidence

    def _first_capability(self, capability_id: str) -> Capability | None:
        for capability in self.runtime.registry.capabilities:
            if capability.id == capability_id:
                return capability
        return None


def _selected_capability(
    contract: DbOperationContract, capability_id: str
) -> dict[str, str] | None:
    for item in contract.metadata.get("selected_capabilities", []):
        if item.get("id") == capability_id:
            return {"id": item["id"], "owner": item["owner"]}
    return None
