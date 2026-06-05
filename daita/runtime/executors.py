"""Shared executor adapters for runtime extension declarations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import inspect
from typing import Any

from .primitives import Evidence, Operation, Task


@dataclass(frozen=True)
class EvidenceWrappingExecutor:
    """Call a handler and wrap payload results as runtime evidence."""

    id: str
    owner: str
    capability_ids: frozenset[str]
    evidence_kind: str
    handler: Callable[[Mapping[str, Any]], Mapping[str, Any] | Evidence | Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        result = self.handler(task.input)
        if inspect.isawaitable(result):
            result = await result
        return [
            self._correlate_evidence(evidence, operation=operation, task=task)
            for evidence in _evidence_records(
                result,
                owner=self.owner,
                evidence_kind=self.evidence_kind,
                metadata=self.metadata,
            )
        ]

    def _correlate_evidence(
        self,
        evidence: Evidence,
        *,
        operation: Operation,
        task: Task,
    ) -> Evidence:
        from dataclasses import replace

        return replace(
            evidence,
            owner=evidence.owner or self.owner,
            operation_id=evidence.operation_id or operation.id,
            task_id=evidence.task_id or task.id,
            metadata={**dict(self.metadata), **evidence.metadata},
        )


def _evidence_records(
    result: Any,
    *,
    owner: str,
    evidence_kind: str,
    metadata: Mapping[str, Any],
) -> tuple[Evidence, ...]:
    if isinstance(result, Evidence):
        return (result,)
    if isinstance(result, (list, tuple)) and all(
        isinstance(item, Evidence) for item in result
    ):
        return tuple(result)
    if isinstance(result, Mapping):
        payload = dict(result)
    else:
        payload = {"result": result}
    return (
        Evidence(
            kind=evidence_kind,
            owner=owner,
            payload=payload,
            metadata=dict(metadata),
        ),
    )
