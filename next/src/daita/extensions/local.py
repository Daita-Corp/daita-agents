"""Explicit local functions projected through the capability boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import inspect
from typing import overload

from ..capabilities import (
    AccessMode,
    Capability,
    EvidenceCandidate,
    ExecutionRequest,
    Executor,
    ExtensionDeclarations,
    RiskLevel,
    ToolView,
)

LocalHandler = Callable[[Mapping[str, object]], object]


class _LocalExecutor:
    def __init__(
        self,
        *,
        executor_id: str,
        evidence_kind: str,
        schema_version: int,
        handler: LocalHandler,
    ) -> None:
        self.executor_id = executor_id
        self._evidence_kind = evidence_kind
        self._schema_version = schema_version
        self._handler = handler

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        result = self._handler(request.arguments)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, Mapping):
            raise TypeError("local capability result must be a mapping")
        return EvidenceCandidate(
            kind=self._evidence_kind,
            schema_version=self._schema_version,
            payload=result,
        )


@dataclass(frozen=True, slots=True)
class LocalCapability:
    """One explicit capability/executor/view bundle produced by :func:`tool`."""

    capability: Capability
    executor: Executor
    tool_view: ToolView

    def declarations(self) -> ExtensionDeclarations:
        return ExtensionDeclarations(
            capabilities=(self.capability,),
            executor_ids=(self.executor.executor_id,),
            tool_views=(self.tool_view,),
        )


@overload
def tool(
    func: None = None,
    *,
    id: str,
    owner: str,
    name: str,
    description: str,
    input_schema: Mapping[str, object],
    output_schema: Mapping[str, object],
    executor_id: str | None = None,
    output_evidence_kind: str | None = None,
    output_schema_version: int = 1,
    access_mode: AccessMode = AccessMode.READ,
    risk: RiskLevel = RiskLevel.LOW,
    side_effecting: bool = False,
    idempotent: bool = True,
    replay_safe: bool = True,
    required_evidence_kinds: tuple[str, ...] = (),
) -> Callable[[LocalHandler], LocalCapability]: ...


@overload
def tool(
    func: LocalHandler,
    *,
    id: str,
    owner: str,
    name: str,
    description: str,
    input_schema: Mapping[str, object],
    output_schema: Mapping[str, object],
    executor_id: str | None = None,
    output_evidence_kind: str | None = None,
    output_schema_version: int = 1,
    access_mode: AccessMode = AccessMode.READ,
    risk: RiskLevel = RiskLevel.LOW,
    side_effecting: bool = False,
    idempotent: bool = True,
    replay_safe: bool = True,
    required_evidence_kinds: tuple[str, ...] = (),
) -> LocalCapability: ...


def tool(
    func: LocalHandler | None = None,
    *,
    id: str,
    owner: str,
    name: str,
    description: str,
    input_schema: Mapping[str, object],
    output_schema: Mapping[str, object],
    executor_id: str | None = None,
    output_evidence_kind: str | None = None,
    output_schema_version: int = 1,
    access_mode: AccessMode = AccessMode.READ,
    risk: RiskLevel = RiskLevel.LOW,
    side_effecting: bool = False,
    idempotent: bool = True,
    replay_safe: bool = True,
    required_evidence_kinds: tuple[str, ...] = (),
) -> LocalCapability | Callable[[LocalHandler], LocalCapability]:
    """Declare a local handler without making it directly callable by the loop.

    Schemas and governance facts are intentionally explicit.  The returned
    object contributes a normal :class:`Capability`, :class:`Executor`, and
    :class:`ToolView`; only the operation runtime may invoke its executor.
    """

    resolved_executor_id = executor_id or f"{id}.executor"
    resolved_evidence_kind = output_evidence_kind or f"{id}.result"

    def declare(handler: LocalHandler) -> LocalCapability:
        if not callable(handler):
            raise TypeError("local capability handler must be callable")
        capability = Capability(
            id=id,
            owner=owner,
            description=description,
            input_schema=input_schema,
            output_evidence_kind=resolved_evidence_kind,
            output_schema_version=output_schema_version,
            output_schema=output_schema,
            executor_id=resolved_executor_id,
            access_mode=access_mode,
            risk=risk,
            side_effecting=side_effecting,
            idempotent=idempotent,
            replay_safe=replay_safe,
            required_evidence_kinds=required_evidence_kinds,
        )
        executor = _LocalExecutor(
            executor_id=resolved_executor_id,
            evidence_kind=resolved_evidence_kind,
            schema_version=output_schema_version,
            handler=handler,
        )
        return LocalCapability(
            capability=capability,
            executor=executor,
            tool_view=ToolView(
                name=name,
                capability_id=id,
                description=description,
            ),
        )

    if func is None:
        return declare
    return declare(func)


__all__ = ["LocalCapability", "tool"]
