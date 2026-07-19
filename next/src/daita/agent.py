"""Thin public facade over the embedded agent composition."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Self

from .adapters.models import SourceRegistration
from .adapters.protocols import ResourceSource
from .capabilities import CapabilityRegistry
from .hosting.embedded import (
    AgentAlreadyExistsError,
    AgentHomeError,
    AgentIdentityMismatchError,
    AgentNameError,
    AgentNotConfiguredError,
    AgentNotFoundError,
    EmbeddedAgent,
    HostActiveError,
)
from .llm.protocols import ModelProvider
from .loop.driver import ContextBuilder, DomainController
from .loop.models import LoopBudgets, LoopExit
from .operations.checkpoints import OperationSnapshot
from .operations.governance import DefaultPolicyEvaluator
from .sessions import SessionTranscript


class Agent:
    """Persistent identity with a deliberately thin local API."""

    def __init__(self, embedded: EmbeddedAgent) -> None:
        self._embedded = embedded

    @classmethod
    async def create(
        cls,
        name: str,
        *,
        root: str | Path | None = None,
        model: ModelProvider | None = None,
        context_builder: ContextBuilder | None = None,
        domain: DomainController | None = None,
        capabilities: CapabilityRegistry | None = None,
        policy: DefaultPolicyEvaluator | None = None,
        budgets: LoopBudgets = LoopBudgets(),
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[str], str] | None = None,
    ) -> Self:
        embedded = await EmbeddedAgent.create(
            name,
            root=root,
            model=model,
            context_builder=context_builder,
            domain=domain,
            capabilities=capabilities,
            policy=policy,
            budgets=budgets,
            clock=clock,
            id_factory=id_factory,
        )
        return cls(embedded)

    @classmethod
    async def open(
        cls,
        name: str,
        *,
        root: str | Path | None = None,
        model: ModelProvider | None = None,
        context_builder: ContextBuilder | None = None,
        domain: DomainController | None = None,
        capabilities: CapabilityRegistry | None = None,
        policy: DefaultPolicyEvaluator | None = None,
        budgets: LoopBudgets = LoopBudgets(),
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[str], str] | None = None,
    ) -> Self:
        embedded = await EmbeddedAgent.open(
            name,
            root=root,
            model=model,
            context_builder=context_builder,
            domain=domain,
            capabilities=capabilities,
            policy=policy,
            budgets=budgets,
            clock=clock,
            id_factory=id_factory,
        )
        return cls(embedded)

    @property
    def id(self) -> str:
        return self._embedded.identity.id

    @property
    def name(self) -> str:
        return self._embedded.identity.display_name

    @property
    def home(self) -> Path:
        return self._embedded.home

    async def run(self, message: str, *, session_id: str | None = None) -> LoopExit:
        return await self._embedded.run(message, session_id=session_id)

    async def attach(self, source: ResourceSource) -> SourceRegistration:
        return await self._embedded.attach(source)

    async def inspect(self, operation_id: str) -> OperationSnapshot:
        return await self._embedded.inspect(operation_id)

    async def resume(self, operation_id: str) -> LoopExit:
        return await self._embedded.resume(operation_id)

    async def transcript(self, session_id: str) -> SessionTranscript:
        return await self._embedded.transcript(session_id)

    async def close(self) -> None:
        await self._embedded.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()


__all__ = [
    "Agent",
    "AgentAlreadyExistsError",
    "AgentHomeError",
    "AgentIdentityMismatchError",
    "AgentNameError",
    "AgentNotConfiguredError",
    "AgentNotFoundError",
    "HostActiveError",
]
