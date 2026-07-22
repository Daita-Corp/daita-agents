"""Focused public API for the MVP data agent."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Self

from ._json import FrozenJsonObject
from .adapters.models import SourceRegistration
from .adapters.protocols import ResourceSource
from .catalog.models import CatalogResource, CatalogSearchRequest, CatalogSearchResult
from .capabilities import ApprovalHandler
from .config import AgentConfig
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
from .llm.models import ModelProfile
from .llm.protocols import ModelProvider
from .llm.routing import ModelRoute
from .loop.driver import ContextBuilder, ToolRuntime
from .loop.models import ConversationRun, LoopExit, LoopLimits, Transcript
from .observation import AgentObserver
from .security import SecretProvider
from .skills import Skill, SkillSummary


class Agent:
    """Persistent identity with one transcript-driven run path."""

    def __init__(self, embedded: EmbeddedAgent) -> None:
        self._embedded = embedded

    @classmethod
    async def create(
        cls,
        name: str,
        *,
        root: str | Path | None = None,
        config: AgentConfig | None = None,
        model: ModelProvider | None = None,
        model_profile: ModelProfile | None = None,
        context_builder: ContextBuilder | None = None,
        tools: ToolRuntime | None = None,
        limits: LoopLimits | None = None,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[str], str] | None = None,
        secret_provider: SecretProvider | None = None,
        observer: AgentObserver | None = None,
        approval_handler: ApprovalHandler | None = None,
    ) -> Self:
        return cls(
            await EmbeddedAgent.create(
                name,
                root=root,
                config=config,
                model=model,
                model_profile=model_profile,
                context_builder=context_builder,
                tools=tools,
                limits=limits,
                clock=clock,
                id_factory=id_factory,
                secret_provider=secret_provider,
                observer=observer,
                approval_handler=approval_handler,
            )
        )

    @classmethod
    async def open(
        cls,
        name: str,
        *,
        root: str | Path | None = None,
        config: AgentConfig | None = None,
        model: ModelProvider | None = None,
        model_profile: ModelProfile | None = None,
        context_builder: ContextBuilder | None = None,
        tools: ToolRuntime | None = None,
        limits: LoopLimits | None = None,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[str], str] | None = None,
        secret_provider: SecretProvider | None = None,
        observer: AgentObserver | None = None,
        approval_handler: ApprovalHandler | None = None,
    ) -> Self:
        return cls(
            await EmbeddedAgent.open(
                name,
                root=root,
                config=config,
                model=model,
                model_profile=model_profile,
                context_builder=context_builder,
                tools=tools,
                limits=limits,
                clock=clock,
                id_factory=id_factory,
                secret_provider=secret_provider,
                observer=observer,
                approval_handler=approval_handler,
            )
        )

    @property
    def id(self) -> str:
        return self._embedded.identity.id

    @property
    def name(self) -> str:
        return self._embedded.identity.display_name

    @property
    def home(self) -> Path:
        return self._embedded.home

    @property
    def model_profile(self) -> ModelProfile | None:
        return self._embedded.model_profile

    @property
    def model_route(self) -> ModelRoute | None:
        return self._embedded.model_route

    async def run(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
    ) -> LoopExit:
        return await self._embedded.run(
            message,
            conversation_id=conversation_id,
        )

    async def transcript(self, run_id: str) -> Transcript:
        return await self._embedded.transcript(run_id)

    async def conversation_runs(
        self,
        conversation_id: str,
    ) -> tuple[ConversationRun, ...]:
        return await self._embedded.conversation_runs(conversation_id)

    async def read_memory(self) -> str:
        return await self._embedded.read_memory()

    async def set_memory(self, text: str) -> None:
        await self._embedded.set_memory(text)

    async def read_user_profile(self) -> str:
        return await self._embedded.read_user_profile()

    async def set_user_profile(self, text: str) -> None:
        await self._embedded.set_user_profile(text)

    async def list_skills(self) -> tuple[SkillSummary, ...]:
        return await self._embedded.list_skills()

    async def read_skill(self, name: str) -> Skill | None:
        return await self._embedded.read_skill(name)

    async def save_skill(
        self,
        name: str,
        description: str,
        instructions: str,
    ) -> bool:
        return await self._embedded.save_skill(name, description, instructions)

    async def delete_skill(self, name: str) -> bool:
        return await self._embedded.delete_skill(name)

    async def attach(self, source: ResourceSource) -> SourceRegistration:
        return await self._embedded.attach(source)

    async def detach(self, source_id: str) -> SourceRegistration:
        return await self._embedded.detach(source_id)

    async def list_sources(self) -> tuple[SourceRegistration, ...]:
        return await self._embedded.list_sources()

    async def list_catalog_resources(
        self, *, source_id: str | None = None
    ) -> tuple[CatalogResource, ...]:
        return await self._embedded.list_catalog_resources(source_id=source_id)

    async def search_catalog(
        self, request: CatalogSearchRequest
    ) -> CatalogSearchResult:
        return await self._embedded.search_catalog(request)

    async def inspect_catalog_resource(self, resource_id: str) -> FrozenJsonObject:
        return await self._embedded.inspect_catalog_resource(resource_id)

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
