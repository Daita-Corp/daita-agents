"""Thin public facade over the embedded agent composition."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from datetime import datetime
from pathlib import Path
from typing import Self

from .adapters.models import SourceRegistration
from .adapters.protocols import ResourceSource
from ._json import FrozenJsonObject
from .catalog.models import CatalogResource, CatalogSearchRequest, CatalogSearchResult
from .capabilities import CapabilityRegistry
from .config import AgentConfig
from .events.models import CommittedEvent, EventCursor
from .events.projection import EventAudience, project_committed_event
from .extensions import ConfiguredExtension, ExtensionBinding
from .hosting.embedded import (
    AgentAlreadyExistsError,
    AgentHomeError,
    AgentIdentityMismatchError,
    AgentNameError,
    AgentNotConfiguredError,
    AgentNotFoundError,
    EmbeddedAgent,
    HostActiveError,
    SessionOperationActiveError,
)
from .llm.models import ModelProfile
from .llm.protocols import ModelProvider
from .llm.routing import ModelRoute
from .learning import LearningProposal, LearningProposalState
from .loop.driver import ContextBuilder, DomainController
from .loop.models import LoopBudgets, LoopExit
from .monitors.models import (
    Monitor,
    MonitorConfirmation,
    MonitorDefinition,
    MonitorInspection,
    MonitorProposal,
    MonitorStatus,
)
from .operations.checkpoints import OperationSnapshot
from .operations.governance import (
    ApprovalRequest,
    ApprovalStatus,
    DefaultPolicyEvaluator,
)
from .operations.models import OperationStatus
from .sessions import SessionTranscript
from .memory.learning import ExplicitCorrectionResult
from .memory.models import (
    MemoryInspection,
    MemoryInspectionRequest,
    MemoryListRequest,
    MemoryListResult,
    MemoryRecallRequest,
    MemoryRecallResult,
    MemoryRestoreRequest,
    MemorySupersessionRequest,
)
from .skills.models import SkillIndex, SkillInspection
from .skills.learning import (
    SkillChangeAcceptanceResult,
    SkillChangeCandidate,
    SkillChangeProposalResult,
)
from .security import SecretProvider


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
        config: AgentConfig | None = None,
        model: ModelProvider | None = None,
        model_profile: ModelProfile | None = None,
        context_builder: ContextBuilder | None = None,
        domain: DomainController | None = None,
        capabilities: CapabilityRegistry | None = None,
        policy: DefaultPolicyEvaluator | None = None,
        budgets: LoopBudgets | None = None,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[str], str] | None = None,
        secret_provider: SecretProvider | None = None,
        extensions: tuple[ConfiguredExtension, ...] = (),
    ) -> Self:
        embedded = await EmbeddedAgent.create(
            name,
            root=root,
            config=config,
            model=model,
            model_profile=model_profile,
            context_builder=context_builder,
            domain=domain,
            capabilities=capabilities,
            policy=policy,
            budgets=budgets,
            clock=clock,
            id_factory=id_factory,
            secret_provider=secret_provider,
            extensions=extensions,
        )
        return cls(embedded)

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
        domain: DomainController | None = None,
        capabilities: CapabilityRegistry | None = None,
        policy: DefaultPolicyEvaluator | None = None,
        budgets: LoopBudgets | None = None,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[str], str] | None = None,
        secret_provider: SecretProvider | None = None,
        extensions: tuple[ConfiguredExtension, ...] = (),
    ) -> Self:
        embedded = await EmbeddedAgent.open(
            name,
            root=root,
            config=config,
            model=model,
            model_profile=model_profile,
            context_builder=context_builder,
            domain=domain,
            capabilities=capabilities,
            policy=policy,
            budgets=budgets,
            clock=clock,
            id_factory=id_factory,
            secret_provider=secret_provider,
            extensions=extensions,
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

    @property
    def model_profile(self) -> ModelProfile | None:
        return self._embedded.model_profile

    @property
    def model_route(self) -> ModelRoute | None:
        return self._embedded.model_route

    @property
    def extension_bindings(self) -> tuple[ExtensionBinding, ...]:
        return self._embedded.extension_bindings

    async def run(self, message: str, *, session_id: str | None = None) -> LoopExit:
        return await self._embedded.run(message, session_id=session_id)

    def stream(
        self,
        message: str,
        *,
        session_id: str | None = None,
    ) -> AsyncGenerator[FrozenJsonObject, None]:
        return self._embedded.stream(message, session_id=session_id)

    async def attach(self, source: ResourceSource) -> SourceRegistration:
        return await self._embedded.attach(source)

    async def detach(self, source_id: str) -> SourceRegistration:
        return await self._embedded.detach(source_id)

    async def list_sources(self) -> tuple[SourceRegistration, ...]:
        return await self._embedded.list_sources()

    async def list_catalog_resources(
        self,
        *,
        source_id: str | None = None,
    ) -> tuple[CatalogResource, ...]:
        return await self._embedded.list_catalog_resources(source_id=source_id)

    async def search_catalog(
        self,
        request: CatalogSearchRequest,
    ) -> CatalogSearchResult:
        return await self._embedded.search_catalog(request)

    async def inspect_catalog_resource(self, resource_id: str) -> FrozenJsonObject:
        return await self._embedded.inspect_catalog_resource(resource_id)

    async def inspect(self, operation_id: str) -> OperationSnapshot:
        return await self._embedded.inspect(operation_id)

    async def list_operations(
        self,
        *,
        statuses: tuple[OperationStatus, ...] | None = None,
        limit: int = 100,
    ) -> tuple[OperationSnapshot, ...]:
        return await self._embedded.list_operations(statuses=statuses, limit=limit)

    async def inspect_approval(self, approval_id: str) -> ApprovalRequest:
        return await self._embedded.inspect_approval(approval_id)

    async def list_approvals(
        self,
        *,
        statuses: tuple[ApprovalStatus, ...] | None = None,
        limit: int = 100,
    ) -> tuple[ApprovalRequest, ...]:
        return await self._embedded.list_approvals(statuses=statuses, limit=limit)

    async def resume(self, operation_id: str) -> LoopExit:
        return await self._embedded.resume(operation_id)

    async def approve(
        self,
        approval_id: str,
        *,
        decided_by: str,
        reason: str,
    ) -> ApprovalRequest:
        return await self._embedded.decide_approval(
            approval_id,
            status=ApprovalStatus.APPROVED,
            decided_by=decided_by,
            reason=reason,
        )

    async def reject(
        self,
        approval_id: str,
        *,
        decided_by: str,
        reason: str,
    ) -> ApprovalRequest:
        return await self._embedded.decide_approval(
            approval_id,
            status=ApprovalStatus.DENIED,
            decided_by=decided_by,
            reason=reason,
        )

    async def cancel(
        self,
        operation_id: str,
        *,
        reason: str = "user_cancelled",
    ) -> LoopExit:
        return await self._embedded.interrupt(operation_id, reason)

    async def events(
        self,
        cursor: EventCursor | None = None,
        *,
        limit: int = 100,
    ) -> tuple[FrozenJsonObject, ...]:
        committed = await self._embedded.read_events(cursor, limit=limit)
        return tuple(
            project_committed_event(event, audience=EventAudience.PUBLIC)
            for event in committed
        )

    def subscribe_events(
        self,
        cursor: EventCursor | None = None,
    ) -> AsyncGenerator[FrozenJsonObject, None]:
        return _project_public_events(self._embedded.subscribe_events(cursor))

    async def propose_monitor(
        self,
        monitor_id: str,
        definition: MonitorDefinition,
        *,
        idempotency_key: str,
        source_operation_id: str | None = None,
    ) -> MonitorProposal:
        return await self._embedded.propose_monitor(
            monitor_id,
            definition,
            idempotency_key=idempotency_key,
            source_operation_id=source_operation_id,
        )

    async def propose_monitor_natural(
        self,
        monitor_id: str,
        request: str,
        *,
        idempotency_key: str,
        source_operation_id: str | None = None,
    ) -> MonitorProposal:
        return await self._embedded.propose_monitor_natural(
            monitor_id,
            request,
            idempotency_key=idempotency_key,
            source_operation_id=source_operation_id,
        )

    async def confirm_monitor(
        self,
        proposal_id: str,
        *,
        candidate_hash: str,
        actor_id: str,
        reason: str,
    ) -> MonitorInspection:
        return await self._embedded.confirm_monitor(
            proposal_id,
            candidate_hash=candidate_hash,
            actor_id=actor_id,
            reason=reason,
        )

    async def reject_monitor(
        self,
        proposal_id: str,
        *,
        candidate_hash: str,
        actor_id: str,
        reason: str,
    ) -> MonitorConfirmation:
        return await self._embedded.reject_monitor(
            proposal_id,
            candidate_hash=candidate_hash,
            actor_id=actor_id,
            reason=reason,
        )

    async def list_monitors(
        self,
        *,
        statuses: tuple[MonitorStatus, ...] | None = None,
        include_deleted: bool = False,
        limit: int = 100,
    ) -> tuple[Monitor, ...]:
        return await self._embedded.list_monitors(
            statuses=statuses,
            include_deleted=include_deleted,
            limit=limit,
        )

    async def list_monitor_proposals(
        self,
        *,
        limit: int = 100,
    ) -> tuple[MonitorProposal, ...]:
        return await self._embedded.list_monitor_proposals(limit=limit)

    async def inspect_monitor(self, monitor_id: str) -> MonitorInspection:
        return await self._embedded.inspect_monitor(monitor_id)

    async def pause_monitor(
        self,
        monitor_id: str,
        *,
        actor_id: str,
        reason: str,
        idempotency_key: str,
        operation_id: str | None = None,
    ) -> MonitorInspection:
        return await self._embedded.pause_monitor(
            monitor_id,
            actor_id=actor_id,
            reason=reason,
            idempotency_key=idempotency_key,
            operation_id=operation_id,
        )

    async def resume_monitor(
        self,
        monitor_id: str,
        *,
        actor_id: str,
        reason: str,
        idempotency_key: str,
        operation_id: str | None = None,
    ) -> MonitorInspection:
        return await self._embedded.resume_monitor(
            monitor_id,
            actor_id=actor_id,
            reason=reason,
            idempotency_key=idempotency_key,
            operation_id=operation_id,
        )

    async def delete_monitor(
        self,
        monitor_id: str,
        *,
        actor_id: str,
        reason: str,
        idempotency_key: str,
        operation_id: str | None = None,
    ) -> MonitorInspection:
        return await self._embedded.delete_monitor(
            monitor_id,
            actor_id=actor_id,
            reason=reason,
            idempotency_key=idempotency_key,
            operation_id=operation_id,
        )

    async def transcript(self, session_id: str) -> SessionTranscript:
        return await self._embedded.transcript(session_id)

    async def recall_memory(
        self,
        request: MemoryRecallRequest,
    ) -> MemoryRecallResult:
        return await self._embedded.recall_memory(request)

    async def list_memories(
        self,
        request: MemoryListRequest,
    ) -> MemoryListResult:
        return await self._embedded.list_memories(request)

    async def inspect_memory(
        self,
        request: MemoryInspectionRequest,
    ) -> MemoryInspection:
        return await self._embedded.inspect_memory(request)

    async def supersede_memory(
        self,
        request: MemorySupersessionRequest,
    ) -> MemoryInspection:
        return await self._embedded.supersede_memory(request)

    async def restore_memory(
        self,
        request: MemoryRestoreRequest,
    ) -> MemoryInspection:
        return await self._embedded.restore_memory(request)

    async def refresh_skills(self) -> tuple[SkillIndex, ...]:
        return await self._embedded.refresh_skills()

    async def list_skills(self) -> tuple[SkillIndex, ...]:
        return await self._embedded.list_skills()

    async def list_learning_proposals(
        self,
        *,
        operation_id: str | None = None,
        states: tuple[LearningProposalState, ...] = (
            LearningProposalState.PROPOSED,
            LearningProposalState.COMMITTED,
            LearningProposalState.REJECTED,
        ),
        limit: int = 100,
    ) -> tuple[LearningProposal, ...]:
        return await self._embedded.list_learning_proposals(
            operation_id=operation_id,
            states=states,
            limit=limit,
        )

    async def inspect_skill(self, skill_id: str) -> SkillInspection:
        return await self._embedded.inspect_skill(skill_id)

    async def activate_skill(
        self,
        skill_id: str,
        version_id: str,
        *,
        expected_active_version_id: str | None,
        actor_id: str,
        reason: str,
    ) -> SkillInspection:
        return await self._embedded.activate_skill(
            skill_id,
            version_id,
            expected_active_version_id=expected_active_version_id,
            actor_id=actor_id,
            reason=reason,
        )

    async def propose_skill_change(
        self,
        source_operation_id: str,
        candidate: SkillChangeCandidate,
    ) -> SkillChangeProposalResult:
        return await self._embedded.propose_skill_change(
            source_operation_id,
            candidate,
        )

    async def accept_skill_change(
        self,
        proposal_id: str,
        *,
        expected_active_version_id: str | None,
        actor_id: str,
        reason: str,
    ) -> SkillChangeAcceptanceResult:
        return await self._embedded.accept_skill_change(
            proposal_id,
            expected_active_version_id=expected_active_version_id,
            actor_id=actor_id,
            reason=reason,
        )

    async def learn_correction(
        self,
        operation_id: str,
    ) -> ExplicitCorrectionResult:
        return await self._embedded.learn_correction(operation_id)

    async def close(self) -> None:
        await self._embedded.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()


async def _project_public_events(
    committed: AsyncGenerator[CommittedEvent, None],
) -> AsyncGenerator[FrozenJsonObject, None]:
    try:
        async for event in committed:
            yield project_committed_event(event, audience=EventAudience.PUBLIC)
    finally:
        await committed.aclose()


__all__ = [
    "Agent",
    "AgentAlreadyExistsError",
    "AgentHomeError",
    "AgentIdentityMismatchError",
    "AgentNameError",
    "AgentNotConfiguredError",
    "AgentNotFoundError",
    "HostActiveError",
    "SessionOperationActiveError",
]
