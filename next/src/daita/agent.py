"""Thin public facade over the embedded agent composition."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import datetime
from pathlib import Path
from typing import Self

from .adapters.models import SourceRegistration
from .adapters.protocols import ResourceSource
from .capabilities import CapabilityRegistry
from .events.models import CommittedEvent, EventCursor
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
        model_profile: ModelProfile | None = None,
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
            model_profile=model_profile,
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
        model_profile: ModelProfile | None = None,
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
            model_profile=model_profile,
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

    @property
    def model_profile(self) -> ModelProfile | None:
        return self._embedded.model_profile

    async def run(self, message: str, *, session_id: str | None = None) -> LoopExit:
        return await self._embedded.run(message, session_id=session_id)

    async def attach(self, source: ResourceSource) -> SourceRegistration:
        return await self._embedded.attach(source)

    async def inspect(self, operation_id: str) -> OperationSnapshot:
        return await self._embedded.inspect(operation_id)

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
    ) -> tuple[CommittedEvent, ...]:
        return await self._embedded.read_events(cursor, limit=limit)

    def subscribe_events(
        self,
        cursor: EventCursor | None = None,
    ) -> AsyncIterator[CommittedEvent]:
        return self._embedded.subscribe_events(cursor)

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
