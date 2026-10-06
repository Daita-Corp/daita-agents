"""Describe durable state operations independently of a database implementation.

This is the complete composition-boundary contract. Domain owners continue to
accept their existing narrower protocols. Construction, home upgrades, filesystem
paths and schema revision checks belong to backend-specific admission code.

Each mutation is one atomic operation: a backend must preserve the existing
compare-and-set, idempotency, fencing, ordering and budget-conservation semantics.
A successful await means durable commit; cancellation or lost acknowledgement
must not cause blind replay of external effects. Identity, payload validation,
bounds and domain errors have the same meaning for every backend.

SQLite and custom SQL adapters can share the SQL operation implementation. Admission
owns connections, schema validation and fencing. Agent composition still uses
local homes; a state backend alone does not provide remote files or writer leases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ..artifacts.store import ArtifactRegistry
from ..capability_runtime import EffectReceiptStore
from ..catalog.protocols import CatalogStore
from ..distribution.owner import DistributionStore
from ..domains.data.catalog import CatalogPermissionStore
from ..domains.mcp import MCPBindingStore
from ..jobs.owner import GraphJobStore, GraphSupervisorStore
from ..learning_candidates import LearningCandidateStore
from ..loop.driver import TranscriptStore
from ..routines.owner import RoutineStore, RoutineSupervisorStore
from ..semantics import SemanticStore

if TYPE_CHECKING:
    from datetime import datetime

    from ..adapters.mcp import MCPServerBinding
    from ..adapters.models import SourceRegistration
    from ..capabilities import ExecutionScope
    from ..catalog.models import (
        SourceCatalogSnapshot,
    )
    from ..identity import AgentIdentity
    from ..jobs.graph.models import (
        AttemptBudgetReservation,
        BudgetLedger,
    )
    from ..loop.models import ConversationRun, LoopExit
    from ..routines.models import (
        ResourceRevisionObservation,
        RoutineOccurrence,
    )
    from .sqlite_records import (
        EffectReceipt,
        EffectResolution,
        RelationalWriteScope,
        SourceReadScope,
    )


class StateStore(
    ArtifactRegistry,
    EffectReceiptStore,
    CatalogStore,
    DistributionStore,
    CatalogPermissionStore,
    MCPBindingStore,
    GraphJobStore,
    LearningCandidateStore,
    TranscriptStore,
    RoutineStore,
    SemanticStore,
    GraphSupervisorStore,
    RoutineSupervisorStore,
    Protocol,
):
    """Composition of domain contracts; consumers accept their narrow views."""

    async def list_graph_budget_ledgers(
        self, agent_id: str, job_id: str
    ) -> tuple[BudgetLedger, ...]: ...

    async def list_graph_attempt_reservations(
        self, agent_id: str, job_id: str, task_id: str, attempt_id: str
    ) -> tuple[AttemptBudgetReservation, ...]: ...

    async def close(self) -> None: ...

    async def initialize_identity(self, identity: AgentIdentity) -> AgentIdentity: ...

    async def load_identity(self) -> AgentIdentity | None: ...

    async def load_deletion_credential_inventory(
        self, agent_id: str
    ) -> tuple[AgentIdentity | None, tuple[str, ...], tuple[str, ...]]: ...

    async def store_mcp_binding(
        self, binding: MCPServerBinding, *, expected_revision: int | None
    ) -> MCPServerBinding: ...

    async def update_mcp_discovery(
        self,
        agent_id: str,
        binding_id: str,
        *,
        summary: str,
        when_to_use: str,
        keywords: tuple[str, ...],
    ) -> MCPServerBinding: ...

    async def update_source_discovery(
        self,
        agent_id: str,
        source_id: str,
        *,
        summary: str,
        when_to_use: str,
        keywords: tuple[str, ...],
    ) -> SourceRegistration: ...

    async def bind_routine_occurrence_run(
        self,
        agent_id: str,
        occurrence_id: str,
        *,
        claim_token: str,
        run_id: str,
        execution_scope: ExecutionScope,
        bound_at: datetime,
        precheck_observation: ResourceRevisionObservation | None = ...,
    ) -> RoutineOccurrence | None: ...

    async def replace_source_permission_scopes(
        self,
        read_scope: SourceReadScope,
        update_scopes: tuple[RelationalWriteScope, ...],
    ) -> SourceRegistration: ...

    async def load_effect_receipt(
        self, agent_id: str, receipt_id: str
    ) -> EffectReceipt | None: ...

    async def list_effect_receipts(
        self,
        agent_id: str,
        *,
        run_id: str | None = ...,
        routine_id: str | None = ...,
        unresolved_only: bool = ...,
        limit: int = ...,
        offset: int = ...,
        caller_principal_id: str | None = ...,
    ) -> tuple[EffectReceipt, ...]: ...

    async def list_effect_receipts_for_graph_attempt(
        self, agent_id: str, job_id: str, task_id: str, attempt_id: str
    ) -> tuple[EffectReceipt, ...]: ...

    async def resolve_effect_receipt(
        self, agent_id: str, resolution: EffectResolution
    ) -> EffectReceipt: ...

    async def commit_snapshot(
        self,
        snapshot: SourceCatalogSnapshot,
        *,
        registration: SourceRegistration | None = ...,
    ) -> SourceCatalogSnapshot: ...

    async def commit_source_edit(
        self,
        snapshot: SourceCatalogSnapshot,
        *,
        registration: SourceRegistration,
        replaced_source_id: str,
        replaced_at: datetime,
        read_scope: SourceReadScope,
    ) -> SourceCatalogSnapshot: ...

    async def recover_unfinished_runs(
        self, agent_id: str, *, created_at: datetime
    ) -> tuple[LoopExit, ...]: ...

    async def recover_started_effect_receipts(
        self, agent_id: str, *, recovered_at: datetime
    ) -> None:
        """Recover only under exclusive ownership, before admitting new work."""
        ...

    async def conversation_run_page(
        self,
        agent_id: str,
        conversation_id: str,
        *,
        after_turn_index: int = -1,
        limit: int = 100,
    ) -> tuple[ConversationRun, ...]:
        """Ascending turns after an exclusive cursor; limit is 1..100."""
        ...

    async def conversation_access(
        self,
        agent_id: str,
        conversation_id: str,
        *,
        caller_principal_id: str | None = None,
    ) -> tuple[bool, bool]:
        """Return (exists, allowed), checking every turn without loading messages."""
        ...

    async def conversation_runs(
        self, agent_id: str, conversation_id: str
    ) -> tuple[ConversationRun, ...]: ...

    async def completed_conversation_tail(
        self, agent_id: str, conversation_id: str, *, limit: int = ...
    ) -> tuple[bool, tuple[ConversationRun, ...], bool]: ...

    async def latest_terminal_conversation_run(
        self, agent_id: str, conversation_id: str
    ) -> ConversationRun | None: ...
