"""Portable persistence seam for the memory service."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from .models import (
    MemoryHistory,
    MemoryRestoreRequest,
    MemoryScope,
    MemorySensitivity,
    MemorySnapshot,
    MemoryState,
    MemorySupersessionRequest,
)


class MemoryStoreError(RuntimeError):
    """Base class for portable memory-store failures."""


class MemoryStoreConflictError(MemoryStoreError):
    """Raised when a version-guarded lifecycle request loses its race."""


@runtime_checkable
class MemoryStore(Protocol):
    """Store current projections and immutable history without exposing SQL.

    Implementations must apply the supplied state, scope, sensitivity, and
    expiry predicates before any FTS or other lexical candidate ranking. The
    service repeats those checks as defense in depth.
    """

    async def recall_candidates(
        self,
        *,
        query: str,
        scope: MemoryScope,
        states: tuple[MemoryState, ...],
        sensitivities: tuple[MemorySensitivity, ...],
        unexpired_at: datetime,
        limit: int,
    ) -> tuple[MemorySnapshot, ...]: ...

    async def list_candidates(
        self,
        *,
        scope: MemoryScope,
        states: tuple[MemoryState, ...],
        sensitivities: tuple[MemorySensitivity, ...],
        limit: int,
    ) -> tuple[MemorySnapshot, ...]: ...

    async def load_history(
        self,
        agent_id: str,
        memory_id: str,
    ) -> MemoryHistory | None: ...

    async def supersede(
        self,
        request: MemorySupersessionRequest,
    ) -> MemoryHistory: ...

    async def restore(
        self,
        request: MemoryRestoreRequest,
    ) -> MemoryHistory: ...


__all__ = [
    "MemoryStore",
    "MemoryStoreConflictError",
    "MemoryStoreError",
]
