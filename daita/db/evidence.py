"""
Evidence storage for DB runtime operations.
"""

from __future__ import annotations

from typing import Any

from daita.runtime import Evidence


async def load_evidence(
    runtime: Any,
    operation_id: str,
    evidence_id: Any,
) -> Evidence | None:
    """Load one operation-scoped evidence item by its durable ID."""
    if not evidence_id:
        return None
    for evidence in await runtime.store.list_evidence(operation_id):
        if evidence.id == evidence_id:
            return evidence
    return None


async def load_evidence_refs_or_latest(
    runtime: Any,
    operation_id: str,
    evidence_ids: Any,
    *,
    kinds: tuple[str, ...],
) -> tuple[Evidence, ...]:
    """Load explicit evidence refs in order, or accepted matching evidence."""
    loaded = tuple(
        item
        for item in [
            await load_evidence(runtime, operation_id, evidence_id)
            for evidence_id in evidence_ids or ()
        ]
        if item is not None
    )
    if loaded:
        return loaded
    evidence = await runtime.store.list_evidence(operation_id)
    return tuple(item for item in evidence if item.kind in kinds and item.accepted)


class DbEvidenceStore:
    """Operation-scoped evidence collection.

    This is intentionally small for the first DB runtime pass. It gives the
    runtime a named owner for accepted evidence before durable stores exist.
    """

    def __init__(self) -> None:
        self._items: list[Evidence] = []

    def add(self, evidence: Evidence) -> None:
        """Store one accepted evidence item."""
        if evidence.accepted:
            self._items.append(evidence)

    def add_many(self, evidence: tuple[Evidence, ...]) -> None:
        """Store multiple accepted evidence items."""
        for item in evidence:
            self.add(item)

    def add_diagnostic(self, evidence: Evidence) -> None:
        """Store one evidence item that must remain visible even if rejected."""
        self._items.append(evidence)

    def discard(self, evidence_id: str | None) -> None:
        """Remove one evidence item from the accepted operation view."""
        if not evidence_id:
            return
        self._items = [item for item in self._items if item.id != evidence_id]

    def list(self) -> tuple[Evidence, ...]:
        """Return accepted evidence in insertion order."""
        return tuple(self._items)

    def kinds(self) -> set[str]:
        """Return the accepted evidence kinds currently present."""
        return {item.kind for item in self._items}

    def refs(self) -> tuple[dict[str, str | None], ...]:
        """Return stable-enough evidence references for diagnostics."""
        return tuple(
            {
                "id": item.id,
                "kind": item.kind,
                "owner": item.owner,
                "task_id": item.task_id,
            }
            for item in self._items
        )


InMemoryDbEvidenceStore = DbEvidenceStore
