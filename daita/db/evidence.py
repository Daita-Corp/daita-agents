"""
Evidence storage for DB runtime operations.
"""

from __future__ import annotations

from daita.runtime import Evidence


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
