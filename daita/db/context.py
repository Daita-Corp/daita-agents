"""
Context rendering helpers for DB runtime synthesis and inspection.
"""

from __future__ import annotations

from daita.runtime import Evidence


class DbContextRenderer:
    """Render compact context from accepted DB evidence."""

    def render_evidence_summary(
        self, evidence: tuple[Evidence, ...], *, max_items: int = 10
    ) -> str:
        """Render a short deterministic evidence summary."""
        parts = []
        for item in evidence[:max_items]:
            label = item.kind
            if item.task_id:
                label = f"{label} from {item.task_id}"
            parts.append(label)
        if len(evidence) > max_items:
            parts.append(f"{len(evidence) - max_items} more")
        return "; ".join(parts)
