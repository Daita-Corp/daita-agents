"""
Memory metadata structure for importance tracking and filtering.

Replaces old text-suffix format with proper structured metadata.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional
import json


@dataclass
class MemoryMetadata:
    """
    Structured metadata for memory entries.

    Enables filtering, sorting, and intelligent pruning based on
    importance, usage, and source.
    """

    content: str
    importance: float = 0.5
    source: str = "agent_inferred"
    category: Optional[str] = None
    created_at: Optional[datetime] = None
    pinned: bool = False
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    ttl_days: Optional[int] = None
    reinforcements: Optional[List[Dict[str, Any]]] = None
    flagged_for_review: bool = False

    def __post_init__(self):
        """Set defaults for datetime fields."""
        if self.created_at is None:
            self.created_at = datetime.now()

        # Validate importance
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError(f"Importance must be 0.0-1.0, got {self.importance}")

        # source is a free-form string; no restriction on custom values

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict with datetime serialization."""
        data = asdict(self)
        # Convert datetime to ISO format
        if self.created_at:
            data["created_at"] = self.created_at.isoformat()
        if self.last_accessed:
            data["last_accessed"] = self.last_accessed.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryMetadata":
        """Create from dict with datetime parsing."""
        # Parse datetime strings
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "last_accessed" in data and isinstance(data["last_accessed"], str):
            data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])

        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    def should_prune(
        self, max_age_days: int = 90, min_importance_threshold: float = 0.3
    ) -> bool:
        """
        Determine if this memory should be pruned.

        Pruning rules:
        - Pinned memories: NEVER prune
        - Age > max_age_days AND importance < min_importance_threshold: prune
        - Never accessed AND age > 60 days AND importance < 0.5: prune sooner

        Args:
            max_age_days: Maximum age in days before considering pruning
            min_importance_threshold: Minimum importance to keep old memories

        Returns:
            True if should be pruned
        """
        # Never prune pinned memories
        if self.pinned:
            return False

        # Calculate age
        age_days = (datetime.now() - self.created_at).days

        # TTL expiry — hard cutoff regardless of importance or access
        if self.ttl_days is not None and age_days > self.ttl_days:
            return True

        # Consistently negative reinforcement: prune at half the normal age threshold
        if self.reinforcements:
            negatives = sum(
                1 for r in self.reinforcements if r.get("outcome") == "negative"
            )
            total = len(self.reinforcements)
            if total >= 3 and negatives / total >= 0.7:
                if age_days > max_age_days / 2 and self.importance < 0.6:
                    return True

        # Never-accessed old memories prune sooner
        if self.access_count == 0 and age_days > 60 and self.importance < 0.5:
            return True

        # Prune if old AND low importance
        return age_days > max_age_days and self.importance < min_importance_threshold
