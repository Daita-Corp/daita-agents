"""
Session-scoped working memory (scratchpad).

In-memory only — no disk, no embeddings, no API calls.
Auto-evicted on agent stop unless explicitly promoted to long-term memory.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class WorkingMemoryItem:
    key: str
    content: str
    created_at: datetime
    promoted: bool = False


class WorkingMemory:
    """In-memory session-scoped scratchpad. No disk, no embeddings."""

    def __init__(self):
        self._store: Dict[str, WorkingMemoryItem] = {}
        self._counter: int = 0

    def scratch(self, content: str, key: Optional[str] = None) -> str:
        """Write to working memory. Returns the assigned key."""
        if key is None:
            self._counter += 1
            key = f"scratch_{self._counter}"

        self._store[key] = WorkingMemoryItem(
            key=key,
            content=content,
            created_at=datetime.now(),
        )
        return key

    def think(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search working memory by case-insensitive substring + keyword match.

        Returns matching items sorted by relevance (keyword overlap count).
        """
        query_lower = query.lower()
        query_tokens = set(query_lower.split())

        scored = []
        for item in self._store.values():
            content_lower = item.content.lower()

            # Substring match
            if query_lower in content_lower:
                score = 2.0
            else:
                # Keyword overlap
                content_tokens = set(content_lower.split())
                overlap = query_tokens & content_tokens
                if not overlap:
                    continue
                score = len(overlap) / len(query_tokens)

            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._item_to_dict(item) for _, item in scored[:limit]]

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific item by key."""
        item = self._store.get(key)
        return self._item_to_dict(item) if item else None

    def promote(self, key: str) -> Optional[Dict[str, Any]]:
        """Mark item as promoted and return its content for remember()."""
        item = self._store.get(key)
        if item is None:
            return None
        item.promoted = True
        return self._item_to_dict(item)

    def clear(self):
        """Discard all working memory (called on agent stop)."""
        self._store.clear()
        self._counter = 0

    def dump(self) -> List[Dict[str, Any]]:
        """Return all items for introspection."""
        return [self._item_to_dict(item) for item in self._store.values()]

    def __len__(self) -> int:
        return len(self._store)

    @staticmethod
    def _item_to_dict(item: WorkingMemoryItem) -> Dict[str, Any]:
        return {
            "key": item.key,
            "content": item.content,
            "created_at": item.created_at.isoformat(),
            "promoted": item.promoted,
        }
