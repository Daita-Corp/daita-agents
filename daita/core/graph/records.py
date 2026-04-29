"""
Reusable graph record primitives.

Domain graph layers can subclass these records to provide their own node and
edge vocabularies without duplicating persistence fields or backend behavior.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GraphNodeRecord(BaseModel):
    """Generic persistable graph node shape."""

    node_id: str
    node_type: str
    name: str
    created_by_agent: Optional[str] = None
    created_at_execution: Optional[str] = None
    confidence: float = 1.0
    health_score: Optional[float] = None
    last_seen: Optional[datetime] = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    tags: List[str] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def make_id(cls, node_type: Enum | str, name: str) -> str:
        value = node_type.value if isinstance(node_type, Enum) else str(node_type)
        return f"{value}:{name}"


class GraphEdgeRecord(BaseModel):
    """Generic persistable graph edge shape."""

    edge_id: str
    from_node_id: str
    to_node_id: str
    edge_type: str
    created_by_agent: Optional[str] = None
    execution_id: Optional[str] = None
    confidence: float = 1.0
    impact_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=_utcnow)
    properties: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def make_id(cls, from_node_id: str, edge_type: Enum | str, to_node_id: str) -> str:
        value = edge_type.value if isinstance(edge_type, Enum) else str(edge_type)
        return f"{from_node_id}:{value}:{to_node_id}"
