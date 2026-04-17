"""
Data models for the agent graph system.

AgentGraphNode and AgentGraphEdge are agent-native graph primitives that carry
provenance, confidence, and impact metadata not found in generic graph libraries.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class NodeType(str, Enum):
    TABLE = "table"
    COLUMN = "column"
    INDEX = "index"
    AGENT = "agent"
    PIPELINE = "pipeline"
    TRANSFORMATION = "transformation"
    MODEL = "model"
    API = "api"
    FILE = "file"
    METRIC = "metric"
    QUERY = "query"
    DATABASE = "database"
    BUCKET = "bucket"
    SERVICE = "service"
    # Memory graph types
    MEMORY = "memory"
    ENTITY = "entity"


class EdgeType(str, Enum):
    READS = "reads"
    WRITES = "writes"
    TRANSFORMS = "transforms"
    TRIGGERS = "triggers"
    CALLS = "calls"
    PRODUCES = "produces"
    SYNCS_TO = "syncs_to"
    DERIVED_FROM = "derived_from"
    HAS_COLUMN = "has_column"
    INDEXED_BY = "indexed_by"
    COVERS = "covers"
    REFERENCES = "references"
    PART_OF = "part_of"
    # Memory graph types
    MENTIONS = "mentions"
    RELATED_TO = "related_to"
    SUPERSEDES = "supersedes"


class AgentGraphNode(BaseModel):
    # Identity
    node_id: str
    node_type: NodeType
    name: str

    # Agent provenance
    created_by_agent: Optional[str] = None
    created_at_execution: Optional[str] = None

    # Confidence: 1.0 = directly observed, <1.0 = agent-inferred
    confidence: float = 1.0

    # Health: None = not assessed, 0.0 = broken, 1.0 = healthy
    health_score: Optional[float] = None

    # Temporal
    last_seen: Optional[datetime] = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    # Categorization
    tags: List[str] = []

    # Type-specific data (row count, schema info, agent config, etc.)
    properties: Dict[str, Any] = {}

    @classmethod
    def make_id(cls, node_type: NodeType, name: str) -> str:
        """Generate a deterministic node_id from type and name."""
        return f"{node_type.value}:{name}"


class AgentGraphEdge(BaseModel):
    # Identity — deterministic ID prevents duplicate edges
    edge_id: str
    from_node_id: str
    to_node_id: str
    edge_type: EdgeType

    # Agent provenance
    created_by_agent: Optional[str] = None
    execution_id: Optional[str] = None
    confidence: float = 1.0

    # Impact weight: 1.0 = critical dependency, 0.0 = loose coupling
    impact_weight: float = Field(default=1.0, ge=0.0, le=1.0)

    # Temporal
    timestamp: datetime = Field(default_factory=_utcnow)

    # Edge-specific metadata (transformation SQL, sync frequency, etc.)
    properties: Dict[str, Any] = {}

    @classmethod
    def make_id(cls, from_node_id: str, edge_type: EdgeType, to_node_id: str) -> str:
        """Generate a deterministic edge_id."""
        return f"{from_node_id}:{edge_type.value}:{to_node_id}"
