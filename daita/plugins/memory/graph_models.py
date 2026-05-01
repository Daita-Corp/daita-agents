"""
Domain models for the memory knowledge graph.

These types intentionally live with the memory plugin instead of the core graph
package. The core graph package provides reusable storage mechanics; this
module owns memory-specific vocabulary such as memories, entities, and semantic
relationships between them.
"""

from enum import Enum

from pydantic import field_validator

from ...core.graph.records import GraphEdgeRecord, GraphNodeRecord


class MemoryNodeType(str, Enum):
    MEMORY = "memory"
    ENTITY = "entity"


class MemoryEdgeType(str, Enum):
    MENTIONS = "mentions"
    RELATED_TO = "related_to"
    SUPERSEDES = "supersedes"


class MemoryGraphNode(GraphNodeRecord):
    """Persistable node record for the memory graph domain."""

    node_type: MemoryNodeType | str

    @field_validator("node_type", mode="before")
    @classmethod
    def _coerce_known_node_type(cls, value):
        try:
            return MemoryNodeType(value)
        except ValueError:
            return value


class MemoryGraphEdge(GraphEdgeRecord):
    """Persistable edge record for the memory graph domain."""

    edge_type: MemoryEdgeType | str

    @field_validator("edge_type", mode="before")
    @classmethod
    def _coerce_known_edge_type(cls, value):
        try:
            return MemoryEdgeType(value)
        except ValueError:
            return value
