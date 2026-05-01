"""
Data models for the agent graph system.

AgentGraphNode and AgentGraphEdge are agent-native graph primitives that carry
provenance, confidence, and impact metadata not found in generic graph libraries.
"""

from enum import Enum

from pydantic import field_validator

from .records import GraphEdgeRecord, GraphNodeRecord


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
    # Compatibility only. Memory graph code owns its domain types in
    # daita.plugins.memory.graph_models.
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
    # Compatibility only. Memory graph code owns its domain types in
    # daita.plugins.memory.graph_models. Will be removed in future updates.
    MENTIONS = "mentions"
    RELATED_TO = "related_to"
    SUPERSEDES = "supersedes"


class AgentGraphNode(GraphNodeRecord):
    node_type: NodeType | str

    @field_validator("node_type", mode="before")
    @classmethod
    def _coerce_known_node_type(cls, value):
        try:
            return NodeType(value)
        except ValueError:
            return value


class AgentGraphEdge(GraphEdgeRecord):
    edge_type: EdgeType | str

    @field_validator("edge_type", mode="before")
    @classmethod
    def _coerce_known_edge_type(cls, value):
        try:
            return EdgeType(value)
        except ValueError:
            return value
