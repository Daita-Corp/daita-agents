"""
Base profiler ABC and normalized schema models.

Profilers connect to a discovered store and extract its full schema
(tables, columns, foreign keys) into a normalized representation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base_discoverer import DiscoveredStore


@dataclass
class NormalizedColumn:
    """A column in a normalized schema."""

    name: str
    type: str  # DB-native type ("varchar", "int4", "ObjectId")
    nullable: bool
    is_primary_key: bool
    comment: Optional[str] = None


@dataclass
class NormalizedIndex:
    """
    A declared access path over one or more columns.

    Unifies every store's notion of "secondary structure":
      * SQL         — ``btree`` / ``hash`` / ``gin`` / ``partial``
      * BigQuery    — ``partition`` / ``cluster``
      * DynamoDB    — ``gsi`` / ``lsi``
      * Firestore   — ``composite``
      * MongoDB     — ``composite`` / ``text`` / ``geo``
      * Elasticsearch — ``text`` / ``keyword``

    Agents can reason uniformly: ``tables[].indexes`` is the queryable surface
    for "what access patterns are efficient on this store".
    """

    name: str
    type: str
    columns: List[str] = field(default_factory=list)
    unique: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedTable:
    """A table (or collection, bucket, etc.) in a normalized schema."""

    name: str
    row_count: Optional[int]
    columns: List[NormalizedColumn] = field(default_factory=list)
    indexes: List[NormalizedIndex] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedForeignKey:
    """A foreign key relationship between two tables."""

    source_table: str
    source_column: str
    target_table: str
    target_column: str


@dataclass
class NormalizedSchema:
    """
    Normalized schema representation produced by profilers.

    The to_dict() method returns the exact same shape as the existing
    normalize_discovery() dict output for backward compatibility.
    """

    database_type: str
    database_name: str
    tables: List[NormalizedTable] = field(default_factory=list)
    foreign_keys: List[NormalizedForeignKey] = field(default_factory=list)
    table_count: int = 0
    store_id: Optional[str] = None  # links back to DiscoveredStore.id
    profiled_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Same shape as normalize_discovery() output, plus metadata when present."""
        result: Dict[str, Any] = {
            "database_type": self.database_type,
            "database_name": self.database_name,
            "tables": [
                {
                    "name": t.name,
                    "row_count": t.row_count,
                    "columns": [
                        {
                            "name": c.name,
                            "type": c.type,
                            "nullable": c.nullable,
                            "is_primary_key": c.is_primary_key,
                            **({"column_comment": c.comment} if c.comment else {}),
                        }
                        for c in t.columns
                    ],
                    "indexes": [
                        {
                            "name": i.name,
                            "type": i.type,
                            "columns": i.columns,
                            "unique": i.unique,
                            **({"metadata": i.metadata} if i.metadata else {}),
                        }
                        for i in t.indexes
                    ],
                    **({"metadata": t.metadata} if t.metadata else {}),
                }
                for t in self.tables
            ],
            "foreign_keys": [
                {
                    "source_table": fk.source_table,
                    "source_column": fk.source_column,
                    "target_table": fk.target_table,
                    "target_column": fk.target_column,
                }
                for fk in self.foreign_keys
            ],
            "table_count": self.table_count,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class BaseProfiler(ABC):
    """
    Abstract base class for store profilers.

    Profilers connect to a data store and extract its schema into a
    NormalizedSchema. Each profiler handles one or more store types.
    """

    @abstractmethod
    def supports(self, store_type: str) -> bool:
        """Return True if this profiler can handle the given store type."""
        ...

    @abstractmethod
    async def profile(self, store: DiscoveredStore) -> NormalizedSchema:
        """
        Connect to the store and extract its schema.

        Args:
            store: The discovered store to profile

        Returns:
            NormalizedSchema with full schema information
        """
        ...
