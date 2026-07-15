"""
Base profiler ABC and normalized schema models.

Profilers connect to a discovered store and extract its full schema
(tables, columns, foreign keys) into a normalized representation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from daita.core.db_type_metadata import native_type_from_db_type, nullable_value

from .base_discoverer import DiscoveredStore


@dataclass
class NormalizedColumn:
    """A column in a normalized schema."""

    name: str
    type: str  # DB-native type ("varchar", "int4", "ObjectId")
    nullable: bool
    is_primary_key: bool
    comment: Optional[str] = None
    physical_type: Optional[str] = None
    native_type: Optional[str] = None
    database_dialect: Optional[str] = None
    logical_type: Optional[str] = None
    logical_type_proof: Dict[str, Any] = field(default_factory=dict)
    is_identity: Optional[bool] = None
    is_generated: Optional[bool] = None
    is_autoincrement: Optional[bool] = None
    is_monotonic: Optional[bool] = None
    identity_proof: Dict[str, Any] = field(default_factory=dict)
    default_value: Any = None
    extra: Optional[str] = None


@dataclass
class NormalizedColumnValue:
    """One bounded observed value for a cataloged column."""

    value: Any
    count: Optional[int] = None
    display: Optional[str] = None
    normalized: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"value": self.value}
        if self.count is not None:
            result["count"] = self.count
        if self.display is not None:
            result["display"] = self.display
        if self.normalized is not None:
            result["normalized"] = self.normalized
        return result

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "NormalizedColumnValue":
        return cls(
            value=value.get("value"),
            count=value.get("count"),
            display=value.get("display"),
            normalized=value.get("normalized"),
        )


@dataclass
class NormalizedColumnValueProfile:
    """Canonical catalog profile of bounded observed values for one column."""

    table: str
    column: str
    profile_kind: str = "categorical_values"
    profile_status: str = "profiled"
    distinct_count: Optional[int] = None
    null_count: Optional[int] = None
    row_count: Optional[int] = None
    top_values: List[NormalizedColumnValue] = field(default_factory=list)
    max_values: int = 25
    sampled: bool = False
    truncated: bool = False
    redacted: bool = False
    skipped_reason: Optional[str] = None
    policy: Dict[str, Any] = field(default_factory=dict)
    profiled_at: Optional[str] = None
    source_evidence_id: Optional[str] = None
    source_fingerprint: Optional[str] = None
    source_fingerprint_status: Optional[str] = None
    source_fingerprint_reason: Optional[str] = None
    source_revision: Optional[str] = None
    logical_type: Optional[str] = None
    logical_type_proof: Dict[str, Any] = field(default_factory=dict)

    @property
    def ref(self) -> str:
        return f"{self.table}.{self.column}"

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "table": self.table,
            "column": self.column,
            "profile_kind": self.profile_kind,
            "profile_status": self.profile_status,
            "top_values": [item.to_dict() for item in self.top_values],
            "max_values": self.max_values,
            "sampled": self.sampled,
            "truncated": self.truncated,
            "redacted": self.redacted,
            "policy": dict(self.policy),
        }
        for key, value in (
            ("distinct_count", self.distinct_count),
            ("null_count", self.null_count),
            ("row_count", self.row_count),
            ("skipped_reason", self.skipped_reason),
            ("profiled_at", self.profiled_at),
            ("source_evidence_id", self.source_evidence_id),
            ("source_fingerprint", self.source_fingerprint),
            ("source_fingerprint_status", self.source_fingerprint_status),
            ("source_fingerprint_reason", self.source_fingerprint_reason),
            ("source_revision", self.source_revision),
            ("logical_type", self.logical_type),
        ):
            if value is not None:
                result[key] = value
        if self.logical_type_proof:
            result["logical_type_proof"] = dict(self.logical_type_proof)
        return result

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "NormalizedColumnValueProfile":
        return cls(
            table=str(value.get("table") or ""),
            column=str(value.get("column") or ""),
            profile_kind=str(value.get("profile_kind") or "categorical_values"),
            profile_status=str(value.get("profile_status") or "profiled"),
            distinct_count=value.get("distinct_count"),
            null_count=value.get("null_count"),
            row_count=value.get("row_count"),
            top_values=[
                NormalizedColumnValue.from_dict(dict(item))
                for item in value.get("top_values", []) or []
                if isinstance(item, dict)
            ],
            max_values=int(value.get("max_values") or 25),
            sampled=bool(value.get("sampled", False)),
            truncated=bool(value.get("truncated", False)),
            redacted=bool(value.get("redacted", False)),
            skipped_reason=value.get("skipped_reason"),
            policy=dict(value.get("policy", {}) or {}),
            profiled_at=value.get("profiled_at"),
            source_evidence_id=value.get("source_evidence_id"),
            source_fingerprint=value.get("source_fingerprint"),
            source_fingerprint_status=value.get("source_fingerprint_status"),
            source_fingerprint_reason=value.get("source_fingerprint_reason"),
            source_revision=value.get("source_revision"),
            logical_type=value.get("logical_type"),
            logical_type_proof=dict(value.get("logical_type_proof", {}) or {}),
        )


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
                            **(
                                {"physical_type": c.physical_type}
                                if c.physical_type
                                else {}
                            ),
                            **({"native_type": c.native_type} if c.native_type else {}),
                            **(
                                {"database_dialect": c.database_dialect}
                                if c.database_dialect
                                else {}
                            ),
                            **(
                                {"logical_type": c.logical_type}
                                if c.logical_type
                                else {}
                            ),
                            **(
                                {"logical_type_proof": c.logical_type_proof}
                                if c.logical_type_proof
                                else {}
                            ),
                            **(
                                {"is_identity": c.is_identity}
                                if c.is_identity is not None
                                else {}
                            ),
                            **(
                                {"is_generated": c.is_generated}
                                if c.is_generated is not None
                                else {}
                            ),
                            **(
                                {"is_autoincrement": c.is_autoincrement}
                                if c.is_autoincrement is not None
                                else {}
                            ),
                            **(
                                {"is_monotonic": c.is_monotonic}
                                if c.is_monotonic is not None
                                else {}
                            ),
                            **(
                                {"identity_proof": c.identity_proof}
                                if c.identity_proof
                                else {}
                            ),
                            **(
                                {"default_value": c.default_value}
                                if c.default_value is not None
                                else {}
                            ),
                            **({"extra": c.extra} if c.extra else {}),
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
        if self.store_id:
            result["store_id"] = self.store_id
        if self.profiled_at:
            result["profiled_at"] = self.profiled_at
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "NormalizedSchema":
        """Hydrate a NormalizedSchema from a persisted catalog record."""
        tables = []
        for table in value.get("tables", []) or []:
            columns = []
            for column in table.get("columns", []) or []:
                physical_type = str(
                    column.get("physical_type")
                    or column.get("data_type")
                    or column.get("type")
                    or ""
                )
                nullable = nullable_value(
                    column.get("nullable")
                    if "nullable" in column
                    else column.get("is_nullable")
                )
                columns.append(
                    NormalizedColumn(
                        name=str(column.get("name", "")),
                        type=str(
                            column.get("type")
                            or column.get("data_type")
                            or physical_type
                        ),
                        nullable=True if nullable is None else nullable,
                        is_primary_key=bool(column.get("is_primary_key", False)),
                        comment=column.get("column_comment") or column.get("comment"),
                        physical_type=physical_type or None,
                        native_type=(
                            str(column.get("native_type"))
                            if column.get("native_type")
                            else native_type_from_db_type(physical_type)
                        ),
                        database_dialect=str(
                            column.get("database_dialect")
                            or column.get("dialect")
                            or value.get("database_type")
                            or ""
                        )
                        or None,
                        logical_type=(
                            str(column.get("logical_type"))
                            if column.get("logical_type")
                            else None
                        ),
                        logical_type_proof=dict(
                            column.get("logical_type_proof", {}) or {}
                        ),
                        is_identity=_optional_bool(column.get("is_identity")),
                        is_generated=_optional_bool(column.get("is_generated")),
                        is_autoincrement=_optional_bool(column.get("is_autoincrement")),
                        is_monotonic=_optional_bool(column.get("is_monotonic")),
                        identity_proof=dict(column.get("identity_proof", {}) or {}),
                        default_value=(
                            column.get("default_value")
                            if "default_value" in column
                            else column.get("column_default")
                        ),
                        extra=(
                            str(column.get("extra"))
                            if column.get("extra") is not None
                            else None
                        ),
                    )
                )
            indexes = [
                NormalizedIndex(
                    name=str(index.get("name", "")),
                    type=str(index.get("type", "")),
                    columns=list(index.get("columns", []) or []),
                    unique=bool(index.get("unique", False)),
                    metadata=dict(index.get("metadata", {}) or {}),
                )
                for index in table.get("indexes", []) or []
            ]
            tables.append(
                NormalizedTable(
                    name=str(table.get("name", "")),
                    row_count=table.get("row_count"),
                    columns=columns,
                    indexes=indexes,
                    metadata=dict(table.get("metadata", {}) or {}),
                )
            )

        foreign_keys = [
            NormalizedForeignKey(
                source_table=str(fk.get("source_table", "")),
                source_column=str(fk.get("source_column", "")),
                target_table=str(fk.get("target_table", "")),
                target_column=str(fk.get("target_column", "")),
            )
            for fk in value.get("foreign_keys", []) or []
        ]

        return cls(
            database_type=str(value.get("database_type", "unknown")),
            database_name=str(value.get("database_name") or value.get("schema") or ""),
            tables=tables,
            foreign_keys=foreign_keys,
            table_count=int(value.get("table_count") or len(tables)),
            store_id=value.get("store_id"),
            profiled_at=value.get("profiled_at"),
            metadata=dict(value.get("metadata", {}) or {}),
        )


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in {"true", "yes", "1"}:
        return True
    if lowered in {"false", "no", "0"}:
        return False
    return None


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
