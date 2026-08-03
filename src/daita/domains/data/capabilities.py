"""Read-only SQL tool declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..._json import FrozenJsonObject
from ...capabilities import (
    AccessMode,
    Capability,
    Executor,
    ExtensionDeclarations,
    ToolApplicability,
    ToolExecution,
    ToolOutput,
    ToolView,
)
from .controller import (
    POSTGRESQL_QUERY_CAPABILITY_ID,
    POSTGRESQL_QUERY_EVIDENCE_KIND,
    SQLITE_QUERY_CAPABILITY_ID,
    SQLITE_QUERY_EVIDENCE_KIND,
)
from .results import BoundedResultProjection
from .sql import MAX_SQL_CHARACTERS, MAX_SQL_PARAMETERS

SQLITE_QUERY_EXECUTOR_ID = "data.sqlite.query.executor"
POSTGRESQL_QUERY_EXECUTOR_ID = "data.postgresql.query.executor"


@dataclass(frozen=True, slots=True)
class SqlReadResult:
    source_id: str
    canonical_sql: str
    sql_fingerprint: str
    resource_ids: tuple[str, ...]
    resource_revisions: tuple[tuple[str, str], ...]
    source_revision: str
    columns: tuple[str, ...]
    projection: BoundedResultProjection

    def __post_init__(self) -> None:
        for value, name in (
            (self.source_id, "source_id"),
            (self.canonical_sql, "canonical_sql"),
            (self.sql_fingerprint, "sql_fingerprint"),
            (self.source_revision, "source_revision"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        resource_ids = tuple(self.resource_ids)
        revisions = tuple(sorted(tuple(item) for item in self.resource_revisions))
        columns = tuple(self.columns)
        if not resource_ids or len(resource_ids) != len(set(resource_ids)):
            raise ValueError("resource_ids must be non-empty and unique")
        if {item[0] for item in revisions} != set(resource_ids):
            raise ValueError("resource_revisions must cover every resource")
        if any(not value.startswith("sha256:") for _, value in revisions):
            raise ValueError("resource revisions must use sha256")
        if any(not isinstance(item, str) or not item for item in columns):
            raise ValueError("columns must contain non-empty strings")
        if not isinstance(self.projection, BoundedResultProjection):
            raise TypeError("projection must be BoundedResultProjection")
        object.__setattr__(self, "resource_ids", resource_ids)
        object.__setattr__(self, "resource_revisions", revisions)
        object.__setattr__(self, "columns", columns)

    def tool_data(self) -> FrozenJsonObject:
        projection = self.projection.to_payload().to_dict()
        return FrozenJsonObject.from_mapping(
            {
                **projection,
                "canonical_sql": self.canonical_sql,
                "columns": self.columns,
                "resource_ids": self.resource_ids,
                "resource_revisions": tuple(
                    {"resource_id": resource_id, "revision": revision}
                    for resource_id, revision in self.resource_revisions
                ),
                "source_id": self.source_id,
                "source_revision": self.source_revision,
                "sql_fingerprint": self.sql_fingerprint,
            }
        )


@dataclass(frozen=True, slots=True)
class SQLiteReadResult(SqlReadResult):
    pass


@dataclass(frozen=True, slots=True)
class PostgreSQLReadResult(SqlReadResult):
    pass


class SqlReadBackend(Protocol):
    async def execute_read(
        self,
        *,
        agent_id: str,
        source_id: str,
        sql: str,
        parameters: tuple[object, ...],
        max_rows: int,
        max_bytes: int,
    ) -> SqlReadResult: ...


class SQLiteReadBackend(SqlReadBackend, Protocol):
    pass


class PostgreSQLReadBackend(SqlReadBackend, Protocol):
    pass


@dataclass(frozen=True, slots=True)
class SQLiteQueryDeclarations:
    capabilities: tuple[Capability, ...]
    executors: tuple[Executor, ...]
    tool_views: tuple[ToolView, ...]


@dataclass(frozen=True, slots=True)
class PostgreSQLQueryDeclarations:
    capabilities: tuple[Capability, ...]
    executors: tuple[Executor, ...]
    tool_views: tuple[ToolView, ...]


class _SqlQueryExecutor:
    executor_id: str
    output_kind: str

    def __init__(
        self,
        agent_id: str,
        backend: SqlReadBackend,
        *,
        max_rows: int = 100,
        max_bytes: int = 65_536,
    ) -> None:
        self._agent_id = agent_id
        self._backend = backend
        self._max_rows = max_rows
        self._max_bytes = max_bytes

    async def execute(self, request: ToolExecution) -> ToolOutput:
        source_id = request.arguments["source_id"]
        sql = request.arguments["sql"]
        parameters = request.arguments.get("parameters", ())
        assert isinstance(source_id, str)
        assert isinstance(sql, str)
        assert isinstance(parameters, tuple)
        result = await self._backend.execute_read(
            agent_id=self._agent_id,
            source_id=source_id,
            sql=sql,
            parameters=parameters,
            max_rows=self._max_rows,
            max_bytes=self._max_bytes,
        )
        if result.source_id != source_id:
            raise ValueError("SQL backend returned a different source")
        return ToolOutput(kind=self.output_kind, data=result.tool_data())


class SQLiteQueryExecutor(_SqlQueryExecutor):
    executor_id = SQLITE_QUERY_EXECUTOR_ID
    output_kind = SQLITE_QUERY_EVIDENCE_KIND


class PostgreSQLQueryExecutor(_SqlQueryExecutor):
    executor_id = POSTGRESQL_QUERY_EXECUTOR_ID
    output_kind = POSTGRESQL_QUERY_EVIDENCE_KIND


def sqlite_query_declarations(
    agent_id: str, backend: SQLiteReadBackend
) -> SQLiteQueryDeclarations:
    executor = SQLiteQueryExecutor(agent_id, backend)
    extension = sqlite_query_extension_declarations()
    return SQLiteQueryDeclarations(
        extension.capabilities, (executor,), extension.tool_views
    )


def postgresql_query_declarations(
    agent_id: str, backend: PostgreSQLReadBackend
) -> PostgreSQLQueryDeclarations:
    executor = PostgreSQLQueryExecutor(agent_id, backend)
    extension = postgresql_query_extension_declarations()
    return PostgreSQLQueryDeclarations(
        extension.capabilities, (executor,), extension.tool_views
    )


def sqlite_query_extension_declarations() -> ExtensionDeclarations:
    return _query_declarations(
        SQLITE_QUERY_CAPABILITY_ID,
        SQLITE_QUERY_EVIDENCE_KIND,
        SQLITE_QUERY_EXECUTOR_ID,
        "sqlite",
        "SQLite",
        "data_query_sqlite",
    )


def postgresql_query_extension_declarations() -> ExtensionDeclarations:
    return _query_declarations(
        POSTGRESQL_QUERY_CAPABILITY_ID,
        POSTGRESQL_QUERY_EVIDENCE_KIND,
        POSTGRESQL_QUERY_EXECUTOR_ID,
        "postgresql",
        "PostgreSQL",
        "data_query_postgresql",
    )


def _query_declarations(
    capability_id: str,
    output_kind: str,
    executor_id: str,
    adapter_id: str,
    adapter_name: str,
    tool_name: str,
) -> ExtensionDeclarations:
    capability = Capability(
        id=capability_id,
        description=f"Run one validated, read-only, bounded {adapter_name} query.",
        input_schema={
            "type": "object",
            "properties": {
                "source_id": {"type": "string"},
                "sql": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_SQL_CHARACTERS,
                },
                "parameters": {
                    "type": "array",
                    "maxItems": MAX_SQL_PARAMETERS,
                },
            },
            "required": ["source_id", "sql"],
            "additionalProperties": False,
        },
        output_kind=output_kind,
        output_schema=_query_output_schema(),
        executor_id=executor_id,
        access_mode=AccessMode.READ,
    )
    view = ToolView(
        name=tool_name,
        capability_id=capability.id,
        description=capability.description,
        applicability=ToolApplicability(
            source_adapter_ids=(adapter_id,), minimum_active_sources=1
        ),
    )
    return ExtensionDeclarations((capability,), (executor_id,), (view,))


def _query_output_schema() -> dict[str, object]:
    names = (
        "byte_limit",
        "canonical_sql",
        "columns",
        "resource_ids",
        "resource_revisions",
        "returned_rows",
        "row_limit",
        "rows",
        "source_id",
        "source_revision",
        "sql_fingerprint",
        "total_rows",
        "truncated",
        "truncation_reasons",
        "trust_classification",
        "utf8_bytes",
    )
    types = {
        "byte_limit": "integer",
        "canonical_sql": "string",
        "columns": "array",
        "resource_ids": "array",
        "resource_revisions": "array",
        "returned_rows": "integer",
        "row_limit": "integer",
        "rows": "array",
        "source_id": "string",
        "source_revision": "string",
        "sql_fingerprint": "string",
        "total_rows": "integer",
        "truncated": "boolean",
        "truncation_reasons": "array",
        "trust_classification": "string",
        "utf8_bytes": "integer",
    }
    return {
        "type": "object",
        "properties": {name: {"type": types[name]} for name in names},
        "required": list(names),
        "additionalProperties": False,
    }


__all__ = [
    "POSTGRESQL_QUERY_EXECUTOR_ID",
    "PostgreSQLQueryDeclarations",
    "PostgreSQLQueryExecutor",
    "PostgreSQLReadBackend",
    "PostgreSQLReadResult",
    "SQLITE_QUERY_EXECUTOR_ID",
    "SQLiteQueryDeclarations",
    "SQLiteQueryExecutor",
    "SQLiteReadBackend",
    "SQLiteReadResult",
    "postgresql_query_declarations",
    "postgresql_query_extension_declarations",
    "sqlite_query_declarations",
    "sqlite_query_extension_declarations",
]
