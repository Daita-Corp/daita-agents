"""SQLite query capability declaration over a source-specific read backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..._json import FrozenJsonObject
from ...capabilities import (
    AccessMode,
    Capability,
    EvidenceCandidate,
    ExtensionDeclarations,
    ExecutionRequest,
    Executor,
    RiskLevel,
    ToolView,
)
from .controller import SQLITE_QUERY_CAPABILITY_ID, SQLITE_QUERY_EVIDENCE_KIND
from .results import BoundedResultProjection

_MAX_EVIDENCE_RESOURCES = 1_000
SQLITE_QUERY_EXECUTOR_ID = "data.sqlite.query.executor"


@dataclass(frozen=True, slots=True)
class SQLiteReadResult:
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
        if not self.sql_fingerprint.startswith("sha256:"):
            raise ValueError("sql_fingerprint must use sha256")
        resource_ids = tuple(self.resource_ids)
        resource_revisions = tuple(
            sorted(tuple(item) for item in self.resource_revisions)
        )
        columns = tuple(self.columns)
        if not resource_ids or any(
            not isinstance(item, str) or not item.strip() for item in resource_ids
        ):
            raise ValueError("resource_ids must contain at least one identifier")
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError("resource_ids cannot contain duplicates")
        if len(resource_ids) > _MAX_EVIDENCE_RESOURCES:
            raise ValueError("resource_ids exceed the evidence bound")
        if any(
            len(item) != 2
            or not all(isinstance(value, str) and value.strip() for value in item)
            for item in resource_revisions
        ):
            raise ValueError(
                "resource_revisions must contain identifier/revision pairs"
            )
        revision_ids = tuple(item[0] for item in resource_revisions)
        if len(revision_ids) != len(set(revision_ids)):
            raise ValueError("resource_revisions cannot contain duplicates")
        if set(revision_ids) != set(resource_ids):
            raise ValueError("resource_revisions must cover every resource_id")
        if any(
            len(revision) != 71
            or any(
                character not in "0123456789abcdef"
                for character in revision.removeprefix("sha256:")
            )
            or not revision.startswith("sha256:")
            for _, revision in resource_revisions
        ):
            raise ValueError("resource revisions must use sha256")
        if len(self.source_revision) > 1_024:
            raise ValueError("source_revision exceeds 1024 characters")
        if any(not isinstance(item, str) or not item.strip() for item in columns):
            raise ValueError("columns must contain non-empty strings")
        if len(columns) != len(set(columns)):
            raise ValueError("columns cannot contain duplicates")
        if not isinstance(self.projection, BoundedResultProjection):
            raise TypeError("projection must be a BoundedResultProjection")
        object.__setattr__(self, "resource_ids", resource_ids)
        object.__setattr__(self, "resource_revisions", resource_revisions)
        object.__setattr__(self, "columns", columns)

    def evidence_payload(self) -> FrozenJsonObject:
        projection = self.projection.to_payload().to_dict()
        return FrozenJsonObject.from_mapping(
            {
                **projection,
                "canonical_sql": self.canonical_sql,
                "columns": self.columns,
                "resource_ids": self.resource_ids,
                "resource_revisions": tuple(
                    {
                        "resource_id": resource_id,
                        "revision": revision,
                    }
                    for resource_id, revision in self.resource_revisions
                ),
                "source_id": self.source_id,
                "source_revision": self.source_revision,
                "sql_fingerprint": self.sql_fingerprint,
            }
        )


class SQLiteReadBackend(Protocol):
    async def execute_read(
        self,
        *,
        agent_id: str,
        source_id: str,
        sql: str,
        parameters: tuple[object, ...],
        max_rows: int,
        max_bytes: int,
    ) -> SQLiteReadResult: ...


@dataclass(frozen=True, slots=True)
class SQLiteQueryDeclarations:
    capabilities: tuple[Capability, ...]
    executors: tuple[Executor, ...]
    tool_views: tuple[ToolView, ...]


class SQLiteQueryExecutor:
    executor_id = SQLITE_QUERY_EXECUTOR_ID

    def __init__(
        self,
        agent_id: str,
        backend: SQLiteReadBackend,
        *,
        max_rows: int = 100,
        max_bytes: int = 65_536,
    ) -> None:
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("agent_id must be a non-empty string")
        if not callable(getattr(backend, "execute_read", None)):
            raise TypeError("backend must provide execute_read")
        for value, name in ((max_rows, "max_rows"), (max_bytes, "max_bytes")):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self._agent_id = agent_id
        self._backend = backend
        self._max_rows = max_rows
        self._max_bytes = max_bytes

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
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
            raise ValueError("SQLite backend returned a different source")
        return EvidenceCandidate(
            kind=SQLITE_QUERY_EVIDENCE_KIND,
            schema_version=1,
            payload=result.evidence_payload(),
        )


def sqlite_query_declarations(
    agent_id: str,
    backend: SQLiteReadBackend,
) -> SQLiteQueryDeclarations:
    executor = SQLiteQueryExecutor(agent_id, backend)
    extension = sqlite_query_extension_declarations()
    return SQLiteQueryDeclarations(
        capabilities=extension.capabilities,
        executors=(executor,),
        tool_views=extension.tool_views,
    )


def sqlite_query_extension_declarations() -> ExtensionDeclarations:
    """Advertise the stable SQLite query contract independently of one source."""

    capability = Capability(
        id=SQLITE_QUERY_CAPABILITY_ID,
        owner="data",
        description="Run one validated, read-only, bounded SQLite query.",
        input_schema={
            "type": "object",
            "properties": {
                "source_id": {"type": "string"},
                "sql": {"type": "string"},
                "parameters": {"type": "array"},
            },
            "required": ["source_id", "sql"],
            "additionalProperties": False,
        },
        output_evidence_kind=SQLITE_QUERY_EVIDENCE_KIND,
        output_schema_version=1,
        output_schema=_query_output_schema(),
        executor_id=SQLITE_QUERY_EXECUTOR_ID,
        access_mode=AccessMode.READ,
        risk=RiskLevel.LOW,
        side_effecting=False,
        idempotent=True,
        replay_safe=True,
    )
    view = ToolView(
        name="data_query_sqlite",
        capability_id=capability.id,
        description=capability.description,
    )
    return ExtensionDeclarations(
        capabilities=(capability,),
        executor_ids=(SQLITE_QUERY_EXECUTOR_ID,),
        tool_views=(view,),
    )


def _query_output_schema() -> dict[str, object]:
    properties = {
        "byte_limit": {"type": "integer"},
        "canonical_sql": {"type": "string"},
        "columns": {"type": "array"},
        "resource_ids": {"type": "array"},
        "resource_revisions": {"type": "array"},
        "returned_rows": {"type": "integer"},
        "row_limit": {"type": "integer"},
        "rows": {"type": "array"},
        "source_id": {"type": "string"},
        "source_revision": {"type": "string"},
        "sql_fingerprint": {"type": "string"},
        "total_rows": {"type": "integer"},
        "truncated": {"type": "boolean"},
        "truncation_reasons": {"type": "array"},
        "trust_classification": {"type": "string"},
        "utf8_bytes": {"type": "integer"},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


__all__ = [
    "SQLITE_QUERY_EXECUTOR_ID",
    "SQLiteQueryDeclarations",
    "SQLiteQueryExecutor",
    "SQLiteReadBackend",
    "SQLiteReadResult",
    "sqlite_query_declarations",
    "sqlite_query_extension_declarations",
]
