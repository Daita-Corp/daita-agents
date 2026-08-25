"""Declare and execute SQL reads and structured PostgreSQL updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..._json import FrozenJsonObject, canonical_json
from ...capabilities import (
    AccessMode,
    Capability,
    CapabilityDeclarations,
    CapabilityInputError,
    Executor,
    OperationalEffect,
    ToolDiscoveryMetadata,
    ToolExecution,
    ToolExposureClass,
    ToolOutput,
    ToolView,
)
from .controller import (
    POSTGRESQL_QUERY_CAPABILITY_ID,
    POSTGRESQL_QUERY_EVIDENCE_KIND,
    POSTGRESQL_UPDATE_CAPABILITY_ID,
    POSTGRESQL_UPDATE_EVIDENCE_KIND,
    POSTGRESQL_UPDATE_PREVIEW_CAPABILITY_ID,
    POSTGRESQL_UPDATE_PREVIEW_EVIDENCE_KIND,
    SQLITE_QUERY_CAPABILITY_ID,
    SQLITE_QUERY_EVIDENCE_KIND,
)
from .results import BoundedResultProjection
from .sql import (
    MAX_SQL_CHARACTERS,
    MAX_SQL_PARAMETERS,
    PostgreSQLUpdateCell,
    PostgreSQLUpdateCommand,
    PostgreSQLUpdateFilter,
    PostgreSQLUpdateIntent,
)

SQLITE_QUERY_EXECUTOR_ID = "data.sqlite.query.executor"
POSTGRESQL_QUERY_EXECUTOR_ID = "data.postgresql.query.executor"
POSTGRESQL_UPDATE_PREVIEW_EXECUTOR_ID = "data.postgresql.update_impact.executor"
POSTGRESQL_UPDATE_PREVIEW_TOOL_NAME = "data_preview_postgresql_update"
POSTGRESQL_UPDATE_EXECUTOR_ID = "data.postgresql.update.executor"
POSTGRESQL_UPDATE_TOOL_NAME = "data_update_postgresql"
_MAX_PREVIEW_OUTPUT_BYTES = 256 * 1_024


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


@dataclass(frozen=True, slots=True)
class PostgreSQLPreviewFingerprint:
    intent_sha256: str
    target_set_sha256: str
    statement_sha256: str
    preview_fingerprint: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.intent_sha256, "intent_sha256"),
            (self.target_set_sha256, "target_set_sha256"),
            (self.statement_sha256, "statement_sha256"),
            (self.preview_fingerprint, "preview_fingerprint"),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 71
                or not value.startswith("sha256:")
                or any(character not in "0123456789abcdef" for character in value[7:])
            ):
                raise ValueError(f"{name} must be a canonical sha256 hash")


@dataclass(frozen=True, slots=True)
class PostgreSQLUpdatePreviewChecks:
    compile_only: str = "passed"
    target_set_fingerprinted: bool = True
    row_level_security: bool = False
    user_triggers: bool = False
    rewrite_rules: bool = False

    def __post_init__(self) -> None:
        if self.compile_only != "passed":
            raise ValueError("preview compile_only must be passed")
        for name in (
            "target_set_fingerprinted",
            "row_level_security",
            "user_triggers",
            "rewrite_rules",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"preview {name} must be boolean")
        if (
            not self.target_set_fingerprinted
            or self.row_level_security
            or self.user_triggers
            or self.rewrite_rules
        ):
            raise ValueError("preview checks must represent admitted read-only state")

    def to_payload(self) -> dict[str, object]:
        return {
            "compile_only": self.compile_only,
            "target_set_fingerprinted": self.target_set_fingerprinted,
            "row_level_security": self.row_level_security,
            "user_triggers": self.user_triggers,
            "rewrite_rules": self.rewrite_rules,
        }


@dataclass(frozen=True, slots=True)
class PostgreSQLUpdateSample:
    primary_key: tuple[PostgreSQLUpdateCell, ...]
    before: tuple[PostgreSQLUpdateCell, ...]
    after: tuple[PostgreSQLUpdateCell, ...]

    def __post_init__(self) -> None:
        primary_key = tuple(self.primary_key)
        before = tuple(self.before)
        after = tuple(self.after)
        if not primary_key or any(
            not isinstance(item, PostgreSQLUpdateCell) for item in primary_key
        ):
            raise ValueError("update sample requires a primary key")
        if (
            not before
            or any(not isinstance(item, PostgreSQLUpdateCell) for item in before)
            or any(not isinstance(item, PostgreSQLUpdateCell) for item in after)
            or tuple(item.column for item in before)
            != tuple(item.column for item in after)
        ):
            raise ValueError("update sample before and after fields must agree")
        object.__setattr__(self, "primary_key", primary_key)
        object.__setattr__(self, "before", before)
        object.__setattr__(self, "after", after)

    def to_payload(self) -> dict[str, object]:
        return {
            "primary_key": tuple(item.to_payload() for item in self.primary_key),
            "before": tuple(item.to_payload() for item in self.before),
            "after": tuple(item.to_payload() for item in self.after),
        }


@dataclass(frozen=True, slots=True)
class PostgreSQLUpdatePreview:
    source_id: str
    resource_id: str
    resource_name: str
    source_revision: str
    resource_revision: str
    where: tuple[PostgreSQLUpdateFilter, ...]
    assignments: tuple[PostgreSQLUpdateCell, ...]
    matched_rows: int
    samples: tuple[PostgreSQLUpdateSample, ...]
    fingerprint: PostgreSQLPreviewFingerprint
    checks: PostgreSQLUpdatePreviewChecks
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for text_value, text_name in (
            (self.source_id, "preview source_id"),
            (self.resource_id, "preview resource_id"),
            (self.resource_name, "preview resource_name"),
            (self.source_revision, "preview source_revision"),
            (self.resource_revision, "preview resource_revision"),
        ):
            if not isinstance(text_value, str) or not text_value:
                raise ValueError(f"{text_name} must be non-empty text")
        where = tuple(self.where)
        assignments = tuple(self.assignments)
        samples = tuple(self.samples)
        if not where or any(
            not isinstance(item, PostgreSQLUpdateFilter) for item in where
        ):
            raise TypeError("preview where must contain update filters")
        if not assignments or any(
            not isinstance(item, PostgreSQLUpdateCell) for item in assignments
        ):
            raise TypeError("preview assignments must contain update cells")
        if (
            not isinstance(self.matched_rows, int)
            or isinstance(self.matched_rows, bool)
            or self.matched_rows < 0
        ):
            raise ValueError("preview matched_rows must be a non-negative integer")
        if len(samples) > 5 or any(
            not isinstance(item, PostgreSQLUpdateSample) for item in samples
        ):
            raise ValueError("preview samples must be bounded update samples")
        if len(samples) > self.matched_rows:
            raise ValueError("preview samples cannot exceed matched rows")
        object.__setattr__(self, "where", where)
        object.__setattr__(self, "assignments", assignments)
        object.__setattr__(self, "samples", samples)
        if not isinstance(self.fingerprint, PostgreSQLPreviewFingerprint):
            raise TypeError("preview fingerprint must be PostgreSQLPreviewFingerprint")
        if not isinstance(self.checks, PostgreSQLUpdatePreviewChecks):
            raise TypeError("preview checks must be PostgreSQLUpdatePreviewChecks")
        warnings = tuple(self.warnings)
        if len(warnings) > 8 or any(
            not isinstance(item, str) or not item or len(item) > 160
            for item in warnings
        ):
            raise ValueError("preview warnings must be bounded text")
        object.__setattr__(self, "warnings", warnings)
        if (
            len(canonical_json(self.tool_data()).encode("utf-8"))
            > _MAX_PREVIEW_OUTPUT_BYTES
        ):
            raise ValueError("preview output exceeds its fixed byte bound")

    def tool_data(self) -> FrozenJsonObject:
        fingerprint = self.fingerprint
        return FrozenJsonObject.from_mapping(
            {
                "source_id": self.source_id,
                "resource_id": self.resource_id,
                "resource_name": self.resource_name,
                "source_revision": self.source_revision,
                "resource_revision": self.resource_revision,
                "where": tuple(item.to_payload() for item in self.where),
                "assignments": tuple(item.to_payload() for item in self.assignments),
                "matched_rows": self.matched_rows,
                "samples": tuple(item.to_payload() for item in self.samples),
                "target_set_sha256": fingerprint.target_set_sha256,
                "statement_sha256": fingerprint.statement_sha256,
                "preview_fingerprint": fingerprint.preview_fingerprint,
                "checks": self.checks.to_payload(),
                "warnings": self.warnings,
                "trust_classification": "untrusted_data",
            }
        )


@dataclass(frozen=True, slots=True)
class PostgreSQLUpdateResult:
    """One positively acknowledged committed PostgreSQL update plan."""

    receipt_id: str
    source_id: str
    resource_id: str
    source_revision: str
    resource_revision: str
    preview_fingerprint: str
    intent_sha256: str
    target_set_sha256: str
    affected_rows: int
    committed_at: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.receipt_id, "update result receipt_id"),
            (self.source_id, "update result source_id"),
            (self.resource_id, "update result resource_id"),
            (self.source_revision, "update result source_revision"),
            (self.resource_revision, "update result resource_revision"),
            (self.committed_at, "update result committed_at"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty text")
        for value, name in (
            (self.preview_fingerprint, "preview_fingerprint"),
            (self.intent_sha256, "intent_sha256"),
            (self.target_set_sha256, "target_set_sha256"),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 71
                or not value.startswith("sha256:")
            ):
                raise ValueError(f"update result {name} must be a sha256 hash")
        if (
            not isinstance(self.affected_rows, int)
            or isinstance(self.affected_rows, bool)
            or self.affected_rows < 1
        ):
            raise ValueError("update result affected_rows must be positive")

    def tool_data(self) -> FrozenJsonObject:
        return FrozenJsonObject.from_mapping(
            {
                "receipt_id": self.receipt_id,
                "outcome": "committed",
                "source_id": self.source_id,
                "resource_id": self.resource_id,
                "source_revision": self.source_revision,
                "resource_revision": self.resource_revision,
                "preview_fingerprint": self.preview_fingerprint,
                "intent_sha256": self.intent_sha256,
                "target_set_sha256": self.target_set_sha256,
                "affected_rows": self.affected_rows,
                "committed_at": self.committed_at,
                "trust_classification": "untrusted_data",
            }
        )


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


class PostgreSQLUpdatePreviewBackend(Protocol):
    async def preview_update(
        self,
        *,
        agent_id: str,
        intent: PostgreSQLUpdateIntent,
    ) -> PostgreSQLUpdatePreview: ...


class PostgreSQLUpdateBackend(PostgreSQLUpdatePreviewBackend, Protocol):
    async def execute_update(
        self,
        *,
        agent_id: str,
        execution: ToolExecution,
        command: PostgreSQLUpdateCommand,
    ) -> PostgreSQLUpdateResult: ...


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


@dataclass(frozen=True, slots=True)
class PostgreSQLUpdatePreviewDeclarations:
    capabilities: tuple[Capability, ...]
    executors: tuple[Executor, ...]
    tool_views: tuple[ToolView, ...]


@dataclass(frozen=True, slots=True)
class PostgreSQLUpdateDeclarations:
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


class PostgreSQLUpdatePreviewExecutor:
    executor_id = POSTGRESQL_UPDATE_PREVIEW_EXECUTOR_ID

    def __init__(
        self,
        agent_id: str,
        backend: PostgreSQLUpdatePreviewBackend,
    ) -> None:
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("preview executor agent_id must be non-empty text")
        if not callable(getattr(backend, "preview_update", None)):
            raise TypeError("preview backend must provide preview_update")
        self._agent_id = agent_id
        self._backend = backend

    async def execute(self, request: ToolExecution) -> ToolOutput:
        intent = PostgreSQLUpdateIntent.from_mapping(request.arguments)
        result = await self._backend.preview_update(
            agent_id=self._agent_id,
            intent=intent,
        )
        if (
            result.source_id != intent.source_id
            or result.resource_id != intent.resource_id
            or result.where != intent.where
        ):
            raise ValueError("preview backend returned a different update identity")
        return ToolOutput(
            kind=POSTGRESQL_UPDATE_PREVIEW_EVIDENCE_KIND,
            data=result.tool_data(),
        )


class PostgreSQLUpdateExecutor:
    executor_id = POSTGRESQL_UPDATE_EXECUTOR_ID

    def __init__(self, agent_id: str, backend: PostgreSQLUpdateBackend) -> None:
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("update executor agent_id must be non-empty text")
        if not callable(getattr(backend, "preview_update", None)) or not callable(
            getattr(backend, "execute_update", None)
        ):
            raise TypeError(
                "update backend must provide preview_update and execute_update"
            )
        self._agent_id = agent_id
        self._backend = backend

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        command = PostgreSQLUpdateCommand.from_mapping(request.arguments)
        preview = await self._backend.preview_update(
            agent_id=self._agent_id,
            intent=command.intent,
        )
        if preview.matched_rows == 0:
            raise CapabilityInputError(
                "write_target_not_found",
                "The previewed target selection does not currently match any rows.",
            )
        if preview.matched_rows != command.expected_affected_rows:
            raise CapabilityInputError(
                "write_state_changed",
                "The exact target count differs from the approved update plan.",
            )
        if preview.fingerprint.preview_fingerprint != command.preview_fingerprint:
            raise CapabilityInputError(
                "write_preview_stale",
                "The supplied preview fingerprint is not the exact current preview.",
            )
        return FrozenJsonObject.from_mapping(
            {
                "intent_sha256": preview.fingerprint.intent_sha256,
                "preview_fingerprint": preview.fingerprint.preview_fingerprint,
                "resource_revision": preview.resource_revision,
                "target_set_sha256": preview.fingerprint.target_set_sha256,
                "source_revision": preview.source_revision,
                "statement_sha256": preview.fingerprint.statement_sha256,
            }
        )

    async def execute(self, request: ToolExecution) -> ToolOutput:
        command = PostgreSQLUpdateCommand.from_mapping(request.arguments)
        result = await self._backend.execute_update(
            agent_id=self._agent_id,
            execution=request,
            command=command,
        )
        if (
            result.source_id != command.intent.source_id
            or result.resource_id != command.intent.resource_id
            or result.preview_fingerprint != command.preview_fingerprint
        ):
            raise ValueError("update backend returned a different update identity")
        return ToolOutput(kind=POSTGRESQL_UPDATE_EVIDENCE_KIND, data=result.tool_data())


def sqlite_query_declarations(
    agent_id: str, backend: SQLiteReadBackend
) -> SQLiteQueryDeclarations:
    executor = SQLiteQueryExecutor(agent_id, backend)
    declarations = sqlite_query_capability_declarations()
    return SQLiteQueryDeclarations(
        declarations.capabilities, (executor,), declarations.tool_views
    )


def postgresql_query_declarations(
    agent_id: str, backend: PostgreSQLReadBackend
) -> PostgreSQLQueryDeclarations:
    executor = PostgreSQLQueryExecutor(agent_id, backend)
    declarations = postgresql_query_capability_declarations()
    return PostgreSQLQueryDeclarations(
        declarations.capabilities, (executor,), declarations.tool_views
    )


def postgresql_update_preview_declarations(
    agent_id: str,
    backend: PostgreSQLUpdatePreviewBackend,
) -> PostgreSQLUpdatePreviewDeclarations:
    executor = PostgreSQLUpdatePreviewExecutor(agent_id, backend)
    declarations = postgresql_update_preview_capability_declarations()
    return PostgreSQLUpdatePreviewDeclarations(
        declarations.capabilities,
        (executor,),
        declarations.tool_views,
    )


def postgresql_update_declarations(
    agent_id: str,
    backend: PostgreSQLUpdateBackend,
) -> PostgreSQLUpdateDeclarations:
    executor = PostgreSQLUpdateExecutor(agent_id, backend)
    declarations = postgresql_update_capability_declarations()
    return PostgreSQLUpdateDeclarations(
        declarations.capabilities,
        (executor,),
        declarations.tool_views,
    )


def sqlite_query_capability_declarations() -> CapabilityDeclarations:
    return _query_declarations(
        SQLITE_QUERY_CAPABILITY_ID,
        SQLITE_QUERY_EVIDENCE_KIND,
        SQLITE_QUERY_EXECUTOR_ID,
        "SQLite",
        "data_query_sqlite",
        ToolDiscoveryMetadata(
            summary="Run one bounded validated read-only SQLite query.",
            when_to_use="Use after catalog_schema provides the exact SQLite structure.",
            keywords=("data", "query", "sqlite", "sql"),
            exposure_class=ToolExposureClass.CORE,
            eager_priority=950,
        ),
    )


def postgresql_query_capability_declarations() -> CapabilityDeclarations:
    return _query_declarations(
        POSTGRESQL_QUERY_CAPABILITY_ID,
        POSTGRESQL_QUERY_EVIDENCE_KIND,
        POSTGRESQL_QUERY_EXECUTOR_ID,
        "PostgreSQL",
        "data_query_postgresql",
        ToolDiscoveryMetadata(
            summary="Run one bounded validated read-only PostgreSQL query.",
            when_to_use="Use after catalog_schema provides the exact PostgreSQL structure.",
            keywords=("data", "query", "postgresql", "sql"),
            exposure_class=ToolExposureClass.CORE,
            eager_priority=950,
        ),
    )


def postgresql_update_preview_capability_declarations() -> CapabilityDeclarations:
    cell_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "column": {"type": "string", "minLength": 1, "maxLength": 256},
            "value": {},
        },
        "required": ["column", "value"],
        "additionalProperties": False,
    }
    filter_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "column": {"type": "string", "minLength": 1, "maxLength": 256},
            "operator": {
                "type": "string",
                "enum": [
                    "eq",
                    "ne",
                    "lt",
                    "lte",
                    "gt",
                    "gte",
                    "in",
                    "not_in",
                    "is_null",
                    "is_not_null",
                ],
            },
            "value": {},
        },
        "required": ["column", "operator", "value"],
        "additionalProperties": False,
    }
    capability = Capability(
        id=POSTGRESQL_UPDATE_PREVIEW_CAPABILITY_ID,
        description=(
            "Validate and preview one structured PostgreSQL update over an exact "
            "catalog-scoped target set without changing the database. When the "
            "user requested approval or execution, pass the exact successful "
            "preview immediately to data_update_postgresql; preview alone does "
            "not request approval."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "pattern": r"^source:sha256:[0-9a-f]{64}$",
                },
                "resource_id": {
                    "type": "string",
                    "pattern": r"^catalog-resource:sha256:[0-9a-f]{64}$",
                },
                "where": {
                    "type": "array",
                    "minItems": 1,
                    "items": filter_schema,
                },
                "assignments": {
                    "type": "array",
                    "minItems": 1,
                    "items": cell_schema,
                },
            },
            "required": ["source_id", "resource_id", "where", "assignments"],
            "additionalProperties": False,
        },
        output_kind=POSTGRESQL_UPDATE_PREVIEW_EVIDENCE_KIND,
        output_schema=_postgresql_update_preview_output_schema(),
        executor_id=POSTGRESQL_UPDATE_PREVIEW_EXECUTOR_ID,
        access_mode=AccessMode.READ,
    )
    view = ToolView(
        name=POSTGRESQL_UPDATE_PREVIEW_TOOL_NAME,
        capability_id=capability.id,
        description=capability.description,
        discovery=ToolDiscoveryMetadata(
            summary="Preview the exact target set for a structured PostgreSQL update.",
            when_to_use="Use before requesting approval for any supported PostgreSQL update.",
            keywords=("data", "postgresql", "update", "preview"),
            exposure_class=ToolExposureClass.CORE,
            eager_priority=940,
        ),
    )
    return CapabilityDeclarations(
        domain_owner_id="data",
        capabilities=(capability,),
        executor_ids=(capability.executor_id,),
        tool_views=(view,),
    )


def postgresql_update_capability_declarations() -> CapabilityDeclarations:
    cell_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "column": {"type": "string", "minLength": 1, "maxLength": 256},
            "value": {},
        },
        "required": ["column", "value"],
        "additionalProperties": False,
    }
    filter_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "column": {"type": "string", "minLength": 1, "maxLength": 256},
            "operator": {
                "type": "string",
                "enum": [
                    "eq",
                    "ne",
                    "lt",
                    "lte",
                    "gt",
                    "gte",
                    "in",
                    "not_in",
                    "is_null",
                    "is_not_null",
                ],
            },
            "value": {},
        },
        "required": ["column", "operator", "value"],
        "additionalProperties": False,
    }
    capability = Capability(
        id=POSTGRESQL_UPDATE_CAPABILITY_ID,
        description=(
            "Submit one exact previewed PostgreSQL update to runtime approval. "
            "Calling this tool opens the approval interaction and applies the "
            "update exactly once only if approved and revalidation succeeds."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "pattern": r"^source:sha256:[0-9a-f]{64}$",
                },
                "resource_id": {
                    "type": "string",
                    "pattern": r"^catalog-resource:sha256:[0-9a-f]{64}$",
                },
                "where": {
                    "type": "array",
                    "minItems": 1,
                    "items": filter_schema,
                },
                "assignments": {
                    "type": "array",
                    "minItems": 1,
                    "items": cell_schema,
                },
                "preview_fingerprint": {
                    "type": "string",
                    "pattern": r"^sha256:[0-9a-f]{64}$",
                },
                "expected_affected_rows": {
                    "type": "integer",
                    "minimum": 1,
                },
            },
            "required": [
                "source_id",
                "resource_id",
                "where",
                "assignments",
                "preview_fingerprint",
                "expected_affected_rows",
            ],
            "additionalProperties": False,
        },
        output_kind=POSTGRESQL_UPDATE_EVIDENCE_KIND,
        output_schema=_postgresql_update_output_schema(),
        executor_id=POSTGRESQL_UPDATE_EXECUTOR_ID,
        access_mode=AccessMode.WRITE,
        operational_effect=OperationalEffect.MUTATE_DATA,
    )
    view = ToolView(
        name=POSTGRESQL_UPDATE_TOOL_NAME,
        capability_id=capability.id,
        description=capability.description,
        discovery=ToolDiscoveryMetadata(
            summary="Request approval and execute one exact previewed PostgreSQL update.",
            when_to_use="Use only after a successful exact update preview.",
            keywords=("data", "postgresql", "update", "approval"),
            exposure_class=ToolExposureClass.CORE,
            eager_priority=930,
        ),
    )
    return CapabilityDeclarations(
        domain_owner_id="data",
        capabilities=(capability,),
        executor_ids=(capability.executor_id,),
        tool_views=(view,),
    )


def _query_declarations(
    capability_id: str,
    output_kind: str,
    executor_id: str,
    adapter_name: str,
    tool_name: str,
    discovery: ToolDiscoveryMetadata,
) -> CapabilityDeclarations:
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
        discovery=discovery,
    )
    return CapabilityDeclarations(
        domain_owner_id="data",
        capabilities=(capability,),
        executor_ids=(executor_id,),
        tool_views=(view,),
    )


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


def _postgresql_update_preview_output_schema() -> dict[str, object]:
    cell_schema = {
        "type": "object",
        "properties": {"column": {"type": "string"}, "value": {}},
        "required": ["column", "value"],
        "additionalProperties": False,
    }
    names = (
        "source_id",
        "resource_id",
        "resource_name",
        "source_revision",
        "resource_revision",
        "where",
        "assignments",
        "matched_rows",
        "samples",
        "target_set_sha256",
        "statement_sha256",
        "preview_fingerprint",
        "checks",
        "warnings",
        "trust_classification",
    )
    hash_rule = {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"}
    return {
        "type": "object",
        "properties": {
            "source_id": {"type": "string"},
            "resource_id": {"type": "string"},
            "resource_name": {"type": "string"},
            "source_revision": {"type": "string"},
            "resource_revision": {"type": "string"},
            "where": {"type": "array", "items": {"type": "object"}},
            "assignments": {"type": "array", "items": cell_schema},
            "matched_rows": {"type": "integer", "minimum": 0},
            "samples": {"type": "array", "items": {"type": "object"}},
            "target_set_sha256": hash_rule,
            "statement_sha256": hash_rule,
            "preview_fingerprint": hash_rule,
            "checks": {
                "type": "object",
                "properties": {
                    "compile_only": {"type": "string", "enum": ["passed"]},
                    "target_set_fingerprinted": {"type": "boolean"},
                    "row_level_security": {"type": "boolean"},
                    "user_triggers": {"type": "boolean"},
                    "rewrite_rules": {"type": "boolean"},
                },
                "required": [
                    "compile_only",
                    "target_set_fingerprinted",
                    "row_level_security",
                    "user_triggers",
                    "rewrite_rules",
                ],
                "additionalProperties": False,
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
            "trust_classification": {
                "type": "string",
                "enum": ["untrusted_data"],
            },
        },
        "required": list(names),
        "additionalProperties": False,
    }


def _postgresql_update_output_schema() -> dict[str, object]:
    hash_rule = {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"}
    names = (
        "receipt_id",
        "outcome",
        "source_id",
        "resource_id",
        "source_revision",
        "resource_revision",
        "preview_fingerprint",
        "intent_sha256",
        "target_set_sha256",
        "affected_rows",
        "committed_at",
        "trust_classification",
    )
    return {
        "type": "object",
        "properties": {
            "receipt_id": {
                "type": "string",
                "pattern": r"^database-write-receipt:sha256:[0-9a-f]{64}$",
            },
            "outcome": {"type": "string", "enum": ["committed"]},
            "source_id": {"type": "string"},
            "resource_id": {"type": "string"},
            "source_revision": {"type": "string"},
            "resource_revision": {"type": "string"},
            "preview_fingerprint": hash_rule,
            "intent_sha256": hash_rule,
            "target_set_sha256": hash_rule,
            "affected_rows": {"type": "integer", "minimum": 1},
            "committed_at": {"type": "string"},
            "trust_classification": {
                "type": "string",
                "enum": ["untrusted_data"],
            },
        },
        "required": list(names),
        "additionalProperties": False,
    }


__all__ = [
    "POSTGRESQL_QUERY_EXECUTOR_ID",
    "POSTGRESQL_UPDATE_PREVIEW_EXECUTOR_ID",
    "POSTGRESQL_UPDATE_PREVIEW_TOOL_NAME",
    "POSTGRESQL_UPDATE_EXECUTOR_ID",
    "POSTGRESQL_UPDATE_TOOL_NAME",
    "PostgreSQLQueryDeclarations",
    "PostgreSQLQueryExecutor",
    "PostgreSQLReadBackend",
    "PostgreSQLReadResult",
    "PostgreSQLPreviewFingerprint",
    "PostgreSQLUpdatePreview",
    "PostgreSQLUpdatePreviewBackend",
    "PostgreSQLUpdatePreviewChecks",
    "PostgreSQLUpdatePreviewDeclarations",
    "PostgreSQLUpdatePreviewExecutor",
    "PostgreSQLUpdateBackend",
    "PostgreSQLUpdateDeclarations",
    "PostgreSQLUpdateExecutor",
    "PostgreSQLUpdateResult",
    "PostgreSQLUpdateSample",
    "SQLITE_QUERY_EXECUTOR_ID",
    "SQLiteQueryDeclarations",
    "SQLiteQueryExecutor",
    "SQLiteReadBackend",
    "SQLiteReadResult",
    "postgresql_query_declarations",
    "postgresql_query_capability_declarations",
    "postgresql_update_preview_declarations",
    "postgresql_update_preview_capability_declarations",
    "postgresql_update_declarations",
    "postgresql_update_capability_declarations",
    "sqlite_query_declarations",
    "sqlite_query_capability_declarations",
]
