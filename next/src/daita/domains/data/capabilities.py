"""SQLite query capability declaration over a source-specific read backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..._json import FrozenJsonObject
from ...capabilities import (
    AccessMode,
    Capability,
    EvidenceCandidate,
    ExecutorKnownNoEffectError,
    ExtensionDeclarations,
    ExecutionRequest,
    Executor,
    RiskLevel,
    ToolView,
)
from .controller import SQLITE_QUERY_CAPABILITY_ID, SQLITE_QUERY_EVIDENCE_KIND
from .controller import (
    SQLITE_UPDATE_CAPABILITY_ID,
    SQLITE_UPDATE_EVIDENCE_KIND,
    SQLITE_UPDATE_IMPACT_CAPABILITY_ID,
    SQLITE_UPDATE_IMPACT_EVIDENCE_KIND,
    SQLITE_UPDATE_IMPACT_TOOL_NAME,
    SQLITE_UPDATE_TOOL_NAME,
)
from .results import BoundedResultProjection
from .sql import sqlite_identifier_key

_MAX_EVIDENCE_RESOURCES = 1_000
SQLITE_QUERY_EXECUTOR_ID = "data.sqlite.query.executor"
SQLITE_UPDATE_IMPACT_EXECUTOR_ID = "data.sqlite.update_impact.executor"
SQLITE_UPDATE_EXECUTOR_ID = "data.sqlite.update.executor"


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


@dataclass(frozen=True, slots=True)
class SQLiteUpdateImpactResult:
    source_id: str
    resource_id: str
    resource_revision: str
    source_revision: str
    key_column: str
    target_column: str
    recipe_fingerprint: str
    matched_rows: int
    eligible_rows: int
    maximum_rows: int

    def __post_init__(self) -> None:
        _validate_update_provenance(
            source_id=self.source_id,
            resource_id=self.resource_id,
            resource_revision=self.resource_revision,
            source_revision=self.source_revision,
            key_column=self.key_column,
            target_column=self.target_column,
            recipe_fingerprint=self.recipe_fingerprint,
        )
        for value, field_name in (
            (self.matched_rows, "matched_rows"),
            (self.eligible_rows, "eligible_rows"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if type(self.maximum_rows) is not int or self.maximum_rows != 1:
            raise ValueError("controlled update maximum_rows must equal one")
        if self.eligible_rows > self.matched_rows:
            raise ValueError("eligible_rows cannot exceed matched_rows")

    def evidence_payload(self) -> FrozenJsonObject:
        return FrozenJsonObject.from_mapping(
            {
                "eligible_rows": self.eligible_rows,
                "key_column": self.key_column,
                "matched_rows": self.matched_rows,
                "maximum_rows": self.maximum_rows,
                "recipe_fingerprint": self.recipe_fingerprint,
                "resource_id": self.resource_id,
                "resource_revision": self.resource_revision,
                "source_id": self.source_id,
                "source_revision": self.source_revision,
                "target_column": self.target_column,
            }
        )


@dataclass(frozen=True, slots=True)
class SQLiteUpdateResult:
    source_id: str
    resource_id: str
    resource_revision: str
    source_revision: str
    key_column: str
    target_column: str
    recipe_fingerprint: str
    impact_evidence_id: str
    affected_rows: int
    maximum_rows: int

    def __post_init__(self) -> None:
        _validate_update_provenance(
            source_id=self.source_id,
            resource_id=self.resource_id,
            resource_revision=self.resource_revision,
            source_revision=self.source_revision,
            key_column=self.key_column,
            target_column=self.target_column,
            recipe_fingerprint=self.recipe_fingerprint,
        )
        if (
            not isinstance(self.impact_evidence_id, str)
            or not self.impact_evidence_id.strip()
            or len(self.impact_evidence_id) > 512
        ):
            raise ValueError("impact_evidence_id must be a bounded non-empty string")
        if type(self.maximum_rows) is not int or self.maximum_rows != 1:
            raise ValueError("controlled update maximum_rows must equal one")
        if type(self.affected_rows) is not int or self.affected_rows != 1:
            raise ValueError(
                "a successful controlled update must affect exactly one row"
            )

    def evidence_payload(self) -> FrozenJsonObject:
        return FrozenJsonObject.from_mapping(
            {
                "affected_rows": self.affected_rows,
                "impact_evidence_id": self.impact_evidence_id,
                "key_column": self.key_column,
                "maximum_rows": self.maximum_rows,
                "recipe_fingerprint": self.recipe_fingerprint,
                "resource_id": self.resource_id,
                "resource_revision": self.resource_revision,
                "source_id": self.source_id,
                "source_revision": self.source_revision,
                "target_column": self.target_column,
            }
        )


def _validate_update_provenance(
    *,
    source_id: str,
    resource_id: str,
    resource_revision: str,
    source_revision: str,
    key_column: str,
    target_column: str,
    recipe_fingerprint: str,
) -> None:
    for value, field_name, maximum in (
        (source_id, "source_id", 512),
        (resource_id, "resource_id", 512),
        (source_revision, "source_revision", 1_024),
        (key_column, "key_column", 256),
        (target_column, "target_column", 256),
    ):
        if not isinstance(value, str) or not value.strip() or len(value) > maximum:
            raise ValueError(f"{field_name} must be a bounded non-empty string")
    for value, field_name in (
        (resource_revision, "resource_revision"),
        (recipe_fingerprint, "recipe_fingerprint"),
    ):
        digest = value.removeprefix("sha256:") if isinstance(value, str) else ""
        if (
            not isinstance(value, str)
            or not value.startswith("sha256:")
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{field_name} must be a canonical sha256 hash")
    if sqlite_identifier_key(key_column) == sqlite_identifier_key(target_column):
        raise ValueError("key_column and target_column must be distinct")


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


class SQLiteUpdateKnownNoEffectError(RuntimeError):
    """Backend failure known to have left no SQLite write effect."""

    def __init__(self, code: str, message: str) -> None:
        if not isinstance(code, str) or not code.strip():
            raise ValueError("known-no-effect code must be a non-empty string")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("known-no-effect message must be a non-empty string")
        self.code = code
        super().__init__(message)


class SQLiteUpdateBackend(Protocol):
    async def execute_update_impact(
        self,
        *,
        agent_id: str,
        source_id: str,
        resource_id: str,
        key_column: str,
        key_value: str,
        target_column: str,
        expected_value: str,
        new_value: str,
        maximum_rows: int,
    ) -> SQLiteUpdateImpactResult: ...

    async def execute_update(
        self,
        *,
        agent_id: str,
        operation_id: str,
        source_id: str,
        resource_id: str,
        key_column: str,
        key_value: str,
        target_column: str,
        expected_value: str,
        new_value: str,
        impact_evidence_id: str,
        maximum_rows: int,
    ) -> SQLiteUpdateResult: ...


@dataclass(frozen=True, slots=True)
class SQLiteQueryDeclarations:
    capabilities: tuple[Capability, ...]
    executors: tuple[Executor, ...]
    tool_views: tuple[ToolView, ...]


@dataclass(frozen=True, slots=True)
class SQLiteUpdateDeclarations:
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


class SQLiteUpdateImpactExecutor:
    executor_id = SQLITE_UPDATE_IMPACT_EXECUTOR_ID

    def __init__(
        self,
        agent_id: str,
        backend: SQLiteUpdateBackend,
        *,
        maximum_rows: int = 1,
    ) -> None:
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("agent_id must be a non-empty string")
        if not callable(getattr(backend, "execute_update_impact", None)):
            raise TypeError("backend must provide execute_update_impact")
        if maximum_rows != 1:
            raise ValueError("controlled SQLite updates require maximum_rows=1")
        self._agent_id = agent_id
        self._backend = backend
        self._maximum_rows = maximum_rows

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        arguments = _update_arguments(request)
        result = await self._backend.execute_update_impact(
            agent_id=self._agent_id,
            **arguments,
            maximum_rows=self._maximum_rows,
        )
        if (
            result.source_id != arguments["source_id"]
            or result.resource_id != arguments["resource_id"]
            or sqlite_identifier_key(result.key_column)
            != sqlite_identifier_key(arguments["key_column"])
            or sqlite_identifier_key(result.target_column)
            != sqlite_identifier_key(arguments["target_column"])
            or result.maximum_rows != self._maximum_rows
        ):
            raise ValueError("SQLite update-impact backend returned a different scope")
        return EvidenceCandidate(
            kind=SQLITE_UPDATE_IMPACT_EVIDENCE_KIND,
            schema_version=1,
            payload=result.evidence_payload(),
        )


class SQLiteUpdateExecutor:
    executor_id = SQLITE_UPDATE_EXECUTOR_ID

    def __init__(
        self,
        agent_id: str,
        backend: SQLiteUpdateBackend,
        *,
        maximum_rows: int = 1,
    ) -> None:
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("agent_id must be a non-empty string")
        if not callable(getattr(backend, "execute_update", None)):
            raise TypeError("backend must provide execute_update")
        if maximum_rows != 1:
            raise ValueError("controlled SQLite updates require maximum_rows=1")
        self._agent_id = agent_id
        self._backend = backend
        self._maximum_rows = maximum_rows

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        arguments = _update_arguments(request)
        impact_evidence_id = request.arguments["impact_evidence_id"]
        assert isinstance(impact_evidence_id, str)
        try:
            result = await self._backend.execute_update(
                agent_id=self._agent_id,
                operation_id=request.operation_id,
                **arguments,
                impact_evidence_id=impact_evidence_id,
                maximum_rows=self._maximum_rows,
            )
        except SQLiteUpdateKnownNoEffectError as error:
            raise ExecutorKnownNoEffectError(error.code, str(error)) from error
        if (
            result.source_id != arguments["source_id"]
            or result.resource_id != arguments["resource_id"]
            or sqlite_identifier_key(result.key_column)
            != sqlite_identifier_key(arguments["key_column"])
            or sqlite_identifier_key(result.target_column)
            != sqlite_identifier_key(arguments["target_column"])
            or result.impact_evidence_id != impact_evidence_id
            or result.maximum_rows != self._maximum_rows
        ):
            raise ValueError("SQLite update backend returned a different scope")
        return EvidenceCandidate(
            kind=SQLITE_UPDATE_EVIDENCE_KIND,
            schema_version=1,
            payload=result.evidence_payload(),
        )


def _update_arguments(request: ExecutionRequest) -> dict[str, str]:
    names = (
        "source_id",
        "resource_id",
        "key_column",
        "key_value",
        "target_column",
        "expected_value",
        "new_value",
    )
    arguments = {name: request.arguments[name] for name in names}
    assert all(isinstance(value, str) for value in arguments.values())
    return {name: value for name, value in arguments.items() if isinstance(value, str)}


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


def sqlite_update_declarations(
    agent_id: str,
    backend: SQLiteUpdateBackend,
) -> SQLiteUpdateDeclarations:
    impact_executor = SQLiteUpdateImpactExecutor(agent_id, backend)
    update_executor = SQLiteUpdateExecutor(agent_id, backend)
    extension = sqlite_update_extension_declarations()
    return SQLiteUpdateDeclarations(
        capabilities=extension.capabilities,
        executors=(impact_executor, update_executor),
        tool_views=extension.tool_views,
    )


def sqlite_update_extension_declarations() -> ExtensionDeclarations:
    """Advertise semantic preview and controlled-write SQLite contracts."""

    common_properties = _update_input_properties()
    common_required = list(common_properties)
    impact = Capability(
        id=SQLITE_UPDATE_IMPACT_CAPABILITY_ID,
        owner="data",
        description=(
            "Preview the bounded impact of one conditional SQLite row update."
        ),
        input_schema={
            "type": "object",
            "properties": common_properties,
            "required": common_required,
            "additionalProperties": False,
        },
        output_evidence_kind=SQLITE_UPDATE_IMPACT_EVIDENCE_KIND,
        output_schema_version=1,
        output_schema=_update_impact_output_schema(),
        executor_id=SQLITE_UPDATE_IMPACT_EXECUTOR_ID,
        access_mode=AccessMode.READ,
        risk=RiskLevel.LOW,
        side_effecting=False,
        idempotent=True,
        replay_safe=True,
    )
    update_properties = {
        **common_properties,
        "impact_evidence_id": {"type": "string"},
    }
    update = Capability(
        id=SQLITE_UPDATE_CAPABILITY_ID,
        owner="data",
        description=(
            "Apply one approved conditional SQLite row update after impact preview."
        ),
        input_schema={
            "type": "object",
            "properties": update_properties,
            "required": list(update_properties),
            "additionalProperties": False,
        },
        output_evidence_kind=SQLITE_UPDATE_EVIDENCE_KIND,
        output_schema_version=1,
        output_schema=_update_output_schema(),
        executor_id=SQLITE_UPDATE_EXECUTOR_ID,
        access_mode=AccessMode.WRITE,
        risk=RiskLevel.HIGH,
        side_effecting=True,
        idempotent=False,
        replay_safe=False,
        required_evidence_kinds=(SQLITE_UPDATE_IMPACT_EVIDENCE_KIND,),
    )
    return ExtensionDeclarations(
        capabilities=(impact, update),
        executor_ids=(SQLITE_UPDATE_IMPACT_EXECUTOR_ID, SQLITE_UPDATE_EXECUTOR_ID),
        tool_views=(
            ToolView(
                name=SQLITE_UPDATE_IMPACT_TOOL_NAME,
                capability_id=impact.id,
                description=impact.description,
            ),
            ToolView(
                name=SQLITE_UPDATE_TOOL_NAME,
                capability_id=update.id,
                description=update.description,
            ),
        ),
    )


def _update_input_properties() -> dict[str, object]:
    return {
        "source_id": {"type": "string"},
        "resource_id": {"type": "string"},
        "key_column": {"type": "string"},
        "key_value": {"type": "string"},
        "target_column": {"type": "string"},
        "expected_value": {"type": "string"},
        "new_value": {"type": "string"},
    }


def _update_impact_output_schema() -> dict[str, object]:
    properties = {
        "eligible_rows": {"type": "integer"},
        "key_column": {"type": "string"},
        "matched_rows": {"type": "integer"},
        "maximum_rows": {"type": "integer"},
        "recipe_fingerprint": {"type": "string"},
        "resource_id": {"type": "string"},
        "resource_revision": {"type": "string"},
        "source_id": {"type": "string"},
        "source_revision": {"type": "string"},
        "target_column": {"type": "string"},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _update_output_schema() -> dict[str, object]:
    properties = {
        "affected_rows": {"type": "integer"},
        "impact_evidence_id": {"type": "string"},
        "key_column": {"type": "string"},
        "maximum_rows": {"type": "integer"},
        "recipe_fingerprint": {"type": "string"},
        "resource_id": {"type": "string"},
        "resource_revision": {"type": "string"},
        "source_id": {"type": "string"},
        "source_revision": {"type": "string"},
        "target_column": {"type": "string"},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


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
    "SQLITE_UPDATE_EXECUTOR_ID",
    "SQLITE_UPDATE_IMPACT_EXECUTOR_ID",
    "SQLiteQueryDeclarations",
    "SQLiteQueryExecutor",
    "SQLiteReadBackend",
    "SQLiteReadResult",
    "SQLiteUpdateBackend",
    "SQLiteUpdateDeclarations",
    "SQLiteUpdateExecutor",
    "SQLiteUpdateImpactExecutor",
    "SQLiteUpdateImpactResult",
    "SQLiteUpdateKnownNoEffectError",
    "SQLiteUpdateResult",
    "sqlite_query_declarations",
    "sqlite_query_extension_declarations",
    "sqlite_update_declarations",
    "sqlite_update_extension_declarations",
]
