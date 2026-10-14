"""Declare and execute SQL reads and structured PostgreSQL updates."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from time import monotonic
from typing import Protocol, cast

from ..._json import FrozenJsonObject, canonical_json
from ...capabilities import (
    AccessMode,
    AutomationEligibility,
    AutomationGrantPolicy,
    Capability,
    CapabilityDeclarations,
    CapabilityInputError,
    EffectEvidenceBasis,
    EffectObservation,
    EffectReceiptPolicy,
    ExecutionAdmissionPolicy,
    Executor,
    OperationalEffect,
    ToolboxId,
    ToolExecution,
    ToolLoadMode,
    ToolOutput,
    ToolPresentation,
    ToolTextTrust,
    ToolView,
)
from ...catalog.models import CATALOG_SCHEMA_MAX_RESOURCE_IDS
from ...llm.models import ModelSensitivity
from .controller import (
    DATA_COUNT_ROWS_CAPABILITY_ID,
    DATA_QUERY_CAPABILITY_ID,
    DATA_QUERY_EVIDENCE_KIND,
    RELATIONAL_UPDATE_CAPABILITY_ID,
    RELATIONAL_UPDATE_EVIDENCE_KIND,
    RELATIONAL_UPDATE_PREVIEW_CAPABILITY_ID,
    RELATIONAL_UPDATE_PREVIEW_EVIDENCE_KIND,
    native_preview_capability,
)
from .results import BoundedResultProjection
from .row_counts import (
    COUNT_INSPECT_CAPABILITY_ID,
    COUNT_OPERATION_CAPABILITY_ID,
    COUNT_PLAN_TOOL_NAME,
    COUNT_SELECTION_TOOL_NAME,
    count_operation_calls,
    reduce_count_operation,
)
from .sql import (
    MAX_SQL_CHARACTERS,
    MAX_SQL_PARAMETERS,
    RelationalUpdateCell,
    RelationalUpdateCommand,
    RelationalUpdateFilter,
    RelationalUpdateIntent,
)
from .sql.relational_upsert import RelationalUpsertIntent

DATA_QUERY_EXECUTOR_ID = "data.query.executor"
DATA_QUERY_TOOL_NAME = "data_query"
COUNT_EXECUTION_POLICY = ExecutionAdmissionPolicy(
    shape="relational_counts",
    inline_eligible=True,
    graph_v1_eligible=False,
    target_count_argument="resource_ids",
    inline_max_targets=4,
    graph_max_targets=0,
)
COUNT_OPERATION_POLICY = ExecutionAdmissionPolicy(
    shape="relational_count_selection",
    inline_eligible=True,
    graph_v1_eligible=True,
    target_count_argument="resource_ids",
    inline_max_targets=CATALOG_SCHEMA_MAX_RESOURCE_IDS,
    graph_max_targets=CATALOG_SCHEMA_MAX_RESOURCE_IDS,
)

RELATIONAL_UPDATE_PREVIEW_EXECUTOR_ID = "data.preview_update_rows.executor"
RELATIONAL_UPDATE_PREVIEW_TOOL_NAME = "data_preview_update_rows"
RELATIONAL_UPDATE_EXECUTOR_ID = "data.update_rows.executor"
RELATIONAL_UPDATE_TOOL_NAME = "data_update_rows"
_MAX_PREVIEW_OUTPUT_BYTES = 256 * 1_024

RELATIONAL_UPDATE_RECEIPT_POLICY = EffectReceiptPolicy(
    receipt_kind="data.update_rows",
    success_evidence_basis=EffectEvidenceBasis.ADAPTER_VERIFIED,
    payload_schema={
        "type": "object",
        "properties": {
            "source_id": {"type": "string", "maxLength": 256},
            "resource_id": {"type": "string", "maxLength": 256},
            "intent_sha256": {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"},
            "preview_fingerprint": {
                "type": "string",
                "pattern": r"^sha256:[0-9a-f]{64}$",
            },
            "target_set_sha256": {"type": ["string", "null"], "maxLength": 71},
            "expected_affected_rows": {"type": "integer", "minimum": 1},
            "affected_rows": {"type": ["integer", "null"], "minimum": 0},
            "normalized_error_code": {"type": ["string", "null"], "maxLength": 128},
        },
        "required": [
            "source_id",
            "resource_id",
            "intent_sha256",
            "preview_fingerprint",
            "target_set_sha256",
            "expected_affected_rows",
            "affected_rows",
            "normalized_error_code",
        ],
        "additionalProperties": False,
    },
)


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
class RelationalPreviewFingerprint:
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
class RelationalUpdatePreviewChecks:
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
class RelationalUpdateSample:
    primary_key: tuple[RelationalUpdateCell, ...]
    before: tuple[RelationalUpdateCell, ...]
    after: tuple[RelationalUpdateCell, ...]

    def __post_init__(self) -> None:
        primary_key = tuple(self.primary_key)
        before = tuple(self.before)
        after = tuple(self.after)
        if not primary_key or any(
            not isinstance(item, RelationalUpdateCell) for item in primary_key
        ):
            raise ValueError("update sample requires a primary key")
        if (
            not before
            or any(not isinstance(item, RelationalUpdateCell) for item in before)
            or any(not isinstance(item, RelationalUpdateCell) for item in after)
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
class RelationalUpdatePreview:
    source_id: str
    resource_id: str
    resource_name: str
    source_revision: str
    resource_revision: str
    where: tuple[RelationalUpdateFilter, ...]
    assignments: tuple[RelationalUpdateCell, ...]
    matched_rows: int
    samples: tuple[RelationalUpdateSample, ...]
    fingerprint: RelationalPreviewFingerprint
    checks: RelationalUpdatePreviewChecks
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
            not isinstance(item, RelationalUpdateFilter) for item in where
        ):
            raise TypeError("preview where must contain update filters")
        if not assignments or any(
            not isinstance(item, RelationalUpdateCell) for item in assignments
        ):
            raise TypeError("preview assignments must contain update cells")
        if (
            not isinstance(self.matched_rows, int)
            or isinstance(self.matched_rows, bool)
            or self.matched_rows < 0
        ):
            raise ValueError("preview matched_rows must be a non-negative integer")
        if len(samples) > 5 or any(
            not isinstance(item, RelationalUpdateSample) for item in samples
        ):
            raise ValueError("preview samples must be bounded update samples")
        if len(samples) > self.matched_rows:
            raise ValueError("preview samples cannot exceed matched rows")
        object.__setattr__(self, "where", where)
        object.__setattr__(self, "assignments", assignments)
        object.__setattr__(self, "samples", samples)
        if not isinstance(self.fingerprint, RelationalPreviewFingerprint):
            raise TypeError("preview fingerprint must be RelationalPreviewFingerprint")
        if not isinstance(self.checks, RelationalUpdatePreviewChecks):
            raise TypeError("preview checks must be RelationalUpdatePreviewChecks")
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
class RelationalUpdateResult:
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
    effect_observation: EffectObservation

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
        expected_resource_revisions: tuple[tuple[str, str], ...] | None = None,
    ) -> SqlReadResult: ...


class SQLiteReadBackend(SqlReadBackend, Protocol):
    pass


class PostgreSQLReadBackend(SqlReadBackend, Protocol):
    pass


class RelationalUpdatePreviewBackend(Protocol):
    async def preview_update(
        self,
        *,
        agent_id: str,
        intent: RelationalUpdateIntent,
    ) -> RelationalUpdatePreview: ...


class RelationalUpdateBackend(RelationalUpdatePreviewBackend, Protocol):
    async def execute_update(
        self,
        *,
        agent_id: str,
        execution: ToolExecution,
        command: RelationalUpdateCommand,
    ) -> RelationalUpdateResult: ...


@dataclass(frozen=True, slots=True)
class DataQueryDeclarations:
    capabilities: tuple[Capability, ...]
    executors: tuple[Executor, ...]
    tool_views: tuple[ToolView, ...]


@dataclass(frozen=True, slots=True)
class RelationalUpdatePreviewDeclarations:
    capabilities: tuple[Capability, ...]
    executors: tuple[Executor, ...]
    tool_views: tuple[ToolView, ...]


@dataclass(frozen=True, slots=True)
class RelationalUpdateDeclarations:
    capabilities: tuple[Capability, ...]
    executors: tuple[Executor, ...]
    tool_views: tuple[ToolView, ...]


class DataQueryExecutor:
    """Dispatch one trusted catalog-bound relational read to one backend."""

    executor_id = DATA_QUERY_EXECUTOR_ID

    def __init__(
        self,
        agent_id: str,
        sqlite_backend: SQLiteReadBackend,
        postgresql_backend: PostgreSQLReadBackend,
        *,
        max_rows: int = 100,
        max_bytes: int = 16_384,
    ) -> None:
        self._agent_id = agent_id
        self._backends: dict[str, SqlReadBackend] = {
            "sqlite": sqlite_backend,
            "postgresql": postgresql_backend,
        }
        self._max_rows = max_rows
        self._max_bytes = max_bytes

    async def execute(self, request: ToolExecution) -> ToolOutput:
        source_id = request.arguments["source_id"]
        sql = request.arguments["sql"]
        parameters = request.arguments.get("parameters", ())
        resource_ids = request.arguments["resource_ids"]
        adapter_id = request.arguments["_adapter_id"]
        assert isinstance(source_id, str)
        assert isinstance(sql, str)
        assert isinstance(parameters, tuple)
        assert isinstance(resource_ids, tuple)
        assert isinstance(adapter_id, str)
        backend = self._backends.get(adapter_id)
        if backend is None:
            raise ValueError("relational query adapter binding is unsupported")
        bindings = request.arguments.get("_resource_revisions")
        if isinstance(bindings, tuple):
            result = await backend.execute_read(
                agent_id=self._agent_id,
                source_id=source_id,
                sql=sql,
                parameters=parameters,
                max_rows=self._max_rows,
                max_bytes=self._max_bytes,
                expected_resource_revisions=cast(tuple[tuple[str, str], ...], bindings),
            )
        else:
            result = await backend.execute_read(
                agent_id=self._agent_id,
                source_id=source_id,
                sql=sql,
                parameters=parameters,
                max_rows=self._max_rows,
                max_bytes=self._max_bytes,
            )
        expected_type = (
            PostgreSQLReadResult if adapter_id == "postgresql" else SQLiteReadResult
        )
        if (
            not isinstance(result, expected_type)
            or result.source_id != source_id
            or frozenset(result.resource_ids) != frozenset(resource_ids)
        ):
            raise ValueError("SQL backend returned different catalog binding facts")
        data = result.tool_data().to_dict()
        data["adapter_id"] = adapter_id
        return ToolOutput(kind=DATA_QUERY_EVIDENCE_KIND, data=data)


class DataCountRowsExecutor:
    """One atomic bounded read statement over exact admitted catalog identities."""

    executor_id = "data.count_rows.executor"

    def __init__(
        self,
        agent_id: str,
        sqlite_backend: SQLiteReadBackend,
        postgresql_backend: PostgreSQLReadBackend,
    ) -> None:
        self._query = DataQueryExecutor(
            agent_id, sqlite_backend, postgresql_backend, max_rows=1, max_bytes=16_384
        )

    async def execute(self, request: ToolExecution) -> ToolOutput:
        started = monotonic()
        observed_at = datetime.now(UTC).isoformat()
        result = await self._query.execute(request)
        data = result.data
        rows = cast(tuple[FrozenJsonObject, ...], data["rows"])
        ids = request.arguments["resource_ids"]
        revisions = request.arguments["_resource_revisions"]
        assert isinstance(ids, tuple) and isinstance(revisions, tuple)
        expected = dict(cast(tuple[tuple[str, str], ...], revisions))
        actual = {
            item["resource_id"]: item["revision"]
            for item in cast(tuple[FrozenJsonObject, ...], data["resource_revisions"])
        }
        if actual != expected or data["truncated"] or len(rows) != 1:
            raise ValueError("count result does not cover the exact bound batch")
        row = rows[0]
        counts = []
        for index, resource_id in enumerate(ids):
            value = row[f"count_{index}"]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    "count backend did not return an exact non-negative integer"
                )
            counts.append(
                {
                    "resource_id": resource_id,
                    "revision": expected[resource_id],
                    "rows": value,
                }
            )
        return ToolOutput(
            kind="data.row_counts",
            data={
                "source_id": data["source_id"],
                "source_name": request.arguments.get("_source_name", data["source_id"]),
                "source_revision": data["source_revision"],
                "adapter_id": data["adapter_id"],
                "counts": counts,
                "total_rows": sum(item["rows"] for item in counts),
                "requested_resources": len(ids),
                "counted_resources": len(counts),
                "coverage": "complete_requested_batch",
                "scope_limitation": "Selected admitted resources only; not the entire physical database. Views, when selected, may overlap base tables.",
                "snapshot_scope": "one_source_one_statement",
                "observed_at": observed_at,
                "duration_ms": int((monotonic() - started) * 1000),
                "sql_fingerprint": data["sql_fingerprint"],
                "trust_classification": "untrusted_external_data",
            },
        )


def data_count_rows_capability_declarations() -> CapabilityDeclarations:
    properties = dict(
        cast(
            Mapping[str, object],
            data_query_capability_declarations()
            .capabilities[0]
            .input_schema["properties"],
        )
    )
    properties.pop("sql")
    properties.pop("parameters")
    properties["include_views"] = {"type": "boolean", "default": False}
    properties["selection_ref"] = {"type": "string", "pattern": "^sel_[0-9a-f]{16}$"}
    properties["cursor"] = {"type": "string", "pattern": "^cnt_[0-9a-f]{16}$"}
    description = (
        "Count rows exactly in a unique bounded set of admitted base tables of one source. "
        "Prefer selection_ref from catalog_schema detail=identity for one source; then pass "
        "only the returned next_cursor as cursor until it is null. Code selects each bounded "
        "batch and returns cumulative totals and fixed-selection coverage in progress. "
        "Explicit source_id/resource_ids remains available for individual batches. "
        "Columns and SQL are not required. Sum table rows, not unique entities. A batch is atomic; "
        "different batches/sources do not share a snapshot. Views require include_views=true."
    )
    fields = {
        "source_id": "string",
        "source_name": "string",
        "source_revision": "string",
        "adapter_id": "string",
        "counts": "array",
        "total_rows": "integer",
        "requested_resources": "integer",
        "counted_resources": "integer",
        "coverage": "string",
        "scope_limitation": "string",
        "snapshot_scope": "string",
        "observed_at": "string",
        "duration_ms": "integer",
        "sql_fingerprint": "string",
        "trust_classification": "string",
    }
    capability = Capability(
        id=DATA_COUNT_ROWS_CAPABILITY_ID,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "oneOf": [
                {
                    "type": "object",
                    "properties": {key: properties[key] for key in keys},
                    "required": required,
                    "additionalProperties": False,
                }
                for keys, required in (
                    (
                        ("source_id", "resource_ids", "include_views"),
                        ["source_id", "resource_ids"],
                    ),
                    (("selection_ref", "include_views"), ["selection_ref"]),
                    (("cursor",), ["cursor"]),
                )
            ],
            "additionalProperties": False,
        },
        output_kind="data.row_counts",
        output_schema={
            "type": "object",
            "properties": {
                **{key: {"type": value} for key, value in fields.items()},
                "selection_ref": properties["selection_ref"],
                "next_cursor": {
                    "type": ["string", "null"],
                    "pattern": "^cnt_[0-9a-f]{16}$",
                },
                "progress": {
                    "type": "object",
                    "properties": {
                        **{
                            key: {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": CATALOG_SCHEMA_MAX_RESOURCE_IDS,
                            }
                            for key in (
                                "selected_resources",
                                "counted_resources",
                                "remaining_resources",
                            )
                        },
                        "total_rows": {"type": "integer", "minimum": 0},
                        "coverage": {
                            "type": "string",
                            "enum": [
                                "complete_admitted_base_tables",
                                "complete_catalog_selection",
                                "partial_catalog_selection",
                            ],
                        },
                        "catalog_enumeration_complete": {"type": "boolean"},
                        "snapshot_scope": {
                            "type": "string",
                            "enum": ["separate_batch_snapshots"],
                        },
                        "evidence_call_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": CATALOG_SCHEMA_MAX_RESOURCE_IDS,
                            "uniqueItems": True,
                        },
                    },
                    "required": [
                        "selected_resources",
                        "counted_resources",
                        "remaining_resources",
                        "total_rows",
                        "coverage",
                        "catalog_enumeration_complete",
                        "snapshot_scope",
                        "evidence_call_ids",
                    ],
                    "additionalProperties": False,
                },
            },
            "required": list(fields),
            "additionalProperties": False,
        },
        executor_id=DataCountRowsExecutor.executor_id,
        access_mode=AccessMode.READ,
        execution_admission_policy=COUNT_EXECUTION_POLICY,
    )
    return CapabilityDeclarations(
        domain_owner_id="data",
        capabilities=(capability,),
        executor_ids=(capability.executor_id,),
        tool_views=(
            ToolView(
                "data_count_rows",
                capability.id,
                description,
                ToolPresentation(
                    ToolboxId.SOURCES,
                    ToolLoadMode.ON_DEMAND,
                    ToolTextTrust.CODE,
                    "Count exact rows across admitted tables without loading column schemas.",
                    "Compare scoped database/table totals; use identity selection references and count cursors.",
                    ("count", "rows", "records", "database", "tables", "total"),
                ),
            ),
        ),
    )


class DataCountSelectionExecutor:
    """Execute a declared finite read operation through host-owned child calls."""

    executor_id = "data.count_selection.executor"

    async def execute(self, request: ToolExecution) -> ToolOutput:
        broker = request.read_broker
        if broker is None or broker.observations is None:
            raise CapabilityInputError(
                "count_operation_unavailable",
                "Bounded operation execution is unavailable; no child read ran.",
            )
        started = monotonic()
        observed_at = datetime.now(UTC).isoformat()
        deadline_reached = False
        try:
            async with asyncio.timeout(
                cast(int, request.arguments["_operation_wall_seconds"])
            ):
                await broker.batch(
                    count_operation_calls(
                        request.arguments, COUNT_EXECUTION_POLICY.inline_max_targets
                    ),
                    request.request_sensitivity,
                    COUNT_OPERATION_POLICY.inline_max_result_bytes,
                )
        except TimeoutError:
            deadline_reached = True
        records = await broker.observations()
        data = reduce_count_operation(
            request.run_id,
            request.call_id,
            request.arguments,
            records,
            width=COUNT_EXECUTION_POLICY.inline_max_targets,
            observed_at=observed_at,
            duration_ms=int((monotonic() - started) * 1000),
            deadline_reached=deadline_reached,
        )
        selection = cast(FrozenJsonObject, request.arguments["_count_selection"])
        classifications = [ModelSensitivity(cast(str, selection["sensitivity"]))]
        classifications.extend(
            ModelSensitivity(cast(str, r.facts["result_sensitivity"]))
            for r in records
            if r.facts.get("result_sensitivity") is not None
        )
        return ToolOutput(
            kind="data.count_operation",
            data=data,
            sensitivity=max(classifications, key=lambda s: s.routing_rank),
            sensitivity_provenance={
                "authority": "authenticated_count_operation",
                "source_ids": (request.arguments["source_id"],),
                "resource_ids": request.arguments["resource_ids"],
                "evidence_call_ids": data["evidence_call_ids"],
                "plan_sha256": data["plan_sha256"],
            },
        )


class DataCountInspectExecutor:
    executor_id = "data.count_inspect.executor"

    async def execute(self, request: ToolExecution) -> ToolOutput:
        return ToolOutput(
            kind="data.count_operation_details",
            data=cast(Mapping[str, object], request.arguments["_inspection"]),
            sensitivity=ModelSensitivity(
                cast(str, request.arguments["_inspection_sensitivity"])
            ),
            sensitivity_provenance=cast(
                Mapping[str, object], request.arguments["_inspection_provenance"]
            ),
        )


def data_count_operation_declarations() -> CapabilityDeclarations:
    description = (
        f"Count an entire authenticated fixed catalog selection of at most {COUNT_OPERATION_POLICY.inline_max_targets} resources. "
        f"Use selection_ref from catalog_schema detail=identity. The framework executes a finite plan of at most "
        f"{(COUNT_OPERATION_POLICY.inline_max_targets + COUNT_EXECUTION_POLICY.inline_max_targets - 1) // COUNT_EXECUTION_POLICY.inline_max_targets} child reads, "
        f"each at most {COUNT_EXECUTION_POLICY.inline_max_targets} resources, within {COUNT_OPERATION_POLICY.inline_max_wall_seconds} seconds and the shared run call budget. "
        "No model continuation or count cursors are needed. Returns code-computed exact total or observed subtotal, fixed coverage and evidence references. "
        "Default base tables only; include_views=true explicitly includes views. Separate batches do not share a snapshot. "
        "Failures are retained; no failed read is retried automatically. Detailed evidence is inspectable where data_count_inspect is admitted."
    )
    properties = {
        "selection_ref": {"type": "string", "pattern": "^sel_[0-9a-f]{16}$"},
        "include_views": {"type": "boolean", "default": False},
    }
    explicit_properties = {
        "source_id": {"type": "string", "minLength": 1, "maxLength": 256},
        "resource_ids": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 256},
            "minItems": 1,
            "maxItems": COUNT_OPERATION_POLICY.inline_max_targets,
            "uniqueItems": True,
        },
        "include_views": properties["include_views"],
    }
    plan_properties = {"plan_ref": {"type": "string", "pattern": "^plan_[0-9a-f]{16}$"}}
    output_properties: dict[str, object] = {
        name: {"type": kind}
        for name, kind in {
            "operation_ref": "string",
            "selection_ref": "string",
            "plan_sha256": "string",
            "source_id": "string",
            "source_name": "string",
            "source_revision": "string",
            "status": "string",
            "observed_total_rows": "integer",
            "selected_resources": "integer",
            "counted_resources": "integer",
            "failed_resources": "integer",
            "interrupted_resources": "integer",
            "unexecuted_resources": "integer",
            "coverage": "string",
            "catalog_enumeration_complete": "boolean",
            "snapshot_scope": "string",
            "scope_limitation": "string",
            "batch_max_resources": "integer",
            "planned_batches": "integer",
            "evidence_call_ids": "array",
            "failures": "array",
            "deadline_reached": "boolean",
            "observed_at": "string",
            "duration_ms": "integer",
        }.items()
    }
    output_properties["total_rows"] = {"type": ["integer", "null"]}
    operation = Capability(
        id=COUNT_OPERATION_CAPABILITY_ID,
        description=description,
        input_schema={
            "type": "object",
            "properties": {**properties, **explicit_properties, **plan_properties},
            "oneOf": [
                {
                    "type": "object",
                    "properties": plan_properties,
                    "required": ["plan_ref"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": properties,
                    "required": ["selection_ref"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": explicit_properties,
                    "required": ["source_id", "resource_ids"],
                    "additionalProperties": False,
                },
            ],
            "additionalProperties": False,
        },
        output_kind="data.count_operation",
        output_schema={
            "type": "object",
            "properties": output_properties,
            "required": list(output_properties),
            "additionalProperties": False,
        },
        executor_id=DataCountSelectionExecutor.executor_id,
        access_mode=AccessMode.READ,
        execution_admission_policy=COUNT_OPERATION_POLICY,
        read_dependency_ids=(DATA_COUNT_ROWS_CAPABILITY_ID,),
    )
    inspect_description = "Inspect exact retained per-table count evidence for an operation_ref returned in this run. Bounded pages perform no source I/O; observations remain historical if permissions or structure changed."
    inspect_properties = {
        "operation_ref": {"type": "string", "pattern": "^op_[0-9a-f]{16}$"},
        "offset": {"type": "integer", "minimum": 0, "maximum": 50, "default": 0},
        "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
    }
    inspect_output = {
        name: {"type": kind}
        for name, kind in {
            "operation_ref": "string",
            "source_id": "string",
            "source_revision": "string",
            "counts": "array",
            "offset": "integer",
            "next_offset": ["integer", "null"],
            "total_counts": "integer",
            "evidence_call_ids": "array",
            "snapshot_scope": "string",
            "historical_evidence": "boolean",
        }.items()
    }
    inspection = Capability(
        id=COUNT_INSPECT_CAPABILITY_ID,
        description=inspect_description,
        input_schema={
            "type": "object",
            "properties": inspect_properties,
            "required": ["operation_ref"],
            "additionalProperties": False,
        },
        output_kind="data.count_operation_details",
        output_schema={
            "type": "object",
            "properties": inspect_output,
            "required": list(inspect_output),
            "additionalProperties": False,
        },
        executor_id=DataCountInspectExecutor.executor_id,
        access_mode=AccessMode.READ,
    )
    return CapabilityDeclarations(
        domain_owner_id="data",
        capabilities=(operation, inspection),
        executor_ids=(operation.executor_id, inspection.executor_id),
        tool_views=tuple(
            ToolView(
                name,
                cap.id,
                (
                    "Use only selection_ref and optional include_views. The authenticated reference already binds its source, resources and revisions; do not pass source_id, resource_ids or plan_ref. "
                    + cap.description
                    if name == COUNT_SELECTION_TOOL_NAME
                    else (
                        "Execute only the exact code-owned plan_ref from this graph task; do not supply source or selection identifiers. "
                        + cap.description
                        if name == COUNT_PLAN_TOOL_NAME
                        else cap.description
                    )
                ),
                ToolPresentation(
                    ToolboxId.SOURCES,
                    ToolLoadMode.ON_DEMAND,
                    ToolTextTrust.CODE,
                    summary,
                    guidance,
                    keywords,
                ),
                input_schema=(
                    {
                        "type": "object",
                        "properties": {
                            "selection_ref": properties["selection_ref"],
                            "include_views": {"type": "boolean"},
                        },
                        "required": ["selection_ref"],
                        "additionalProperties": False,
                    }
                    if name == COUNT_SELECTION_TOOL_NAME
                    else (
                        {
                            "type": "object",
                            "properties": plan_properties,
                            "required": ["plan_ref"],
                            "additionalProperties": False,
                        }
                        if name == COUNT_PLAN_TOOL_NAME
                        else None
                    )
                ),
            )
            for name, cap, summary, guidance, keywords in (
                (
                    COUNT_SELECTION_TOOL_NAME,
                    operation,
                    "Count a fixed table selection with framework-managed bounded reads and totals.",
                    "Prefer for source comparisons. Use catalog identity selection references; no per-page model calls.",
                    (
                        "count",
                        "rows",
                        "records",
                        "database",
                        "tables",
                        "total",
                        "compare",
                        "operation",
                    ),
                ),
                (
                    COUNT_PLAN_TOOL_NAME,
                    operation,
                    "Execute the frozen count plan for this graph task.",
                    "Use only the task's exact plan_ref; bounded reads and totals use the same count execution owner.",
                    ("count", "plan", "graph", "rows"),
                ),
                (
                    "data_count_inspect",
                    inspection,
                    "Inspect retained exact table counts for a count operation.",
                    "Use operation_ref for per-table evidence; unnecessary for answering completed source totals.",
                    ("count", "inspect", "evidence", "rows", "operation"),
                ),
            )
        ),
    )


class RelationalUpdatePreviewExecutor:
    executor_id = RELATIONAL_UPDATE_PREVIEW_EXECUTOR_ID

    def __init__(
        self,
        agent_id: str,
        backend: RelationalUpdatePreviewBackend,
    ) -> None:
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("preview executor agent_id must be non-empty text")
        if not callable(getattr(backend, "preview_update", None)):
            raise TypeError("preview backend must provide preview_update")
        self._agent_id = agent_id
        self._backend = backend

    async def execute(self, request: ToolExecution) -> ToolOutput:
        intent = RelationalUpdateIntent.from_mapping(request.arguments)
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
            kind=RELATIONAL_UPDATE_PREVIEW_EVIDENCE_KIND,
            data=result.tool_data(),
        )


class RelationalUpdateExecutor:
    executor_id = RELATIONAL_UPDATE_EXECUTOR_ID

    def __init__(self, agent_id: str, backend: RelationalUpdateBackend) -> None:
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
        command = RelationalUpdateCommand.from_mapping(request.arguments)
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
                "review": {
                    "matched_rows": preview.matched_rows,
                    "samples": tuple(sample.to_payload() for sample in preview.samples),
                    "warnings": preview.warnings,
                },
            }
        )

    async def execute(self, request: ToolExecution) -> ToolOutput:
        command = RelationalUpdateCommand.from_mapping(request.arguments)
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
        return ToolOutput(
            kind=RELATIONAL_UPDATE_EVIDENCE_KIND,
            data=result.tool_data(),
            effect_observation=result.effect_observation,
        )


def data_query_declarations(
    agent_id: str,
    sqlite_backend: SQLiteReadBackend,
    postgresql_backend: PostgreSQLReadBackend,
) -> DataQueryDeclarations:
    executor = DataQueryExecutor(agent_id, sqlite_backend, postgresql_backend)
    declarations = data_query_capability_declarations()
    return DataQueryDeclarations(
        (
            *declarations.capabilities,
            *data_count_rows_capability_declarations().capabilities,
            *data_count_operation_declarations().capabilities,
        ),
        (
            executor,
            DataCountRowsExecutor(agent_id, sqlite_backend, postgresql_backend),
            DataCountSelectionExecutor(),
            DataCountInspectExecutor(),
        ),
        (
            *declarations.tool_views,
            *data_count_rows_capability_declarations().tool_views,
            *data_count_operation_declarations().tool_views,
        ),
    )


def relational_update_preview_declarations(
    agent_id: str,
    backend: RelationalUpdatePreviewBackend,
) -> RelationalUpdatePreviewDeclarations:
    executor = RelationalUpdatePreviewExecutor(agent_id, backend)
    declarations = relational_update_preview_capability_declarations()
    return RelationalUpdatePreviewDeclarations(
        declarations.capabilities,
        (executor,),
        declarations.tool_views,
    )


def relational_update_declarations(
    agent_id: str,
    backend: RelationalUpdateBackend,
) -> RelationalUpdateDeclarations:
    executor = RelationalUpdateExecutor(agent_id, backend)
    declarations = relational_update_capability_declarations()
    return RelationalUpdateDeclarations(
        declarations.capabilities,
        (executor,),
        declarations.tool_views,
    )


def data_query_capability_declarations() -> CapabilityDeclarations:
    return _query_declarations(
        ToolPresentation(
            toolbox_id=ToolboxId.SOURCES,
            load_mode=ToolLoadMode.PINNED,
            text_trust=ToolTextTrust.CODE,
            summary="Run one bounded validated read-only relational query.",
            when_to_use=(
                "Use exact source/resource IDs and columns from current catalog evidence; "
                "obtain missing structure only through admitted, callable tools."
            ),
            keywords=("data", "query", "relational", "sql"),
        ),
    )


def relational_update_preview_capability_declarations() -> CapabilityDeclarations:
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
        id=RELATIONAL_UPDATE_PREVIEW_CAPABILITY_ID,
        description=(
            "Validate and preview one structured PostgreSQL update over an exact "
            "catalog-scoped target set without changing the database. When the "
            "user requested approval or execution, ground the intended target "
            "and pass its exact positive preview to data_update_rows when the run "
            "can proceed. Zero matches require correction or explanation; "
            "preview alone does not request approval."
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
        output_kind=RELATIONAL_UPDATE_PREVIEW_EVIDENCE_KIND,
        output_schema=_relational_update_preview_output_schema(),
        executor_id=RELATIONAL_UPDATE_PREVIEW_EXECUTOR_ID,
        access_mode=AccessMode.READ,
        automation_eligibility=AutomationEligibility.AUTOMATION_DIRECT,
    )
    view = ToolView(
        name=RELATIONAL_UPDATE_PREVIEW_TOOL_NAME,
        capability_id=capability.id,
        description=capability.description,
        presentation=ToolPresentation(
            toolbox_id=ToolboxId.SOURCES,
            load_mode=ToolLoadMode.ON_DEMAND,
            text_trust=ToolTextTrust.CODE,
            summary="Preview matching rows and proposed changes for a relational update.",
            when_to_use="Use before applying an admitted update to existing rows.",
            keywords=("data", "relational", "postgresql", "rows", "update", "preview"),
        ),
    )
    return CapabilityDeclarations(
        domain_owner_id="data",
        capabilities=(capability,),
        executor_ids=(capability.executor_id,),
        tool_views=(view,),
    )


def relational_update_capability_declarations() -> CapabilityDeclarations:
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
        id=RELATIONAL_UPDATE_CAPABILITY_ID,
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
        output_kind=RELATIONAL_UPDATE_EVIDENCE_KIND,
        output_schema=_relational_update_output_schema(),
        executor_id=RELATIONAL_UPDATE_EXECUTOR_ID,
        access_mode=AccessMode.WRITE,
        operational_effect=OperationalEffect.MUTATE_DATA,
        automation_eligibility=AutomationEligibility.AUTOMATION_DIRECT,
        effect_receipt_policy=RELATIONAL_UPDATE_RECEIPT_POLICY,
        automation_grant_policy=NATIVE_WRITE_GRANT_POLICY,
        execution_admission_policy=ExecutionAdmissionPolicy(
            shape="preview_bound_postgresql_update",
            inline_eligible=True,
            graph_v1_eligible=True,
            target_count_argument=None,
            inline_max_targets=1,
            graph_max_targets=1,
        ),
    )
    view = ToolView(
        name=RELATIONAL_UPDATE_TOOL_NAME,
        capability_id=capability.id,
        description=capability.description,
        presentation=ToolPresentation(
            toolbox_id=ToolboxId.SOURCES,
            load_mode=ToolLoadMode.ON_DEMAND,
            text_trust=ToolTextTrust.CODE,
            summary="Apply an exact previewed update to existing relational rows.",
            when_to_use="Use after a successful update preview, with approval or an exact standing grant.",
            keywords=(
                "data",
                "relational",
                "postgresql",
                "rows",
                "update",
                "apply",
                "approval",
            ),
        ),
    )
    return CapabilityDeclarations(
        domain_owner_id="data",
        capabilities=(capability,),
        executor_ids=(capability.executor_id,),
        tool_views=(view,),
    )


def _query_declarations(
    presentation: ToolPresentation,
) -> CapabilityDeclarations:
    capability = Capability(
        id=DATA_QUERY_CAPABILITY_ID,
        description=(
            "Run one validated, read-only, bounded relational query against exact "
            "current catalog resources."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "source_id": {"type": "string"},
                "resource_ids": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "maxLength": 256},
                    "minItems": 1,
                    "maxItems": 64,
                    "uniqueItems": True,
                },
                "sql": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_SQL_CHARACTERS,
                },
                "parameters": {
                    "type": "array",
                    "maxItems": MAX_SQL_PARAMETERS,
                    "description": "SQLite ? values follow SQL occurrence order; PostgreSQL $1, $2, ... by index. Verify filter bindings.",
                },
            },
            "required": ["source_id", "resource_ids", "sql"],
            "additionalProperties": False,
        },
        output_kind=DATA_QUERY_EVIDENCE_KIND,
        output_schema=_query_output_schema(),
        executor_id=DATA_QUERY_EXECUTOR_ID,
        access_mode=AccessMode.READ,
        automation_eligibility=AutomationEligibility.AUTOMATION_DIRECT,
        execution_admission_policy=ExecutionAdmissionPolicy(
            shape="relational_read",
            inline_eligible=True,
            graph_v1_eligible=True,
            target_count_argument="resource_ids",
            inline_max_targets=4,
            graph_max_targets=16,
        ),
    )
    view = ToolView(
        name=DATA_QUERY_TOOL_NAME,
        capability_id=capability.id,
        description=capability.description,
        presentation=presentation,
    )
    return CapabilityDeclarations(
        domain_owner_id="data",
        capabilities=(capability,),
        executor_ids=(DATA_QUERY_EXECUTOR_ID,),
        tool_views=(view,),
    )


def _query_output_schema() -> dict[str, object]:
    names = (
        "byte_limit",
        "adapter_id",
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
        "adapter_id": "string",
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


def _relational_update_preview_output_schema() -> dict[str, object]:
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


def _relational_update_output_schema() -> dict[str, object]:
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
                "pattern": r"^effect-receipt:sha256:[0-9a-f]{64}$",
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
    "DATA_QUERY_EXECUTOR_ID",
    "DATA_QUERY_TOOL_NAME",
    "DataQueryDeclarations",
    "DataQueryExecutor",
    "RELATIONAL_UPDATE_PREVIEW_EXECUTOR_ID",
    "RELATIONAL_UPDATE_PREVIEW_TOOL_NAME",
    "RELATIONAL_UPDATE_EXECUTOR_ID",
    "RELATIONAL_UPDATE_TOOL_NAME",
    "PostgreSQLReadBackend",
    "PostgreSQLReadResult",
    "RelationalPreviewFingerprint",
    "RelationalUpdatePreview",
    "RelationalUpdatePreviewBackend",
    "RelationalUpdatePreviewChecks",
    "RelationalUpdatePreviewDeclarations",
    "RelationalUpdatePreviewExecutor",
    "RelationalUpdateBackend",
    "RelationalUpdateDeclarations",
    "RelationalUpdateExecutor",
    "RelationalUpdateResult",
    "RelationalUpdateSample",
    "SQLiteReadBackend",
    "SQLiteReadResult",
    "data_query_declarations",
    "data_query_capability_declarations",
    "relational_update_preview_declarations",
    "relational_update_preview_capability_declarations",
    "relational_upsert_declarations",
    "relational_upsert_capability_declarations",
    "RelationalUpsertResult",
    "RelationalUpsertBackend",
    "relational_update_declarations",
    "relational_update_capability_declarations",
]

# The native grant is one exact resource and one bounded invocation per occurrence.
_NATIVE_COLUMNS_SCHEMA = {
    "type": "array",
    "items": {"type": "string", "minLength": 1, "maxLength": 256},
    "maxItems": 512,
    "uniqueItems": True,
}
NATIVE_WRITE_GRANT_POLICY = AutomationGrantPolicy(
    constraints_kind="data.relational_write",
    constraints_schema={
        "type": "object",
        "description": "The proposed allowed_capability_ids must include the matching preview: "
        + "; ".join(
            f"{capability_id} requires {native_preview_capability(capability_id)}"
            for capability_id in (RELATIONAL_UPDATE_CAPABILITY_ID, "data.upsert_rows")
        )
        + ". Each occurrence must obtain its own authenticated preview before writing.",
        "properties": {
            "source_id": {"type": "string", "maxLength": 256},
            "resource_id": {"type": "string", "maxLength": 256},
            "resource_revision": {
                "type": "string",
                "pattern": r"^sha256:[0-9a-f]{64}$",
            },
            "key_columns": _NATIVE_COLUMNS_SCHEMA,
            "allowed_insert_columns": _NATIVE_COLUMNS_SCHEMA,
            "allowed_update_columns": _NATIVE_COLUMNS_SCHEMA,
            "generated_identity_columns": _NATIVE_COLUMNS_SCHEMA,
            "max_rows": {"type": "integer", "minimum": 1, "maximum": 10000},
        },
        "required": [
            "source_id",
            "resource_id",
            "resource_revision",
            "key_columns",
            "allowed_insert_columns",
            "allowed_update_columns",
            "generated_identity_columns",
            "max_rows",
        ],
        "additionalProperties": False,
    },
)


@dataclass(frozen=True, slots=True)
class RelationalUpsertResult:
    data: FrozenJsonObject
    effect_observation: EffectObservation


class RelationalUpsertBackend(Protocol):
    async def preview_upsert(
        self, *, agent_id: str, intent: RelationalUpsertIntent
    ) -> FrozenJsonObject: ...
    async def execute_upsert(
        self,
        *,
        agent_id: str,
        execution: ToolExecution,
        intent: RelationalUpsertIntent,
        preview_fingerprint: str,
    ) -> RelationalUpsertResult: ...


class RelationalUpsertPreviewExecutor:
    executor_id = "data.preview_upsert_rows.executor"

    def __init__(self, agent_id: str, backend: RelationalUpsertBackend) -> None:
        self._agent_id, self._backend = agent_id, backend

    async def execute(self, request: ToolExecution) -> ToolOutput:
        intent = RelationalUpsertIntent.from_mapping(request.arguments)
        return ToolOutput(
            kind="data.preview_upsert_rows",
            data=await self._backend.preview_upsert(
                agent_id=self._agent_id, intent=intent
            ),
        )


class RelationalUpsertExecutor:
    executor_id = "data.upsert_rows.executor"

    def __init__(self, agent_id: str, backend: RelationalUpsertBackend) -> None:
        self._agent_id, self._backend = agent_id, backend

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        intent = RelationalUpsertIntent.from_mapping(request.arguments)
        preview = await self._backend.preview_upsert(
            agent_id=self._agent_id, intent=intent
        )
        if preview["preview_fingerprint"] != request.arguments["preview_fingerprint"]:
            raise CapabilityInputError(
                "write_preview_stale",
                "The exact upsert preview changed before dispatch.",
            )
        return FrozenJsonObject.from_mapping(
            {
                **{
                    key: preview[key]
                    for key in (
                        "intent_sha256",
                        "preview_fingerprint",
                        "resource_revision",
                        "target_set_sha256",
                        "permission_fingerprint",
                    )
                },
                "review": {
                    **{
                        key: preview[key]
                        for key in (
                            "input_count",
                            "inserted_count",
                            "updated_count",
                            "unchanged_count",
                            "identity_sequence_gaps_possible",
                        )
                    },
                    "classifications": cast(
                        tuple[object, ...], preview["classifications"]
                    )[:5],
                },
            }
        )

    async def execute(self, request: ToolExecution) -> ToolOutput:
        result = await self._backend.execute_upsert(
            agent_id=self._agent_id,
            execution=request,
            intent=RelationalUpsertIntent.from_mapping(request.arguments),
            preview_fingerprint=cast(str, request.arguments["preview_fingerprint"]),
        )
        return ToolOutput(
            kind="data.upsert_rows_result",
            data=result.data,
            effect_observation=result.effect_observation,
        )


_UPSERT_RECEIPT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "source_id": {"type": "string", "maxLength": 256},
        "resource_id": {"type": "string", "maxLength": 256},
        "intent_sha256": {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"},
        "preview_fingerprint": {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"},
        "target_set_sha256": {"type": ["string", "null"], "maxLength": 71},
        "input_count": {"type": "integer", "minimum": 1, "maximum": 1000},
        "inserted_count": {"type": ["integer", "null"], "minimum": 0, "maximum": 1000},
        "updated_count": {"type": ["integer", "null"], "minimum": 0, "maximum": 1000},
        "unchanged_count": {"type": ["integer", "null"], "minimum": 0, "maximum": 1000},
        "normalized_error_code": {"type": ["string", "null"], "maxLength": 128},
        "identity_sequence_gaps_possible": {"type": "boolean"},
    },
    "required": [
        "source_id",
        "resource_id",
        "intent_sha256",
        "preview_fingerprint",
        "target_set_sha256",
        "input_count",
        "inserted_count",
        "updated_count",
        "unchanged_count",
        "normalized_error_code",
        "identity_sequence_gaps_possible",
    ],
    "additionalProperties": False,
}


def relational_upsert_capability_declarations() -> CapabilityDeclarations:
    properties: dict[str, object] = {
        "source_id": {"type": "string", "pattern": r"^source:sha256:[0-9a-f]{64}$"},
        "resource_id": {
            "type": "string",
            "pattern": r"^catalog-resource:sha256:[0-9a-f]{64}$",
        },
        "key_columns": {**_NATIVE_COLUMNS_SCHEMA, "minItems": 1},
        "insert_columns": {**_NATIVE_COLUMNS_SCHEMA, "minItems": 1},
        "update_columns": {**_NATIVE_COLUMNS_SCHEMA, "minItems": 1},
        "rows": {
            "type": "array",
            "minItems": 1,
            "maxItems": 1000,
            "items": {
                "type": "object",
                "additionalProperties": {
                    "type": ["string", "number", "boolean", "null"]
                },
            },
        },
        "evidence_call_ids": {
            "type": "array",
            "maxItems": 32,
            "uniqueItems": True,
            "items": {"type": "string", "maxLength": 256},
        },
    }
    required = [
        "source_id",
        "resource_id",
        "key_columns",
        "insert_columns",
        "update_columns",
        "rows",
    ]
    preview = Capability(
        id="data.preview_upsert_rows",
        description=(
            "Preview one exact uniform batch as inserts, updates or unchanged rows. Values are model-derived claims; "
            "research call IDs retain current-run evidence, not proof of truth. Identity values are allocated only on insert. "
            "Execution briefly takes a table-wide EXCLUSIVE lock permitting ordinary reads, with bounded timeouts."
        ),
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        output_kind="data.preview_upsert_rows",
        output_schema=_upsert_output_schema(preview=True),
        executor_id=RelationalUpsertPreviewExecutor.executor_id,
        access_mode=AccessMode.READ,
        automation_eligibility=AutomationEligibility.AUTOMATION_DIRECT,
    )
    write = Capability(
        id="data.upsert_rows",
        description=(
            "Apply one authenticated current-run upsert preview atomically under exact approval or a standing grant. "
            "No chunking or replay. Unchanged rows consume the invocation and produce verified zero-change evidence. "
            "Rollback does not restore identity sequence allocations."
        ),
        input_schema={
            "type": "object",
            "properties": {
                **properties,
                "preview_fingerprint": {
                    "type": "string",
                    "pattern": r"^sha256:[0-9a-f]{64}$",
                },
            },
            "required": [*required, "preview_fingerprint"],
            "additionalProperties": False,
        },
        output_kind="data.upsert_rows_result",
        output_schema=_upsert_output_schema(preview=False),
        executor_id=RelationalUpsertExecutor.executor_id,
        access_mode=AccessMode.WRITE,
        operational_effect=OperationalEffect.MUTATE_DATA,
        automation_eligibility=AutomationEligibility.AUTOMATION_DIRECT,
        automation_grant_policy=NATIVE_WRITE_GRANT_POLICY,
        effect_receipt_policy=EffectReceiptPolicy(
            receipt_kind="data.upsert_rows",
            payload_schema=_UPSERT_RECEIPT_SCHEMA,
            success_evidence_basis=EffectEvidenceBasis.ADAPTER_VERIFIED,
        ),
        execution_admission_policy=ExecutionAdmissionPolicy(
            shape="preview_bound_postgresql_upsert",
            inline_eligible=True,
            graph_v1_eligible=True,
            target_count_argument=None,
            inline_max_targets=1,
            graph_max_targets=1,
        ),
    )
    return CapabilityDeclarations(
        domain_owner_id="data",
        capabilities=(preview, write),
        executor_ids=(preview.executor_id, write.executor_id),
        tool_views=tuple(
            ToolView(
                name=cap.id.replace(".", "_"),
                capability_id=cap.id,
                description=cap.description,
                presentation=ToolPresentation(
                    toolbox_id=ToolboxId.SOURCES,
                    load_mode=ToolLoadMode.ON_DEMAND,
                    text_trust=ToolTextTrust.CODE,
                    summary=summary,
                    when_to_use=when_to_use,
                    keywords=(
                        "data",
                        "relational",
                        "upsert",
                        "insert",
                        "update",
                        "rows",
                        "row",
                        "table",
                        "batch",
                        "research",
                        *action_keywords,
                    ),
                ),
            )
            for cap, summary, when_to_use, action_keywords in (
                (
                    preview,
                    "Preview an exact relational batch: insert missing rows, update existing rows, or leave them unchanged.",
                    "Use before applying an admitted upsert batch identified by exact conflict keys.",
                    ("preview",),
                ),
                (
                    write,
                    "Insert missing rows or update existing rows in a relational table using one exact previewed batch.",
                    "Use after a successful upsert preview, with approval or an exact standing grant.",
                    ("apply", "save", "approval"),
                ),
            )
        ),
    )


def _upsert_output_schema(*, preview: bool) -> dict[str, object]:
    properties: dict[str, object] = {
        "source_id": {"type": "string"},
        "resource_id": {"type": "string"},
        "resource_revision": {"type": "string"},
        "intent_sha256": {"type": "string"},
        "preview_fingerprint": {"type": "string"},
        "target_set_sha256": {"type": "string"},
        "permission_fingerprint": {"type": "string"},
        "input_count": {"type": "integer", "minimum": 1, "maximum": 1000},
        "inserted_count": {"type": "integer", "minimum": 0, "maximum": 1000},
        "updated_count": {"type": "integer", "minimum": 0, "maximum": 1000},
        "unchanged_count": {"type": "integer", "minimum": 0, "maximum": 1000},
        "classifications": {
            "type": "array",
            "maxItems": 1000,
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "object"},
                    "action": {
                        "type": "string",
                        "enum": ["insert", "update", "unchanged"],
                    },
                },
                "required": ["key", "action"],
                "additionalProperties": False,
            },
        },
        "evidence_call_ids": {
            "type": "array",
            "maxItems": 32,
            "items": {"type": "string"},
        },
        "authorship": {"type": "string", "enum": ["model_derived"]},
        "identity_sequence_gaps_possible": {"type": "boolean"},
    }
    if not preview:
        properties.update(
            {
                "receipt_id": {"type": "string"},
                "committed_at": {"type": "string"},
                "generated_identities": {
                    "type": "array",
                    "maxItems": 1000,
                    "items": {"type": "object"},
                },
            }
        )
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def relational_upsert_declarations(
    agent_id: str, backend: RelationalUpsertBackend
) -> DataQueryDeclarations:
    declarations = relational_upsert_capability_declarations()
    return DataQueryDeclarations(
        declarations.capabilities,
        (
            RelationalUpsertPreviewExecutor(agent_id, backend),
            RelationalUpsertExecutor(agent_id, backend),
        ),
        declarations.tool_views,
    )
