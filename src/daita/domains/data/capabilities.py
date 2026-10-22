"""Declare and execute SQL reads and structured PostgreSQL updates."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Protocol, TypeVar, cast

from ..._json import FrozenJsonObject, canonical_json
from ...adapters.local_file_query import LocalFileQueryError, QueryRelation
from ...artifacts.renderers import MAX_CSV_ROWS, MAX_CSV_SECONDS
from ...capabilities import (
    RESOURCE_ID_PATTERN,
    SOURCE_ID_PATTERN,
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
    RecoveryKind,
    ToolboxId,
    ToolExecution,
    ToolLoadMode,
    ToolOutput,
    ToolPresentation,
    ToolRecovery,
    ToolTextTrust,
    ToolView,
    resource_id_schema,
    source_id_schema,
)
from ...errors import DaitaError
from .controller import (
    DATA_EXPORT_TABULAR_CAPABILITY_ID,
    DATA_QUERY_CAPABILITY_ID,
    DATA_QUERY_COMBINED_CAPABILITY_ID,
    DATA_QUERY_COMBINED_EVIDENCE_KIND,
    DATA_QUERY_EVIDENCE_KIND,
    RELATIONAL_UPDATE_CAPABILITY_ID,
    RELATIONAL_UPDATE_EVIDENCE_KIND,
    RELATIONAL_UPDATE_PREVIEW_CAPABILITY_ID,
    RELATIONAL_UPDATE_PREVIEW_EVIDENCE_KIND,
    native_preview_capability,
)
from .results import BoundedResultProjection
from .source_files import (
    PATH_QUERY_EXECUTOR_ID,
    PathPatternReadBackend,
    PathQueryExecutor,
    path_query_capability,
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
DATA_QUERY_COMBINED_EXECUTOR_ID = "data.query_combined.executor"
DATA_QUERY_TOOL_NAME = "data_query"
RELATIONAL_UPDATE_PREVIEW_EXECUTOR_ID = "data.preview_update_rows.executor"
RELATIONAL_UPDATE_EXECUTOR_ID = "data.update_rows.executor"
RELATIONAL_UPDATE_TOOL_NAME = "data_update_rows"
RELATIONAL_UPSERT_TOOL_NAME = "data_upsert_rows"
# The data.query capability description is part of its contract digest, which
# retained routine and graph bindings revalidate; the model reads the view's.
_DATA_QUERY_CAPABILITY_DESCRIPTION = (
    "Run read-only SQL on the cataloged tables of one source, named by its "
    "display name. Use sql, or statements for up to 8 reads in one call. "
    "Returns columns, rows, tables_touched, truncation "
    "and timing. Limits: 5s per statement, 15s total, 100 rows and 16 KB "
    "shared by all statements. "
    'Example: {"source": "Shop", '
    '"sql": "SELECT region, sum(total) FROM orders GROUP BY region"}. '
    "Mistakes: columns absent from catalog evidence; joining on matching "
    "names alone; treating truncated rows as complete."
)
DATA_QUERY_DESCRIPTION = (
    "Read-only SQL on the cataloged tables of one source, named by its display "
    "name. Use sql, or statements for up to 8 reads. "
    "Limits: 5s per statement, 15s "
    "total, 100 rows and 16 KB shared."
)
_DATA_QUERY_USAGE = (
    " Profile with aggregates (COUNT, COUNT(DISTINCT), null counts, MIN/MAX)."
    ' Example: "SELECT region, sum(total) FROM orders GROUP BY region".'
    " Mistakes: columns absent from catalog evidence; trusting truncated rows."
)


_DATA_QUERY_COMBINED_CAPABILITY_DESCRIPTION = (
    "Join or aggregate data from 2 to 4 sources in one bounded read. Each input "
    "runs read-only SQL in its own source's dialect on exact catalog resources, "
    "or reads exact catalog columns of one resource with optional eq, in and "
    "range filters; sql is one DuckDB SELECT or UNION, at most 8 KiB, that "
    "reads every input alias and nothing else. Limits: 5s per input, 15s for "
    "all inputs, 100 rows and 16 KB in the result. An input too large to join "
    "completely refuses the call, so aggregate or filter inside each input."
)
_DATA_QUERY_COMBINED_USAGE = (
    " Join sources: inputs is 2-4 {alias, source, sql} in each "
    "source's dialect, or {alias, source, table, columns, filters} for "
    "a source without SQL; sql is one DuckDB SELECT or UNION over the aliases "
    "(8 KiB). Aggregate or filter in each input: one too large refuses the call."
)
_DATA_QUERY_PATH_USAGE = (
    " Folder files: query relation data by path_pattern (live folder); an "
    "attached folder reads its cataloged files as relation data."
)


def _data_query_description(file_outputs: tuple[str, ...]) -> str:
    file_output = (
        f" output={'|'.join(file_outputs)} exports one sql in full as a file "
        f"({MAX_CSV_ROWS:,} rows, {MAX_CSV_SECONDS:.0f}s), no rows; to save "
        "rows already shown in full, use artifact_create from=table."
        if file_outputs
        else ""
    )
    return DATA_QUERY_DESCRIPTION + file_output + _DATA_QUERY_USAGE


RELATIONAL_UPDATE_DESCRIPTION = (
    "Update rows in a PostgreSQL table. mode=preview changes nothing and "
    "returns matched_rows, samples, preview_fingerprint. mode=apply "
    "resends the arguments plus preview_fingerprint and "
    "expected_affected_rows=matched_rows; calling it is what requests "
    "approval. Only outcome=committed proves it; a change after approval is "
    "final for the request. Limits: structured filters, no SQL. "
    'Example: {"where": [{"column": "id", "operator": "eq", "value": 7}], '
    '"assignments": [{"column": "status", "value": "paid"}], ...}. '
    "Mistakes: asking for approval in chat; applying a zero-match preview; "
    "retrying after denial."
)
RELATIONAL_UPSERT_DESCRIPTION = (
    "Insert missing rows and update existing ones in one PostgreSQL table by "
    "key_columns; needs upsert permission. mode=preview changes nothing and "
    "classifies rows as insert, update or unchanged. mode=apply sends the "
    "same batch plus preview_fingerprint, requests approval, locks the table "
    "briefly and reports verified counts; a change after approval is final "
    "for the request. Limits: one uniform batch of up to 1000 scalar rows, no "
    'chunking. Example: {"key_columns": ["sku"], "rows": [{"sku": "A1", '
    '"qty": 3}], ...}. Mistakes: omitting a column to clear it (send null); '
    "retrying after denial or failure."
)
_REPREVIEW = ToolRecovery(
    RecoveryKind.REINSPECT_BINDING,
    "The table changed since the preview; run mode=preview again and apply "
    "only the new preview_fingerprint.",
)
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
    resource_names: tuple[tuple[str, str], ...] = ()

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
        names = tuple(sorted(tuple(item) for item in self.resource_names))
        if names and {item[0] for item in names} != set(resource_ids):
            raise ValueError("resource_names must cover every resource")
        object.__setattr__(self, "resource_names", names)
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
                "tables_touched": tables_touched(self.resource_names),
            }
        )


def tables_touched(
    resource_names: Iterable[tuple[str, str]],
) -> tuple[dict[str, str], ...]:
    """Name the catalog tables a read touched, ordered for stable evidence."""

    return tuple(
        {"name": name, "resource_id": resource_id}
        for resource_id, name in sorted(
            dict(resource_names).items(), key=lambda item: (item[1], item[0])
        )
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
        timeout_seconds: float | None = None,
    ) -> SqlReadResult: ...


TABULAR_FILTER_OPERATORS = ("eq", "in", "lt", "lte", "gt", "gte")
TABULAR_READ_MAX_COLUMNS = 64
TABULAR_READ_MAX_FILTERS = 16
TABULAR_FILTER_MAX_VALUES = 100


def _tabular_scalar(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    return isinstance(value, (str, int, bool))


@dataclass(frozen=True, slots=True)
class TabularFilter:
    """One AND-combined predicate: equality, IN or one side of a range."""

    column: str
    operator: str
    value: object

    def __post_init__(self) -> None:
        if not isinstance(self.column, str) or not self.column.strip():
            raise ValueError("a filter needs a column name")
        if self.operator not in TABULAR_FILTER_OPERATORS:
            raise ValueError(
                "filter operator must be one of " + ", ".join(TABULAR_FILTER_OPERATORS)
            )
        if self.operator == "in":
            if (
                not isinstance(self.value, (tuple, list))
                or not 1 <= len(self.value) <= TABULAR_FILTER_MAX_VALUES
                or not all(_tabular_scalar(item) for item in self.value)
            ):
                raise ValueError(
                    "an in filter needs an array of 1 to "
                    f"{TABULAR_FILTER_MAX_VALUES} strings, numbers or booleans"
                )
            object.__setattr__(self, "value", tuple(self.value))
        elif not _tabular_scalar(self.value):
            raise ValueError(
                f"an {self.operator} filter needs one string, number or boolean"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> TabularFilter:
        return cls(
            cast(str, value["column"]), cast(str, value["operator"]), value["value"]
        )

    def to_payload(self) -> dict[str, object]:
        return {"column": self.column, "operator": self.operator, "value": self.value}


@dataclass(frozen=True, slots=True)
class ReadTabularRequest:
    """One bounded read of named columns from one catalog resource.

    The backend refuses the read when ``expected_revision`` is no longer the
    resource's current catalog revision.
    """

    agent_id: str
    source_id: str
    resource_id: str
    expected_revision: str
    columns: tuple[str, ...]
    filters: tuple[TabularFilter, ...]
    limit: int
    max_bytes: int
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.agent_id, "agent_id"),
            (self.source_id, "source_id"),
            (self.resource_id, "resource_id"),
            (self.expected_revision, "expected_revision"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"tabular read {name} must be non-empty text")
        columns = tuple(self.columns)
        if (
            not 1 <= len(columns) <= TABULAR_READ_MAX_COLUMNS
            or len(set(columns)) != len(columns)
            or any(not isinstance(item, str) or not item for item in columns)
        ):
            raise ValueError("tabular read columns must be unique non-empty names")
        filters = tuple(self.filters)
        if len(filters) > TABULAR_READ_MAX_FILTERS or any(
            not isinstance(item, TabularFilter) for item in filters
        ):
            raise ValueError("tabular read filters must be TabularFilter values")
        for bound, name in ((self.limit, "limit"), (self.max_bytes, "max_bytes")):
            if not isinstance(bound, int) or isinstance(bound, bool) or bound < 1:
                raise ValueError(f"tabular read {name} must be a positive integer")
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "filters", filters)


class TabularReadBackend(Protocol):
    async def read_tabular(
        self, request: ReadTabularRequest
    ) -> BoundedResultProjection: ...


SourceAdapterLookup = Callable[[str, str], Awaitable[str | None]]
_Backend = TypeVar("_Backend")


async def _source_backend(
    backends: Mapping[str, _Backend],
    source_adapter_id: SourceAdapterLookup,
    agent_id: str,
    source_id: str,
) -> _Backend:
    """Route one write to the backend of the source's registered type."""

    adapter_id = await source_adapter_id(agent_id, source_id)
    backend = None if adapter_id is None else backends.get(adapter_id)
    if backend is None:
        raise ValueError("relational write adapter binding is unsupported")
    return backend


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


DATA_QUERY_MAX_STATEMENTS = 8
# Resource safety ceiling for one foreground read; graph tasks keep their own.
DATA_QUERY_MAX_RESOURCES = 64
DATA_QUERY_MAX_ROWS = 100
DATA_QUERY_MAX_BYTES = 16_384
DATA_QUERY_STATEMENT_TIMEOUT_SECONDS = 5.0
DATA_QUERY_TOTAL_TIMEOUT_SECONDS = 15.0


class DataQueryError(DaitaError):
    """Stable model-visible failure of a bounded multi-statement read."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message, error_code=code)


class DataQueryExecutor:
    """Run catalog-bound read-only statements against one relational source.

    Every statement shares one row budget, one UTF-8 byte budget and one total
    deadline, and each statement also has its own statement deadline. All
    statements must observe the catalog revisions validated before dispatch.
    """

    executor_id = DATA_QUERY_EXECUTOR_ID

    def __init__(
        self,
        agent_id: str,
        backends: Mapping[str, SqlReadBackend],
        *,
        max_rows: int = DATA_QUERY_MAX_ROWS,
        max_bytes: int = DATA_QUERY_MAX_BYTES,
        statement_timeout_seconds: float = DATA_QUERY_STATEMENT_TIMEOUT_SECONDS,
        total_timeout_seconds: float = DATA_QUERY_TOTAL_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0 < statement_timeout_seconds <= total_timeout_seconds:
            raise ValueError("statement timeout must be positive and within total")
        self._agent_id = agent_id
        self._backends = dict(backends)
        self._max_rows = max_rows
        self._max_bytes = max_bytes
        self._statement_timeout_seconds = float(statement_timeout_seconds)
        self._total_timeout_seconds = float(total_timeout_seconds)
        self._clock = clock

    async def execute(self, request: ToolExecution) -> ToolOutput:
        source_id = request.arguments["source_id"]
        resource_ids = request.arguments["resource_ids"]
        adapter_id = request.arguments["_adapter_id"]
        assert isinstance(source_id, str)
        assert isinstance(resource_ids, tuple)
        assert isinstance(adapter_id, str)
        statements = data_query_statements(request.arguments)
        expected = _expected_resource_revisions(request.arguments)
        backend = self._backends.get(adapter_id)
        if backend is None:
            raise ValueError("relational query adapter binding is unsupported")
        started = self._clock()
        deadline = started + self._total_timeout_seconds
        remaining_rows = self._max_rows
        remaining_bytes = self._max_bytes
        results: list[tuple[SqlReadResult | None, int]] = []
        for sql, parameters in statements:
            if remaining_rows < 1 or remaining_bytes < 2:
                # The shared budget is spent; later statements are reported as
                # skipped rather than silently omitted.
                results.append((None, 0))
                continue
            remaining_seconds = deadline - self._clock()
            if remaining_seconds <= 0:
                raise DataQueryError(
                    "query_time_limit",
                    "The statements exceeded the total read deadline of "
                    f"{self._total_timeout_seconds:g} seconds.",
                )
            statement_started = self._clock()
            result = await backend.execute_read(
                agent_id=self._agent_id,
                source_id=source_id,
                sql=sql,
                parameters=parameters,
                max_rows=remaining_rows,
                max_bytes=remaining_bytes,
                expected_resource_revisions=expected,
                timeout_seconds=min(self._statement_timeout_seconds, remaining_seconds),
            )
            elapsed_ms = _elapsed_ms(statement_started, self._clock())
            if (
                not isinstance(result, SqlReadResult)
                or result.source_id != source_id
                or not frozenset(result.resource_ids) <= frozenset(resource_ids)
            ):
                raise ValueError("SQL backend returned different catalog binding facts")
            if results and any(
                earlier is not None
                and earlier.source_revision != result.source_revision
                for earlier, _ in results
            ):
                raise DataQueryError(
                    "query_binding_stale",
                    "The source changed between statements of one read.",
                )
            results.append((result, elapsed_ms))
            remaining_rows -= result.projection.returned_rows
            remaining_bytes -= result.projection.utf8_bytes
        executed = tuple(result for result, _ in results if result is not None)
        if frozenset(
            resource_id for result in executed for resource_id in result.resource_ids
        ) != frozenset(resource_ids):
            raise ValueError("SQL backend returned different catalog binding facts")
        timing = {
            "elapsed_ms": _elapsed_ms(started, self._clock()),
            "statement_timeout_ms": int(self._statement_timeout_seconds * 1_000),
            "total_timeout_ms": int(self._total_timeout_seconds * 1_000),
        }
        if len(statements) == 1:
            data = executed[0].tool_data().to_dict()
            data["adapter_id"] = adapter_id
            data["timing"] = timing
            return ToolOutput(kind=DATA_QUERY_EVIDENCE_KIND, data=data)
        return ToolOutput(
            kind=DATA_QUERY_EVIDENCE_KIND,
            data=_multi_statement_data(
                adapter_id,
                source_id,
                results,
                timing=timing,
                row_limit=self._max_rows,
                byte_limit=self._max_bytes,
            ),
        )


def data_query_statements(
    arguments: Mapping[str, object],
) -> tuple[tuple[str, tuple[object, ...]], ...]:
    """Return the ordered statements of one data_query call."""

    raw_statements = arguments.get("statements")
    if raw_statements is None:
        sql = arguments.get("sql")
        parameters = arguments.get("parameters", ())
        if not isinstance(sql, str) or not isinstance(parameters, tuple):
            raise ValueError("data_query requires sql or statements")
        return ((sql, parameters),)
    if not isinstance(raw_statements, tuple) or not raw_statements:
        raise ValueError("data_query statements must be a non-empty array")
    statements: list[tuple[str, tuple[object, ...]]] = []
    for item in raw_statements:
        if not isinstance(item, Mapping):
            raise ValueError("data_query statements must be objects")
        sql = item.get("sql")
        parameters = item.get("parameters", ())
        if not isinstance(sql, str) or not isinstance(parameters, tuple):
            raise ValueError("data_query statements need sql and parameters")
        statements.append((sql, parameters))
    return tuple(statements)


COMBINED_QUERY_MAX_INPUTS = 4
# Internal per-input ceilings the model never sees. An input that reaches
# either refuses the whole call: a join over incomplete input is never returned.
COMBINED_QUERY_INPUT_MAX_ROWS = 10_000
COMBINED_QUERY_INPUT_MAX_BYTES = 4 * 1_024 * 1_024


class CombinedQueryEngine(Protocol):
    async def query_relations(
        self,
        *,
        relations: tuple[QueryRelation, ...],
        canonical_sql: str,
        sql_fingerprint: str,
    ) -> FrozenJsonObject: ...


_ENGINE_ERRORS = {
    "folder_query_timeout": (
        "combined_query_timeout",
        "The combining query exceeded its bounded wall time.",
    ),
    "folder_query_limited": (
        "combined_query_limited",
        "The combining query exceeded a bounded worker limit.",
    ),
}


@dataclass(frozen=True, slots=True)
class _InputRead:
    """One complete input read and the lineage it contributes."""

    columns: tuple[str, ...]
    projection: BoundedResultProjection
    resource_revisions: tuple[tuple[str, str], ...]
    resource_names: tuple[tuple[str, str], ...]
    lineage: dict[str, object]


class DataQueryCombinedExecutor:
    """Join complete bounded per-source reads in the private DuckDB worker.

    Each input runs through its source's registry backend, as SQL or as a
    tabular read, under the catalog revisions validated before dispatch,
    within one shared input deadline.
    """

    executor_id = DATA_QUERY_COMBINED_EXECUTOR_ID

    def __init__(
        self,
        agent_id: str,
        backends: Mapping[str, SqlReadBackend],
        engine: CombinedQueryEngine,
        *,
        tabular_backends: Mapping[str, TabularReadBackend] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._agent_id = agent_id
        self._backends = dict(backends)
        self._tabular_backends = dict(tabular_backends or {})
        self._engine = engine
        self._clock = clock

    async def execute(self, request: ToolExecution) -> ToolOutput:
        inputs = cast(tuple[Mapping[str, object], ...], request.arguments["inputs"])
        started = self._clock()
        deadline = started + DATA_QUERY_TOTAL_TIMEOUT_SECONDS
        reads: list[tuple[str, _InputRead]] = []
        for item in inputs:
            alias = cast(str, item["alias"])
            remaining_seconds = deadline - self._clock()
            if remaining_seconds <= 0:
                raise DataQueryError(
                    "query_time_limit",
                    "The inputs exceeded the total read deadline of "
                    f"{DATA_QUERY_TOTAL_TIMEOUT_SECONDS:g} seconds.",
                )
            timeout = min(DATA_QUERY_STATEMENT_TIMEOUT_SECONDS, remaining_seconds)
            read = await (
                self._read_sql(item, timeout)
                if "sql" in item
                else self._read_tabular(item, timeout)
            )
            if read.projection.truncated:
                raise CapabilityInputError(
                    "combined_input_truncated",
                    f"Input {alias} returned more rows or bytes than can be "
                    "joined completely, so no result was returned.",
                    {
                        "alias": alias,
                        "source_id": item["source_id"],
                        "truncation_reasons": read.projection.truncation_reasons,
                    },
                    recovery=ToolRecovery(
                        RecoveryKind.CORRECT_ARGUMENTS,
                        f"Aggregate or filter inside input {alias} so it "
                        "returns fewer rows, then retry the combined query.",
                    ),
                )
            reads.append((alias, read))
        try:
            payload = await self._engine.query_relations(
                relations=tuple(
                    QueryRelation(alias, read.columns, read.projection.rows)
                    for alias, read in reads
                ),
                canonical_sql=cast(str, request.arguments["sql"]),
                sql_fingerprint=cast(str, request.arguments["_sql_fingerprint"]),
            )
        except LocalFileQueryError as error:
            code, message = _ENGINE_ERRORS.get(
                error.code,
                (
                    "combined_query_failed",
                    "The combining query could not run over the inputs; check "
                    "each alias's column names and types.",
                ),
            )
            raise DataQueryError(code, message) from error
        projection = BoundedResultProjection(
            rows=tuple(
                FrozenJsonObject.from_mapping(row)
                for row in cast(tuple[Mapping[str, object], ...], payload["rows"])
            ),
            total_rows=cast(int, payload["observed_rows"]),
            returned_rows=cast(int, payload["returned_rows"]),
            utf8_bytes=cast(int, payload["utf8_bytes"]),
            row_limit=cast(int, payload["row_limit"]),
            byte_limit=cast(int, payload["byte_limit"]),
            truncated=cast(bool, payload["truncated"]),
            truncation_reasons=cast(tuple[str, ...], payload["truncation_reasons"]),
        )
        revisions = dict(
            revision for _, read in reads for revision in read.resource_revisions
        )
        names = dict(name for _, read in reads for name in read.resource_names)
        data = projection.to_payload().to_dict()
        data.update(
            {
                "canonical_sql": payload["canonical_sql"],
                "columns": payload["columns"],
                "inputs": tuple(
                    {"alias": alias, **read.lineage} for alias, read in reads
                ),
                "resource_ids": tuple(sorted(revisions)),
                "resource_revisions": tuple(
                    {"resource_id": resource_id, "revision": revision}
                    for resource_id, revision in sorted(revisions.items())
                ),
                "sql_fingerprint": payload["sql_fingerprint"],
                "tables_touched": tables_touched(names.items()),
                "timing": {
                    "elapsed_ms": _elapsed_ms(started, self._clock()),
                    "statement_timeout_ms": int(
                        DATA_QUERY_STATEMENT_TIMEOUT_SECONDS * 1_000
                    ),
                    "total_timeout_ms": int(DATA_QUERY_TOTAL_TIMEOUT_SECONDS * 1_000),
                },
            }
        )
        return ToolOutput(kind=DATA_QUERY_COMBINED_EVIDENCE_KIND, data=data)

    async def _read_sql(self, item: Mapping[str, object], timeout: float) -> _InputRead:
        source_id = cast(str, item["source_id"])
        backend = self._backends.get(cast(str, item["_adapter_id"]))
        if backend is None:
            raise ValueError("relational query adapter binding is unsupported")
        result = await backend.execute_read(
            agent_id=self._agent_id,
            source_id=source_id,
            sql=cast(str, item["sql"]),
            parameters=(),
            max_rows=COMBINED_QUERY_INPUT_MAX_ROWS,
            max_bytes=COMBINED_QUERY_INPUT_MAX_BYTES,
            expected_resource_revisions=_expected_resource_revisions(item),
            timeout_seconds=timeout,
        )
        if (
            not isinstance(result, SqlReadResult)
            or result.source_id != source_id
            or frozenset(result.resource_ids)
            != frozenset(cast(tuple[str, ...], item["resource_ids"]))
        ):
            raise ValueError("SQL backend returned different catalog binding facts")
        return _InputRead(
            columns=result.columns,
            projection=result.projection,
            resource_revisions=result.resource_revisions,
            resource_names=result.resource_names,
            lineage={
                "source_id": result.source_id,
                "source_revision": result.source_revision,
                "canonical_sql": result.canonical_sql,
                "sql_fingerprint": result.sql_fingerprint,
                "resource_ids": result.resource_ids,
                "resource_revisions": _revision_payload(result.resource_revisions),
                "rows": result.projection.returned_rows,
            },
        )

    async def _read_tabular(
        self, item: Mapping[str, object], timeout: float
    ) -> _InputRead:
        backend = self._tabular_backends.get(cast(str, item["_adapter_id"]))
        if backend is None:
            raise ValueError("tabular read adapter binding is unsupported")
        resource_id = cast(str, item["resource_id"])
        ((_, revision),) = cast(
            tuple[tuple[str, str], ...], item["_resource_revisions"]
        )
        columns = cast(tuple[str, ...], item["columns"])
        filters = tuple(
            TabularFilter.from_mapping(cast(Mapping[str, object], value))
            for value in cast(tuple[object, ...], item.get("filters", ()))
        )
        projection = await backend.read_tabular(
            ReadTabularRequest(
                agent_id=self._agent_id,
                source_id=cast(str, item["source_id"]),
                resource_id=resource_id,
                expected_revision=revision,
                columns=columns,
                filters=filters,
                limit=COMBINED_QUERY_INPUT_MAX_ROWS,
                max_bytes=COMBINED_QUERY_INPUT_MAX_BYTES,
                timeout_seconds=timeout,
            )
        )
        if not isinstance(projection, BoundedResultProjection) or any(
            set(row) != set(columns) for row in projection.rows
        ):
            raise ValueError("tabular backend returned different columns")
        revisions = ((resource_id, revision),)
        return _InputRead(
            columns=columns,
            projection=projection,
            resource_revisions=revisions,
            resource_names=((resource_id, cast(str, item["_resource_name"])),),
            lineage={
                "source_id": item["source_id"],
                # The backend confirmed the bound revision, so the catalog
                # source revision bound with it is the one that was read.
                "source_revision": item["_source_revision"],
                "columns": columns,
                "filters": tuple(value.to_payload() for value in filters),
                "resource_ids": (resource_id,),
                "resource_revisions": _revision_payload(revisions),
                "rows": projection.returned_rows,
            },
        )


def _revision_payload(
    revisions: tuple[tuple[str, str], ...],
) -> tuple[dict[str, str], ...]:
    return tuple(
        {"resource_id": resource_id, "revision": revision}
        for resource_id, revision in revisions
    )


def _expected_resource_revisions(
    arguments: Mapping[str, object],
) -> tuple[tuple[str, str], ...] | None:
    raw = arguments.get("_resource_revisions")
    if raw is None:
        return None
    if not isinstance(raw, tuple):
        raise ValueError("validated resource revisions are malformed")
    revisions: list[tuple[str, str]] = []
    for item in raw:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not all(isinstance(value, str) for value in item)
        ):
            raise ValueError("validated resource revisions are malformed")
        revisions.append((item[0], item[1]))
    return tuple(revisions)


def _elapsed_ms(started: float, finished: float) -> int:
    return max(0, int(round((finished - started) * 1_000)))


def _multi_statement_data(
    adapter_id: str,
    source_id: str,
    results: list[tuple[SqlReadResult | None, int]],
    *,
    timing: Mapping[str, int],
    row_limit: int,
    byte_limit: int,
) -> dict[str, object]:
    statements: list[dict[str, object]] = []
    revisions: dict[str, str] = {}
    names: dict[str, str] = {}
    reasons: list[str] = []
    source_revision = ""
    for index, (result, elapsed_ms) in enumerate(results):
        if result is None:
            statements.append({"index": index, "status": "skipped_budget_spent"})
            reasons.append("budget_spent")
            continue
        source_revision = result.source_revision
        revisions.update(result.resource_revisions)
        names.update(result.resource_names)
        payload = result.tool_data().to_dict()
        for name in ("source_id", "source_revision", "resource_revisions"):
            payload.pop(name)
        for name in ("row_limit", "byte_limit"):
            payload.pop(name)
        payload["elapsed_ms"] = elapsed_ms
        payload["index"] = index
        payload["status"] = "completed"
        statements.append(payload)
        reasons.extend(result.projection.truncation_reasons)
    executed = [result for result, _ in results if result is not None]
    fingerprint = canonical_json([result.sql_fingerprint for result in executed])
    return {
        "adapter_id": adapter_id,
        "byte_limit": byte_limit,
        "resource_ids": tuple(sorted(revisions)),
        "resource_revisions": tuple(
            {"resource_id": resource_id, "revision": revision}
            for resource_id, revision in sorted(revisions.items())
        ),
        "returned_rows": sum(result.projection.returned_rows for result in executed),
        "row_limit": row_limit,
        "source_id": source_id,
        "source_revision": source_revision,
        "sql_fingerprint": "sha256:"
        + hashlib.sha256(fingerprint.encode("utf-8")).hexdigest(),
        "statements": tuple(statements),
        "tables_touched": tables_touched(names.items()),
        "timing": dict(timing),
        "total_rows": sum(result.projection.total_rows for result in executed),
        "truncated": bool(reasons),
        "truncation_reasons": tuple(dict.fromkeys(reasons)),
        "trust_classification": "untrusted_external_data",
        "utf8_bytes": sum(result.projection.utf8_bytes for result in executed),
    }


class RelationalUpdatePreviewExecutor:
    executor_id = RELATIONAL_UPDATE_PREVIEW_EXECUTOR_ID

    def __init__(
        self,
        agent_id: str,
        backends: Mapping[str, RelationalUpdatePreviewBackend],
        source_adapter_id: SourceAdapterLookup,
    ) -> None:
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("preview executor agent_id must be non-empty text")
        if any(
            not callable(getattr(backend, "preview_update", None))
            for backend in backends.values()
        ):
            raise TypeError("preview backend must provide preview_update")
        self._agent_id = agent_id
        self._backends = dict(backends)
        self._source_adapter_id = source_adapter_id

    async def execute(self, request: ToolExecution) -> ToolOutput:
        intent = RelationalUpdateIntent.from_mapping(request.arguments)
        backend = await _source_backend(
            self._backends, self._source_adapter_id, self._agent_id, intent.source_id
        )
        result = await backend.preview_update(
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

    def __init__(
        self,
        agent_id: str,
        backends: Mapping[str, RelationalUpdateBackend],
        source_adapter_id: SourceAdapterLookup,
    ) -> None:
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("update executor agent_id must be non-empty text")
        if any(
            not callable(getattr(backend, "preview_update", None))
            or not callable(getattr(backend, "execute_update", None))
            for backend in backends.values()
        ):
            raise TypeError(
                "update backend must provide preview_update and execute_update"
            )
        self._agent_id = agent_id
        self._backends = dict(backends)
        self._source_adapter_id = source_adapter_id

    async def _backend(self, source_id: str) -> RelationalUpdateBackend:
        return await _source_backend(
            self._backends, self._source_adapter_id, self._agent_id, source_id
        )

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        command = RelationalUpdateCommand.from_mapping(request.arguments)
        backend = await self._backend(command.intent.source_id)
        preview = await backend.preview_update(
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
                recovery=_REPREVIEW,
            )
        if preview.fingerprint.preview_fingerprint != command.preview_fingerprint:
            raise CapabilityInputError(
                "write_preview_stale",
                "The supplied preview fingerprint is not the exact current preview.",
                recovery=_REPREVIEW,
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
        backend = await self._backend(command.intent.source_id)
        result = await backend.execute_update(
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
    backends: Mapping[str, SqlReadBackend],
    *,
    file_outputs: tuple[str, ...] = (),
    combined_engine: CombinedQueryEngine | None = None,
    tabular_backends: Mapping[str, TabularReadBackend] | None = None,
    path_backends: Mapping[str, PathPatternReadBackend] | None = None,
) -> DataQueryDeclarations:
    """Declare data_query over SQL backends.

    Path-pattern backends add the ``path_pattern`` route; a combined engine
    adds the multi-source ``inputs`` route.
    """

    executors: tuple[Executor, ...] = (DataQueryExecutor(agent_id, backends),)
    if path_backends:
        executors += (PathQueryExecutor(agent_id, path_backends),)
    if combined_engine is not None:
        executors += (
            DataQueryCombinedExecutor(
                agent_id,
                backends,
                combined_engine,
                tabular_backends=tabular_backends,
            ),
        )
    declarations = data_query_capability_declarations(
        file_outputs=file_outputs,
        file_capability_id=DATA_EXPORT_TABULAR_CAPABILITY_ID if file_outputs else None,
        combined=combined_engine is not None,
        path_query=bool(path_backends),
    )
    return DataQueryDeclarations(
        declarations.capabilities, executors, declarations.tool_views
    )


def relational_update_preview_declarations(
    agent_id: str,
    backends: Mapping[str, RelationalUpdatePreviewBackend],
    source_adapter_id: SourceAdapterLookup,
) -> RelationalUpdatePreviewDeclarations:
    executor = RelationalUpdatePreviewExecutor(agent_id, backends, source_adapter_id)
    declarations = relational_update_preview_capability_declarations()
    return RelationalUpdatePreviewDeclarations(
        declarations.capabilities,
        (executor,),
        declarations.tool_views,
    )


def relational_update_declarations(
    agent_id: str,
    backends: Mapping[str, RelationalUpdateBackend],
    source_adapter_id: SourceAdapterLookup,
) -> RelationalUpdateDeclarations:
    executor = RelationalUpdateExecutor(agent_id, backends, source_adapter_id)
    declarations = relational_update_capability_declarations()
    return RelationalUpdateDeclarations(
        declarations.capabilities,
        (executor,),
        declarations.tool_views,
    )


def data_query_capability_declarations(
    *,
    file_outputs: tuple[str, ...] = (),
    file_capability_id: str | None = None,
    combined: bool = False,
    path_query: bool = False,
) -> CapabilityDeclarations:
    """Declare data_query; file outputs route to the exact export capability.

    A call that supplies ``inputs`` routes to the combined multi-source read,
    and one that supplies ``path_pattern`` to the live folder-file query.
    """

    return _query_declarations(
        ToolPresentation(
            toolbox_id=ToolboxId.SOURCES,
            load_mode=ToolLoadMode.PINNED,
            text_trust=ToolTextTrust.CODE,
            summary=(
                "Run one bounded validated read-only relational query, or export "
                "one exact complete query result as a CSV or XLSX file."
                if file_outputs
                else "Run one bounded validated read-only relational query."
            ),
            when_to_use=(
                "Use exact source, table and column names from current catalog evidence; "
                "obtain missing structure only through admitted, callable tools."
                + (
                    " Exports cover all database rows the query selects."
                    if file_outputs
                    else ""
                )
            ),
            keywords=(
                "data",
                "query",
                "relational",
                "sql",
                "profile",
                *(("join", "combine") if combined else ()),
                *(
                    (
                        "export",
                        "csv",
                        "xlsx",
                        "database",
                        "complete dataset",
                        "all rows",
                    )
                    if file_outputs
                    else ()
                ),
            ),
        ),
        file_outputs=file_outputs,
        file_capability_id=file_capability_id,
        combined=combined,
        path_query=path_query,
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
                    "pattern": SOURCE_ID_PATTERN,
                },
                "resource_id": {
                    "type": "string",
                    "pattern": RESOURCE_ID_PATTERN,
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
    return CapabilityDeclarations(
        domain_owner_id="data",
        capabilities=(capability,),
        executor_ids=(capability.executor_id,),
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
                    "pattern": SOURCE_ID_PATTERN,
                },
                "resource_id": {
                    "type": "string",
                    "pattern": RESOURCE_ID_PATTERN,
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
        description=RELATIONAL_UPDATE_DESCRIPTION,
        mode_capability_ids=(
            ("preview", RELATIONAL_UPDATE_PREVIEW_CAPABILITY_ID),
            ("apply", capability.id),
        ),
        presentation=ToolPresentation(
            toolbox_id=ToolboxId.SOURCES,
            load_mode=ToolLoadMode.ON_DEMAND,
            text_trust=ToolTextTrust.CODE,
            summary="Preview, then apply, an exact update to existing relational rows.",
            when_to_use="Use mode=preview first; apply that exact preview with approval or a standing grant.",
            keywords=(
                "data",
                "relational",
                "postgresql",
                "rows",
                "update",
                "preview",
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
    *,
    file_outputs: tuple[str, ...] = (),
    file_capability_id: str | None = None,
    combined: bool = False,
    path_query: bool = False,
) -> CapabilityDeclarations:
    if bool(file_outputs) != (file_capability_id is not None):
        raise ValueError("data_query file outputs require their export capability")
    sql_rule = {"type": "string", "minLength": 1, "maxLength": MAX_SQL_CHARACTERS}
    parameters_rule = {
        "type": "array",
        "maxItems": MAX_SQL_PARAMETERS,
        "description": "SQLite ? values follow SQL occurrence order; PostgreSQL $1, $2, ... by index. Verify filter bindings.",
    }
    capability = Capability(
        id=DATA_QUERY_CAPABILITY_ID,
        description=_DATA_QUERY_CAPABILITY_DESCRIPTION,
        input_schema={
            "type": "object",
            "properties": {
                "source_id": source_id_schema(),
                # The tables a read touches come from the SQL; the controller
                # records them here for the executor. readOnly keeps the
                # argument out of the model-facing schema.
                "resource_ids": {
                    "type": "array",
                    "items": resource_id_schema(),
                    "minItems": 1,
                    "maxItems": DATA_QUERY_MAX_RESOURCES,
                    "uniqueItems": True,
                    "readOnly": True,
                },
                "sql": sql_rule,
                "parameters": parameters_rule,
                "statements": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": DATA_QUERY_MAX_STATEMENTS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "sql": sql_rule,
                            "parameters": parameters_rule,
                        },
                        "required": ["sql"],
                        "additionalProperties": False,
                    },
                    "description": "Use instead of sql to run several reads in one call.",
                },
            },
            "required": ["source_id"],
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
            inline_max_targets=DATA_QUERY_MAX_RESOURCES,
            graph_max_targets=16,
        ),
    )
    alias_rule = {"type": "string", "pattern": r"^[a-z][a-z0-9_]{0,31}$"}
    column_rule = {"type": "string", "minLength": 1, "maxLength": 256}
    resource_ids_rule = {
        "type": "array",
        "items": resource_id_schema(),
        "minItems": 1,
        "maxItems": DATA_QUERY_MAX_RESOURCES,
        "uniqueItems": True,
        "readOnly": True,
    }
    combined_capability = Capability(
        id=DATA_QUERY_COMBINED_CAPABILITY_ID,
        description=_DATA_QUERY_COMBINED_CAPABILITY_DESCRIPTION,
        input_schema={
            "type": "object",
            "properties": {
                "inputs": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": COMBINED_QUERY_MAX_INPUTS,
                    "items": {
                        "oneOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "alias": alias_rule,
                                    "source_id": source_id_schema(),
                                    "resource_ids": resource_ids_rule,
                                    "sql": sql_rule,
                                },
                                "required": [
                                    "alias",
                                    "source_id",
                                    "sql",
                                ],
                                "additionalProperties": False,
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "alias": alias_rule,
                                    "source_id": source_id_schema(),
                                    "resource_id": resource_id_schema(),
                                    "columns": {
                                        "type": "array",
                                        "items": column_rule,
                                        "minItems": 1,
                                        "maxItems": TABULAR_READ_MAX_COLUMNS,
                                        "uniqueItems": True,
                                    },
                                    "filters": {
                                        "type": "array",
                                        "maxItems": TABULAR_READ_MAX_FILTERS,
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "column": column_rule,
                                                "operator": {
                                                    "type": "string",
                                                    "enum": list(
                                                        TABULAR_FILTER_OPERATORS
                                                    ),
                                                },
                                                "value": {},
                                            },
                                            "required": ["column", "operator", "value"],
                                            "additionalProperties": False,
                                        },
                                    },
                                },
                                "required": [
                                    "alias",
                                    "source_id",
                                    "resource_id",
                                    "columns",
                                ],
                                "additionalProperties": False,
                            },
                        ]
                    },
                    "description": (
                        "Use with sql as one DuckDB SELECT over these aliases "
                        "to join sources. Each input is either sql in its own "
                        "source's dialect, or exact catalog columns of one "
                        "resource_id with optional AND-combined filters "
                        "(eq, in, lt, lte, gt, gte)."
                    ),
                },
                "sql": sql_rule,
            },
            "required": ["inputs", "sql"],
            "additionalProperties": False,
        },
        output_kind=DATA_QUERY_COMBINED_EVIDENCE_KIND,
        output_schema=_combined_query_output_schema(),
        executor_id=DATA_QUERY_COMBINED_EXECUTOR_ID,
        access_mode=AccessMode.READ,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
    )
    path_capability = path_query_capability()
    presence_modes = (
        *(("inputs",) if combined else ()),
        *(("path_pattern",) if path_query else ()),
    )
    routes: tuple[tuple[str, str], ...] = (
        ("rows", capability.id),
        *((output, cast(str, file_capability_id)) for output in file_outputs),
        *((("inputs", combined_capability.id),) if combined else ()),
        *((("path_pattern", path_capability.id),) if path_query else ()),
    )
    routed = len(routes) > 1
    view = ToolView(
        name=DATA_QUERY_TOOL_NAME,
        capability_id=capability.id,
        description=_data_query_description(file_outputs),
        presentation=presentation,
        mode_capability_ids=routes if routed else (),
        mode_argument="output" if routed else "mode",
        default_mode="rows" if routed else None,
        presence_modes=presence_modes,
        route_notes=(
            *(
                ((combined_capability.id, _DATA_QUERY_COMBINED_USAGE),)
                if combined
                else ()
            ),
            *(((path_capability.id, _DATA_QUERY_PATH_USAGE),) if path_query else ()),
        ),
        # The description states the input shape; the full item rules stay in
        # the capability contract.
        summarized_arguments=("inputs",) if combined else (),
    )
    return CapabilityDeclarations(
        domain_owner_id="data",
        capabilities=(
            capability,
            *((combined_capability,) if combined else ()),
            *((path_capability,) if path_query else ()),
        ),
        executor_ids=(
            DATA_QUERY_EXECUTOR_ID,
            *((DATA_QUERY_COMBINED_EXECUTOR_ID,) if combined else ()),
            *((PATH_QUERY_EXECUTOR_ID,) if path_query else ()),
        ),
        tool_views=(view,),
    )


def _combined_query_output_schema() -> dict[str, object]:
    types = {
        "byte_limit": "integer",
        "canonical_sql": "string",
        "columns": "array",
        "inputs": "array",
        "resource_ids": "array",
        "resource_revisions": "array",
        "returned_rows": "integer",
        "row_limit": "integer",
        "rows": "array",
        "sql_fingerprint": "string",
        "tables_touched": "array",
        "timing": "object",
        "total_rows": "integer",
        "truncated": "boolean",
        "truncation_reasons": "array",
        "trust_classification": "string",
        "utf8_bytes": "integer",
    }
    return {
        "type": "object",
        "properties": {name: {"type": kind} for name, kind in sorted(types.items())},
        "required": sorted(types),
        "additionalProperties": False,
    }


def _query_output_schema() -> dict[str, object]:
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
        "statements": "array",
        "tables_touched": "array",
        "timing": "object",
        "total_rows": "integer",
        "truncated": "boolean",
        "truncation_reasons": "array",
        "trust_classification": "string",
        "utf8_bytes": "integer",
    }
    # One statement keeps its columns, rows and SQL at the top level; several
    # statements report them per entry under statements.
    per_statement = {"canonical_sql", "columns", "rows", "statements"}
    return {
        "type": "object",
        "properties": {name: {"type": kind} for name, kind in sorted(types.items())},
        "required": sorted(set(types) - per_statement),
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
    "RELATIONAL_UPDATE_EXECUTOR_ID",
    "RELATIONAL_UPDATE_TOOL_NAME",
    "RELATIONAL_UPSERT_TOOL_NAME",
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
    "SQLiteReadResult",
    "ReadTabularRequest",
    "TABULAR_FILTER_OPERATORS",
    "TabularFilter",
    "TabularReadBackend",
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
        "description": "Only for routine proposals: allowed_capability_ids lists both "
        "routes of the one write tool: "
        + "; ".join(
            f"{tool_name} needs {capability_id} and "
            f"{native_preview_capability(capability_id)}"
            for tool_name, capability_id in (
                (RELATIONAL_UPDATE_TOOL_NAME, RELATIONAL_UPDATE_CAPABILITY_ID),
                (RELATIONAL_UPSERT_TOOL_NAME, "data.upsert_rows"),
            )
        )
        + ". These are its mode=apply and mode=preview routes, not separate tools "
        "to load. Each occurrence calls mode=preview before mode=apply.",
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


def _upsert_intent(request: ToolExecution) -> RelationalUpsertIntent:
    """Bind the model's batch to the evidence the runtime derived for it."""

    return replace(
        RelationalUpsertIntent.from_mapping(request.arguments),
        evidence_call_ids=request.provenance.evidence_call_ids,
    )


class _RelationalUpsertRouting:
    def __init__(
        self,
        agent_id: str,
        backends: Mapping[str, RelationalUpsertBackend],
        source_adapter_id: SourceAdapterLookup,
    ) -> None:
        self._agent_id = agent_id
        self._backends = dict(backends)
        self._source_adapter_id = source_adapter_id

    async def _backend(self, intent: RelationalUpsertIntent) -> RelationalUpsertBackend:
        return await _source_backend(
            self._backends, self._source_adapter_id, self._agent_id, intent.source_id
        )


class RelationalUpsertPreviewExecutor(_RelationalUpsertRouting):
    executor_id = "data.preview_upsert_rows.executor"

    async def execute(self, request: ToolExecution) -> ToolOutput:
        intent = _upsert_intent(request)
        backend = await self._backend(intent)
        return ToolOutput(
            kind="data.preview_upsert_rows",
            data=await backend.preview_upsert(agent_id=self._agent_id, intent=intent),
        )


class RelationalUpsertExecutor(_RelationalUpsertRouting):
    executor_id = "data.upsert_rows.executor"

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        intent = _upsert_intent(request)
        backend = await self._backend(intent)
        preview = await backend.preview_upsert(agent_id=self._agent_id, intent=intent)
        if preview["preview_fingerprint"] != request.arguments["preview_fingerprint"]:
            raise CapabilityInputError(
                "write_preview_stale",
                "The exact upsert preview changed before dispatch.",
                recovery=_REPREVIEW,
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
        intent = _upsert_intent(request)
        backend = await self._backend(intent)
        result = await backend.execute_upsert(
            agent_id=self._agent_id,
            execution=request,
            intent=intent,
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
        "source_id": {"type": "string", "pattern": SOURCE_ID_PATTERN},
        "resource_id": {
            "type": "string",
            "pattern": RESOURCE_ID_PATTERN,
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
            "earlier reads of the same table in this run are recorded as evidence, not proof of truth. Identity values are allocated only on insert. "
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
        derives_call_provenance=True,
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
        derives_call_provenance=True,
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
        tool_views=(
            ToolView(
                name=RELATIONAL_UPSERT_TOOL_NAME,
                capability_id=write.id,
                description=RELATIONAL_UPSERT_DESCRIPTION,
                mode_capability_ids=(("preview", preview.id), ("apply", write.id)),
                presentation=ToolPresentation(
                    toolbox_id=ToolboxId.SOURCES,
                    load_mode=ToolLoadMode.ON_DEMAND,
                    text_trust=ToolTextTrust.CODE,
                    summary=(
                        "Preview, then apply, an exact relational batch: insert missing "
                        "rows, update existing rows, or leave them unchanged."
                    ),
                    when_to_use=(
                        "Use mode=preview first for a batch identified by exact conflict "
                        "keys; apply that exact preview with approval or a standing grant."
                    ),
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
                        "preview",
                        "apply",
                        "save",
                        "approval",
                    ),
                ),
            ),
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
    agent_id: str,
    backends: Mapping[str, RelationalUpsertBackend],
    source_adapter_id: SourceAdapterLookup,
) -> DataQueryDeclarations:
    declarations = relational_upsert_capability_declarations()
    return DataQueryDeclarations(
        declarations.capabilities,
        (
            RelationalUpsertPreviewExecutor(agent_id, backends, source_adapter_id),
            RelationalUpsertExecutor(agent_id, backends, source_adapter_id),
        ),
        declarations.tool_views,
    )
