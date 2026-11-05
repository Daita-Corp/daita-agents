"""Declare and execute SQL reads and structured PostgreSQL updates."""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, TypeVar, cast

from ..._json import FrozenJsonObject, canonical_json
from ...adapters.local_file_query import LocalFileQueryError, QueryRelation
from ...artifacts.renderers import MAX_CSV_ROWS, MAX_CSV_SECONDS
from ...capabilities import (
    SOURCE_ID_PATTERN,
    AccessMode,
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
    recovery_details,
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
    RELATIONAL_WRITE_CAPABILITY_ID,
    RELATIONAL_WRITE_EVIDENCE_KIND,
    RELATIONAL_WRITE_PREVIEW_CAPABILITY_ID,
    RELATIONAL_WRITE_PREVIEW_EVIDENCE_KIND,
    WriteDescriber,
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
    CheckedRead,
    WriteFacts,
    WritePreview,
    checked_reads,
    write_facts,
)

DATA_QUERY_EXECUTOR_ID = "data.query.executor"
DATA_QUERY_COMBINED_EXECUTOR_ID = "data.query_combined.executor"
DATA_QUERY_TOOL_NAME = "data_query"
RELATIONAL_WRITE_PREVIEW_EXECUTOR_ID = "data.preview_write.executor"
RELATIONAL_WRITE_EXECUTOR_ID = "data.write.executor"
RELATIONAL_WRITE_TOOL_NAME = "data_write"
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


COMBINED_QUERY_MAX_INPUTS = 16
# One budget shared by every input, enforced while the reads run. An input that
# cannot fit refuses the whole call: a join over incomplete input is never returned.
COMBINED_QUERY_TOTAL_MAX_ROWS = 50_000
COMBINED_QUERY_TOTAL_MAX_BYTES = 32 * 1_024 * 1_024
_DATA_QUERY_COMBINED_CAPABILITY_DESCRIPTION = (
    f"Join or aggregate data from 2 to {COMBINED_QUERY_MAX_INPUTS} sources in "
    "one bounded read. Each input runs read-only SQL in its own source's dialect "
    "on exact tables, or reads exact columns of one table with optional eq, in "
    "and range filters; sql is one DuckDB SELECT or UNION, at most 8 KiB, that "
    "reads every input alias and nothing else. Limits: 5s per input, 15s for all "
    f"inputs, {COMBINED_QUERY_TOTAL_MAX_ROWS:,} rows shared by all inputs, 100 "
    "rows and 16 KB in the result. An input too large to join completely "
    "refuses the call, so aggregate or filter inside each input. The result "
    "reports join match counts."
)
_DATA_QUERY_COMBINED_USAGE = (
    f" Join sources: inputs is 2-{COMBINED_QUERY_MAX_INPUTS} {{alias, source, "
    "sql} in each source's dialect, with path_pattern for a folder, or {alias, "
    "source, table, columns, filters} for a source without SQL; sql is one "
    "DuckDB SELECT or UNION over the aliases (8 KiB). Inputs share "
    f"{COMBINED_QUERY_TOTAL_MAX_ROWS:,} rows; aggregate or filter in each "
    "input: one too large refuses the call. The result reports join match counts."
)
_DATA_QUERY_PATH_USAGE = (
    " Folder files: query relation data by path_pattern, matched against the "
    "folder's current files."
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


RELATIONAL_WRITE_DESCRIPTION = (
    "Write to a PostgreSQL table with one SQL statement in one call: INSERT with a "
    "column list, UPDATE or DELETE with WHERE, including INSERT ... SELECT, ON "
    "CONFLICT (columns), UPDATE ... FROM and DELETE ... USING. Daita previews the "
    "rows on the approval card, rechecks them and writes once. mode=preview only "
    "shows the count and samples. Only outcome=committed proves it. Limits: one "
    'table with write permission, no RETURNING, 64 KiB. Example: {"sql": "UPDATE '
    'orders SET status = $1 WHERE id = $2", "parameters": ["paid", 7]}. Mistakes: '
    "asking for approval in chat; retrying after denial."
)
RELATIONAL_WRITE_RECEIPT_POLICY = EffectReceiptPolicy(
    receipt_kind="data.write",
    success_evidence_basis=EffectEvidenceBasis.ADAPTER_VERIFIED,
    payload_schema={
        "type": "object",
        "properties": {
            "source_id": {"type": "string", "maxLength": 256},
            "resource_id": {"type": "string", "maxLength": 256},
            "table": {"type": "string", "maxLength": 1024},
            "operation": {"type": "string", "enum": ["insert", "update", "delete"]},
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
            "table",
            "operation",
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
    """One read's facts: the tables come from the parse, the rest from the source."""

    source_id: str
    canonical_sql: str
    sql_fingerprint: str
    resource_ids: tuple[str, ...]
    columns: tuple[str, ...]
    projection: BoundedResultProjection
    resource_names: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for value, name in (
            (self.source_id, "source_id"),
            (self.canonical_sql, "canonical_sql"),
            (self.sql_fingerprint, "sql_fingerprint"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        resource_ids = tuple(self.resource_ids)
        columns = tuple(self.columns)
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError("resource_ids must be unique")
        if any(not isinstance(item, str) or not item for item in columns):
            raise ValueError("columns must contain non-empty strings")
        if not isinstance(self.projection, BoundedResultProjection):
            raise TypeError("projection must be BoundedResultProjection")
        names = tuple(sorted(tuple(item) for item in self.resource_names))
        if names and {item[0] for item in names} != set(resource_ids):
            raise ValueError("resource_names must cover every resource")
        object.__setattr__(self, "resource_names", names)
        object.__setattr__(self, "resource_ids", resource_ids)
        object.__setattr__(self, "columns", columns)

    def tool_data(self) -> FrozenJsonObject:
        projection = self.projection.to_payload().to_dict()
        return FrozenJsonObject.from_mapping(
            {
                **projection,
                "canonical_sql": self.canonical_sql,
                "columns": self.columns,
                "resource_ids": self.resource_ids,
                "source_id": self.source_id,
                "sql_fingerprint": self.sql_fingerprint,
                "tables_touched": tables_touched(self.resource_names),
            }
        )


def tables_touched(
    resource_names: Iterable[tuple[str, str]],
) -> tuple[dict[str, str], ...]:
    """Name the tables a read touched, ordered for stable evidence."""

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


_CORRECT_READ = ToolRecovery(
    RecoveryKind.CORRECT_ARGUMENTS,
    "Fix the SQL using this message; the same call fails the same way.",
)


def refused_read_details(details: Mapping[str, object]) -> dict[str, object]:
    """Details for a read the source refused before reading a row.

    Marking it undispatched lets the loop warn when the model repeats it.
    """

    return recovery_details(
        _CORRECT_READ, {**details, "execution_disposition": "not_dispatched"}
    )


class SqlReadBackend(Protocol):
    async def execute_read(
        self,
        *,
        agent_id: str,
        source_id: str,
        read: CheckedRead,
        max_rows: int,
        max_bytes: int,
        timeout_seconds: float | None = None,
    ) -> SqlReadResult: ...


RefreshRequest = Callable[[str, tuple[str, ...]], None]


async def _execute_read(
    backend: SqlReadBackend,
    request_refresh: RefreshRequest,
    *,
    source_id: str,
    read: CheckedRead,
    **arguments: Any,
) -> SqlReadResult:
    """Run one read; a missing object asks for a catalog refresh and still fails."""

    try:
        return await backend.execute_read(source_id=source_id, read=read, **arguments)
    except Exception as error:
        if getattr(error, "undefined_object", False):
            request_refresh(source_id, tuple(name for _, name in read.resource_names))
        raise


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
    """One bounded read of named columns from one catalog resource."""

    agent_id: str
    source_id: str
    resource_id: str
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


@dataclass(frozen=True, slots=True)
class DataQueryDeclarations:
    capabilities: tuple[Capability, ...]
    executors: tuple[Executor, ...]
    tool_views: tuple[ToolView, ...]


DATA_QUERY_MAX_STATEMENTS = 8
# Resource safety ceiling for one foreground read; job runs keep their own.
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
    """Run read-only statements that passed policy against one relational source.

    Every statement shares one row budget, one UTF-8 byte budget and one total
    deadline, and each statement also has its own statement deadline.
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
        wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        request_refresh: RefreshRequest = lambda source_id, names: None,
    ) -> None:
        if not 0 < statement_timeout_seconds <= total_timeout_seconds:
            raise ValueError("statement timeout must be positive and within total")
        self._agent_id = agent_id
        self._request_refresh = request_refresh
        self._backends = dict(backends)
        self._max_rows = max_rows
        self._max_bytes = max_bytes
        self._statement_timeout_seconds = float(statement_timeout_seconds)
        self._total_timeout_seconds = float(total_timeout_seconds)
        self._clock = clock
        self._wall_clock = wall_clock

    async def execute(self, request: ToolExecution) -> ToolOutput:
        source_id = request.arguments["source_id"]
        adapter_id = request.arguments["_adapter_id"]
        assert isinstance(source_id, str)
        assert isinstance(adapter_id, str)
        reads = checked_reads(request.arguments)
        backend = self._backends.get(adapter_id)
        if backend is None:
            raise ValueError("relational query adapter binding is unsupported")
        started = self._clock()
        deadline = started + self._total_timeout_seconds
        remaining_rows = self._max_rows
        remaining_bytes = self._max_bytes
        results: list[tuple[SqlReadResult | None, int, str]] = []
        for read in reads:
            if remaining_rows < 1 or remaining_bytes < 2:
                # The shared budget is spent; later statements are reported as
                # skipped rather than silently omitted.
                results.append((None, 0, ""))
                continue
            remaining_seconds = deadline - self._clock()
            if remaining_seconds <= 0:
                raise DataQueryError(
                    "query_time_limit",
                    "The statements exceeded the total read deadline of "
                    f"{self._total_timeout_seconds:g} seconds.",
                )
            statement_started = self._clock()
            read_at = self._wall_clock().isoformat()
            result = await _execute_read(
                backend,
                self._request_refresh,
                agent_id=self._agent_id,
                source_id=source_id,
                read=read,
                max_rows=remaining_rows,
                max_bytes=remaining_bytes,
                timeout_seconds=min(self._statement_timeout_seconds, remaining_seconds),
            )
            elapsed_ms = _elapsed_ms(statement_started, self._clock())
            if not isinstance(result, SqlReadResult) or result.source_id != source_id:
                raise ValueError("SQL backend returned a different source")
            results.append((result, elapsed_ms, read_at))
            remaining_rows -= result.projection.returned_rows
            remaining_bytes -= result.projection.utf8_bytes
        executed = tuple(result for result, _, _ in results if result is not None)
        timing = {
            "elapsed_ms": _elapsed_ms(started, self._clock()),
            "statement_timeout_ms": int(self._statement_timeout_seconds * 1_000),
            "total_timeout_ms": int(self._total_timeout_seconds * 1_000),
        }
        if len(reads) == 1:
            data = executed[0].tool_data().to_dict()
            data["adapter_id"] = adapter_id
            data["read_at"] = results[0][2]
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
    """One input read and the lineage it contributes."""

    columns: tuple[str, ...]
    projection: BoundedResultProjection
    resource_ids: tuple[str, ...]
    resource_names: tuple[tuple[str, str], ...]
    lineage: dict[str, object]


class _ReadBudget:
    """Rows and bytes not yet granted to a running or finished read.

    Outstanding grants plus settled usage never exceed the total, so memory is
    bounded while the reads run. One event loop owns it: no lock.
    """

    def __init__(self, rows: int, size: int) -> None:
        self._rows = rows
        self._bytes = size

    def grant(self, waiting: int) -> tuple[int, int]:
        """Reserve an equal share of what is left for one of ``waiting`` reads."""

        share = (self._rows // waiting, self._bytes // waiting)
        self._rows -= share[0]
        self._bytes -= share[1]
        return share

    def settle(self, granted: tuple[int, int], used: tuple[int, int]) -> None:
        """Return the part of a grant its read did not use."""

        self._rows += granted[0] - used[0]
        self._bytes += granted[1] - used[1]


# LocalFileQueryLimits rejects a shorter query time.
_MIN_READ_SECONDS = 0.01
_Attempts = dict[int, tuple["_InputRead | None", tuple[int, int], tuple[str, ...]]]
_READ_CONSISTENCY = (
    "Inputs were read separately at the read_at times shown, so they are not one "
    "consistent snapshot."
)
_NUMERIC_TYPES = (
    "BIGINT",
    "INTEGER",
    "HUGEINT",
    "SMALLINT",
    "TINYINT",
    "DOUBLE",
    "FLOAT",
    "DECIMAL",
)


def _type_family(name: str) -> str:
    upper = name.upper()
    return "NUMERIC" if upper.startswith(_NUMERIC_TYPES) else upper


def _join_warnings(checks: object, status: object) -> tuple[str, ...]:
    """Advisory notes on how the joins matched; none of them blocks the result."""

    warnings: list[str] = []
    for check in cast(tuple[Mapping[str, object], ...], checks or ()):
        left, right = check.get("left"), check.get("right")
        parts = []
        for side, key in (("left", left), ("right", right)):
            unmatched = cast(int, check.get(f"{side}_unmatched") or 0)
            if unmatched > 0:
                rows = check.get(f"{side}_rows")
                of = "" if rows is None else f" of {rows}"
                parts.append(f"{unmatched}{of} rows in {key} have no match")
        if parts:
            warnings.append(f"{left} = {right}: {'; '.join(parts)}.")
        for side, key in (("left", left), ("right", right)):
            near = cast(int, check.get(f"{side}_near_matches") or 0)
            if near > 0:
                warnings.append(
                    f"{left} = {right}: {near} unmatched rows in {key} would match "
                    "if case and surrounding spaces were ignored, so the keys may "
                    "be formatted differently. If they name the same thing, "
                    "normalize them inside the inputs and retry, and tell the user."
                )
        types = (check.get("left_type"), check.get("right_type"))
        if all(isinstance(item, str) and item != "NULL" for item in types) and (
            _type_family(cast(str, types[0])) != _type_family(cast(str, types[1]))
        ):
            warnings.append(
                f"{left} = {right}: key types differ ({types[0]} and {types[1]}), "
                "so matches may be missed or coerced."
            )
    if status in ("partial", "unavailable"):
        warnings.append(
            "The join match check could not finish; the result itself is unaffected."
        )
    return tuple(warnings)


def _first_error(failure: BaseException) -> BaseException:
    while isinstance(failure, BaseExceptionGroup):
        failure = failure.exceptions[0]
    return failure


def _name_input(error: Exception, alias: str) -> None:
    """Add the failing alias to the details the model is shown with the error."""

    details = getattr(error, "details", None)
    if details is None or isinstance(details, Mapping):
        try:
            error.details = FrozenJsonObject.from_mapping(  # type: ignore[attr-defined]
                {**(details or {}), "input_alias": alias}
            )
        except AttributeError:
            pass


def _truncated(item: Mapping[str, object], reasons: tuple[str, ...]) -> Exception:
    alias = item["alias"]
    return CapabilityInputError(
        "combined_input_truncated",
        f"Input {alias} returned more rows or bytes than can be joined "
        "completely, so no result was returned.",
        {
            "alias": alias,
            "source_id": item["source_id"],
            "truncation_reasons": reasons or ("budget_spent",),
        },
        recovery=ToolRecovery(
            RecoveryKind.CORRECT_ARGUMENTS,
            f"Aggregate or filter inside input {alias} so it returns fewer "
            "rows, then retry the combined query.",
        ),
    )


class DataQueryCombinedExecutor:
    """Join complete bounded per-source reads in the private DuckDB worker.

    Each input runs through its source's registry backend, as SQL or as a
    tabular read. Sources are read concurrently and the inputs of one source one
    at a time, inside one shared deadline and one shared row and byte budget.
    The caller already holds one admission permit per source for the whole
    call, so this executor takes none.
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
        request_refresh: RefreshRequest = lambda source_id, names: None,
    ) -> None:
        self._agent_id = agent_id
        self._request_refresh = request_refresh
        self._backends = dict(backends)
        self._tabular_backends = dict(tabular_backends or {})
        self._engine = engine
        self._clock = clock

    async def execute(self, request: ToolExecution) -> ToolOutput:
        inputs = cast(tuple[Mapping[str, object], ...], request.arguments["inputs"])
        started = self._clock()
        deadline = started + DATA_QUERY_TOTAL_TIMEOUT_SECONDS
        budget = _ReadBudget(
            COMBINED_QUERY_TOTAL_MAX_ROWS, COMBINED_QUERY_TOTAL_MAX_BYTES
        )
        attempts: _Attempts = {}
        # A child task keeps the TaskGroup's own cancel of its parent (which
        # Python 3.11 never takes back) off this call's task, so
        # ``cancelling()`` below means only an outside cancellation.
        reading = asyncio.ensure_future(
            self._read_groups(inputs, budget, deadline, attempts)
        )
        try:
            await reading
        except BaseExceptionGroup as failure:
            task = asyncio.current_task()
            if task is not None and task.cancelling() > 0:
                raise asyncio.CancelledError from None
            raise _first_error(failure) from None
        # An input that did not fit its share is read again, alone, with what the
        # finished reads left; if it still does not fit, the call is refused.
        for index in sorted(attempts):
            read, grant, reasons = attempts[index]
            if read is None:
                read, _, again = await self._read_input(
                    inputs[index], budget, 1, deadline, beyond=grant
                )
                if read is None:
                    raise _truncated(inputs[index], again or reasons)
                attempts[index] = read, grant, ()
        reads = [
            (cast(str, item["alias"]), cast(_InputRead, attempts[index][0]))
            for index, item in enumerate(inputs)
        ]
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
        touched = {item for _, read in reads for item in read.resource_ids}
        names = dict(name for _, read in reads for name in read.resource_names)
        data = projection.to_payload().to_dict()
        data.update(
            {
                "canonical_sql": payload["canonical_sql"],
                "columns": payload["columns"],
                "inputs": tuple(
                    {"alias": alias, **read.lineage} for alias, read in reads
                ),
                "read_consistency": _READ_CONSISTENCY,
                "resource_ids": tuple(sorted(touched)),
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
        if "join_checks" in payload:
            data["join_checks"] = payload["join_checks"]
        warnings = _join_warnings(
            payload.get("join_checks"), payload.get("join_checks_status")
        )
        if warnings:
            data["warnings"] = warnings
        return ToolOutput(kind=DATA_QUERY_COMBINED_EVIDENCE_KIND, data=data)

    async def _read_groups(
        self,
        inputs: tuple[Mapping[str, object], ...],
        budget: _ReadBudget,
        deadline: float,
        attempts: _Attempts,
    ) -> None:
        """Read each source's inputs in order, the sources at the same time."""

        groups: dict[str, list[int]] = {}
        for index, item in enumerate(inputs):
            groups.setdefault(cast(str, item["source_id"]), []).append(index)
        ungranted = len(inputs)

        async def read_group(indexes: list[int]) -> None:
            nonlocal ungranted
            for index in indexes:
                waiting, ungranted = ungranted, ungranted - 1
                attempts[index] = await self._read_input(
                    inputs[index], budget, waiting, deadline
                )

        async with asyncio.TaskGroup() as group:
            for indexes in groups.values():
                group.create_task(read_group(indexes))

    async def _read_input(
        self,
        item: Mapping[str, object],
        budget: _ReadBudget,
        waiting: int,
        deadline: float,
        beyond: tuple[int, int] | None = None,
    ) -> tuple[_InputRead | None, tuple[int, int], tuple[str, ...]]:
        """Read one input within its share; None when it did not fit.

        ``beyond`` is the grant a first attempt failed with: a repeat that gets
        no more than that cannot do better, so it is not run. The rows of an
        input that did not fit are dropped, never joined.
        """

        remaining = deadline - self._clock()
        if remaining < _MIN_READ_SECONDS:
            raise DataQueryError(
                "query_time_limit",
                "The inputs exceeded the total read deadline of "
                f"{DATA_QUERY_TOTAL_TIMEOUT_SECONDS:g} seconds.",
            )
        grant = budget.grant(waiting)
        rows, size = grant
        if (
            rows < 1
            or size < 2
            or (beyond is not None and rows <= beyond[0] and size <= beyond[1])
        ):
            budget.settle(grant, (0, 0))
            return None, grant, ()
        timeout = min(DATA_QUERY_STATEMENT_TIMEOUT_SECONDS, remaining)
        used = (0, 0)
        try:
            began = self._clock()
            read = await (
                self._read_sql(item, timeout, rows, size)
                if "sql" in item
                else self._read_tabular(item, timeout, rows, size)
            )
            read.lineage["read_ms"] = _elapsed_ms(began, self._clock())
            if read.projection.truncated:
                return None, grant, read.projection.truncation_reasons
            used = (read.projection.returned_rows, read.projection.utf8_bytes)
            return read, grant, ()
        except Exception as error:
            _name_input(error, cast(str, item["alias"]))
            raise
        finally:
            budget.settle(grant, used)

    async def _read_sql(
        self, item: Mapping[str, object], timeout: float, rows: int, size: int
    ) -> _InputRead:
        source_id = cast(str, item["source_id"])
        backend = self._backends.get(cast(str, item["_adapter_id"]))
        if backend is None:
            raise ValueError("relational query adapter binding is unsupported")
        read_at = datetime.now(UTC).isoformat()
        result = await _execute_read(
            backend,
            self._request_refresh,
            agent_id=self._agent_id,
            source_id=source_id,
            read=checked_reads(item)[0],
            max_rows=rows,
            max_bytes=size,
            timeout_seconds=timeout,
        )
        if not isinstance(result, SqlReadResult) or result.source_id != source_id:
            raise ValueError("SQL backend returned a different source")
        return _InputRead(
            columns=result.columns,
            projection=result.projection,
            resource_ids=result.resource_ids,
            resource_names=result.resource_names,
            lineage={
                "source_id": result.source_id,
                "read_at": read_at,
                "canonical_sql": result.canonical_sql,
                "sql_fingerprint": result.sql_fingerprint,
                "resource_ids": result.resource_ids,
                "rows": result.projection.returned_rows,
            },
        )

    async def _read_tabular(
        self, item: Mapping[str, object], timeout: float, rows: int, size: int
    ) -> _InputRead:
        backend = self._tabular_backends.get(cast(str, item["_adapter_id"]))
        if backend is None:
            raise ValueError("tabular read adapter binding is unsupported")
        resource_id = cast(str, item["resource_id"])
        columns = cast(tuple[str, ...], item["columns"])
        filters = tuple(
            TabularFilter.from_mapping(cast(Mapping[str, object], value))
            for value in cast(tuple[object, ...], item.get("filters", ()))
        )
        read_at = datetime.now(UTC).isoformat()
        projection = await backend.read_tabular(
            ReadTabularRequest(
                agent_id=self._agent_id,
                source_id=cast(str, item["source_id"]),
                resource_id=resource_id,
                columns=columns,
                filters=filters,
                limit=rows,
                max_bytes=size,
                timeout_seconds=timeout,
            )
        )
        if not isinstance(projection, BoundedResultProjection):
            raise ValueError("tabular backend returned a different result type")
        if any(set(row) != set(columns) for row in projection.rows):
            # The source could not serve a requested column.
            raise CapabilityInputError(
                "tabular_columns_unavailable",
                f"Input {item['alias']} could not return exactly the requested "
                "columns of that table.",
                {"alias": item["alias"], "source_id": item["source_id"]},
                recovery=ToolRecovery(
                    RecoveryKind.CORRECT_ARGUMENTS,
                    "Request only columns this table has; catalog_describe "
                    "lists the ones it knows.",
                ),
            )
        return _InputRead(
            columns=columns,
            projection=projection,
            resource_ids=(resource_id,),
            resource_names=(
                ((resource_id, cast(str, item["_resource_name"])),)
                if "_resource_name" in item
                else ()
            ),
            lineage={
                "source_id": item["source_id"],
                "read_at": read_at,
                "columns": columns,
                "filters": tuple(value.to_payload() for value in filters),
                "resource_ids": (resource_id,),
                "rows": projection.returned_rows,
            },
        )


def _elapsed_ms(started: float, finished: float) -> int:
    return max(0, int(round((finished - started) * 1_000)))


def _multi_statement_data(
    adapter_id: str,
    source_id: str,
    results: list[tuple[SqlReadResult | None, int, str]],
    *,
    timing: Mapping[str, int],
    row_limit: int,
    byte_limit: int,
) -> dict[str, object]:
    statements: list[dict[str, object]] = []
    touched: set[str] = set()
    names: dict[str, str] = {}
    reasons: list[str] = []
    for index, (result, elapsed_ms, read_at) in enumerate(results):
        if result is None:
            statements.append({"index": index, "status": "skipped_budget_spent"})
            reasons.append("budget_spent")
            continue
        touched.update(result.resource_ids)
        names.update(result.resource_names)
        payload = result.tool_data().to_dict()
        for name in ("source_id", "row_limit", "byte_limit"):
            payload.pop(name)
        payload["elapsed_ms"] = elapsed_ms
        payload["index"] = index
        payload["read_at"] = read_at
        payload["status"] = "completed"
        statements.append(payload)
        reasons.extend(result.projection.truncation_reasons)
    executed = [result for result, _, _ in results if result is not None]
    fingerprint = canonical_json([result.sql_fingerprint for result in executed])
    return {
        "adapter_id": adapter_id,
        "byte_limit": byte_limit,
        "resource_ids": tuple(sorted(touched)),
        "returned_rows": sum(result.projection.returned_rows for result in executed),
        "row_limit": row_limit,
        "source_id": source_id,
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


@dataclass(frozen=True, slots=True)
class VerifiedEffect:
    """One committed write whose source compared what it touched with the card."""

    receipt_id: str
    source_id: str
    table: str
    operation: str
    canonical_sql: str
    preview_fingerprint: str
    intent_sha256: str
    target_set_sha256: str
    affected_rows: int
    inserted_rows: int
    updated_rows: int
    deleted_rows: int
    skipped_rows: int
    committed_at: str
    effect_observation: EffectObservation

    def tool_data(self) -> FrozenJsonObject:
        return FrozenJsonObject.from_mapping(
            {
                "receipt_id": self.receipt_id,
                "outcome": "committed",
                "source_id": self.source_id,
                "table": self.table,
                "operation": self.operation,
                "sql": self.canonical_sql,
                "preview_fingerprint": self.preview_fingerprint,
                "intent_sha256": self.intent_sha256,
                "target_set_sha256": self.target_set_sha256,
                "affected_rows": self.affected_rows,
                "inserted_rows": self.inserted_rows,
                "updated_rows": self.updated_rows,
                "deleted_rows": self.deleted_rows,
                "skipped_rows": self.skipped_rows,
                "committed_at": self.committed_at,
                "trust_classification": "untrusted_data",
            }
        )


class RelationalWriteBackend(WriteDescriber, Protocol):
    """The seam: everything about a source's language stays behind these."""

    async def preview(
        self, *, agent_id: str, source_id: str, facts: WriteFacts
    ) -> WritePreview: ...

    async def apply(
        self,
        *,
        agent_id: str,
        source_id: str,
        execution: ToolExecution,
        facts: WriteFacts,
        preview_fingerprint: str,
        expected_affected_rows: int,
    ) -> VerifiedEffect: ...


class _RelationalWriteRouting:
    def __init__(
        self,
        agent_id: str,
        backends: Mapping[str, RelationalWriteBackend],
        source_adapter_id: SourceAdapterLookup,
    ) -> None:
        self._agent_id = agent_id
        self._backends = dict(backends)
        self._source_adapter_id = source_adapter_id

    async def _preview(self, request: ToolExecution) -> WritePreview:
        source_id = cast(str, request.arguments["source_id"])
        backend = await _source_backend(
            self._backends, self._source_adapter_id, self._agent_id, source_id
        )
        return await backend.preview(
            agent_id=self._agent_id,
            source_id=source_id,
            facts=write_facts(request.arguments),
        )


class RelationalWritePreviewExecutor(_RelationalWriteRouting):
    executor_id = RELATIONAL_WRITE_PREVIEW_EXECUTOR_ID

    async def execute(self, request: ToolExecution) -> ToolOutput:
        preview = (await self._preview(request)).to_mapping()
        # The fingerprint binds approval to dispatch inside Daita; the model never
        # sends it.
        del preview["preview_fingerprint"]
        return ToolOutput(
            kind=RELATIONAL_WRITE_PREVIEW_EVIDENCE_KIND,
            data=FrozenJsonObject.from_mapping(
                {**preview, "trust_classification": "untrusted_data"}
            ),
        )


class RelationalWriteExecutor(_RelationalWriteRouting):
    executor_id = RELATIONAL_WRITE_EXECUTOR_ID

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        preview = await self._preview(request)
        if preview.affected_rows == 0:
            raise CapabilityInputError(
                "write_target_not_found",
                "The statement does not currently change any rows.",
            )
        return FrozenJsonObject.from_mapping(
            {
                "intent_sha256": preview.intent_sha256,
                "preview_fingerprint": preview.preview_fingerprint,
                "target_set_sha256": preview.target_set_sha256,
                "structure_sha256": preview.structure_sha256,
                "review": {
                    key: getattr(preview, key)
                    for key in (
                        "sql",
                        "operation",
                        "columns_changed",
                        "affected_rows",
                        "inserted_rows",
                        "updated_rows",
                        "deleted_rows",
                        "skipped_rows",
                        "samples",
                        "triggers",
                        "cascades",
                        "warnings",
                        "verification",
                    )
                },
            }
        )

    async def execute(self, request: ToolExecution) -> ToolOutput:
        approved = request.approved_preflight
        if approved is None:
            raise ValueError("write dispatch requires its approved preflight")
        source_id = cast(str, request.arguments["source_id"])
        backend = await _source_backend(
            self._backends, self._source_adapter_id, self._agent_id, source_id
        )
        result = await backend.apply(
            agent_id=self._agent_id,
            source_id=source_id,
            execution=request,
            facts=write_facts(request.arguments),
            preview_fingerprint=cast(str, approved["preview_fingerprint"]),
            expected_affected_rows=cast(
                int, cast(Mapping[str, object], approved["review"])["affected_rows"]
            ),
        )
        return ToolOutput(
            kind=RELATIONAL_WRITE_EVIDENCE_KIND,
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
    request_refresh: RefreshRequest = lambda source_id, names: None,
) -> DataQueryDeclarations:
    """Declare data_query over SQL backends.

    Path-pattern backends add the ``path_pattern`` route; a combined engine
    adds the multi-source ``inputs`` route.
    """

    executors: tuple[Executor, ...] = (
        DataQueryExecutor(agent_id, backends, request_refresh=request_refresh),
    )
    if path_backends:
        executors += (PathQueryExecutor(agent_id, path_backends),)
    if combined_engine is not None:
        executors += (
            DataQueryCombinedExecutor(
                agent_id,
                backends,
                combined_engine,
                tabular_backends=tabular_backends,
                request_refresh=request_refresh,
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
                                    "path_pattern": {
                                        "type": "string",
                                        "minLength": 1,
                                        "maxLength": 2_048,
                                    },
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
                        "source's dialect, or exact columns of one "
                        "table with optional AND-combined filters "
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


# Results stored before reads stopped binding catalog revisions still validate.
_STORED_READ_RESULT_KEYS: dict[str, object] = {
    "resource_revisions": {"type": "array"},
    "source_revision": {"type": "string"},
}


def _combined_query_output_schema() -> dict[str, object]:
    # Results stored before these were reported still validate.
    later = {"join_checks": "array", "read_consistency": "string", "warnings": "array"}
    types = {
        "byte_limit": "integer",
        "canonical_sql": "string",
        "columns": "array",
        "inputs": "array",
        "resource_ids": "array",
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
        "properties": {
            **{name: {"type": kind} for name, kind in sorted((types | later).items())},
            **_STORED_READ_RESULT_KEYS,
        },
        "required": sorted(types),
        "additionalProperties": False,
    }


def _query_output_schema() -> dict[str, object]:
    types = {
        "byte_limit": "integer",
        "adapter_id": "string",
        "canonical_sql": "string",
        "columns": "array",
        "read_at": "string",
        "resource_ids": "array",
        "returned_rows": "integer",
        "row_limit": "integer",
        "rows": "array",
        "source_id": "string",
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
    per_statement = {"canonical_sql", "columns", "read_at", "rows", "statements"}
    return {
        "type": "object",
        "properties": {
            **{name: {"type": kind} for name, kind in sorted(types.items())},
            **_STORED_READ_RESULT_KEYS,
        },
        "required": sorted(set(types) - per_statement),
        "additionalProperties": False,
    }


def relational_write_capability_declarations() -> CapabilityDeclarations:
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "source_id": {"type": "string", "pattern": SOURCE_ID_PATTERN},
            "sql": {"type": "string", "minLength": 1, "maxLength": MAX_SQL_CHARACTERS},
            "parameters": {
                "type": "array",
                "maxItems": 32_767,
                "description": "Values for $1, $2, ... by index.",
            },
        },
        "required": ["source_id", "sql"],
        "additionalProperties": False,
    }
    count: dict[str, object] = {"type": "integer", "minimum": 0}
    preview_properties: dict[str, object] = {
        "source_id": {"type": "string"},
        "table": {"type": "string"},
        "operation": {"type": "string", "enum": ["insert", "update", "delete"]},
        "sql": {"type": "string"},
        "columns_changed": {"type": "array", "items": {"type": "string"}},
        "affected_rows": count,
        "inserted_rows": count,
        "updated_rows": count,
        "deleted_rows": count,
        "skipped_rows": count,
        "samples": {"type": "array", "maxItems": 5, "items": {"type": "object"}},
        "triggers": {"type": "array", "items": {"type": "string"}},
        "cascades": {"type": "array", "items": {"type": "string"}},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "verification": {"type": "string", "enum": ["rows", "count"]},
        "intent_sha256": {"type": "string"},
        "target_set_sha256": {"type": "string"},
        "structure_sha256": {"type": "string"},
        "trust_classification": {"type": "string"},
    }
    write_properties: dict[str, object] = {
        "receipt_id": {"type": "string"},
        "outcome": {"type": "string", "enum": ["committed"]},
        "source_id": {"type": "string"},
        "table": {"type": "string"},
        "operation": {"type": "string", "enum": ["insert", "update", "delete"]},
        "sql": {"type": "string"},
        "preview_fingerprint": {"type": "string"},
        "intent_sha256": {"type": "string"},
        "target_set_sha256": {"type": "string"},
        "affected_rows": {"type": "integer", "minimum": 1},
        "inserted_rows": count,
        "updated_rows": count,
        "deleted_rows": count,
        "skipped_rows": count,
        "committed_at": {"type": "string"},
        "trust_classification": {"type": "string"},
    }
    preview = Capability(
        id=RELATIONAL_WRITE_PREVIEW_CAPABILITY_ID,
        description=(
            "Preview one SQL write without writing: the rows it would touch, counted "
            "once each, with a bounded sample."
        ),
        input_schema=input_schema,
        output_kind=RELATIONAL_WRITE_PREVIEW_EVIDENCE_KIND,
        output_schema={
            "type": "object",
            "properties": preview_properties,
            "required": list(preview_properties),
            "additionalProperties": False,
        },
        executor_id=RELATIONAL_WRITE_PREVIEW_EXECUTOR_ID,
        access_mode=AccessMode.READ,
    )
    write = Capability(
        id=RELATIONAL_WRITE_CAPABILITY_ID,
        description=(
            "Preview one SQL write for runtime approval, then apply it exactly once "
            "only if approved and the recheck finds the same rows and table."
        ),
        input_schema=input_schema,
        output_kind=RELATIONAL_WRITE_EVIDENCE_KIND,
        output_schema={
            "type": "object",
            "properties": write_properties,
            "required": list(write_properties),
            "additionalProperties": False,
        },
        executor_id=RELATIONAL_WRITE_EXECUTOR_ID,
        access_mode=AccessMode.WRITE,
        operational_effect=OperationalEffect.MUTATE_DATA,
        effect_receipt_policy=RELATIONAL_WRITE_RECEIPT_POLICY,
        execution_admission_policy=ExecutionAdmissionPolicy(
            shape="preview_bound_postgresql_write",
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
                name=RELATIONAL_WRITE_TOOL_NAME,
                capability_id=write.id,
                description=RELATIONAL_WRITE_DESCRIPTION,
                mode_capability_ids=(("preview", preview.id), ("apply", write.id)),
                default_mode="apply",
                presentation=ToolPresentation(
                    toolbox_id=ToolboxId.SOURCES,
                    load_mode=ToolLoadMode.ON_DEMAND,
                    text_trust=ToolTextTrust.CODE,
                    summary="Insert, update or delete rows with one approved SQL statement.",
                    when_to_use=(
                        "Call once to write; the approval card shows the rows. Use "
                        "mode=preview only to look first."
                    ),
                    keywords=(
                        "data",
                        "postgresql",
                        "rows",
                        "row",
                        "table",
                        "database",
                        "write",
                        "insert",
                        "update",
                        "delete",
                        "upsert",
                        "batch",
                        "save",
                        "approval",
                    ),
                ),
            ),
        ),
    )


def relational_write_declarations(
    agent_id: str,
    backends: Mapping[str, RelationalWriteBackend],
    source_adapter_id: SourceAdapterLookup,
) -> DataQueryDeclarations:
    declarations = relational_write_capability_declarations()
    return DataQueryDeclarations(
        declarations.capabilities,
        (
            RelationalWritePreviewExecutor(agent_id, backends, source_adapter_id),
            RelationalWriteExecutor(agent_id, backends, source_adapter_id),
        ),
        declarations.tool_views,
    )


__all__ = [
    "DATA_QUERY_EXECUTOR_ID",
    "DATA_QUERY_TOOL_NAME",
    "DataQueryDeclarations",
    "DataQueryExecutor",
    "PostgreSQLReadResult",
    "RELATIONAL_WRITE_EXECUTOR_ID",
    "RELATIONAL_WRITE_PREVIEW_EXECUTOR_ID",
    "RELATIONAL_WRITE_TOOL_NAME",
    "RelationalWriteBackend",
    "RelationalWriteExecutor",
    "RelationalWritePreviewExecutor",
    "VerifiedEffect",
    "SQLiteReadResult",
    "ReadTabularRequest",
    "TABULAR_FILTER_OPERATORS",
    "TabularFilter",
    "TabularReadBackend",
    "data_query_declarations",
    "data_query_capability_declarations",
    "relational_write_capability_declarations",
    "relational_write_declarations",
]
