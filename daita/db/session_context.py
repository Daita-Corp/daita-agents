"""Typed, read-only session context for DB conversational workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Iterable, Mapping

from daita.runtime import Evidence, Operation, OperationStatus

from .models import DbRequest
from .sql_analysis import SqlAnalysisError, analyze_sql

_MAX_MESSAGES = 12
_MAX_OPERATIONS = 8
_MAX_REFERENTS = 24
_MAX_QUERY_SCOPES = 4
_MAX_QUERY_FILTERS = 12
_MAX_FILTER_VALUES = 8
_MAX_FILTER_VALUE_CHARS = 80
_MAX_IDENTIFIER_CHARS = 160
_IDENTIFIER_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]*)`")
_SCHEMA_LINE_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_.]*)\s*:\s*([^.;\n]+)")
_SENSITIVE_FILTER_COLUMN_RE = re.compile(
    r"(?i)(password|passwd|secret|token|api_?key|email|phone|mobile|ssn|"
    r"social_security|credit_card|card_number|cvv|pin|dob|date_of_birth|"
    r"birth_date|address|street|zip|postal|passport|national_id)"
)


@dataclass(frozen=True)
class DbSessionOperationRef:
    """Compact reference to a recent DB runtime operation."""

    operation_id: str
    operation_type: str
    status: str
    prompt: str | None = None
    monitor_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "status": self.status,
            "monitor_id": self.monitor_id,
        }
        if self.prompt:
            result["prompt_fingerprint"] = _fingerprint_text(self.prompt)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DbSessionOperationRef":
        prompt = value.get("prompt")
        if prompt is None:
            prompt = value.get("prompt_snippet")
        return cls(
            operation_id=str(value.get("operation_id") or ""),
            operation_type=str(value.get("operation_type") or ""),
            status=str(value.get("status") or ""),
            prompt=str(prompt) if prompt is not None else None,
            monitor_id=(
                str(value["monitor_id"])
                if value.get("monitor_id") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class DbSessionQueryScope:
    """Compact predicate scope from a prior same-session query."""

    operation_id: str
    tables: tuple[str, ...] = ()
    filters: tuple[dict[str, Any], ...] = ()
    selected_columns: tuple[str, ...] = ()
    result_row_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "operation_id": _clip(self.operation_id, _MAX_IDENTIFIER_CHARS),
            "tables": [
                _clip(item, _MAX_IDENTIFIER_CHARS)
                for item in self.tables[:_MAX_REFERENTS]
            ],
            "filters": [
                item
                for item in (
                    _compact_query_filter(filter_item)
                    for filter_item in self.filters[:_MAX_QUERY_FILTERS]
                )
                if item is not None
            ],
            "selected_columns": [
                _clip(item, _MAX_IDENTIFIER_CHARS)
                for item in self.selected_columns[:_MAX_REFERENTS]
            ],
        }
        if self.result_row_count is not None:
            result["result_row_count"] = max(0, int(self.result_row_count))
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DbSessionQueryScope":
        filters = tuple(
            item
            for item in (
                _compact_query_filter(filter_item)
                for filter_item in value.get("filters", ()) or ()
                if isinstance(filter_item, Mapping)
            )
            if item is not None
        )
        row_count = value.get("result_row_count")
        return cls(
            operation_id=str(value.get("operation_id") or ""),
            tables=_strings(value.get("tables")),
            filters=filters,
            selected_columns=_strings(value.get("selected_columns")),
            result_row_count=int(row_count) if isinstance(row_count, int) else None,
        )


@dataclass(frozen=True)
class DbSessionReferents:
    """Structured referents carried across same-session DB turns."""

    tables: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()
    schemas: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    monitors: tuple[str, ...] = ()
    approvals: tuple[str, ...] = ()
    operations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "tables": list(self.tables),
            "columns": list(self.columns),
            "schemas": list(self.schemas),
            "metrics": list(self.metrics),
            "monitors": list(self.monitors),
            "approvals": list(self.approvals),
            "operations": list(self.operations),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "DbSessionReferents":
        value = dict(value or {})
        return cls(
            tables=_strings(value.get("tables")),
            columns=_strings(value.get("columns")),
            schemas=_strings(value.get("schemas")),
            metrics=_strings(value.get("metrics")),
            monitors=_strings(value.get("monitors")),
            approvals=_strings(value.get("approvals")),
            operations=_strings(value.get("operations")),
        )


@dataclass(frozen=True)
class DbSessionContext:
    """Bounded context assembled for one DB request from existing owners."""

    session_id: str | None
    user_id: str | None
    current_prompt: str
    conversation_messages: tuple[dict[str, str], ...] = ()
    recent_operations: tuple[DbSessionOperationRef, ...] = ()
    query_scopes: tuple[DbSessionQueryScope, ...] = ()
    referents: DbSessionReferents = field(default_factory=DbSessionReferents)
    durable_ids: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.to_request_dict()

    def to_request_dict(self) -> dict[str, Any]:
        """Return a JSON-safe projection suitable for DbRequest and planning."""

        return {
            "session_id": _optional_clip(self.session_id),
            "user_id": _optional_clip(self.user_id),
            "recent_operations": [item.to_dict() for item in self.recent_operations],
            "query_scopes": [item.to_dict() for item in self.query_scopes],
            "referents": _compact_referents(self.referents),
            "durable_ids": _compact_string_mapping(self.durable_ids),
            "diagnostics": _compact_diagnostics(
                {
                    **self.diagnostics,
                    "conversation_message_count": len(self.conversation_messages),
                    "recent_operation_count": len(self.recent_operations),
                    "query_scope_count": len(self.query_scopes),
                }
            ),
        }

    def to_diagnostic_dict(self) -> dict[str, Any]:
        """Return bounded diagnostics without raw transcript or full prompts."""

        return {
            "session_id": _optional_clip(self.session_id),
            "user_id": _optional_clip(self.user_id),
            "conversation_message_count": len(self.conversation_messages),
            "recent_operation_count": len(self.recent_operations),
            "query_scope_count": len(self.query_scopes),
            "referents": _compact_referents(self.referents),
            "durable_ids": _compact_string_mapping(self.durable_ids),
            "diagnostics": _compact_diagnostics(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "DbSessionContext | None":
        if not isinstance(value, Mapping):
            return None
        return cls(
            session_id=(
                str(value["session_id"])
                if value.get("session_id") is not None
                else None
            ),
            user_id=str(value["user_id"]) if value.get("user_id") is not None else None,
            current_prompt=str(value.get("current_prompt") or ""),
            conversation_messages=tuple(
                message
                for item in value.get("conversation_messages", ()) or ()
                if (message := _message_dict(item)) is not None
            ),
            recent_operations=tuple(
                DbSessionOperationRef.from_dict(item)
                for item in value.get("recent_operations", ()) or ()
                if isinstance(item, Mapping)
            ),
            query_scopes=tuple(
                DbSessionQueryScope.from_dict(item)
                for item in value.get("query_scopes", ()) or ()
                if isinstance(item, Mapping)
            ),
            referents=DbSessionReferents.from_dict(value.get("referents")),
            durable_ids={
                _clip(str(key), _MAX_IDENTIFIER_CHARS): _clip(
                    str(item), _MAX_IDENTIFIER_CHARS
                )
                for key, item in dict(value.get("durable_ids") or {}).items()
                if item is not None
            },
            diagnostics=dict(value.get("diagnostics") or {}),
        )


class DbSessionContextBuilder:
    """Build a read-only DB session context from transcript and runtime facts."""

    def __init__(self, runtime: Any, *, max_operations: int = _MAX_OPERATIONS) -> None:
        self.runtime = runtime
        self.max_operations = max_operations

    async def build(
        self,
        request: DbRequest,
        *,
        conversation_messages: Iterable[Mapping[str, Any]] = (),
    ) -> DbSessionContext:
        messages = _compact_messages(conversation_messages)
        durable_ids = _durable_ids_from_metadata(request.metadata)
        diagnostics: dict[str, Any] = {
            "sources": [],
            "conversation_message_count": len(messages),
            "bounded": {
                "max_messages": _MAX_MESSAGES,
                "max_operations": self.max_operations,
                "max_referents": _MAX_REFERENTS,
            },
        }
        if messages:
            diagnostics["sources"].append("conversation_history")
        if durable_ids:
            diagnostics["sources"].append("request.metadata")

        operations = await self._recent_operations(request, durable_ids)
        if operations:
            diagnostics["sources"].append("runtime.operations")

        evidence_by_operation = {
            operation.id: tuple(await self.runtime.store.list_evidence(operation.id))
            for operation in operations
        }
        approvals = await self._approval_ids(operations)
        query_scopes: list[DbSessionQueryScope] = []

        tables = _OrderedStrings()
        columns = _OrderedStrings()
        monitors = _OrderedStrings()
        operation_ids = _OrderedStrings()
        approval_ids = _OrderedStrings()
        schemas = _OrderedStrings()
        metrics = _OrderedStrings()

        for item in request.source_scope:
            tables.add(item, source="request.source_scope")
        for key in ("last_monitor_id", "monitor_id"):
            if durable_ids.get(key):
                monitors.add(durable_ids[key], source="request.metadata")
        for key in ("last_runtime_operation_id", "operation_id"):
            if durable_ids.get(key):
                operation_ids.add(durable_ids[key], source="request.metadata")
        for key in ("last_approval_id", "approval_id"):
            if durable_ids.get(key):
                approval_ids.add(durable_ids[key], source="request.metadata")

        for operation in operations:
            operation_ids.add(operation.id, source="runtime.operations")
            monitor_id = operation.metadata.get("monitor_id")
            if monitor_id:
                monitors.add(str(monitor_id), source="runtime.operations")
            extracted = _extract_evidence_referents(evidence_by_operation[operation.id])
            tables.extend(extracted["tables"], source="runtime.evidence")
            columns.extend(extracted["columns"], source="runtime.evidence")
            monitors.extend(extracted["monitors"], source="runtime.evidence")
            schemas.extend(extracted["schemas"], source="runtime.evidence")
            metrics.extend(extracted["metrics"], source="runtime.evidence")
            query_scope = _query_scope_from_evidence(
                operation.id,
                evidence_by_operation[operation.id],
            )
            if query_scope is not None:
                query_scopes.append(query_scope)

        approval_ids.extend(approvals, source="runtime.approvals")

        transcript_refs = _extract_transcript_referents(messages)
        if not tables.values:
            tables.extend(transcript_refs["tables"], source="conversation_history")
        if not columns.values:
            columns.extend(transcript_refs["columns"], source="conversation_history")

        diagnostics["referent_sources"] = {
            "tables": tables.sources,
            "columns": columns.sources,
            "schemas": schemas.sources,
            "metrics": metrics.sources,
            "monitors": monitors.sources,
            "approvals": approval_ids.sources,
            "operations": operation_ids.sources,
        }
        diagnostics["recent_operation_count"] = len(operations)
        diagnostics["evidence_operation_count"] = len(evidence_by_operation)
        diagnostics["query_scope_count"] = len(query_scopes)

        return DbSessionContext(
            session_id=request.session_id,
            user_id=request.user_id,
            current_prompt=request.prompt,
            conversation_messages=messages,
            recent_operations=tuple(_operation_ref(item) for item in operations),
            query_scopes=tuple(query_scopes[:_MAX_QUERY_SCOPES]),
            referents=DbSessionReferents(
                tables=tables.values,
                columns=columns.values,
                schemas=schemas.values,
                metrics=metrics.values,
                monitors=monitors.values,
                approvals=approval_ids.values,
                operations=operation_ids.values,
            ),
            durable_ids=durable_ids,
            diagnostics=diagnostics,
        )

    async def _recent_operations(
        self,
        request: DbRequest,
        durable_ids: Mapping[str, str],
    ) -> tuple[Operation, ...]:
        operations = await self.runtime.store.list_operations()
        selected: list[Operation] = []
        explicit_ids = {
            durable_ids[key]
            for key in ("last_runtime_operation_id", "operation_id")
            if durable_ids.get(key)
        }
        approval_operation_ids = {
            approval.operation_id
            for approval in await self.runtime.store.list_approval_requests()
            if approval.approval_id
            in {
                durable_ids.get("last_approval_id"),
                durable_ids.get("approval_id"),
            }
        }
        explicit_ids.update(item for item in approval_operation_ids if item)
        for operation in reversed(operations):
            if operation.id in explicit_ids:
                selected.append(operation)
                continue
            if (
                request.session_id
                and _operation_session_id(operation) == request.session_id
            ):
                selected.append(operation)
            if len(selected) >= self.max_operations:
                break
        return tuple(selected)

    async def _approval_ids(self, operations: tuple[Operation, ...]) -> tuple[str, ...]:
        operation_ids = {operation.id for operation in operations}
        if not operation_ids:
            return ()
        approvals = await self.runtime.store.list_approval_requests()
        return tuple(
            approval.approval_id
            for approval in reversed(approvals)
            if approval.operation_id in operation_ids
        )


def db_session_context_from_request(request: DbRequest) -> DbSessionContext | None:
    context = getattr(request, "session_context", None)
    if isinstance(context, DbSessionContext):
        return context
    return DbSessionContext.from_dict(context)


def _operation_session_id(operation: Operation) -> str | None:
    value = operation.metadata.get("session_id")
    if value is None:
        value = operation.request.get("session_id")
    return str(value) if value is not None else None


def _operation_ref(operation: Operation) -> DbSessionOperationRef:
    status = operation.status
    if isinstance(status, OperationStatus):
        status_value = status.value
    else:
        status_value = str(status)
    prompt = operation.request.get("prompt")
    return DbSessionOperationRef(
        operation_id=operation.id,
        operation_type=operation.operation_type,
        status=status_value,
        prompt=str(prompt) if prompt is not None else None,
        monitor_id=(
            str(operation.metadata["monitor_id"])
            if operation.metadata.get("monitor_id") is not None
            else None
        ),
    )


def _durable_ids_from_metadata(metadata: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in (
        "last_monitor_id",
        "monitor_id",
        "last_runtime_operation_id",
        "operation_id",
        "last_approval_id",
        "approval_id",
    ):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            result[key] = str(value)
    return result


def _compact_messages(
    messages: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, str], ...]:
    compact = []
    for item in messages:
        message = _message_dict(item)
        if message is not None:
            compact.append(message)
    return tuple(compact[-_MAX_MESSAGES:])


def _message_dict(value: Any) -> dict[str, str] | None:
    if not isinstance(value, Mapping):
        return None
    role = str(value.get("role") or "")
    content = _clip(str(value.get("content") or ""), 1000)
    if role not in {"system", "user", "assistant", "tool"} or not content:
        return None
    return {"role": role, "content": content}


def _extract_evidence_referents(
    evidence: tuple[Evidence, ...],
) -> dict[str, tuple[str, ...]]:
    asset_tables = _OrderedStrings()
    search_tables = _OrderedStrings()
    database_tables = _OrderedStrings()
    columns = _OrderedStrings()
    monitors = _OrderedStrings()
    schemas = _OrderedStrings()
    metrics = _OrderedStrings()
    for item in evidence:
        if not item.accepted:
            continue
        payload = item.payload
        if item.kind == "schema.asset_profile":
            table_payloads = _tables_from_schema_payload(payload)
            target = asset_tables if _schema_scope(item) == "asset" else database_tables
            for table in table_payloads:
                table_name = table.get("name")
                if table_name:
                    target.add(str(table_name), source="evidence")
                for column in _columns_from_table(table):
                    columns.add(column, source="evidence")
        elif item.kind == "schema.search_result":
            for table in payload.get("tables", []) or []:
                if not _schema_search_table_matched(table):
                    continue
                if isinstance(table, Mapping) and table.get("name"):
                    search_tables.add(str(table["name"]), source="evidence")
                if isinstance(table, Mapping):
                    for column in table.get("matched_columns", []) or []:
                        if isinstance(column, Mapping) and column.get("name"):
                            columns.add(str(column["name"]), source="evidence")
        elif item.kind == "query.plan.proposal":
            structured = payload.get("structured_plan")
            if isinstance(structured, Mapping):
                for table in structured.get("tables", []) or []:
                    asset_tables.add(str(table), source="evidence")
            for table in payload.get("tables", []) or []:
                asset_tables.add(str(table), source="evidence")
        elif item.kind == "sql.validation":
            for table in payload.get("tables", []) or []:
                asset_tables.add(str(table), source="evidence")
            for column in payload.get("columns", []) or []:
                columns.add(str(column), source="evidence")
        elif item.kind in {"monitor.definition", "monitor.proposal"}:
            monitor_id = payload.get("monitor_id")
            if monitor_id is None and isinstance(payload.get("monitor"), Mapping):
                monitor_id = payload["monitor"].get("id")
            if monitor_id is not None:
                monitors.add(str(monitor_id), source="evidence")
            source_scope = payload.get("source_scope")
            if isinstance(source_scope, list):
                asset_tables.extend(source_scope, source="evidence")
        elif item.kind in {"metric.result", "metric.definition"}:
            metric = payload.get("metric_id") or payload.get("name")
            if metric is not None:
                metrics.add(str(metric), source="evidence")
    return {
        "tables": asset_tables.values or search_tables.values or database_tables.values,
        "columns": columns.values,
        "monitors": monitors.values,
        "schemas": schemas.values,
        "metrics": metrics.values,
    }


def _query_scope_from_evidence(
    operation_id: str,
    evidence: tuple[Evidence, ...],
) -> DbSessionQueryScope | None:
    tables = _OrderedStrings()
    selected_columns = _OrderedStrings()
    filters: list[dict[str, Any]] = []
    result_row_count: int | None = None

    for item in evidence:
        if not item.accepted:
            continue
        payload = item.payload
        if item.kind == "query.plan.proposal":
            structured = payload.get("structured_plan")
            if isinstance(structured, Mapping):
                tables.extend(
                    structured.get("selected_tables", ()) or (),
                    source="query.plan",
                )
                for table in structured.get("tables", ()) or ():
                    tables.add(table, source="query.plan")
                for column in structured.get("selected_columns", ()) or ():
                    selected_columns.add(column, source="query.plan")
                for filter_item in structured.get("filters", ()) or ():
                    _append_query_filter(filters, filter_item)
            _add_sql_scope(
                _first_sql_value(payload),
                tables=tables,
                selected_columns=selected_columns,
                filters=filters,
            )
        elif item.kind == "sql.validation":
            for table in payload.get("referenced_tables", ()) or ():
                tables.add(table, source="sql.validation")
            for table in payload.get("tables", ()) or ():
                tables.add(table, source="sql.validation")
            for column in payload.get("selected_columns", ()) or ():
                selected_columns.add(column, source="sql.validation")
            _add_sql_scope(
                _first_sql_value(payload),
                tables=tables,
                selected_columns=selected_columns,
                filters=filters,
            )
        elif item.kind == "query.result":
            rows = payload.get("rows")
            if isinstance(rows, list):
                result_row_count = len(rows)

    if not tables.values and not filters and not selected_columns.values:
        return None
    return DbSessionQueryScope(
        operation_id=operation_id,
        tables=tables.values,
        filters=tuple(filters[:_MAX_QUERY_FILTERS]),
        selected_columns=selected_columns.values,
        result_row_count=result_row_count,
    )


def _add_sql_scope(
    sql: str,
    *,
    tables: _OrderedStrings,
    selected_columns: _OrderedStrings,
    filters: list[dict[str, Any]],
) -> None:
    if not sql.strip():
        return
    try:
        analysis = analyze_sql(sql)
    except (ImportError, SqlAnalysisError, ValueError):
        return
    for table in analysis.tables:
        if not table.is_cte:
            tables.add(table.short_key, source="sql.analysis")
    for item in analysis.select_items:
        selected = item.alias or item.expression_sql
        selected_columns.add(selected, source="sql.analysis")
    for predicate in analysis.literal_predicates:
        column = predicate.column.name
        if predicate.column.table:
            column = f"{predicate.column.table}.{predicate.column.name}"
        _append_query_filter(
            filters,
            {
                "column": column,
                "operator": predicate.operator,
                "values": list(predicate.values),
            },
        )


def _append_query_filter(filters: list[dict[str, Any]], raw: Any) -> None:
    if len(filters) >= _MAX_QUERY_FILTERS:
        return
    compact = _compact_query_filter(raw)
    if compact is None:
        return
    key = (
        compact["column"].lower(),
        compact["operator"].lower(),
        tuple(str(item).lower() for item in compact.get("values", ())),
    )
    existing = {
        (
            item["column"].lower(),
            item["operator"].lower(),
            tuple(str(value).lower() for value in item.get("values", ())),
        )
        for item in filters
    }
    if key not in existing:
        filters.append(compact)


def _compact_query_filter(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    column = str(raw.get("column") or raw.get("ref") or "").strip()
    operator = str(raw.get("operator") or raw.get("op") or "").strip()
    values = _filter_value_strings(
        raw.get("values") if "values" in raw else raw.get("value")
    )
    if not column or not operator or not values:
        return None
    if _SENSITIVE_FILTER_COLUMN_RE.search(column):
        return None
    return {
        "column": _clip(column, _MAX_IDENTIFIER_CHARS),
        "operator": _clip(operator, _MAX_IDENTIFIER_CHARS),
        "values": [
            _clip(value, _MAX_FILTER_VALUE_CHARS)
            for value in values[:_MAX_FILTER_VALUES]
        ],
    }


def _filter_value_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = (value,)
    result = []
    for item in values:
        if item is None or isinstance(item, Mapping):
            continue
        text = str(item).strip()
        if text:
            result.append(text)
    return tuple(result)


def _first_sql_value(payload: Any) -> str:
    if isinstance(payload, Mapping):
        for key in ("sql", "planned_sql", "selected_sql", "query", "statement"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in payload.values():
            nested = _first_sql_value(value)
            if nested:
                return nested
    if isinstance(payload, (list, tuple)):
        for value in payload:
            nested = _first_sql_value(value)
            if nested:
                return nested
    return ""


def _schema_search_table_matched(table: Any) -> bool:
    if not isinstance(table, Mapping):
        return False
    try:
        if float(table.get("score") or 0) > 0:
            return True
    except (TypeError, ValueError):
        pass
    return bool(table.get("matched_columns") or table.get("match_reasons"))


def _schema_scope(evidence: Evidence) -> str | None:
    if evidence.metadata.get("scope"):
        return str(evidence.metadata["scope"])
    metadata = evidence.payload.get("metadata")
    if isinstance(metadata, Mapping) and metadata.get("scope"):
        return str(metadata["scope"])
    asset = evidence.payload.get("asset")
    if isinstance(asset, Mapping):
        return "asset"
    table = evidence.payload.get("table")
    if isinstance(table, Mapping):
        return "asset"
    return None


def _tables_from_schema_payload(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    nested_schema = payload.get("schema")
    if isinstance(nested_schema, Mapping):
        nested_tables = _tables_from_schema_payload(nested_schema)
        if nested_tables:
            return nested_tables
    tables = payload.get("tables")
    if isinstance(tables, list):
        return tuple(table for table in tables if isinstance(table, Mapping))
    asset = payload.get("asset")
    if isinstance(asset, Mapping):
        return (
            {
                "name": asset.get("name") or payload.get("table_name"),
                "columns": payload.get("fields") or payload.get("columns") or [],
            },
        )
    table = payload.get("table")
    if isinstance(table, Mapping):
        return (
            {
                "name": table.get("name") or payload.get("table_name"),
                "columns": payload.get("fields") or payload.get("columns") or [],
            },
        )
    return ()


def _columns_from_table(table: Mapping[str, Any]) -> tuple[str, ...]:
    columns = []
    for column in table.get("columns") or table.get("fields") or []:
        if isinstance(column, Mapping) and (
            column.get("name") or column.get("column_name")
        ):
            columns.append(str(column.get("name") or column.get("column_name")))
    return tuple(columns)


def _extract_transcript_referents(
    messages: tuple[dict[str, str], ...],
) -> dict[str, tuple[str, ...]]:
    tables = _OrderedStrings()
    columns = _OrderedStrings()
    for message in messages:
        if message["role"] != "assistant":
            continue
        content = message["content"]
        for match in _IDENTIFIER_RE.finditer(content):
            tables.add(match.group(1), source="conversation")
        for match in _SCHEMA_LINE_RE.finditer(content):
            tables.add(match.group(1), source="conversation")
            for part in match.group(2).split(","):
                value = part.strip()
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
                    columns.add(value, source="conversation")
    return {"tables": tables.values, "columns": columns.values}


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (_clip(value, _MAX_IDENTIFIER_CHARS),)
    return tuple(
        _clip(str(item), _MAX_IDENTIFIER_CHARS) for item in value if item is not None
    )


def _compact_referents(referents: DbSessionReferents) -> dict[str, list[str]]:
    return {
        key: [_clip(item, _MAX_IDENTIFIER_CHARS) for item in values[:_MAX_REFERENTS]]
        for key, values in referents.to_dict().items()
    }


def _compact_string_mapping(value: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, item in value.items():
        if item is None:
            continue
        result[_clip(str(key), _MAX_IDENTIFIER_CHARS)] = _clip(
            str(item), _MAX_IDENTIFIER_CHARS
        )
    return result


def _compact_diagnostics(value: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = dict(value)
    result: dict[str, Any] = {
        "sources": [
            _clip(str(item), _MAX_IDENTIFIER_CHARS)
            for item in diagnostics.get("sources", []) or []
        ],
        "bounded": dict(diagnostics.get("bounded") or {}),
    }
    for key in (
        "conversation_message_count",
        "recent_operation_count",
        "evidence_operation_count",
    ):
        if key in diagnostics:
            result[key] = diagnostics[key]
    referent_sources = diagnostics.get("referent_sources")
    if isinstance(referent_sources, Mapping):
        result["referent_sources"] = {
            _clip(str(ref_type), _MAX_IDENTIFIER_CHARS): _compact_string_mapping(
                sources if isinstance(sources, Mapping) else {}
            )
            for ref_type, sources in referent_sources.items()
        }
    return result


def _optional_clip(value: str | None) -> str | None:
    return _clip(value, _MAX_IDENTIFIER_CHARS) if value is not None else None


def _clip(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def _fingerprint_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _OrderedStrings:
    def __init__(self) -> None:
        self._values: list[str] = []
        self._seen: set[str] = set()
        self.sources: dict[str, str] = {}

    @property
    def values(self) -> tuple[str, ...]:
        return tuple(
            _clip(value, _MAX_IDENTIFIER_CHARS)
            for value in self._values[:_MAX_REFERENTS]
        )

    def add(self, value: Any, *, source: str) -> None:
        text = _clip(str(value or "").strip(), _MAX_IDENTIFIER_CHARS)
        if not text:
            return
        key = text.lower()
        if key in self._seen:
            return
        self._seen.add(key)
        self._values.append(text)
        self.sources[text] = source

    def extend(self, values: Iterable[Any], *, source: str) -> None:
        for value in values:
            self.add(value, source=source)
