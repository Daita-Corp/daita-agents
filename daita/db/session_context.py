"""Typed, read-only session context for DB conversational workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Iterable, Mapping

from daita.runtime import (
    Evidence,
    Operation,
    OperationStatus,
    RuntimeStore,
    Task,
    TaskStatus,
)

from .fingerprints import persisted_fingerprint, text_fingerprint
from .models import DbRequest
from .sql_analysis import SqlAnalysisError, analyze_sql

_MAX_MESSAGES = 12
_MAX_OPERATIONS = 8
_MAX_REFERENTS = 24
_MAX_QUERY_SCOPES = 4
_MAX_QUERY_FILTERS = 12
_MAX_QUERY_JOINS = 12
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
            result["prompt_fingerprint"] = text_fingerprint(self.prompt)
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
    scope_id: str | None = None
    tables: tuple[str, ...] = ()
    filters: tuple[dict[str, Any], ...] = ()
    joins: tuple[dict[str, Any], ...] = ()
    selected_columns: tuple[str, ...] = ()
    result_row_count: int | None = None
    source_scope: tuple[str, ...] = ()
    schema_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            **(
                {"scope_id": _clip(self.scope_id, _MAX_IDENTIFIER_CHARS)}
                if self.scope_id
                else {}
            ),
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
            "joins": [
                item
                for item in (
                    _compact_query_join(join_item)
                    for join_item in self.joins[:_MAX_QUERY_JOINS]
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
        if self.source_scope:
            result["source_scope"] = [
                _clip(item, _MAX_IDENTIFIER_CHARS)
                for item in self.source_scope[:_MAX_REFERENTS]
            ]
        if self.schema_fingerprint:
            result["schema_fingerprint"] = _clip(
                self.schema_fingerprint, _MAX_IDENTIFIER_CHARS
            )
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
        joins = tuple(
            item
            for item in (
                _compact_query_join(join_item)
                for join_item in value.get("joins", ()) or ()
                if isinstance(join_item, Mapping)
            )
            if item is not None
        )
        row_count = value.get("result_row_count")
        return cls(
            operation_id=str(
                value.get("operation_id") or value.get("source_operation_id") or ""
            ),
            scope_id=(
                str(value["scope_id"]) if value.get("scope_id") is not None else None
            ),
            tables=_strings(value.get("tables")),
            filters=filters,
            joins=joins,
            selected_columns=_strings(value.get("selected_columns")),
            result_row_count=int(row_count) if isinstance(row_count, int) else None,
            source_scope=_strings(value.get("source_scope")),
            schema_fingerprint=(
                str(value["schema_fingerprint"])
                if value.get("schema_fingerprint") is not None
                else None
            ),
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


def session_query_scope_evidence_for(
    operation: Operation,
    evidence: Iterable[Evidence],
    *,
    task_id: str | None = None,
) -> Evidence | None:
    """Build durable scope evidence from accepted runtime query facts."""

    if not _operation_has_session_scope(operation):
        return None
    evidence_tuple = tuple(evidence)
    if any(
        item.kind == "session.query_scope" and item.accepted
        for item in evidence_tuple
        if item.operation_id in {None, operation.id}
    ):
        return None
    if not _has_successful_query_result(evidence_tuple):
        return None
    scope = _query_scope_from_evidence(operation.id, evidence_tuple)
    if scope is None:
        return None
    payload = scope.to_dict()
    payload["source_operation_id"] = operation.id
    if not payload.get("source_scope"):
        source_scope = _strings(operation.request.get("source_scope"))
        if source_scope:
            payload["source_scope"] = list(source_scope)
    refs = _query_scope_source_refs(evidence_tuple)
    if refs:
        payload["source_evidence_refs"] = refs
    scope_fingerprint = persisted_fingerprint(
        {
            "operation_id": operation.id,
            "tables": payload.get("tables") or [],
            "filters": payload.get("filters") or [],
            "joins": payload.get("joins") or [],
            "selected_columns": payload.get("selected_columns") or [],
            "result_row_count": payload.get("result_row_count"),
            "source_evidence_refs": refs,
        }
    )
    payload.setdefault("scope_id", f"session-scope-{scope_fingerprint[:16]}")
    payload["scope_fingerprint"] = scope_fingerprint
    payload.setdefault("version", 1)
    payload_fingerprint = persisted_fingerprint(payload)
    return Evidence(
        id=f"evidence-{payload_fingerprint}",
        kind="session.query_scope",
        owner="db_runtime",
        operation_id=operation.id,
        task_id=task_id,
        accepted=True,
        payload=payload,
        metadata={
            "payload_fingerprint": payload_fingerprint,
            "source_operation_id": operation.id,
            "scope_id": payload["scope_id"],
        },
    )


async def persist_session_query_scopes(
    store: RuntimeStore,
    operation: Operation,
    tasks: Iterable[Task],
    evidence: Iterable[Evidence],
) -> tuple[Evidence, ...]:
    """Persist deterministic query scopes during successful ``db.run`` finalization."""

    if operation.operation_type != "db.run" or not _operation_has_session_scope(
        operation
    ):
        return ()

    task_tuple = tuple(tasks)
    evidence_tuple = tuple(evidence)
    stored_evidence = tuple(await store.list_evidence(operation.id))
    existing_scope_ids = {
        str(item.payload.get("scope_id"))
        for item in (*evidence_tuple, *stored_evidence)
        if item.kind == "session.query_scope" and item.payload.get("scope_id")
    }
    persisted: list[Evidence] = []

    for task in task_tuple:
        if (
            task.operation_id != operation.id
            or task.capability_id != "db.sql.execute_read"
            or task.status is not TaskStatus.SUCCEEDED
        ):
            continue
        result_evidence = tuple(
            item
            for item in evidence_tuple
            if item.operation_id in {None, operation.id}
            and item.task_id == task.id
            and item.kind == "query.result"
        )
        if not _has_successful_query_result(result_evidence):
            continue
        scope_facts = _session_query_scope_facts_for_task(
            task,
            task_tuple,
            evidence_tuple,
        )
        scope_evidence = session_query_scope_evidence_for(
            operation,
            scope_facts,
            task_id=task.id,
        )
        if scope_evidence is None:
            continue
        scope_id = str(scope_evidence.payload.get("scope_id") or "")
        if not scope_id or scope_id in existing_scope_ids:
            continue
        await store.save_evidence(scope_evidence)
        existing_scope_ids.add(scope_id)
        persisted.append(scope_evidence)

    return tuple(persisted)


def _session_query_scope_facts_for_task(
    task: Task,
    tasks: tuple[Task, ...],
    evidence: tuple[Evidence, ...],
) -> tuple[Evidence, ...]:
    tasks_by_id = {
        item.id: item for item in tasks if item.operation_id == task.operation_id
    }
    related_task_ids = {task.id}
    related_evidence_ids: set[str] = set()
    pending_task_ids = [task.id]
    while pending_task_ids:
        related_task = tasks_by_id.get(pending_task_ids.pop())
        if related_task is None:
            continue
        for dependency in related_task.dependencies:
            if dependency.kind.value != "evidence":
                continue
            if dependency.evidence_id:
                related_evidence_ids.add(dependency.evidence_id)
            if (
                dependency.producer_task_id
                and dependency.producer_task_id not in related_task_ids
            ):
                related_task_ids.add(dependency.producer_task_id)
                pending_task_ids.append(dependency.producer_task_id)

    return tuple(
        item
        for item in evidence
        if item.operation_id in {None, task.operation_id}
        and item.kind != "session.query_scope"
        and (
            item.task_id in related_task_ids
            or (item.id is not None and item.id in related_evidence_ids)
        )
    )


def session_scope_binding_evidence_for(
    operation: Operation,
    plan: Any,
    planning_context: Mapping[str, Any],
    *,
    plan_payload: Mapping[str, Any] | None = None,
    task_id: str | None = None,
) -> Evidence | None:
    """Build a binding artifact for a plan that uses prior session scope."""

    session_context = planning_context.get("session_context")
    if not isinstance(session_context, Mapping):
        return None
    raw_scopes = [
        dict(item)
        for item in session_context.get("query_scopes", ()) or ()
        if isinstance(item, Mapping)
    ]
    if not raw_scopes:
        return None

    explicit = _explicit_scope_binding(plan_payload)
    explicit_status = str(
        explicit.get("binding_status") or explicit.get("status") or ""
    ).strip()
    selected_scope = _select_scope_for_binding(
        raw_scopes,
        plan=plan,
        explicit=explicit,
    )
    if selected_scope is None:
        return None

    status = explicit_status or "bound"
    required_filters = _required_scope_filters(selected_scope)
    required_joins = _required_scope_joins(selected_scope)
    if not explicit and not required_filters and not required_joins:
        return None
    omitted = [
        dict(item)
        for item in selected_scope.get("omitted_unsafe_referents", ()) or ()
        if isinstance(item, Mapping)
    ]
    scope_id = str(selected_scope.get("scope_id") or "").strip()
    source_operation_id = str(
        selected_scope.get("operation_id")
        or selected_scope.get("source_operation_id")
        or ""
    ).strip()
    payload: dict[str, Any] = {
        "version": 1,
        "binding_status": status,
        "source_scope_id": scope_id or None,
        "source_operation_id": source_operation_id or None,
        "required_filters": required_filters,
        "required_joins": required_joins,
        "binding_confidence": _binding_confidence(explicit, selected_scope),
        "omitted_unsafe_referents": omitted,
        "source_tables": list(_strings(selected_scope.get("tables"))),
        "source_selected_columns": list(
            _strings(selected_scope.get("selected_columns"))
        ),
    }
    if isinstance(selected_scope.get("result_row_count"), int):
        payload["source_result_row_count"] = int(selected_scope["result_row_count"])
    if selected_scope.get("schema_fingerprint"):
        payload["schema_fingerprint"] = str(selected_scope["schema_fingerprint"])
    if explicit:
        payload["planner_binding"] = {
            str(key): value
            for key, value in explicit.items()
            if key in {"binding_status", "status", "reason", "confidence"}
        }
    binding_fingerprint = persisted_fingerprint(
        {
            "operation_id": operation.id,
            "source_scope_id": payload.get("source_scope_id"),
            "source_operation_id": payload.get("source_operation_id"),
            "required_filters": required_filters,
            "required_joins": required_joins,
            "binding_status": status,
        }
    )
    payload["binding_fingerprint"] = binding_fingerprint
    payload_fingerprint = persisted_fingerprint(payload)
    return Evidence(
        id=f"evidence-{payload_fingerprint}",
        kind="session.scope_binding",
        owner="db_runtime",
        operation_id=operation.id,
        task_id=task_id,
        accepted=True,
        payload=payload,
        metadata={
            "payload_fingerprint": payload_fingerprint,
            "binding_fingerprint": binding_fingerprint,
            **({"source_scope_id": scope_id} if scope_id else {}),
            **(
                {"source_operation_id": source_operation_id}
                if source_operation_id
                else {}
            ),
        },
    )


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
        elif item.kind == "session.query_scope":
            for table in payload.get("tables", []) or []:
                asset_tables.add(str(table), source="evidence")
            for column in payload.get("selected_columns", []) or []:
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
    for item in reversed(evidence):
        if item.accepted and item.kind == "session.query_scope":
            return DbSessionQueryScope.from_dict(item.payload)

    # Compatibility path for operations completed before durable
    # session.query_scope evidence existed. Successful db.run finalization
    # now emits the artifact directly from accepted runtime facts.
    tables = _OrderedStrings()
    selected_columns = _OrderedStrings()
    filters: list[dict[str, Any]] = []
    joins: list[dict[str, Any]] = []
    result_row_count: int | None = None
    source_scope = _OrderedStrings()
    schema_fingerprint: str | None = None

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
                for join_item in structured.get("joins", ()) or ():
                    _append_query_join(joins, join_item)
            _add_sql_scope(
                _first_sql_value(payload),
                tables=tables,
                selected_columns=selected_columns,
                filters=filters,
                joins=joins,
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
                joins=joins,
            )
        elif item.kind == "query.result":
            rows = payload.get("rows")
            if isinstance(rows, list):
                result_row_count = len(rows)
            elif isinstance(payload.get("total_rows"), int):
                result_row_count = int(payload["total_rows"])
        elif item.kind == "planning.context":
            for scope in payload.get("source_scope", ()) or ():
                source_scope.add(scope, source="planning.context")
            if payload.get("schema_fingerprint"):
                schema_fingerprint = str(payload["schema_fingerprint"])

    if not tables.values and not filters and not selected_columns.values:
        return None
    scope_payload = {
        "operation_id": operation_id,
        "tables": list(tables.values),
        "filters": filters[:_MAX_QUERY_FILTERS],
        "joins": joins[:_MAX_QUERY_JOINS],
        "selected_columns": list(selected_columns.values),
        "result_row_count": result_row_count,
        "source_scope": list(source_scope.values),
        "schema_fingerprint": schema_fingerprint,
    }
    scope_id = f"session-scope-{persisted_fingerprint(scope_payload)[:16]}"
    return DbSessionQueryScope(
        operation_id=operation_id,
        scope_id=scope_id,
        tables=tables.values,
        filters=tuple(filters[:_MAX_QUERY_FILTERS]),
        joins=tuple(joins[:_MAX_QUERY_JOINS]),
        selected_columns=selected_columns.values,
        result_row_count=result_row_count,
        source_scope=source_scope.values,
        schema_fingerprint=schema_fingerprint,
    )


def _add_sql_scope(
    sql: str,
    *,
    tables: _OrderedStrings,
    selected_columns: _OrderedStrings,
    filters: list[dict[str, Any]],
    joins: list[dict[str, Any]],
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
    for predicate in getattr(analysis, "column_predicates", ()) or ():
        if str(getattr(predicate, "operator", "") or "").strip() != "=":
            continue
        left = getattr(predicate, "left", None)
        right = getattr(predicate, "right", None)
        _append_query_join(
            joins,
            {
                "left_table": getattr(left, "table", "") if left is not None else "",
                "left_column": getattr(left, "name", "") if left is not None else "",
                "right_table": (
                    getattr(right, "table", "") if right is not None else ""
                ),
                "right_column": (
                    getattr(right, "name", "") if right is not None else ""
                ),
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


def _append_query_join(joins: list[dict[str, Any]], raw: Any) -> None:
    if len(joins) >= _MAX_QUERY_JOINS:
        return
    compact = _compact_query_join(raw)
    if compact is None:
        return
    key = (
        compact["left_table"].lower(),
        compact.get("left_column", "").lower(),
        compact["right_table"].lower(),
        compact.get("right_column", "").lower(),
    )
    reverse = (key[2], key[3], key[0], key[1])
    existing = {
        (
            item["left_table"].lower(),
            item.get("left_column", "").lower(),
            item["right_table"].lower(),
            item.get("right_column", "").lower(),
        )
        for item in joins
    }
    if key not in existing and reverse not in existing:
        joins.append(compact)


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


def _compact_query_join(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    left_table = _short_table_name(
        raw.get("left_table")
        or raw.get("left_asset")
        or raw.get("source_table")
        or raw.get("source_asset")
    )
    right_table = _short_table_name(
        raw.get("right_table")
        or raw.get("right_asset")
        or raw.get("target_table")
        or raw.get("target_asset")
    )
    left_column = _column_name(
        raw.get("left_column")
        or raw.get("left_field")
        or raw.get("source_column")
        or raw.get("source_field")
        or raw.get("left_key")
    )
    right_column = _column_name(
        raw.get("right_column")
        or raw.get("right_field")
        or raw.get("target_column")
        or raw.get("target_field")
        or raw.get("right_key")
    )
    if (not left_table or not right_table) and raw.get("condition"):
        parsed = _join_from_condition(str(raw.get("condition") or ""))
        if parsed is not None:
            left_table = left_table or parsed["left_table"]
            left_column = left_column or parsed.get("left_column", "")
            right_table = right_table or parsed["right_table"]
            right_column = right_column or parsed.get("right_column", "")
    if not left_table or not right_table or left_table.lower() == right_table.lower():
        return None
    item = {
        "left_table": _clip(left_table, _MAX_IDENTIFIER_CHARS),
        "right_table": _clip(right_table, _MAX_IDENTIFIER_CHARS),
    }
    if left_column:
        item["left_column"] = _clip(left_column, _MAX_IDENTIFIER_CHARS)
    if right_column:
        item["right_column"] = _clip(right_column, _MAX_IDENTIFIER_CHARS)
    return item


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


def _operation_has_session_scope(operation: Operation) -> bool:
    return bool(
        _operation_session_id(operation)
        or operation.request.get("session_id")
        or (
            isinstance(operation.request.get("session_context"), Mapping)
            and operation.request["session_context"].get("session_id")
        )
    )


def _has_successful_query_result(evidence: tuple[Evidence, ...]) -> bool:
    for item in evidence:
        if item.kind != "query.result" or not item.accepted:
            continue
        if item.payload.get("success") is False:
            continue
        return True
    return False


def _query_scope_source_refs(evidence: tuple[Evidence, ...]) -> list[dict[str, Any]]:
    refs = []
    for item in evidence:
        if not item.accepted or item.kind not in {
            "planning.context",
            "query.plan.proposal",
            "query.plan.validation",
            "sql.validation",
            "query.result",
            "schema.relationship_path",
        }:
            continue
        if not item.id:
            continue
        refs.append(
            {
                "id": item.id,
                "kind": item.kind,
                "owner": item.owner,
                "payload_fingerprint": item.metadata.get("payload_fingerprint")
                or persisted_fingerprint(item.payload),
            }
        )
    return refs


def _explicit_scope_binding(plan_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(plan_payload, Mapping):
        return {}
    for key in ("session_scope_binding", "scope_binding"):
        value = plan_payload.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    metadata = plan_payload.get("metadata")
    if isinstance(metadata, Mapping):
        return _explicit_scope_binding(metadata)
    return {}


def _select_scope_for_binding(
    scopes: list[dict[str, Any]],
    *,
    plan: Any,
    explicit: Mapping[str, Any],
) -> dict[str, Any] | None:
    requested_scope_id = str(
        explicit.get("source_scope_id") or explicit.get("scope_id") or ""
    ).strip()
    requested_operation_id = str(
        explicit.get("source_operation_id") or explicit.get("operation_id") or ""
    ).strip()
    if requested_scope_id or requested_operation_id:
        for scope in scopes:
            if (
                requested_scope_id
                and str(scope.get("scope_id") or "") == requested_scope_id
            ):
                return scope
            if (
                requested_operation_id
                and str(
                    scope.get("operation_id") or scope.get("source_operation_id") or ""
                )
                == requested_operation_id
            ):
                return scope
        return None

    plan_tables = _plan_table_keys(plan)
    if plan_tables:
        for scope in scopes:
            scope_tables = {
                _short_table_name(item) for item in scope.get("tables") or ()
            }
            if plan_tables & {item.lower() for item in scope_tables if item}:
                return scope
        return None

    status = str(explicit.get("binding_status") or explicit.get("status") or "")
    if status and len(scopes) == 1:
        return scopes[0]
    return None


def _plan_table_keys(plan: Any) -> set[str]:
    tables = {
        _short_table_name(item).lower()
        for item in getattr(plan, "selected_tables", ()) or ()
        if _short_table_name(item)
    }
    sql = str(getattr(plan, "selected_sql", "") or "")
    if sql.strip():
        try:
            analysis = analyze_sql(sql)
        except (ImportError, SqlAnalysisError, ValueError):
            analysis = None
        if analysis is not None:
            tables.update(
                table.short_key.lower()
                for table in analysis.tables
                if not table.is_cte and table.short_key
            )
    return tables


def _required_scope_filters(scope: Mapping[str, Any]) -> list[dict[str, Any]]:
    filters = []
    for item in scope.get("filters", ()) or ():
        compact = _compact_query_filter(item)
        if compact is not None:
            filters.append(compact)
    return filters[:_MAX_QUERY_FILTERS]


def _required_scope_joins(scope: Mapping[str, Any]) -> list[dict[str, Any]]:
    joins = []
    for item in scope.get("joins", ()) or ():
        compact = _compact_query_join(item)
        if compact is not None:
            joins.append(compact)
    return joins[:_MAX_QUERY_JOINS]


def _binding_confidence(
    explicit: Mapping[str, Any],
    scope: Mapping[str, Any],
) -> float:
    for value in (explicit.get("confidence"), scope.get("confidence")):
        if value is None:
            continue
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            continue
        return max(0.0, min(1.0, confidence))
    return 1.0


def _join_from_condition(value: str) -> dict[str, str] | None:
    if value.count("=") != 1:
        return None
    left, right = (_column_ref(part) for part in value.split("=", maxsplit=1))
    if left is None or right is None:
        return None
    return {
        "left_table": left[0],
        "left_column": left[1],
        "right_table": right[0],
        "right_column": right[1],
    }


def _column_ref(value: Any) -> tuple[str, str] | None:
    parts = [
        part.strip().strip('"`[]')
        for part in str(value or "").split(".")
        if part.strip().strip('"`[]')
    ]
    if len(parts) < 2:
        return None
    return _short_table_name(parts[-2]), _column_name(parts[-1])


def _short_table_name(value: Any) -> str:
    text = str(value or "").strip().strip('"`[]')
    if not text:
        return ""
    return text.split(".")[-1].strip().strip('"`[]')


def _column_name(value: Any) -> str:
    text = str(value or "").strip().strip('"`[]')
    if not text:
        return ""
    return text.split(".")[-1].strip().strip('"`[]')


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
