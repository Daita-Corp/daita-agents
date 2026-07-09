"""Governed DB context projections for planner, result, and audit views."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import re
from typing import Any, Iterable, Mapping

from daita.runtime import Evidence

from .memory import PII_COLUMN_PATTERNS, _detect_pii_value as _memory_detect_pii_value
from .memory_contracts import (
    DB_MEMORY_SEMANTIC_CONTRACT_KEY,
    db_memory_contract_refs,
)


class ProjectionMode(str, Enum):
    """Caller mode for DB-owned context and evidence projection."""

    PLANNER = "planner"
    DIAGNOSTIC = "diagnostic"
    PUBLIC_RESULT = "public_result"
    AUDIT = "audit"


@dataclass(frozen=True)
class ProjectionContext:
    """Facts produced by existing owners and consumed by projection helpers."""

    mode: ProjectionMode = ProjectionMode.PLANNER
    operation_intent: str | None = None
    safety_frame: dict[str, Any] | None = None
    policy_summary: dict[str, Any] | None = None
    guardrail_summary: dict[str, Any] | None = None
    source_identity: str | None = None
    schema_fingerprint: str | None = None
    session_id: str | None = None
    user_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", ProjectionMode(self.mode))
        object.__setattr__(self, "safety_frame", dict(self.safety_frame or {}))
        object.__setattr__(self, "policy_summary", dict(self.policy_summary or {}))
        object.__setattr__(
            self,
            "guardrail_summary",
            dict(self.guardrail_summary or {}),
        )

    @property
    def blocked_tables(self) -> frozenset[str]:
        values = []
        for source in (self.policy_summary, self.safety_frame, self.guardrail_summary):
            values.extend(_string_list(source.get("blocked_tables")))
            values.extend(_string_list(source.get("deny_tables")))
            values.extend(_string_list(source.get("restricted_tables")))
        return frozenset(_ref_key(value) for value in values if _ref_key(value))

    @property
    def blocked_columns(self) -> frozenset[str]:
        values = []
        for source in (self.policy_summary, self.safety_frame, self.guardrail_summary):
            values.extend(_string_list(source.get("blocked_columns")))
            values.extend(_string_list(source.get("blocked_fields")))
            values.extend(_string_list(source.get("sensitive_columns")))
            values.extend(_string_list(source.get("deny_columns")))
        return frozenset(_ref_key(value) for value in values if _ref_key(value))

    @property
    def blocked_values(self) -> frozenset[str]:
        values = []
        for source in (self.policy_summary, self.safety_frame, self.guardrail_summary):
            values.extend(_string_list(source.get("blocked_values")))
            values.extend(_string_list(source.get("sensitive_values")))
        return frozenset(str(value).strip().lower() for value in values if str(value))


def policy_summary_from_source(source: Any) -> dict[str, Any]:
    """Return connector-owned policy facts without evaluating policy."""

    return {
        "read_only": getattr(source, "read_only", None),
        "allowed_tables": sorted(getattr(source, "allowed_tables", set()) or []),
        "blocked_tables": sorted(getattr(source, "blocked_tables", set()) or []),
        "blocked_columns": sorted(getattr(source, "blocked_columns", set()) or []),
    }


def project_policy_summary(
    policy_summary: Mapping[str, Any],
    projection: ProjectionContext,
) -> dict[str, Any]:
    """Project connector policy facts without exposing denied identifiers."""

    if projection.mode is ProjectionMode.AUDIT:
        return dict(policy_summary)
    blocked_tables = _string_list(policy_summary.get("blocked_tables"))
    blocked_columns = _string_list(policy_summary.get("blocked_columns"))
    result = {
        "read_only": policy_summary.get("read_only"),
        "allowed_tables": _string_list(policy_summary.get("allowed_tables")),
        "blocked_tables": ["<redacted>"] if blocked_tables else [],
        "blocked_columns": ["<redacted>"] if blocked_columns else [],
        "blocked_table_count": len(blocked_tables),
        "blocked_column_count": len(blocked_columns),
    }
    return result


def project_session_context(
    session_context: Mapping[str, Any] | None,
    projection: ProjectionContext,
) -> dict[str, Any]:
    """Project compact session context for one caller mode."""

    if not isinstance(session_context, Mapping):
        return {}
    if projection.mode is ProjectionMode.AUDIT:
        return dict(session_context)

    result: dict[str, Any] = {}
    if session_context.get("session_id") is not None:
        result["session_id"] = session_context.get("session_id")
    if (
        projection.mode is not ProjectionMode.PUBLIC_RESULT
        and session_context.get("user_id") is not None
    ):
        result["user_id"] = session_context.get("user_id")

    referents = _project_session_referents(session_context.get("referents"), projection)
    if referents:
        result["referents"] = referents

    recent_operations = session_context.get("recent_operations")
    if (
        isinstance(recent_operations, list)
        and projection.mode is not ProjectionMode.PUBLIC_RESULT
    ):
        result["recent_operations"] = [
            dict(item) for item in recent_operations if isinstance(item, Mapping)
        ]

    query_scopes = project_session_query_scopes(
        session_context.get("query_scopes") or (),
        projection,
    )
    if query_scopes:
        result["query_scopes"] = list(query_scopes)

    durable_ids = session_context.get("durable_ids")
    if (
        isinstance(durable_ids, Mapping)
        and projection.mode is not ProjectionMode.PUBLIC_RESULT
    ):
        result["durable_ids"] = dict(durable_ids)

    diagnostics = session_context.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        result["diagnostics"] = {
            **dict(diagnostics),
            "projection": _projection_summary(projection),
        }
    elif projection.mode is not ProjectionMode.PUBLIC_RESULT:
        result["diagnostics"] = {"projection": _projection_summary(projection)}
    return result


def project_session_query_scopes(
    query_scopes: Iterable[Any],
    projection: ProjectionContext,
) -> tuple[dict[str, Any], ...]:
    """Project prior query scopes without blocked filters or values."""

    if projection.mode is ProjectionMode.AUDIT:
        return tuple(dict(item) for item in query_scopes if isinstance(item, Mapping))

    projected: list[dict[str, Any]] = []
    for scope in query_scopes:
        if not isinstance(scope, Mapping):
            continue
        tables = [
            table
            for table in _string_list(scope.get("tables"))
            if not _table_blocked(table, projection)
        ]
        filters = [
            item
            for item in (
                _project_session_filter(filter_item, projection)
                for filter_item in scope.get("filters", ()) or ()
            )
            if item is not None
        ]
        joins = _project_session_joins(scope.get("joins"), projection)
        selected_columns = [
            column
            for column in _string_list(scope.get("selected_columns"))
            if not _column_blocked_or_sensitive(column, projection)
        ]

        item: dict[str, Any] = {}
        if (
            projection.mode is not ProjectionMode.PUBLIC_RESULT
            and scope.get("scope_id") is not None
        ):
            item["scope_id"] = scope.get("scope_id")
        if (
            projection.mode is not ProjectionMode.PUBLIC_RESULT
            and scope.get("operation_id") is not None
        ):
            item["operation_id"] = scope.get("operation_id")
        if tables:
            item["tables"] = tables
        if filters:
            item["filters"] = filters
        if joins:
            item["joins"] = joins
        if selected_columns:
            item["selected_columns"] = selected_columns
        if isinstance(scope.get("result_row_count"), int):
            item["result_row_count"] = max(0, int(scope["result_row_count"]))
        if item and (
            tables or filters or joins or selected_columns or "result_row_count" in item
        ):
            projected.append(item)
    return tuple(projected)


def project_memory_refs(
    refs: Iterable[Mapping[str, Any]],
    projection: ProjectionContext,
) -> tuple[dict[str, Any], ...]:
    """Project recalled DB memory refs for planner/result visibility."""

    if projection.mode is ProjectionMode.AUDIT:
        return tuple(dict(ref) for ref in refs)

    projected: list[dict[str, Any]] = []
    for ref in refs:
        if not isinstance(ref, Mapping):
            continue
        reason = _memory_ref_redaction_reason(ref, projection)
        if reason:
            projected.append(_redacted_memory_ref(ref, reason, projection))
            continue
        item = dict(ref)
        if projection.mode is ProjectionMode.PUBLIC_RESULT:
            item.pop("text", None)
            item.pop(DB_MEMORY_SEMANTIC_CONTRACT_KEY, None)
        projected.append(item)
    return tuple(projected)


def project_memory_semantics(
    semantics: Iterable[Mapping[str, Any]],
    projection: ProjectionContext,
) -> tuple[dict[str, Any], ...]:
    """Project compact memory semantic contracts without blocked refs."""

    if projection.mode is ProjectionMode.AUDIT:
        return tuple(dict(item) for item in semantics)

    projected: list[dict[str, Any]] = []
    for semantic in semantics:
        if not isinstance(semantic, Mapping):
            continue
        item = dict(semantic)
        if _memory_semantic_blocked(item, projection):
            item["enforceable"] = False
            item["projection"] = {
                "redacted": True,
                "reason": "blocked_by_policy",
            }
            for key in (
                "required_refs",
                "required_relationships",
                "required_filters",
                "required_aggregations",
                "result_shape",
                "unit_conversion",
            ):
                item.pop(key, None)
        elif projection.mode is ProjectionMode.PUBLIC_RESULT:
            for key in (
                "required_refs",
                "required_relationships",
                "required_filters",
                "required_aggregations",
            ):
                item.pop(key, None)
        projected.append(item)
    return tuple(projected)


def project_catalog_hints(
    hints: Iterable[Mapping[str, Any]],
    projection: ProjectionContext,
) -> tuple[dict[str, Any], ...]:
    """Project catalog value hints for planner-safe rendering."""

    if projection.mode is ProjectionMode.AUDIT:
        return tuple(dict(item) for item in hints)

    projected: list[dict[str, Any]] = []
    for hint in hints:
        if not isinstance(hint, Mapping):
            continue
        table = str(hint.get("table") or "").strip()
        column = str(hint.get("column") or "").strip()
        if _table_blocked(table, projection) or _column_blocked_or_sensitive(
            f"{table}.{column}" if table and column else column,
            projection,
        ):
            continue
        item = dict(hint)
        if projection.blocked_values:
            item["observed_values"] = [
                value
                for value in hint.get("observed_values", []) or []
                if not _value_blocked(value, projection)
            ]
        if projection.mode is ProjectionMode.PUBLIC_RESULT:
            item.pop("observed_values", None)
            item.pop("candidate_mapping", None)
        projected.append(item)
    return tuple(projected)


def project_operation_evidence(
    evidence: Iterable[Evidence],
    projection: ProjectionContext,
) -> tuple[Evidence, ...]:
    """Project persisted evidence for caller-facing result views."""

    if projection.mode is ProjectionMode.AUDIT:
        return tuple(evidence)
    return tuple(
        replace(
            item,
            payload=_project_evidence_payload(item, projection),
            metadata={
                **item.metadata,
                "projection_mode": projection.mode.value,
                "projected": True,
            },
        )
        for item in evidence
    )


def _project_evidence_payload(
    evidence: Evidence,
    projection: ProjectionContext,
) -> dict[str, Any]:
    payload = evidence.payload if isinstance(evidence.payload, Mapping) else {}
    base: dict[str, Any] = {
        "projection_mode": projection.mode.value,
        "source_kind": evidence.kind,
        "accepted": evidence.accepted,
        "payload_keys": sorted(str(key) for key in payload.keys()),
    }
    if projection.mode is ProjectionMode.PUBLIC_RESULT:
        base["redacted"] = True

    if evidence.kind in {"sql.validation", "query.plan.validation"}:
        if "valid" in payload:
            base["valid"] = payload.get("valid") is True
        operation = payload.get("operation")
        if operation is not None:
            base["operation"] = str(operation)
        facts = _project_validation_items(
            payload.get("validation_facts")
            or payload.get("warnings")
            or payload.get("validation_warnings"),
            projection,
        )
        if facts and projection.mode is ProjectionMode.DIAGNOSTIC:
            base["validation_facts"] = facts
    elif evidence.kind == "query.result":
        rows = payload.get("rows")
        if isinstance(rows, list):
            base["row_count"] = len(rows)
        for key in ("total_rows", "truncated", "success"):
            if key in payload:
                base[key] = payload[key]
        if "error" in payload:
            error = payload["error"]
            if error is not None and _text_contains_blocked_or_sensitive(
                str(error),
                projection,
            ):
                base["error"] = "<redacted>"
                base["redacted_fields"] = ["error"]
            else:
                base["error"] = error
    elif evidence.kind == "planning.context":
        base["included_sections"] = list(payload.get("included_sections") or [])
        base["omitted_sections"] = list(payload.get("omitted_sections") or [])
        memory_refs = project_memory_refs(
            payload.get("db_memory_refs") or (),
            projection,
        )
        if projection.mode is ProjectionMode.PUBLIC_RESULT:
            memory_refs = tuple(
                item
                for item in memory_refs
                if not (
                    isinstance(item.get("projection"), Mapping)
                    and item["projection"].get("redacted") is True
                )
            )
        if memory_refs:
            base["db_memory_refs"] = [dict(item) for item in memory_refs]
        memory_semantics = project_memory_semantics(
            payload.get("db_memory_semantics") or (),
            projection,
        )
        if projection.mode is ProjectionMode.PUBLIC_RESULT:
            memory_semantics = tuple(
                item
                for item in memory_semantics
                if not (
                    isinstance(item.get("projection"), Mapping)
                    and item["projection"].get("redacted") is True
                )
            )
        if memory_semantics:
            base["db_memory_semantics"] = [dict(item) for item in memory_semantics]
        memory_diagnostics = _project_memory_diagnostics(
            payload.get("db_memory_diagnostics"),
        )
        if memory_diagnostics and memory_refs:
            base["db_memory_diagnostics"] = memory_diagnostics
        contract_diagnostics = _project_memory_contract_diagnostics(
            payload.get("db_memory_contract_diagnostics"),
        )
        if contract_diagnostics and memory_semantics:
            base["db_memory_contract_diagnostics"] = contract_diagnostics
        diagnostics = payload.get("diagnostics")
        if isinstance(diagnostics, Mapping):
            base["diagnostics"] = {
                key: diagnostics[key]
                for key in (
                    "schema_table_count",
                    "column_value_hint_count",
                    "db_memory_ref_count",
                    "db_memory_contract_count",
                    "schema_fingerprint",
                )
                if key in diagnostics
            }
        if projection.mode is ProjectionMode.DIAGNOSTIC:
            session_context = project_session_context(
                payload.get("session_context"),
                replace(projection, mode=ProjectionMode.DIAGNOSTIC),
            )
            if session_context:
                base["session_context"] = session_context
            base["column_value_hints"] = list(
                project_catalog_hints(
                    payload.get("column_value_hints") or (),
                    replace(projection, mode=ProjectionMode.DIAGNOSTIC),
                )
            )
            base["db_memory_ref_count"] = len(payload.get("db_memory_refs") or ())
    elif evidence.kind == "memory.semantic.recall":
        results = payload.get("results")
        if isinstance(results, list):
            base["result_count"] = len(results)
        diagnostics = _project_recall_diagnostics(payload.get("diagnostics"))
        if diagnostics:
            base["diagnostics"] = diagnostics
    elif evidence.kind == "db.memory.selection":
        base["raw_candidate_count"] = int(payload.get("raw_candidate_count") or 0)
        base["included_count"] = int(payload.get("included_count") or 0)
        omitted = payload.get("omitted_counts_by_reason")
        if isinstance(omitted, Mapping):
            base["omitted_count"] = sum(
                int(count) for count in omitted.values() if isinstance(count, int)
            )
        if projection.mode is ProjectionMode.DIAGNOSTIC:
            base["omitted_counts_by_reason"] = dict(
                payload.get("omitted_counts_by_reason") or {}
            )
            base["safe_diagnostic_omission_summaries"] = [
                dict(item)
                for item in payload.get("safe_diagnostic_omission_summaries", ()) or ()
                if isinstance(item, Mapping)
            ]
            budget = payload.get("budget_usage")
            if isinstance(budget, Mapping):
                base["budget_usage"] = dict(budget)
    elif evidence.kind == "db.memory.contracts":
        enforceable = payload.get("enforceable_contracts")
        advisory = payload.get("advisory_contracts")
        if isinstance(enforceable, list):
            base["enforceable_count"] = len(enforceable)
        if isinstance(advisory, list):
            base["advisory_count"] = len(advisory)
        omitted = payload.get("contract_omission_reasons")
        if isinstance(omitted, Mapping):
            base["omitted_count"] = sum(
                int(count) for count in omitted.values() if isinstance(count, int)
            )
        if projection.mode is ProjectionMode.DIAGNOSTIC:
            base["contract_omission_reasons"] = dict(
                payload.get("contract_omission_reasons") or {}
            )
            base["safe_diagnostic_summaries"] = [
                dict(item)
                for item in payload.get("safe_diagnostic_summaries", ()) or ()
                if isinstance(item, Mapping)
            ]
            applicability = payload.get("source_schema_applicability")
            if isinstance(applicability, Mapping):
                base["source_schema_applicability"] = dict(applicability)
    elif evidence.kind == "session.query_scope":
        base["table_count"] = len(_string_list(payload.get("tables")))
        base["filter_count"] = len(
            [
                item
                for item in payload.get("filters", ()) or ()
                if isinstance(item, Mapping)
            ]
        )
        base["join_count"] = len(
            [
                item
                for item in payload.get("joins", ()) or ()
                if isinstance(item, Mapping)
            ]
        )
        if isinstance(payload.get("result_row_count"), int):
            base["result_row_count"] = max(0, int(payload["result_row_count"]))
        if projection.mode is ProjectionMode.DIAGNOSTIC:
            base["scope_id"] = payload.get("scope_id")
            base["source_operation_id"] = payload.get(
                "source_operation_id"
            ) or payload.get("operation_id")
            base["tables"] = [
                table
                for table in _string_list(payload.get("tables"))
                if not _table_blocked(table, projection)
            ]
            base["filters"] = [
                item
                for item in (
                    _project_session_filter(filter_item, projection)
                    for filter_item in payload.get("filters", ()) or ()
                )
                if item is not None
            ]
            base["joins"] = _project_session_joins(payload.get("joins"), projection)
    elif evidence.kind == "session.scope_binding":
        filters = [
            item
            for item in payload.get("required_filters", ()) or ()
            if isinstance(item, Mapping)
        ]
        joins = [
            item
            for item in payload.get("required_joins", ()) or ()
            if isinstance(item, Mapping)
        ]
        omitted = [
            item
            for item in payload.get("omitted_unsafe_referents", ()) or ()
            if isinstance(item, Mapping)
        ]
        base["required_filter_count"] = len(filters)
        base["required_join_count"] = len(joins)
        base["omitted_unsafe_referent_count"] = len(omitted)
        if projection.mode is ProjectionMode.DIAGNOSTIC:
            base["binding_status"] = payload.get("binding_status")
            base["source_scope_id"] = payload.get("source_scope_id")
            base["source_operation_id"] = payload.get("source_operation_id")
            base["required_filters"] = [
                item
                for item in (
                    _project_session_filter(filter_item, projection)
                    for filter_item in filters
                )
                if item is not None
            ]
            base["required_joins"] = _project_session_joins(joins, projection)
            base["omitted_unsafe_referents"] = [
                {"reason": item.get("reason"), "count": item.get("count", 1)}
                for item in omitted
            ]
    elif evidence.kind == "schema.column_value_hint":
        hints = project_catalog_hints(
            payload.get("hints") or (),
            replace(projection, mode=ProjectionMode.DIAGNOSTIC),
        )
        base["hint_count"] = len(hints)
        if projection.mode is ProjectionMode.DIAGNOSTIC:
            base["hints"] = list(hints)
    elif evidence.kind == "schema.asset_profile":
        tables = payload.get("tables")
        if isinstance(tables, list):
            safe_tables = [
                table
                for table in tables
                if isinstance(table, Mapping)
                and not _table_blocked(str(table.get("name") or ""), projection)
            ]
            base["table_count"] = len(safe_tables)
            if projection.mode is ProjectionMode.DIAGNOSTIC:
                base["tables"] = [
                    {
                        "name": table.get("name"),
                        "columns": [
                            column.get("name")
                            for column in table.get("columns", []) or []
                            if isinstance(column, Mapping)
                            and not _column_blocked_or_sensitive(
                                f"{table.get('name')}.{column.get('name')}",
                                projection,
                            )
                        ],
                    }
                    for table in safe_tables
                ]
    else:
        for key in ("status", "success", "error", "reason"):
            value = payload.get(key)
            if value is not None and not _text_contains_blocked_or_sensitive(
                str(value),
                projection,
            ):
                base[key] = value
    return base


def _project_recall_diagnostics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        key: value[key]
        for key in (
            "retrieval_mode",
            "embedding_available",
            "structured_candidate_count",
            "embedding_candidate_count",
            "returned_count",
            "limit",
        )
        if key in value
    }


def _project_memory_diagnostics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        key: value[key]
        for key in (
            "candidate_count",
            "included_count",
            "used_chars",
            "char_budget",
            "limit",
            "score_threshold",
            "omitted_reasons",
        )
        if key in value
    }


def _project_memory_contract_diagnostics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        key: value[key]
        for key in (
            "candidate_count",
            "enforced_count",
            "advisory_count",
            "omitted_count",
            "omitted_reasons",
        )
        if key in value
    }


def _project_session_referents(
    referents: Any,
    projection: ProjectionContext,
) -> dict[str, list[str]]:
    if not isinstance(referents, Mapping):
        return {}
    result: dict[str, list[str]] = {}
    for key, values in referents.items():
        strings = _string_list(values)
        if key == "tables":
            strings = [item for item in strings if not _table_blocked(item, projection)]
        elif key == "columns":
            strings = [
                item
                for item in strings
                if not _column_blocked_or_sensitive(item, projection)
            ]
        result[str(key)] = strings
    return result


def _project_session_filter(
    value: Any,
    projection: ProjectionContext,
) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    column = str(value.get("column") or "").strip()
    if not column or _column_blocked_or_sensitive(column, projection):
        return None
    values = [
        item
        for item in value.get("values", []) or []
        if not _value_blocked(item, projection)
    ]
    if not values:
        return None
    return {
        "column": column,
        "operator": str(value.get("operator") or "").strip(),
        "values": values,
    }


def _project_session_joins(
    value: Any,
    projection: ProjectionContext,
) -> list[dict[str, Any]]:
    joins = value if isinstance(value, (list, tuple)) else []
    projected = []
    for item in joins:
        if not isinstance(item, Mapping):
            continue
        left_table = str(item.get("left_table") or "").strip()
        right_table = str(item.get("right_table") or "").strip()
        left_column = str(item.get("left_column") or "").strip()
        right_column = str(item.get("right_column") or "").strip()
        if _table_blocked(left_table, projection) or _table_blocked(
            right_table,
            projection,
        ):
            continue
        left_ref = f"{left_table}.{left_column}" if left_column else left_table
        right_ref = f"{right_table}.{right_column}" if right_column else right_table
        if _column_blocked_or_sensitive(
            left_ref,
            projection,
        ) or _column_blocked_or_sensitive(right_ref, projection):
            continue
        projected.append(
            {
                "left_table": left_table,
                **({"left_column": left_column} if left_column else {}),
                "right_table": right_table,
                **({"right_column": right_column} if right_column else {}),
            }
        )
    return projected


def _project_validation_items(
    value: Any,
    projection: ProjectionContext,
) -> list[dict[str, Any]]:
    items = value if isinstance(value, (list, tuple)) else [value]
    projected: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        table = str(item.get("table") or item.get("table_name") or "").strip()
        column = str(item.get("column") or item.get("column_name") or "").strip()
        column_ref = f"{table}.{column}" if table and column else column
        if _table_blocked(table, projection) or _column_blocked_or_sensitive(
            column_ref,
            projection,
        ):
            projected.append(
                {
                    "kind": item.get("kind") or "validation_fact",
                    "redacted": True,
                    "reason": "blocked_by_policy",
                }
            )
            continue
        safe = {
            key: item[key]
            for key in (
                "kind",
                "table",
                "table_name",
                "column",
                "column_name",
                "operator",
                "candidates",
            )
            if key in item
        }
        for key in ("literal", "value", "filter_literal"):
            if key in item and not _value_blocked(item[key], projection):
                safe[key] = item[key]
        projected.append(safe)
    return projected


def _memory_ref_redaction_reason(
    ref: Mapping[str, Any],
    projection: ProjectionContext,
) -> str | None:
    contract = ref.get(DB_MEMORY_SEMANTIC_CONTRACT_KEY)
    if _contract_has_blocked_refs(contract, projection):
        return "blocked_by_policy"
    text = " ".join(
        str(ref.get(key) or "")
        for key in ("key", "text", "schema_fingerprint")
        if ref.get(key) is not None
    )
    if _text_contains_blocked_or_sensitive(text, projection):
        return "blocked_by_policy"
    return None


def _redacted_memory_ref(
    ref: Mapping[str, Any],
    reason: str,
    projection: ProjectionContext,
) -> dict[str, Any]:
    key = str(ref.get("key") or "").strip()
    if _text_contains_blocked_or_sensitive(key, projection):
        key = "redacted"
    result = {
        "chunk_id": ref.get("chunk_id"),
        "kind": ref.get("kind"),
        "key": key or "redacted",
        "confidence": ref.get("confidence"),
        "importance": ref.get("importance"),
        "source_identity": ref.get("source_identity"),
        "evidence_refs": list(ref.get("evidence_refs") or []),
        "schema_fingerprint": ref.get("schema_fingerprint"),
        "projection": {"redacted": True, "reason": reason},
    }
    if projection.mode is not ProjectionMode.PUBLIC_RESULT:
        result["text"] = "Memory reference redacted by projection policy."
    return {key: value for key, value in result.items() if value is not None}


def _memory_semantic_blocked(
    semantic: Mapping[str, Any],
    projection: ProjectionContext,
) -> bool:
    values = []
    values.extend(_string_list(semantic.get("required_refs")))
    values.extend(_string_list(semantic.get("required_relationships")))
    for item in semantic.get("required_filters", []) or []:
        if isinstance(item, Mapping):
            values.append(str(item.get("ref") or ""))
            values.append(str(item.get("value") or ""))
    for item in semantic.get("required_aggregations", []) or []:
        if isinstance(item, Mapping):
            values.append(str(item.get("ref") or ""))
    return any(
        _text_contains_blocked_or_sensitive(value, projection) for value in values
    )


def _contract_has_blocked_refs(
    contract: Any,
    projection: ProjectionContext,
) -> bool:
    if not isinstance(contract, Mapping):
        return False
    for ref in db_memory_contract_refs(dict(contract)):
        table = ref.get("table")
        column = ref.get("column")
        if table and _table_blocked(table, projection):
            return True
        if column and _column_blocked_or_sensitive(
            f"{table}.{column}" if table else column,
            projection,
        ):
            return True
    return False


def _table_blocked(table: Any, projection: ProjectionContext) -> bool:
    table_key = _ref_key(table)
    if not table_key:
        return False
    short = table_key.split(".")[-1]
    return table_key in projection.blocked_tables or short in projection.blocked_tables


def _column_blocked_or_sensitive(
    column_ref: Any,
    projection: ProjectionContext,
) -> bool:
    if _column_blocked(column_ref, projection):
        return True
    return _looks_sensitive_column(column_ref)


def _column_blocked(column_ref: Any, projection: ProjectionContext) -> bool:
    ref = _ref_key(column_ref)
    if not ref:
        return False
    table, column = _split_column_ref(ref)
    if table and _table_blocked(table, projection):
        return True
    if ref in projection.blocked_columns or column in projection.blocked_columns:
        return True
    for blocked in projection.blocked_columns:
        blocked_table, blocked_column = _split_column_ref(blocked)
        if ref == blocked:
            return True
        if column == blocked_column and (
            not table or not blocked_table or table == blocked_table
        ):
            return True
    return False


def _value_blocked(value: Any, projection: ProjectionContext) -> bool:
    raw = value.get("value") if isinstance(value, Mapping) else value
    text = str(raw or "").strip().lower()
    if not text:
        return False
    if text in projection.blocked_values:
        return True
    return bool(_memory_detect_pii_value(text))


def _text_contains_blocked_or_sensitive(
    text: str,
    projection: ProjectionContext,
) -> bool:
    lowered = str(text or "").lower()
    if not lowered:
        return False
    if _memory_detect_pii_value(lowered):
        return True
    for table in projection.blocked_tables:
        if table and _ref_token_in_text(table, lowered):
            return True
    for column in projection.blocked_columns:
        if column and _ref_token_in_text(column, lowered):
            return True
        _table, short = _split_column_ref(column)
        if short and _ref_token_in_text(short, lowered):
            return True
    return False


def _looks_sensitive_column(column_ref: Any) -> bool:
    lowered = _ref_key(column_ref).replace(".", "_")
    if not lowered:
        return False
    return any(pattern in lowered for pattern in PII_COLUMN_PATTERNS)


def _ref_token_in_text(ref: str, text: str) -> bool:
    normalized = _ref_key(ref)
    if not normalized:
        return False
    variants = {normalized, normalized.replace(".", "_")}
    _table, column = _split_column_ref(normalized)
    if column:
        variants.add(column)
    return any(
        re.search(rf"(?<![a-z0-9_]){re.escape(value)}(?![a-z0-9_])", text)
        for value in variants
        if value
    )


def _split_column_ref(ref: str) -> tuple[str | None, str]:
    cleaned = _ref_key(ref)
    parts = [part for part in cleaned.split(".") if part]
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return None, parts[-1] if parts else ""


def _ref_key(value: Any) -> str:
    text = str(value or "").strip().strip('`"[]').lower()
    return ".".join(part.strip('`"[]') for part in text.split(".") if part)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [str(value)]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _projection_summary(projection: ProjectionContext) -> dict[str, Any]:
    return {
        "mode": projection.mode.value,
        "blocked_table_count": len(projection.blocked_tables),
        "blocked_column_count": len(projection.blocked_columns),
    }
