"""Evidence-backed planning context for DB query planning."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

from daita.runtime import Evidence, Operation

from .analysis import structural_schema_fingerprint
from .context_projection import (
    ProjectionContext,
    ProjectionMode,
    policy_summary_from_source,
    project_catalog_hints,
    project_memory_refs,
    project_memory_semantics,
    project_policy_summary,
    project_session_context,
)
from .memory import (
    db_memory_options_from_from_db_options,
    db_memory_refs_from_recall_evidence,
)
from .memory_contracts import project_db_memory_semantic_contracts
from .models import DbIntent, DbRequest, DbRuntimeConfig
from .session_context import db_session_context_from_request


@dataclass(frozen=True)
class DbPlanningContext:
    """Compact context artifact used by deterministic and LLM planners."""

    operation_id: str
    prompt: str
    intent_kind: str
    source_scope: tuple[str, ...]
    dialect: str | None
    schema: dict[str, Any]
    schema_evidence_refs: tuple[str, ...] = ()
    catalog_evidence_refs: tuple[str, ...] = ()
    relationship_evidence_refs: tuple[str, ...] = ()
    column_value_evidence_refs: tuple[str, ...] = ()
    column_value_hints: tuple[dict[str, Any], ...] = ()
    db_memory_refs: tuple[dict[str, Any], ...] = ()
    db_memory_semantics: tuple[dict[str, Any], ...] = ()
    db_memory_evidence_refs: tuple[str, ...] = ()
    db_memory_diagnostics: dict[str, Any] = field(default_factory=dict)
    db_memory_contract_diagnostics: dict[str, Any] = field(default_factory=dict)
    source_evidence_refs: tuple[dict[str, Any], ...] = ()
    source_fingerprints: dict[str, str] = field(default_factory=dict)
    capability_summaries: tuple[dict[str, Any], ...] = ()
    included_sections: tuple[str, ...] = ()
    omitted_sections: tuple[str, ...] = ()
    budget_usage: dict[str, Any] = field(default_factory=dict)
    context_selection_diagnostics: dict[str, Any] = field(default_factory=dict)
    policy_summary: dict[str, Any] = field(default_factory=dict)
    limit_summary: dict[str, Any] = field(default_factory=dict)
    redaction_policy: dict[str, Any] = field(default_factory=dict)
    session_context: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    rendered_context: str = ""
    schema_fingerprint: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "prompt": self.prompt,
            "intent_kind": self.intent_kind,
            "source_scope": list(self.source_scope),
            "dialect": self.dialect,
            "schema": self.schema,
            "schema_evidence_refs": list(self.schema_evidence_refs),
            "catalog_evidence_refs": list(self.catalog_evidence_refs),
            "relationship_evidence_refs": list(self.relationship_evidence_refs),
            "column_value_evidence_refs": list(self.column_value_evidence_refs),
            "column_value_hints": [dict(item) for item in self.column_value_hints],
            "db_memory_refs": [dict(item) for item in self.db_memory_refs],
            "db_memory_semantics": [dict(item) for item in self.db_memory_semantics],
            "db_memory_evidence_refs": list(self.db_memory_evidence_refs),
            "db_memory_diagnostics": dict(self.db_memory_diagnostics),
            "db_memory_contract_diagnostics": dict(self.db_memory_contract_diagnostics),
            "source_evidence_refs": [dict(item) for item in self.source_evidence_refs],
            "source_fingerprints": dict(self.source_fingerprints),
            "capability_summaries": [dict(item) for item in self.capability_summaries],
            "included_sections": list(self.included_sections),
            "omitted_sections": list(self.omitted_sections),
            "budget_usage": dict(self.budget_usage),
            "context_selection_diagnostics": dict(self.context_selection_diagnostics),
            "policy_summary": self.policy_summary,
            "limit_summary": self.limit_summary,
            "redaction_policy": self.redaction_policy,
            "session_context": self.session_context,
            "diagnostics": self.diagnostics,
            "rendered_context": self.rendered_context,
            "schema_fingerprint": self.schema_fingerprint,
        }


class DbPlanningContextBuilder:
    """Build bounded, auditable planning context from accepted evidence."""

    def __init__(self, config: DbRuntimeConfig) -> None:
        self.config = config

    def build(
        self,
        *,
        request: DbRequest,
        intent: DbIntent,
        operation: Operation,
        schema_evidence: Evidence | None,
        catalog_evidence: tuple[Evidence, ...] = (),
        relationship_evidence: tuple[Evidence, ...] = (),
        memory_recall_evidence: tuple[Evidence, ...] = (),
        memory_recall_diagnostics: dict[str, Any] | None = None,
        capability_summaries: tuple[dict[str, Any], ...] = (),
        source: Any = None,
    ) -> DbPlanningContext:
        schema = dict(schema_evidence.payload) if schema_evidence is not None else {}
        schema_fingerprint = structural_schema_fingerprint(schema)
        dialect = (
            str(
                schema.get("database_type")
                or getattr(source, "sql_dialect", None)
                or ""
            )
            or None
        )
        options = _from_db_options(self.config.metadata)
        source_evidence = tuple(
            item
            for item in (schema_evidence, *catalog_evidence, *relationship_evidence)
            if item is not None and item.accepted
        )
        source_refs = tuple(_evidence_ref(item) for item in source_evidence)
        source_fingerprints = {
            str(item.id): str(
                item.metadata.get("payload_fingerprint") or _fingerprint(item.payload)
            )
            for item in source_evidence
            if item.id
        }
        included_sections = ["schema", "limits", "policy", "capabilities"]
        if catalog_evidence:
            included_sections.append("catalog")
        if relationship_evidence:
            included_sections.append("relationships")
        policy_summary = policy_summary_from_source(source)
        projection = ProjectionContext(
            mode=ProjectionMode.PLANNER,
            operation_intent=intent.kind.value,
            safety_frame=operation.metadata.get("safety_frame"),
            policy_summary=policy_summary,
            source_identity=None,
            schema_fingerprint=schema_fingerprint,
            session_id=request.session_id,
            user_id=request.user_id,
        )
        column_value_hints = project_catalog_hints(
            _column_value_hints(
                catalog_evidence,
                schema,
            ),
            projection,
        )
        column_value_evidence = tuple(
            item
            for item in catalog_evidence
            if item.kind
            in {
                "schema.column_value_profile",
                "schema.column_value_search_result",
                "schema.column_value_hint",
            }
        )
        if column_value_hints:
            included_sections.append("column_value_hints")
        memory_options = db_memory_options_from_from_db_options(options)
        projection = ProjectionContext(
            mode=ProjectionMode.PLANNER,
            operation_intent=intent.kind.value,
            safety_frame=operation.metadata.get("safety_frame"),
            policy_summary=policy_summary,
            source_identity=memory_options.get("source_identity"),
            schema_fingerprint=schema_fingerprint,
            session_id=request.session_id,
            user_id=request.user_id,
        )
        raw_db_memory_refs, db_memory_evidence_refs, db_memory_diagnostics = (
            db_memory_refs_from_recall_evidence(
                tuple(item for item in memory_recall_evidence if item.accepted),
                prompt=request.prompt,
                schema=schema,
                source_identity=memory_options.get("source_identity"),
                schema_fingerprint=schema_fingerprint,
                limit=int(memory_options.get("limit") or 3),
                char_budget=int(memory_options.get("char_budget") or 800),
                score_threshold=_float_option(
                    memory_options,
                    "score_threshold",
                    0.45,
                ),
            )
        )
        db_memory_refs = project_memory_refs(raw_db_memory_refs, projection)
        db_memory_diagnostics = {
            **db_memory_diagnostics,
            **dict(memory_recall_diagnostics or {}),
        }
        raw_db_memory_semantics, db_memory_contract_diagnostics = (
            project_db_memory_semantic_contracts(
                raw_db_memory_refs,
                prompt=request.prompt,
                schema=schema,
                policy_summary=policy_summary,
                source_identity=memory_options.get("source_identity"),
            )
        )
        db_memory_semantics = project_memory_semantics(
            raw_db_memory_semantics,
            projection,
        )
        if db_memory_refs:
            included_sections.append("db_memory")
        if db_memory_semantics:
            included_sections.append("db_memory_semantics")
        diagnostics = {
            "schema_table_count": len(schema.get("tables", []) or []),
            "context_budget": options.get("planner_context_budget"),
            "omitted_sections": [],
            "schema_fingerprint": schema_fingerprint,
            "source_evidence_count": len(source_evidence),
            "capability_summary_count": len(capability_summaries),
            "column_value_hint_count": len(column_value_hints),
            "db_memory_ref_count": len(db_memory_refs),
            "db_memory_contract_count": len(db_memory_semantics),
        }
        session_context = project_session_context(
            _compact_session_context(request),
            projection,
        )
        if session_context:
            included_sections.append("session_context")
        context = DbPlanningContext(
            operation_id=operation.id,
            prompt=request.prompt,
            intent_kind=intent.kind.value,
            source_scope=request.source_scope,
            dialect=dialect,
            schema=_compact_schema(schema),
            schema_evidence_refs=_evidence_refs((schema_evidence,)),
            catalog_evidence_refs=_evidence_refs(catalog_evidence),
            relationship_evidence_refs=_evidence_refs(relationship_evidence),
            column_value_evidence_refs=_evidence_refs(column_value_evidence),
            column_value_hints=column_value_hints,
            db_memory_refs=db_memory_refs,
            db_memory_semantics=db_memory_semantics,
            db_memory_evidence_refs=db_memory_evidence_refs,
            db_memory_diagnostics=db_memory_diagnostics,
            db_memory_contract_diagnostics=db_memory_contract_diagnostics,
            source_evidence_refs=source_refs,
            source_fingerprints=source_fingerprints,
            capability_summaries=tuple(capability_summaries),
            included_sections=tuple(included_sections),
            omitted_sections=(),
            budget_usage={
                "rendered_context_chars": 0,
                "capability_summary_count": len(capability_summaries),
                "source_evidence_count": len(source_evidence),
                "db_memory_chars": int(db_memory_diagnostics.get("used_chars") or 0),
            },
            context_selection_diagnostics={
                "mode": "runtime_bounded",
                "capability_summary_owners": sorted(
                    {
                        str(item.get("owner"))
                        for item in capability_summaries
                        if item.get("owner")
                    }
                ),
            },
            policy_summary=project_policy_summary(policy_summary, projection),
            limit_summary={
                "max_rows": self.config.limits.max_rows,
                "timeout_seconds": self.config.limits.timeout_seconds,
                "query_default_limit": getattr(source, "query_default_limit", None),
                "query_max_rows": getattr(source, "query_max_rows", None),
                "query_max_chars": getattr(source, "query_max_chars", None),
            },
            redaction_policy={
                "redact_pii_columns": bool(options.get("redact_pii_columns", True)),
                "include_sample_values": bool(
                    options.get("include_sample_values", False)
                ),
            },
            session_context=session_context,
            diagnostics=diagnostics,
            schema_fingerprint=schema_fingerprint,
        )
        return DbPlanningContext(
            **{
                **context.__dict__,
                "rendered_context": _render_context_summary(context),
                "budget_usage": {
                    **context.budget_usage,
                    "rendered_context_chars": len(_render_context_summary(context)),
                },
            }
        )

    def evidence_for(self, context: DbPlanningContext) -> Evidence:
        return Evidence(
            kind="planning.context",
            owner="db_runtime",
            operation_id=context.operation_id,
            payload=context.to_payload(),
            metadata={
                "schema_fingerprint": context.schema_fingerprint,
                "payload_fingerprint": _fingerprint(context.to_payload()),
            },
        )


def _compact_schema(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "database_type": schema.get("database_type"),
        "database_name": schema.get("database_name"),
        "table_count": schema.get("table_count") or len(schema.get("tables", []) or []),
        "tables": [
            {
                "name": table.get("name"),
                "columns": [
                    {
                        "name": column.get("name"),
                        "data_type": column.get("data_type"),
                        "is_primary_key": column.get("is_primary_key"),
                    }
                    for column in table.get("columns", []) or []
                    if column.get("name")
                ],
                "metadata": dict(table.get("metadata") or {}),
            }
            for table in schema.get("tables", []) or []
            if table.get("name")
        ],
        "foreign_keys": list(schema.get("foreign_keys", []) or []),
    }


def _render_context_summary(context: DbPlanningContext) -> str:
    lines = [
        f"Prompt: {context.prompt}",
        f"Intent: {context.intent_kind}",
        f"Dialect: {context.dialect or 'unknown'}",
        "Tables:",
    ]
    for table in context.schema.get("tables", []) or []:
        columns = ", ".join(
            str(column.get("name"))
            for column in table.get("columns", []) or []
            if column.get("name")
        )
        lines.append(f"- {table.get('name')}: {columns}")
    foreign_keys = context.schema.get("foreign_keys", []) or []
    if foreign_keys:
        lines.append("Relationships:")
        for item in foreign_keys[:50]:
            lines.append(
                "- "
                f"{item.get('source_table')}.{item.get('source_column')} -> "
                f"{item.get('target_table')}.{item.get('target_column')}"
            )
    if context.db_memory_refs:
        lines.append("Database memory:")
        for ref in context.db_memory_refs:
            lines.append(
                "- "
                f"{ref.get('kind')} {ref.get('key')}: "
                f"{str(ref.get('text') or '').strip()}"
            )
    if context.column_value_hints:
        lines.append("Known filter values:")
        for hint in context.column_value_hints[:20]:
            values = []
            for item in hint.get("observed_values", []) or []:
                if isinstance(item, dict):
                    label = str(item.get("value"))
                    if item.get("count") is not None:
                        label = f"{label} ({item.get('count')})"
                    values.append(label)
                else:
                    values.append(str(item))
            if values:
                lines.append(
                    f"- {hint.get('table')}.{hint.get('column')}: "
                    + ", ".join(values[:25])
                )
    referents = context.session_context.get("referents")
    if isinstance(referents, dict):
        tables = referents.get("tables") or []
        if tables:
            lines.append("Session table referents:")
            for table in tables[:20]:
                lines.append(f"- {table}")
        monitors = referents.get("monitors") or []
        if monitors:
            lines.append("Session monitor referents:")
            for monitor in monitors[:10]:
                lines.append(f"- {monitor}")
    query_scopes = context.session_context.get("query_scopes")
    if isinstance(query_scopes, list) and query_scopes:
        lines.append("Session query scopes:")
        for scope in query_scopes[:4]:
            if not isinstance(scope, dict):
                continue
            parts = []
            tables = scope.get("tables") or []
            if tables:
                parts.append("tables " + ", ".join(str(item) for item in tables[:8]))
            filter_text = _session_filter_text(scope.get("filters"))
            if filter_text:
                parts.append("filters " + filter_text)
            row_count = scope.get("result_row_count")
            if isinstance(row_count, int):
                parts.append(f"result rows {row_count}")
            if parts:
                lines.append("- " + "; ".join(parts))
    return "\n".join(lines)


def _session_filter_text(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    filters = []
    for item in value[:12]:
        if not isinstance(item, dict):
            continue
        column = str(item.get("column") or "").strip()
        operator = str(item.get("operator") or "").strip()
        values = item.get("values")
        if not column or not operator or not isinstance(values, list) or not values:
            continue
        filters.append(f"{column} {operator} {', '.join(str(v) for v in values[:8])}")
    return "; ".join(filters)


def _compact_session_context(request: DbRequest) -> dict[str, Any]:
    session_context = db_session_context_from_request(request)
    if session_context is not None:
        context = session_context.to_request_dict()
    else:
        context = getattr(request, "session_context", None)
    if context is None:
        return {}
    if not isinstance(context, dict):
        return {}
    referents = context.get("referents")
    recent_operations = context.get("recent_operations")
    query_scopes = context.get("query_scopes")
    durable_ids = context.get("durable_ids")
    diagnostics = context.get("diagnostics")
    compact: dict[str, Any] = {}
    if isinstance(referents, dict):
        compact["referents"] = {
            key: list(referents.get(key) or [])[:20]
            for key in (
                "tables",
                "columns",
                "schemas",
                "metrics",
                "monitors",
                "approvals",
                "operations",
            )
        }
    if isinstance(recent_operations, list):
        compact["recent_operations"] = recent_operations[:8]
    if isinstance(query_scopes, list):
        compact["query_scopes"] = [
            dict(item) for item in query_scopes[:4] if isinstance(item, dict)
        ]
    if isinstance(durable_ids, dict):
        compact["durable_ids"] = dict(durable_ids)
    if isinstance(diagnostics, dict):
        compact["diagnostics"] = {
            "sources": list(diagnostics.get("sources") or []),
            "referent_sources": dict(diagnostics.get("referent_sources") or {}),
            "bounded": dict(diagnostics.get("bounded") or {}),
        }
        for key in (
            "conversation_message_count",
            "recent_operation_count",
            "evidence_operation_count",
            "query_scope_count",
        ):
            if key in diagnostics:
                compact["diagnostics"][key] = diagnostics[key]
    return compact


def _column_value_hints(
    catalog_evidence: tuple[Evidence, ...],
    schema: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    hints: list[dict[str, Any]] = []
    for evidence in catalog_evidence:
        if evidence.kind == "schema.column_value_hint":
            for hint in evidence.payload.get("hints", []) or []:
                if isinstance(hint, dict):
                    hints.append(_compact_column_value_hint(hint))

    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for hint in hints:
        key = (str(hint.get("table") or ""), str(hint.get("column") or ""))
        if not key[0] or not key[1] or not planner_eligible_column_value_hint(hint):
            continue
        existing = deduped.get(key)
        if existing is None or _prefer_column_value_hint(hint, existing):
            deduped[key] = hint
    return tuple(deduped.values())


def _prefer_column_value_hint(
    candidate: dict[str, Any],
    existing: dict[str, Any],
) -> bool:
    if candidate.get("candidate_mapping") and not existing.get("candidate_mapping"):
        return True
    return False


def _compact_column_value_hint(hint: dict[str, Any]) -> dict[str, Any]:
    observed = []
    for item in hint.get("observed_values", []) or []:
        if isinstance(item, dict):
            observed.append(
                {
                    "value": item.get("value"),
                    **(
                        {"count": item.get("count")}
                        if item.get("count") is not None
                        else {}
                    ),
                }
            )
        else:
            observed.append({"value": item})
    result = {
        "table": hint.get("table"),
        "column": hint.get("column"),
        "profile_ref": hint.get("profile_ref")
        or f"{hint.get('table')}.{hint.get('column')}",
        "distinct_count": hint.get("distinct_count"),
        "observed_values": observed[:25],
        "profile_status": hint.get("profile_status") or "profiled",
        "sampled": bool(hint.get("sampled", False)),
        "truncated": bool(hint.get("truncated", False)),
        "redacted": bool(hint.get("redacted", False)),
        "stale": bool(hint.get("stale", False)),
    }
    if hint.get("stale_reason"):
        result["stale_reason"] = hint.get("stale_reason")
    if hint.get("candidate_mapping"):
        result["candidate_mapping"] = dict(hint["candidate_mapping"])
    return result


def planner_eligible_column_value_hint(hint: dict[str, Any]) -> bool:
    if hint.get("profile_status") != "profiled":
        return False
    if hint.get("stale") or hint.get("redacted") or hint.get("sampled"):
        return False
    if hint.get("truncated"):
        return False
    observed = hint.get("observed_values")
    if not isinstance(observed, list) or not observed or len(observed) > 25:
        return False
    for item in observed:
        value = item.get("value") if isinstance(item, dict) else item
        if value is None:
            return False
    return True


def _evidence_refs(values: tuple[Evidence | None, ...]) -> tuple[str, ...]:
    return tuple(item.id for item in values if item is not None and item.id)


def _evidence_ref(evidence: Evidence) -> dict[str, Any]:
    return {
        "id": evidence.id,
        "kind": evidence.kind,
        "owner": evidence.owner,
        "task_id": evidence.task_id,
        "payload_fingerprint": evidence.metadata.get("payload_fingerprint")
        or _fingerprint(evidence.payload),
    }


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _from_db_options(metadata: dict[str, Any]) -> dict[str, Any]:
    options = metadata.get("from_db_options")
    return options if isinstance(options, dict) else {}


def _float_option(options: dict[str, Any], key: str, default: float) -> float:
    value = options.get(key)
    if value is None:
        return default
    return float(value)
