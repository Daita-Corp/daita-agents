"""Evidence-backed planning context for DB query planning."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

from daita.runtime import Evidence, Operation

from .models import DbIntent, DbRequest, DbRuntimeConfig


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
    policy_summary: dict[str, Any] = field(default_factory=dict)
    limit_summary: dict[str, Any] = field(default_factory=dict)
    redaction_policy: dict[str, Any] = field(default_factory=dict)
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
            "policy_summary": self.policy_summary,
            "limit_summary": self.limit_summary,
            "redaction_policy": self.redaction_policy,
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
        source: Any = None,
    ) -> DbPlanningContext:
        schema = dict(schema_evidence.payload) if schema_evidence is not None else {}
        schema_fingerprint = _fingerprint(schema)
        dialect = (
            str(
                schema.get("database_type")
                or getattr(source, "sql_dialect", None)
                or ""
            )
            or None
        )
        options = _from_db_options(self.config.metadata)
        diagnostics = {
            "schema_table_count": len(schema.get("tables", []) or []),
            "context_budget": options.get("planner_context_budget"),
            "omitted_sections": [],
            "schema_fingerprint": schema_fingerprint,
        }
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
            policy_summary={
                "read_only": getattr(source, "read_only", None),
                "allowed_tables": sorted(
                    getattr(source, "allowed_tables", set()) or []
                ),
                "blocked_tables": sorted(
                    getattr(source, "blocked_tables", set()) or []
                ),
                "blocked_columns": sorted(
                    getattr(source, "blocked_columns", set()) or []
                ),
            },
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
            diagnostics=diagnostics,
            schema_fingerprint=schema_fingerprint,
        )
        return DbPlanningContext(
            **{
                **context.__dict__,
                "rendered_context": _render_context_summary(context),
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
    return "\n".join(lines)


def _evidence_refs(values: tuple[Evidence | None, ...]) -> tuple[str, ...]:
    return tuple(item.id for item in values if item is not None and item.id)


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _from_db_options(metadata: dict[str, Any]) -> dict[str, Any]:
    options = metadata.get("from_db_options")
    return options if isinstance(options, dict) else {}
