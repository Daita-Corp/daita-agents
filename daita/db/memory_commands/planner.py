"""Schema-aware proposal planning for explicit DB memory commands."""

from __future__ import annotations

from typing import Any

from daita.db.analysis import stable_fingerprint
from daita.db.memory import (
    DB_SEMANTIC_MEMORY_KINDS,
    DBMemoryRecord,
    db_memory_pii_error,
    normalize_db_memory_record,
)

from .types import DB_MEMORY_MUTATION_ACTIONS, DbMemoryIntent, DbMemoryValidation

SCHEMA_REF_REQUIRED_KINDS = frozenset({"unit_convention", "schema_interpretation"})
CATALOG_REF_REQUIRED_KINDS = frozenset({"value_alias"})


class DbMemoryProposalPlanner:
    """Build accepted or rejected DB memory proposals from typed intents."""

    def plan(
        self,
        intent: DbMemoryIntent,
        *,
        schema: dict[str, Any] | None = None,
        source_identity: str | None,
        schema_fingerprint: str | None = None,
    ) -> tuple[dict[str, Any], DbMemoryValidation]:
        reasons: list[str] = []
        diagnostics: dict[str, Any] = {}
        schema = schema if isinstance(schema, dict) else {}

        if intent.action not in DB_MEMORY_MUTATION_ACTIONS:
            reasons.append("non_mutating_memory_action_not_implemented")
        if intent.kind not in DB_SEMANTIC_MEMORY_KINDS:
            reasons.append("unsupported_or_ambiguous_memory_kind")
        if not intent.key:
            reasons.append("memory_key_required")
        if not intent.text:
            reasons.append("memory_text_required")
        if not source_identity or intent.source_identity != source_identity:
            reasons.append("source_identity_required")
        if intent.workspace_scope != "source":
            reasons.append("workspace_scope_must_be_source")
        if intent.confidence < 0.5:
            reasons.append("confidence_too_low")

        schema_refs = tuple(intent.schema_refs)
        if intent.kind in SCHEMA_REF_REQUIRED_KINDS and not schema_refs:
            reasons.append("schema_refs_required")
        missing_refs = _missing_schema_refs(schema, schema_refs)
        if missing_refs:
            reasons.append("schema_refs_not_found")
            diagnostics["missing_schema_refs"] = missing_refs

        catalog_refs = tuple(intent.catalog_refs)
        metadata = dict(intent.metadata)
        if catalog_refs:
            metadata.setdefault("catalog_refs", list(catalog_refs))
            metadata.setdefault("catalog_profile_ref", catalog_refs[0])
        if intent.kind in CATALOG_REF_REQUIRED_KINDS and not catalog_refs:
            reasons.append("catalog_refs_required")

        metadata.update(
            {
                "source_identity": source_identity,
                "workspace_scope": "source",
                "active": True,
                "confidence": intent.confidence,
                "creation_path": "explicit_intent",
            }
        )
        if schema_fingerprint:
            metadata.setdefault("source_schema_fingerprint", schema_fingerprint)

        record_payload = {
            "kind": intent.kind or "",
            "key": intent.key or "",
            "text": intent.text or "",
            "metadata": metadata,
            "importance": intent.importance,
        }
        try:
            record = normalize_db_memory_record(record_payload)
            pii_error = db_memory_pii_error(
                key=record.key,
                text=record.text,
                metadata=record.metadata,
            )
            if pii_error:
                reasons.append("pii_or_row_level_memory_rejected")
                diagnostics["pii_error"] = pii_error
        except Exception as exc:
            record = None
            reasons.append("memory_record_invalid")
            diagnostics["record_error"] = str(exc)

        accepted = not reasons and record is not None
        validation = DbMemoryValidation(
            accepted=accepted,
            status="accepted" if accepted else "rejected",
            reasons=tuple(dict.fromkeys(reasons)),
            diagnostics=diagnostics,
        )
        proposal = {
            "kind": "db.memory.proposal",
            "action": intent.action,
            "intent": intent.to_dict(),
            "source_identity": source_identity,
            "workspace_scope": "source",
            "schema_fingerprint": schema_fingerprint,
            "schema_refs": [dict(item) for item in schema_refs],
            "catalog_refs": list(catalog_refs),
            "record": record.to_dict() if record is not None else record_payload,
            "validation": validation.to_dict(),
        }
        proposal["proposal_fingerprint"] = stable_fingerprint(proposal)
        return proposal, validation


def _missing_schema_refs(
    schema: dict[str, Any],
    refs: tuple[dict[str, str], ...],
) -> list[dict[str, str]]:
    if not refs:
        return []
    tables = {
        str(table.get("name") or ""): {
            str(column.get("name") or "")
            for column in table.get("columns", []) or []
            if isinstance(column, dict)
        }
        for table in schema.get("tables", []) or []
        if isinstance(table, dict)
    }
    if not tables:
        return []
    missing: list[dict[str, str]] = []
    for ref in refs:
        table = ref.get("table") or ""
        column = ref.get("column")
        if table not in tables:
            missing.append(dict(ref))
            continue
        if column and column not in tables[table]:
            missing.append(dict(ref))
    return missing
