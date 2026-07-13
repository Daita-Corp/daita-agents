"""Deterministic extractor for explicit DB memory command intents."""

from __future__ import annotations

import re
from typing import Any

from daita.db.models import DbRequest

from .types import DB_MEMORY_COMMAND_ACTIONS, DbMemoryCommand, DbMemoryIntent


class DeterministicDbMemoryIntentExtractor:
    """Extract only obvious DB memory intents from user-authored commands."""

    def extract(
        self,
        command: DbMemoryCommand,
        request: DbRequest,
        *,
        source_identity: str | None,
        workspace_scope: str = "source",
    ) -> DbMemoryIntent:
        raw = {**command.constraints, **command.metadata}
        action = str(raw.get("action") or command.action or "").strip().lower()
        if action not in DB_MEMORY_COMMAND_ACTIONS:
            action = "remember"
        metadata = _metadata_object(raw.get("metadata"))
        schema_refs = _schema_refs(
            raw.get("schema_refs") or metadata.get("schema_refs")
        )
        if not schema_refs:
            schema_refs = _schema_refs(
                {
                    "table": raw.get("table") or metadata.get("table"),
                    "column": raw.get("column") or metadata.get("column"),
                }
            )
        catalog_refs = _catalog_refs(
            raw.get("catalog_refs")
            or metadata.get("catalog_refs")
            or raw.get("catalog_ref")
            or metadata.get("catalog_ref")
        )
        kind = _clean(raw.get("kind")) or _infer_kind(command.prompt, raw, metadata)
        text = _clean(raw.get("text") or raw.get("content")) or _text_from_prompt(
            command.prompt
        )
        if _generic_memory_text(text):
            text = None
        key = _clean(raw.get("key")) or _infer_key(kind, text, raw, metadata)
        confidence = _float(raw.get("confidence") or metadata.get("confidence"), 1.0)
        importance = _float(raw.get("importance"), 0.7)
        return DbMemoryIntent(
            action=action,
            kind=kind,
            key=key,
            text=text,
            schema_refs=schema_refs,
            catalog_refs=catalog_refs,
            source_identity=_clean(raw.get("source_identity")) or source_identity,
            workspace_scope=_clean(raw.get("workspace_scope")) or workspace_scope,
            confidence=confidence,
            metadata={
                **metadata,
                **_schema_ref_metadata(schema_refs),
                **({"catalog_refs": list(catalog_refs)} if catalog_refs else {}),
            },
            importance=importance,
            diagnostics={"extractor": "deterministic_db_memory"},
        )


def _metadata_object(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _clean(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _schema_refs(value: Any) -> tuple[dict[str, str], ...]:
    refs: list[dict[str, str]] = []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return ()
    for item in value:
        if not isinstance(item, dict):
            continue
        table = _clean(item.get("table") or item.get("table_name"))
        column = _clean(item.get("column") or item.get("column_name"))
        if table:
            ref = {"table": table}
            if column:
                ref["column"] = column
            refs.append(ref)
    return tuple(refs)


def _catalog_refs(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _schema_ref_metadata(refs: tuple[dict[str, str], ...]) -> dict[str, Any]:
    if not refs:
        return {}
    first = refs[0]
    return {
        "schema_refs": [dict(item) for item in refs],
        **({"table": first["table"]} if first.get("table") else {}),
        **({"column": first["column"]} if first.get("column") else {}),
    }


def _infer_kind(
    prompt: str,
    raw: dict[str, Any],
    metadata: dict[str, Any],
) -> str | None:
    lowered = prompt.lower()
    if "value_alias" in raw or "alias" in lowered:
        return "value_alias"
    if "unit" in lowered or raw.get("unit") or metadata.get("unit"):
        return "unit_convention"
    if "schema" in lowered or raw.get("schema_refs") or metadata.get("schema_refs"):
        return "schema_interpretation"
    if "contract" in lowered:
        return "data_contract_note"
    if "metric" in lowered or "definition" in lowered or "revenue" in lowered:
        return "metric_definition"
    if "rule" in lowered or "remember" in lowered or "note" in lowered:
        return "business_rule"
    return None


def _text_from_prompt(prompt: str) -> str | None:
    text = str(prompt or "").strip()
    text = re.sub(r"(?i)\bplease\b", "", text).strip()
    text = re.sub(r"(?i)\b(update_memory|memory\.update)\b.*$", "", text).strip()
    text = re.sub(
        r"(?i)^(remember|note|update|replace|change)\s+(that\s+|the\s+)?",
        "",
        text,
    ).strip()
    return text.rstrip(". ") + "." if text else None


def _infer_key(
    kind: str | None,
    text: str | None,
    raw: dict[str, Any],
    metadata: dict[str, Any],
) -> str | None:
    if not kind or not text:
        return None
    metric = _clean(raw.get("metric") or metadata.get("metric"))
    lowered = text.lower()
    if kind == "metric_definition":
        if metric:
            return f"metric:{_slug(metric)}"
        if "revenue" in lowered:
            return "metric:revenue"
    table = _clean(raw.get("table") or metadata.get("table"))
    column = _clean(raw.get("column") or metadata.get("column"))
    if kind == "unit_convention" and table and column:
        return f"unit_convention:{table}.{column}"
    if kind == "schema_interpretation" and table:
        return f"schema:{table}.{column}" if column else f"schema:{table}"
    if kind == "value_alias" and table and column:
        alias = _slug(_first_alias(text))
        return f"value_alias:{table}.{column}:{alias}"
    return f"{kind}:{_slug(text)[:60]}"


def _first_alias(text: str) -> str:
    match = re.search(r"['\"]([^'\"]+)['\"]", text)
    return match.group(1) if match else text


def _generic_memory_text(text: str | None) -> bool:
    normalized = re.sub(r"[^a-z0-9 ]+", "", str(text or "").lower()).strip()
    return normalized in {"this", "that", "it", "this thing", "important thing"}


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "memory"
