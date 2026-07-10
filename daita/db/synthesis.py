"""
Evidence-driven final answer synthesis for DB runtime operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from difflib import get_close_matches
import json
import re
from typing import Any, Literal

from daita.runtime import Evidence, Operation, Task

from .context import DbContextRenderer
from .models import (
    DbIntent,
    DbIntentKind,
    DbOperationContract,
    DbRequest,
    db_optional_int,
)
from .session_context import db_session_context_from_request
from .verification import DbVerificationResult

_ALLOWED_SUFFICIENCY = frozenset(
    {
        "answered",
        "partial",
        "needs_clarification",
        "insufficient_evidence",
    }
)
_DEFAULT_CONTEXT_ROW_BUDGET = 25
_DEFAULT_CONTEXT_CHAR_BUDGET = 16000
_DEFAULT_FIELD_CHAR_BUDGET = 500
_SENSITIVE_FIELD_RE = re.compile(
    r"(password|passcode|secret|token|api[_-]?key|auth|credential|ssn|social[_-]?security|email|phone)",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_PHONE_RE = re.compile(r"\+?\d[\d\-\s().]{7,}\d")
_DATABASE_WIDE_SCHEMA_RE = re.compile(
    r"\b("
    r"what\s+tables\s+(?:exist|are\s+(?:there|available))|"
    r"list\s+(?:all\s+)?tables|"
    r"all\s+tables|"
    r"database\s+schema|"
    r"schema\s+summary|"
    r"summarize\s+(?:the\s+)?(?:database\s+)?schema|"
    r"available\s+(?:tables|data|schema)"
    r")\b",
    re.IGNORECASE,
)
_TABLE_NAME_PROMPT_RE = re.compile(
    r"\b(?:about|for|from|in|named|called)\s+(?:the\s+)?([A-Za-z_][\w.]*)(?:\s+table)?\b|"
    r"\b([A-Za-z_][\w.]*)\s+table\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DbSynthesisResult:
    """Final answer and diagnostics derived from accepted evidence."""

    answer: str
    evidence_refs: tuple[dict[str, str | None], ...]
    warnings: tuple[str, ...]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "evidence_refs": list(self.evidence_refs),
            "warnings": list(self.warnings),
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class SchemaAnswerScope:
    """Prompt-aware schema evidence selected for the final answer."""

    mode: Literal["asset", "database", "ambiguous", "none"]
    requested_assets: tuple[str, ...]
    selected_tables: tuple[dict[str, Any], ...]
    evidence_refs: tuple[str, ...]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "requested_assets": list(self.requested_assets),
            "selected_table_names": [
                str(table.get("name"))
                for table in self.selected_tables
                if table.get("name")
            ],
            "evidence_refs": list(self.evidence_refs),
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class DbAnswerScalarFact:
    """Scalar fact derived from accepted query result evidence."""

    label: str
    value: Any
    aggregate_kind: str | None
    source_evidence_id: str | None
    source_evidence_kind: str = "query.result"
    confidence: str = "high"
    redacted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "value": self.value,
            "aggregate_kind": self.aggregate_kind,
            "source_evidence_id": self.source_evidence_id,
            "source_evidence_kind": self.source_evidence_kind,
            "confidence": self.confidence,
            "redacted": self.redacted,
        }


@dataclass(frozen=True)
class DbAnswerFacts:
    """Typed facts that final answer synthesis must preserve."""

    result_shape: Literal["empty", "scalar", "record", "table"]
    row_count: int
    sampled_row_count: int
    columns: tuple[str, ...]
    scalars: tuple[DbAnswerScalarFact, ...] = ()
    primary_scalar: DbAnswerScalarFact | None = None
    truncated: bool = False
    source_evidence_id: str | None = None
    source_evidence_kind: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_shape": self.result_shape,
            "row_count": self.row_count,
            "sampled_row_count": self.sampled_row_count,
            "columns": list(self.columns),
            "scalars": [fact.to_dict() for fact in self.scalars],
            "primary_scalar": (
                self.primary_scalar.to_dict() if self.primary_scalar else None
            ),
            "truncated": self.truncated,
            "source_evidence_id": self.source_evidence_id,
            "source_evidence_kind": self.source_evidence_kind,
            "diagnostics": self.diagnostics,
        }


class DbSynthesizer:
    """Create final answers only from accepted, verified evidence."""

    def __init__(self, context_renderer: DbContextRenderer | None = None) -> None:
        self.context_renderer = context_renderer or DbContextRenderer()

    def synthesize(
        self,
        request: DbRequest,
        intent: DbIntent,
        contract: DbOperationContract,
        evidence: tuple[Evidence, ...],
        verification: DbVerificationResult,
    ) -> DbSynthesisResult:
        """Return a deterministic answer from verified evidence."""
        if not verification.passed:
            raise ValueError(
                "cannot synthesize final answer before verification passes"
            )

        answer_facts: DbAnswerFacts | None = None
        if intent.kind is DbIntentKind.SCHEMA_QUERY:
            scope = _schema_answer_scope(request, contract, evidence)
            answer = _append_db_memory_annotation(
                _schema_answer(scope),
                evidence,
            )
        elif intent.kind is DbIntentKind.SCHEMA_RELATIONSHIP_QUERY:
            scope = None
            answer = _append_db_memory_annotation(
                _schema_relationship_answer(evidence),
                evidence,
            )
        elif intent.kind in {
            DbIntentKind.DATA_QUERY,
            DbIntentKind.CATALOG_ASSISTED_DATA_QUERY,
        }:
            scope = None
            answer_facts = derive_answer_facts(
                request=request,
                intent=intent,
                contract=contract,
                evidence=evidence,
            )
            answer = _data_answer_from_facts(answer_facts)
        elif _has_monitor_evidence(evidence) or contract.operation_type.startswith(
            "monitor."
        ):
            scope = None
            answer = _monitor_answer(evidence)
        else:
            scope = None
            answer = "The DB operation completed with verified evidence."

        diagnostics = {
            "synthesis": "deterministic",
            "operation_type": contract.operation_type,
            "context": self.context_renderer.render_evidence_summary(evidence),
            "prompt": request.prompt,
            "skill_synthesis_metadata": contract.metadata.get(
                "skill_synthesis_metadata", {}
            ),
        }
        if scope is not None:
            diagnostics["schema_answer_scope"] = scope.to_dict()
        if answer_facts is not None:
            diagnostics["answer_facts"] = answer_facts.to_dict()
        return DbSynthesisResult(
            answer=answer,
            evidence_refs=verification.evidence_refs,
            warnings=(),
            diagnostics=diagnostics,
        )


def _schema_answer(scope: SchemaAnswerScope) -> str:
    if scope.mode == "none":
        return "No schema evidence was produced."
    if scope.mode == "ambiguous":
        requested = (
            f" named {scope.requested_assets[0]}"
            if len(scope.requested_assets) == 1
            else ""
        )
        matches = tuple(scope.diagnostics.get("closest_matches") or ())
        if matches:
            return (
                f"I could not find an exact table{requested}. "
                f"Closest matches: {', '.join(str(item) for item in matches)}."
            )
        return (
            f"I could not find an exact table{requested}. "
            "Please clarify which table you want."
        )
    tables = list(scope.selected_tables)
    parts = []
    for table in tables:
        columns = [
            str(column.get("name") or column.get("column_name"))
            for column in _columns_from_schema_table(table)
            if column.get("name") or column.get("column_name")
        ]
        parts.append(f"{table.get('name')}: {', '.join(columns)}")
    if scope.mode == "asset":
        prefix = f"Found {len(tables)} matching tables. " if len(tables) != 1 else ""
        return prefix + "; ".join(parts)
    return f"Found {len(tables)} tables. " + "; ".join(parts)


def _schema_answer_scope(
    request: DbRequest,
    contract: DbOperationContract,
    evidence: tuple[Evidence, ...],
) -> SchemaAnswerScope:
    accepted = tuple(item for item in evidence if item.accepted)
    inventory = _schema_table_inventory(accepted)
    table_names = tuple(inventory["table_names"])
    if not table_names:
        return SchemaAnswerScope(
            mode="none",
            requested_assets=(),
            selected_tables=(),
            evidence_refs=(),
            diagnostics={"reason": "no_schema_tables"},
        )

    source_requested = _requested_assets_from_source_scope(request, table_names)
    hinted_requested = _requested_assets_from_scope_hint(contract, table_names)
    session_requested = _requested_assets_from_session(request, table_names)
    prompt_requested = _requested_assets_from_prompt(request.prompt, inventory)
    for reason, assets in (
        ("request.source_scope", source_requested),
        ("schema_answer_scope_hint", hinted_requested),
        ("structured_session_referents", session_requested),
        ("best_table_for_prompt", prompt_requested),
    ):
        selected = _tables_for_names(assets, inventory)
        if selected:
            return SchemaAnswerScope(
                mode="asset",
                requested_assets=tuple(assets),
                selected_tables=tuple(table for table, _ in selected),
                evidence_refs=tuple(
                    dict.fromkeys(ref for _, refs in selected for ref in refs if ref)
                ),
                diagnostics={
                    "reason": reason,
                    "database_wide_prompt": _is_database_wide_schema_prompt(
                        request.prompt
                    ),
                },
            )

    if _is_database_wide_schema_prompt(request.prompt):
        tables, evidence_refs = _database_tables(inventory)
        return SchemaAnswerScope(
            mode="database",
            requested_assets=(),
            selected_tables=tuple(tables),
            evidence_refs=evidence_refs,
            diagnostics={"reason": "database_wide_prompt"},
        )

    requested_names = _prompt_table_like_names(request.prompt)
    closest = _closest_table_matches(requested_names, table_names)
    search_matches = _search_result_table_names(accepted)
    if requested_names and not _exact_name_match(requested_names, table_names):
        return SchemaAnswerScope(
            mode="ambiguous",
            requested_assets=tuple(requested_names),
            selected_tables=(),
            evidence_refs=tuple(
                item.id
                for item in accepted
                if item.id and item.kind == "schema.search_result"
            ),
            diagnostics={
                "reason": "missing_or_ambiguous_table",
                "closest_matches": tuple(closest or search_matches[:5]),
            },
        )

    asset_tables = _asset_tables(inventory)
    if asset_tables:
        return SchemaAnswerScope(
            mode="asset",
            requested_assets=tuple(table.get("name") for table, _ in asset_tables),
            selected_tables=tuple(table for table, _ in asset_tables),
            evidence_refs=tuple(
                dict.fromkeys(ref for _, refs in asset_tables for ref in refs if ref)
            ),
            diagnostics={"reason": "asset_profile_evidence"},
        )

    tables, evidence_refs = _database_tables(inventory)
    return SchemaAnswerScope(
        mode="database",
        requested_assets=(),
        selected_tables=tuple(tables),
        evidence_refs=evidence_refs,
        diagnostics={"reason": "database_schema_fallback"},
    )


def _schema_relationship_answer(evidence: tuple[Evidence, ...]) -> str:
    relationship = next(
        (item.payload for item in evidence if item.kind == "schema.relationship_path"),
        {},
    )
    if relationship.get("reachable") is False:
        return "No relationship path was found between the requested tables."
    paths = relationship.get("paths") or []
    if not paths:
        return "No relationship path evidence was produced."
    parts = []
    for path in paths[:3]:
        assets = path.get("assets") or []
        if assets:
            parts.append(" -> ".join(str(asset) for asset in assets))
    if parts:
        return "Found relationship path: " + "; ".join(parts)
    return "Found relationship path evidence for the requested tables."


def _append_db_memory_annotation(answer: str, evidence: tuple[Evidence, ...]) -> str:
    if "Semantic memory note:" in answer:
        return answer
    refs = _db_memory_refs_from_planning_context(evidence)
    if not refs:
        return answer
    notes = []
    for ref in refs[:2]:
        text = str(ref.get("text") or "").strip()
        if not text:
            continue
        notes.append(text)
    if not notes:
        return answer
    return answer.rstrip() + " Semantic memory note: " + " ".join(notes)


def _apply_schema_db_memory_annotation(
    payload: DbAnswerSynthesisPayload,
    *,
    intent: DbIntent,
    evidence: tuple[Evidence, ...],
) -> DbAnswerSynthesisPayload:
    if intent.kind not in {
        DbIntentKind.SCHEMA_QUERY,
        DbIntentKind.SCHEMA_RELATIONSHIP_QUERY,
    }:
        return payload
    answer = _append_db_memory_annotation(payload.answer, evidence)
    if answer == payload.answer:
        return payload
    return replace(payload, answer=answer)


def _db_memory_refs_from_planning_context(
    evidence: tuple[Evidence, ...],
) -> tuple[dict[str, Any], ...]:
    planning = next(
        (
            item
            for item in reversed(evidence)
            if item.accepted and item.kind == "planning.context"
        ),
        None,
    )
    if planning is None:
        return ()
    refs = planning.payload.get("db_memory_refs")
    if not isinstance(refs, list):
        return ()
    return tuple(item for item in refs if isinstance(item, dict))


def _schema_table_inventory(evidence: tuple[Evidence, ...]) -> dict[str, Any]:
    tables_by_name: dict[str, list[tuple[dict[str, Any], tuple[str, ...], str]]] = {}
    table_names: list[str] = []
    database_tables: list[tuple[dict[str, Any], tuple[str, ...], str]] = []
    asset_tables: list[tuple[dict[str, Any], tuple[str, ...], str]] = []
    search_tables: list[tuple[dict[str, Any], tuple[str, ...], str]] = []
    for item in evidence:
        if item.kind not in {"schema.asset_profile", "schema.search_result"}:
            continue
        scope = _schema_scope(item)
        source = (
            "search"
            if item.kind == "schema.search_result"
            else ("asset" if scope == "asset" else "database")
        )
        for table in _tables_from_schema_payload(item.payload):
            name = _table_name(table)
            if not name:
                continue
            normalized = _normalize_table_name(name)
            copied = dict(table)
            copied["name"] = name
            entry = (copied, (item.id,) if item.id else (), source)
            tables_by_name.setdefault(normalized, []).append(entry)
            table_names.append(name)
            if source == "asset":
                asset_tables.append(entry)
            elif source == "search":
                search_tables.append(entry)
            else:
                database_tables.append(entry)
    return {
        "tables_by_name": tables_by_name,
        "table_names": tuple(dict.fromkeys(table_names)),
        "database_tables": database_tables,
        "asset_tables": asset_tables,
        "search_tables": search_tables,
    }


def _requested_assets_from_source_scope(
    request: DbRequest, table_names: tuple[str, ...]
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            resolved
            for item in request.source_scope
            for resolved in (_resolve_table_name(str(item), table_names),)
            if resolved
        )
    )


def _requested_assets_from_scope_hint(
    contract: DbOperationContract, table_names: tuple[str, ...]
) -> tuple[str, ...]:
    hint = contract.metadata.get("schema_answer_scope")
    if not isinstance(hint, dict):
        return ()
    names = hint.get("requested_assets") or hint.get("selected_tables") or ()
    if isinstance(names, str):
        names = (names,)
    return tuple(
        dict.fromkeys(
            resolved
            for item in names
            for resolved in (_resolve_table_name(str(item), table_names),)
            if resolved
        )
    )


def _requested_assets_from_session(
    request: DbRequest, table_names: tuple[str, ...]
) -> tuple[str, ...]:
    session_context = db_session_context_from_request(request)
    if session_context is None:
        return ()
    referent_sources = (
        session_context.diagnostics.get("referent_sources", {}).get("tables", {})
        if isinstance(session_context.diagnostics.get("referent_sources"), dict)
        else {}
    )
    structured_referents = tuple(
        table
        for table in session_context.referents.tables
        if referent_sources.get(table) != "conversation_history"
    )
    candidate_referents = structured_referents or session_context.referents.tables
    return tuple(
        dict.fromkeys(
            resolved
            for item in candidate_referents
            for resolved in (_resolve_table_name(str(item), table_names),)
            if resolved
        )
    )


def _requested_assets_from_prompt(
    prompt: str, inventory: dict[str, Any]
) -> tuple[str, ...]:
    table_names = tuple(inventory["table_names"])
    explicit = tuple(
        dict.fromkeys(
            resolved
            for item in _prompt_table_like_names(prompt)
            for resolved in (_resolve_table_name(item, table_names),)
            if resolved
        )
    )
    if explicit:
        return explicit
    return ()


def _tables_for_names(
    names: tuple[str, ...], inventory: dict[str, Any]
) -> list[tuple[dict[str, Any], tuple[str, ...]]]:
    selected: list[tuple[dict[str, Any], tuple[str, ...]]] = []
    for name in names:
        entries = inventory["tables_by_name"].get(_normalize_table_name(name)) or []
        if not entries:
            continue
        table, refs, _ = _preferred_table_entry(entries)
        selected.append((table, refs))
    return selected


def _preferred_table_entry(
    entries: list[tuple[dict[str, Any], tuple[str, ...], str]],
) -> tuple[dict[str, Any], tuple[str, ...], str]:
    order = {"asset": 0, "search": 1, "database": 2}
    return sorted(entries, key=lambda item: order.get(item[2], 99))[0]


def _database_tables(
    inventory: dict[str, Any],
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    entries = inventory["database_tables"] or _all_table_entries(inventory)
    tables: list[dict[str, Any]] = []
    refs: list[str] = []
    seen: set[str] = set()
    for table, entry_refs, _ in entries:
        normalized = _normalize_table_name(_table_name(table))
        if normalized in seen:
            continue
        seen.add(normalized)
        tables.append(table)
        refs.extend(entry_refs)
    return tables, tuple(dict.fromkeys(refs))


def _asset_tables(
    inventory: dict[str, Any],
) -> list[tuple[dict[str, Any], tuple[str, ...]]]:
    selected = []
    seen: set[str] = set()
    for table, refs, _ in inventory["asset_tables"]:
        normalized = _normalize_table_name(_table_name(table))
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append((table, refs))
    return selected


def _all_table_entries(
    inventory: dict[str, Any],
) -> list[tuple[dict[str, Any], tuple[str, ...], str]]:
    entries: list[tuple[dict[str, Any], tuple[str, ...], str]] = []
    seen: set[tuple[str, str]] = set()
    for source in ("asset_tables", "search_tables", "database_tables"):
        for table, refs, kind in inventory[source]:
            key = (_normalize_table_name(_table_name(table)), kind)
            if key in seen:
                continue
            seen.add(key)
            entries.append((table, refs, kind))
    return entries


def _search_result_table_names(evidence: tuple[Evidence, ...]) -> tuple[str, ...]:
    names: list[str] = []
    for item in evidence:
        if item.kind != "schema.search_result":
            continue
        names.extend(
            name
            for table in _tables_from_schema_payload(item.payload)
            for name in (_table_name(table),)
            if name
        )
    return tuple(dict.fromkeys(names))


def _is_database_wide_schema_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    if _DATABASE_WIDE_SCHEMA_RE.search(lowered):
        return True
    return "schema" in lowered and not _prompt_table_like_names(prompt)


def _prompt_table_like_names(prompt: str) -> tuple[str, ...]:
    names: list[str] = []
    for match in _TABLE_NAME_PROMPT_RE.finditer(prompt):
        name = match.group(1) or match.group(2)
        if not name:
            continue
        lowered = name.lower()
        if lowered in {"the", "a", "an", "this", "that", "what", "which"}:
            continue
        names.append(name)
    return tuple(dict.fromkeys(names))


def _closest_table_matches(
    requested_names: tuple[str, ...], table_names: tuple[str, ...]
) -> tuple[str, ...]:
    if not requested_names:
        return ()
    display_by_normalized = {
        _normalize_table_name(name): name for name in table_names if name
    }
    matches: list[str] = []
    for requested in requested_names:
        normalized = _normalize_table_name(requested)
        for match in get_close_matches(
            normalized, tuple(display_by_normalized), n=3, cutoff=0.55
        ):
            matches.append(display_by_normalized[match])
    return tuple(dict.fromkeys(matches))


def _exact_name_match(
    requested_names: tuple[str, ...], table_names: tuple[str, ...]
) -> bool:
    normalized = {_normalize_table_name(name) for name in table_names}
    return any(_normalize_table_name(name) in normalized for name in requested_names)


def _resolve_table_name(raw: str, table_names: tuple[str, ...]) -> str | None:
    if not raw:
        return None
    normalized_raw = _normalize_table_name(raw)
    short_raw = _normalize_table_name(raw.split(".")[-1])
    for table_name in table_names:
        normalized = _normalize_table_name(table_name)
        short = _normalize_table_name(table_name.split(".")[-1])
        if normalized_raw in {normalized, short} or short_raw in {normalized, short}:
            return table_name
    return None


def _normalize_table_name(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def _table_name(table: dict[str, Any]) -> str:
    return str(table.get("name") or table.get("table_name") or "").strip()


def _schema_scope(evidence: Evidence) -> str | None:
    if evidence.metadata.get("scope"):
        return str(evidence.metadata["scope"])
    payload_metadata = evidence.payload.get("metadata")
    if isinstance(payload_metadata, dict) and payload_metadata.get("scope"):
        return str(payload_metadata["scope"])
    if isinstance(evidence.payload.get("asset"), dict):
        return "asset"
    if isinstance(evidence.payload.get("table"), dict):
        return "asset"
    return None


def _data_answer(evidence: tuple[Evidence, ...]) -> str:
    return _data_answer_from_facts(derive_answer_facts(evidence=evidence))


def derive_answer_facts(
    *,
    evidence: tuple[Evidence, ...],
    request: DbRequest | None = None,
    intent: DbIntent | None = None,
    contract: DbOperationContract | None = None,
) -> DbAnswerFacts:
    """Derive typed answer facts from accepted result evidence."""
    del request, intent, contract
    accepted = tuple(item for item in evidence if item.accepted)
    query_result = next(
        (item for item in reversed(accepted) if item.kind == "query.result"), None
    )
    if query_result is None:
        return DbAnswerFacts(
            result_shape="empty",
            row_count=0,
            sampled_row_count=0,
            columns=(),
            diagnostics={"reason": "query_result_missing"},
        )

    raw_rows = query_result.payload.get("rows") or []
    row_dicts = _row_dicts(raw_rows if isinstance(raw_rows, list) else [])
    row_count = _safe_int(query_result.payload.get("total_rows"), len(row_dicts))
    sampled_row_count = len(row_dicts)
    truncated = bool(query_result.payload.get("truncated"))
    columns = tuple(dict.fromkeys(column for row in row_dicts for column in row.keys()))
    source_id = query_result.id

    if not row_dicts:
        return DbAnswerFacts(
            result_shape="empty",
            row_count=row_count,
            sampled_row_count=sampled_row_count,
            columns=columns,
            truncated=truncated,
            source_evidence_id=source_id,
            source_evidence_kind=query_result.kind,
        )

    if sampled_row_count == 1:
        row = row_dicts[0]
        aggregate_kinds = _aggregate_kinds_by_column(accepted, tuple(row.keys()))
        scalars = tuple(
            _scalar_fact(
                label=label,
                value=value,
                aggregate_kind=aggregate_kinds.get(_normalize_answer_label(label)),
                source_evidence_id=source_id,
                source_evidence_kind=query_result.kind,
            )
            for label, value in row.items()
        )
        result_shape = "scalar" if len(row) == 1 else "record"
        return DbAnswerFacts(
            result_shape=result_shape,
            row_count=row_count,
            sampled_row_count=sampled_row_count,
            columns=tuple(row.keys()),
            scalars=scalars,
            primary_scalar=scalars[0] if result_shape == "scalar" and scalars else None,
            truncated=truncated,
            source_evidence_id=source_id,
            source_evidence_kind=query_result.kind,
        )

    return DbAnswerFacts(
        result_shape="table",
        row_count=row_count,
        sampled_row_count=sampled_row_count,
        columns=columns,
        truncated=truncated,
        source_evidence_id=source_id,
        source_evidence_kind=query_result.kind,
    )


def _row_dicts(rows: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            out.append({str(key): value for key, value in row.items()})
        else:
            out.append({"value": row})
    return out


def _safe_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _scalar_fact(
    *,
    label: str,
    value: Any,
    aggregate_kind: str | None,
    source_evidence_id: str | None,
    source_evidence_kind: str,
) -> DbAnswerScalarFact:
    if aggregate_kind == "count":
        redacted = False
        rendered = value
        truncated = False
    else:
        redacted, rendered, truncated = _redact_value(label, value)
    return DbAnswerScalarFact(
        label=label,
        value=rendered,
        aggregate_kind=aggregate_kind,
        source_evidence_id=source_evidence_id,
        source_evidence_kind=source_evidence_kind,
        confidence="high",
        redacted=bool(redacted or truncated),
    )


def _data_answer_from_facts(facts: DbAnswerFacts) -> str:
    if facts.diagnostics.get("reason") == "query_result_missing":
        return "No query result was produced."
    if facts.result_shape == "empty":
        return "The query returned no rows."
    if facts.primary_scalar is not None:
        return _scalar_answer(facts.primary_scalar)
    if facts.scalars:
        return "; ".join(_scalar_clause(fact) for fact in facts.scalars) + "."
    count = facts.row_count or facts.sampled_row_count
    if facts.truncated and facts.sampled_row_count and count > facts.sampled_row_count:
        return (
            f"Returned {facts.sampled_row_count} of {count} rows; "
            "additional rows were truncated."
        )
    return f"Returned {count} row{'s' if count != 1 else ''}."


def _scalar_answer(fact: DbAnswerScalarFact) -> str:
    if (
        fact.aggregate_kind == "count"
        and _normalize_answer_label(fact.label) == "count"
    ):
        return f"The count is {_format_answer_value(fact.value)}."
    return f"{fact.label} is {_format_answer_value(fact.value)}."


def _scalar_clause(fact: DbAnswerScalarFact) -> str:
    return f"{fact.label} is {_format_answer_value(fact.value)}"


def _format_answer_value(value: Any) -> str:
    return str(value)


def _answer_facts_from_mapping(value: Any) -> DbAnswerFacts | None:
    if not isinstance(value, dict):
        return None
    scalars = tuple(
        _scalar_fact_from_mapping(item)
        for item in value.get("scalars") or ()
        if isinstance(item, dict)
    )
    primary = (
        _scalar_fact_from_mapping(value.get("primary_scalar"))
        if isinstance(value.get("primary_scalar"), dict)
        else None
    )
    if primary is None and value.get("result_shape") == "scalar" and scalars:
        primary = scalars[0]
    result_shape = str(value.get("result_shape") or "empty")
    if result_shape not in {"empty", "scalar", "record", "table"}:
        result_shape = "empty"
    return DbAnswerFacts(
        result_shape=result_shape,  # type: ignore[arg-type]
        row_count=_safe_int(value.get("row_count"), 0),
        sampled_row_count=_safe_int(value.get("sampled_row_count"), 0),
        columns=tuple(str(item) for item in value.get("columns") or ()),
        scalars=scalars,
        primary_scalar=primary,
        truncated=bool(value.get("truncated")),
        source_evidence_id=(
            str(value["source_evidence_id"])
            if value.get("source_evidence_id") is not None
            else None
        ),
        source_evidence_kind=(
            str(value["source_evidence_kind"])
            if value.get("source_evidence_kind") is not None
            else None
        ),
        diagnostics=(
            dict(value.get("diagnostics"))
            if isinstance(value.get("diagnostics"), dict)
            else {}
        ),
    )


def _scalar_fact_from_mapping(value: Any) -> DbAnswerScalarFact:
    payload = value if isinstance(value, dict) else {}
    return DbAnswerScalarFact(
        label=str(payload.get("label") or "value"),
        value=payload.get("value"),
        aggregate_kind=(
            str(payload["aggregate_kind"])
            if payload.get("aggregate_kind") is not None
            else None
        ),
        source_evidence_id=(
            str(payload["source_evidence_id"])
            if payload.get("source_evidence_id") is not None
            else None
        ),
        source_evidence_kind=str(payload.get("source_evidence_kind") or "query.result"),
        confidence=str(payload.get("confidence") or "high"),
        redacted=bool(payload.get("redacted")),
    )


def _aggregate_kinds_by_column(
    evidence: tuple[Evidence, ...],
    columns: tuple[str, ...],
) -> dict[str, str]:
    sql = _answer_sql_from_evidence(evidence)
    if not sql:
        return {}
    by_label: dict[str, str] = {}
    positional: list[str | None] = []
    try:
        from .sql_analysis import analyze_sql

        analysis = analyze_sql(sql)
    except Exception:
        positional = _aggregate_kinds_from_sql_text(sql)
    else:
        for item in analysis.select_items:
            kind = _aggregate_kind_from_expression(item.expression_sql)
            if kind is None and item.is_count:
                kind = "count"
            positional.append(kind)
            if item.alias and kind is not None:
                by_label[_normalize_answer_label(item.alias)] = kind

    if not positional:
        positional = _aggregate_kinds_from_sql_text(sql)

    out = dict(by_label)
    for index, column in enumerate(columns):
        normalized = _normalize_answer_label(column)
        if normalized in out:
            continue
        if index < len(positional) and positional[index] is not None:
            out[normalized] = str(positional[index])
    if len(columns) == 1 and columns:
        normalized = _normalize_answer_label(columns[0])
        if normalized not in out and len(positional) == 1 and positional[0] is not None:
            out[normalized] = str(positional[0])
    return out


def _answer_sql_from_evidence(evidence: tuple[Evidence, ...]) -> str:
    for kind in (
        "query.result",
        "sql.validation",
        "query.plan.validation",
        "query.plan.proposal",
    ):
        for item in reversed(evidence):
            if item.kind != kind or not item.accepted:
                continue
            sql = _sql_from_payload(item.payload)
            if sql:
                return sql
    return ""


def _sql_from_payload(payload: dict[str, Any]) -> str:
    for key in ("sql", "accepted_sql", "selected_sql"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    structured = payload.get("structured_plan")
    if isinstance(structured, dict):
        for key in ("selected_sql", "sql"):
            value = structured.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            value = candidate.get("sql")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _aggregate_kinds_from_sql_text(sql: str) -> list[str | None]:
    return [
        match.group(1).lower()
        for match in re.finditer(
            r"\b(count|sum|avg|min|max)\s*\(",
            sql or "",
            flags=re.IGNORECASE,
        )
    ]


def _aggregate_kind_from_expression(expression: str) -> str | None:
    match = re.search(
        r"\b(count|sum|avg|min|max)\s*\(",
        expression or "",
        flags=re.IGNORECASE,
    )
    return match.group(1).lower() if match else None


def _normalize_answer_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def _has_monitor_evidence(evidence: tuple[Evidence, ...]) -> bool:
    return any(item.kind.startswith("monitor.") for item in evidence if item.accepted)


def _monitor_answer(evidence: tuple[Evidence, ...]) -> str:
    accepted = tuple(item for item in evidence if item.accepted)
    for kind in (
        "monitor.definition",
        "monitor.deleted",
        "monitor.disabled",
        "monitor.paused",
        "monitor.resumed",
        "monitor.state_update",
        "monitor.listing",
        "monitor.inspection",
        "monitor.run_summary",
        "monitor.approval_state",
        "monitor.approval_resolution",
        "monitor.proposal",
    ):
        item = next(
            (candidate for candidate in reversed(accepted) if candidate.kind == kind),
            None,
        )
        if item is not None:
            return _monitor_answer_from_evidence(item)
    return "The monitor operation completed with verified evidence."


def _monitor_answer_from_evidence(evidence: Evidence) -> str:
    payload = evidence.payload
    if evidence.kind == "monitor.listing":
        monitors = [
            item for item in payload.get("monitors") or () if isinstance(item, dict)
        ]
        if not monitors:
            return "No monitors are currently defined."
        lines = ["Monitors:"]
        for monitor in monitors:
            lines.append(
                f"- {monitor.get('id')}: {monitor.get('name')} [{monitor.get('status')}]"
            )
        return "\n".join(lines)
    if evidence.kind in {"monitor.inspection", "monitor.run_summary"}:
        inspection = payload.get("inspection")
        if not isinstance(inspection, dict):
            return "The monitor could not be inspected from the accepted evidence."
        monitor = dict(inspection.get("monitor") or {})
        monitor_id = monitor.get("id")
        name = monitor.get("name") or monitor_id
        if evidence.kind == "monitor.run_summary":
            runs = inspection.get("runs") or ()
            if not runs:
                return f"Monitor {name} ({monitor_id}) has no recorded runs yet."
            last_run = dict(runs[-1])
            return (
                f"Monitor {name} ({monitor_id}) last run "
                f"{last_run.get('status')}; operation {last_run.get('operation_id')}."
            )
        schedule = monitor.get("schedule")
        schedule_text = ""
        if isinstance(schedule, dict):
            schedule_text = _monitor_schedule_text(schedule)
        suffix = f", schedule {schedule_text}" if schedule_text else ""
        return f"Monitor {name} ({monitor_id}) is {monitor.get('status')}{suffix}."
    if evidence.kind == "monitor.approval_state":
        approvals = [
            item for item in payload.get("approvals") or () if isinstance(item, dict)
        ]
        if not approvals:
            return "No pending monitor approvals were found."
        lines = ["Pending monitor approvals:"]
        for approval in approvals:
            context = dict(approval.get("context") or {})
            monitor_id = context.get("monitor_id") or payload.get("monitor_id")
            lines.append(
                f"- {approval.get('approval_id')}: {approval.get('status')}"
                + (f" for {monitor_id}" if monitor_id else "")
            )
        return "\n".join(lines)
    if evidence.kind == "monitor.approval_resolution":
        status = str(payload.get("status") or "")
        if status == "not_found":
            return "No matching pending monitor approval was found."
        if status == "ambiguous":
            return "Multiple pending monitor approvals matched; specify an approval id."
        action = str(payload.get("approval_action") or "approve").lower()
        verb = {
            "approve": "Approved",
            "reject": "Rejected",
            "cancel": "Cancelled",
        }.get(action, "Updated")
        return (
            f"{verb} monitor approval {payload.get('approval_id')}; "
            f"approval is {payload.get('approval_status')}."
        )
    if evidence.kind == "monitor.definition":
        monitor = dict(payload.get("monitor") or {})
        schedule = monitor.get("schedule")
        suffix = ""
        if isinstance(schedule, dict):
            schedule_text = _monitor_schedule_text(schedule)
            suffix = f" on {schedule_text}" if schedule_text else ""
        return f"Created monitor {monitor.get('name')} ({monitor.get('id')}){suffix}."
    if evidence.kind in {
        "monitor.deleted",
        "monitor.disabled",
        "monitor.paused",
        "monitor.resumed",
        "monitor.state_update",
    }:
        action = str(payload.get("action") or evidence.kind.removeprefix("monitor."))
        monitor = dict(
            payload.get("monitor")
            or payload.get("after")
            or payload.get("before")
            or {}
        )
        monitor_id = monitor.get("id") or payload.get("monitor_id")
        name = monitor.get("name") or monitor_id
        verb = {
            "delete": "Deleted",
            "deleted": "Deleted",
            "disable": "Disabled",
            "disabled": "Disabled",
            "pause": "Paused",
            "paused": "Paused",
            "resume": "Resumed",
            "resumed": "Resumed",
            "update": "Updated",
            "state_update": "Updated",
        }.get(action, "Updated")
        return f"{verb} monitor {name} ({monitor_id})."
    if evidence.kind == "monitor.proposal":
        validation = dict(payload.get("validation") or {})
        if validation.get("accepted") is False:
            missing = ", ".join(
                str(item) for item in validation.get("missing_capabilities") or ()
            )
            if missing:
                return (
                    "Monitor was not created because required capabilities are missing: "
                    f"{missing}."
                )
            return "Monitor was not created because its definition did not pass validation."
        return f"Prepared monitor proposal {payload.get('name')} ({payload.get('monitor_id')})."
    return "The monitor operation completed with verified evidence."


def _monitor_schedule_text(schedule: dict[str, Any]) -> str:
    interval_seconds = schedule.get("interval_seconds") or schedule.get("every_seconds")
    if isinstance(interval_seconds, (int, float)) and interval_seconds > 0:
        if interval_seconds % 3600 == 0:
            hours = int(interval_seconds // 3600)
            return "hourly" if hours == 1 else f"every {hours} hours"
        if interval_seconds % 60 == 0:
            minutes = int(interval_seconds // 60)
            return "every minute" if minutes == 1 else f"every {minutes} minutes"
        seconds = int(interval_seconds)
        return "every second" if seconds == 1 else f"every {seconds} seconds"
    expression = str(schedule.get("expression") or "").strip()
    return expression


def _tables_from_schema_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    nested_schema = payload.get("schema")
    if isinstance(nested_schema, dict):
        nested_tables = _tables_from_schema_payload(nested_schema)
        if nested_tables:
            return nested_tables
    if payload.get("tables"):
        return [
            table
            for table in payload.get("tables", []) or []
            if isinstance(table, dict)
        ]
    asset = payload.get("asset")
    if isinstance(asset, dict):
        return [
            {
                "name": asset.get("name") or payload.get("table_name"),
                "columns": payload.get("fields") or payload.get("columns") or [],
            }
        ]
    table = payload.get("table")
    if isinstance(table, dict):
        return [
            {
                "name": table.get("name") or payload.get("table_name"),
                "columns": payload.get("fields") or payload.get("columns") or [],
            }
        ]
    return []


def _columns_from_schema_table(table: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        column
        for column in (table.get("columns") or table.get("fields") or [])
        if isinstance(column, dict)
    ]


@dataclass(frozen=True)
class DbAnswerCitation:
    """Citation to accepted runtime evidence used by final synthesis."""

    id: str
    kind: str
    purpose: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "kind": self.kind, "purpose": self.purpose}


@dataclass(frozen=True)
class DbAnswerSynthesisPayload:
    """Typed payload for `answer.synthesis` evidence."""

    answer: str
    reasoning_summary: str
    cited_evidence_refs: tuple[DbAnswerCitation, ...]
    assumptions: tuple[str, ...]
    limitations: tuple[str, ...]
    warnings: tuple[str, ...]
    follow_up_questions: tuple[str, ...]
    sufficiency: str
    confidence: float
    truncation: dict[str, bool]
    grounding: dict[str, bool]
    diagnostics: dict[str, Any]
    answer_facts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "reasoning_summary": self.reasoning_summary,
            "cited_evidence_refs": [
                citation.to_dict() for citation in self.cited_evidence_refs
            ],
            "assumptions": list(self.assumptions),
            "limitations": list(self.limitations),
            "warnings": list(self.warnings),
            "follow_up_questions": list(self.follow_up_questions),
            "sufficiency": self.sufficiency,
            "confidence": self.confidence,
            "truncation": dict(self.truncation),
            "grounding": dict(self.grounding),
            "answer_facts": dict(self.answer_facts),
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class DbSynthesisContext:
    """Bounded accepted-evidence context supplied to final synthesis."""

    payload: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DbAnswerSynthesisExecutor:
    """Executor for runtime-owned `db.answer.synthesize` tasks."""

    runtime: Any
    id: str = "db.answer.synthesize.runtime"
    owner: str = "db_runtime"
    capability_ids: frozenset[str] = frozenset({"db.answer.synthesize"})

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: dict[str, Any],
    ) -> list[Evidence]:
        runtime = _runtime_from_boundary(self.runtime)
        request = runtime._db_request_from_operation(operation)
        intent = runtime._db_intent_from_operation(operation)
        contract = runtime._db_contract_from_context(operation)
        dependency_evidence = await _accepted_dependency_evidence(
            runtime, task, operation
        )
        verification_evidence = next(
            (
                item
                for item in dependency_evidence
                if item.kind == "verification.result"
            ),
            None,
        )
        if verification_evidence is None:
            raise RuntimeError("verification.result evidence is required")
        verification = verification_from_evidence(verification_evidence)
        synthesis_context = build_synthesis_context(
            request=request,
            intent=intent,
            contract=contract,
            evidence=dependency_evidence,
            row_budget=int(task.input.get("row_budget") or _DEFAULT_CONTEXT_ROW_BUDGET),
            char_budget=int(
                task.input.get("char_budget") or _DEFAULT_CONTEXT_CHAR_BUDGET
            ),
        )
        payload: DbAnswerSynthesisPayload | None = None
        fallback_reason: str | None = None
        if runtime.db_llm_service.available:
            try:
                response = await runtime.db_llm_service.generate_synthesis_json(
                    _synthesis_messages(synthesis_context.payload)
                )
                parsed = parse_synthesis_json(response.content)
                payload = validate_synthesis_payload(
                    parsed,
                    dependency_evidence=dependency_evidence,
                    context_metadata=synthesis_context.metadata,
                    llm_diagnostics=response.diagnostics,
                )
            except Exception as exc:
                fallback_reason = f"{type(exc).__name__}:{exc}"
        else:
            fallback_reason = "db_llm_service_unavailable"

        if payload is None:
            payload = deterministic_synthesis_payload(
                request=request,
                intent=intent,
                contract=contract,
                evidence=dependency_evidence,
                verification=verification,
                synthesizer=runtime.synthesizer,
                context_metadata=synthesis_context.metadata,
                fallback_reason=fallback_reason or "llm_synthesis_invalid",
            )
        payload = _apply_schema_db_memory_annotation(
            payload,
            intent=intent,
            evidence=dependency_evidence,
        )

        return [
            Evidence(
                kind="answer.synthesis",
                owner=self.owner,
                operation_id=operation.id,
                task_id=task.id,
                accepted=True,
                payload=payload.to_dict(),
                metadata={
                    "mode": payload.diagnostics.get("mode"),
                    "sufficiency": payload.sufficiency,
                    "cited_evidence_refs": [
                        citation.id for citation in payload.cited_evidence_refs
                    ],
                },
            )
        ]


def verification_from_evidence(evidence: Evidence) -> DbVerificationResult:
    """Rehydrate verifier result from compact `verification.result` evidence."""
    payload = evidence.payload
    evidence_refs = tuple(
        dict(item)
        for item in payload.get("evidence_details", ())
        if isinstance(item, dict)
    )
    if not evidence_refs:
        evidence_refs = tuple(
            {"id": item, "kind": None, "owner": None, "task_id": None}
            for item in payload.get("evidence_refs", ())
        )
    return DbVerificationResult(
        passed=bool(payload.get("passed")),
        missing_evidence=tuple(payload.get("missing_evidence") or ()),
        warnings=tuple(str(item) for item in payload.get("warnings") or ()),
        diagnostics=dict(payload.get("diagnostics") or {}),
        evidence_refs=evidence_refs,
    )


def build_synthesis_context(
    *,
    request: DbRequest,
    intent: DbIntent,
    contract: DbOperationContract,
    evidence: tuple[Evidence, ...],
    row_budget: int = _DEFAULT_CONTEXT_ROW_BUDGET,
    char_budget: int = _DEFAULT_CONTEXT_CHAR_BUDGET,
) -> DbSynthesisContext:
    """Build bounded synthesis context from accepted evidence records only."""
    accepted = tuple(item for item in evidence if item.accepted)
    evidence_summaries = [_evidence_summary(item) for item in accepted]
    query_result = next(
        (item for item in accepted if item.kind == "query.result"), None
    )
    rows, row_metadata = _bounded_rows(query_result, row_budget=row_budget)
    answer_facts = derive_answer_facts(
        request=request,
        intent=intent,
        contract=contract,
        evidence=accepted,
    )
    schema_scope = (
        _schema_answer_scope(request, contract, accepted)
        if intent.kind is DbIntentKind.SCHEMA_QUERY
        else None
    )
    semantics = _catalog_semantics(accepted, schema_scope=schema_scope)
    payload = {
        "prompt": request.prompt,
        "intent_kind": intent.kind.value,
        "operation_type": contract.operation_type,
        "answer_facts": answer_facts.to_dict(),
        "evidence": evidence_summaries,
        "query_result": {
            "evidence_id": query_result.id if query_result else None,
            "columns": row_metadata["columns"],
            "column_types": row_metadata["column_types"],
            "rows": rows,
            "total_rows": row_metadata["total_rows"],
            "sampled_rows": row_metadata["sampled_rows"],
            "truncated": row_metadata["rows_truncated"],
        },
        "planning": _planning_summary(accepted),
        "semantics": semantics,
        "verification": _verification_summary(accepted),
    }
    if schema_scope is not None:
        payload["schema_answer_scope"] = schema_scope.to_dict()
    required_caveats = _required_caveats(payload, row_metadata)
    payload["required_caveats"] = list(required_caveats)
    rendered = json.dumps(payload, sort_keys=True, default=str)
    context_chars_truncated = len(rendered) > char_budget
    diagnostic_caveats: tuple[str, ...] = ()
    if context_chars_truncated:
        payload["rendered_context"] = rendered[:char_budget]
        diagnostic_caveats = ("synthesis_context_truncated",)
        payload["diagnostic_caveats"] = list(diagnostic_caveats)
    else:
        payload["rendered_context"] = rendered
    truncation = {
        "rows_truncated": bool(row_metadata["rows_truncated"]),
        "fields_truncated": bool(row_metadata["fields_truncated"]),
        "context_chars_truncated": context_chars_truncated,
    }
    metadata = {
        "dependency_evidence_refs": [
            {"id": item.id, "kind": item.kind, "operation_id": item.operation_id}
            for item in accepted
            if item.id
        ],
        "accepted_evidence_ids": [item.id for item in accepted if item.id],
        "required_caveats": list(required_caveats),
        "diagnostic_caveats": list(diagnostic_caveats),
        "answer_facts": answer_facts.to_dict(),
        "truncation": truncation,
        "redaction": {
            "values_redacted": bool(row_metadata["redacted_value_count"]),
            "redacted_value_count": row_metadata["redacted_value_count"],
        },
        "context_budget": {
            "row_budget": row_budget,
            "char_budget": char_budget,
            "field_char_budget": _DEFAULT_FIELD_CHAR_BUDGET,
        },
    }
    if schema_scope is not None:
        metadata["schema_answer_scope"] = schema_scope.to_dict()
    payload["truncation"] = truncation
    payload["redaction"] = metadata["redaction"]
    return DbSynthesisContext(payload=payload, metadata=metadata)


def deterministic_synthesis_payload(
    *,
    request: DbRequest,
    intent: DbIntent,
    contract: DbOperationContract,
    evidence: tuple[Evidence, ...],
    verification: DbVerificationResult,
    synthesizer: DbSynthesizer | None = None,
    context_metadata: dict[str, Any],
    fallback_reason: str | None,
) -> DbAnswerSynthesisPayload:
    """Build deterministic fallback through the same answer evidence path."""
    result = (synthesizer or DbSynthesizer()).synthesize(
        request, intent, contract, evidence, verification
    )
    answer_facts = _answer_facts_from_mapping(context_metadata.get("answer_facts"))
    answer = (
        _data_answer_from_facts(answer_facts)
        if answer_facts is not None
        and intent.kind
        in {DbIntentKind.DATA_QUERY, DbIntentKind.CATALOG_ASSISTED_DATA_QUERY}
        else result.answer
    )
    citations = tuple(
        DbAnswerCitation(
            id=str(item["id"]),
            kind=str(item.get("kind") or "unknown"),
            purpose="verified runtime evidence",
        )
        for item in _accepted_citation_refs(evidence, verification)
        if item.get("id")
    )
    caveats = tuple(str(item) for item in context_metadata.get("required_caveats", ()))
    truncation = dict(context_metadata.get("truncation") or {})
    answer_critical_truncated = bool(
        truncation.get("rows_truncated") or truncation.get("fields_truncated")
    )
    sufficiency = "partial" if answer_critical_truncated else "answered"
    return DbAnswerSynthesisPayload(
        answer=answer,
        reasoning_summary="Deterministic summary generated from verified evidence.",
        cited_evidence_refs=citations,
        assumptions=(),
        limitations=caveats,
        warnings=tuple((*result.warnings, *caveats)),
        follow_up_questions=(),
        sufficiency=sufficiency,
        confidence=0.8 if verification.passed else 0.0,
        truncation={
            "rows_truncated": bool(truncation.get("rows_truncated")),
            "fields_truncated": bool(truncation.get("fields_truncated")),
            "context_chars_truncated": bool(truncation.get("context_chars_truncated")),
        },
        grounding={"all_claims_from_evidence": True},
        answer_facts=answer_facts.to_dict() if answer_facts is not None else {},
        diagnostics={
            "mode": "deterministic_fallback",
            "model": "deterministic",
            "provider": "daita.db",
            "latency_ms": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost": 0.0,
            "llm_calls": 0,
            "fallback_reason": fallback_reason,
            "evidence_refs": [citation.id for citation in citations],
            "context": context_metadata,
            "sufficiency": sufficiency,
        },
    )


def parse_synthesis_json(content: str) -> dict[str, Any]:
    raw = _strip_json_fence(content)
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("synthesis_json_not_object")
    return parsed


def validate_synthesis_payload(
    parsed: dict[str, Any],
    *,
    dependency_evidence: tuple[Evidence, ...],
    context_metadata: dict[str, Any],
    llm_diagnostics: dict[str, Any],
) -> DbAnswerSynthesisPayload:
    """Validate strict LLM synthesis before accepting it as evidence."""
    answer = str(parsed.get("answer") or "").strip()
    if not answer:
        raise ValueError("synthesis_answer_empty")
    if _requests_db_work(parsed):
        raise ValueError("synthesis_requests_db_work")
    confidence = parsed.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        raise ValueError("synthesis_confidence_out_of_bounds")
    sufficiency = str(parsed.get("sufficiency") or "")
    if sufficiency not in _ALLOWED_SUFFICIENCY:
        raise ValueError("synthesis_sufficiency_invalid")
    grounding = dict(parsed.get("grounding") or {})
    if grounding.get("all_claims_from_evidence") is not True:
        raise ValueError("synthesis_grounding_missing")

    accepted_by_id = {
        item.id: item for item in dependency_evidence if item.accepted and item.id
    }
    citations = _parse_citations(parsed.get("cited_evidence_refs"), accepted_by_id)
    if not citations:
        raise ValueError("synthesis_citations_missing")
    _validate_required_citations(citations, dependency_evidence)
    _validate_caveats_preserved(parsed, context_metadata)
    _validate_relationship_claims(answer, citations, accepted_by_id)
    answer_facts = _answer_facts_from_mapping(context_metadata.get("answer_facts"))
    _validate_answer_facts_preserved(answer, answer_facts)
    truncation = dict(parsed.get("truncation") or {})
    context_truncation = dict(context_metadata.get("truncation") or {})
    normalized_truncation = {
        "rows_truncated": bool(
            truncation.get("rows_truncated") or context_truncation.get("rows_truncated")
        ),
        "fields_truncated": bool(
            truncation.get("fields_truncated")
            or context_truncation.get("fields_truncated")
        ),
        "context_chars_truncated": bool(
            truncation.get("context_chars_truncated")
            or context_truncation.get("context_chars_truncated")
        ),
    }
    sufficiency = _normalized_sufficiency(
        sufficiency,
        answer_facts=answer_facts,
        truncation=normalized_truncation,
    )
    diagnostics = _diagnostics_from_llm(llm_diagnostics)
    diagnostics.update(
        {
            "mode": "llm",
            "fallback_reason": None,
            "evidence_refs": [citation.id for citation in citations],
            "context": context_metadata,
            "sufficiency": sufficiency,
        }
    )
    return DbAnswerSynthesisPayload(
        answer=answer,
        reasoning_summary=str(parsed.get("reasoning_summary") or "").strip(),
        cited_evidence_refs=tuple(citations),
        assumptions=_string_tuple(parsed.get("assumptions")),
        limitations=_string_tuple(parsed.get("limitations")),
        warnings=_string_tuple(parsed.get("warnings")),
        follow_up_questions=_string_tuple(parsed.get("follow_up_questions")),
        sufficiency=sufficiency,
        confidence=float(confidence),
        truncation=normalized_truncation,
        grounding={"all_claims_from_evidence": True},
        answer_facts=answer_facts.to_dict() if answer_facts is not None else {},
        diagnostics=diagnostics,
    )


async def _accepted_dependency_evidence(
    runtime: Any,
    task: Task,
    operation: Operation,
) -> tuple[Evidence, ...]:
    evidence: list[Evidence] = []
    for dependency in task.dependencies:
        if dependency.kind.value != "evidence":
            continue
        item = await runtime.tasks.accepted_evidence_for_dependency(
            operation.id, dependency
        )
        if item is not None and item.accepted and item.operation_id == operation.id:
            evidence.append(item)
    return tuple(evidence)


def _synthesis_messages(context_payload: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You synthesize final database answers inside a governed runtime. "
                "Use only the provided accepted evidence. Return strict JSON only. "
                "The sufficiency field must be exactly one string from: "
                "answered, partial, needs_clarification, insufficient_evidence. "
                "If verification.result evidence is present, cite it. If "
                "query.result evidence is present, cite it. "
                "Never request SQL execution, tool calls, connector access, or any "
                "new database work."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "context": context_payload,
                    "schema": {
                        "answer": "string",
                        "reasoning_summary": "string",
                        "cited_evidence_refs": [
                            {
                                "id": "accepted evidence id",
                                "kind": "evidence kind",
                                "purpose": "why cited",
                            }
                        ],
                        "assumptions": ["string"],
                        "limitations": ["string"],
                        "warnings": ["string"],
                        "follow_up_questions": ["string"],
                        "sufficiency": (
                            "one of: answered, partial, needs_clarification, "
                            "insufficient_evidence"
                        ),
                        "confidence": "number from 0 to 1",
                        "truncation": {
                            "rows_truncated": "boolean",
                            "fields_truncated": "boolean",
                            "context_chars_truncated": "boolean",
                        },
                        "grounding": {"all_claims_from_evidence": True},
                        "answer_facts": (
                            "object copied from context.answer_facts; the answer "
                            "must preserve scalar values from these facts"
                        ),
                    },
                },
                sort_keys=True,
                default=str,
            ),
        },
    ]


def _runtime_from_boundary(boundary: Any) -> Any:
    runtime = getattr(boundary, "runtime", None)
    return runtime if runtime is not None else boundary


def _accepted_citation_refs(
    evidence: tuple[Evidence, ...],
    verification: DbVerificationResult,
) -> tuple[dict[str, str | None], ...]:
    accepted_by_id = {item.id: item for item in evidence if item.accepted and item.id}
    refs = [
        item for item in verification.evidence_refs if item.get("id") in accepted_by_id
    ]
    if refs:
        return tuple(refs)
    return tuple(
        {
            "id": item.id,
            "kind": item.kind,
            "owner": item.owner,
            "task_id": item.task_id,
        }
        for item in evidence
        if item.accepted and item.id
    )


def _evidence_summary(evidence: Evidence) -> dict[str, Any]:
    return {
        "id": evidence.id,
        "kind": evidence.kind,
        "owner": evidence.owner,
        "operation_id": evidence.operation_id,
        "task_id": evidence.task_id,
        "payload_fingerprint": evidence.metadata.get("payload_fingerprint"),
        "payload_keys": sorted(str(key) for key in evidence.payload),
        "accepted": evidence.accepted,
    }


def _bounded_rows(
    query_result: Evidence | None,
    *,
    row_budget: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if query_result is None:
        return [], {
            "columns": [],
            "column_types": {},
            "total_rows": 0,
            "sampled_rows": 0,
            "rows_truncated": False,
            "fields_truncated": False,
            "redacted_value_count": 0,
        }
    raw_rows = query_result.payload.get("rows") or []
    rows = raw_rows if isinstance(raw_rows, list) else []
    bounded: list[dict[str, Any]] = []
    fields_truncated = False
    redacted_count = 0
    for row in rows[: max(row_budget, 0)]:
        if not isinstance(row, dict):
            row = {"value": row}
        redacted_row: dict[str, Any] = {}
        for key, value in row.items():
            redacted, redacted_value, truncated = _redact_value(str(key), value)
            redacted_row[str(key)] = redacted_value
            redacted_count += int(redacted)
            fields_truncated = fields_truncated or truncated
        bounded.append(redacted_row)
    columns = sorted({column for row in bounded for column in row})
    column_types = {
        column: _first_type_for_column(bounded, column) for column in columns
    }
    total_rows = int(query_result.payload.get("total_rows") or len(rows))
    rows_truncated = bool(query_result.payload.get("truncated")) or len(rows) > len(
        bounded
    )
    return bounded, {
        "columns": columns,
        "column_types": column_types,
        "total_rows": total_rows,
        "sampled_rows": len(bounded),
        "rows_truncated": rows_truncated,
        "fields_truncated": fields_truncated,
        "redacted_value_count": redacted_count,
    }


def _redact_value(key: str, value: Any) -> tuple[bool, Any, bool]:
    if _SENSITIVE_FIELD_RE.search(key):
        return True, "[REDACTED]", False
    if isinstance(value, str):
        redacted = False
        rendered = value
        if _EMAIL_RE.search(rendered) or _PHONE_RE.fullmatch(rendered.strip()):
            rendered = "[REDACTED]"
            redacted = True
        truncated = len(rendered) > _DEFAULT_FIELD_CHAR_BUDGET
        if truncated:
            rendered = rendered[:_DEFAULT_FIELD_CHAR_BUDGET] + "...[TRUNCATED]"
        return redacted, rendered, truncated
    return False, value, False


def _first_type_for_column(rows: list[dict[str, Any]], column: str) -> str:
    for row in rows:
        value = row.get(column)
        if value is not None:
            return type(value).__name__
    return "null"


def _catalog_semantics(
    evidence: tuple[Evidence, ...],
    *,
    schema_scope: SchemaAnswerScope | None = None,
) -> dict[str, Any]:
    planning = next(
        (item for item in evidence if item.kind == "planning.context"), None
    )
    if planning is None:
        planning = next(
            (
                item
                for item in evidence
                if item.kind.startswith("catalog.")
                or item.kind == "schema.asset_profile"
            ),
            None,
        )
    payload = planning.payload if planning is not None else {}
    schema = (
        payload.get("schema") if isinstance(payload.get("schema"), dict) else payload
    )
    tables = []
    source_tables = (
        list(schema_scope.selected_tables)
        if schema_scope is not None
        else _tables_from_schema_payload(schema)
    )
    for table in source_tables:
        tables.append(
            {
                "name": table.get("name"),
                "columns": [
                    {
                        "name": column.get("name") or column.get("column_name"),
                        "data_type": column.get("data_type") or column.get("type"),
                    }
                    for column in _columns_from_schema_table(table)
                    if column.get("name") or column.get("column_name")
                ],
                "metadata": dict(table.get("metadata") or {}),
            }
        )
    relationships = list(schema.get("foreign_keys", []) or [])[:50]
    if schema_scope is not None and schema_scope.mode == "asset":
        selected_names = {
            str(table.get("name"))
            for table in schema_scope.selected_tables
            if table.get("name")
        }
        relationships = [
            relationship
            for relationship in relationships
            if _relationship_mentions_table(relationship, selected_names)
        ]
    return {
        "evidence_id": planning.id if planning is not None else None,
        "tables": tables,
        "relationships": relationships,
    }


def _relationship_mentions_table(relationship: Any, selected_names: set[str]) -> bool:
    if not isinstance(relationship, dict):
        return False
    values = {
        relationship.get("source_table"),
        relationship.get("target_table"),
        relationship.get("source_asset"),
        relationship.get("target_asset"),
        relationship.get("from_table"),
        relationship.get("to_table"),
    }
    return bool(selected_names & {str(value) for value in values if value})


def _planning_summary(evidence: tuple[Evidence, ...]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for kind in (
        "planning.context",
        "query.plan.proposal",
        "query.plan.validation",
        "sql.validation",
    ):
        item = next(
            (candidate for candidate in evidence if candidate.kind == kind), None
        )
        if item is None:
            continue
        summary[kind] = {
            "evidence_id": item.id,
            "payload_fingerprint": item.metadata.get("payload_fingerprint"),
            "payload": _compact_payload(item.payload),
        }
    return summary


def _verification_summary(evidence: tuple[Evidence, ...]) -> dict[str, Any]:
    item = next(
        (
            candidate
            for candidate in evidence
            if candidate.kind == "verification.result"
        ),
        None,
    )
    return dict(item.payload) if item is not None else {}


def _compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "valid",
        "warnings",
        "assumptions",
        "strategy",
        "sql_fingerprint",
        "schema_fingerprint",
        "plan_fingerprint",
        "tables",
        "columns",
        "statement_type",
        "is_read",
        "accepted_sql",
    }
    return {key: value for key, value in payload.items() if key in allowed}


def _required_caveats(
    payload: dict[str, Any], row_metadata: dict[str, Any]
) -> tuple[str, ...]:
    caveats: list[str] = []
    if row_metadata["rows_truncated"]:
        caveats.append("query_result_truncated")
    if row_metadata["fields_truncated"]:
        caveats.append("fields_truncated")
    if row_metadata["redacted_value_count"]:
        caveats.append("sensitive_values_redacted")
    verification = payload.get("verification") or {}
    for warning in verification.get("warnings") or ():
        caveats.append(str(warning))
    return tuple(dict.fromkeys(caveats))


def _parse_citations(
    value: Any, accepted_by_id: dict[str | None, Evidence]
) -> list[DbAnswerCitation]:
    if not isinstance(value, list):
        raise ValueError("synthesis_citations_not_list")
    citations: list[DbAnswerCitation] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("synthesis_citation_not_object")
        evidence_id = str(item.get("id") or "")
        if evidence_id not in accepted_by_id:
            raise ValueError(f"synthesis_unknown_citation:{evidence_id}")
        evidence = accepted_by_id[evidence_id]
        kind = str(item.get("kind") or evidence.kind)
        if kind != evidence.kind:
            raise ValueError(f"synthesis_citation_kind_mismatch:{evidence_id}")
        citations.append(
            DbAnswerCitation(
                id=evidence_id,
                kind=kind,
                purpose=str(item.get("purpose") or "supporting evidence"),
            )
        )
    return citations


def _validate_required_citations(
    citations: list[DbAnswerCitation],
    dependency_evidence: tuple[Evidence, ...],
) -> None:
    cited_kinds = {citation.kind for citation in citations}
    available_kinds = {item.kind for item in dependency_evidence if item.accepted}
    if "query.result" in available_kinds and "query.result" not in cited_kinds:
        raise ValueError("synthesis_query_result_citation_missing")
    if (
        "verification.result" in available_kinds
        and "verification.result" not in cited_kinds
    ):
        raise ValueError("synthesis_verification_citation_missing")


def _validate_caveats_preserved(
    parsed: dict[str, Any], context_metadata: dict[str, Any]
) -> None:
    required = tuple(context_metadata.get("required_caveats") or ())
    if not required:
        return
    provided = " ".join(
        str(item)
        for item in (
            *(parsed.get("limitations") or ()),
            *(parsed.get("warnings") or ()),
            *(parsed.get("assumptions") or ()),
        )
    )
    for caveat in required:
        if str(caveat) not in provided:
            raise ValueError(f"synthesis_dropped_caveat:{caveat}")


def _validate_answer_facts_preserved(
    answer: str,
    answer_facts: DbAnswerFacts | None,
) -> None:
    if answer_facts is None:
        return
    required = (
        (answer_facts.primary_scalar,)
        if answer_facts.primary_scalar is not None
        else tuple(
            fact
            for fact in answer_facts.scalars
            if fact.confidence == "high" and not fact.redacted
        )
    )
    for fact in required:
        if fact is None or fact.redacted:
            continue
        if not _answer_contains_value(answer, fact.value):
            raise ValueError(f"synthesis_missing_answer_fact:{fact.label}")


def _answer_contains_value(answer: str, value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return str(value).lower() in answer.lower()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _answer_contains_numeric_value(answer, float(value))
    rendered = str(value).strip()
    if not rendered:
        return True
    numeric = _numeric_value(rendered)
    if numeric is not None:
        return _answer_contains_numeric_value(answer, numeric)
    return rendered.lower() in answer.lower()


def _answer_contains_numeric_value(answer: str, expected: float) -> bool:
    for match in re.finditer(r"[-+]?\d[\d,]*(?:\.\d+)?", answer):
        numeric = _numeric_value(match.group(0))
        if numeric is not None and abs(numeric - expected) < 1e-9:
            return True
    return False


def _numeric_value(value: str) -> float | None:
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def _normalized_sufficiency(
    sufficiency: str,
    *,
    answer_facts: DbAnswerFacts | None,
    truncation: dict[str, bool],
) -> str:
    if sufficiency != "partial":
        return sufficiency
    if answer_facts is None or answer_facts.primary_scalar is None:
        return sufficiency
    if truncation.get("rows_truncated") or truncation.get("fields_truncated"):
        return sufficiency
    if truncation.get("context_chars_truncated"):
        return "answered"
    return sufficiency


def _validate_relationship_claims(
    answer: str,
    citations: list[DbAnswerCitation],
    accepted_by_id: dict[str | None, Evidence],
) -> None:
    lowered = answer.lower()
    if not any(
        term in lowered
        for term in ("relationship", "foreign key", "joined", "join path", "linked")
    ):
        return
    cited_kinds = {accepted_by_id[citation.id].kind for citation in citations}
    if not any(
        kind == "planning.context"
        or kind == "schema.asset_profile"
        or kind.startswith("catalog.")
        for kind in cited_kinds
    ):
        raise ValueError("synthesis_ungrounded_relationship_claim")


def _requests_db_work(parsed: dict[str, Any]) -> bool:
    text = json.dumps(parsed, sort_keys=True, default=str).lower()
    forbidden = (
        "execute sql",
        "run sql",
        "run a query",
        "query the database",
        "call a tool",
        "tool call",
        "connector access",
        "db.sql.",
    )
    return any(term in text for term in forbidden)


def _diagnostics_from_llm(diagnostics: dict[str, Any]) -> dict[str, Any]:
    tokens = (
        diagnostics.get("tokens") if isinstance(diagnostics.get("tokens"), dict) else {}
    )
    input_tokens = db_optional_int(
        diagnostics.get("input_tokens")
        if diagnostics.get("input_tokens") is not None
        else tokens.get("input_tokens", tokens.get("prompt_tokens"))
    )
    output_tokens = db_optional_int(
        diagnostics.get("output_tokens")
        if diagnostics.get("output_tokens") is not None
        else tokens.get("output_tokens", tokens.get("completion_tokens"))
    )
    total_tokens = db_optional_int(
        diagnostics.get("total_tokens")
        if diagnostics.get("total_tokens") is not None
        else tokens.get("total_tokens")
    )
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    estimated_cost = diagnostics.get("estimated_cost")
    if estimated_cost is None:
        estimated_cost = diagnostics.get("estimated_cost_usd")
    return {
        "model": diagnostics.get("model"),
        "provider": diagnostics.get("provider"),
        "latency_ms": diagnostics.get("latency_ms"),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost": estimated_cost,
        "llm_calls": diagnostics.get("llm_calls") or 1,
    }


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    return (str(value),)


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL)
    return match.group(1).strip() if match else stripped
