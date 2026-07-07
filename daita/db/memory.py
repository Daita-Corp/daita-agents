"""
DB-specific memory semantics for the operation-centric runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import decimal
from inspect import getattr_static, isawaitable
import json
import logging
import re
from typing import Any, Iterable

from daita.db.query_catalog import has_likely_catalog_match
from daita.db.memory_contracts import (
    DB_MEMORY_SEMANTIC_CONTRACT_KEY,
    extract_db_memory_semantic_contract,
    normalize_db_memory_semantic_contract as _normalize_contract,
)

logger = logging.getLogger(__name__)

DB_SEMANTIC_CATEGORY = "db_semantics"
DB_MARKER_CATEGORY = "db_cache_marker"

DB_MEMORY_KINDS = frozenset(
    {
        "unit_convention",
        "metric_definition",
        "business_rule",
        "data_contract_note",
        "schema_interpretation",
        "value_alias",
        "cache_marker",
    }
)
DB_SEMANTIC_MEMORY_KINDS = (
    "unit_convention",
    "metric_definition",
    "business_rule",
    "data_contract_note",
    "schema_interpretation",
    "value_alias",
)
DB_PLANNING_MEMORY_KINDS = frozenset(DB_SEMANTIC_MEMORY_KINDS)
DB_MEMORY_SEMANTIC_QUERY_INTENTS = frozenset(
    {
        "data.query",
        "data.query.catalog_assisted",
        "metric.query",
        "report.generate",
        "quality.check",
        "anomaly.investigate",
    }
)
DB_MEMORY_METADATA_RECALL_INTENTS = frozenset(
    {
        "schema.query",
        "schema.relationship_query",
    }
)

PII_COLUMN_PATTERNS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "email",
    "phone",
    "mobile",
    "ssn",
    "social_security",
    "credit_card",
    "card_number",
    "cvv",
    "pin",
    "dob",
    "date_of_birth",
    "birth_date",
    "address",
    "street",
    "zip",
    "postal",
    "passport",
    "national_id",
)

SENSITIVE_METADATA_KEYS = frozenset(
    (
        *PII_COLUMN_PATTERNS,
        "address",
        "authorization",
        "bearer",
        "credential",
        "credentials",
        "credit_card",
        "email",
        "phone",
        "private_key",
        "ssn",
    )
)
PII_VALUE_PATTERNS = (
    ("email address", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("US SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit card number", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    (
        "phone number",
        re.compile(
            r"(?<!\w)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)"
        ),
    ),
)


@dataclass(frozen=True)
class DBMemoryRecord:
    """Structured DB memory record stored through a MemoryPlugin backend."""

    kind: str
    key: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    importance: float = 0.7

    @property
    def category(self) -> str:
        if self.kind == "cache_marker":
            return DB_MARKER_CATEGORY
        return DB_SEMANTIC_CATEGORY

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "key": self.key,
            "text": self.text,
            "metadata": _json_safe(self.metadata),
            "importance": self.importance,
            "category": self.category,
        }

    def to_memory_content(self) -> str:
        return f"DB memory record:\n{json.dumps(self.to_dict(), sort_keys=True)}"


class DBMemory:
    """Thin adapter that gives MemoryPlugin DB-specific semantics."""

    def __init__(self, plugin: Any):
        self.plugin = plugin

    async def remember(self, record: Any) -> dict[str, Any] | None:
        normalized = normalize_db_memory_record(record)
        return await _upsert_record(self.plugin, normalized)

    async def remember_many(self, records: list[Any]) -> list[dict[str, Any]]:
        results = []
        for record in records:
            result = await self.remember(record)
            if result is not None:
                results.append(result)
        return results

    async def recall(
        self,
        query: str,
        *,
        kinds: list[str] | tuple[str, ...] | set[str] | None = None,
        limit: int = 5,
        score_threshold: float = 0.45,
    ) -> list[dict[str, Any]]:
        return await recall_db_memory_records(
            self.plugin,
            query,
            kinds=kinds,
            limit=limit,
            score_threshold=score_threshold,
        )

    async def has_marker(self, key: str) -> bool:
        return await has_db_memory_marker(self.plugin, key)

    async def mark(self, key: str) -> dict[str, Any] | None:
        return await mark_db_memory(self.plugin, key)


async def write_db_memory_record(plugin: Any, raw: Any) -> dict[str, Any]:
    """Validate and upsert one DB semantic memory record."""
    try:
        record = normalize_db_memory_record(raw)
        pii_error = db_memory_pii_error(
            key=record.key,
            text=record.text,
            metadata=record.metadata,
        )
        if pii_error:
            return {"success": False, "error": pii_error}
        result = await _upsert_record(plugin, record)
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    return {
        "success": True,
        "content": record.to_memory_content(),
        "result": (result or {}).get("stored") or result,
        "kind": record.kind,
        "key": record.key,
        "category": record.category,
        "status": (result or {}).get("status"),
        "updated": (result or {}).get("updated", 0),
        "stored": result,
    }


async def write_db_memory_records(
    plugin: Any, records: list[Any]
) -> list[dict[str, Any]]:
    """Validate and write multiple DB semantic memory records."""
    return [await write_db_memory_record(plugin, record) for record in records]


async def db_memory_record_chunk_ids_by_key(plugin: Any, raw: Any) -> list[str]:
    """Return existing stored chunk IDs for a normalized DB memory key."""
    record = normalize_db_memory_record(raw)
    return await _find_record_chunk_ids_by_key(plugin, record)


async def recall_db_memory_records(
    plugin: Any,
    query: str,
    *,
    kinds: list[str] | tuple[str, ...] | set[str] | None = None,
    limit: int = 5,
    score_threshold: float = 0.45,
) -> list[dict[str, Any]]:
    """Recall DB semantic memory records from the single DB semantics category."""
    results = await _recall_records(
        plugin,
        query,
        limit=max(limit * 3, limit),
        score_threshold=score_threshold,
        category=DB_SEMANTIC_CATEGORY,
        kinds=kinds,
    )
    if kinds:
        allowed = set(kinds)
        results = [
            result for result in results if _record_kind_from_result(result) in allowed
        ]
    return _dedupe_recall_results(results)[:limit]


def db_memory_planning_recall_decision(
    *,
    prompt: str,
    intent_kind: str,
    schema: dict[str, Any],
    memory_config: dict[str, Any],
    matched_schema_terms: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return whether planning should recall DB semantic memory."""
    if not bool(memory_config.get("enabled", False)):
        return {"recall": False, "reason": "memory_disabled"}
    if memory_config.get("recall") == "off":
        return {"recall": False, "reason": "recall_disabled"}
    if int(memory_config.get("limit") or 0) <= 0:
        return {"recall": False, "reason": "limit_zero"}
    if int(memory_config.get("char_budget") or 0) <= 0:
        return {"recall": False, "reason": "char_budget_zero"}
    if intent_kind not in (
        DB_MEMORY_SEMANTIC_QUERY_INTENTS | DB_MEMORY_METADATA_RECALL_INTENTS
    ):
        return {"recall": False, "reason": "intent_not_semantic_query"}
    if _looks_row_level(prompt):
        return {"recall": False, "reason": "row_level_or_pii_prompt"}

    text = str(prompt or "").lower()
    semantic_matches = _matched_terms(text, FALLBACK_SEMANTIC_RECALL_TERMS)
    if semantic_matches:
        return {
            "recall": True,
            "reason": "semantic_prompt",
            "matched_terms": semantic_matches,
            "query": db_memory_planning_recall_query(
                prompt,
                schema,
                intent_kind,
                matched_schema_terms=matched_schema_terms,
            ),
        }
    if _looks_direct_query(text) and _prompt_matches_schema(prompt, schema):
        return {"recall": False, "reason": "direct_schema_matched_query"}
    return {
        "recall": True,
        "reason": "semantic_fallback",
        "matched_terms": [],
        "query": db_memory_planning_recall_query(
            prompt,
            schema,
            intent_kind,
            matched_schema_terms=matched_schema_terms,
        ),
    }


def db_memory_planning_recall_query(
    prompt: str,
    schema: dict[str, Any],
    intent_kind: str,
    *,
    matched_schema_terms: Iterable[str] | None = None,
) -> str:
    """Build the bounded text used for planning-time memory recall."""
    schema_terms = _bounded_recall_terms(
        (
            matched_schema_terms
            if matched_schema_terms is not None
            else _matched_schema_terms_for_recall(prompt, schema)
        ),
        limit=24,
    )
    recall_terms = _bounded_recall_terms(
        _recall_terms_for_prompt(prompt, schema_terms, intent_kind),
        limit=32,
    )
    lines = [str(prompt or "").strip(), f"Intent: {intent_kind}"]
    if schema_terms:
        lines.append(f"Matched schema terms: {' '.join(schema_terms)}")
    if recall_terms:
        lines.append(f"Recall terms: {' '.join(recall_terms)}")
    return "\n".join(line for line in lines if line).strip()


def _matched_schema_terms_for_recall(
    prompt: str,
    schema: dict[str, Any],
) -> tuple[str, ...]:
    """Return schema terms that are actually mentioned by the prompt."""
    prompt_terms = set(_meaningful_tokens(prompt))
    if not prompt_terms:
        return ()
    matches: list[str] = []
    column_tables: dict[str, list[str]] = {}
    for table in schema.get("tables", []) or []:
        table_name = str(table.get("name") or "").strip()
        if not table_name:
            continue
        if _identifier_matches_terms(table_name, prompt_terms):
            matches.append(table_name)
        for column in table.get("columns", []) or []:
            column_name = str(column.get("name") or "").strip()
            if column_name:
                column_tables.setdefault(_identifier_key(column_name), []).append(
                    table_name
                )

    for table in schema.get("tables", []) or []:
        table_name = str(table.get("name") or "").strip()
        if not table_name:
            continue
        table_matched = table_name in matches
        for column in table.get("columns", []) or []:
            column_name = str(column.get("name") or "").strip()
            if not column_name or not _identifier_matches_terms(
                column_name, prompt_terms
            ):
                continue
            owners = column_tables.get(_identifier_key(column_name), [])
            if table_matched or len(set(owners)) == 1:
                matches.append(f"{table_name}.{column_name}")
            else:
                matches.append(column_name)
    return tuple(_bounded_recall_terms(matches, limit=24))


def _recall_terms_for_prompt(
    prompt: str,
    schema_terms: Iterable[str],
    intent_kind: str,
) -> tuple[str, ...]:
    terms: list[str] = []
    prompt_terms = set(_meaningful_tokens(prompt))
    for term in schema_terms:
        cleaned = str(term or "").strip()
        if not cleaned:
            continue
        terms.append(cleaned)
        if "." not in cleaned and "table" in prompt_terms:
            terms.append(f"{cleaned} table")
    if intent_kind == "schema.relationship_query":
        terms.append("relationship")
    return tuple(terms)


def _bounded_recall_terms(
    terms: Iterable[str],
    *,
    limit: int,
) -> tuple[str, ...]:
    bounded: list[str] = []
    seen: set[str] = set()
    for raw in terms:
        term = re.sub(r"\s+", " ", str(raw or "").strip())
        if not term:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        bounded.append(term)
        if len(bounded) >= max(0, int(limit)):
            break
    return tuple(bounded)


def _identifier_matches_terms(identifier: str, prompt_terms: set[str]) -> bool:
    identifier_terms = set(_meaningful_tokens(identifier))
    if identifier_terms & prompt_terms:
        return True
    return bool(_singular_terms(identifier_terms) & _singular_terms(prompt_terms))


def _identifier_key(identifier: str) -> str:
    return " ".join(_meaningful_tokens(identifier))


def _singular_terms(terms: Iterable[str]) -> set[str]:
    singular: set[str] = set()
    for term in terms:
        text = str(term)
        singular.add(text)
        if len(text) > 3 and text.endswith("ies"):
            singular.add(f"{text[:-3]}y")
        elif len(text) > 3 and text.endswith("s"):
            singular.add(text[:-1])
    return singular


def db_memory_refs_from_recall_evidence(
    recall_evidence: tuple[Any, ...],
    *,
    prompt: str,
    schema: dict[str, Any],
    source_identity: str | None,
    schema_fingerprint: str | None,
    limit: int = 3,
    char_budget: int = 800,
    score_threshold: float = 0.45,
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...], dict[str, Any]]:
    """Project semantic recall evidence into compact, planner-safe DB refs."""
    diagnostics: dict[str, Any] = {
        "registered": bool(recall_evidence),
        "queried": bool(recall_evidence),
        "candidate_count": 0,
        "included_count": 0,
        "omitted_reasons": {},
    }
    candidates: list[tuple[dict[str, Any], Any]] = []
    for evidence in recall_evidence:
        payload = getattr(evidence, "payload", {}) or {}
        for result in payload.get("results", []) or []:
            if isinstance(result, dict):
                candidates.append((result, evidence))
    diagnostics["candidate_count"] = len(candidates)

    refs: list[dict[str, Any]] = []
    evidence_refs: list[str] = []
    used_chars = 0
    for result, evidence in candidates:
        if len(refs) >= max(0, int(limit)):
            _bump_omitted(diagnostics, "limit")
            continue
        record = _db_memory_record_from_recall_result(result)
        reason = _planner_memory_omit_reason(
            record,
            result,
            prompt=prompt,
            schema=schema,
            source_identity=source_identity,
            schema_fingerprint=schema_fingerprint,
            score_threshold=score_threshold,
        )
        if reason:
            _bump_omitted(diagnostics, reason)
            continue
        text = str(record.get("text") or "").strip()
        key = str(record.get("key") or "").strip()
        kind = str(record.get("kind") or "").strip()
        line = f"- {kind} {key}: {text}"
        if used_chars + len(line) > max(0, int(char_budget)):
            _bump_omitted(diagnostics, "budget")
            continue
        metadata = (
            record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        )
        ref_evidence = _memory_evidence_ref_ids(metadata)
        if getattr(evidence, "id", None):
            evidence_refs.append(evidence.id)
        evidence_refs.extend(ref_evidence)
        ref = {
            "chunk_id": result.get("chunk_id"),
            "kind": kind,
            "key": key,
            "text": text,
            "confidence": _confidence_value(metadata.get("confidence"), default=1.0),
            "importance": float(record.get("importance") or 0.0),
            "source_identity": metadata.get("source_identity"),
            "evidence_refs": ref_evidence,
            "schema_fingerprint": metadata.get("source_schema_fingerprint")
            or metadata.get("schema_fingerprint"),
        }
        if metadata.get("active") is False:
            ref["active"] = False
        if metadata.get("stale") is True:
            ref["stale"] = True
        if metadata.get("creation_path"):
            ref["creation_path"] = metadata.get("creation_path")
        if metadata.get("semantic_contract_status"):
            ref["semantic_contract_status"] = metadata.get("semantic_contract_status")
        try:
            contract = _normalize_contract(
                metadata.get(DB_MEMORY_SEMANTIC_CONTRACT_KEY)
            )
        except Exception:
            contract = None
        if contract is not None:
            ref[DB_MEMORY_SEMANTIC_CONTRACT_KEY] = contract
        refs.append(ref)
        used_chars += len(line)
    diagnostics["included_count"] = len(refs)
    diagnostics["char_budget"] = int(char_budget)
    diagnostics["used_chars"] = used_chars
    return tuple(refs), tuple(dict.fromkeys(evidence_refs)), diagnostics


def db_memory_record_refs_known_schema(
    metadata: dict[str, Any],
    schema: dict[str, Any],
) -> bool:
    """Return whether DB memory metadata references known schema objects."""
    return _record_refs_known_schema(metadata, schema)


async def calibrate_db_memory(*args: Any, **kwargs: Any) -> dict[str, Any] | None:
    """Calibrate DB memory for either runtime-owned or legacy adapter callers."""
    if len(args) >= 3:
        await _calibrate_db_memory_agent(*args, **kwargs)
        return None
    if not args:
        raise TypeError("calibrate_db_memory() missing runtime or agent argument")
    return await _calibrate_db_memory_runtime(args[0], **kwargs)


async def _calibrate_db_memory_runtime(
    runtime: Any,
    *,
    source_owner: str,
    marker_key: str,
) -> dict[str, Any]:
    """Infer simple DB unit conventions and persist them as DB memory records."""
    try:
        memory_plugin = runtime.registry.get_plugin("memory")
    except KeyError:
        return {"calibrated": False, "reason": "memory_not_registered"}

    if await has_db_memory_marker(memory_plugin, marker_key):
        return {"calibrated": False, "reason": "marker_exists"}

    schema_evidence = await runtime.execute_capability(
        "db.schema.inspect",
        owner=source_owner,
        operation_type="source.profile",
        input={},
    )
    if schema_evidence:
        runtime.remember_schema_evidence(schema_evidence[0])
    schema = schema_evidence[0].payload if schema_evidence else {}
    records = unit_records_from_schema(schema)
    results = [
        await _write_db_memory_record_runtime(runtime, record) for record in records
    ]
    marker = await _write_db_memory_record_runtime(
        runtime,
        DBMemoryRecord(
            kind="cache_marker",
            key=marker_key,
            text=_marker_content(marker_key),
            importance=0.1,
            metadata={"exact": True},
        ),
    )
    return {
        "calibrated": True,
        "record_count": len(records),
        "records": results,
        "marker": marker,
    }


async def _calibrate_db_memory_agent(
    agent: Any,
    schema: dict[str, Any],
    db_memory: DBMemory,
    *,
    marker_key: str,
) -> None:
    """Infer numeric unit conventions for legacy generic-agent DB memory."""
    try:
        if await db_memory.has_marker(marker_key):
            return
    except Exception as exc:
        logger.debug(f"DB memory marker check failed: {exc}")

    numeric_cols = _numeric_columns(schema)
    if not numeric_cols:
        try:
            await db_memory.mark(marker_key)
        except Exception as exc:
            logger.debug(f"DB memory marker write failed: {exc}")
        return

    prompt = (
        "You are calibrating database memory. Infer unit conventions for the "
        "numeric columns below. Respond only with a JSON array. Each item must "
        'have keys: "table", "column", "unit", "confidence", and optional '
        '"reason". Use "unknown" when evidence is weak.\n\n'
        f"Columns: {json.dumps(numeric_cols, default=_json_default)}"
    )

    try:
        calibration_result = await agent.run(prompt)
        records = _unit_records_from_calibration(calibration_result)
        if records:
            await db_memory.remember_many(records)
        await db_memory.mark(marker_key)
    except Exception as exc:
        logger.debug(f"DB memory calibration failed: {exc}")


async def _write_db_memory_record_runtime(
    runtime: Any, record: DBMemoryRecord
) -> dict[str, Any]:
    """Write one DB memory record through the runtime capability boundary."""
    record = _record_with_runtime_source_identity(runtime, record)
    evidence = await runtime.execute_capability(
        "memory.semantic.write",
        owner="memory",
        operation_type="memory.update",
        input={
            "db_memory_payload": record.to_dict(),
            "db_memory_prompt": record.text,
        },
    )
    if not evidence:
        return {"success": False, "error": "memory write produced no evidence"}
    return dict(evidence[0].payload)


def _record_with_runtime_source_identity(
    runtime: Any,
    record: DBMemoryRecord,
) -> DBMemoryRecord:
    memory_options = db_memory_options_from_runtime_metadata(
        getattr(runtime.config, "metadata", {})
    )
    source_identity = memory_options.get("source_identity")
    metadata = dict(record.metadata)
    if source_identity:
        metadata.setdefault("source_identity", source_identity)
    metadata.setdefault("workspace_scope", "source")
    metadata.setdefault("active", True)
    metadata.setdefault("creation_path", "runtime_calibration")
    return DBMemoryRecord(
        kind=record.kind,
        key=record.key,
        text=record.text,
        metadata=metadata,
        importance=record.importance,
    )


def db_memory_payload_with_runtime_source(
    payload: dict[str, Any],
    runtime_metadata: dict[str, Any],
    *,
    creation_path: str = "explicit_intent",
) -> dict[str, Any]:
    """Attach runtime source-scope metadata to a DB memory payload."""
    from_db_options = _from_db_options(runtime_metadata)
    memory_options = db_memory_options_from_runtime_metadata(runtime_metadata)
    source_identity = memory_options.get("source_identity")
    metadata = dict(payload.get("metadata") or {})
    if source_identity:
        metadata.setdefault("source_identity", source_identity)
    metadata.setdefault("workspace_scope", "source")
    metadata.setdefault("active", True)
    metadata.setdefault("confidence", 1.0)
    metadata.setdefault("creation_path", creation_path)
    catalog_store_id = (
        from_db_options.get("catalog_store_id")
        if isinstance(from_db_options, dict)
        else None
    )
    if catalog_store_id:
        metadata.setdefault("catalog_store_id", catalog_store_id)
    return {**payload, "metadata": metadata}


def db_memory_options_from_runtime_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return normalized DB memory options from runtime metadata."""
    return db_memory_options_from_from_db_options(_from_db_options(metadata))


def db_memory_options_from_from_db_options(options: dict[str, Any]) -> dict[str, Any]:
    """Return normalized DB memory options from `from_db_options`."""
    memory = options.get("memory")
    return memory if isinstance(memory, dict) else {}


def _from_db_options(metadata: dict[str, Any]) -> dict[str, Any]:
    options = metadata.get("from_db_options")
    return options if isinstance(options, dict) else {}


async def has_db_memory_marker(plugin: Any, key: str) -> bool:
    """Return whether an exact DB memory marker exists."""
    marker = _marker_content(key)
    results = await _list_records_by_category(plugin, DB_MARKER_CATEGORY, limit=1000)
    if results is not None:
        return any(marker in str(result.get("content", "")) for result in results)
    return False


async def mark_db_memory(plugin: Any, key: str) -> dict[str, Any]:
    """Write an exact DB memory marker."""
    return await write_db_memory_record(
        plugin,
        DBMemoryRecord(
            kind="cache_marker",
            key=key,
            text=_marker_content(key),
            importance=0.1,
            metadata={"exact": True},
        ),
    )


def unit_records_from_schema(schema: dict[str, Any]) -> tuple[DBMemoryRecord, ...]:
    """Infer obvious numeric unit conventions from schema metadata."""
    records: list[DBMemoryRecord] = []
    for column in _numeric_columns(schema):
        unit, reason = _infer_unit_from_column_name(column["column"])
        if unit is None:
            continue
        confidence = "high" if unit in {"cents", "percent", "basis_points"} else "low"
        table_name = column["table"]
        column_name = column["column"]
        metadata = {
            "table": table_name,
            "column": column_name,
            "unit": unit,
            "confidence": confidence,
            "reason": reason,
        }
        draft = DBMemoryRecord(
            kind="unit_convention",
            key=f"unit_convention:{table_name}.{column_name}",
            text=(
                f"{table_name}.{column_name} is stored as {unit} "
                f"(confidence: {confidence}). Reason: {reason}"
            ),
            metadata=metadata,
            importance=0.75 if confidence == "high" else 0.65,
        )
        contract = extract_db_memory_semantic_contract(draft, schema=schema)
        if contract is not None:
            metadata[DB_MEMORY_SEMANTIC_CONTRACT_KEY] = contract
            metadata["semantic_contract_status"] = "validated"
        records.append(
            DBMemoryRecord(
                kind=draft.kind,
                key=draft.key,
                text=draft.text,
                metadata=metadata,
                importance=draft.importance,
            )
        )
    return tuple(records)


async def recall_db_memory_context(
    agent: Any,
    prompt: str,
    *,
    limit: int = 5,
    classification: Any | None = None,
) -> list[str]:
    """Recall compact DB-specific memory snippets for a user prompt."""
    db_memory = getattr(agent, "_db_memory_semantics", None)
    if db_memory is None:
        return []
    decision = db_memory_recall_decision(agent, prompt, classification=classification)
    setattr(agent, "_db_last_memory_recall_decision", decision)
    if not decision["recall"]:
        return []
    try:
        results = await db_memory.recall(
            prompt,
            kinds=[
                "business_rule",
                "metric_definition",
                "unit_convention",
                "data_contract_note",
                "schema_interpretation",
            ],
            limit=limit,
        )
    except Exception as exc:
        logger.debug(f"DB memory recall failed: {exc}")
        return []

    snippets = []
    for result in results:
        content = str(result.get("content", "")).strip()
        if content:
            snippets.append(_memory_preview(content))
    return snippets[:limit]


def db_memory_recall_decision(
    agent: Any,
    prompt: str,
    *,
    classification: Any | None = None,
) -> dict[str, Any]:
    """Return whether DB semantic memory is useful for this prompt."""

    text = str(prompt or "").lower()
    if classification is not None:
        return _classified_memory_recall_decision(agent, prompt, classification)

    semantic_matches = _matched_terms(text, FALLBACK_SEMANTIC_RECALL_TERMS)
    if semantic_matches:
        return {
            "recall": True,
            "reason": "semantic_prompt",
            "matched_terms": semantic_matches,
        }

    if _looks_row_level(prompt):
        return {"recall": False, "reason": "row_level_prompt", "matched_terms": []}

    identifier_matches = _identifier_matches(text)
    if identifier_matches:
        return {
            "recall": False,
            "reason": "identifier_prompt",
            "matched_terms": identifier_matches,
        }

    if _looks_direct_query(text) and has_likely_catalog_match(agent, text):
        return {
            "recall": False,
            "reason": "direct_schema_matched_query",
            "matched_terms": _matched_terms(text, FALLBACK_DIRECT_QUERY_TERMS),
        }

    return {"recall": True, "reason": "semantic_fallback", "matched_terms": []}


def _classified_memory_recall_decision(
    agent: Any,
    prompt: str,
    classification: Any,
) -> dict[str, Any]:
    if classification.needs_semantic_memory:
        return {
            "recall": True,
            "reason": "semantic_prompt",
            "matched_terms": list(classification.semantic_memory_terms),
        }

    text = str(prompt or "").lower()
    if _looks_row_level(prompt):
        return {"recall": False, "reason": "row_level_prompt", "matched_terms": []}

    identifier_matches = _identifier_matches(text)
    if identifier_matches:
        return {
            "recall": False,
            "reason": "identifier_prompt",
            "matched_terms": identifier_matches,
        }

    if (
        classification.intent.needs_sql_execution
        and classification.likely_catalog_match
        and not classification.needs_memory_tools
    ):
        return {
            "recall": False,
            "reason": "direct_schema_matched_query",
            "matched_terms": [classification.intent.value],
        }

    if classification.needs_memory_tools:
        return {
            "recall": True,
            "reason": "memory_intent",
            "matched_terms": [classification.intent.value],
        }

    return {"recall": True, "reason": "semantic_fallback", "matched_terms": []}


FALLBACK_SEMANTIC_RECALL_TERMS = (
    "business rule",
    "business rules",
    "calculate",
    "calculation",
    "caveat",
    "caveats",
    "definition",
    "definitions",
    "exclude",
    "excludes",
    "include",
    "includes",
    "known issue",
    "known issues",
    "meaning",
    "metric",
    "metrics",
    "remember",
    "stored rule",
    "unit",
    "units",
    "what does",
    "you said",
)
FALLBACK_DIRECT_QUERY_TERMS = (
    "average",
    "avg",
    "count",
    "find",
    "group by",
    "how many",
    "list",
    "max",
    "minimum",
    "min",
    "show",
    "sum",
    "top",
    "total",
)


def _looks_row_level(prompt: str) -> bool:
    """Return True for prompts that ask for row/entity-level values."""

    text = (prompt or "").lower()
    row_value_terms = (
        "email",
        "phone",
        "address",
        "ssn",
        "social security",
        "credit card",
        "customer_id",
        "order_id",
        "user_id",
        "account_id",
    )
    row_action_terms = (
        "show",
        "list",
        "lookup",
        "look up",
        "find",
        "give me",
        "what is",
        "who is",
    )
    row_entity_terms = (" row", " record", " customer", " order", " user", " account")

    if any(term in text for term in row_value_terms):
        return True
    return any(action in text for action in row_action_terms) and any(
        entity in text for entity in row_entity_terms
    )


def _looks_direct_query(text: str) -> bool:
    return _looks_count_intent(text) or bool(
        _matched_terms(text, FALLBACK_DIRECT_QUERY_TERMS)
    )


def _looks_count_intent(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(term in lowered for term in ("count", "how many", "number of", "total"))


def _matched_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if term in text]


def _identifier_matches(text: str) -> list[str]:
    matches = re.findall(
        r"\b(?:[a-z]+_id|[0-9a-f]{8}-[0-9a-f-]{13,}|[\w.+-]+@[\w-]+(?:\.[\w-]+)+)\b",
        text,
    )
    return matches[:5]


def db_memory_record_from_payload(
    payload: dict[str, Any],
    prompt: str,
    *,
    task_metadata: dict[str, Any] | None = None,
) -> DBMemoryRecord:
    """Build a DB memory record from runtime request metadata and constraints."""
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    metadata = _direct_write_contract_metadata(
        dict(metadata),
        task_metadata=task_metadata or {},
    )

    kind = str(payload.get("kind") or "business_rule").strip()
    text = str(payload.get("text") or payload.get("content") or prompt).strip()
    key = str(payload.get("key") or _default_key(kind, text)).strip()
    importance = float(payload.get("importance", 0.7))
    record = normalize_db_memory_record(
        {
            "kind": kind,
            "key": key,
            "text": text,
            "metadata": metadata,
            "importance": importance,
        }
    )
    return record


def normalize_db_memory_record(raw: Any) -> DBMemoryRecord:
    """Validate and normalize a DB memory record."""
    if isinstance(raw, DBMemoryRecord):
        record = raw
    elif isinstance(raw, dict):
        record = DBMemoryRecord(
            kind=str(raw.get("kind") or "").strip(),
            key=str(raw.get("key") or "").strip(),
            text=str(raw.get("text") or raw.get("content") or "").strip(),
            metadata=dict(raw.get("metadata") or {}),
            importance=float(raw.get("importance", 0.7)),
        )
    else:
        raise TypeError("DB memory records must be dictionaries or DBMemoryRecord")

    if record.kind not in DB_MEMORY_KINDS:
        raise ValueError(
            f"Unsupported DB memory kind {record.kind!r}; expected one of "
            f"{sorted(DB_MEMORY_KINDS)}"
        )
    if not record.key:
        raise ValueError("DB memory record requires a key")
    if not record.text:
        raise ValueError("DB memory record requires text")
    if record.kind == "value_alias":
        _validate_value_alias_memory(record)
    metadata = _json_safe(record.metadata)
    if DB_MEMORY_SEMANTIC_CONTRACT_KEY in metadata:
        try:
            metadata[DB_MEMORY_SEMANTIC_CONTRACT_KEY] = _normalize_contract(
                metadata.get(DB_MEMORY_SEMANTIC_CONTRACT_KEY)
            )
        except Exception as exc:
            metadata.pop(DB_MEMORY_SEMANTIC_CONTRACT_KEY, None)
            metadata["semantic_contract_diagnostics"] = {
                "valid": False,
                "reason": str(exc),
            }
    return DBMemoryRecord(
        kind=record.kind,
        key=record.key,
        text=record.text,
        metadata=metadata,
        importance=max(0.0, min(1.0, record.importance)),
    )


def _direct_write_contract_metadata(
    metadata: dict[str, Any],
    *,
    task_metadata: dict[str, Any],
) -> dict[str, Any]:
    if DB_MEMORY_SEMANTIC_CONTRACT_KEY not in metadata:
        return metadata
    if metadata.get("semantic_contract_status") == "validated" and task_metadata.get(
        "reason"
    ) in {"db_memory_commit_update", "db_memory_learning_promotion"}:
        return metadata
    metadata.pop(DB_MEMORY_SEMANTIC_CONTRACT_KEY, None)
    metadata.pop("semantic_contract_status", None)
    metadata["semantic_contract_diagnostics"] = {
        "created": False,
        "reason": "direct_write_unvalidated",
    }
    return metadata


def _validate_value_alias_memory(record: DBMemoryRecord) -> None:
    """Require catalog citation and keep observed values out of memory metadata."""
    metadata = record.metadata
    catalog_ref = str(metadata.get("catalog_profile_ref") or "").strip()
    catalog_evidence_id = str(metadata.get("catalog_evidence_id") or "").strip()
    if not catalog_ref and not catalog_evidence_id:
        raise ValueError(
            "value_alias memory requires catalog_profile_ref or catalog_evidence_id"
        )

    forbidden = _find_forbidden_value_alias_key(metadata)
    if forbidden:
        raise ValueError(
            f"value_alias memory cannot store observed value field {forbidden!r}; "
            "cite catalog evidence instead"
        )


def db_memory_pii_error(*, key: str, text: str, metadata: dict[str, Any]) -> str | None:
    """Return a user-facing validation error when a DB memory stores row values."""
    violation = _detect_pii_value(text) or _detect_pii_value(key)
    if violation:
        return (
            f"DB memory cannot store row-level or PII values ({violation}); "
            "store durable database semantics only."
        )

    sensitive_key = _find_sensitive_metadata_key(metadata)
    if sensitive_key:
        return (
            f"DB memory metadata cannot include sensitive field {sensitive_key!r}; "
            "store durable database semantics only."
        )

    metadata_value = _detect_pii_value(json.dumps(_json_safe(metadata), sort_keys=True))
    if metadata_value:
        return (
            f"DB memory metadata cannot store row-level or PII values ({metadata_value}); "
            "store durable database semantics only."
        )
    return None


def _find_forbidden_value_alias_key(value: Any, prefix: str = "") -> str | None:
    forbidden_keys = {
        "canonical_value",
        "observed_value",
        "observed_values",
        "top_values",
        "value",
        "values",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text.lower() in forbidden_keys:
                return path
            nested = _find_forbidden_value_alias_key(item, path)
            if nested:
                return nested
    elif isinstance(value, list):
        for index, item in enumerate(value):
            nested = _find_forbidden_value_alias_key(item, f"{prefix}[{index}]")
            if nested:
                return nested
    return None


async def _upsert_record(plugin: Any, record: DBMemoryRecord) -> dict[str, Any] | None:
    structured_upsert = _backend_method(plugin, "upsert_db_record")
    if structured_upsert is not None:
        result = structured_upsert(record.to_dict())
        if isawaitable(result):
            result = await result
        if isinstance(result, dict):
            return {
                "status": result.get("status", "created"),
                "updated": 1 if result.get("status") == "updated" else 0,
                "stored": result,
                "structured": True,
            }
        return {"status": "created", "updated": 0, "stored": result, "structured": True}

    existing_chunk_ids = await _find_record_chunk_ids_by_key(plugin, record)
    if existing_chunk_ids:
        deleted = await _delete_record_chunks(plugin, existing_chunk_ids)
        result = await _remember_record(plugin, record)
        return {
            "status": "updated" if deleted else "stored",
            "updated": len(existing_chunk_ids) if deleted else 0,
            "replaced_chunk_ids": existing_chunk_ids if deleted else [],
            "stored": result,
            "upsert_fallback": None if deleted else "append",
        }

    result = await _remember_record(plugin, record)
    return {"status": "created", "updated": 0, "stored": result}


async def _remember_record(
    plugin: Any, record: DBMemoryRecord
) -> dict[str, Any] | None:
    content = record.to_memory_content()
    backend = getattr(plugin, "backend", None)
    if backend is not None and hasattr(backend, "remember"):
        from daita.plugins.memory.metadata import MemoryMetadata

        metadata = MemoryMetadata(
            content=content,
            importance=record.importance,
            source="agent_inferred",
            category=record.category,
        )
        return await backend.remember(
            content,
            category=record.category,
            metadata=metadata,
            extra_metadata={"db_memory": record.to_dict()},
            index_content=record.text,
        )

    remember = getattr(plugin, "remember", None)
    if remember is None:
        return None
    result = remember(
        content,
        importance=record.importance,
        category=record.category,
    )
    if isawaitable(result):
        result = await result
    return result


async def _find_record_chunk_ids_by_key(
    plugin: Any, record: DBMemoryRecord
) -> list[str]:
    structured_list = _backend_method(plugin, "list_db_records")
    if structured_list is not None:
        result = structured_list(
            category=record.category,
            key=record.key,
            source_identity=record.metadata.get("source_identity"),
            limit=1000,
        )
        if isawaitable(result):
            result = await result
        if isinstance(result, list):
            return [
                str(item.get("chunk_id") or item.get("record_id"))
                for item in result
                if item.get("chunk_id") or item.get("record_id")
            ]

    results = await _list_records_by_category(plugin, record.category, limit=1000)
    if results is None:
        return []

    chunk_ids = []
    for result in results:
        if _record_key_from_result(result) != record.key:
            continue
        chunk_id = result.get("chunk_id")
        if chunk_id:
            chunk_ids.append(str(chunk_id))
    return chunk_ids


async def _list_records_by_category(
    plugin: Any, category: str, *, limit: int
) -> list[dict[str, Any]] | None:
    structured_list = _backend_method(plugin, "list_db_records")
    if structured_list is not None:
        result = structured_list(category=category, limit=limit)
        if isawaitable(result):
            result = await result
        return result if isinstance(result, list) else None

    backend = getattr(plugin, "backend", None)
    if backend is not None and hasattr(backend, "list_by_category"):
        result = backend.list_by_category(category=category, limit=limit)
        if isawaitable(result):
            result = await result
        return result if isinstance(result, list) else None
    return None


async def _recall_records(
    plugin: Any,
    query: str,
    *,
    limit: int,
    score_threshold: float,
    category: str,
    kinds: list[str] | tuple[str, ...] | set[str] | None = None,
) -> list[dict[str, Any]]:
    structured_recall = _backend_method(plugin, "recall_db_records")
    if structured_recall is not None and category == DB_SEMANTIC_CATEGORY:
        result = structured_recall(
            query,
            limit=limit,
            score_threshold=score_threshold,
            category=category,
            kinds=kinds,
        )
        if isawaitable(result):
            result = await result
        return result if isinstance(result, list) else []

    backend = getattr(plugin, "backend", None)
    if backend is not None and hasattr(backend, "recall"):
        return await backend.recall(
            query,
            limit=limit,
            score_threshold=score_threshold,
            category=category,
        )
    recall = getattr(plugin, "recall", None)
    if recall is None:
        return []
    result = recall(
        query,
        limit=limit,
        score_threshold=score_threshold,
        category=category,
    )
    if isawaitable(result):
        result = await result
    return result if isinstance(result, list) else []


async def _delete_record_chunks(plugin: Any, chunk_ids: list[str]) -> bool:
    if not chunk_ids:
        return True
    backend = getattr(plugin, "backend", None)
    if backend is not None and hasattr(backend, "delete_chunks"):
        await backend.delete_chunks(chunk_ids)
        return True
    delete_chunks = getattr(plugin, "delete_chunks", None)
    if delete_chunks is None:
        return False
    result = delete_chunks(chunk_ids)
    if isawaitable(result):
        await result
    return True


def _backend_method(plugin: Any, name: str) -> Any | None:
    backend = getattr(plugin, "backend", None)
    if backend is None:
        return None
    try:
        getattr_static(backend, name)
    except AttributeError:
        return None
    method = getattr(backend, name, None)
    return method if callable(method) else None


def _record_kind_from_result(result: dict[str, Any]) -> str | None:
    metadata = result.get("metadata") or {}
    db_memory = metadata.get("db_memory")
    if isinstance(db_memory, dict) and db_memory.get("kind"):
        return str(db_memory["kind"])

    content = str(result.get("content", ""))
    try:
        marker = "DB memory record:\n"
        if content.startswith(marker):
            payload = json.loads(content[len(marker) :])
            if isinstance(payload, dict) and payload.get("kind"):
                return str(payload["kind"])
    except Exception:
        return None
    return None


def _record_key_from_result(result: dict[str, Any]) -> str | None:
    metadata = result.get("metadata") or {}
    db_memory = metadata.get("db_memory")
    if isinstance(db_memory, dict) and db_memory.get("key"):
        return str(db_memory["key"])

    content = str(result.get("content", ""))
    try:
        marker = "DB memory record:\n"
        if content.startswith(marker):
            payload = json.loads(content[len(marker) :])
            if isinstance(payload, dict) and payload.get("key"):
                return str(payload["key"])
    except Exception:
        return None
    return None


def _db_memory_record_from_recall_result(result: dict[str, Any]) -> dict[str, Any]:
    metadata = result.get("metadata") or {}
    db_memory = metadata.get("db_memory") if isinstance(metadata, dict) else None
    if isinstance(db_memory, dict):
        return dict(db_memory)

    content = str(result.get("content", ""))
    try:
        marker = "DB memory record:\n"
        if content.startswith(marker):
            payload = json.loads(content[len(marker) :])
            if isinstance(payload, dict):
                return payload
    except Exception:
        return {}
    return {}


def _planner_memory_omit_reason(
    record: dict[str, Any],
    result: dict[str, Any],
    *,
    prompt: str,
    schema: dict[str, Any],
    source_identity: str | None,
    schema_fingerprint: str | None,
    score_threshold: float,
) -> str | None:
    kind = str(record.get("kind") or "")
    key = str(record.get("key") or "")
    text = str(record.get("text") or "")
    metadata = (
        record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    )
    if kind not in DB_PLANNING_MEMORY_KINDS:
        return "unsupported_kind"
    if not key or not text:
        return "malformed"
    if not _score_is_high_enough(result, score_threshold):
        return "low_score"
    record_source = metadata.get("source_identity")
    if source_identity and record_source != source_identity:
        return "cross_source" if record_source else "missing_source_identity"
    if metadata.get("active") is False:
        return "inactive"
    if bool(metadata.get("stale")) or _is_memory_expired(metadata):
        return "stale"
    record_schema = metadata.get("source_schema_fingerprint") or metadata.get(
        "schema_fingerprint"
    )
    if record_schema and schema_fingerprint and record_schema != schema_fingerprint:
        return "stale_schema"
    if _confidence_value(metadata.get("confidence"), default=1.0) < 0.5:
        return "low_confidence"
    if kind == "value_alias" and not _value_alias_has_catalog_citation(metadata):
        return "missing_catalog_citation"
    if db_memory_pii_error(key=key, text=text, metadata=metadata):
        return "unsafe"
    if not _record_refs_known_schema(metadata, schema):
        if _has_valid_semantic_contract(metadata):
            return None
        return "schema_scope_mismatch"
    if not _memory_relevant_to_prompt(prompt, record):
        return "irrelevant"
    return None


def _has_valid_semantic_contract(metadata: dict[str, Any]) -> bool:
    try:
        return (
            _normalize_contract(metadata.get(DB_MEMORY_SEMANTIC_CONTRACT_KEY))
            is not None
        )
    except Exception:
        return False


def _score_is_high_enough(result: dict[str, Any], threshold: float) -> bool:
    score = result.get("relevance_score", result.get("score"))
    if score is None:
        return True
    try:
        return float(score) >= float(threshold)
    except (TypeError, ValueError):
        return True


def _confidence_value(value: Any, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "high":
            return 0.9
        if lowered == "medium":
            return 0.7
        if lowered == "low":
            return 0.4
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _is_memory_expired(metadata: dict[str, Any]) -> bool:
    expires_at = metadata.get("expires_at")
    if not expires_at:
        return False
    try:
        parsed = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
    except ValueError:
        return True
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed <= datetime.now(timezone.utc)


def _value_alias_has_catalog_citation(metadata: dict[str, Any]) -> bool:
    return bool(
        metadata.get("catalog_profile_ref")
        or metadata.get("catalog_evidence_id")
        or metadata.get("catalog_refs")
    )


def _record_refs_known_schema(metadata: dict[str, Any], schema: dict[str, Any]) -> bool:
    schema_refs = metadata.get("schema_refs")
    if isinstance(schema_refs, list) and schema_refs:
        return _schema_refs_known_schema(
            _schema_refs_from_metadata(schema_refs), schema
        )
    tables = {
        str(table.get("name") or "").lower(): {
            str(column.get("name") or "").lower()
            for column in table.get("columns", []) or []
            if column.get("name")
        }
        for table in schema.get("tables", []) or []
        if table.get("name")
    }
    table = str(metadata.get("table") or "").lower()
    column = str(metadata.get("column") or "").lower()
    if table and table not in tables:
        return False
    if table and column and column not in tables.get(table, set()):
        return False
    return True


def _schema_refs_from_metadata(value: Any) -> tuple[dict[str, str], ...]:
    refs: list[dict[str, str]] = []
    if not isinstance(value, (list, tuple, set)):
        return ()
    for item in value:
        if isinstance(item, dict):
            table = str(item.get("table") or "").strip()
            column = str(item.get("column") or "").strip()
        else:
            parts = [
                part.strip('"`[] ')
                for part in str(item or "").split(".")
                if part.strip()
            ]
            if len(parts) >= 2:
                table, column = parts[-2], parts[-1]
            else:
                table, column = "", ""
        if table and column:
            refs.append({"table": table, "column": column})
        elif table:
            refs.append({"table": table})
    return tuple(refs)


def _schema_refs_known_schema(
    refs: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    schema: dict[str, Any],
) -> bool:
    tables = {
        str(table.get("name") or "").lower(): {
            str(column.get("name") or "").lower()
            for column in table.get("columns", []) or []
            if column.get("name")
        }
        for table in schema.get("tables", []) or []
        if table.get("name")
    }
    if not tables:
        return True
    for ref in refs:
        table = str(ref.get("table") or "").lower()
        column = str(ref.get("column") or "").lower()
        if table and table not in tables:
            return False
        if table and column and column not in tables.get(table, set()):
            return False
    return True


def _memory_relevant_to_prompt(prompt: str, record: dict[str, Any]) -> bool:
    prompt_tokens = set(_meaningful_tokens(prompt))
    if not prompt_tokens:
        return True
    metadata = (
        record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    )
    record_text = " ".join(
        str(value)
        for value in (
            record.get("kind"),
            record.get("key"),
            record.get("text"),
            metadata.get("metric"),
            metadata.get("table"),
            metadata.get("column"),
            metadata.get("alias"),
        )
        if value
    )
    record_tokens = set(_meaningful_tokens(record_text))
    return bool(prompt_tokens & record_tokens)


def _prompt_matches_schema(prompt: str, schema: dict[str, Any]) -> bool:
    prompt_tokens = set(_meaningful_tokens(prompt))
    if not prompt_tokens:
        return False
    for table in schema.get("tables", []) or []:
        names = [table.get("name")]
        names.extend(column.get("name") for column in table.get("columns", []) or [])
        if prompt_tokens & set(
            _meaningful_tokens(" ".join(str(n) for n in names if n))
        ):
            return True
    return False


def _meaningful_tokens(text: Any) -> list[str]:
    stop = {
        "about",
        "after",
        "before",
        "calculate",
        "computed",
        "does",
        "from",
        "have",
        "many",
        "show",
        "that",
        "their",
        "there",
        "this",
        "what",
        "when",
        "where",
        "which",
        "with",
    }
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", str(text or "").lower())
    expanded: list[str] = []
    for token in tokens:
        expanded.extend(part for part in token.split("_") if len(part) > 2)
        expanded.append(token)
    return [token for token in expanded if token not in stop]


def _memory_evidence_ref_ids(metadata: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    raw = metadata.get("evidence_refs")
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and item.get("id"):
                refs.append(str(item["id"]))
            elif isinstance(item, str):
                refs.append(item)
    for key in ("catalog_evidence_id", "proposal_evidence_id"):
        if metadata.get(key):
            refs.append(str(metadata[key]))
    return list(dict.fromkeys(refs))


def _bump_omitted(diagnostics: dict[str, Any], reason: str) -> None:
    omitted = diagnostics.setdefault("omitted_reasons", {})
    omitted[reason] = int(omitted.get(reason) or 0) + 1


def _dedupe_recall_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for result in results:
        key = result.get("chunk_id") or result.get("content")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


def _unit_records_from_calibration(result: Any) -> list[DBMemoryRecord]:
    items = _parse_json_array(result)
    records = []
    for item in items:
        table = str(item.get("table") or "").strip()
        column = str(item.get("column") or "").strip()
        unit = str(item.get("unit") or "unknown").strip()
        confidence = str(item.get("confidence") or "low").strip()
        if not table or not column:
            continue
        key = f"unit_convention:{table}.{column}"
        text = f"{table}.{column} is stored as {unit} (confidence: {confidence})."
        if item.get("reason"):
            text += f" Reason: {item['reason']}"
        records.append(
            DBMemoryRecord(
                kind="unit_convention",
                key=key,
                text=text,
                metadata={
                    "table": table,
                    "column": column,
                    "unit": unit,
                    "confidence": confidence,
                    "reason": item.get("reason"),
                },
                importance=0.75 if confidence == "high" else 0.65,
            )
        )
    return records


def _parse_json_array(result: Any) -> list[dict[str, Any]]:
    text = result if isinstance(result, str) else str(result)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _memory_preview(content: str, max_chars: int = 240) -> str:
    try:
        marker = "DB memory record:\n"
        if content.startswith(marker):
            payload = json.loads(content[len(marker) :])
            content = payload.get("text") or content
    except Exception:
        pass
    return content if len(content) <= max_chars else content[: max_chars - 3] + "..."


def _numeric_columns(schema: dict[str, Any]) -> list[dict[str, Any]]:
    numeric = []
    for table in schema.get("tables", []) or []:
        table_name = str(table.get("name") or "").strip()
        if not table_name:
            continue
        for column in table.get("columns", []) or []:
            column_name = str(column.get("name") or "").strip()
            column_type = str(
                column.get("data_type") or column.get("type") or ""
            ).strip()
            if column_name and _is_numeric_type(column_type):
                entry: dict[str, Any] = {
                    "table": table_name,
                    "column": column_name,
                    "type": column_type,
                }
                if column.get("_samples"):
                    entry["samples"] = column["_samples"]
                if column.get("column_comment"):
                    entry["comment"] = column["column_comment"]
                numeric.append(entry)
    return numeric


def _is_numeric_type(value: str) -> bool:
    text = value.lower()
    return any(
        token in text
        for token in (
            "bigint",
            "decimal",
            "double",
            "float",
            "int",
            "numeric",
            "number",
            "real",
        )
    )


def _infer_unit_from_column_name(column: str) -> tuple[str | None, str | None]:
    text = column.lower()
    if "cents" in text or text.endswith("_cent"):
        return "cents", "column name contains cents"
    if "basis_points" in text or text.endswith("_bps") or text == "bps":
        return "basis_points", "column name indicates basis points"
    if "percent" in text or text.endswith("_pct") or text.endswith("_percentage"):
        return "percent", "column name indicates percent"
    return None, None


def _detect_pii_value(value: str) -> str | None:
    text = str(value or "")
    if not text:
        return None
    for label, pattern in PII_VALUE_PATTERNS:
        for match in pattern.finditer(text):
            candidate = match.group(0)
            if label == "credit card number" and not _looks_like_credit_card(candidate):
                continue
            return label
    return None


def _looks_like_credit_card(candidate: str) -> bool:
    digits = [int(ch) for ch in candidate if ch.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _find_sensitive_metadata_key(
    metadata: dict[str, Any], prefix: str = ""
) -> str | None:
    for key, value in metadata.items():
        key_text = str(key).lower()
        if _metadata_key_is_sensitive(key_text):
            return f"{prefix}{key}" if prefix else str(key)
        if isinstance(value, dict):
            nested = _find_sensitive_metadata_key(
                value, prefix=f"{prefix}{key}." if prefix else f"{key}."
            )
            if nested:
                return nested
    return None


def _metadata_key_is_sensitive(key_text: str) -> bool:
    if key_text in SENSITIVE_METADATA_KEYS:
        return True
    tokens = {token for token in re.split(r"[^a-z0-9]+", key_text) if token}
    if tokens & SENSITIVE_METADATA_KEYS:
        return True
    for pattern in SENSITIVE_METADATA_KEYS:
        pattern_tokens = {token for token in re.split(r"[^a-z0-9]+", pattern) if token}
        if pattern_tokens and pattern_tokens <= tokens:
            return True
    return False


def _default_key(kind: str, text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())[:8]
    slug = "_".join(words) or "memory"
    return f"{kind}:{slug}"


def _marker_content(key: str) -> str:
    return f"DB exact cache marker: {key}"


def _json_default(value: Any) -> Any:
    if isinstance(value, decimal.Decimal):
        return float(value)
    return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return _json_default(value)
