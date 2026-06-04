"""
DB-specific memory semantics for the operation-centric runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import decimal
from inspect import isawaitable
import json
import logging
import re
from typing import Any

from daita.db.query_catalog import has_likely_catalog_match

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
        "cache_marker",
    }
)
DB_SEMANTIC_MEMORY_KINDS = (
    "unit_convention",
    "metric_definition",
    "business_rule",
    "data_contract_note",
    "schema_interpretation",
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


def create_db_memory_tools(db_memory: DBMemory) -> list[Any]:
    """Create LLM-callable DB memory tool views with strict from_db semantics."""
    from daita.core.tools import LocalTool
    from daita.db.capabilities import MEMORY_SEMANTIC_WRITE_CAPABILITY

    async def db_remember_handler(args: dict[str, Any]) -> dict[str, Any]:
        kind = str(args.get("kind") or "").strip()
        if kind not in DB_SEMANTIC_MEMORY_KINDS:
            return {
                "success": False,
                "error": (
                    f"Unsupported DB memory kind {kind!r}; expected one of "
                    f"{list(DB_SEMANTIC_MEMORY_KINDS)}"
                ),
            }

        metadata = args.get("metadata") or {}
        if not isinstance(metadata, dict):
            return {"success": False, "error": "metadata must be an object"}

        pii_error = db_memory_pii_error(
            key=str(args.get("key") or ""),
            text=str(args.get("text") or ""),
            metadata=metadata,
        )
        if pii_error:
            return {"success": False, "error": pii_error}

        try:
            record = DBMemoryRecord(
                kind=kind,
                key=str(args.get("key") or "").strip(),
                text=str(args.get("text") or "").strip(),
                metadata=metadata,
                importance=float(args.get("importance", 0.7)),
            )
            result = await db_memory.remember(record)
        except Exception as exc:
            return {"success": False, "error": str(exc)}

        return {
            "success": True,
            "kind": record.kind,
            "key": record.key,
            "category": record.category,
            "status": (result or {}).get("status"),
            "updated": (result or {}).get("updated", 0),
            "stored": result,
        }

    return [
        LocalTool(
            name="db_remember",
            description=(
                "Store or revise durable database semantics for this from_db agent. "
                "Use only for metric definitions, business rules, unit conventions, "
                "data contracts, or schema interpretation; never store row values."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": list(DB_SEMANTIC_MEMORY_KINDS),
                        "description": "The structured DB memory kind.",
                    },
                    "key": {
                        "type": "string",
                        "description": (
                            "Stable unique key, such as metric:revenue or "
                            "business_rule:refunds."
                        ),
                    },
                    "text": {
                        "type": "string",
                        "description": "Self-contained semantic memory text.",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional structured context for the memory.",
                    },
                    "importance": {
                        "type": "number",
                        "description": "Importance from 0.0 to 1.0.",
                    },
                },
                "required": ["kind", "key", "text"],
            },
            handler=db_remember_handler,
            source="from_db",
            category="memory",
            capability_ids=(MEMORY_SEMANTIC_WRITE_CAPABILITY,),
        )
    ]


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
    )
    if kinds:
        allowed = set(kinds)
        results = [
            result for result in results if _record_kind_from_result(result) in allowed
        ]
    return _dedupe_recall_results(results)[:limit]


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
        records.append(
            DBMemoryRecord(
                kind="unit_convention",
                key=f"unit_convention:{table_name}.{column_name}",
                text=(
                    f"{table_name}.{column_name} is stored as {unit} "
                    f"(confidence: {confidence}). Reason: {reason}"
                ),
                metadata={
                    "table": table_name,
                    "column": column_name,
                    "unit": unit,
                    "confidence": confidence,
                    "reason": reason,
                },
                importance=0.75 if confidence == "high" else 0.65,
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
    payload: dict[str, Any], prompt: str
) -> DBMemoryRecord:
    """Build a DB memory record from runtime request metadata and constraints."""
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")

    kind = str(payload.get("kind") or "business_rule").strip()
    text = str(payload.get("text") or payload.get("content") or prompt).strip()
    key = str(payload.get("key") or _default_key(kind, text)).strip()
    importance = float(payload.get("importance", 0.7))
    return normalize_db_memory_record(
        {
            "kind": kind,
            "key": key,
            "text": text,
            "metadata": metadata,
            "importance": importance,
        }
    )


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
    return DBMemoryRecord(
        kind=record.kind,
        key=record.key,
        text=record.text,
        metadata=_json_safe(record.metadata),
        importance=max(0.0, min(1.0, record.importance)),
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


async def _upsert_record(plugin: Any, record: DBMemoryRecord) -> dict[str, Any] | None:
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
) -> list[dict[str, Any]]:
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
        if key_text in SENSITIVE_METADATA_KEYS or any(
            pattern in key_text for pattern in SENSITIVE_METADATA_KEYS
        ):
            return f"{prefix}{key}" if prefix else str(key)
        if isinstance(value, dict):
            nested = _find_sensitive_metadata_key(
                value, prefix=f"{prefix}{key}." if prefix else f"{key}."
            )
            if nested:
                return nested
    return None


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
