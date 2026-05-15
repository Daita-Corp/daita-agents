"""
DB-specific memory semantics for ``from_db()`` agents.
"""

import decimal
import json
import logging
import re
from dataclasses import dataclass, field
from inspect import isawaitable
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from .schema.discovery import is_numeric_type
from .schema.sampling import PII_COLUMN_PATTERNS

if TYPE_CHECKING:
    from ..agent import Agent

logger = logging.getLogger(__name__)

DB_MEMORY_KINDS = {
    "unit_convention",
    "metric_definition",
    "business_rule",
    "data_contract_note",
    "schema_interpretation",
    "cache_marker",
}
DB_SEMANTIC_MEMORY_KINDS = (
    "unit_convention",
    "metric_definition",
    "business_rule",
    "data_contract_note",
    "schema_interpretation",
)
DB_SEMANTIC_CATEGORY = "db_semantics"
DB_MARKER_CATEGORY = "db_cache_marker"
SENSITIVE_METADATA_KEYS = frozenset(
    PII_COLUMN_PATTERNS
    + [
        "authorization",
        "bearer",
        "credential",
        "credentials",
        "private_key",
    ]
)
PII_VALUE_PATTERNS = (
    ("email address", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("US SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    (
        "credit card number",
        re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    ),
    (
        "phone number",
        re.compile(
            r"(?<!\w)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)"
        ),
    ),
)


@dataclass(frozen=True)
class DBMemoryRecord:
    """Structured DB memory record stored through MemoryPlugin."""

    kind: str
    key: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.7

    @property
    def category(self) -> str:
        if self.kind == "cache_marker":
            return DB_MARKER_CATEGORY
        return DB_SEMANTIC_CATEGORY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "key": self.key,
            "text": self.text,
            "metadata": _json_safe(self.metadata),
            "importance": self.importance,
            "category": self.category,
        }

    def to_memory_content(self) -> str:
        payload = self.to_dict()
        return f"DB memory record:\n{json.dumps(payload, sort_keys=True)}"


class DBMemory:
    """Thin adapter that gives MemoryPlugin DB-specific semantics."""

    def __init__(self, plugin: Any):
        self.plugin = plugin

    async def remember(self, record: Any) -> Optional[Dict[str, Any]]:
        normalized = normalize_db_memory_record(record)
        return await _upsert_record(self.plugin, normalized)

    async def remember_many(self, records: Iterable[Any]) -> List[Dict[str, Any]]:
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
        kinds: Optional[Iterable[str]] = None,
        limit: int = 5,
        score_threshold: float = 0.45,
    ) -> List[Dict[str, Any]]:
        results = await _recall_records(
            self.plugin,
            query,
            limit=max(limit * 3, limit),
            score_threshold=score_threshold,
            category=DB_SEMANTIC_CATEGORY,
        )
        if kinds:
            allowed = set(kinds)
            results = [
                result
                for result in results
                if _record_kind_from_result(result) in allowed
            ]
        return _dedupe_recall_results(results)[:limit]

    async def has_marker(self, key: str) -> bool:
        marker = _marker_content(key)
        exact_results = await _list_records_by_category(
            self.plugin,
            DB_MARKER_CATEGORY,
            limit=1000,
        )
        if exact_results is not None:
            return any(
                marker in str(result.get("content", "")) for result in exact_results
            )
        results = await _recall_records(
            self.plugin,
            marker,
            limit=20,
            score_threshold=0.0,
            category=DB_MARKER_CATEGORY,
        )
        return any(marker in str(result.get("content", "")) for result in results)

    async def mark(self, key: str) -> Optional[Dict[str, Any]]:
        return await self.remember(
            DBMemoryRecord(
                kind="cache_marker",
                key=key,
                text=_marker_content(key),
                importance=0.1,
                metadata={"exact": True},
            )
        )


def create_db_memory_tools(db_memory: DBMemory) -> List[Any]:
    """Create LLM-callable DB memory tools with strict from_db semantics."""
    from ...core.tools import AgentTool

    async def db_remember_handler(args: Dict[str, Any]) -> Dict[str, Any]:
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

        pii_error = _db_memory_pii_error(
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
        AgentTool(
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
        )
    ]


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


def _db_memory_pii_error(
    *, key: str, text: str, metadata: Dict[str, Any]
) -> Optional[str]:
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


def _detect_pii_value(value: str) -> Optional[str]:
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
    metadata: Dict[str, Any], prefix: str = ""
) -> Optional[str]:
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


async def calibrate_db_memory(
    agent: "Agent",
    schema: Dict[str, Any],
    db_memory: DBMemory,
    *,
    marker_key: str,
) -> None:
    """Infer numeric unit conventions and store structured DB memory records."""
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


async def recall_db_memory_context(
    agent: "Agent",
    prompt: str,
    *,
    limit: int = 5,
) -> List[str]:
    """Recall compact DB-specific memory snippets for a user prompt."""
    db_memory = getattr(agent, "_db_memory_semantics", None)
    if db_memory is None:
        return []
    if _looks_row_level(prompt):
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


def _looks_row_level(prompt: str) -> bool:
    """Return True for prompts that ask for row/entity-level values.

    DB memory is for durable semantics: metric definitions, business rules,
    units, and schema interpretation. Exact row values must come from guarded
    database tools so memory cannot become an ungoverned row cache.
    """

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


def _numeric_columns(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    numeric_cols: List[Dict[str, Any]] = []
    for table in schema.get("tables", []):
        for col in table.get("columns", []):
            if is_numeric_type(col.get("type", "")):
                entry: Dict[str, Any] = {
                    "table": table["name"],
                    "column": col["name"],
                    "type": col["type"],
                }
                if col.get("_samples"):
                    entry["samples"] = col["_samples"]
                if col.get("column_comment"):
                    entry["comment"] = col["column_comment"]
                numeric_cols.append(entry)
    return numeric_cols


def _unit_records_from_calibration(result: Any) -> List[DBMemoryRecord]:
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
        text = f"{table}.{column} is stored as {unit} " f"(confidence: {confidence})."
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


async def _remember_record(
    plugin: Any, record: DBMemoryRecord
) -> Optional[Dict[str, Any]]:
    content = record.to_memory_content()
    backend = getattr(plugin, "backend", None)
    if backend is not None and hasattr(backend, "remember"):
        try:
            from ...plugins.memory.metadata import MemoryMetadata

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
        except Exception as exc:
            logger.debug(f"Direct DB memory store failed: {exc}")

    if hasattr(plugin, "remember"):
        return await plugin.remember(
            content,
            importance=record.importance,
            category=record.category,
        )
    return None


async def _upsert_record(
    plugin: Any, record: DBMemoryRecord
) -> Optional[Dict[str, Any]]:
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


async def _find_record_chunk_ids_by_key(
    plugin: Any, record: DBMemoryRecord
) -> List[str]:
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


async def _delete_record_chunks(plugin: Any, chunk_ids: List[str]) -> bool:
    if not chunk_ids:
        return True
    backend = getattr(plugin, "backend", None)
    if backend is not None and hasattr(backend, "delete_chunks"):
        await backend.delete_chunks(chunk_ids)
        return True
    if hasattr(plugin, "delete_chunks"):
        await plugin.delete_chunks(chunk_ids)
        return True
    return False


async def _recall_records(
    plugin: Any,
    query: str,
    *,
    limit: int,
    score_threshold: float,
    category: Optional[str],
) -> List[Dict[str, Any]]:
    backend = getattr(plugin, "backend", None)
    if backend is not None and hasattr(backend, "recall"):
        return await backend.recall(
            query,
            limit=limit,
            score_threshold=score_threshold,
            category=category,
        )
    if hasattr(plugin, "recall"):
        return await plugin.recall(
            query,
            limit=limit,
            score_threshold=score_threshold,
            category=category,
        )
    return []


async def _list_records_by_category(
    plugin: Any, category: str, *, limit: int
) -> Optional[List[Dict[str, Any]]]:
    backend = getattr(plugin, "backend", None)
    if backend is not None and hasattr(backend, "list_by_category"):
        result = backend.list_by_category(category=category, limit=limit)
        if isawaitable(result):
            result = await result
        return result if isinstance(result, list) else None
    return None


def _parse_json_array(result: Any) -> List[Dict[str, Any]]:
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


def _dedupe_recall_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for result in results:
        key = result.get("chunk_id") or result.get("content")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


def _record_kind_from_result(result: Dict[str, Any]) -> Optional[str]:
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


def _record_key_from_result(result: Dict[str, Any]) -> Optional[str]:
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


def _memory_preview(content: str, max_chars: int = 240) -> str:
    try:
        marker = "DB memory record:\n"
        if content.startswith(marker):
            payload = json.loads(content[len(marker) :])
            content = payload.get("text") or content
    except Exception:
        pass
    return content if len(content) <= max_chars else content[: max_chars - 1] + "…"


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
    return str(value)
