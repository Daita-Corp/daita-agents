"""Structured storage and indexed recall for DB semantic memory."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

DB_SEMANTIC_CATEGORY = "db_semantics"
DB_MARKER_CATEGORY = "db_cache_marker"

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


class DBSemanticMemoryStore:
    """SQLite-backed structured DB semantic memory substore."""

    def __init__(self, path: Path, *, default_source_identity: str | None = None):
        self.path = Path(path)
        self.default_source_identity = default_source_identity
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    async def upsert_db_record(self, record: dict[str, Any]) -> dict[str, Any]:
        return self.upsert_db_record_sync(record)

    def upsert_db_record_sync(self, record: dict[str, Any]) -> dict[str, Any]:
        normalized = _normalize_record(record, self.default_source_identity)
        now = _utc_now()
        identity_key = normalized["source_identity"] or ""
        record_id = _record_id(identity_key, normalized["key"])
        lexical_document = _lexical_document(normalized)

        conn = sqlite3.connect(str(self.path))
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT record_id FROM db_memory_records
            WHERE source_identity_key = ? AND key = ?
            """,
            (identity_key, normalized["key"]),
        )
        existing = cursor.fetchone()
        status = "updated" if existing else "created"
        if existing:
            record_id = str(existing[0])

        cursor.execute(
            """
            INSERT INTO db_memory_records (
                record_id, source_identity_key, key, kind, text, category,
                source_identity, workspace_scope, schema_refs_json,
                catalog_refs_json, aliases_json, confidence, importance,
                active, stale, expires_at, metadata_json, lexical_document,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_identity_key, key) DO UPDATE SET
                kind = excluded.kind,
                text = excluded.text,
                category = excluded.category,
                source_identity = excluded.source_identity,
                workspace_scope = excluded.workspace_scope,
                schema_refs_json = excluded.schema_refs_json,
                catalog_refs_json = excluded.catalog_refs_json,
                aliases_json = excluded.aliases_json,
                confidence = excluded.confidence,
                importance = excluded.importance,
                active = excluded.active,
                stale = excluded.stale,
                expires_at = excluded.expires_at,
                metadata_json = excluded.metadata_json,
                lexical_document = excluded.lexical_document,
                updated_at = excluded.updated_at
            """,
            (
                record_id,
                identity_key,
                normalized["key"],
                normalized["kind"],
                normalized["text"],
                normalized["category"],
                normalized["source_identity"],
                normalized["workspace_scope"],
                json.dumps(normalized["schema_refs"], sort_keys=True),
                json.dumps(normalized["catalog_refs"], sort_keys=True),
                json.dumps(normalized["aliases"], sort_keys=True),
                normalized["confidence"],
                normalized["importance"],
                1 if normalized["active"] else 0,
                1 if normalized["stale"] else 0,
                normalized["expires_at"],
                json.dumps(normalized["metadata"], sort_keys=True),
                lexical_document,
                now,
                now,
            ),
        )
        _refresh_index_rows(cursor, record_id, normalized, lexical_document)
        conn.commit()
        conn.close()

        return {
            "status": status,
            "chunk_id": record_id,
            "record_id": record_id,
            "indexed": True,
            "structured": True,
            "db_memory": normalized,
        }

    async def delete_db_records_by_key(
        self,
        key: str,
        *,
        source_identity: str | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        identity_key = source_identity or self.default_source_identity or ""
        clauses = ["key = ?", "source_identity_key = ?"]
        params: list[Any] = [key, identity_key]
        if category:
            clauses.append("category = ?")
            params.append(category)

        conn = sqlite3.connect(str(self.path))
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT record_id FROM db_memory_records WHERE {' AND '.join(clauses)}",
            params,
        )
        record_ids = [str(row[0]) for row in cursor.fetchall()]
        cursor.execute(
            f"DELETE FROM db_memory_records WHERE {' AND '.join(clauses)}",
            params,
        )
        deleted = cursor.rowcount
        _delete_index_rows(cursor, record_ids)
        conn.commit()
        conn.close()
        return {"deleted": int(deleted)}

    async def list_db_records(
        self,
        *,
        category: str | None = None,
        key: str | None = None,
        source_identity: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if category:
            clauses.append("category = ?")
            params.append(category)
        if key:
            clauses.append("key = ?")
            params.append(key)
        identity = source_identity if source_identity is not None else None
        if identity is not None:
            clauses.append("source_identity_key = ?")
            params.append(identity or "")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(0, int(limit)))

        conn = sqlite3.connect(str(self.path))
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT record_id, key, kind, text, category, source_identity,
                   workspace_scope, schema_refs_json, catalog_refs_json,
                   aliases_json, confidence, importance, active, stale,
                   expires_at, metadata_json, created_at, updated_at
            FROM db_memory_records
            {where}
            ORDER BY importance DESC, updated_at DESC
            LIMIT ?
            """,
            params,
        )
        rows = cursor.fetchall()
        conn.close()
        return [_row_to_result(row) for row in rows]

    async def recall_db_records(
        self,
        query: str,
        *,
        limit: int = 5,
        score_threshold: float = 0.45,
        source_identity: str | None = None,
        kinds: list[str] | tuple[str, ...] | set[str] | None = None,
        category: str = DB_SEMANTIC_CATEGORY,
    ) -> list[dict[str, Any]]:
        identity = source_identity
        if identity is None:
            identity = self.default_source_identity
        clauses = [
            "category = ?",
            "active = 1",
            "stale = 0",
            "workspace_scope = 'source'",
            "(expires_at IS NULL OR expires_at = '' OR expires_at > ?)",
        ]
        params: list[Any] = [category, _utc_now()]
        if identity:
            clauses.append("source_identity_key = ?")
            params.append(identity)
        if kinds:
            allowed = [str(kind) for kind in kinds]
            placeholders = ",".join("?" * len(allowed))
            clauses.append(f"kind IN ({placeholders})")
            params.extend(allowed)

        conn = sqlite3.connect(str(self.path))
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT record_id, key, kind, text, category, source_identity,
                   workspace_scope, schema_refs_json, catalog_refs_json,
                   aliases_json, confidence, importance, active, stale,
                   expires_at, metadata_json, created_at, updated_at,
                   lexical_document
            FROM db_memory_records
            WHERE {' AND '.join(clauses)}
            """,
            params,
        )
        rows = cursor.fetchall()
        record_ids = [str(row[0]) for row in rows]
        fts_scores = _fts_scores(cursor, query, record_ids)
        candidate_ids = set(fts_scores)
        conn.close()

        results = []
        for row in rows:
            result = _row_to_result(row[:-1])
            record = result["metadata"]["db_memory"]
            record_id = str(result["record_id"])
            structured_candidate = _has_structured_query_match(record, query)
            if record_id not in candidate_ids and not structured_candidate:
                continue
            score, breakdown = _score_record(
                record,
                query,
                fts_scores.get(record_id),
                updated_at=record.get("updated_at"),
            )
            if score < score_threshold:
                continue
            result["score"] = score
            result["relevance_score"] = score
            result["score_breakdown"] = breakdown
            results.append(result)

        results.sort(
            key=lambda item: (
                item.get("score", 0.0),
                item.get("metadata", {}).get("db_memory", {}).get("importance", 0.0),
                item.get("metadata", {}).get("db_memory", {}).get("updated_at", ""),
            ),
            reverse=True,
        )
        return results[: max(0, int(limit))]

    def _ensure_schema(self) -> None:
        conn = sqlite3.connect(str(self.path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS db_memory_records (
                record_id TEXT PRIMARY KEY,
                source_identity_key TEXT NOT NULL,
                key TEXT NOT NULL,
                kind TEXT NOT NULL,
                text TEXT NOT NULL,
                category TEXT NOT NULL,
                source_identity TEXT,
                workspace_scope TEXT NOT NULL,
                schema_refs_json TEXT NOT NULL,
                catalog_refs_json TEXT NOT NULL,
                aliases_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                importance REAL NOT NULL,
                active INTEGER NOT NULL,
                stale INTEGER NOT NULL,
                expires_at TEXT,
                metadata_json TEXT NOT NULL,
                lexical_document TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(source_identity_key, key)
            )
            """)
        _migrate_record_columns(cursor)
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_db_memory_unique_identity_key
            ON db_memory_records(source_identity_key, key)
            """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_db_memory_scope
            ON db_memory_records(source_identity_key, category, kind, active, stale)
            """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_db_memory_workspace
            ON db_memory_records(source_identity_key, workspace_scope)
            """)
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS db_memory_fts
            USING fts5(
                record_id UNINDEXED,
                key,
                kind,
                text,
                aliases,
                schema_refs,
                catalog_refs,
                lexical_document,
                tokenize='unicode61'
            )
            """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS db_memory_schema_refs (
                record_id TEXT NOT NULL,
                source_identity_key TEXT NOT NULL,
                source_identity TEXT,
                ref TEXT NOT NULL,
                ref_key TEXT NOT NULL,
                PRIMARY KEY(record_id, ref_key)
            )
            """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS db_memory_catalog_refs (
                record_id TEXT NOT NULL,
                source_identity_key TEXT NOT NULL,
                source_identity TEXT,
                ref TEXT NOT NULL,
                ref_key TEXT NOT NULL,
                PRIMARY KEY(record_id, ref_key)
            )
            """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS db_memory_aliases (
                record_id TEXT NOT NULL,
                source_identity_key TEXT NOT NULL,
                source_identity TEXT,
                alias TEXT NOT NULL,
                alias_key TEXT NOT NULL,
                PRIMARY KEY(record_id, alias_key)
            )
            """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_db_memory_schema_ref_lookup
            ON db_memory_schema_refs(source_identity_key, ref_key)
            """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_db_memory_catalog_ref_lookup
            ON db_memory_catalog_refs(source_identity_key, ref_key)
            """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_db_memory_alias_lookup
            ON db_memory_aliases(source_identity_key, alias_key)
            """)
        _backfill_index_rows(cursor)
        conn.commit()
        conn.close()


def _migrate_record_columns(cursor: sqlite3.Cursor) -> None:
    """Add Phase 3.2 columns to Phase 3.1 structured DB memory files."""
    cursor.execute("PRAGMA table_info(db_memory_records)")
    columns = {str(row[1]) for row in cursor.fetchall()}
    migrations = {
        "source_identity_key": "TEXT NOT NULL DEFAULT ''",
        "category": f"TEXT NOT NULL DEFAULT '{DB_SEMANTIC_CATEGORY}'",
        "source_identity": "TEXT",
        "workspace_scope": "TEXT NOT NULL DEFAULT 'source'",
        "schema_refs_json": "TEXT NOT NULL DEFAULT '[]'",
        "catalog_refs_json": "TEXT NOT NULL DEFAULT '[]'",
        "aliases_json": "TEXT NOT NULL DEFAULT '[]'",
        "confidence": "REAL NOT NULL DEFAULT 1.0",
        "importance": "REAL NOT NULL DEFAULT 0.7",
        "active": "INTEGER NOT NULL DEFAULT 1",
        "stale": "INTEGER NOT NULL DEFAULT 0",
        "expires_at": "TEXT",
        "metadata_json": "TEXT NOT NULL DEFAULT '{}'",
        "lexical_document": "TEXT NOT NULL DEFAULT ''",
        "created_at": "TEXT NOT NULL DEFAULT ''",
        "updated_at": "TEXT NOT NULL DEFAULT ''",
    }
    for column, definition in migrations.items():
        if column not in columns:
            cursor.execute(
                f"ALTER TABLE db_memory_records ADD COLUMN {column} {definition}"
            )

    now = _utc_now()
    if "source_identity_key" not in columns and "source_identity" in columns:
        cursor.execute("""
            UPDATE db_memory_records
            SET source_identity_key = COALESCE(source_identity, '')
            WHERE source_identity_key = ''
            """)
    cursor.execute(
        "UPDATE db_memory_records SET created_at = ? WHERE created_at = ''",
        (now,),
    )
    cursor.execute(
        "UPDATE db_memory_records SET updated_at = ? WHERE updated_at = ''",
        (now,),
    )
    cursor.execute("""
        UPDATE db_memory_records
        SET lexical_document = trim(
            COALESCE(kind, '') || ' ' ||
            COALESCE(key, '') || ' ' ||
            COALESCE(text, '') || ' ' ||
            COALESCE(schema_refs_json, '') || ' ' ||
            COALESCE(catalog_refs_json, '') || ' ' ||
            COALESCE(aliases_json, '')
        )
        WHERE lexical_document = ''
        """)


def _normalize_record(
    record: dict[str, Any], default_source_identity: str | None
) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    kind = str(record.get("kind") or "").strip()
    key = str(record.get("key") or "").strip()
    text = str(record.get("text") or record.get("content") or "").strip()
    source_identity = (
        record.get("source_identity")
        or metadata.get("source_identity")
        or default_source_identity
    )
    workspace_scope = (
        record.get("workspace_scope") or metadata.get("workspace_scope") or "source"
    )
    schema_refs = [
        ref
        for item in _list_value(
            record.get("schema_refs") or metadata.get("schema_refs")
        )
        for ref in [_schema_ref_string(item)]
        if ref
    ]
    catalog_refs = _list_value(
        record.get("catalog_refs") or metadata.get("catalog_refs")
    )
    catalog_refs.extend(
        ref
        for ref in (
            metadata.get("catalog_profile_ref"),
            metadata.get("catalog_evidence_id"),
            metadata.get("catalog_store_id"),
        )
        if ref and ref not in catalog_refs
    )
    aliases = _list_value(record.get("aliases") or metadata.get("aliases"))
    if metadata.get("alias"):
        alias = str(metadata["alias"])
        if alias not in aliases:
            aliases.append(alias)
    schema_ref = _schema_ref_from_metadata(metadata)
    if schema_ref and schema_ref not in schema_refs:
        schema_refs.append(schema_ref)

    active = record.get("active", metadata.get("active", True))
    stale = record.get("stale", metadata.get("stale", False))
    confidence = _confidence(record.get("confidence", metadata.get("confidence", 1.0)))
    importance = _float_clamped(record.get("importance", 0.7), 0.7)
    expires_at = record.get("expires_at") or metadata.get("expires_at")
    category = str(record.get("category") or DB_SEMANTIC_CATEGORY)
    normalized_source_identity = str(source_identity) if source_identity else None
    normalized_workspace_scope = str(workspace_scope or "source")
    normalized_schema_refs = [str(item) for item in schema_refs if str(item).strip()]
    normalized_catalog_refs = [str(item) for item in catalog_refs if str(item).strip()]
    normalized_aliases = [str(item) for item in aliases if str(item).strip()]
    metadata.setdefault("source_identity", normalized_source_identity)
    metadata.setdefault("workspace_scope", normalized_workspace_scope)
    metadata.setdefault("active", bool(active))
    metadata.setdefault("stale", bool(stale))
    metadata.setdefault("confidence", confidence)
    if normalized_schema_refs:
        metadata.setdefault("schema_refs", normalized_schema_refs)
    if normalized_catalog_refs:
        metadata.setdefault("catalog_refs", normalized_catalog_refs)
    if normalized_aliases:
        metadata.setdefault("aliases", normalized_aliases)

    normalized = {
        "kind": kind,
        "key": key,
        "text": text,
        "category": category,
        "source_identity": normalized_source_identity,
        "workspace_scope": normalized_workspace_scope,
        "schema_refs": normalized_schema_refs,
        "catalog_refs": normalized_catalog_refs,
        "aliases": normalized_aliases,
        "confidence": confidence,
        "importance": importance,
        "active": bool(active),
        "stale": bool(stale),
        "expires_at": str(expires_at) if expires_at else None,
        "metadata": metadata,
    }
    return normalized


def _row_to_result(row: tuple[Any, ...]) -> dict[str, Any]:
    (
        record_id,
        key,
        kind,
        text,
        category,
        source_identity,
        workspace_scope,
        schema_refs_json,
        catalog_refs_json,
        aliases_json,
        confidence,
        importance,
        active,
        stale,
        expires_at,
        metadata_json,
        created_at,
        updated_at,
    ) = row
    metadata = _json_object(metadata_json)
    record = {
        "kind": kind,
        "key": key,
        "text": text,
        "category": category,
        "source_identity": source_identity,
        "workspace_scope": workspace_scope,
        "schema_refs": _json_list(schema_refs_json),
        "catalog_refs": _json_list(catalog_refs_json),
        "aliases": _json_list(aliases_json),
        "confidence": float(confidence),
        "importance": float(importance),
        "active": bool(active),
        "stale": bool(stale),
        "expires_at": expires_at,
        "metadata": metadata,
        "created_at": created_at,
        "updated_at": updated_at,
    }
    metadata = {**metadata, "db_memory": record}
    return {
        "chunk_id": record_id,
        "record_id": record_id,
        "content": f"DB memory record:\n{json.dumps(record, sort_keys=True)}",
        "metadata": metadata,
        "score": 1.0,
        "relevance_score": 1.0,
        "source": "structured_db_memory",
    }


def _refresh_index_rows(
    cursor: sqlite3.Cursor,
    record_id: str,
    record: dict[str, Any],
    lexical_document: str,
) -> None:
    _delete_index_rows(cursor, [record_id])
    source_identity = record.get("source_identity")
    source_identity_key = str(source_identity or "")
    cursor.execute(
        """
        INSERT INTO db_memory_fts (
            record_id, key, kind, text, aliases, schema_refs, catalog_refs,
            lexical_document
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record_id,
            record.get("key") or "",
            record.get("kind") or "",
            record.get("text") or "",
            " ".join(record.get("aliases") or ()),
            " ".join(record.get("schema_refs") or ()),
            " ".join(record.get("catalog_refs") or ()),
            lexical_document,
        ),
    )
    for ref in record.get("schema_refs") or ():
        ref_value = str(ref).strip()
        if not ref_value:
            continue
        cursor.execute(
            """
            INSERT OR REPLACE INTO db_memory_schema_refs
            (record_id, source_identity_key, source_identity, ref, ref_key)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                record_id,
                source_identity_key,
                source_identity,
                ref_value,
                _ref_key(ref_value),
            ),
        )
    for ref in record.get("catalog_refs") or ():
        ref_value = str(ref).strip()
        if not ref_value:
            continue
        cursor.execute(
            """
            INSERT OR REPLACE INTO db_memory_catalog_refs
            (record_id, source_identity_key, source_identity, ref, ref_key)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                record_id,
                source_identity_key,
                source_identity,
                ref_value,
                _ref_key(ref_value),
            ),
        )
    for alias in record.get("aliases") or ():
        alias_value = str(alias).strip()
        if not alias_value:
            continue
        cursor.execute(
            """
            INSERT OR REPLACE INTO db_memory_aliases
            (record_id, source_identity_key, source_identity, alias, alias_key)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                record_id,
                source_identity_key,
                source_identity,
                alias_value,
                _ref_key(alias_value),
            ),
        )


def _delete_index_rows(cursor: sqlite3.Cursor, record_ids: list[str]) -> None:
    if not record_ids:
        return
    placeholders = ",".join("?" * len(record_ids))
    cursor.execute(
        f"DELETE FROM db_memory_fts WHERE record_id IN ({placeholders})", record_ids
    )
    cursor.execute(
        f"DELETE FROM db_memory_schema_refs WHERE record_id IN ({placeholders})",
        record_ids,
    )
    cursor.execute(
        f"DELETE FROM db_memory_catalog_refs WHERE record_id IN ({placeholders})",
        record_ids,
    )
    cursor.execute(
        f"DELETE FROM db_memory_aliases WHERE record_id IN ({placeholders})",
        record_ids,
    )


def _backfill_index_rows(cursor: sqlite3.Cursor) -> None:
    cursor.execute("""
        SELECT record_id, key, kind, text, category, source_identity,
               workspace_scope, schema_refs_json, catalog_refs_json,
               aliases_json, confidence, importance, active, stale, expires_at,
               metadata_json, created_at, updated_at, lexical_document
        FROM db_memory_records r
        WHERE NOT EXISTS (
            SELECT 1 FROM db_memory_fts f WHERE f.record_id = r.record_id
        )
        """)
    rows = cursor.fetchall()
    for row in rows:
        result = _row_to_result(row[:-1])
        record = result["metadata"]["db_memory"]
        lexical_document = str(row[-1] or _lexical_document(record))
        _refresh_index_rows(cursor, str(row[0]), record, lexical_document)


def _fts_scores(
    cursor: sqlite3.Cursor, query: str, eligible_record_ids: list[str]
) -> dict[str, dict[str, float]]:
    if not eligible_record_ids:
        return {}
    fts_query = _fts_query(query)
    if not fts_query:
        return {}
    cursor.execute(
        """
        SELECT record_id, bm25(db_memory_fts, 1.6, 1.0, 1.2, 1.5, 1.4, 1.3, 0.8)
        FROM db_memory_fts
        WHERE db_memory_fts MATCH ?
        """,
        (fts_query,),
    )
    eligible = set(eligible_record_ids)
    raw_scores = {
        str(record_id): float(bm25)
        for record_id, bm25 in cursor.fetchall()
        if str(record_id) in eligible
    }
    if not raw_scores:
        return {}
    best = min(raw_scores.values())
    worst = max(raw_scores.values())
    normalized: dict[str, dict[str, float]] = {}
    for record_id, raw in raw_scores.items():
        if best == worst:
            fts_normalized = 1.0
        else:
            fts_normalized = (worst - raw) / (worst - best)
        normalized[record_id] = {
            "fts_bm25": raw,
            "fts_normalized": max(0.0, min(1.0, float(fts_normalized))),
        }
    return normalized


def _fts_query(query: str) -> str:
    terms = list(dict.fromkeys(_tokens(query)))
    return " OR ".join(f'"{term}"' for term in terms[:24])


def _score_record(
    record: dict[str, Any],
    query: str,
    fts_score: dict[str, float] | None,
    *,
    updated_at: str | None = None,
) -> tuple[float, dict[str, Any]]:
    query_terms = _tokens(query)
    if not query_terms:
        return 0.0, {"fts_bm25": None, "fts_normalized": 0.0}
    query_term_set = set(query_terms)
    query_text = str(query or "").lower()
    key = str(record.get("key") or "").lower()
    aliases = [str(item).lower() for item in record.get("aliases") or ()]
    schema_refs = [str(item).lower() for item in record.get("schema_refs") or ()]
    catalog_refs = [str(item).lower() for item in record.get("catalog_refs") or ()]

    fts_bm25 = None if fts_score is None else fts_score.get("fts_bm25")
    fts_normalized = 0.0 if fts_score is None else fts_score.get("fts_normalized", 0.0)
    key_terms = set(_tokens(key))
    key_overlap = len(key_terms & query_term_set) / max(1, len(key_terms))
    alias_match = any(alias and alias in query_text for alias in aliases)
    schema_overlap = _ref_overlap(query_term_set, schema_refs)
    catalog_overlap = _ref_overlap(query_term_set, catalog_refs)
    exact_key = bool(key and key in query_text)

    confidence = _confidence(record.get("confidence", 1.0))
    importance = _float_clamped(record.get("importance", 0.7), 0.7)
    recency = _recency_score(updated_at)
    score = (
        (0.34 * float(fts_normalized or 0.0))
        + (0.12 * key_overlap)
        + (0.18 if exact_key else 0.0)
        + (0.16 if alias_match else 0.0)
        + (0.14 * schema_overlap)
        + (0.10 * catalog_overlap)
        + (0.08 * confidence)
        + (0.06 * importance)
        + (0.02 * recency)
    )
    score = min(1.0, float(score))
    return score, {
        "fts_bm25": fts_bm25,
        "fts_normalized": round(float(fts_normalized or 0.0), 4),
        "key_overlap": round(float(key_overlap), 4),
        "exact_key": exact_key,
        "alias_match": alias_match,
        "schema_ref_overlap": round(float(schema_overlap), 4),
        "catalog_ref_overlap": round(float(catalog_overlap), 4),
        "confidence": confidence,
        "importance": importance,
        "recency": round(float(recency), 4),
    }


def _has_structured_query_match(record: dict[str, Any], query: str) -> bool:
    query_terms = set(_tokens(query))
    if not query_terms:
        return False
    query_text = str(query or "").lower()
    key = str(record.get("key") or "").lower()
    if key and (key in query_text or set(_tokens(key)) & query_terms):
        return True
    for alias in record.get("aliases") or ():
        alias_text = str(alias).lower()
        if alias_text and (
            alias_text in query_text or set(_tokens(alias_text)) & query_terms
        ):
            return True
    return any(
        _ref_overlap(query_terms, [str(ref)]) > 0
        for ref in (
            *(record.get("schema_refs") or ()),
            *(record.get("catalog_refs") or ()),
        )
    )


def _lexical_document(record: dict[str, Any]) -> str:
    raw_metadata = record.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    parts = [
        record.get("kind"),
        record.get("key"),
        record.get("text"),
        *(record.get("aliases") or ()),
        *(record.get("schema_refs") or ()),
        *(record.get("catalog_refs") or ()),
        metadata.get("table"),
        metadata.get("column"),
        metadata.get("metric"),
        metadata.get("unit"),
    ]
    return " ".join(str(part) for part in parts if part)


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(str(text or ""))]


def _ref_overlap(query_terms: set[str], refs: list[str]) -> float:
    if not refs:
        return 0.0
    ref_terms = set()
    for ref in refs:
        ref_terms.update(_tokens(ref))
    if not ref_terms:
        return 0.0
    return len(ref_terms & query_terms) / len(ref_terms)


def _ref_key(value: str) -> str:
    return " ".join(_tokens(value))


def _recency_score(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return 0.0
    age_seconds = max(0.0, (_now_dt() - parsed).total_seconds())
    age_days = age_seconds / 86400.0
    return max(0.0, min(1.0, 1.0 - (age_days / 365.0)))


def _record_id(source_identity: str, key: str) -> str:
    digest = hashlib.sha256(f"{source_identity}\0{key}".encode()).hexdigest()[:24]
    return f"dbmem:{digest}"


def _utc_now() -> str:
    return _now_dt().isoformat()


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _json_object(value: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_list(value: str | None) -> list[Any]:
    try:
        parsed = json.loads(value or "[]")
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _list_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _schema_ref_from_metadata(metadata: dict[str, Any]) -> str | None:
    table = str(metadata.get("table") or "").strip()
    column = str(metadata.get("column") or "").strip()
    if table and column:
        return f"{table}.{column}"
    return table or None


def _schema_ref_string(value: Any) -> str | None:
    if isinstance(value, dict):
        table = str(value.get("table") or "").strip()
        column = str(value.get("column") or "").strip()
        if table and column:
            return f"{table}.{column}"
        return table or None
    rendered = str(value or "").strip()
    return rendered or None


def _confidence(value: Any) -> float:
    if isinstance(value, str):
        mapped = {"low": 0.35, "medium": 0.65, "high": 0.9}
        if value.lower() in mapped:
            return mapped[value.lower()]
    return _float_clamped(value, 1.0)


def _float_clamped(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))
