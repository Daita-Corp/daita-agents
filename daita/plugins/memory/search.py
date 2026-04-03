"""
Vector search using SQLite for local development.

Uses embeddings for semantic search over memory files.
Supports importance-weighted scoring and temporal decay.
"""

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .metadata import MemoryMetadata
from .utils import (
    build_where_clause,
    generate_chunk_id,
    merge_metadata_json,
    parse_metadata_json,
)


@dataclass
class SearchResult:
    content: str
    file_path: str
    score: float
    line_start: int
    line_end: int
    chunk_id: str
    metadata: Optional[Dict[str, Any]] = None
    raw_semantic_score: Optional[float] = None


class SQLiteVectorSearch:
    """Local vector search using SQLite with importance-weighted, temporally-decayed scoring."""

    def __init__(
        self,
        db_path: Path,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
    ):
        self.db_path = db_path
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self._embed_cache: Dict[str, List[float]] = {}

        if embedding_provider == "openai":
            from openai import AsyncOpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable required for embeddings"
                )
            self.embedder = AsyncOpenAI(api_key=api_key)
        else:
            raise NotImplementedError(
                f"Provider {embedding_provider} not yet supported"
            )

        self._init_database()

    def _init_database(self):
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                content TEXT NOT NULL,
                line_start INTEGER,
                line_end INTEGER,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id TEXT PRIMARY KEY,
                embedding TEXT NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
            )
        """)

        # Indexes for common filter patterns
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON chunks(file_path)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_category ON chunks(json_extract(metadata, '$.category'))"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_created_at ON chunks(json_extract(metadata, '$.created_at'))"
        )

        # Migration: add norm column for pre-computed vector norms
        try:
            cursor.execute("ALTER TABLE embeddings ADD COLUMN norm REAL")
        except sqlite3.OperationalError:
            pass  # Column already exists

        conn.commit()
        conn.close()

    async def embed_text(self, text: str) -> List[float]:
        if text in self._embed_cache:
            return self._embed_cache[text]
        response = await self.embedder.embeddings.create(
            model=self.embedding_model, input=text
        )
        embedding = response.data[0].embedding
        self._embed_cache[text] = embedding
        return embedding

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Batch embed multiple texts in a single API call.

        Returns embeddings in the same order as the input texts.
        Leverages the cache for already-seen texts.
        """
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            if text in self._embed_cache:
                results[i] = self._embed_cache[text]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            response = await self.embedder.embeddings.create(
                model=self.embedding_model, input=uncached_texts
            )
            for j, data in enumerate(response.data):
                idx = uncached_indices[j]
                self._embed_cache[uncached_texts[j]] = data.embedding
                results[idx] = data.embedding

        return results

    async def store_chunk(
        self,
        content: str,
        metadata: MemoryMetadata,
        chunk_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a single chunk directly into the vector DB.

        This is the primary write path. Checks for existence before embedding
        to avoid redundant API calls (incremental indexing).

        Args:
            extra_metadata: Optional dict merged into the metadata JSON before
                            storage. Used to attach e.g. "extracted_facts".

        Returns:
            chunk_id of the stored chunk
        """
        if chunk_id is None:
            chunk_id = generate_chunk_id(content)

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Incremental: skip if already exists with same content
        cursor.execute("SELECT chunk_id FROM chunks WHERE chunk_id = ?", (chunk_id,))
        if cursor.fetchone():
            conn.close()
            return chunk_id

        conn.close()

        # Generate embedding only for new chunks
        embedding = await self.embed_text(content)

        metadata_json = merge_metadata_json(metadata, extra_metadata)

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO chunks (chunk_id, file_path, content, line_start, line_end, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (chunk_id, "memory://direct", content, 0, 0, metadata_json),
        )

        norm = float(np.linalg.norm(embedding))
        cursor.execute(
            """
            INSERT INTO embeddings (chunk_id, embedding, norm)
            VALUES (?, ?, ?)
        """,
            (chunk_id, json.dumps(embedding), norm),
        )

        conn.commit()
        conn.close()
        return chunk_id

    async def store_chunks_batch(
        self,
        items: List[dict],
    ) -> List[dict]:
        """Store multiple chunks in a single batch with one embedding API call.

        Args:
            items: List of dicts with keys: content, metadata (MemoryMetadata),
                   and optional extra_metadata and chunk_id.

        Returns:
            List of dicts with chunk_id and status ("stored" or "skipped").
        """
        prepared = []
        for item in items:
            cid = item.get("chunk_id") or generate_chunk_id(item["content"])
            prepared.append({**item, "chunk_id": cid})

        # Check existing in one query
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        all_cids = [p["chunk_id"] for p in prepared]
        placeholders = ",".join("?" * len(all_cids))
        cursor.execute(
            f"SELECT chunk_id FROM chunks WHERE chunk_id IN ({placeholders})",
            all_cids,
        )
        existing_ids = {row[0] for row in cursor.fetchall()}
        conn.close()

        results = [
            {
                "chunk_id": p["chunk_id"],
                "status": "skipped" if p["chunk_id"] in existing_ids else "stored",
            }
            for p in prepared
        ]
        new_items = [p for p in prepared if p["chunk_id"] not in existing_ids]

        if not new_items:
            return results

        # Batch embed all new items
        texts = [item["content"] for item in new_items]
        embeddings = await self.embed_texts(texts)

        # Batch insert
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        for i, item in enumerate(new_items):
            metadata_json = merge_metadata_json(
                item["metadata"], item.get("extra_metadata")
            )
            cursor.execute(
                "INSERT INTO chunks (chunk_id, file_path, content, line_start, line_end, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (item["chunk_id"], "memory://direct", item["content"], 0, 0, metadata_json),
            )
            norm = float(np.linalg.norm(embeddings[i]))
            cursor.execute(
                "INSERT INTO embeddings (chunk_id, embedding, norm) VALUES (?, ?, ?)",
                (item["chunk_id"], json.dumps(embeddings[i]), norm),
            )
        conn.commit()
        conn.close()

        return results

    async def index_file(
        self,
        file_path: str,
        chunk_size: int = 400,
        metadata: Optional[MemoryMetadata] = None,
    ):
        """
        Index a file by chunking and embedding. Incremental: skips unchanged chunks.
        Used by the curator when re-indexing MEMORY.md.
        """
        from .chunking import chunk_markdown

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = chunk_markdown(content, chunk_size)

        if metadata is None:
            metadata = MemoryMetadata(
                content=content[:100], importance=0.5, source="agent_inferred"
            )

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        for chunk in chunks:
            chunk_id = hashlib.md5(
                f"{file_path}:{chunk.start_line}:{chunk.content[:50]}".encode()
            ).hexdigest()

            # Incremental: skip existing chunks
            cursor.execute(
                "SELECT chunk_id FROM chunks WHERE chunk_id = ?", (chunk_id,)
            )
            if cursor.fetchone():
                continue

            embedding = await self.embed_text(chunk.content)

            chunk_metadata = MemoryMetadata(
                content=chunk.content,
                importance=metadata.importance,
                source=metadata.source,
                category=metadata.category,
                created_at=metadata.created_at,
                pinned=metadata.pinned,
            )

            cursor.execute(
                """
                INSERT INTO chunks (chunk_id, file_path, content, line_start, line_end, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    chunk_id,
                    file_path,
                    chunk.content,
                    chunk.start_line,
                    chunk.end_line,
                    chunk_metadata.to_json(),
                ),
            )

            cursor.execute(
                """
                INSERT INTO embeddings (chunk_id, embedding)
                VALUES (?, ?)
            """,
                (chunk_id, json.dumps(embedding)),
            )

        conn.commit()
        conn.close()

    def track_access(self, chunk_ids: List[str]):
        """
        Increment access_count and update last_accessed for recalled chunks.
        Uses atomic SQL to avoid read-modify-write race conditions.
        """
        if not chunk_ids:
            return

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        for chunk_id in chunk_ids:
            cursor.execute(
                """
                UPDATE chunks SET metadata = json_set(
                    metadata,
                    '$.access_count', COALESCE(json_extract(metadata, '$.access_count'), 0) + 1,
                    '$.last_accessed', ?
                ) WHERE chunk_id = ? AND metadata IS NOT NULL
                """,
                (now, chunk_id),
            )

        conn.commit()
        conn.close()

    def _apply_score_adjustments(
        self, base_score: float, metadata_dict: Optional[Dict]
    ) -> float:
        """
        Apply importance boost and temporal decay to a base relevance score.

        - Importance boost: +/- 0.1 based on distance from 0.5
        - Temporal decay: floors at 0.7 after 1 year (pinned memories exempt)
        """
        if not metadata_dict:
            return base_score

        # Importance boost: range [-0.1, +0.1]
        importance = metadata_dict.get("importance", 0.5)
        importance_boost = (importance - 0.5) * 0.2

        # Temporal decay (pinned memories are exempt)
        decay = 1.0
        if not metadata_dict.get("pinned", False):
            created_at_str = metadata_dict.get("created_at")
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                    age_days = (datetime.now() - created_at).days
                    decay = max(0.7, 1.0 - (age_days / 365) * 0.3)
                except Exception:
                    pass

        return min((base_score + importance_boost) * decay, 1.0)

    async def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.7,
        category: Optional[str] = None,
        since: Optional[str] = None,
        before: Optional[str] = None,
    ) -> List[SearchResult]:
        """Semantic search with importance weighting and temporal decay.

        Args:
            since: Only return memories created at or after this datetime (ISO format).
            before: Only return memories created before this datetime (ISO format).
        """
        query_embedding = await self.embed_text(query)
        query_vec = np.array(query_embedding)

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        where_sql, params = build_where_clause(
            category=category, since=since, before=before, table_alias="c"
        )
        cursor.execute(
            f"""
            SELECT c.chunk_id, c.file_path, c.content, c.line_start, c.line_end,
                   c.metadata, e.embedding, e.norm
            FROM chunks c
            JOIN embeddings e ON c.chunk_id = e.chunk_id
            {where_sql}
            """,
            params,
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return []

        # Batch vectorized cosine similarity
        embedding_jsons = [r[6] for r in rows]
        stored_norms = [r[7] for r in rows]
        all_vecs = np.array([json.loads(e) for e in embedding_jsons])
        query_norm = np.linalg.norm(query_vec)
        all_norms = np.array([
            n if n is not None else float(np.linalg.norm(v))
            for n, v in zip(stored_norms, all_vecs)
        ])
        denoms = query_norm * all_norms
        similarities = np.where(denoms > 0, all_vecs @ query_vec / denoms, 0.0)

        # Apply score adjustments and filter
        results = []
        for i, row in enumerate(rows):
            chunk_id, file_path, content, line_start, line_end, metadata_json = row[:6]
            metadata_dict = parse_metadata_json(metadata_json)
            adjusted = self._apply_score_adjustments(
                float(similarities[i]), metadata_dict
            )
            if adjusted >= score_threshold:
                results.append(
                    SearchResult(
                        content=content,
                        file_path=file_path,
                        score=adjusted,
                        line_start=line_start,
                        line_end=line_end,
                        chunk_id=chunk_id,
                        metadata=metadata_dict,
                        raw_semantic_score=float(similarities[i]),
                    )
                )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
