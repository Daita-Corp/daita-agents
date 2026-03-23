"""
Vector search using SQLite for local development.

Uses embeddings for semantic search over memory files.
Supports importance-weighted scoring and temporal decay.
"""

import sqlite3
import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
from datetime import datetime
from .metadata import MemoryMetadata


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

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON chunks(file_path)")
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
            # Content-based hash only (no timestamp) so identical content
            # submitted in the same session produces the same chunk_id and is
            # caught by the incremental-check below instead of creating a duplicate.
            chunk_id = hashlib.md5(f"direct:{content.strip()}".encode()).hexdigest()

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

        # Build final metadata JSON, merging any extra fields
        if extra_metadata:
            metadata_dict = json.loads(metadata.to_json())
            metadata_dict.update(extra_metadata)
            metadata_json = json.dumps(metadata_dict)
        else:
            metadata_json = metadata.to_json()

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO chunks (chunk_id, file_path, content, line_start, line_end, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (chunk_id, "memory://direct", content, 0, 0, metadata_json),
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
        return chunk_id

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
        Used to build usage signals for smarter pruning over time.
        """
        if not chunk_ids:
            return

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        for chunk_id in chunk_ids:
            cursor.execute(
                "SELECT metadata FROM chunks WHERE chunk_id = ?", (chunk_id,)
            )
            row = cursor.fetchone()
            if row and row[0]:
                try:
                    metadata_dict = json.loads(row[0])
                    metadata_dict["access_count"] = (
                        metadata_dict.get("access_count", 0) + 1
                    )
                    metadata_dict["last_accessed"] = now
                    cursor.execute(
                        "UPDATE chunks SET metadata = ? WHERE chunk_id = ?",
                        (json.dumps(metadata_dict), chunk_id),
                    )
                except Exception:
                    pass

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
    ) -> List[SearchResult]:
        """Semantic search with importance weighting and temporal decay."""
        query_embedding = await self.embed_text(query)
        query_vec = np.array(query_embedding)

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        if category:
            cursor.execute(
                """
                SELECT c.chunk_id, c.file_path, c.content, c.line_start, c.line_end, c.metadata, e.embedding
                FROM chunks c
                JOIN embeddings e ON c.chunk_id = e.chunk_id
                WHERE json_extract(c.metadata, '$.category') = ?
            """,
                (category,),
            )
        else:
            cursor.execute("""
                SELECT c.chunk_id, c.file_path, c.content, c.line_start, c.line_end, c.metadata, e.embedding
                FROM chunks c
                JOIN embeddings e ON c.chunk_id = e.chunk_id
            """)

        results = []
        for row in cursor.fetchall():
            (
                chunk_id,
                file_path,
                content,
                line_start,
                line_end,
                metadata_json,
                embedding_json,
            ) = row

            metadata_dict = None
            if metadata_json:
                try:
                    metadata_dict = json.loads(metadata_json)
                except json.JSONDecodeError:
                    metadata_dict = {
                        "importance": 0.5,
                        "source": "agent_inferred",
                        "pinned": False,
                    }

            chunk_vec = np.array(json.loads(embedding_json))
            denom = np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec)
            similarity = (
                float(np.dot(query_vec, chunk_vec) / denom) if denom > 0 else 0.0
            )

            adjusted = self._apply_score_adjustments(similarity, metadata_dict)

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
                        raw_semantic_score=similarity,
                    )
                )

        conn.close()
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
