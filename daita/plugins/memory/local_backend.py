"""
Local file-based memory backend.

Storage hierarchy:
- Project-scoped (default): .daita/memory/workspaces/{workspace}/
- Global (opt-in): ~/.daita/memory/workspaces/{workspace}/

Design: vector DB is the source of truth for recall. MEMORY.md is a
curator-generated human-readable summary, never written by the agent directly.
"""

import json
import aiofiles
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from .metadata import MemoryMetadata


def _find_project_root(start_path: Optional[Path] = None) -> Optional[Path]:
    current = start_path or Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "daita-project.yaml").exists():
            return parent
    return None


class LocalMemoryBackend:
    """
    Memory backend using local files + SQLite vectors.

    Write paths:
    - agent.remember() -> daily log + vector DB (immediate recall)
    - curator -> MEMORY.md + vector DB (structured long-term)

    Read paths:
    - recall() -> vector DB (semantic + keyword hybrid)
    - read_memory() -> raw file contents
    """

    def __init__(
        self,
        workspace: str,
        agent_id: Optional[str] = None,
        scope: str = "project",
        base_dir: Optional[Path] = None,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        max_chunks: int = 2000
    ):
        self.workspace = workspace
        self.agent_id = agent_id
        self.scope = scope

        if base_dir is None:
            if scope == "global":
                base_dir = Path.home() / ".daita" / "memory" / "workspaces"
            else:
                project_root = _find_project_root()
                base_dir = (project_root or Path.cwd()) / ".daita" / "memory" / "workspaces"

        self.workspace_dir = base_dir / workspace
        self.logs_dir = self.workspace_dir / "logs"
        self.memory_file = self.workspace_dir / "MEMORY.md"
        self.vector_db = self.workspace_dir / "vectors.db"

        self.logs_dir.mkdir(parents=True, exist_ok=True)

        from .storage import FileStorage
        from .search import SQLiteVectorSearch

        self.max_chunks = max_chunks
        self.storage = FileStorage(self.workspace_dir, agent_id=agent_id)
        self.search = SQLiteVectorSearch(self.vector_db, embedding_provider, embedding_model)

    async def remember(
        self,
        content: str,
        category: Optional[str] = None,
        metadata: Optional[MemoryMetadata] = None
    ) -> Dict[str, Any]:
        """
        Store a memory: appends to today's daily log and indexes in the vector DB.
        MEMORY.md is managed exclusively by the curator.
        """
        # Always write to daily log (human-readable history + curator input)
        await self.storage.append_to_daily_log(content, category)

        # Store directly in vector DB for immediate recall
        if metadata is None:
            metadata = MemoryMetadata(
                content=content,
                importance=0.5,
                source='agent_inferred',
                category=category
            )

        chunk_id = await self.search.store_chunk(content, metadata)

        return {
            "status": "success",
            "message": "Memory stored and indexed",
            "chunk_id": chunk_id,
            "indexed": True
        }

    async def update_memory(
        self,
        query: str,
        new_content: str,
        importance: float = 0.5
    ) -> Dict[str, Any]:
        """
        Find memories matching query, delete them, and store updated content.
        Use when a fact has changed, been resolved, or needs correction.
        """
        import sqlite3

        matches = await self.recall(query, limit=5, score_threshold=0.7)
        if not matches:
            return {"status": "not_found", "updated": 0, "message": "No matching memories found"}

        chunk_ids = [m['chunk_id'] for m in matches if m.get('chunk_id')]
        if not chunk_ids:
            return {"status": "not_found", "updated": 0, "message": "No matching memories with valid chunk IDs found"}

        conn = sqlite3.connect(str(self.vector_db))
        cursor = conn.cursor()
        placeholders = ','.join('?' * len(chunk_ids))
        cursor.execute(f"DELETE FROM embeddings WHERE chunk_id IN ({placeholders})", chunk_ids)
        cursor.execute(f"DELETE FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids)
        conn.commit()
        conn.close()

        metadata = MemoryMetadata(
            content=new_content,
            importance=importance,
            source='agent_inferred'
        )
        await self.search.store_chunk(new_content, metadata)

        return {
            "status": "success",
            "updated": len(chunk_ids),
            "message": f"Replaced {len(chunk_ids)} memories with updated content"
        }

    async def recall(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.6,
        strategy: str = "hybrid",
        min_importance: Optional[float] = None,
        max_importance: Optional[float] = None,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search memories using hybrid semantic + keyword search.
        Results are importance-weighted and temporally decayed.
        """
        if strategy == "semantic":
            results = await self._semantic_recall(query, limit, score_threshold, category)
        elif strategy == "keyword":
            results = await self._keyword_recall(query, limit, score_threshold, category)
        else:
            results = await self._hybrid_recall(query, limit, score_threshold, category)

        if min_importance is not None or max_importance is not None:
            results = self._filter_by_importance(results, min_importance, max_importance)

        return results

    async def _semantic_recall(self, query: str, limit: int, score_threshold: float, category: Optional[str] = None) -> List[Dict]:
        results = await self.search.search(query, limit, score_threshold, category=category)
        return [
            {
                "content": r.content,
                "file": r.file_path,
                "score": r.score,
                "relevance_score": r.score,
                "raw_semantic_score": r.raw_semantic_score if r.raw_semantic_score is not None else r.score,
                "lines": f"{r.line_start}-{r.line_end}",
                "chunk_id": r.chunk_id,
                "metadata": r.metadata or {}
            }
            for r in results
        ]

    async def _keyword_recall(self, query: str, limit: int, score_threshold: float, category: Optional[str] = None) -> List[Dict]:
        from .keyword_search import BM25Scorer
        from .text_utils import extract_keywords
        import sqlite3

        keywords = extract_keywords(query)

        conn = sqlite3.connect(str(self.vector_db))
        cursor = conn.cursor()
        if category:
            cursor.execute(
                "SELECT chunk_id, file_path, content, line_start, line_end, metadata FROM chunks"
                " WHERE json_extract(metadata, '$.category') = ?",
                (category,)
            )
        else:
            cursor.execute("SELECT chunk_id, file_path, content, line_start, line_end, metadata FROM chunks")
        chunks = cursor.fetchall()
        conn.close()

        if not chunks:
            return []

        documents = [c[2] for c in chunks]
        bm25 = BM25Scorer(documents)

        results = []
        for chunk_id, file_path, content, line_start, line_end, metadata_json in chunks:
            metadata_dict = {}
            if metadata_json:
                try:
                    metadata_dict = json.loads(metadata_json)
                except json.JSONDecodeError:
                    metadata_dict = {'importance': 0.5, 'source': 'agent_inferred', 'pinned': False}

            raw_score = bm25.score(keywords, content)
            keyword_score = bm25.normalize_score(raw_score)
            adjusted = self.search._apply_score_adjustments(keyword_score, metadata_dict)

            if adjusted >= score_threshold:
                results.append({
                    "content": content,
                    "file": file_path,
                    "score": float(adjusted),
                    "relevance_score": float(adjusted),
                    "lines": f"{line_start}-{line_end}",
                    "chunk_id": chunk_id,
                    "metadata": metadata_dict
                })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]

    async def _hybrid_recall(self, query: str, limit: int, score_threshold: float, category: Optional[str] = None) -> List[Dict]:
        from .keyword_search import BM25Scorer
        from .text_utils import extract_keywords, contains_exact_phrase
        import sqlite3
        import numpy as np

        keywords = extract_keywords(query)

        conn = sqlite3.connect(str(self.vector_db))
        cursor = conn.cursor()
        if category:
            cursor.execute("""
                SELECT c.chunk_id, c.file_path, c.content, c.line_start, c.line_end, c.metadata, e.embedding
                FROM chunks c
                JOIN embeddings e ON c.chunk_id = e.chunk_id
                WHERE json_extract(c.metadata, '$.category') = ?
            """, (category,))
        else:
            cursor.execute("""
                SELECT c.chunk_id, c.file_path, c.content, c.line_start, c.line_end, c.metadata, e.embedding
                FROM chunks c
                JOIN embeddings e ON c.chunk_id = e.chunk_id
            """)
        chunks = cursor.fetchall()
        conn.close()

        if not chunks:
            return []

        documents = [c[2] for c in chunks]
        bm25 = BM25Scorer(documents)
        query_embedding = await self.search.embed_text(query)
        query_vec = np.array(query_embedding)

        results = []
        for chunk_id, file_path, content, line_start, line_end, metadata_json, embedding_json in chunks:
            metadata_dict = {}
            if metadata_json:
                try:
                    metadata_dict = json.loads(metadata_json)
                except json.JSONDecodeError:
                    metadata_dict = {'importance': 0.5, 'source': 'agent_inferred', 'pinned': False}

            chunk_vec = np.array(json.loads(embedding_json))
            denom = np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec)
            semantic_score = float(np.dot(query_vec, chunk_vec) / denom) if denom > 0 else 0.0

            keyword_raw = bm25.score(keywords, content)
            keyword_score = bm25.normalize_score(keyword_raw)

            base_score = (0.6 * semantic_score) + (0.4 * keyword_score)
            phrase_bonus = 0.15 if contains_exact_phrase(query, content) else 0.0
            consensus_bonus = 0.10 if semantic_score > 0.5 and keyword_score > 0.5 else 0.0

            raw_score = min(base_score + phrase_bonus + consensus_bonus, 1.0)
            adjusted = self.search._apply_score_adjustments(raw_score, metadata_dict)

            if adjusted >= score_threshold:
                results.append({
                    "content": content,
                    "file": file_path,
                    "score": float(adjusted),
                    "relevance_score": float(adjusted),
                    "raw_semantic_score": float(semantic_score),
                    "lines": f"{line_start}-{line_end}",
                    "chunk_id": chunk_id,
                    "metadata": metadata_dict,
                    "score_breakdown": {
                        "semantic": float(semantic_score),
                        "keyword": float(keyword_score),
                        "phrase_bonus": float(phrase_bonus),
                        "consensus_bonus": float(consensus_bonus)
                    }
                })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]

    async def list_by_category(
        self,
        category: str,
        min_importance: float = 0.0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Direct category dump — no embedding call, no semantic ranking."""
        import sqlite3
        conn = sqlite3.connect(str(self.vector_db))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT chunk_id, content, metadata FROM chunks"
            " WHERE json_extract(metadata, '$.category') = ?"
            " AND (json_extract(metadata, '$.importance') IS NULL"
            "      OR json_extract(metadata, '$.importance') >= ?)"
            " ORDER BY json_extract(metadata, '$.importance') DESC"
            " LIMIT ?",
            (category, min_importance, limit)
        )
        rows = cursor.fetchall()
        conn.close()

        results = []
        for chunk_id, content, metadata_json in rows:
            metadata_dict = {}
            if metadata_json:
                try:
                    metadata_dict = json.loads(metadata_json)
                except Exception:
                    pass
            results.append({"chunk_id": chunk_id, "content": content, "metadata": metadata_dict})
        return results

    async def regenerate_memory_md(self, min_importance: float = 0.4) -> str:
        """
        Regenerate MEMORY.md from the vector DB. Curator-only write path.

        Groups memories by category, sorts by importance descending.
        This replaces the old append-only approach - MEMORY.md is always
        a clean, current snapshot of what's worth remembering.
        """
        import sqlite3

        conn = sqlite3.connect(str(self.vector_db))
        cursor = conn.cursor()
        cursor.execute("SELECT content, metadata FROM chunks WHERE file_path = 'memory://direct' OR file_path LIKE '%MEMORY%'")
        rows = cursor.fetchall()
        conn.close()

        by_category: Dict[str, List] = {}
        for content, metadata_json in rows:
            if not metadata_json:
                continue
            try:
                m = json.loads(metadata_json)
                importance = m.get('importance', 0.5)
                if importance < min_importance:
                    continue
                category = m.get('category') or 'General'
                by_category.setdefault(category, []).append((importance, content.strip()))
            except Exception:
                continue

        lines = ["# Long-Term Memory\n"]
        for category in sorted(by_category.keys()):
            entries = sorted(by_category[category], reverse=True)
            lines.append(f"\n## {category.title()}\n")
            for _importance, content in entries:
                lines.append(f"- {content}\n")

        async with aiofiles.open(self.memory_file, 'w', encoding='utf-8') as f:
            await f.write(''.join(lines))

        return str(self.memory_file)

    async def update_chunk_metadata(self, chunk_id: str, updates: dict):
        """Update metadata fields on a stored chunk."""
        import sqlite3
        conn = sqlite3.connect(str(self.vector_db))
        cursor = conn.cursor()
        cursor.execute("SELECT metadata FROM chunks WHERE chunk_id = ?", (chunk_id,))
        row = cursor.fetchone()
        if row and row[0]:
            try:
                meta = json.loads(row[0])
                meta.update(updates)
                cursor.execute(
                    "UPDATE chunks SET metadata = ? WHERE chunk_id = ?",
                    (json.dumps(meta), chunk_id)
                )
            except Exception:
                pass
        conn.commit()
        conn.close()

    async def delete_chunks(self, chunk_ids: list):
        """Hard-delete chunks from the local vector DB."""
        if not chunk_ids:
            return
        import sqlite3
        conn = sqlite3.connect(str(self.vector_db))
        cursor = conn.cursor()
        placeholders = ','.join('?' * len(chunk_ids))
        cursor.execute(f"DELETE FROM embeddings WHERE chunk_id IN ({placeholders})", chunk_ids)
        cursor.execute(f"DELETE FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids)
        conn.commit()
        conn.close()

    async def read_memory_md(self) -> str:
        """Return MEMORY.md content."""
        if self.memory_file.exists():
            async with aiofiles.open(self.memory_file, 'r', encoding='utf-8') as f:
                return await f.read()
        return "# Long-Term Memory\n\n(No memories yet)"

    async def read_today_log(self) -> str:
        """Return today's daily log content."""
        from datetime import date
        today = date.today().isoformat()
        today_log = self.logs_dir / f"{today}.md"
        if today_log.exists():
            async with aiofiles.open(today_log, 'r', encoding='utf-8') as f:
                return await f.read()
        return f"# Daily Log - {today}\n\n(No entries today)"

    async def read_memory(self, file_path: str) -> str:
        return await self.storage.read_file(file_path)

    async def list_memories(self) -> List[Dict[str, Any]]:
        return await self.storage.list_files()

    async def _enforce_size_limit(self):
        """
        Evict the lowest-scored non-pinned chunks when over max_chunks.

        Score = importance * log(1 + access_count). Pinned memories are immune.
        Called by the curator after each curation pass.
        """
        import sqlite3
        import math
        conn = sqlite3.connect(str(self.vector_db))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks")
        count = cursor.fetchone()[0]
        if count <= self.max_chunks:
            conn.close()
            return
        excess = count - self.max_chunks
        cursor.execute("SELECT chunk_id, metadata FROM chunks WHERE metadata IS NOT NULL")
        rows = cursor.fetchall()
        conn.close()

        evict_candidates = []
        for chunk_id, metadata_json in rows:
            if not metadata_json:
                continue
            try:
                meta = json.loads(metadata_json)
            except Exception:
                continue
            if meta.get('pinned', False):
                continue
            importance = float(meta.get('importance', 0.5))
            access_count = int(meta.get('access_count', 0))
            score = importance * math.log1p(access_count)
            evict_candidates.append((score, chunk_id))

        evict_candidates.sort(key=lambda x: x[0])
        to_evict = [chunk_id for _, chunk_id in evict_candidates[:excess]]
        if to_evict:
            conn = sqlite3.connect(str(self.vector_db))
            cursor = conn.cursor()
            placeholders = ','.join('?' * len(to_evict))
            cursor.execute(f"DELETE FROM embeddings WHERE chunk_id IN ({placeholders})", to_evict)
            cursor.execute(f"DELETE FROM chunks WHERE chunk_id IN ({placeholders})", to_evict)
            conn.commit()
            conn.close()

    @staticmethod
    def _filter_by_importance(
        results: List[Dict],
        min_importance: Optional[float],
        max_importance: Optional[float]
    ) -> List[Dict]:
        filtered = []
        for result in results:
            importance = result.get('metadata', {}).get('importance', 0.5)
            if min_importance is not None and importance < min_importance:
                continue
            if max_importance is not None and importance > max_importance:
                continue
            filtered.append(result)
        return filtered
