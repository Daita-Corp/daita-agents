"""
Standalone tool handler functions for MemoryPlugin.

Each function receives the plugin instance as its first argument,
making them independently testable without closure capture.
"""

from __future__ import annotations

import asyncio
import re
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from .metadata import MemoryMetadata
from .utils import serialize_results

if TYPE_CHECKING:
    from .memory_plugin import MemoryPlugin


def _parse_time_param(value: Optional[str]) -> Optional[str]:
    """Parse a time parameter into an ISO datetime string.

    Accepts ISO datetimes directly or relative shorthand: "24h", "7d", "30d".
    """
    if value is None:
        return None
    match = re.fullmatch(r"(\d+)([hd])", value.strip())
    if match:
        amount, unit = int(match.group(1)), match.group(2)
        delta = timedelta(hours=amount) if unit == "h" else timedelta(days=amount)
        return (datetime.now() - delta).isoformat()
    return value


async def handle_remember(
    plugin: MemoryPlugin,
    content: Union[str, List[Dict[str, Any]]],
    importance: float = 0.5,
    category: Optional[str] = None,
    ttl_days: Optional[int] = None,
    promote_key: Optional[str] = None,
):
    """Store information in memory for future recall."""
    from .auto_classify import infer_category, infer_importance

    backend = plugin.backend
    _fact_extractor = plugin._fact_extractor
    _working_memory = plugin._working_memory
    _memory_graph = plugin._memory_graph
    _checker = plugin._checker

    # --- Promote from working memory ---
    if promote_key and _working_memory:
        promoted = _working_memory.promote(promote_key)
        if promoted is None:
            return {
                "status": "not_found",
                "message": f"No scratch item with key '{promote_key}'",
            }
        content = promoted["content"]

    # --- Batch path ---
    if isinstance(content, list):
        items = content

        for item in items:
            if item.get("category") is None:
                item["category"] = infer_category(item["content"])
            if item.get("importance", 0.5) == 0.5:
                item["importance"] = infer_importance(item["content"], 0.5)

        extra_metadata_list = None
        if _fact_extractor is not None:
            extra_metadata_list = [{"_facts_extracted": False} for _ in items]

        return await backend.remember_batch(
            items, extra_metadata_list=extra_metadata_list
        )

    # --- Single-item path ---
    importance = max(0.0, min(1.0, importance))

    if category is None:
        category = infer_category(content)
    if importance == 0.5:
        importance = infer_importance(content, importance)

    # Preprocess: split into storage (original) and index (cleaned) content.
    # The index version strips code blocks, markdown formatting, and template
    # noise so embeddings capture factual signal rather than structure.
    from .preprocessor import preprocess_content

    _storage_content, index_content = preprocess_content(content)

    # Semantic dedup: skip if a near-identical memory already exists.
    # Uses index_content so template-identical but factually different
    # memories (e.g. two table schemas) have distinct embeddings.
    # Scoped by category so different memory types (schema vs incident) about
    # the same topic are never falsely deduped against each other.
    existing = await backend.recall(
        index_content, limit=1, score_threshold=0.6, category=category
    )
    if existing:
        raw_sim = (
            existing[0].get("raw_semantic_score")
            or existing[0].get("score_breakdown", {}).get("semantic")
            or existing[0].get("score", 0)
        )
        if raw_sim >= plugin.dedup_threshold:
            return {
                "status": "duplicate_skipped",
                "message": "A near-identical memory already exists",
                "chunk_id": existing[0].get("chunk_id"),
                "existing_score": round(float(raw_sim), 3),
            }

    # Build extra metadata flags
    extra_metadata = {}
    if _fact_extractor is not None:
        extra_metadata["_facts_extracted"] = False
    if importance >= 0.7 and _checker is not None:
        extra_metadata["_contradiction_checked"] = False

    metadata = MemoryMetadata(
        content=content,
        importance=importance,
        source="agent_inferred",
        category=category,
        ttl_days=ttl_days or plugin.default_ttl_days,
    )

    result = await backend.remember(
        content,
        category=category,
        metadata=metadata,
        extra_metadata=extra_metadata or None,
        index_content=index_content,
    )

    # Queue deferred contradiction check for important facts
    if importance >= 0.7 and _checker is not None and result.get("chunk_id"):
        plugin._pending_contradiction_checks.append(
            (result["chunk_id"], content, importance)
        )

    # Background eager fact extraction (fire-and-forget)
    # Uses index_content (cleaned) so the LLM doesn't extract SQL/YAML/markdown
    # fragments as entities.
    if _fact_extractor is not None and result.get("chunk_id"):
        from .fact_extractor import FactExtractor

        async def _bg_extract():
            try:
                facts = await _fact_extractor.extract(index_content)
                facts_meta = (
                    FactExtractor.facts_to_metadata(facts) if facts else []
                )
                await backend.update_chunk_metadata(
                    result["chunk_id"],
                    {
                        "_facts_extracted": True,
                        "extracted_facts": facts_meta,
                    },
                )
                if _memory_graph and facts_meta:
                    try:
                        await _memory_graph.index_memory(
                            result["chunk_id"], index_content, facts=facts_meta
                        )
                    except Exception:
                        pass
            except Exception:
                pass  # Falls back to lazy extraction in query_facts()

        task = asyncio.create_task(_bg_extract())
        plugin._background_tasks.append(task)

    # Index in memory graph (skip heuristic indexing when fact extraction
    # is enabled — facts will populate the graph on query_facts() instead)
    if (
        _memory_graph
        and not plugin.enable_fact_extraction
        and result.get("chunk_id")
    ):
        try:
            await _memory_graph.index_memory(result["chunk_id"], index_content)
        except Exception:
            pass  # Graph indexing is best-effort

    return result


async def handle_recall(
    plugin: MemoryPlugin,
    query: str,
    limit: int = 5,
    score_threshold: float = 0.6,
    min_importance: Optional[float] = None,
    max_importance: Optional[float] = None,
    category: Optional[str] = None,
    since: Optional[str] = None,
    before: Optional[str] = None,
):
    """Search previously stored agent memories by meaning."""
    backend = plugin.backend
    _reranker = plugin._reranker
    _memory_graph = plugin._memory_graph

    results = await backend.recall(
        query=query,
        limit=limit,
        score_threshold=score_threshold,
        min_importance=min_importance,
        max_importance=max_importance,
        category=category,
        reranker=_reranker,
        since=_parse_time_param(since),
        before=_parse_time_param(before),
    )

    # Track access for usage-based pruning signals
    chunk_ids = [r["chunk_id"] for r in results if r.get("chunk_id")]
    if chunk_ids:
        backend.search.track_access(chunk_ids)

    # Graph expansion: pull in connected memories for high-confidence hits
    if _memory_graph and results:
        try:
            seen = {r["chunk_id"] for r in results if r.get("chunk_id")}
            graph_results = []
            for r in results[:3]:
                if r.get("score", 0) >= 0.7:
                    connected = await _memory_graph.get_connected_memories(
                        r["chunk_id"]
                    )
                    for cid in connected:
                        if cid not in seen:
                            chunk = await backend.get_chunk(cid)
                            if chunk:
                                chunk["source"] = "graph"
                                graph_results.append(chunk)
                                seen.add(cid)
            results.extend(graph_results[:limit])
        except Exception:
            pass  # Graph expansion is best-effort

    return serialize_results(results)


async def handle_list_by_category(
    plugin: MemoryPlugin,
    category: str,
    min_importance: float = 0.0,
    limit: int = 100,
):
    """Enumerate ALL stored memories in a category without semantic ranking."""
    results = await plugin.backend.list_by_category(
        category=category, min_importance=min_importance, limit=limit
    )
    return serialize_results(results)


async def handle_update_memory(
    plugin: MemoryPlugin,
    query: str,
    new_content: str,
    importance: float = 0.5,
):
    """Replace an existing memory with updated information."""
    return await plugin.backend.update_memory(query, new_content, importance)


async def handle_read_memory(plugin: MemoryPlugin, file: str = "MEMORY.md"):
    """Read a memory file."""
    if file == "MEMORY.md":
        return await plugin.backend.read_memory_md()
    elif file == "today":
        return await plugin.backend.read_today_log()
    else:
        return await plugin.backend.read_memory(file)


async def handle_list_memories(plugin: MemoryPlugin, include_stats: bool = False):
    """List available memory files and optionally show memory statistics."""
    backend = plugin.backend
    today = date.today().isoformat()
    files = []
    try:
        content = await backend.read_memory_md()
        if content and not content.startswith("# Long-Term Memory\n\n(No"):
            files.append({"file": "MEMORY.md", "size_bytes": len(content.encode())})
    except Exception:
        pass
    try:
        log_content = await backend.read_today_log()
        if log_content and not log_content.startswith(
            f"# Daily Log - {today}\n\n(No"
        ):
            files.append(
                {
                    "file": f"logs/{today}.md",
                    "size_bytes": len(log_content.encode()),
                }
            )
    except Exception:
        pass

    if not include_stats:
        return files

    stats = await backend.get_stats()
    return {"files": files, "stats": stats}


async def handle_query_facts(
    plugin: MemoryPlugin,
    entity: Optional[str] = None,
    relation: Optional[str] = None,
    value: Optional[str] = None,
    limit: int = 50,
):
    """Query structured facts extracted from memories."""
    backend = plugin.backend
    _fact_extractor = plugin._fact_extractor
    _memory_graph = plugin._memory_graph

    # Lazy extraction: extract facts for chunks that need them (parallelized)
    unextracted = await backend.get_unextracted_chunks(limit=50)
    if unextracted:
        from .fact_extractor import FactExtractor

        sem = asyncio.Semaphore(5)

        async def _extract_one(cid: str, text: str):
            async with sem:
                try:
                    facts = await _fact_extractor.extract(text)
                    facts_meta = (
                        FactExtractor.facts_to_metadata(facts) if facts else []
                    )
                    await backend.update_chunk_metadata(
                        cid,
                        {
                            "_facts_extracted": True,
                            "extracted_facts": facts_meta,
                        },
                    )
                    if _memory_graph and facts_meta:
                        try:
                            await _memory_graph.index_memory(
                                cid, text, facts=facts_meta
                            )
                        except Exception:
                            pass
                except Exception:
                    pass  # Chunk stays unextracted, retried on next call

        await asyncio.gather(
            *[_extract_one(cid, text) for cid, text in unextracted]
        )

    results = await backend.query_facts(
        entity=entity, relation=relation, value=value, limit=limit
    )
    return serialize_results(results)


async def handle_scratch(
    plugin: MemoryPlugin, content: str, key: Optional[str] = None
):
    """Store temporary info in session working memory."""
    assigned_key = plugin._working_memory.scratch(content, key)
    return {
        "status": "stored",
        "key": assigned_key,
        "message": f"Stored in working memory as '{assigned_key}'",
    }


async def handle_think(plugin: MemoryPlugin, query: str, limit: int = 5):
    """Search session working memory (scratch items only)."""
    return plugin._working_memory.think(query, limit)


async def handle_reinforce(
    plugin: MemoryPlugin,
    memory_ids: Union[str, List[str]],
    outcome: str,
    signal_strength: float = 0.5,
    context: Optional[str] = None,
):
    """Record whether recalled memories led to good or bad outcomes."""
    if outcome not in ("positive", "negative", "neutral"):
        return {
            "status": "error",
            "message": "outcome must be 'positive', 'negative', or 'neutral'",
        }
    signal_strength = max(0.0, min(1.0, signal_strength))

    if isinstance(memory_ids, str):
        memory_ids = [memory_ids]

    backend = plugin.backend
    reinforcement = {
        "outcome": outcome,
        "signal": signal_strength,
        "timestamp": datetime.now().isoformat(),
        "context": context,
    }

    importance_delta = signal_strength * 0.1 if outcome == "positive" else 0.0

    updated = 0
    for chunk_id in memory_ids:
        try:
            await backend.append_reinforcement(
                chunk_id, reinforcement, importance_delta
            )
            if outcome == "negative":
                await backend.update_chunk_metadata(
                    chunk_id, {"flagged_for_review": True}
                )
            updated += 1
        except Exception:
            pass

    return {
        "status": "success",
        "reinforced": updated,
        "outcome": outcome,
        "message": f"Reinforced {updated} memories as {outcome}",
    }


async def handle_traverse_memory(
    plugin: MemoryPlugin, entity: str, max_depth: int = 2
):
    """Walk the memory knowledge graph to find all connected knowledge."""
    return await plugin._memory_graph.traverse_entity(
        entity, direction="both", max_depth=max_depth
    )
