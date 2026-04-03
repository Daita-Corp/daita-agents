"""
Unified memory plugin with automatic environment detection.

Provides production-ready memory for DAITA agents.
Project-scoped by default, global as opt-in.
"""

import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from ..base import LifecyclePlugin
from ...core.tools import AgentTool, tool
from .metadata import MemoryMetadata
from .utils import serialize_results


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
    # Assume ISO format
    return value


class MemoryPlugin(LifecyclePlugin):
    """
    Production-ready memory plugin for DAITA agents.

    Features:
    - Automatic context injection before each run (on_before_run)
    - Hybrid semantic + keyword recall with importance weighting and temporal decay
    - Access tracking: recalled memories increment access_count for smarter pruning
    - Curator-maintained MEMORY.md: clean, categorized, never append-bloated
    - update_memory: agents can supersede stale or resolved facts

    Storage:
    - Project-scoped (default): .daita/memory/workspaces/{workspace}/
    - Global (opt-in): ~/.daita/memory/workspaces/{workspace}/

    Usage:
        # Isolated, project-scoped (default)
        agent.add_plugin(MemoryPlugin())

        # Shared workspace across agents
        memory = MemoryPlugin(workspace="research_team")
        agent1.add_plugin(memory)
        agent2.add_plugin(memory)

        # Global scope
        agent.add_plugin(MemoryPlugin(scope="global"))
    """

    def __init__(
        self,
        workspace: Optional[str] = None,
        scope: str = "project",
        auto_curate: str = "on_stop",
        curation_provider: Optional[str] = None,
        curation_model: Optional[str] = None,
        curation_api_key: Optional[str] = None,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        enable_reranking: bool = False,
        enable_fact_extraction: bool = False,
        max_chunks: int = 2000,
        default_ttl_days: Optional[int] = None,
    ):
        """
        Args:
            workspace: Workspace name (default: derived from agent_id for isolation)
            scope: "project" (default) or "global"
            auto_curate: When to curate: "on_stop" (default) or "manual"
            curation_provider: LLM for curation (default: "openai")
            curation_model: Model for curation (default: "gpt-4o-mini")
            curation_api_key: API key for curation (default: global settings)
            embedding_provider: Provider for embeddings (default: "openai")
            embedding_model: Embedding model (default: "text-embedding-3-small")
            enable_reranking: Rerank top recall results with an LLM call for
                higher accuracy at the cost of latency. Requires curator LLM.
            enable_fact_extraction: Extract structured facts from memories at
                ingestion time for richer temporal and relational recall.
                Requires curator LLM.
            max_chunks: Maximum stored chunks before eviction (default: 2000).
            default_ttl_days: Default time-to-live in days for new memories.
                None means no expiry (default). Can be overridden per-memory.
        """
        self.workspace = workspace
        self.scope = scope
        self.auto_curate = auto_curate
        self.curation_provider = curation_provider
        self.curation_model = curation_model
        self.curation_api_key = curation_api_key
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.enable_reranking = enable_reranking
        self.enable_fact_extraction = enable_fact_extraction
        self.max_chunks = max_chunks
        self.default_ttl_days = default_ttl_days

        self._agent_id = None
        self.backend = None
        self.curator = None
        self.environment = None
        self._reranker = None
        self._fact_extractor = None

    def initialize(self, agent_id: str):
        """Called by Agent.add_plugin() to inject agent context."""
        if self.backend is not None:
            self._agent_id = agent_id
            return

        self._agent_id = agent_id

        # Derive stable workspace name from agent_id (strip UUID suffix)
        if self.workspace:
            workspace = self.workspace
        elif "_" in agent_id and len(agent_id) > 9:
            workspace = "_".join(agent_id.split("_")[:-1])
        else:
            workspace = agent_id

        runtime = os.getenv("DAITA_RUNTIME", "local")

        if runtime == "lambda":
            backend_module = os.getenv("DAITA_MEMORY_BACKEND_MODULE")
            backend_class_name = os.getenv("DAITA_MEMORY_BACKEND_CLASS")

            if not backend_module or not backend_class_name:
                raise RuntimeError(
                    "Cloud memory is a Daita Cloud platform feature and is not available "
                    "in the open-source SDK. See https://daita-tech.io for managed deployment."
                )

            import importlib

            mod = importlib.import_module(backend_module)
            BackendClass = getattr(mod, backend_class_name)

            org_id = os.getenv("DAITA_ORG_ID") or os.getenv("DAITA_ORGANIZATION_ID")
            project_name = os.getenv("DAITA_PROJECT_NAME")

            if not org_id or not project_name:
                raise RuntimeError(
                    "Missing DAITA_ORG_ID and DAITA_PROJECT_NAME for cloud memory."
                )

            self.backend = BackendClass(
                org_id=org_id,
                project_name=project_name,
                workspace=workspace,
                agent_id=agent_id,
                scope=self.scope,
                embedding_provider=self.embedding_provider,
                embedding_model=self.embedding_model,
            )
            self.environment = "cloud"
            print(
                f"Memory: cloud backend, {self.scope}-scoped, workspace='{workspace}'"
            )
        else:
            from .local_backend import LocalMemoryBackend

            self.backend = LocalMemoryBackend(
                workspace=workspace,
                agent_id=agent_id,
                scope=self.scope,
                embedding_provider=self.embedding_provider,
                embedding_model=self.embedding_model,
                max_chunks=self.max_chunks,
            )
            self.environment = "local"

        curator_module = os.getenv("DAITA_MEMORY_CURATOR_MODULE")
        curator_class_name = os.getenv("DAITA_MEMORY_CURATOR_CLASS")
        if curator_module and curator_class_name:
            import importlib

            mod = importlib.import_module(curator_module)
            CuratorClass = getattr(mod, curator_class_name)
            self.curator = CuratorClass(
                backend=self.backend,
                agent_id=agent_id,
                llm_provider=self.curation_provider,
                llm_model=self.curation_model,
                api_key=self.curation_api_key,
            )
        else:
            self.curator = None

        if self.environment == "local":
            scope_label = f"{self.scope}-scoped"
            workspace_label = "shared" if self.workspace else "isolated"
            print(f"Memory: {scope_label}, {workspace_label} workspace='{workspace}'")
            print(f"  Location: {self.backend.workspace_dir}")

        if self.auto_curate != "manual" and self.curator is not None:
            print(
                f"  Auto-curation: {self.auto_curate} (provider: {self.curator.llm_provider}, model: {self.curator.llm_model})"
            )

        if self.enable_reranking and self.curator is not None:
            from .reranker import MemoryReranker

            self._reranker = MemoryReranker(llm=self.curator.llm)

        if self.enable_fact_extraction and self.curator is not None:
            from .fact_extractor import FactExtractor

            self._fact_extractor = FactExtractor(llm=self.curator.llm)

    def get_tools(self) -> List[AgentTool]:
        """Get memory tools for the LLM to use."""
        if self.backend is None:
            raise RuntimeError(
                "MemoryPlugin not initialized. Add via agent.add_plugin()."
            )

        backend = self.backend
        _reranker = self._reranker
        _fact_extractor = self._fact_extractor

        # Instantiate contradiction checker if curation LLM is available.
        _checker = None
        if self.curator is not None:
            from .contradiction import ContradictionChecker

            _checker = ContradictionChecker(
                llm=self.curator.llm,
                recall_fn=backend.recall,
                importance_threshold=0.7,
            )

        _default_ttl = self.default_ttl_days

        @tool
        async def remember(
            content: Union[str, List[Dict[str, Any]]],
            importance: float = 0.5,
            category: Optional[str] = None,
            ttl_days: Optional[int] = None,
        ):
            """
            Store information in memory for future recall.

            Use for facts, decisions, action items, and context worth preserving.

            Args:
                content: The information to store. Either:
                    - A string (single memory, be specific and self-contained)
                    - A list of dicts for batch storage, each with keys:
                        "content" (required), "importance" (optional, 0.0-1.0),
                        "category" (optional). Uses a single embedding API call.
                importance: How critical this memory is (0.0-1.0), used when
                    content is a string. Ignored for batch input (use per-item):
                    0.9+ = critical (security issues, hard deadlines, major decisions)
                    0.7-0.8 = important (key facts, events, action items)
                    0.5-0.6 = useful (general context, preferences)
                    0.3-0.4 = low priority (minor notes)
                category: Optional tag (e.g. "security", "project", "contact", "event").
                    Used when content is a string. Ignored for batch input.
                ttl_days: Days until this memory expires (default: no expiry).
                    Expired memories are pruned at session start/stop.

            Returns:
                Confirmation with chunk_id (single) or batch summary (batch)
            """
            from .auto_classify import infer_category, infer_importance

            # --- Batch path ---
            if isinstance(content, list):
                items = content

                # Auto-classify items with default values
                for item in items:
                    if item.get("category") is None:
                        item["category"] = infer_category(item["content"])
                    if item.get("importance", 0.5) == 0.5:
                        item["importance"] = infer_importance(
                            item["content"], 0.5
                        )

                # Lazy fact extraction: mark for later extraction
                extra_metadata_list = None
                if _fact_extractor is not None:
                    extra_metadata_list = [
                        {"_facts_extracted": False} for _ in items
                    ]

                return await backend.remember_batch(
                    items, extra_metadata_list=extra_metadata_list
                )

            # --- Single-item path ---
            importance = max(0.0, min(1.0, importance))

            # Auto-classify when agent uses defaults
            if category is None:
                category = infer_category(content)
            if importance == 0.5:
                importance = infer_importance(content, importance)

            # Semantic dedup: skip if a near-identical memory already exists.
            # Compare against raw_semantic_score (not the hybrid score) to prevent
            # importance-boost inflation from falsely flagging distinct facts as
            # duplicates. E.g. "FK: api_keys.org_id → org" and
            # "FK: agents.org_id → org" share template tokens (raw ~0.80) but
            # score above 0.92 after a 0.9-importance boost — they are NOT duplicates.
            existing = await backend.recall(content, limit=1, score_threshold=0.6)
            if existing:
                raw_sim = (
                    existing[0].get("raw_semantic_score")
                    or existing[0].get("score_breakdown", {}).get("semantic")
                    or existing[0].get("score", 0)
                )
                if raw_sim >= 0.92:
                    return {
                        "status": "duplicate_skipped",
                        "message": "A near-identical memory already exists",
                        "chunk_id": existing[0].get("chunk_id"),
                        "existing_score": round(float(raw_sim), 3),
                    }

            # Contradiction/evolution check for important facts (importance >= 0.7).
            # - CONTRADICTION: new fact is a reasoning error conflicting with an
            #   established memory — block storage and explain what conflicts.
            # - EVOLUTION: a real-world change (e.g. FK dropped, constraint removed)
            #   that makes an existing memory outdated — auto-replace it so memory
            #   stays accurate without forcing the agent to call update_memory().
            if importance >= 0.7 and _checker is not None:
                conflict = await _checker.check(content, importance)
                if conflict.status == "contradiction":
                    return {
                        "status": "conflict_detected",
                        "message": (
                            "New fact contradicts an existing high-importance memory. "
                            "If this is a real change, use update_memory() to explicitly "
                            "replace the old fact."
                        ),
                        "conflicting_chunk_id": conflict.conflicting_chunk_id,
                        "conflicting_content": conflict.conflicting_content,
                        "conflict_reason": conflict.conflict_reason,
                    }
                elif conflict.status == "evolution":
                    # Real-world change: replace the stale fact automatically.
                    await backend.update_memory(
                        conflict.conflicting_content,
                        content,
                        importance,
                    )
                    return {
                        "status": "evolution_replaced",
                        "message": "Existing memory was outdated and has been replaced.",
                        "replaced_chunk_id": conflict.conflicting_chunk_id,
                        "conflict_reason": conflict.conflict_reason,
                    }

            metadata = MemoryMetadata(
                content=content,
                importance=importance,
                source="agent_inferred",
                category=category,
                ttl_days=ttl_days or _default_ttl,
            )

            # Lazy fact extraction: mark for later extraction on query_facts()
            extra_metadata = None
            if _fact_extractor is not None:
                extra_metadata = {"_facts_extracted": False}

            return await backend.remember(
                content,
                category=category,
                metadata=metadata,
                extra_metadata=extra_metadata,
            )

        @tool
        async def recall(
            query: str,
            limit: int = 5,
            score_threshold: float = 0.6,
            min_importance: Optional[float] = None,
            max_importance: Optional[float] = None,
            category: Optional[str] = None,
            since: Optional[str] = None,
            before: Optional[str] = None,
        ):
            """
            Search previously stored agent memories by meaning.

            Use this ONLY to retrieve facts that were explicitly stored with
            remember() — things like business rules, unit conventions, or
            analyst notes (e.g. "prices are in cents", "exclude refunded orders").

            Do NOT use this to query database records or find patterns in data.
            For anything involving rows, columns, or data similarity, write a
            SQL query instead.

            Results are ranked by relevance, weighted by importance, and
            adjusted for age (recent memories rank slightly higher).

            Args:
                query: What you're looking for (natural language)
                limit: Maximum results (default: 5)
                score_threshold: Minimum relevance score 0-1 (default: 0.6)
                min_importance: Filter to memories with importance >= this value
                max_importance: Filter to memories with importance <= this value
                category: Restrict search to a specific category (e.g. "fk_constraint")
                since: Only return memories created at or after this time.
                    Accepts ISO datetime (e.g. "2026-04-01T00:00:00") or
                    relative shorthand: "24h", "7d", "30d".
                before: Only return memories created before this time.
                    Same format as since.

            Returns:
                Ranked list of relevant memories with scores and metadata
            """
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

            return serialize_results(results)

        @tool
        async def list_by_category(
            category: str, min_importance: float = 0.0, limit: int = 100
        ):
            """
            Enumerate ALL stored memories in a category without semantic ranking.

            PREFER this over recall() whenever the question asks to list or count
            everything of a known type:
              - "Which tables have jsonb columns?"  -> list_by_category("column_type")
              - "List all FK constraints"           -> list_by_category("fk_constraint")
              - "What are the row counts?"          -> list_by_category("table_stats")
              - "What schema observations exist?"   -> list_by_category("schema_observation")

            Use recall() only when you need semantic similarity ranking (e.g.
            "what do I know about the users table?"). For exhaustive enumeration,
            always use this tool — recall() truncates at its limit and may miss facts.

            Args:
                category: The category tag to enumerate (e.g. "fk_constraint", "column_type")
                min_importance: Only return memories with importance >= this value (default: 0.0)
                limit: Maximum results (default: 100)

            Returns:
                All matching memories ordered by importance descending
            """
            results = await backend.list_by_category(
                category=category, min_importance=min_importance, limit=limit
            )
            return serialize_results(results)

        @tool
        async def update_memory(query: str, new_content: str, importance: float = 0.5):
            """
            Replace an existing memory with updated information.

            Use when a fact has changed, been resolved, or was incorrect.
            Finds the closest matching memory, removes it, and stores the update.

            Args:
                query: Description of the memory to find and replace
                new_content: The corrected or updated information
                importance: Importance score for the updated memory (0.0-1.0)

            Returns:
                Result with count of memories replaced
            """
            return await backend.update_memory(query, new_content, importance)

        @tool
        async def read_memory(file: str = "MEMORY.md"):
            """
            Read a memory file. Accepts shorthand values or absolute paths.

            Args:
                file: "MEMORY.md" (default) for long-term memory summary,
                      "today" for today's interaction log,
                      or an absolute file path

            Returns:
                Full file contents
            """
            if file == "MEMORY.md":
                return await backend.read_memory_md()
            elif file == "today":
                return await backend.read_today_log()
            else:
                return await backend.read_memory(file)

        @tool
        async def list_memories(include_stats: bool = False):
            """
            List available memory files and optionally show memory statistics.

            Args:
                include_stats: When true, include category counts, importance
                    distribution, time range, and pinned count. Useful for
                    understanding what the agent already knows before storing
                    or recalling. (default: false)

            Returns:
                Summary of available memory files with paths and sizes.
                When include_stats=True, also includes a "stats" key with:
                  total_memories, categories (with count and avg_importance),
                  oldest/newest timestamps, and pinned_count.
            """
            from datetime import date as _date

            today = _date.today().isoformat()
            files = []
            try:
                content = await backend.read_memory_md()
                if content and not content.startswith("# Long-Term Memory\n\n(No"):
                    files.append(
                        {"file": "MEMORY.md", "size_bytes": len(content.encode())}
                    )
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

        tools = [
            remember,
            recall,
            list_by_category,
            update_memory,
            read_memory,
            list_memories,
        ]

        # Conditionally add query_facts when fact extraction is enabled.
        if _fact_extractor is not None:

            @tool
            async def query_facts(
                entity: Optional[str] = None,
                relation: Optional[str] = None,
                value: Optional[str] = None,
                limit: int = 50,
            ):
                """
                Query structured facts extracted from memories.

                Use when you need to look up specific entities, relationships,
                or values rather than doing a semantic search. Returns facts
                as (entity, relation, value, temporal_context) tuples.

                Examples:
                  - query_facts(entity="users") -> all facts about the users table
                  - query_facts(relation="FK references") -> all foreign key relationships
                  - query_facts(entity="orders", relation="has column") -> columns of orders

                Args:
                    entity: Filter by subject (partial match, case-insensitive)
                    relation: Filter by relationship type (partial match)
                    value: Filter by object/value (partial match)
                    limit: Maximum results (default: 50)

                Returns:
                    List of structured facts with source content and metadata
                """
                # Lazy extraction: extract facts for chunks that need them
                unextracted = await backend.get_unextracted_chunks(limit=50)
                if unextracted:
                    from .fact_extractor import FactExtractor

                    for chunk_id, chunk_content in unextracted:
                        facts = await _fact_extractor.extract(chunk_content)
                        await backend.update_chunk_metadata(chunk_id, {
                            "_facts_extracted": True,
                            "extracted_facts": (
                                FactExtractor.facts_to_metadata(facts)
                                if facts
                                else []
                            ),
                        })

                results = await backend.query_facts(
                    entity=entity, relation=relation, value=value, limit=limit
                )
                return serialize_results(results)

            tools.append(query_facts)

        return tools

    async def on_before_run(self, prompt: str) -> Optional[str]:
        """
        Auto-inject relevant memories into agent context before each run.

        Called by the agent framework before the first LLM turn. Returns a
        formatted memory block that gets prepended to the system prompt,
        so the agent arrives with relevant context already loaded — no
        explicit recall call needed.
        """
        if self.backend is None:
            return None

        # Prune expired memories at session start
        if hasattr(self.backend, "prune"):
            try:
                await self.backend.prune()
            except Exception:
                pass

        try:
            results = await self.backend.recall(
                prompt, limit=10, score_threshold=0.55, reranker=self._reranker
            )

            # Fallback: no semantic matches — inject top-3 by importance
            if not results:
                all_results = await self.backend.recall(
                    prompt, limit=10, score_threshold=0.0
                )
                results = sorted(
                    all_results,
                    key=lambda r: r.get("metadata", {}).get("importance", 0),
                    reverse=True,
                )[:3]
            else:
                # Always surface top-3 high-importance facts (importance >= 0.8),
                # deduplicated against the semantic results already in the list.
                seen = {r["chunk_id"] for r in results if r.get("chunk_id")}
                top_important = await self.backend.recall(
                    prompt, limit=10, score_threshold=0.0, min_importance=0.8
                )
                for r in sorted(
                    top_important,
                    key=lambda r: r.get("metadata", {}).get("importance", 0),
                    reverse=True,
                )[:3]:
                    if r.get("chunk_id") not in seen:
                        results.append(r)
                        seen.add(r["chunk_id"])

            if not results:
                return None

            # Always inject pinned memories — these are org rules / critical
            # facts that must be visible regardless of semantic similarity.
            seen = {r.get("chunk_id") for r in results if r.get("chunk_id")}
            try:
                pinned = await self.backend.get_pinned_memories()
                for p in pinned:
                    if p.get("chunk_id") not in seen:
                        results.append(p)
                        seen.add(p["chunk_id"])
            except Exception:
                pass

            # Track access for these auto-injected memories
            chunk_ids = [r["chunk_id"] for r in results if r.get("chunk_id")]
            if chunk_ids:
                self.backend.search.track_access(chunk_ids)

            lines = ["## Relevant Memory"]
            for r in results:
                importance = r.get("metadata", {}).get("importance", 0.5)
                category = r.get("metadata", {}).get("category", "")
                tag = f"[{category}] " if category else ""
                lines.append(
                    f"- {tag}{r['content'].strip()} (importance: {importance:.1f})"
                )

            return "\n".join(lines)
        except Exception as e:
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "Memory recall failed in on_before_run: %s. "
                "Agent will proceed without memory context.",
                e,
            )
            return None

    async def curate(self):
        """Manually trigger curation. Returns CurationResult. Requires Daita Cloud."""
        if not self.curator:
            raise RuntimeError(
                "Memory curation is a Daita Cloud platform feature. "
                "See https://daita-tech.io for managed cloud deployment."
            )
        return await self.curator.curate()

    async def on_agent_stop(self):
        """Flush pending changes, prune stale memories, and auto-curate."""
        if hasattr(self.backend, "flush"):
            await self.backend.flush()

        # Prune expired and over-limit memories
        if hasattr(self.backend, "prune"):
            try:
                await self.backend.prune()
            except Exception:
                pass

        if self.auto_curate == "on_stop":
            if self.curator is not None:
                result = await self.curate()
                if result.success:
                    msg = f"Memory curated: +{result.facts_added} new"
                    if result.memories_updated > 0:
                        msg += f", ~{result.memories_updated} updated"
                    if result.memories_pruned > 0:
                        msg += f", -{result.memories_pruned} pruned"
                    if getattr(result, "existing_memories", 0) > 0:
                        msg += f" ({result.existing_memories} stored by agent)"
                    msg += f", ${result.cost_usd:.4f}"
                    print(msg)
            elif hasattr(self.backend, "regenerate_memory_md"):
                path = await self.backend.regenerate_memory_md()
                print(f"Memory summary written to: {path}")

    def get_pending_metrics(self) -> dict:
        if self.backend is None:
            return {"memory_count_delta": 0, "memory_retrieval_count": 0}
        if hasattr(self.backend, "get_pending_metrics"):
            return self.backend.get_pending_metrics()
        return {
            "memory_count_delta": 0,
            "memory_retrieval_count": getattr(self.backend, "_retrieval_count", 0),
        }

    async def mark_important(
        self, query: str, importance: float, source: str = "user_explicit"
    ) -> dict:
        """Mark memories matching query with a specific importance score."""
        if not 0.0 <= importance <= 1.0:
            raise ValueError(f"Importance must be 0.0-1.0, got {importance}")

        matches = await self.backend.recall(query, limit=100, score_threshold=0.7)
        if not matches:
            return {
                "status": "success",
                "updated": 0,
                "message": "No matching memories found",
            }

        updated = 0
        for match in matches:
            chunk_id = match.get("chunk_id")
            if chunk_id:
                await self.backend.update_chunk_metadata(
                    chunk_id,
                    {
                        "importance": importance,
                        "source": source,
                    },
                )
                updated += 1

        return {
            "status": "success",
            "updated": updated,
            "message": f"Updated importance to {importance} for {updated} memories",
        }

    async def pin(self, query: str) -> dict:
        """Pin memories matching query (importance=1.0, never pruned)."""
        return await self.mark_important(query, 1.0, source="user_explicit")

    async def forget(self, query: str) -> dict:
        """Delete memories matching query."""
        matches = await self.backend.recall(query, limit=100, score_threshold=0.7)
        if not matches:
            return {
                "status": "success",
                "deleted": 0,
                "message": "No matching memories found",
            }

        chunk_ids = [m["chunk_id"] for m in matches if m.get("chunk_id")]
        await self.backend.delete_chunks(chunk_ids)

        return {
            "status": "success",
            "deleted": len(chunk_ids),
            "message": f"Deleted {len(chunk_ids)} memories",
        }

    def configure(self, **kwargs) -> dict:
        """Update plugin configuration at runtime. Supports: auto_curate."""
        updated = {}
        if "auto_curate" in kwargs:
            value = kwargs["auto_curate"]
            if isinstance(value, str) and value in ["on_stop", "manual"]:
                self.auto_curate = value
                updated["auto_curate"] = self.auto_curate
            else:
                raise ValueError("auto_curate must be 'on_stop' or 'manual'")
        return {"status": "success", "updated": updated}


def memory(**kwargs) -> MemoryPlugin:
    """Create Memory plugin with simplified interface."""
    return MemoryPlugin(**kwargs)
