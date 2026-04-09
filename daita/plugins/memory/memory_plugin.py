"""
Unified memory plugin with automatic environment detection.

Provides production-ready memory for DAITA agents.
Project-scoped by default, global as opt-in.
"""

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from ..base import LifecyclePlugin
from ...core.tools import AgentTool, tool

if TYPE_CHECKING:
    from ...embeddings.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)

_TOOL_TIERS = {
    "basic": {"remember", "recall", "read_memory", "list_memories"},
    "analysis": {
        "remember",
        "recall",
        "read_memory",
        "list_memories",
        "query_facts",
        "traverse_memory",
        "reinforce",
        "list_by_category",
    },
    "full": None,  # None means all tools
}


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
        embedder: Optional["BaseEmbeddingProvider"] = None,
        enable_reranking: bool = False,
        enable_fact_extraction: bool = False,
        enable_working_memory: bool = False,
        enable_reinforcement: bool = False,
        enable_memory_graph: bool = False,
        graph_auto_promote_specificity: float = 0.7,
        graph_mention_promote_specificity: float = 0.3,
        graph_mention_promote_count: int = 2,
        max_chunks: int = 2000,
        default_ttl_days: Optional[int] = None,
        tier: str = "basic",
        memory_tools: Optional[List[str]] = None,
        dedup_threshold: float = 0.95,
    ):
        """
        Args:
            workspace: Workspace name (default: derived from agent_id for isolation)
            scope: "project" (default) or "global"
            auto_curate: When to curate: "on_stop" (default) or "manual"
            curation_provider: LLM for curation (default: "openai")
            curation_model: Model for curation (default: "gpt-4o-mini")
            curation_api_key: API key for curation (default: global settings)
            embedding_provider: Provider name for embeddings (default: "openai").
                Ignored if ``embedder`` is provided.
            embedding_model: Embedding model (default: "text-embedding-3-small").
                Ignored if ``embedder`` is provided.
            embedder: Pre-constructed BaseEmbeddingProvider instance.
                Takes precedence over embedding_provider/embedding_model strings.
            enable_reranking: Rerank top recall results with an LLM call for
                higher accuracy at the cost of latency. Uses curator LLM if
                available, otherwise creates a standalone LLM via
                curation_provider/curation_model.
            enable_fact_extraction: Extract structured facts from memories at
                ingestion time for richer temporal and relational recall.
                Uses curator LLM if available, otherwise creates a standalone
                LLM via curation_provider/curation_model.
            enable_working_memory: Enable session-scoped scratchpad. Adds
                scratch() and think() tools. Working memory auto-evicts on
                agent stop unless promoted via remember(promote_key=...).
            enable_reinforcement: Enable outcome-based learning. Adds
                reinforce() tool for recording whether recalled memories
                led to good or bad outcomes.
            enable_memory_graph: Enable knowledge graph over memories. Adds
                traverse_memory() tool and auto-expands recall results via
                graph traversal. Works best with enable_fact_extraction=True.
            graph_auto_promote_specificity: Minimum specificity score (0.0-1.0)
                for an entity to be promoted on first mention (default: 0.7).
                Proper nouns and technical identifiers typically score >= 0.7.
            graph_mention_promote_specificity: Minimum specificity score for
                mention-based promotion (default: 0.3). Entities scoring
                between this and auto_promote need multiple mentions.
            graph_mention_promote_count: Number of distinct memory mentions
                required to promote an entity below auto_promote_specificity
                (default: 2).
            max_chunks: Maximum stored chunks before eviction (default: 2000).
            default_ttl_days: Default time-to-live in days for new memories.
                None means no expiry (default). Can be overridden per-memory.
            tier: Tool tier controlling which tools are exposed to the LLM.
                "basic" (default): remember, recall, read_memory, list_memories
                "analysis": basic + query_facts, traverse_memory, reinforce,
                    list_by_category
                "full": all tools including scratch, think, update_memory
                Tier only controls which tools are registered; feature flags
                still control whether optional backends are initialized.
            memory_tools: Explicit whitelist of tool names to expose.
                Overrides tier when provided. Use for fine-grained control.
            dedup_threshold: Minimum raw semantic similarity (cosine) to
                consider a new memory a duplicate of an existing one
                (default: 0.95). Dedup is also scoped by category — a
                memory is only deduped against others in the same category.
        """
        self.workspace = workspace
        self.scope = scope
        self.auto_curate = auto_curate
        self.curation_provider = curation_provider
        self.curation_model = curation_model
        self.curation_api_key = curation_api_key
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self._embedder = embedder
        self.enable_reranking = enable_reranking
        self.enable_fact_extraction = enable_fact_extraction
        self.enable_working_memory = enable_working_memory
        self.enable_reinforcement = enable_reinforcement
        self.enable_memory_graph = enable_memory_graph
        self._graph_auto_promote_specificity = graph_auto_promote_specificity
        self._graph_mention_promote_specificity = graph_mention_promote_specificity
        self._graph_mention_promote_count = graph_mention_promote_count
        self.max_chunks = max_chunks
        self.default_ttl_days = default_ttl_days
        self.tier = tier
        self.memory_tools = memory_tools
        self.dedup_threshold = dedup_threshold

        self._agent_id = None
        self.backend = None
        self.curator = None
        self.environment = None
        self._curation_llm = None
        self._reranker = None
        self._fact_extractor = None
        self._checker = None
        self._working_memory = None
        self._memory_graph = None
        self._pending_contradiction_checks: list = []
        self._background_tasks: list = []

    def _ensure_curation_llm(self):
        """Return the curator LLM if available, else create a standalone one.

        Cached so all consumers (fact extractor, reranker, contradiction
        checker) share a single instance.
        """
        if self.curator is not None:
            return self.curator.llm
        if self._curation_llm is None:
            from ...llm.factory import create_llm_provider

            self._curation_llm = create_llm_provider(
                provider=self.curation_provider or "openai",
                model=self.curation_model or "gpt-4o-mini",
                api_key=self.curation_api_key,
            )
        return self._curation_llm

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

        # Build embedding provider (factory or pre-constructed)
        if self._embedder is None:
            from ...embeddings import create_embedding_provider

            self._embedder = create_embedding_provider(
                provider=self.embedding_provider,
                model=self.embedding_model,
            )

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
                embedder=self._embedder,
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
                embedder=self._embedder,
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
        elif self.auto_curate != "manual" and self.curator is None:
            logger.info(
                "Auto-curation set to '%s' but no curator available. "
                "MEMORY.md will be regenerated from vector DB on stop instead.",
                self.auto_curate,
            )

        if self.enable_reranking:
            from .reranker import MemoryReranker

            self._reranker = MemoryReranker(llm=self._ensure_curation_llm())

        if self.enable_fact_extraction:
            from .fact_extractor import FactExtractor

            self._fact_extractor = FactExtractor(llm=self._ensure_curation_llm())

        if self.enable_working_memory:
            from .working_memory import WorkingMemory

            self._working_memory = WorkingMemory()

        if self.enable_memory_graph:
            from .memory_graph import MemoryGraph

            self._memory_graph = MemoryGraph(
                agent_id=agent_id,
                default_properties={"workspace": workspace},
                auto_promote_specificity=self._graph_auto_promote_specificity,
                mention_promote_specificity=self._graph_mention_promote_specificity,
                mention_promote_count=self._graph_mention_promote_count,
            )
            if not self.enable_fact_extraction:
                logger.info(
                    "Memory graph using keyword heuristics only. "
                    "Enable fact_extraction for richer entity graphs."
                )

        # Contradiction checker for high-importance fact validation
        from .contradiction import ContradictionChecker

        self._checker = ContradictionChecker(
            llm=self._ensure_curation_llm(),
            recall_fn=self.backend.recall,
            importance_threshold=0.7,
        )

    def get_tools(self) -> List[AgentTool]:
        """Get memory tools for the LLM to use."""
        if self.backend is None:
            raise RuntimeError(
                "MemoryPlugin not initialized. Add via agent.add_plugin()."
            )

        from .memory_tools import (
            handle_remember,
            handle_recall,
            handle_list_by_category,
            handle_update_memory,
            handle_read_memory,
            handle_list_memories,
            handle_query_facts,
            handle_scratch,
            handle_think,
            handle_reinforce,
            handle_traverse_memory,
        )

        plugin = self

        @tool
        async def remember(
            content: Union[str, List[Dict[str, Any]]],
            importance: float = 0.5,
            category: Optional[str] = None,
            ttl_days: Optional[int] = None,
            promote_key: Optional[str] = None,
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
                promote_key: Optional key from working memory (scratch) to promote
                    to long-term storage. When provided, uses the scratch item's
                    content (overrides the content parameter).
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
            return await handle_remember(
                plugin, content, importance, category, ttl_days, promote_key
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
            return await handle_recall(
                plugin,
                query,
                limit,
                score_threshold,
                min_importance,
                max_importance,
                category,
                since,
                before,
            )

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
            return await handle_list_by_category(
                plugin, category, min_importance, limit
            )

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
            return await handle_update_memory(plugin, query, new_content, importance)

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
            return await handle_read_memory(plugin, file)

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
            return await handle_list_memories(plugin, include_stats)

        tools = [
            remember,
            recall,
            list_by_category,
            update_memory,
            read_memory,
            list_memories,
        ]

        # Conditionally add query_facts when fact extraction is enabled.
        if self._fact_extractor is not None:

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
                return await handle_query_facts(plugin, entity, relation, value, limit)

            tools.append(query_facts)

        # Working memory tools
        if self._working_memory is not None:

            @tool
            async def scratch(content: str, key: Optional[str] = None):
                """
                Store temporary info in session working memory.

                Fast (no embedding cost). Discarded when agent stops unless
                promoted to long-term memory via remember(promote_key=key).

                Use for intermediate results, checklists, or temporary state
                during multi-step tasks.

                Args:
                    content: Information to store temporarily
                    key: Optional key for later reference (auto-generated if omitted)

                Returns:
                    The key assigned to this scratch item
                """
                return await handle_scratch(plugin, content, key)

            @tool
            async def think(query: str, limit: int = 5):
                """
                Search session working memory (scratch items only).

                Use to retrieve temporary notes, intermediate results, or
                scratchpad items stored during this session with scratch().

                Args:
                    query: What to search for (keyword matching)
                    limit: Maximum results (default: 5)

                Returns:
                    Matching scratch items from this session
                """
                return await handle_think(plugin, query, limit)

            tools.extend([scratch, think])

        # Reinforcement tool
        if self.enable_reinforcement:

            @tool
            async def reinforce(
                memory_ids: Union[str, List[str]],
                outcome: str,
                signal_strength: float = 0.5,
                context: Optional[str] = None,
            ):
                """
                Record whether recalled memories led to good or bad outcomes.

                Call after acting on recalled memories to improve future recall.
                Positive reinforcement increases importance; negative flags for
                review and accelerates pruning.

                Args:
                    memory_ids: chunk_id(s) to reinforce (from recall results).
                        Accepts a single string or a list.
                    outcome: "positive", "negative", or "neutral"
                    signal_strength: How strong the signal (0.0-1.0, default 0.5)
                    context: Optional description of what happened

                Returns:
                    Summary of reinforcement applied
                """
                return await handle_reinforce(
                    plugin, memory_ids, outcome, signal_strength, context
                )

            tools.append(reinforce)

        # Memory graph traversal tool
        if self._memory_graph is not None:

            @tool
            async def traverse_memory(entity: str, max_depth: int = 2):
                """
                Walk the memory knowledge graph to find all connected knowledge.

                Use when you need to find everything related to a concept,
                especially facts that semantic search might miss. For example:
                "what are all the infrastructure constraints for Project Orion?"

                Args:
                    entity: Entity name to start from (e.g. "PostgreSQL",
                        "users table", "Project Orion")
                    max_depth: How many hops to traverse (default: 2)

                Returns:
                    Connected entities and the memories that reference them
                """
                return await handle_traverse_memory(plugin, entity, max_depth)

            tools.append(traverse_memory)

        # Filter tools by tier or explicit whitelist
        allowed = self.memory_tools
        if allowed is None:
            tier_set = _TOOL_TIERS.get(self.tier)
            if tier_set is not None:
                allowed = tier_set
        if allowed is not None:
            tools = [t for t in tools if t.name in allowed]

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

            if not results and not (
                self._working_memory and len(self._working_memory) > 0
            ):
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

            lines = []
            if results:
                lines.append("## Relevant Memory")
                for r in results:
                    importance = r.get("metadata", {}).get("importance", 0.5)
                    category = r.get("metadata", {}).get("category", "")
                    tag = f"[{category}] " if category else ""
                    lines.append(
                        f"- {tag}{r['content'].strip()} (importance: {importance:.1f})"
                    )

            # Inject working memory if it has items
            if self._working_memory and len(self._working_memory) > 0:
                lines.append("")
                lines.append("## Working Memory (session scratchpad)")
                for item in self._working_memory.dump():
                    status = " [promoted]" if item["promoted"] else ""
                    lines.append(f"- [{item['key']}] {item['content'].strip()}{status}")

            return "\n".join(lines)
        except Exception as e:
            logger.warning(
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

    async def _process_deferred_contradictions(self):
        """Batch-process queued contradiction checks concurrently.

        Called during on_agent_stop(). Memories are already stored — if a
        contradiction is found the chunk is flagged (not deleted) so the
        next session or a human can resolve it. Evolutions trigger the same
        auto-replace as the former synchronous path.
        """
        import asyncio

        sem = asyncio.Semaphore(3)

        async def _check_one(chunk_id: str, content: str, importance: float):
            async with sem:
                try:
                    conflict = await self._checker.check(content, importance)
                    if conflict.status == "contradiction":
                        await self.backend.update_chunk_metadata(
                            chunk_id,
                            {
                                "_contradiction_checked": True,
                                "flagged_contradiction": True,
                                "conflict_reason": conflict.conflict_reason,
                                "conflicting_chunk_id": conflict.conflicting_chunk_id,
                            },
                        )
                    elif conflict.status == "evolution":
                        # Delete the specific stale chunk by ID — not a broad
                        # recall-and-delete which can cascade into collateral
                        # deletions of unrelated memories that happen to share
                        # similar embeddings (e.g. template-identical schemas).
                        if conflict.conflicting_chunk_id:
                            await self.backend.delete_chunks(
                                [conflict.conflicting_chunk_id]
                            )
                        await self.backend.update_chunk_metadata(
                            chunk_id, {"_contradiction_checked": True}
                        )
                    else:
                        await self.backend.update_chunk_metadata(
                            chunk_id, {"_contradiction_checked": True}
                        )
                except Exception:
                    pass  # Non-fatal — check will be skipped

        await asyncio.gather(
            *[
                _check_one(cid, content, imp)
                for cid, content, imp in self._pending_contradiction_checks
                if cid is not None
            ]
        )

    async def on_agent_stop(self):
        """Flush pending changes, prune stale memories, and auto-curate.

        MEMORY.md is a human-readable artifact for inspecting what the agent
        has learned. Agent context injection uses semantic recall via
        on_before_run(), not MEMORY.md.
        """
        import asyncio

        # Await background extraction tasks
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

        # Process deferred contradiction checks
        if self._pending_contradiction_checks and self._checker:
            await self._process_deferred_contradictions()
        self._pending_contradiction_checks.clear()

        # Clear working memory (session-scoped)
        if self._working_memory:
            self._working_memory.clear()

        # Flush memory graph
        if self._memory_graph:
            try:
                await self._memory_graph.flush()
            except Exception:
                pass

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
                if path is not None:
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
