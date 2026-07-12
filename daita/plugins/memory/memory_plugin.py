"""
Unified memory plugin with automatic environment detection.

Provides production-ready memory for DAITA agents.
Project-scoped by default, global as opt-in.
"""

import logging
import inspect
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from ..base import DomainServicePlugin
from .extensions import (
    MEMORY_MANIFEST,
    MemoryContextProvider,
    MemoryExecutor,
    memory_capabilities,
    memory_evidence_schemas,
)

if TYPE_CHECKING:
    from ...embeddings.base import BaseEmbeddingProvider
    from .graph_store import MemoryGraphStore
    from .local_backend import LocalMemoryBackend

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


def _memory_graph_type(scope: str, workspace: str, project_name: Optional[str]) -> str:
    """Return a safe graph namespace matching memory's isolation boundary."""
    parts = ["memory", scope]
    if project_name:
        parts.append(project_name)
    parts.append(workspace)
    raw = "_".join(parts)
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", raw).strip("_").lower()
    return safe or "memory"


async def _maybe_await(result: Any) -> Any:
    if inspect.isawaitable(result):
        return await result
    return result


def _merge_recall_results(
    structured_results: list[dict[str, Any]],
    embedding_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for result in (*structured_results, *embedding_results):
        key = str(result.get("chunk_id") or result.get("record_id") or id(result))
        existing = merged.get(key)
        if existing is None:
            merged[key] = dict(result)
            continue
        existing_score = float(
            existing.get("relevance_score", existing.get("score", 0))
        )
        score = float(result.get("relevance_score", result.get("score", 0)))
        if score > existing_score:
            merged[key] = dict(result)
    results = list(merged.values())
    results.sort(
        key=lambda item: float(item.get("relevance_score", item.get("score", 0))),
        reverse=True,
    )
    return results


def _declared_method(obj: Any, name: str) -> Any | None:
    if obj is None:
        return None
    try:
        inspect.getattr_static(obj, name)
    except AttributeError:
        return None
    method = getattr(obj, name, None)
    return method if callable(method) else None


def _backend_name(backend: Any, environment: str | None) -> str:
    if backend is None:
        return environment or "unconfigured"
    name = type(backend).__name__.lower()
    if "supabase" in name:
        return "supabase"
    if "local" in name:
        return "local"
    return name or (environment or "custom")


def _declared_attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    try:
        inspect.getattr_static(obj, name)
    except AttributeError:
        return default
    return getattr(obj, name, default)


def _db_semantic_result_allowed(
    result: dict[str, Any],
    *,
    source_identity: str | None,
) -> bool:
    raw_metadata = result.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    record = metadata.get("db_memory")
    if not isinstance(record, dict):
        return False
    raw_record_metadata = record.get("metadata")
    record_metadata = (
        raw_record_metadata if isinstance(raw_record_metadata, dict) else {}
    )
    record_source = record.get("source_identity") or record_metadata.get(
        "source_identity"
    )
    if source_identity and record_source != source_identity:
        return False
    if record.get("category", metadata.get("category")) != "db_semantics":
        return False
    if (
        record.get("workspace_scope", record_metadata.get("workspace_scope", "source"))
        != "source"
    ):
        return False
    if record.get("active", record_metadata.get("active", True)) is False:
        return False
    if record.get("stale", record_metadata.get("stale", False)) is True:
        return False
    return True


class MemoryPlugin(DomainServicePlugin):
    """
    Production-ready memory plugin for DAITA agents.

    Features:
    - Automatic context injection through MemoryContextProvider
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

    manifest = MEMORY_MANIFEST

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
        memory_graph_store: Optional["MemoryGraphStore"] = None,
        memory_graph_backend: Optional[Any] = None,
        memory_graph_type: Optional[str] = None,
        max_chunks: int = 2000,
        default_ttl_days: Optional[int] = None,
        tier: str = "basic",
        memory_tools: Optional[List[str]] = None,
        dedup_threshold: float = 0.95,
        db_memory_mode: bool = False,
        db_memory_retrieval_mode: str = "structured",
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
            memory_graph_store: Optional memory graph store implementation.
                Use this for fully custom memory graph persistence/traversal.
            memory_graph_backend: Optional core GraphBackend instance for the
                memory graph. Ignored when ``memory_graph_store`` is provided.
            memory_graph_type: Optional graph namespace passed to registered
                graph backends. Defaults to a workspace-scoped namespace in
                cloud and "memory" locally.
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
            db_memory_mode: Enable the structured DB semantic memory lane used
                by Agent.from_db(). Generic MemoryPlugin behavior is unchanged
                when this is False.
            db_memory_retrieval_mode: DB semantic recall mode: "structured",
                "hybrid", or "embedding".
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
        self._memory_graph_store = memory_graph_store
        self._memory_graph_backend = memory_graph_backend
        self._memory_graph_type = memory_graph_type
        self.max_chunks = max_chunks
        self.default_ttl_days = default_ttl_days
        self.tier = tier
        self.memory_tools = memory_tools
        self.dedup_threshold = dedup_threshold
        self.db_memory_mode = bool(db_memory_mode)
        self.db_memory_retrieval_mode = str(db_memory_retrieval_mode or "structured")

        self._agent_id: str | None = None
        self.backend: LocalMemoryBackend | None = None
        self.curator = None
        self.environment: str | None = None
        self._curation_llm = None
        self._reranker = None
        self._fact_extractor = None
        self._checker = None
        self._working_memory = None
        self._memory_graph = None
        self._pending_contradiction_checks: list = []
        self._background_tasks: list = []

    @property
    def embedding_available(self) -> bool:
        if self._embedder is not None:
            return True
        return bool(getattr(self.backend, "embedding_available", False))

    async def setup(self, context):
        """Set up the memory backend for an extension-runtime host."""
        self._configure_backend(context.agent_id or context.runtime_id)

    def declare_capabilities(self):
        return memory_capabilities()

    def get_executors(self):
        return (
            MemoryExecutor(
                id="memory.semantic.recall",
                capability_ids=frozenset({"memory.semantic.recall"}),
                evidence_kind="memory.semantic.recall",
                handler=self._execute_semantic_recall,
            ),
            MemoryExecutor(
                id="memory.semantic.write",
                capability_ids=frozenset({"memory.semantic.write"}),
                evidence_kind="memory.semantic.write",
                handler=self._execute_semantic_write,
            ),
            MemoryExecutor(
                id="memory.fact.query",
                capability_ids=frozenset({"memory.fact.query"}),
                evidence_kind="memory.fact.query",
                handler=self._execute_fact_query,
            ),
            MemoryExecutor(
                id="memory.context.render",
                capability_ids=frozenset({"memory.context.render"}),
                evidence_kind="memory.context",
                handler=self._execute_context_render,
            ),
        )

    def declare_evidence_schemas(self):
        return memory_evidence_schemas()

    def get_context_providers(self):
        return (MemoryContextProvider(plugin=self),)

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

    def _configure_backend(self, agent_id: str) -> None:
        """Configure the memory backend for generic-agent and runtime hosts."""
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

        structured_db_only = (
            self.db_memory_mode and self.db_memory_retrieval_mode == "structured"
        )

        # Build embedding provider (factory or pre-constructed).
        # The from_db structured DB lane intentionally avoids constructing the
        # generic vector lane unless hybrid/embedding recall is explicit.
        if self._embedder is None and not structured_db_only:
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
                retrieval_mode=self.db_memory_retrieval_mode,
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
                default_source_identity=workspace if self.db_memory_mode else None,
            )
            self.environment = "local"

        backend = self.backend
        if backend is None:
            raise RuntimeError("Memory backend configuration did not produce a backend")

        curator_module = os.getenv("DAITA_MEMORY_CURATOR_MODULE")
        curator_class_name = os.getenv("DAITA_MEMORY_CURATOR_CLASS")
        if curator_module and curator_class_name:
            import importlib

            mod = importlib.import_module(curator_module)
            CuratorClass = getattr(mod, curator_class_name)
            self.curator = CuratorClass(
                backend=backend,
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
            print(f"  Location: {backend.workspace_dir}")

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
            self._memory_graph = self._build_memory_graph(agent_id, workspace)
            if not self.enable_fact_extraction:
                logger.info(
                    "Memory graph using keyword heuristics only. "
                    "Enable fact_extraction for richer entity graphs."
                )

        # Contradiction checking belongs to the generic vector lane. The
        # structured DB lane validates DB records before storage and should not
        # construct an LLM solely to support unused generic memory behavior.
        if getattr(backend, "embedding_available", True):
            from .contradiction import ContradictionChecker

            self._checker = ContradictionChecker(
                llm=self._ensure_curation_llm(),
                recall_fn=backend.recall,
                importance_threshold=0.7,
            )
        else:
            self._checker = None

    def _build_memory_graph(self, agent_id: str, workspace: str):
        """Build the optional memory graph without coupling to a backend."""
        from .memory_graph import MemoryGraph

        graph_storage_dir = None
        graph_type = self._memory_graph_type or "memory"
        backend = self.backend

        if (
            self.environment == "local"
            and backend is not None
            and hasattr(backend, "workspace_dir")
        ):
            graph_storage_dir = backend.workspace_dir / "graph"
        elif self.environment == "cloud" and self._memory_graph_type is None:
            graph_type = _memory_graph_type(
                self.scope,
                workspace,
                os.getenv("DAITA_PROJECT_NAME") if self.scope != "global" else None,
            )

        return MemoryGraph(
            agent_id=agent_id,
            default_properties={"workspace": workspace},
            store=self._memory_graph_store,
            backend=self._memory_graph_backend,
            storage_dir=graph_storage_dir,
            graph_type=graph_type,
            auto_promote_specificity=self._graph_auto_promote_specificity,
            mention_promote_specificity=self._graph_mention_promote_specificity,
            mention_promote_count=self._graph_mention_promote_count,
        )

    async def _execute_semantic_recall(self, payload: Any) -> Dict[str, Any]:
        from .memory_tools import handle_recall

        args = dict(payload or {})
        category = args.get("category")
        if category == "db_semantics":
            return await self._execute_db_semantic_recall(args)

        results = await handle_recall(
            self,
            query=str(args.get("query") or ""),
            limit=int(args.get("limit") or 5),
            score_threshold=float(args.get("score_threshold", 0.6)),
            min_importance=args.get("min_importance"),
            max_importance=args.get("max_importance"),
            category=args.get("category"),
            since=args.get("since"),
            before=args.get("before"),
        )
        return {"query": str(args.get("query") or ""), "results": results}

    async def _execute_db_semantic_recall(self, args: dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query") or "")
        requested_mode = args.get("retrieval_mode") or self.db_memory_retrieval_mode
        retrieval_mode = str(requested_mode or "structured")
        if retrieval_mode not in {"structured", "hybrid", "embedding"}:
            return {
                "query": query,
                "results": [],
                "diagnostics": {
                    "retrieval_mode": retrieval_mode,
                    "embedding_available": self.embedding_available,
                    "structured_candidate_count": 0,
                    "embedding_candidate_count": 0,
                    "fallback": "invalid_retrieval_mode",
                },
            }

        backend = self.backend
        if backend is None:
            raise RuntimeError("MemoryPlugin must be set up before recall")

        limit = int(args.get("limit") or 5)
        score_threshold = float(args.get("score_threshold", 0.45))
        source_identity = args.get("source_identity") or (
            self.workspace if self.db_memory_mode else None
        )
        diagnostics = {
            "backend": _backend_name(backend, self.environment),
            "retrieval_mode": retrieval_mode,
            "embedding_available": self.embedding_available,
            "structured_index": _declared_attr(backend, "structured_index", None),
            "structured_candidate_count": 0,
            "embedding_candidate_count": 0,
            "fallback": None,
        }

        structured_results = []
        if retrieval_mode in {"structured", "hybrid"}:
            recall_db_records = _declared_method(backend, "recall_db_records")
            if recall_db_records is not None:
                structured_results = await recall_db_records(
                    query,
                    limit=limit,
                    score_threshold=score_threshold,
                    source_identity=source_identity,
                    category="db_semantics",
                    kinds=args.get("kinds"),
                )
            elif retrieval_mode == "structured":
                diagnostics["fallback"] = "structured_backend_unavailable"
            diagnostics["structured_candidate_count"] = len(structured_results)

        embedding_results = []
        if retrieval_mode in {"hybrid", "embedding"}:
            if not self.embedding_available:
                diagnostics["fallback"] = "embedding_unavailable"
                if retrieval_mode == "embedding":
                    return {
                        "query": query,
                        "results": [],
                        "diagnostics": diagnostics,
                    }
            else:
                try:
                    embedding_results = await backend.recall(
                        query=query,
                        limit=limit,
                        score_threshold=score_threshold,
                        category="db_semantics",
                        reranker=self._reranker,
                    )
                    embedding_results = [
                        result
                        for result in embedding_results
                        if _db_semantic_result_allowed(
                            result, source_identity=source_identity
                        )
                    ]
                except Exception as exc:
                    diagnostics["fallback"] = f"embedding_recall_failed:{exc}"
                    embedding_results = []
            diagnostics["embedding_candidate_count"] = len(embedding_results)

        if retrieval_mode == "embedding":
            results = embedding_results
        else:
            results = _merge_recall_results(structured_results, embedding_results)
        return {
            "query": query,
            "results": results[:limit],
            "diagnostics": diagnostics,
        }

    async def _execute_semantic_write(self, payload: Any) -> Dict[str, Any]:
        from .memory_tools import handle_remember

        args = dict(payload or {})
        if "db_memory_payload" in args:
            from daita.db.memory import (
                db_memory_record_from_payload,
                write_db_memory_record,
            )

            try:
                record = db_memory_record_from_payload(
                    dict(args.get("db_memory_payload") or {}),
                    str(args.get("db_memory_prompt") or ""),
                    task_metadata=dict(args.get("_runtime_task_metadata") or {}),
                )
            except Exception as exc:
                return {"success": False, "error": str(exc)}
            return await write_db_memory_record(self, record)

        content = args.get("content") or args.get("memory") or ""
        result = await handle_remember(
            self,
            content=content,
            importance=float(args.get("importance", 0.5)),
            category=args.get("category"),
            ttl_days=args.get("ttl_days"),
            promote_key=args.get("promote_key"),
        )
        return {"content": content, "result": result}

    async def _execute_fact_query(self, payload: Any) -> Dict[str, Any]:
        from .memory_tools import handle_query_facts

        args = dict(payload or {})
        results = await handle_query_facts(
            self,
            entity=args.get("entity"),
            relation=args.get("relation"),
            value=args.get("value"),
            limit=int(args.get("limit") or 50),
        )
        return {"results": results}

    async def _execute_context_render(self, payload: Any) -> Dict[str, Any]:
        args = dict(payload or {})
        content = await self.render_context(
            prompt=str(args.get("prompt") or ""),
            token_budget=int(args.get("token_budget") or 2000),
        )
        return {"content": content, "rendered": bool(content)}

    async def render_context(self, prompt: str, token_budget: int = 2000) -> str | None:
        """Render bounded memory context for runtime context providers."""
        content = await self._render_relevant_context(prompt)
        if content and token_budget:
            return content[: max(token_budget, 0)]
        return content

    async def _render_relevant_context(self, prompt: str) -> Optional[str]:
        """Render relevant memories for runtime context providers."""
        backend = self.backend
        if backend is None:
            return None

        # Prune expired memories at session start
        if hasattr(backend, "prune"):
            try:
                await backend.prune()
            except Exception:
                pass

        try:
            results = await backend.recall(
                prompt, limit=10, score_threshold=0.55, reranker=self._reranker
            )

            # Fallback: no semantic matches — inject top-3 by importance
            if not results:
                all_results = await backend.recall(
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
                top_important = await backend.recall(
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
                pinned = await backend.get_pinned_memories()
                for p in pinned:
                    if p.get("chunk_id") not in seen:
                        results.append(p)
                        seen.add(p["chunk_id"])
            except Exception:
                pass

            # Track access for these auto-injected memories
            chunk_ids = [r["chunk_id"] for r in results if r.get("chunk_id")]
            if chunk_ids:
                search = backend.search
                if search is None:
                    raise AttributeError("memory backend search is unavailable")
                search.track_access(chunk_ids)

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
                "Memory recall failed while rendering context: %s. "
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

        Called during teardown(). Memories are already stored — if a
        contradiction is found the chunk is flagged (not deleted) so the
        next session or a human can resolve it. Evolutions trigger the same
        auto-replace as the former synchronous path.
        """
        import asyncio

        checker = self._checker
        backend = self.backend
        if checker is None or backend is None:
            return

        sem = asyncio.Semaphore(3)

        async def _check_one(chunk_id: str, content: str, importance: float):
            async with sem:
                try:
                    conflict = await checker.check(content, importance)
                    if conflict.status == "contradiction":
                        await backend.update_chunk_metadata(
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
                            await backend.delete_chunks([conflict.conflicting_chunk_id])
                        await backend.update_chunk_metadata(
                            chunk_id, {"_contradiction_checked": True}
                        )
                    else:
                        await backend.update_chunk_metadata(
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

    async def teardown(self):
        """Flush pending changes, prune stale memories, and auto-curate.

        MEMORY.md is a human-readable artifact for inspecting what the agent
        has learned. Agent context injection uses semantic recall via
        MemoryContextProvider, not MEMORY.md.
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
                await _maybe_await(self._memory_graph.flush())
            except Exception:
                pass

        backend = self.backend
        flush = _declared_method(backend, "flush")
        if flush is not None:
            await _maybe_await(flush())

        # Prune expired and over-limit memories
        prune = _declared_method(backend, "prune")
        if prune is not None:
            try:
                await _maybe_await(prune())
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
            else:
                regenerate_memory_md = _declared_method(backend, "regenerate_memory_md")
                path = (
                    await _maybe_await(regenerate_memory_md())
                    if regenerate_memory_md is not None
                    else None
                )
                if path is not None:
                    print(f"Memory summary written to: {path}")

    def get_pending_metrics(self) -> dict:
        if self.backend is None:
            return {"memory_count_delta": 0, "memory_retrieval_count": 0}
        get_pending_metrics = _declared_method(self.backend, "get_pending_metrics")
        if get_pending_metrics is not None:
            return get_pending_metrics()
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

        backend = self.backend
        if backend is None:
            raise RuntimeError("MemoryPlugin must be set up before updating memory")
        matches = await backend.recall(query, limit=100, score_threshold=0.7)
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
                await backend.update_chunk_metadata(
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
        backend = self.backend
        if backend is None:
            raise RuntimeError("MemoryPlugin must be set up before deleting memory")
        matches = await backend.recall(query, limit=100, score_threshold=0.7)
        if not matches:
            return {
                "status": "success",
                "deleted": 0,
                "message": "No matching memories found",
            }

        chunk_ids = [m["chunk_id"] for m in matches if m.get("chunk_id")]
        await backend.delete_chunks(chunk_ids)

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
