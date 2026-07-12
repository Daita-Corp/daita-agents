"""
Exa Search Plugin for Daita Agents using the Exa AI search API.

This plugin provides AI-powered web search capabilities to agents using Exa,
a search engine designed for LLMs that returns clean, semantically relevant
results with optional content retrieval (full text, highlights, summaries).

Features:
- Neural / fast / auto search modes with semantic ranking
- AI-extracted highlights and LLM-generated summaries on every result
- Domain include/exclude filtering and date range filtering
- Category filtering (research papers, news, companies, etc.)
- Find-similar lookups by URL
- Automatic error handling with DAITA error hierarchy

Usage:
    ```python
    from daita.plugins import exa_search
    from daita import Agent
    import os

    # Option 1: Use with agent
    agent = Agent(
        name="researcher",
        plugins=[exa_search(api_key=os.getenv("EXA_API_KEY"))],
        model="gpt-4o-mini"
    )

    await agent.start()
    result = await agent.run("What are the latest AI developments?")
    await agent.stop()

    # Option 2: Direct usage
    async with exa_search(api_key=os.getenv("EXA_API_KEY")) as search:
        results = await search.search("Python async best practices", num_results=5)
        for r in results['results']:
            print(f"- {r['title']}: {r['url']}")
            print(f"  {r['snippet']}")
    ```

Getting Started:
    1. Sign up at https://exa.ai
    2. Get your API key from the dashboard
    3. Set environment variable: export EXA_API_KEY=...
    4. Use the plugin in your agents
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Mapping, Optional, TYPE_CHECKING

from daita.runtime import (
    AccessMode,
    Capability,
    Evidence,
    EvidenceSchema,
    Operation,
    RiskLevel,
    Task,
    ToolView,
)

from .base import ConnectorPlugin
from .manifest import PluginKind, PluginManifest

if TYPE_CHECKING:
    from exa_py import Exa
from ..core.exceptions import (
    TransientError,
    RetryableError,
    PermanentError,
    RateLimitError,
    TimeoutError,
    ConnectionError as DaitaConnectionError,
    AuthenticationError,
)
from ..core.tools import LocalTool

logger = logging.getLogger(__name__)


def _required_string_arg(args: Dict[str, Any], field: str) -> str:
    value = args.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


VALID_SEARCH_TYPES = {"neural", "fast", "auto", "keyword"}
# Exa supports additional internal types ("deep", "deep-lite", etc.) but the
# four above are the stable, user-facing options exposed by this plugin.

VALID_CATEGORIES = {
    "company",
    "research paper",
    "news",
    "personal site",
    "financial report",
    "people",
}


def _search_web_parameters(num_results: int, search_type: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (e.g., 'Python async best practices')",
            },
            "num_results": {
                "type": "integer",
                "description": f"Number of results to return (optional, default: {num_results}, max: 100)",
            },
            "search_type": {
                "type": "string",
                "enum": sorted(VALID_SEARCH_TYPES),
                "description": f"Search mode (optional, default: {search_type})",
            },
            "category": {
                "type": "string",
                "enum": sorted(VALID_CATEGORIES),
                "description": "Restrict results to a category (optional)",
            },
            "include_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Only return results from these domains (optional)",
            },
            "exclude_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Skip results from these domains (optional)",
            },
            "start_published_date": {
                "type": "string",
                "description": "ISO 8601 earliest publish date (optional, e.g., '2025-01-01')",
            },
            "end_published_date": {
                "type": "string",
                "description": "ISO 8601 latest publish date (optional)",
            },
            "include_text": {
                "type": "boolean",
                "description": "Include full page text on each result (optional)",
            },
            "include_highlights": {
                "type": "boolean",
                "description": "Include AI-extracted highlight passages (optional)",
            },
            "include_summary": {
                "type": "boolean",
                "description": "Include an AI-generated summary per result (optional)",
            },
        },
        "required": ["query"],
    }


def _find_similar_parameters(num_results: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to find similar pages for (HTTP/HTTPS)",
            },
            "num_results": {
                "type": "integer",
                "description": f"Number of results (optional, default: {num_results})",
            },
            "include_text": {
                "type": "boolean",
                "description": "Include full page text on each result (optional)",
            },
            "include_highlights": {
                "type": "boolean",
                "description": "Include AI-extracted highlight passages (optional)",
            },
            "include_summary": {
                "type": "boolean",
                "description": "Include an AI-generated summary per result (optional)",
            },
        },
        "required": ["url"],
    }


def _tool_definitions(num_results: int, search_type: str) -> tuple[Dict[str, Any], ...]:
    return (
        {
            "name": "search_web",
            "capability_id": "exa_search.web.search",
            "operation_type": "web.search",
            "description": (
                "Search the web for information using Exa's AI-powered semantic search. "
                "Returns high-quality results with optional AI highlights or summaries. "
                "Supports filtering by domain, date range, and category "
                "(news, research paper, company, etc.). Use for: research, fact-checking, "
                "current information, technical documentation, finding sources."
            ),
            "parameters": _search_web_parameters(num_results, search_type),
        },
        {
            "name": "find_similar",
            "capability_id": "exa_search.web.find_similar",
            "operation_type": "web.find_similar",
            "description": (
                "Find pages on the web that are semantically similar to a given URL "
                "using Exa. Use for: discovering related sources, expanding research "
                "from a known good page, finding alternatives or comparisons."
            ),
            "parameters": _find_similar_parameters(num_results),
        },
    )


_TOOL_BY_CAPABILITY = {
    "exa_search.web.search": "search_web",
    "exa_search.web.find_similar": "find_similar",
}


class _ExaSearchExecutor:
    """Executor backing Exa search capability declarations."""

    id = "exa_search.web"
    capability_ids = frozenset(_TOOL_BY_CAPABILITY)

    def __init__(self, plugin: "ExaSearchPlugin") -> None:
        self._plugin = plugin

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        tool_name = _TOOL_BY_CAPABILITY.get(task.capability_id)
        if tool_name is None:
            raise ValueError(f"Unsupported Exa capability: {task.capability_id}")

        handler = self._plugin._tool_handler(tool_name)
        result = await handler(dict(task.input))
        tool_view = context.get("tool_view")
        tool_view_name = (
            tool_view.get("name") if isinstance(tool_view, Mapping) else None
        )
        return [
            Evidence(
                kind="web.search.results",
                owner="exa_search",
                operation_id=operation.id,
                task_id=task.id,
                payload={
                    "operation": tool_name,
                    "request": dict(task.input),
                    "response": result,
                },
                metadata={
                    "capability_id": task.capability_id,
                    "tool_view": tool_view_name,
                },
            )
        ]


@dataclass
class ExaSearchResult:
    """Typed view over a single Exa API result."""

    title: str
    url: str
    snippet: str
    score: float = 0.0
    published_date: Optional[str] = None
    author: Optional[str] = None
    text: Optional[str] = None
    highlights: List[str] = field(default_factory=list)
    summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "score": self.score,
            "published_date": self.published_date,
            "author": self.author,
            "text": self.text,
            "highlights": self.highlights,
            "summary": self.summary,
        }


def _extract_snippet(item: Dict[str, Any]) -> str:
    """Pick the best available snippet from an Exa result.

    Exa may return any combination of summary / highlights / text depending on
    the contents flags requested. Cascade through them so the consumer always
    gets a usable preview.
    """
    summary = item.get("summary")
    if summary:
        return summary

    highlights = item.get("highlights") or []
    if highlights:
        return " ... ".join(h for h in highlights if h)

    text = item.get("text")
    if text:
        return text[:500]

    return ""


def _result_from_item(item: Dict[str, Any]) -> ExaSearchResult:
    return ExaSearchResult(
        title=item.get("title") or "",
        url=item.get("url") or "",
        snippet=_extract_snippet(item),
        score=float(item.get("score") or 0.0),
        published_date=item.get("publishedDate"),
        author=item.get("author"),
        text=item.get("text"),
        highlights=list(item.get("highlights") or []),
        summary=item.get("summary"),
    )


class ExaSearchPlugin(ConnectorPlugin):
    """
    Exa search plugin using the Exa AI search API.

    Provides semantically ranked web search with optional content retrieval
    (highlights, summaries, full text) and rich filtering (domains, dates,
    categories).

    Args:
        api_key: Exa API key (or from EXA_API_KEY env var)
        num_results: Default number of results (default: 5, max: 100)
        search_type: "neural", "fast", "auto", or "keyword" (default: "auto")
        include_text: Include full page text in results (default: False)
        include_highlights: Include AI-extracted highlights (default: True)
        include_summary: Include AI-generated summary (default: False)
        max_text_chars: Max chars when include_text is True (default: 1000)
        category: Optional default category filter
        include_domains: Optional default domain include list
        exclude_domains: Optional default domain exclude list
    """

    manifest = PluginManifest(
        id="exa_search",
        display_name="Exa Search",
        version="2.0.0",
        kind=PluginKind.CONNECTOR,
        domains=frozenset({"web", "search"}),
        provides=frozenset({"web_search", "similarity_search"}),
        optional_dependencies=frozenset({"exa-py"}),
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        num_results: int = 5,
        search_type: str = "auto",
        include_text: bool = False,
        include_highlights: bool = True,
        include_summary: bool = False,
        max_text_chars: int = 1000,
        category: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        **kwargs,
    ):
        """Initialize ExaSearchPlugin with Exa configuration."""
        self._api_key = api_key or os.getenv("EXA_API_KEY")

        if not self._api_key or not self._api_key.strip():
            raise ValueError(
                "Exa API key is required. "
                "Provide via api_key parameter or EXA_API_KEY environment variable. "
                "Get a key at https://exa.ai"
            )

        if search_type not in VALID_SEARCH_TYPES:
            raise ValueError(
                f"search_type must be one of {sorted(VALID_SEARCH_TYPES)}, "
                f"got: {search_type}"
            )

        if category is not None and category not in VALID_CATEGORIES:
            raise ValueError(
                f"category must be one of {sorted(VALID_CATEGORIES)}, got: {category}"
            )

        self._num_results = num_results
        self._search_type = search_type
        self._include_text = include_text
        self._include_highlights = include_highlights
        self._include_summary = include_summary
        self._max_text_chars = max_text_chars
        self._category = category
        self._include_domains = include_domains
        self._exclude_domains = exclude_domains

        self._client: Exa | None = None
        self._executor = _ExaSearchExecutor(self)

        logger.info(
            f"ExaSearchPlugin initialized (type: {search_type}, "
            f"num_results: {num_results}, highlights: {include_highlights}, "
            f"summary: {include_summary})"
        )

    @property
    def is_connected(self) -> bool:
        """Whether the Exa client is currently initialized."""
        return self._client is not None

    async def teardown(self) -> None:
        """Release runtime-owned Exa resources."""
        await self.disconnect()

    def declare_capabilities(self) -> tuple[Capability, ...]:
        """Declare Exa search operations as runtime-plannable capabilities."""
        return tuple(
            Capability(
                id=definition["capability_id"],
                owner=self.manifest.id,
                description=definition["description"],
                domains=frozenset({"web", "search"}),
                operation_types=frozenset({definition["operation_type"]}),
                access=AccessMode.READ,
                risk=RiskLevel.LOW,
                input_schema=definition["parameters"],
                output_evidence=frozenset({"web.search.results"}),
                executor=self._executor.id,
                model_visible=True,
                retry_safe=True,
                idempotent=True,
                side_effecting=False,
                timeout_seconds=30,
                metadata={"tool_name": definition["name"]},
            )
            for definition in _tool_definitions(self._num_results, self._search_type)
        )

    def declare_evidence_schemas(self) -> tuple[EvidenceSchema, ...]:
        """Declare typed evidence returned by Exa search execution."""
        return (
            EvidenceSchema(
                kind="web.search.results",
                owner=self.manifest.id,
                description="Search results returned by Exa web search operations.",
                json_schema={
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string"},
                        "request": {"type": "object"},
                        "response": {"type": "object"},
                    },
                    "required": ["operation", "request", "response"],
                },
            ),
        )

    def get_executors(self) -> tuple[_ExaSearchExecutor, ...]:
        """Return the executor for Exa search runtime capabilities."""
        return (self._executor,)

    def get_tool_views(self) -> tuple[ToolView, ...]:
        """Expose Exa capabilities as model-visible tool views."""
        return tuple(
            ToolView(
                name=definition["name"],
                capability_id=definition["capability_id"],
                description=definition["description"],
                parameters=definition["parameters"],
            )
            for definition in _tool_definitions(self._num_results, self._search_type)
        )

    async def connect(self):
        """Initialize the Exa client."""
        if self._client is not None:
            return

        try:
            from exa_py import Exa

            client = Exa(api_key=self._api_key)
            # Tag API calls so Exa can attribute usage to this integration.
            client.headers["x-exa-integration"] = "daita-agents"
            self._client = client
            logger.info("Connected to Exa Search API")
        except ImportError as e:
            raise ImportError(
                "exa-py not installed. Install with: pip install 'daita-agents[exa]'"
            ) from e

    async def disconnect(self):
        """Cleanup the Exa client."""
        self._client = None
        logger.info("Disconnected from Exa Search API")

    @property
    def client(self) -> Exa:
        if self._client is None:
            raise RuntimeError("Exa plugin is not connected")
        return self._client

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    def _build_contents(
        self,
        include_text: Optional[bool],
        include_highlights: Optional[bool],
        include_summary: Optional[bool],
    ) -> Optional[Dict[str, Any]]:
        """Build the ``contents`` payload for the Exa API.

        Each content type can be requested independently. The Exa API accepts
        them simultaneously, so this returns a dict combining whichever flags
        are enabled, or ``None`` when none are.
        """
        text_on = self._include_text if include_text is None else include_text
        highlights_on = (
            self._include_highlights
            if include_highlights is None
            else include_highlights
        )
        summary_on = (
            self._include_summary if include_summary is None else include_summary
        )

        contents: Dict[str, Any] = {}
        if text_on:
            contents["text"] = {"maxCharacters": self._max_text_chars}
        if highlights_on:
            contents["highlights"] = True
        if summary_on:
            contents["summary"] = True

        return contents or None

    async def search(
        self,
        query: str,
        num_results: Optional[int] = None,
        search_type: Optional[str] = None,
        category: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_text: Optional[bool] = None,
        include_highlights: Optional[bool] = None,
        include_summary: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Search the web using Exa's semantic search.

        Args:
            query: Search query
            num_results: Number of results (default: from constructor)
            search_type: Override default search type
            category: Filter by category (e.g. "news", "research paper")
            include_domains: Only return results from these domains
            exclude_domains: Skip results from these domains
            start_published_date: ISO 8601 lower bound for publish date
            end_published_date: ISO 8601 upper bound for publish date
            include_text: Override default text inclusion
            include_highlights: Override default highlight inclusion
            include_summary: Override default summary inclusion

        Returns:
            Dict with keys: success, query, results, count
        """
        if not self._client:
            await self.connect()

        num_results = num_results if num_results is not None else self._num_results
        search_type = search_type or self._search_type
        category = category if category is not None else self._category
        include_domains = (
            include_domains if include_domains is not None else self._include_domains
        )
        exclude_domains = (
            exclude_domains if exclude_domains is not None else self._exclude_domains
        )

        if search_type not in VALID_SEARCH_TYPES:
            raise ValueError(
                f"search_type must be one of {sorted(VALID_SEARCH_TYPES)}, "
                f"got: {search_type}"
            )
        if category is not None and category not in VALID_CATEGORIES:
            raise ValueError(
                f"category must be one of {sorted(VALID_CATEGORIES)}, got: {category}"
            )

        contents = self._build_contents(
            include_text, include_highlights, include_summary
        )

        kwargs: Dict[str, Any] = {
            "query": query,
            "num_results": num_results,
            "type": search_type,
        }
        if category:
            kwargs["category"] = category
        if include_domains:
            kwargs["include_domains"] = include_domains
        if exclude_domains:
            kwargs["exclude_domains"] = exclude_domains
        if start_published_date:
            kwargs["start_published_date"] = start_published_date
        if end_published_date:
            kwargs["end_published_date"] = end_published_date

        try:
            if contents is not None:
                response = await self._call_search_with_contents(kwargs, contents)
            else:
                response = await self._call_search(kwargs)

            items = self._coerce_items(response)
            results = [_result_from_item(item).to_dict() for item in items]

            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results),
            }

        except Exception as e:
            logger.error(f"Exa search failed: {e}")
            raise self._handle_search_error(e)

    async def find_similar(
        self,
        url: str,
        num_results: Optional[int] = None,
        include_text: Optional[bool] = None,
        include_highlights: Optional[bool] = None,
        include_summary: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Find pages semantically similar to a given URL.

        Args:
            url: URL to find similar pages for
            num_results: Number of results (default: from constructor)
            include_text: Override default text inclusion
            include_highlights: Override default highlight inclusion
            include_summary: Override default summary inclusion

        Returns:
            Dict with keys: success, url, results, count
        """
        if not self._client:
            await self.connect()

        num_results = num_results if num_results is not None else self._num_results
        contents = self._build_contents(
            include_text, include_highlights, include_summary
        )

        try:
            if contents is not None:
                response = await self._call_find_similar_with_contents(
                    url, num_results, contents
                )
            else:
                response = await self._call_find_similar(url, num_results)

            items = self._coerce_items(response)
            results = [_result_from_item(item).to_dict() for item in items]

            return {
                "success": True,
                "url": url,
                "results": results,
                "count": len(results),
            }

        except Exception as e:
            logger.error(f"Exa find_similar failed for {url}: {e}")
            raise self._handle_search_error(e)

    async def _call_search(self, kwargs: Dict[str, Any]) -> Any:
        import asyncio

        return await asyncio.to_thread(self.client.search, **kwargs)

    async def _call_search_with_contents(
        self, kwargs: Dict[str, Any], contents: Dict[str, Any]
    ) -> Any:
        import asyncio

        return await asyncio.to_thread(
            self.client.search_and_contents, **kwargs, **contents
        )

    async def _call_find_similar(self, url: str, num_results: int) -> Any:
        import asyncio

        return await asyncio.to_thread(
            self.client.find_similar, url=url, num_results=num_results
        )

    async def _call_find_similar_with_contents(
        self, url: str, num_results: int, contents: Dict[str, Any]
    ) -> Any:
        import asyncio

        return await asyncio.to_thread(
            self.client.find_similar_and_contents,
            url=url,
            num_results=num_results,
            **contents,
        )

    @staticmethod
    def _coerce_items(response: Any) -> List[Dict[str, Any]]:
        """Normalize the SDK response into a list of plain dicts.

        ``exa-py`` returns a ``SearchResponse`` whose ``results`` are typed
        objects. We convert them to dicts so downstream extraction stays simple.
        """
        results = getattr(response, "results", None)
        if results is None and isinstance(response, dict):
            results = response.get("results", [])
        if results is None:
            return []

        items: List[Dict[str, Any]] = []
        for r in results:
            if isinstance(r, dict):
                items.append(r)
            elif hasattr(r, "model_dump"):
                items.append(r.model_dump())
            elif hasattr(r, "__dict__"):
                items.append(
                    {k: v for k, v in vars(r).items() if not k.startswith("_")}
                )
        return items

    def _handle_search_error(self, e: Exception) -> Exception:
        """Convert Exa API errors to DAITA error hierarchy."""
        error_msg = str(e).lower()

        if (
            "rate limit" in error_msg
            or "429" in error_msg
            or "too many requests" in error_msg
        ):
            return RateLimitError(
                message=f"Exa API rate limit exceeded: {e}",
                retry_after=60,
            )

        if (
            "unauthorized" in error_msg
            or "401" in error_msg
            or "invalid api key" in error_msg
        ):
            return AuthenticationError(
                message=f"Invalid Exa API key. Get one at https://exa.ai: {e}"
            )

        if "forbidden" in error_msg or "403" in error_msg:
            return PermanentError(message=f"Exa API access forbidden: {e}")

        if "bad request" in error_msg or "400" in error_msg:
            return PermanentError(message=f"Invalid search request: {e}")

        if "timeout" in error_msg or "timed out" in error_msg:
            return TimeoutError(
                message=f"Exa API request timed out: {e}", timeout_duration=30
            )

        if "connection" in error_msg or "network" in error_msg:
            return DaitaConnectionError(message=f"Connection to Exa API failed: {e}")

        if "503" in error_msg or "unavailable" in error_msg:
            return TransientError(message=f"Exa API temporarily unavailable: {e}")

        return RetryableError(message=f"Exa search error: {e}")

    # Tool handlers

    async def _tool_search_web(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for search_web."""
        return await self.search(
            query=_required_string_arg(args, "query"),
            num_results=args.get("num_results"),
            search_type=args.get("search_type"),
            category=args.get("category"),
            include_domains=args.get("include_domains"),
            exclude_domains=args.get("exclude_domains"),
            start_published_date=args.get("start_published_date"),
            end_published_date=args.get("end_published_date"),
            include_text=args.get("include_text"),
            include_highlights=args.get("include_highlights"),
            include_summary=args.get("include_summary"),
        )

    async def _tool_find_similar(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for find_similar."""
        return await self.find_similar(
            url=_required_string_arg(args, "url"),
            num_results=args.get("num_results"),
            include_text=args.get("include_text"),
            include_highlights=args.get("include_highlights"),
            include_summary=args.get("include_summary"),
        )

    def _tool_handler(self, name: str):
        """Return the legacy tool handler for an Exa tool name."""
        handlers = {
            "search_web": self._tool_search_web,
            "find_similar": self._tool_find_similar,
        }
        return handlers[name]


def exa_search(**kwargs) -> ExaSearchPlugin:
    """
    Create ExaSearchPlugin with simplified interface.

    Args:
        api_key: Exa API key (or from EXA_API_KEY env var)
        num_results: Default number of results (default: 5)
        search_type: "neural", "fast", "auto", or "keyword" (default: "auto")
        include_text: Include full page text (default: False)
        include_highlights: Include AI highlights (default: True)
        include_summary: Include AI summary (default: False)
        max_text_chars: Max chars for full text (default: 1000)
        category: Default category filter (optional)
        include_domains: Default domain include list (optional)
        exclude_domains: Default domain exclude list (optional)

    Returns:
        ExaSearchPlugin instance

    Example:
        ```python
        from daita.plugins import exa_search
        import os

        # API key from environment
        search = exa_search()

        # Or pass directly with custom defaults
        search = exa_search(api_key="...", search_type="neural", num_results=10)

        # Use with agent
        from daita import Agent
        agent = Agent(
            name="researcher",
            plugins=[exa_search()],
            model="gpt-4o-mini"
        )
        ```
    """
    return ExaSearchPlugin(**kwargs)
