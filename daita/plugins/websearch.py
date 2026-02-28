"""
Web Search Plugin for Daita Agents using Tavily Search API.

This plugin provides AI-optimized web search capabilities to agents using Tavily,
the industry-standard search API for AI agents. Tavily provides LLM-optimized results
with automatic answer extraction and relevance scoring.

Features:
- General web search with AI-extracted answers
- News search with date filtering
- URL content extraction (fallback tool)
- Automatic error handling with DAITA error hierarchy
- Generous free tier: 1000 searches/month

Usage:
    ```python
    from daita.plugins import websearch
    from daita import Agent
    import os

    # Option 1: Use with agent
    agent = Agent(
        name="researcher",
        tools=[websearch(api_key=os.getenv("TAVILY_API_KEY"))],
        model="gpt-4o-mini"
    )

    await agent.start()
    result = await agent.run("What are the latest AI developments?")
    await agent.stop()

    # Option 2: Direct usage
    async with websearch(api_key=os.getenv("TAVILY_API_KEY")) as search:
        results = await search.search("Python async best practices", max_results=5)
        print(f"Answer: {results['answer']}")
        for r in results['results']:
            print(f"- {r['title']}: {r['url']}")
    ```

Getting Started:
    1. Sign up at https://tavily.com (free account)
    2. Get your API key from the dashboard
    3. Set environment variable: export TAVILY_API_KEY=tvly-xxxxx
    4. Use the plugin in your agents

Cost:
    - Free tier: 1000 searches/month
    - Paid tier: $0.002/search ($2 per 1000)
"""

import os
import logging
from typing import List, Dict, Any, Optional

from .base import BasePlugin
from ..core.exceptions import (
    TransientError,
    RetryableError,
    PermanentError,
    RateLimitError,
    TimeoutError,
    ConnectionError as DaitaConnectionError,
    AuthenticationError,
)

logger = logging.getLogger(__name__)


class WebSearchPlugin(BasePlugin):
    """
    Web search plugin using Tavily Search API.

    Provides AI-optimized web search with automatic answer extraction,
    news search with date filtering, and URL content extraction.

    Args:
        api_key: Tavily API key (or from TAVILY_API_KEY env var)
        max_results: Default number of results to return (default: 5)
        search_depth: Search depth - "basic" or "advanced" (default: "basic")
        include_answer: Include AI-extracted answer (default: True)
        include_raw_content: Include full page HTML (default: False)
        max_page_length: Max characters for fetch_page (default: 10000)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 5,
        search_depth: str = "basic",
        include_answer: bool = True,
        include_raw_content: bool = False,
        max_page_length: int = 10000,
        **kwargs
    ):
        """Initialize WebSearch plugin with Tavily configuration."""
        # Get API key from parameter or environment variable
        self._api_key = api_key or os.getenv("TAVILY_API_KEY")

        if not self._api_key or not self._api_key.strip():
            raise ValueError(
                "Tavily API key is required. "
                "Provide via api_key parameter or TAVILY_API_KEY environment variable. "
                "Get a free key at https://tavily.com"
            )

        # Validate search depth
        if search_depth not in ["basic", "advanced"]:
            raise ValueError(f"search_depth must be 'basic' or 'advanced', got: {search_depth}")

        # Store configuration
        self._max_results = max_results
        self._search_depth = search_depth
        self._include_answer = include_answer
        self._include_raw_content = include_raw_content
        self._max_page_length = max_page_length

        # Initialize state
        self._client = None
        self._session = None  # For fetch_page

        logger.info(f"WebSearchPlugin initialized (depth: {search_depth}, max_results: {max_results})")

    async def connect(self):
        """Initialize Tavily client and HTTP session."""
        if self._client is not None:
            return  # Already connected

        try:
            # Lazy import of Tavily client
            from tavily import TavilyClient

            # Initialize Tavily client
            self._client = TavilyClient(api_key=self._api_key)

            # Initialize aiohttp session for fetch_page
            import aiohttp
            self._session = aiohttp.ClientSession()

            logger.info("Connected to Tavily Search API")
        except ImportError as e:
            if "tavily" in str(e):
                raise RuntimeError(
                    "tavily-python not installed. Run: pip install tavily-python"
                )
            elif "aiohttp" in str(e):
                raise RuntimeError(
                    "aiohttp not installed. Run: pip install aiohttp"
                )
            else:
                raise

    async def disconnect(self):
        """Close HTTP session and cleanup."""
        if self._session:
            await self._session.close()
            self._session = None

        self._client = None
        logger.info("Disconnected from Tavily Search API")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        include_answer: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Search the web using Tavily's AI-optimized search.

        Args:
            query: Search query
            max_results: Number of results (default: from constructor)
            include_answer: Include AI-extracted answer (default: from constructor)

        Returns:
            Dict with keys: success, query, answer, results, count
        """
        if not self._client:
            await self.connect()

        max_results = max_results if max_results is not None else self._max_results
        include_answer = include_answer if include_answer is not None else self._include_answer

        try:
            # Call Tavily search API
            response = self._client.search(
                query=query,
                max_results=max_results,
                search_depth=self._search_depth,
                include_answer=include_answer,
                include_raw_content=self._include_raw_content
            )

            # Extract results
            results = []
            for item in response.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0.0),
                    "published_date": item.get("published_date")
                })

            return {
                "success": True,
                "query": query,
                "answer": response.get("answer", ""),
                "results": results,
                "count": len(results)
            }

        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            raise self._handle_search_error(e)

    async def search_news(
        self,
        query: str,
        days: int = 7,
        max_results: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Search for recent news articles.

        Args:
            query: Search query
            days: How many days back to search (default: 7)
            max_results: Number of results (default: from constructor)

        Returns:
            Dict with keys: success, query, results, count
        """
        if not self._client:
            await self.connect()

        max_results = max_results if max_results is not None else self._max_results

        try:
            # Call Tavily search API with news topic
            response = self._client.search(
                query=query,
                topic="news",
                days=days,
                max_results=max_results,
                search_depth=self._search_depth
            )

            # Extract results
            results = []
            for item in response.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0.0),
                    "published_date": item.get("published_date")
                })

            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results)
            }

        except Exception as e:
            logger.error(f"Tavily news search failed: {e}")
            raise self._handle_search_error(e)

    async def fetch_page(self, url: str) -> Dict[str, Any]:
        """
        Fetch and extract text content from a URL.

        Fallback tool for when Tavily doesn't have the page cached.
        Uses aiohttp + BeautifulSoup to extract clean text content.

        Args:
            url: URL to fetch

        Returns:
            Dict with keys: success, url, content, length, truncated
        """
        if not self._session:
            await self.connect()

        try:
            # Fetch URL
            async with self._session.get(url, timeout=30) as response:
                response.raise_for_status()
                html = await response.text()

            # Parse HTML and extract text
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "lxml")

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Get text
            text = soup.get_text()

            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

            # Truncate if needed
            truncated = len(text) > self._max_page_length
            if truncated:
                text = text[:self._max_page_length]

            return {
                "success": True,
                "url": url,
                "content": text,
                "length": len(text),
                "truncated": truncated
            }

        except Exception as e:
            logger.error(f"Failed to fetch page {url}: {e}")
            raise self._handle_fetch_error(e, url)

    def _handle_search_error(self, e: Exception) -> Exception:
        """Convert Tavily API errors to DAITA error hierarchy."""
        error_msg = str(e).lower()

        # Rate limiting (429)
        if "rate limit" in error_msg or "429" in error_msg or "too many requests" in error_msg:
            return RateLimitError(
                message=f"Tavily API rate limit exceeded: {e}",
                retry_after=60  # Assume 1 minute
            )

        # Authentication errors (401, 403)
        if "unauthorized" in error_msg or "401" in error_msg or "invalid api key" in error_msg:
            return AuthenticationError(
                message=f"Invalid Tavily API key. Get one at https://tavily.com: {e}"
            )

        if "forbidden" in error_msg or "403" in error_msg:
            return PermanentError(
                message=f"Tavily API access forbidden: {e}"
            )

        # Bad requests (400)
        if "bad request" in error_msg or "400" in error_msg:
            return PermanentError(
                message=f"Invalid search query: {e}"
            )

        # Timeout errors
        if "timeout" in error_msg or "timed out" in error_msg:
            return TimeoutError(
                message=f"Tavily API request timed out: {e}",
                timeout_duration=30
            )

        # Connection errors
        if "connection" in error_msg or "network" in error_msg:
            return DaitaConnectionError(
                message=f"Connection to Tavily API failed: {e}"
            )

        # Service unavailable (503)
        if "503" in error_msg or "unavailable" in error_msg:
            return TransientError(
                message=f"Tavily API temporarily unavailable: {e}"
            )

        # Default to retryable error
        return RetryableError(
            message=f"Tavily search error: {e}"
        )

    def _handle_fetch_error(self, e: Exception, url: str) -> Exception:
        """Convert fetch_page errors to DAITA error hierarchy."""
        error_msg = str(e).lower()

        # Timeout errors
        if "timeout" in error_msg or "timed out" in error_msg:
            return TimeoutError(
                message=f"Timeout fetching {url}: {e}",
                timeout_duration=30
            )

        # Connection errors
        if "connection" in error_msg or "cannot connect" in error_msg:
            return DaitaConnectionError(
                message=f"Connection failed for {url}: {e}"
            )

        # HTTP errors
        if "404" in error_msg:
            return PermanentError(
                message=f"URL not found: {url}"
            )

        if "403" in error_msg or "forbidden" in error_msg:
            return PermanentError(
                message=f"Access forbidden for {url}: {e}"
            )

        # Default to retryable error
        return RetryableError(
            message=f"Failed to fetch {url}: {e}"
        )

    # Tool handlers
    async def _tool_search_web(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for search_web."""
        query = args.get("query")
        max_results = args.get("max_results")
        include_answer = args.get("include_answer")

        return await self.search(query, max_results, include_answer)

    async def _tool_search_news(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for search_news."""
        query = args.get("query")
        days = args.get("days", 7)
        max_results = args.get("max_results")

        return await self.search_news(query, days, max_results)

    async def _tool_fetch_page(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for fetch_page."""
        url = args.get("url")
        return await self.fetch_page(url)

    def get_tools(self) -> List['AgentTool']:
        """Expose web search operations as agent tools."""
        from ..core.tools import AgentTool

        return [
            AgentTool(
                name="search_web",
                description=(
                    "Search the web for information using Tavily's AI-optimized search. "
                    "Returns an AI-extracted direct answer to the query plus LLM-optimized "
                    "search results with relevance scores. Use for: research, fact-checking, "
                    "current information, technical documentation, how-to guides, etc."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (e.g., 'Python async best practices', 'what is quantum computing')"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": f"Number of results to return (optional, default: {self._max_results})"
                        },
                        "include_answer": {
                            "type": "boolean",
                            "description": f"Include AI-extracted direct answer (optional, default: {self._include_answer})"
                        }
                    },
                    "required": ["query"]
                },
                handler=self._tool_search_web,
                category="search",
                source="plugin",
                plugin_name="WebSearch",
                timeout_seconds=30
            ),

            AgentTool(
                name="search_news",
                description=(
                    "Search for recent news articles using Tavily. Returns news articles "
                    "with published dates and relevance scores. Use for: latest developments, "
                    "recent announcements, current events, trending topics, breaking news."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "News search query (e.g., 'AI developments', 'Tesla earnings')"
                        },
                        "days": {
                            "type": "integer",
                            "description": "How many days back to search (optional, default: 7)"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": f"Number of results to return (optional, default: {self._max_results})"
                        }
                    },
                    "required": ["query"]
                },
                handler=self._tool_search_news,
                category="search",
                source="plugin",
                plugin_name="WebSearch",
                timeout_seconds=30
            ),

            AgentTool(
                name="fetch_page",
                description=(
                    "Fetch and extract clean text content from a specific URL. "
                    "Use as a fallback when you have a specific URL to read but Tavily "
                    "doesn't have it cached. Returns plain text content extracted from HTML."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to fetch (must be a valid HTTP/HTTPS URL)"
                        }
                    },
                    "required": ["url"]
                },
                handler=self._tool_fetch_page,
                category="search",
                source="plugin",
                plugin_name="WebSearch",
                timeout_seconds=30
            )
        ]


def websearch(**kwargs) -> WebSearchPlugin:
    """
    Create WebSearch plugin with simplified interface.

    Args:
        api_key: Tavily API key (or from TAVILY_API_KEY env var)
        max_results: Default number of results (default: 5)
        search_depth: "basic" or "advanced" (default: "basic")
        include_answer: Include AI answer (default: True)
        include_raw_content: Include full HTML (default: False)
        max_page_length: Max chars for fetch_page (default: 10000)

    Returns:
        WebSearchPlugin instance

    Example:
        ```python
        from daita.plugins import websearch
        import os

        # API key from environment
        search = websearch()

        # Or pass directly
        search = websearch(api_key="tvly-xxxxx", max_results=10)

        # Use with agent
        from daita import Agent
        agent = Agent(
            name="researcher",
            tools=[websearch()],
            model="gpt-4o-mini"
        )
        ```
    """
    return WebSearchPlugin(**kwargs)
