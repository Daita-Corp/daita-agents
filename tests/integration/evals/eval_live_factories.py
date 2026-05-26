"""Live eval factory targets used by integration tests."""

from __future__ import annotations

import os
from typing import Any

from daita.agents.agent import Agent
from daita.core.tools import tool
from daita.plugins.sqlite import SQLitePlugin


def _openai_kwargs(name: str, prompt: str | None = None) -> dict[str, Any]:
    kwargs = {
        "name": name,
        "llm_provider": "openai",
        "model": os.environ.get("OPENAI_TEST_MODEL", "gpt-5.4-mini"),
        "api_key": os.environ["OPENAI_API_KEY"],
        "temperature": 0,
        "max_tokens": 256,
    }
    if prompt:
        kwargs["prompt"] = prompt
    return kwargs


def create_plain_live_agent() -> Agent:
    return Agent(
        **_openai_kwargs(
            "LiveEvalPlain",
            "Follow the user's formatting request exactly and keep answers brief.",
        )
    )


def create_tool_live_agent() -> Agent:
    @tool(
        name="multiply",
        description="Multiply two integers. Use this whenever multiplication is requested.",
    )
    def multiply(a: int, b: int) -> int:
        """Multiply two integers.

        Args:
            a: First integer.
            b: Second integer.
        """
        return a * b

    return Agent(
        tools=[multiply],
        **_openai_kwargs(
            "LiveEvalTool",
            "When a relevant tool exists, call the tool before answering.",
        ),
    )


def create_skill_plugin_live_agent():
    agent = create_tool_live_agent()

    class SkillPluginTraceAgent:
        name = "LiveEvalSkillPlugin"
        agent_id = "live_eval_skill_plugin"
        llm = agent.llm

        async def start(self):
            start = getattr(agent, "start", None)
            if start is not None:
                result = start()
                if hasattr(result, "__await__"):
                    await result

        async def stop(self):
            stop = getattr(agent, "stop", None)
            if stop is not None:
                result = stop()
                if hasattr(result, "__await__"):
                    await result

        async def run(self, prompt: str, detailed: bool = False, **kwargs):
            result = await agent.run(prompt, detailed=True, **kwargs)
            if not isinstance(result, dict):
                result = {"result": str(result)}
            result.setdefault(
                "skill_calls",
                [
                    {
                        "name": "math_reasoning",
                        "operation": "plan",
                        "status": "passed",
                        "latency_ms": 35,
                    }
                ],
            )
            result.setdefault(
                "plugin_calls",
                [
                    {
                        "name": "calculator",
                        "operation": "multiply",
                        "status": "passed",
                        "latency_ms": 12,
                    }
                ],
            )
            for call in result.get("tool_calls", []):
                if isinstance(call, dict) and call.get("tool") == "multiply":
                    call.setdefault("skill", "math_reasoning")
                    call.setdefault("plugin", "calculator")
                    call.setdefault("operation", "multiply")
                    call.setdefault("latency_ms", 12)
            return result if detailed else result.get("result", "")

    return SkillPluginTraceAgent()


async def create_sqlite_sales_agent(db_path: str) -> Agent:
    db = SQLitePlugin(path=db_path, wal_mode=False, read_only=True)
    await db.connect()
    await db.execute_script("""
        CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            product TEXT NOT NULL,
            revenue REAL NOT NULL
        );
        INSERT INTO sales (product, revenue) VALUES
            ('Widget A', 8000.00),
            ('Widget A', 4840.50),
            ('Widget B', 4000.00),
            ('Widget C', 2500.00),
            ('Widget D', 1000.00);
        """)
    return Agent(
        tools=[db],
        **_openai_kwargs(
            "LiveEvalSQLite",
            (
                "You answer questions about the SQLite sales table. "
                "Use sqlite_query for calculations. Include an explicit LIMIT in SELECT queries. "
                "For top product revenue, aggregate with SUM(revenue) and GROUP BY product."
            ),
        ),
    )


def create_data_ops_live_agent() -> Agent:
    @tool(name="file_read", description="Read a local data file by path.")
    def file_read(path: str) -> dict[str, Any]:
        """Read a local data file.

        Args:
            path: File path to read.
        """
        return {"path": path, "rows": 12, "columns": ["customer_id", "revenue"]}

    @tool(name="rest_request", description="Make an HTTP request to a URL.")
    def rest_request(method: str, url: str) -> dict[str, Any]:
        """Make an HTTP request.

        Args:
            method: HTTP method such as GET.
            url: URL to request.
        """
        return {"method": method, "url": url, "status": 200}

    @tool(name="s3_get_object", description="Read an object from storage.")
    def s3_get_object(bucket: str, key: str) -> dict[str, Any]:
        """Read an object from a bucket.

        Args:
            bucket: Bucket name.
            key: Object key.
        """
        return {"bucket": bucket, "key": key, "bytes": 128}

    @tool(name="vector_search", description="Search a vector index with filters.")
    def vector_search(
        index: str, top_k: int, filters: dict[str, Any]
    ) -> dict[str, Any]:
        """Search a vector index.

        Args:
            index: Vector index name.
            top_k: Maximum number of matches.
            filters: Metadata filters to apply.
        """
        return {"index": index, "top_k": top_k, "filters": filters, "matches": []}

    return Agent(
        tools=[file_read, rest_request, s3_get_object, vector_search],
        **_openai_kwargs(
            "LiveEvalDataOps",
            (
                "Use each requested tool exactly once when the user asks for data operation checks. "
                "Do not use write, delete, POST, PUT, PATCH, or DELETE operations."
            ),
        ),
    )
