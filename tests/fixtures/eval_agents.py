"""Factories used by eval unit tests."""

from __future__ import annotations


class FakeEvalAgent:
    def __init__(self, responses):
        self.responses = list(responses)
        self.index = 0
        self.started = False
        self.stopped = False
        self.name = "Fake Eval Agent"
        self.agent_id = "fake_eval_agent"

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    async def run(self, prompt, detailed=False, **kwargs):
        if self.index < len(self.responses):
            result = self.responses[self.index]
            self.index += 1
        else:
            result = self.responses[-1]
        return result if detailed else result["result"]


def create_passing_agent():
    return FakeEvalAgent(
        [
            {
                "result": "Widget A revenue: 12840.50",
                "tool_calls": [
                    {
                        "tool": "sqlite_query",
                        "arguments": {
                            "sql": "SELECT product, SUM(revenue) FROM sales GROUP BY product LIMIT 5"
                        },
                        "result": {
                            "rows": [{"product": "Widget A", "revenue": 12840.50}]
                        },
                    }
                ],
                "iterations": 2,
                "tokens": {"total_tokens": 100},
                "cost": 0.001,
                "processing_time_ms": 12,
            }
        ]
    )


def create_sql_failure_agent():
    return FakeEvalAgent(
        [
            {
                "result": "Done",
                "tool_calls": [
                    {
                        "tool": "sqlite_query",
                        "arguments": {"sql": "SELECT * FROM users_pii"},
                        "result": {"rows": []},
                    }
                ],
                "iterations": 1,
            }
        ]
    )


def create_data_ops_agent():
    return FakeEvalAgent(
        [
            {
                "result": "Loaded sales.csv and checked the customer API.",
                "tool_calls": [
                    {
                        "tool": "file_read",
                        "arguments": {"path": "data/sales.csv"},
                        "result": {"rows": 10},
                    },
                    {
                        "tool": "rest_request",
                        "arguments": {
                            "method": "GET",
                            "url": "https://api.example.com/customers",
                        },
                        "result": {"status": 200},
                    },
                    {
                        "tool": "s3_get_object",
                        "arguments": {"bucket": "analytics", "key": "sales.csv"},
                        "result": {"bytes": 128},
                    },
                    {
                        "tool": "vector_search",
                        "arguments": {
                            "index": "docs",
                            "top_k": 5,
                            "filters": {"tenant_id": "acme"},
                        },
                        "result": {"matches": []},
                    },
                ],
                "iterations": 1,
            }
        ]
    )


def create_execution_agent():
    return FakeEvalAgent(
        [
            {
                "result": "Schema discovery and SQLite query completed.",
                "tool_calls": [
                    {
                        "tool": "sqlite_query",
                        "plugin": "sqlite",
                        "skill": "schema_discovery",
                        "operation": "query",
                        "arguments": {"sql": "SELECT product FROM sales LIMIT 5"},
                        "result": {"rows": [{"product": "Widget A"}]},
                        "latency_ms": 18,
                    }
                ],
                "skill_calls": [
                    {
                        "name": "schema_discovery",
                        "operation": "inspect",
                        "latency_ms": 25,
                        "status": "passed",
                    }
                ],
                "plugin_calls": [
                    {
                        "name": "sqlite",
                        "operation": "query",
                        "latency_ms": 18,
                        "status": "passed",
                    }
                ],
                "iterations": 1,
            }
        ]
    )


def create_unstable_agent():
    return FakeEvalAgent(
        [
            {
                "result": "Answer A",
                "tool_calls": [{"tool": "file_read", "arguments": {"path": "a.csv"}}],
            },
            {
                "result": "Answer B",
                "tool_calls": [
                    {
                        "tool": "api_get",
                        "arguments": {"url": "https://api.example.com/b"},
                    }
                ],
            },
        ]
    )
