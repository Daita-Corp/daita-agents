"""
Unit tests for Agent execution loop (daita/agents/agent.py).

Covers:
  - run() returns string, run(detailed=True) returns dict with expected keys
  - System prompt injection
  - Max iterations limit raises AgentError
  - Tool calling: tool is executed, result appears in run(detailed=True)
  - JSON serializer handles datetime / Decimal / UUID / bytes
  - on_event callback receives ITERATION and COMPLETE events
"""

import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any, List
from uuid import uuid4

import pytest

from daita.agents.agent import Agent
from daita.agents.db.runtime.orchestrator import DbRunContract, DbRunOrchestrator
from daita.agents.db.runtime.state import DbRunState, set_db_run_state
from daita.agents.runtime.contextvars import get_active_run_state
from daita.core.exceptions import AgentError
from daita.core.streaming import EventType
from daita.core.tools import AgentTool
from daita.core.tracing import get_trace_manager
from daita.config.settings import Settings
from daita.llm.mock import MockLLMProvider
from daita.plugins.base import LifecyclePlugin

from tests.conftest import SequentialMockLLM

# ===========================================================================
# Helpers
# ===========================================================================


def _make_agent(responses: List[Any], tools=None) -> Agent:
    """Create an Agent with a SequentialMockLLM loaded with the given responses."""
    llm = SequentialMockLLM(response_sequence=responses)
    return Agent(name="ExecAgent", llm_provider=llm, tools=tools or [])


def _add_tool():
    """Return an 'add' AgentTool used in tool-calling tests."""

    async def h(args):
        return args["a"] + args["b"]

    return AgentTool(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First"},
                "b": {"type": "integer", "description": "Second"},
            },
            "required": ["a", "b"],
        },
        handler=h,
    )


# ===========================================================================
# Basic execution results
# ===========================================================================


class TestBasicExecution:
    async def test_run_returns_string(self):
        agent = _make_agent(["Hello from the agent."])
        result = await agent.run("Say hi")
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_run_detailed_returns_dict(self):
        agent = _make_agent(["Hi there."])
        result = await agent.run("Say hi", detailed=True)
        assert isinstance(result, dict)

    async def test_run_detailed_has_result_key(self):
        agent = _make_agent(["Answer text."])
        result = await agent.run("prompt", detailed=True)
        assert "result" in result
        assert isinstance(result["result"], str)

    async def test_run_detailed_has_iterations_key(self):
        agent = _make_agent(["Done."])
        result = await agent.run("prompt", detailed=True)
        assert "iterations" in result
        assert result["iterations"] >= 1

    async def test_run_detailed_has_tool_calls_key(self):
        agent = _make_agent(["Done."])
        result = await agent.run("prompt", detailed=True)
        assert "tool_calls" in result
        assert isinstance(result["tool_calls"], list)

    async def test_run_detailed_has_tokens_key(self):
        agent = _make_agent(["Done."])
        result = await agent.run("prompt", detailed=True)
        assert "tokens" in result

    async def test_run_detailed_has_cost_key(self):
        agent = _make_agent(["Done."])
        result = await agent.run("prompt", detailed=True)
        assert "cost" in result

    async def test_run_detailed_has_agent_id(self):
        agent = _make_agent(["Done."])
        result = await agent.run("prompt", detailed=True)
        assert result.get("agent_id") == agent.agent_id

    async def test_run_detailed_has_agent_name(self):
        agent = _make_agent(["Done."])
        result = await agent.run("prompt", detailed=True)
        assert result.get("agent_name") == agent.name

    async def test_run_no_llm_raises_agent_error(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(
            Settings, "get_llm_api_key", lambda _settings, _provider: None
        )
        agent = Agent(name="NoLLM", llm_provider="openai")
        # No API key → llm property returns None
        with pytest.raises(AgentError, match="No API key"):
            await agent.run("hello")


# ===========================================================================
# System prompt injection
# ===========================================================================


class TestSystemPromptInjection:
    async def test_system_prompt_sent_in_conversation(self):
        llm = SequentialMockLLM(response_sequence=["Done."])
        agent = Agent(name="X", llm_provider=llm, prompt="You are a helpful assistant.")
        await agent.run("hi")
        # The first call's messages should contain a system message
        first_call = llm.call_history[0]
        messages = first_call["messages"]
        roles = [m["role"] for m in messages]
        assert "system" in roles

    async def test_system_prompt_content_is_injected(self):
        llm = SequentialMockLLM(response_sequence=["Done."])
        agent = Agent(name="X", llm_provider=llm, prompt="Be concise.")
        await agent.run("hi")
        messages = llm.call_history[0]["messages"]
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert any("Be concise." in m["content"] for m in system_msgs)

    async def test_no_system_prompt_when_none(self):
        llm = SequentialMockLLM(response_sequence=["Done."])
        agent = Agent(name="X", llm_provider=llm)  # no prompt
        await agent.run("hi")
        messages = llm.call_history[0]["messages"]
        roles = [m["role"] for m in messages]
        assert "system" not in roles


# ===========================================================================
# Max iterations
# ===========================================================================


class TestMaxIterations:
    async def test_max_iterations_raises_agent_error(self):
        # LLM always returns tool calls; no tool registered so the result
        # is an error message, but the loop still repeats.  After
        # max_iterations the agent should raise AgentError.
        always_tool_call = [
            {
                "content": "Calling tool...",
                "tool_calls": [{"id": "tc", "name": "nonexistent", "arguments": {}}],
            }
        ] * 10  # more than max_iterations

        agent = _make_agent(always_tool_call)
        with pytest.raises(AgentError, match="Max iterations"):
            await agent.run("go", max_iterations=2)

    async def test_partial_exit_returns_evidence_backed_result_when_enabled(self):
        async def lookup(args):
            return {"answer": 7}

        tool = AgentTool(
            name="lookup",
            description="Lookup a value",
            parameters={"type": "object", "properties": {}},
            handler=lookup,
        )
        repeated_call = {
            "content": "Looking.",
            "tool_calls": [{"id": "tc", "name": "lookup", "arguments": {}}],
        }
        agent = _make_agent([repeated_call], tools=[tool])

        result = await agent.run(
            "find the answer",
            max_iterations=1,
            detailed=True,
            partial_exit=True,
        )

        assert result["partial"] is True
        assert "Partial result based on evidence" in result["result"]
        assert "lookup({}) returned {answer=7}" in result["result"]
        assert result["diagnostics"]["exit_reason"] == "max_iterations_partial"
        assert result["diagnostics"]["evidence_count"] == 1

    async def test_partial_exit_is_plain_string_when_not_detailed(self):
        async def lookup(args):
            return {"answer": 9}

        tool = AgentTool(
            name="lookup",
            description="Lookup a value",
            parameters={"type": "object", "properties": {}},
            handler=lookup,
        )
        repeated_call = {
            "content": "Looking.",
            "tool_calls": [{"id": "tc", "name": "lookup", "arguments": {}}],
        }
        agent = _make_agent([repeated_call], tools=[tool])

        result = await agent.run(
            "find the answer",
            max_iterations=1,
            partial_exit=True,
        )

        assert isinstance(result, str)
        assert "answer=9" in result

    async def test_terminal_tool_result_adds_final_synthesis_guidance(self):
        async def query(args):
            return {"rows": [{"answer": 7}]}

        tool = AgentTool(
            name="db_query",
            description="Run a database query",
            parameters={"type": "object", "properties": {}},
            handler=query,
        )
        llm = SequentialMockLLM(
            response_sequence=[
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "db_query",
                            "arguments": {"sql": "select"},
                        }
                    ],
                },
                "The answer is 7.",
            ]
        )
        agent = Agent(name="T", llm_provider=llm, tools=[tool])

        result = await agent.run(
            "answer from db",
            detailed=True,
            final_synthesis_without_tools=True,
            terminal_tools={"db_query"},
        )

        assert result["result"] == "The answer is 7."
        second_call = llm.call_history[1]
        assert second_call["tools"] in (None, [])
        assert second_call["messages"][-1]["role"] == "user"
        assert "Provide the final answer now" in second_call["messages"][-1]["content"]

    async def test_terminal_tool_result_synthesizes_at_iteration_limit(self):
        async def query(args):
            return {"rows": [{"answer": 7}]}

        tool = AgentTool(
            name="db_query",
            description="Run a database query",
            parameters={"type": "object", "properties": {}},
            handler=query,
        )
        repeated_tool_call = {
            "content": "",
            "tool_calls": [{"id": "c1", "name": "db_query", "arguments": {}}],
        }
        llm = SequentialMockLLM(response_sequence=[repeated_tool_call] * 4)
        agent = Agent(name="T", llm_provider=llm, tools=[tool])

        result = await agent.run(
            "answer from db",
            detailed=True,
            max_iterations=3,
            final_synthesis_without_tools=True,
            terminal_tools={"db_query"},
        )

        assert result["result"] == 'Query result: [{"answer": 7}]'
        assert result["diagnostics"]["exit_reason"] == "terminal_evidence_synthesis"

    async def test_guardrail_payload_is_not_terminal_evidence(self):
        async def query(args):
            return {
                "guardrail": "repeated_tool_error",
                "message": "Try something else.",
            }

        tool = AgentTool(
            name="db_query",
            description="Run a database query",
            parameters={"type": "object", "properties": {}},
            handler=query,
        )
        repeated_tool_call = {
            "content": "",
            "tool_calls": [{"id": "c1", "name": "db_query", "arguments": {}}],
        }
        llm = SequentialMockLLM(response_sequence=[repeated_tool_call] * 2)
        agent = Agent(name="T", llm_provider=llm, tools=[tool])

        with pytest.raises(AgentError, match="Max iterations"):
            await agent.run(
                "answer from db",
                max_iterations=1,
                final_synthesis_without_tools=True,
                terminal_tools={"db_query"},
            )

    async def test_suggested_next_tool_constrains_next_turn_tools(self):
        async def plan(args):
            return {
                "ok": True,
                "suggested_next_tool": "db_query",
                "suggested_next_arguments": {"plan_id": "plan_1"},
            }

        async def query(args):
            return {"rows": [{"answer": 7}]}

        tools = [
            AgentTool(
                name="db_plan_query",
                description="Plan",
                parameters={"type": "object", "properties": {}},
                handler=plan,
            ),
            AgentTool(
                name="db_query",
                description="Query",
                parameters={"type": "object", "properties": {}},
                handler=query,
            ),
        ]
        llm = SequentialMockLLM(
            response_sequence=[
                {
                    "content": "",
                    "tool_calls": [
                        {"id": "c1", "name": "db_plan_query", "arguments": {}}
                    ],
                },
                "The answer is 7.",
            ]
        )
        agent = Agent(name="T", llm_provider=llm, tools=tools)

        result = await agent.run("answer from db", detailed=True)

        assert result["result"] == "The answer is 7."
        second_call_tools = llm.call_history[1]["tools"]
        assert [tool["function"]["name"] for tool in second_call_tools] == ["db_query"]
        assert (
            "suggested_next_arguments" in llm.call_history[1]["messages"][-2]["content"]
        )

    async def test_unavailable_suggested_next_tool_does_not_empty_tools(self):
        async def search(args):
            return {
                "ok": False,
                "suggested_next_tool": "db_plan_query",
                "suggested_next_arguments": {"goal": "answer"},
            }

        tool = AgentTool(
            name="catalog_search_schema",
            description="Search schema",
            parameters={"type": "object", "properties": {}},
            handler=search,
        )
        llm = SequentialMockLLM(
            response_sequence=[
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "catalog_search_schema",
                            "arguments": {},
                        }
                    ],
                },
                "I can answer from the schema evidence.",
            ]
        )
        agent = Agent(name="T", llm_provider=llm, tools=[tool])

        result = await agent.run("answer from schema", detailed=True)

        assert result["result"] == "I can answer from the schema evidence."
        second_call_tools = llm.call_history[1]["tools"]
        assert [tool["function"]["name"] for tool in second_call_tools] == [
            "catalog_search_schema"
        ]
        assert (
            result["diagnostics"]["unavailable_suggested_next_tool"] == "db_plan_query"
        )
        assert "suggested_next_tool_unavailable:db_plan_query" in (
            result["diagnostics"]["warnings"]
        )

    async def test_schema_only_db_run_answers_from_catalog_evidence(self):
        async def search(args):
            return {
                "tables": [
                    {
                        "name": "payments",
                        "score": 0.95,
                        "matched_fields": [{"name": "amount", "score": 0.9}],
                    }
                ]
            }

        tool = AgentTool(
            name="catalog_search_schema",
            description="Search schema",
            parameters={"type": "object", "properties": {}},
            handler=search,
            category="database",
            source="from_db",
        )
        llm = SequentialMockLLM(
            response_sequence=[
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "catalog_search_schema",
                            "arguments": {"query": "revenue"},
                        }
                    ],
                },
                "payments.amount is the safest revenue-like column.",
            ]
        )
        agent = Agent(name="SchemaRuntimeAgent", llm_provider=llm, tools=[tool])
        prompt = "Search the schema for revenue columns."
        orchestrator = DbRunOrchestrator(agent, prompt, state=DbRunState())
        prepared = await orchestrator.prepare()

        result = await agent.run(
            prompt,
            tools=list(prepared.contract.tools),
            run_orchestrator=orchestrator,
            detailed=True,
            max_iterations=3,
        )

        assert result["result"] == "payments.amount is the safest revenue-like column."
        assert result["diagnostics"]["db_completeness"]["status"] == (
            "answerable_schema"
        )
        assert result["diagnostics"]["db_completeness"]["queries_executed"] == 0

    async def test_db_catalog_join_result_suggests_plan_query(self):
        async def find_join_paths(args):
            return {
                "success": True,
                "from_assets": ["orders"],
                "to_assets": ["customers"],
                "reachable": True,
                "path_count": 1,
                "paths": [
                    {
                        "tables": ["orders", "customers"],
                        "joins": [
                            {
                                "left_table": "orders",
                                "left_column": "customer_id",
                                "right_table": "customers",
                                "right_column": "customer_id",
                            }
                        ],
                    }
                ],
            }

        async def plan(args):
            return {"ok": True}

        tools = [
            AgentTool(
                name="catalog_find_join_paths",
                description="Find joins",
                parameters={"type": "object", "properties": {}},
                handler=find_join_paths,
            ),
            AgentTool(
                name="db_plan_query",
                description="Plan",
                parameters={"type": "object", "properties": {}},
                handler=plan,
            ),
        ]
        llm = SequentialMockLLM(
            response_sequence=[
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "catalog_find_join_paths",
                            "arguments": {
                                "from_tables": ["orders"],
                                "to_tables": ["customers"],
                            },
                        }
                    ],
                },
                "Ready to plan.",
            ]
        )
        agent = Agent(name="T", llm_provider=llm, tools=tools)
        prompt = "Show sales by customer and find the join path"
        orchestrator = DbRunOrchestrator(agent, prompt, state=DbRunState())
        prepared = await orchestrator.prepare()

        result = await agent.run(
            prompt,
            tools=list(prepared.contract.tools),
            run_orchestrator=orchestrator,
            detailed=True,
        )

        assert result["result"] == "Ready to plan."
        raw_catalog_result = result["tool_calls"][0]["result"]
        assert raw_catalog_result["suggested_next_tool"] == "db_plan_query"
        assert raw_catalog_result["suggested_next_arguments"] == {
            "candidate_tables": ["orders", "customers"],
            "required_joins": [{"from_tables": ["orders"], "to_tables": ["customers"]}],
        }
        second_call_tools = llm.call_history[1]["tools"]
        assert [tool["function"]["name"] for tool in second_call_tools] == [
            "db_plan_query"
        ]

    async def test_db_catalog_tool_uses_active_store_for_unprofiled_store_id(self):
        seen = {}

        async def search_schema(args):
            seen["args"] = dict(args)
            return {"store_id": args["store_id"], "tables": [{"name": "orders"}]}

        tool = AgentTool(
            name="catalog_search_schema",
            description="Search catalog",
            parameters={"type": "object", "properties": {}},
            handler=search_schema,
        )
        llm = SequentialMockLLM(
            response_sequence=[
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "catalog_search_schema",
                            "arguments": {
                                "store_id": "invented-store",
                                "query": "orders",
                            },
                        }
                    ],
                },
                "Done.",
            ]
        )
        agent = Agent(name="T", llm_provider=llm, tools=[tool])
        agent._db_catalog_store_id = "active-store"
        agent._db_catalog = type(
            "Catalog",
            (),
            {
                "get_schema": lambda _self, store_id: (
                    {"tables": []} if store_id == "active-store" else None
                ),
                "summarize_store": lambda _self, store_id, limit=None: {},
            },
        )()
        prompt = "inspect schema"
        orchestrator = DbRunOrchestrator(agent, prompt, state=DbRunState())
        prepared = await orchestrator.prepare()

        result = await agent.run(
            prompt,
            tools=list(prepared.contract.tools),
            run_orchestrator=orchestrator,
            detailed=True,
        )

        assert result["result"] == "Done."
        assert seen["args"]["store_id"] == "active-store"
        assert result["tool_calls"][0]["arguments"]["store_id"] == "active-store"

    def test_schema_catalog_join_result_does_not_emit_db_handoff(self):
        state = DbRunState()

        handoff = state.record_catalog_tool_result(
            "catalog_find_join_paths",
            {"from_tables": ["orders"], "to_tables": ["customers"]},
            {
                "success": True,
                "from_assets": ["orders"],
                "to_assets": ["customers"],
                "reachable": True,
                "path_count": 1,
                "paths": [{"tables": ["orders", "customers"], "joins": [{}]}],
            },
        )

        assert handoff is None
        assert state.summary()["catalog_join_evidence_count"] == 1

    async def test_db_tool_budget_blocks_extra_tool_calls(self):
        async def search(args):
            return {
                "tables": [
                    {
                        "name": "payments",
                        "score": 0.95,
                        "matched_fields": [{"name": "amount", "score": 0.9}],
                    }
                ]
            }

        tool = AgentTool(
            name="catalog_search_schema",
            description="Search schema",
            parameters={"type": "object", "properties": {}},
            handler=search,
            category="database",
            source="from_db",
        )
        llm = SequentialMockLLM(
            response_sequence=[
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "catalog_search_schema",
                            "arguments": {"query": "revenue"},
                        },
                        {
                            "id": "c2",
                            "name": "catalog_search_schema",
                            "arguments": {"query": "amount"},
                        },
                    ],
                },
                "payments.amount is the best match.",
            ]
        )
        agent = Agent(name="BudgetAgent", llm_provider=llm, tools=[tool])
        prompt = "Search the schema for revenue columns."
        orchestrator = DbRunOrchestrator(agent, prompt, state=DbRunState())
        await orchestrator.prepare()
        orchestrator.contract = DbRunContract(
            **{
                **orchestrator.contract.__dict__,
                "max_tool_calls": 1,
            }
        )
        orchestrator.state.run_contract = orchestrator.contract

        result = await agent.run(
            prompt,
            tools=list(orchestrator.contract.tools),
            run_orchestrator=orchestrator,
            detailed=True,
            max_iterations=3,
        )

        assert result["tool_calls"][1]["result"]["guardrail"] == (
            "db_tool_budget_exhausted"
        )
        assert "db_tool_budget_exhausted" in result["diagnostics"]["warnings"]

    async def test_partial_exit_without_evidence_still_raises(self):
        repeated_call = {
            "content": "Calling missing tool.",
            "tool_calls": [{"id": "tc", "name": "missing", "arguments": {}}],
        }
        agent = _make_agent([repeated_call])

        with pytest.raises(AgentError, match="Max iterations"):
            await agent.run("go", max_iterations=1, partial_exit=True)

    async def test_single_iteration_when_no_tool_calls(self):
        agent = _make_agent(["Direct answer."])
        result = await agent.run("prompt", max_iterations=5, detailed=True)
        assert result["iterations"] == 1


# ===========================================================================
# Tool calling loop
# ===========================================================================


class TestToolCallingLoop:
    async def test_tool_handler_is_called(self):
        call_log = []

        async def h(args):
            call_log.append(args)
            return args["a"] + args["b"]

        tool = AgentTool(
            name="add",
            description="Add",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "a"},
                    "b": {"type": "integer", "description": "b"},
                },
                "required": ["a", "b"],
            },
            handler=h,
        )

        llm = SequentialMockLLM(
            response_sequence=[
                {
                    "content": "Adding.",
                    "tool_calls": [
                        {"id": "c1", "name": "add", "arguments": {"a": 2, "b": 3}}
                    ],
                },
                "The answer is 5.",
            ]
        )
        agent = Agent(name="T", llm_provider=llm, tools=[tool])

        await agent.run("add 2 and 3")
        assert len(call_log) == 1
        assert call_log[0] == {"a": 2, "b": 3}

    async def test_tool_call_appears_in_run_detailed(self):  # noqa: N802
        tool = _add_tool()
        llm = SequentialMockLLM(
            response_sequence=[
                {
                    "content": "Adding.",
                    "tool_calls": [
                        {"id": "c1", "name": "add", "arguments": {"a": 1, "b": 1}}
                    ],
                },
                "Done.",
            ]
        )
        agent = Agent(name="T", llm_provider=llm, tools=[tool])

        result = await agent.run("add 1 and 1", detailed=True)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["tool"] == "add"
        assert result["tool_calls"][0]["duration_ms"] >= 0

    async def test_tool_handler_can_read_active_run_state(self):
        seen_run_ids = []

        async def h(args):
            run_state = get_active_run_state()
            seen_run_ids.append(run_state.run_id if run_state else None)
            return "ok"

        tool = AgentTool(
            name="stateful",
            description="Read run state",
            parameters={"type": "object", "properties": {}},
            handler=h,
        )
        llm = SequentialMockLLM(
            response_sequence=[
                {
                    "content": "Checking.",
                    "tool_calls": [{"id": "c1", "name": "stateful", "arguments": {}}],
                },
                "Done.",
            ]
        )
        agent = Agent(name="T", llm_provider=llm, tools=[tool])

        result = await agent.run("check state", detailed=True)

        assert seen_run_ids == [result["diagnostics"]["run_id"]]
        assert get_active_run_state() is None

    async def test_db_run_rejects_final_answer_without_query_evidence(self):
        async def h(args):
            run_state = get_active_run_state()
            db_state = run_state.domains["db"]
            db_state.record_executed_query(
                {
                    "sql": args["sql"],
                    "row_count": 1,
                    "returned_rows": 1,
                    "truncated": False,
                }
            )
            return {"rows": [{"customer": "Bob", "total_revenue": 250}]}

        tool = AgentTool(
            name="db_query",
            description="Run SQL",
            parameters={
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
            handler=h,
            category="database",
            source="from_db",
        )
        llm = SequentialMockLLM(
            response_sequence=[
                "I need to inspect the schema first.",
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "db_query",
                            "arguments": {
                                "sql": "SELECT 'Bob' AS customer, 250 AS total_revenue"
                            },
                        }
                    ],
                },
                "Bob has the highest total order revenue: 250.",
            ]
        )
        agent = Agent(name="DbRuntimeAgent", llm_provider=llm, tools=[tool])
        set_db_run_state(agent, DbRunState())

        result = await agent.run(
            "Which customer has the highest total order revenue?",
            tools=["db_query"],
            detailed=True,
            max_iterations=4,
        )

        assert result["result"] == "Bob has the highest total order revenue: 250."
        assert [call["tool"] for call in result["tool_calls"]] == ["db_query"]
        assert result["diagnostics"]["db_completeness"]["can_answer"] is True
        assert (
            "db_final_answer_without_query_evidence"
            in result["diagnostics"]["warnings"]
        )
        assert len(llm.call_history) == 3

    async def test_db_run_rejects_incomplete_final_answer_after_query(self):
        async def h(args):
            run_state = get_active_run_state()
            db_state = run_state.domains["db"]
            db_state.record_executed_query(
                {
                    "sql": args["sql"],
                    "row_count": 1,
                    "returned_rows": 1,
                    "truncated": False,
                }
            )
            return {"rows": [{"customer": "Bob", "total_revenue": 250}]}

        tool = AgentTool(
            name="db_query",
            description="Run SQL",
            parameters={
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
            handler=h,
            category="database",
            source="from_db",
        )
        llm = SequentialMockLLM(
            response_sequence=[
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "db_query",
                            "arguments": {
                                "sql": "SELECT 'Bob' AS customer, 250 AS total_revenue"
                            },
                        }
                    ],
                },
                "I will write a corrected SQL query now.",
                "Bob has the highest total order revenue: 250.",
            ]
        )
        agent = Agent(name="DbRuntimeAgent", llm_provider=llm, tools=[tool])
        set_db_run_state(agent, DbRunState())

        result = await agent.run(
            "Which customer has the highest total order revenue?",
            tools=["db_query"],
            detailed=True,
            max_iterations=4,
        )

        assert result["result"] == "Bob has the highest total order revenue: 250."
        assert "db_final_answer_incomplete" in result["diagnostics"]["warnings"]
        assert len(llm.call_history) == 3

    async def test_iterations_incremented_per_llm_call(self):
        tool = _add_tool()
        llm = SequentialMockLLM(
            response_sequence=[
                {
                    "content": "Step 1.",
                    "tool_calls": [
                        {"id": "c1", "name": "add", "arguments": {"a": 1, "b": 1}}
                    ],
                },
                "Final answer.",
            ]
        )
        agent = Agent(name="T", llm_provider=llm, tools=[tool])
        result = await agent.run("go", detailed=True)
        # 1 iteration for tool call + 1 iteration for final answer = 2
        assert result["iterations"] == 2

    async def test_repeated_tool_error_gets_guidance_before_hard_stop(self):
        async def broken(args):
            return {"error": "same failure"}

        tool = AgentTool(
            name="broken",
            description="Always fails",
            parameters={"type": "object", "properties": {}},
            handler=broken,
        )
        repeated_call = {
            "content": "Trying.",
            "tool_calls": [{"id": "c1", "name": "broken", "arguments": {}}],
        }
        llm = SequentialMockLLM(response_sequence=[repeated_call] * 4)
        agent = Agent(name="T", llm_provider=llm, tools=[tool])

        with pytest.raises(AgentError, match="Loop detected"):
            await agent.run("try it", max_iterations=4)

        third_turn_messages = llm.call_history[2]["messages"]
        guidance_messages = [
            json.loads(message["content"])
            for message in third_turn_messages
            if message.get("role") == "tool"
        ]
        assert guidance_messages[-1]["guardrail"] == "repeated_tool_error"
        assert guidance_messages[-1]["suggested_next_step"] == (
            "change_arguments_or_synthesize"
        )

    async def test_repeated_identical_result_gets_no_progress_guidance(self):
        async def read(args):
            return {"value": 1}

        tool = AgentTool(
            name="read_value",
            description="Read a value",
            parameters={"type": "object", "properties": {}},
            handler=read,
        )
        llm = SequentialMockLLM(
            response_sequence=[
                {
                    "content": "Reading.",
                    "tool_calls": [{"id": "c1", "name": "read_value", "arguments": {}}],
                },
                {
                    "content": "Reading again.",
                    "tool_calls": [{"id": "c2", "name": "read_value", "arguments": {}}],
                },
                "Done.",
            ]
        )
        agent = Agent(name="T", llm_provider=llm, tools=[tool])

        result = await agent.run("read twice", detailed=True)

        assert result["tool_calls"][1]["result"]["guardrail"] == "repeated_no_progress"
        assert "repeated_no_progress" in result["diagnostics"]["warnings"]
        assert result["diagnostics"]["evidence_count"] == 2

    async def test_successful_db_plan_result_constrains_repeated_plan_turn(self):
        plan_counter = 0

        async def plan(args):
            nonlocal plan_counter
            plan_counter += 1
            return {
                "ok": True,
                "plan_id": f"plan_{plan_counter}",
                "route": "aggregation",
                "compiled_sql": "SELECT 7 AS answer",
                "validation": {"ok": True},
                "suggested_next_tool": "db_query",
                "suggested_next_arguments": {"plan_id": f"plan_{plan_counter}"},
                "resolved_tables": ["answers"],
            }

        async def query(args):
            return {"rows": [{"answer": 7}]}

        tools = [
            AgentTool(
                name="db_plan_query",
                description="Plan a database query",
                parameters={"type": "object", "properties": {}},
                handler=plan,
            ),
            AgentTool(
                name="db_query",
                description="Run a database query",
                parameters={"type": "object", "properties": {}},
                handler=query,
            ),
        ]
        repeated_plan = lambda call_id: {
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "name": "db_plan_query",
                    "arguments": {"goal": "answer"},
                }
            ],
        }
        llm = SequentialMockLLM(
            response_sequence=[
                repeated_plan("c1"),
                repeated_plan("c2"),
                "The answer is 7.",
            ]
        )
        agent = Agent(name="T", llm_provider=llm, tools=tools)

        result = await agent.run("answer", detailed=True)

        second_call_tools = llm.call_history[1]["tools"]
        assert [tool["function"]["name"] for tool in second_call_tools] == ["db_query"]
        assert (
            "Tool 'db_plan_query' not found"
            in result["tool_calls"][1]["result"]["error"]
        )

    async def test_different_arguments_do_not_trigger_repeat_guidance(self):
        async def read(args):
            return {"value": 1}

        tool = AgentTool(
            name="read_value",
            description="Read a value",
            parameters={
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "required": ["id"],
            },
            handler=read,
        )
        llm = SequentialMockLLM(
            response_sequence=[
                {
                    "content": "Reading.",
                    "tool_calls": [
                        {"id": "c1", "name": "read_value", "arguments": {"id": 1}}
                    ],
                },
                {
                    "content": "Reading another.",
                    "tool_calls": [
                        {"id": "c2", "name": "read_value", "arguments": {"id": 2}}
                    ],
                },
                "Done.",
            ]
        )
        agent = Agent(name="T", llm_provider=llm, tools=[tool])

        result = await agent.run("read both", detailed=True)

        assert "guardrail" not in result["tool_calls"][1]["result"]
        assert result["diagnostics"]["evidence_count"] == 2

    async def test_tool_span_records_input_and_output_events(self):
        tm = get_trace_manager()
        tm._memory_exporter.clear()
        tool = _add_tool()
        llm = SequentialMockLLM(
            response_sequence=[
                {
                    "content": "Adding.",
                    "tool_calls": [
                        {"id": "c1", "name": "add", "arguments": {"a": 2, "b": 4}}
                    ],
                },
                "Done.",
            ]
        )
        agent = Agent(name="T", llm_provider=llm, tools=[tool])

        await agent.run("add 2 and 4")
        tm.flush(timeout_millis=2000)

        tool_spans = [
            s for s in tm._memory_exporter.get_finished_spans() if s.name == "tool_add"
        ]
        assert tool_spans
        events = tool_spans[-1].events
        input_event = next(e for e in events if e.name == "daita.input")
        output_event = next(e for e in events if e.name == "daita.output")
        assert json.loads(input_event.attributes["data"]) == {"a": 2, "b": 4}
        assert output_event.attributes["data"] == "6"

    async def test_agent_run_span_records_prompt_and_result_events(self):
        tm = get_trace_manager()
        tm._memory_exporter.clear()
        agent = _make_agent(["Traceable answer."])

        await agent.run("trace this")
        tm.flush(timeout_millis=2000)

        agent_spans = [
            s for s in tm._memory_exporter.get_finished_spans() if s.name == "agent_run"
        ]
        assert agent_spans
        events = agent_spans[-1].events
        input_event = next(e for e in events if e.name == "daita.input")
        output_event = next(e for e in events if e.name == "daita.output")
        assert "trace this" in input_event.attributes["data"]
        assert "Traceable answer." in output_event.attributes["data"]


# ===========================================================================
# JSON serializer for tool result messages
# ===========================================================================


class TestJsonSerialiser:
    """
    Tests for the custom json_serializer used in tool result messages.
    We exercise it by making a tool return values that are not natively
    JSON-serialisable and verifying that run() completes without error.
    """

    async def _run_with_tool_returning(self, value):
        async def h(args):
            return value

        tool = AgentTool(
            name="special",
            description="Returns special type",
            parameters={},
            handler=h,
        )
        llm = SequentialMockLLM(
            response_sequence=[
                {
                    "content": "Calling.",
                    "tool_calls": [{"id": "c1", "name": "special", "arguments": {}}],
                },
                "Done.",
            ]
        )
        agent = Agent(name="T", llm_provider=llm, tools=[tool])
        # If serialisation fails this will raise; we just need it not to.
        await agent.run("go")

    async def test_datetime_serialised(self):
        await self._run_with_tool_returning(datetime(2024, 1, 15, 12, 0, 0))

    async def test_date_serialised(self):
        await self._run_with_tool_returning(date(2024, 1, 15))

    async def test_decimal_serialised(self):
        await self._run_with_tool_returning(Decimal("3.14"))

    async def test_uuid_serialised(self):
        await self._run_with_tool_returning(uuid4())

    async def test_bytes_serialised(self):
        await self._run_with_tool_returning(b"raw bytes")


# ===========================================================================
# _build_initial_conversation
# ===========================================================================


class MemoryPlugin(LifecyclePlugin):
    """LifecyclePlugin that injects a fixed string via on_before_run."""

    def __init__(self, context: str):
        self._context = context

    async def on_before_run(self, prompt: str):
        return self._context


class TestBuildInitialConversation:
    async def test_user_message_always_last(self):
        agent = _make_agent(["Done."])
        conv = await agent._build_initial_conversation("Hello")
        assert conv[-1] == {"role": "user", "content": "Hello"}

    async def test_no_system_when_no_prompt_and_no_plugins(self):
        agent = _make_agent(["Done."])
        conv = await agent._build_initial_conversation("Hi")
        roles = [m["role"] for m in conv]
        assert "system" not in roles

    async def test_system_included_when_prompt_configured(self):
        llm = SequentialMockLLM(["Done."])
        agent = Agent(name="X", llm_provider=llm, prompt="You are helpful.")
        conv = await agent._build_initial_conversation("Hi")
        assert conv[0]["role"] == "system"
        assert "You are helpful." in conv[0]["content"]

    async def test_initial_messages_injected_before_user_message(self):
        agent = _make_agent(["Done."])
        history = [
            {"role": "user", "content": "prev"},
            {"role": "assistant", "content": "reply"},
        ]
        conv = await agent._build_initial_conversation("new", history)
        assert conv[-3]["content"] == "prev"
        assert conv[-2]["content"] == "reply"
        assert conv[-1]["content"] == "new"

    async def test_on_before_run_context_injected_into_system(self):
        llm = SequentialMockLLM(["Done."])
        plugin = MemoryPlugin("Relevant memory: user likes Python.")
        agent = Agent(name="X", llm_provider=llm, tools=[plugin])
        conv = await agent._build_initial_conversation("Tell me something")
        system_msg = next(m for m in conv if m["role"] == "system")
        assert "Relevant memory" in system_msg["content"]

    async def test_multiple_plugin_contexts_combined(self):
        llm = SequentialMockLLM(["Done."])
        agent = Agent(
            name="X",
            llm_provider=llm,
            tools=[MemoryPlugin("Memory A"), MemoryPlugin("Memory B")],
        )
        conv = await agent._build_initial_conversation("Go")
        system_msg = next(m for m in conv if m["role"] == "system")
        assert "Memory A" in system_msg["content"]
        assert "Memory B" in system_msg["content"]

    async def test_base_plugin_not_called_for_lifecycle(self):
        # BasePlugin (not LifecyclePlugin) is never consulted for on_before_run
        from daita.plugins.base import BasePlugin

        class ToolOnlyPlugin(BasePlugin):
            def get_tools(self):
                return []

        llm = SequentialMockLLM(["Done."])
        agent = Agent(name="X", llm_provider=llm, tools=[ToolOnlyPlugin()])
        conv = await agent._build_initial_conversation("Hi")
        roles = [m["role"] for m in conv]
        assert "system" not in roles

    async def test_lifecycle_plugin_returning_none_not_added(self):
        # LifecyclePlugin.on_before_run returns None by default — no system message
        llm = SequentialMockLLM(["Done."])
        agent = Agent(name="X", llm_provider=llm, tools=[LifecyclePlugin()])
        conv = await agent._build_initial_conversation("Hi")
        roles = [m["role"] for m in conv]
        assert "system" not in roles


# ===========================================================================
# on_event streaming callback
# ===========================================================================


class TestOnEventCallback:
    async def test_no_on_event_does_not_crash(self):
        agent = _make_agent(["Done."])
        # on_event=None is the default — should not raise
        result = await agent.run("hi", on_event=None)
        assert isinstance(result, str)

    async def test_on_event_receives_iteration_event(self):
        events = []

        def collect(event):
            events.append(event)

        llm = MockLLMProvider(delay=0)
        agent = Agent(name="X", llm_provider=llm)

        await agent.run("hi", on_event=collect)

        iteration_events = [e for e in events if e.type == EventType.ITERATION]
        assert len(iteration_events) >= 1

    async def test_on_event_receives_complete_event(self):
        events = []

        def collect(event):
            events.append(event)

        llm = MockLLMProvider(delay=0)
        agent = Agent(name="X", llm_provider=llm)

        await agent.run("hi", on_event=collect)

        complete_events = [e for e in events if e.type == EventType.COMPLETE]
        assert len(complete_events) == 1
