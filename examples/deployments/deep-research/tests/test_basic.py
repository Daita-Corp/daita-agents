"""
Tests for the Deep Research pipeline.

All tool functions (decompose_query, extract_scope, format_citation,
build_report_structure) are pure Python — no LLM or API key required.
Integration tests are skipped when OPENAI_API_KEY or TAVILY_API_KEY are absent.
"""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------


class TestAgentCreation:
    def test_orchestrator_can_be_created(self):
        from agents.orchestrator import create_agent

        agent = create_agent()
        assert agent is not None
        assert agent.name == "Orchestrator"

    @pytest.mark.skipif(
        not os.getenv("TAVILY_API_KEY"),
        reason="TAVILY_API_KEY not set",
    )
    def test_researcher_can_be_created(self):
        from agents.researcher import create_agent

        agent = create_agent()
        assert agent is not None
        assert agent.name == "Web Researcher"

    def test_analyst_can_be_created(self):
        from agents.analyst import create_agent

        agent = create_agent()
        assert agent is not None
        assert agent.name == "Analyst"

    def test_report_writer_can_be_created(self):
        from agents.report_writer import create_agent

        agent = create_agent()
        assert agent is not None
        assert agent.name == "Report Writer"


# ---------------------------------------------------------------------------
# Workflow structure
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.getenv("TAVILY_API_KEY"),
    reason="TAVILY_API_KEY not set — websearch plugin requires key at construction",
)
class TestWorkflowCreation:
    def test_workflow_has_four_agents(self):
        from workflows.research_workflow import create_workflow

        wf = create_workflow()
        assert len(wf.agents) == 4
        assert "orchestrator" in wf.agents
        assert "researcher" in wf.agents
        assert "analyst" in wf.agents
        assert "report_writer" in wf.agents

    def test_workflow_has_three_connections(self):
        from workflows.research_workflow import create_workflow

        wf = create_workflow()
        assert len(wf.connections) == 3

    def test_relay_channels(self):
        from workflows.research_workflow import create_workflow

        wf = create_workflow()
        channels = {c.channel for c in wf.connections}
        assert "research_plan" in channels
        assert "raw_findings" in channels
        assert "synthesis" in channels

    def test_connection_order(self):
        from workflows.research_workflow import create_workflow

        wf = create_workflow()
        conns = {c.channel: (c.from_agent, c.to_agent) for c in wf.connections}
        assert conns["research_plan"] == ("orchestrator", "researcher")
        assert conns["raw_findings"] == ("researcher", "analyst")
        assert conns["synthesis"] == ("analyst", "report_writer")


# ---------------------------------------------------------------------------
# Orchestrator tool — pure Python
# ---------------------------------------------------------------------------


class TestDecomposeQueryTool:
    async def test_returns_framework(self):
        from agents.orchestrator import decompose_query

        result = await decompose_query.execute({"query": "What are the latest advances in quantum computing?"})
        assert "original_query" in result
        assert "research_angles" in result
        assert "output_format" in result
        assert len(result["research_angles"]) == 5

    async def test_preserves_query(self):
        from agents.orchestrator import decompose_query

        q = "How does mRNA vaccine technology work?"
        result = await decompose_query.execute({"query": q})
        assert result["original_query"] == q


# ---------------------------------------------------------------------------
# Analyst tool — pure Python
# ---------------------------------------------------------------------------


SAMPLE_FINDINGS = {
    "query": "Solid-state battery breakthroughs",
    "findings": [
        {
            "sub_question": "What is a solid-state battery?",
            "answer": "A battery using solid electrolyte instead of liquid.",
            "key_facts": ["Higher energy density", "Safer than lithium-ion"],
            "sources": [
                {"title": "Battery Tech Review", "url": "https://example.com/1", "snippet": "..."},
                {"title": "Science Daily", "url": "https://example.com/2", "snippet": "..."},
            ],
        },
        {
            "sub_question": "Who is leading solid-state battery R&D?",
            "answer": "Toyota, QuantumScape, and several Chinese manufacturers.",
            "key_facts": ["Toyota plans commercial launch by 2027"],
            "sources": [
                {"title": "EV Magazine", "url": "https://example.com/3", "snippet": "..."},
            ],
        },
    ],
    "total_sources": 3,
    "search_date": "2026-03-11",
}


class TestExtractScopeTool:
    async def test_returns_scope(self):
        from agents.analyst import extract_scope

        result = await extract_scope.execute({"findings_json": json.dumps(SAMPLE_FINDINGS)})
        assert "error" not in result
        assert result["query"] == "Solid-state battery breakthroughs"
        assert result["sub_question_count"] == 2
        assert result["total_sources"] == 3

    async def test_returns_sub_questions(self):
        from agents.analyst import extract_scope

        result = await extract_scope.execute({"findings_json": json.dumps(SAMPLE_FINDINGS)})
        assert len(result["sub_questions"]) == 2

    async def test_invalid_json(self):
        from agents.analyst import extract_scope

        result = await extract_scope.execute({"findings_json": "not json"})
        assert "error" in result

    async def test_empty_findings(self):
        from agents.analyst import extract_scope

        empty = {"query": "test", "findings": [], "total_sources": 0}
        result = await extract_scope.execute({"findings_json": json.dumps(empty)})
        assert result["sub_question_count"] == 0
        assert result["total_sources"] == 0


# ---------------------------------------------------------------------------
# Report Writer tools — pure Python
# ---------------------------------------------------------------------------


class TestFormatCitationTool:
    async def test_format_citation(self):
        from agents.report_writer import format_citation

        result = await format_citation.execute({
            "title": "Battery Tech Review",
            "url": "https://example.com/1",
            "index": 1,
        })
        assert result == "[1] [Battery Tech Review](https://example.com/1)"

    async def test_citation_index(self):
        from agents.report_writer import format_citation

        result = await format_citation.execute({"title": "Source", "url": "https://example.com", "index": 5})
        assert result.startswith("[5]")


class TestBuildReportStructureTool:
    async def test_returns_structure(self):
        from agents.report_writer import build_report_structure

        result = await build_report_structure.execute({
            "query": "Solid-state battery breakthroughs", "section_count": 4
        })
        assert "sections" in result
        assert "Executive Summary" in result["sections"]
        assert "References" in result["sections"]
        assert "style_guidance" in result

    async def test_title_matches_query(self):
        from agents.report_writer import build_report_structure

        q = "Latest AI safety research"
        result = await build_report_structure.execute({"query": q, "section_count": 3})
        assert result["title"] == q


# ---------------------------------------------------------------------------
# LLM + web integration tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (os.getenv("OPENAI_API_KEY") and os.getenv("TAVILY_API_KEY")),
    reason="OPENAI_API_KEY and TAVILY_API_KEY both required",
)
class TestPipelineIntegration:
    async def test_orchestrator_produces_plan(self):
        from agents.orchestrator import create_agent

        agent = create_agent()
        await agent.start()
        try:
            result = await agent.run(
                "Research query: What are the latest breakthroughs in solid-state batteries?"
            )
            assert isinstance(result, str)
            # Should contain JSON with sub_questions
            assert "sub_questions" in result or "question" in result.lower()
        finally:
            await agent.stop()
