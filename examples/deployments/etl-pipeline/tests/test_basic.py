"""
Tests for the ETL Pipeline.

The transformer tool (validate_and_clean) runs pure Python with pandas, so it
can be fully tested without any database or LLM. Database and LLM integration
tests are skipped when the relevant environment variables are not set.
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
    def test_extractor_can_be_created(self):
        from agents.extractor import create_agent

        agent = create_agent()
        assert agent is not None
        assert agent.name == "Extractor"

    def test_transformer_can_be_created(self):
        from agents.transformer import create_agent

        agent = create_agent()
        assert agent is not None
        assert agent.name == "Transformer"

    def test_loader_can_be_created(self):
        from agents.loader import create_agent

        agent = create_agent()
        assert agent is not None
        assert agent.name == "Loader"


# ---------------------------------------------------------------------------
# Workflow creation
# ---------------------------------------------------------------------------


class TestWorkflowCreation:
    def test_workflow_has_three_agents(self):
        from workflows.etl_workflow import create_workflow

        wf = create_workflow()
        assert len(wf.agents) == 3
        assert "extractor" in wf.agents
        assert "transformer" in wf.agents
        assert "loader" in wf.agents

    def test_workflow_has_two_connections(self):
        from workflows.etl_workflow import create_workflow

        wf = create_workflow()
        assert len(wf.connections) == 2

    def test_workflow_channels(self):
        from workflows.etl_workflow import create_workflow

        wf = create_workflow()
        channels = {c.channel for c in wf.connections}
        assert "raw_data" in channels
        assert "transformed_data" in channels


# ---------------------------------------------------------------------------
# Transformer tool tests — no database or LLM required
# ---------------------------------------------------------------------------


class TestValidateAndClean:
    SAMPLE_RECORDS = [
        {
            "id": "1",
            "event_type": "Page View",
            "user_id": "u1",
            "session_id": "s1",
            "properties": '{"page": "/home"}',
            "created_at": "2026-03-11T08:00:00",
        },
        {
            "id": "2",
            "event_type": "page view",  # duplicate after normalisation
            "user_id": "u1",
            "session_id": "s1",
            "properties": '{"page": "/home"}',
            "created_at": "2026-03-11T08:00:00",
        },
        {
            "id": "3",
            "event_type": "click",
            "user_id": None,  # missing user_id — should be rejected
            "session_id": "s2",
            "properties": "{}",
            "created_at": "2026-03-11T08:01:00",
        },
        {
            "id": "4",
            "event_type": "  Sign Up  ",
            "user_id": "u2",
            "session_id": "s3",
            "properties": {"plan": "pro"},  # already a dict
            "created_at": "2026-03-11T08:02:00",
        },
    ]

    async def test_rejects_missing_user_id(self):
        from agents.transformer import validate_and_clean

        result = await validate_and_clean.execute({"records_json": json.dumps(self.SAMPLE_RECORDS)})
        assert "error" not in result
        rejected_reasons = [r.get("rejection_reason") for r in result["rejected"]]
        assert "missing_user_id" in rejected_reasons

    async def test_normalises_event_type(self):
        from agents.transformer import validate_and_clean

        result = await validate_and_clean.execute({"records_json": json.dumps(self.SAMPLE_RECORDS)})
        assert "error" not in result
        event_types = {r["event_type"] for r in result["clean"]}
        # "Page View" → "page_view", "  Sign Up  " → "sign_up"
        assert "page_view" in event_types
        assert "sign_up" in event_types

    async def test_deduplicates(self):
        from agents.transformer import validate_and_clean

        result = await validate_and_clean.execute({"records_json": json.dumps(self.SAMPLE_RECORDS)})
        assert "error" not in result
        assert result["stats"]["duplicates_removed"] >= 1

    async def test_parses_properties_json_string(self):
        from agents.transformer import validate_and_clean

        result = await validate_and_clean.execute({"records_json": json.dumps(self.SAMPLE_RECORDS)})
        assert "error" not in result
        for record in result["clean"]:
            assert isinstance(record["properties"], dict)

    async def test_empty_input(self):
        from agents.transformer import validate_and_clean

        result = await validate_and_clean.execute({"records_json": "[]"})
        assert "error" not in result
        assert result["clean"] == []
        assert result["stats"]["input"] == 0

    async def test_invalid_json(self):
        from agents.transformer import validate_and_clean

        result = await validate_and_clean.execute({"records_json": "not json"})
        assert "error" in result

    async def test_adds_processed_at(self):
        from agents.transformer import validate_and_clean

        records = [
            {
                "id": "1",
                "event_type": "click",
                "user_id": "u1",
                "session_id": "s1",
                "properties": "{}",
                "created_at": "2026-03-11T08:00:00",
            }
        ]
        result = await validate_and_clean.execute({"records_json": json.dumps(records)})
        assert "error" not in result
        assert len(result["clean"]) == 1
        assert "processed_at" in result["clean"][0]


# ---------------------------------------------------------------------------
# Database integration tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.getenv("SOURCE_DATABASE_URL"),
    reason="SOURCE_DATABASE_URL not set",
)
class TestExtractorIntegration:
    async def test_get_unprocessed_events(self):
        from agents.extractor import get_unprocessed_events

        result = await get_unprocessed_events.execute({"batch_size": 10})
        assert "error" not in result
        assert "records" in result
        assert "count" in result


@pytest.mark.skipif(
    not os.getenv("DEST_DATABASE_URL"),
    reason="DEST_DATABASE_URL not set",
)
class TestLoaderIntegration:
    async def test_load_empty_batch(self):
        from agents.loader import load_to_destination

        result = await load_to_destination.execute({"records_json": "[]"})
        assert "error" not in result
        assert result["total"] == 0
