"""
Tests for the Slack Reporter agent.

Agent creation and configuration are tested without any external services.
Integration tests are skipped when DATABASE_URL or SLACK_BOT_TOKEN are not set.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (os.getenv("SLACK_BOT_TOKEN") and os.getenv("DATABASE_URL")),
    reason="SLACK_BOT_TOKEN and DATABASE_URL required to instantiate the reporter agent",
)
class TestAgentCreation:
    def test_agent_can_be_created(self):
        from agents.reporter_agent import create_agent

        agent = create_agent()
        assert agent is not None
        assert agent.name == "Slack Reporter"

    def test_agent_has_two_plugins(self):
        from agents.reporter_agent import create_agent

        agent = create_agent()
        # Agent should have both postgresql and slack plugins
        assert len(agent.plugins) == 2

    def test_agent_model(self):
        from agents.reporter_agent import create_agent

        agent = create_agent()
        assert agent.model is not None


# ---------------------------------------------------------------------------
# LLM + service integration tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (
        os.getenv("OPENAI_API_KEY")
        and os.getenv("DATABASE_URL")
        and os.getenv("SLACK_BOT_TOKEN")
    ),
    reason="OPENAI_API_KEY, DATABASE_URL, and SLACK_BOT_TOKEN all required",
)
class TestAgentIntegration:
    async def test_daily_digest_runs(self):
        from agents.reporter_agent import create_agent

        agent = create_agent()
        await agent.start()
        try:
            result = await agent.run(
                "Run the daily sales digest for today and post it to Slack."
            )
            assert isinstance(result, str)
            assert len(result) > 10
        finally:
            await agent.stop()
