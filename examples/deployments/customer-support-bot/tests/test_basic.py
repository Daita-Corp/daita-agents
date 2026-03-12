"""
Tests for the Customer Support Bot.

classify_ticket and draft_response are pure Python tools, so they can be
tested without any LLM or external services.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------


class TestAgentCreation:
    def test_classifier_can_be_created(self):
        from agents.classifier import create_agent

        agent = create_agent()
        assert agent is not None
        assert agent.name == "Classifier"

    def test_billing_specialist_can_be_created(self):
        from agents.specialist import create_specialist

        agent = create_specialist("billing")
        assert agent is not None
        assert "Billing" in agent.name

    def test_technical_specialist_can_be_created(self):
        from agents.specialist import create_specialist

        agent = create_specialist("technical")
        assert "Technical" in agent.name

    def test_general_specialist_can_be_created(self):
        from agents.specialist import create_specialist

        agent = create_specialist("general")
        assert "General" in agent.name

    def test_invalid_category_raises(self):
        from agents.specialist import create_specialist

        with pytest.raises(ValueError, match="Unknown category"):
            create_specialist("unknown")


# ---------------------------------------------------------------------------
# Workflow structure
# ---------------------------------------------------------------------------


class TestWorkflowCreation:
    def test_workflow_has_four_agents(self):
        from workflows.support_workflow import create_workflow

        wf = create_workflow()
        assert len(wf.agents) == 4
        assert "classifier" in wf.agents
        assert "billing" in wf.agents
        assert "technical" in wf.agents
        assert "general" in wf.agents

    def test_workflow_has_three_connections(self):
        from workflows.support_workflow import create_workflow

        wf = create_workflow()
        assert len(wf.connections) == 3

    def test_routing_channels(self):
        from workflows.support_workflow import create_workflow

        wf = create_workflow()
        channels = {c.channel for c in wf.connections}
        assert "billing_queue" in channels
        assert "technical_queue" in channels
        assert "general_queue" in channels


# ---------------------------------------------------------------------------
# classify_ticket tool — no LLM required (pure Python)
# ---------------------------------------------------------------------------


class TestClassifyTicketTool:
    async def test_billing_ticket(self):
        from agents.classifier import classify_ticket

        result = await classify_ticket.execute({
            "subject": "Invoice discrepancy",
            "body": "I was charged twice for my subscription this month. "
                    "Please refund the duplicate payment.",
        })
        assert result["suggested_category"] == "billing"
        assert result["billing_signals"] > 0

    async def test_technical_ticket(self):
        from agents.classifier import classify_ticket

        result = await classify_ticket.execute({
            "subject": "API returns 500 error",
            "body": "My integration keeps getting a 500 error when calling the webhooks endpoint. "
                    "The issue started after yesterday's deployment.",
        })
        assert result["suggested_category"] == "technical"
        assert result["technical_signals"] > 0

    async def test_general_ticket(self):
        from agents.classifier import classify_ticket

        result = await classify_ticket.execute({
            "subject": "Feature request: dark mode",
            "body": "Would love a dark mode option in the dashboard. "
                    "Keep up the great work!",
        })
        # No strong billing or technical signals → general
        assert result["suggested_category"] == "general"

    async def test_routing_matches_category(self):
        from agents.classifier import classify_ticket

        result = await classify_ticket.execute({
            "subject": "Refund request",
            "body": "I'd like a refund for my annual subscription.",
        })
        assert result["suggested_routing"] == f"{result['suggested_category']}_queue"


# ---------------------------------------------------------------------------
# draft_response tool — no LLM required (pure Python)
# ---------------------------------------------------------------------------


class TestDraftResponseTool:
    async def test_basic_response(self):
        from agents.specialist import draft_response

        result = await draft_response.execute({
            "category": "billing",
            "issue_summary": "Duplicate charge in March",
            "resolution": "We have confirmed the duplicate charge and issued a full refund.",
            "next_steps": ["Check your bank statement in 3-5 business days"],
            "escalate": False,
        })
        assert "subject" in result
        assert "body" in result
        assert result["escalate"] is False
        assert "refund" in result["body"].lower()

    async def test_escalation_flag(self):
        from agents.specialist import draft_response

        result = await draft_response.execute({
            "category": "technical",
            "issue_summary": "Platform outage affecting all users",
            "resolution": "We are investigating a platform-wide issue.",
            "next_steps": ["Monitor our status page for updates"],
            "escalate": True,
        })
        assert result["escalate"] is True
        assert "ESCALATION FLAG" in result["body"]

    async def test_next_steps_limited(self):
        from agents.specialist import draft_response

        result = await draft_response.execute({
            "category": "general",
            "issue_summary": "Feature request",
            "resolution": "Thank you for your feedback.",
            "next_steps": ["step 1", "step 2", "step 3", "step 4", "step 5", "step 6"],
            "escalate": False,
        })
        # Should cap at 5 next steps
        assert len(result["next_steps"]) <= 5


# ---------------------------------------------------------------------------
# LLM integration tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
class TestClassifierIntegration:
    async def test_classifies_billing_ticket(self):
        from agents.classifier import create_agent

        agent = create_agent()
        await agent.start()
        try:
            result = await agent.run(
                "New ticket — Subject: Incorrect charge on invoice "
                "Body: I was charged $99 but my plan is only $49/month."
            )
            assert isinstance(result, str)
            assert "billing" in result.lower()
        finally:
            await agent.stop()
