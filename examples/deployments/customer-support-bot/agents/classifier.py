"""
Classifier Agent

Reads incoming support tickets, classifies them into billing, technical,
or general categories, and routes them to the right specialist via relay channels.
"""

from typing import Any, Dict

from daita import Agent
from daita.core.tools import tool
from daita.plugins import MemoryPlugin


@tool
async def classify_ticket(subject: str, body: str) -> Dict[str, Any]:
    """
    Score a support ticket against billing and technical keyword signals.

    Uses simple keyword matching to produce a preliminary classification signal
    that the LLM uses alongside its own understanding to make a final decision.

    Args:
        subject: Ticket subject line.
        body: Full ticket body text.

    Returns:
        Dict with keyword scores and a suggested category/routing.
    """
    text = f"{subject} {body}".lower()

    billing_keywords = {
        "invoice",
        "charge",
        "payment",
        "refund",
        "billing",
        "subscription",
        "price",
        "cost",
        "fee",
        "credit card",
        "overcharge",
        "receipt",
        "trial",
        "plan",
        "upgrade",
        "downgrade",
        "cancel",
    }
    technical_keywords = {
        "error",
        "bug",
        "crash",
        "broken",
        "not working",
        "fail",
        "issue",
        "problem",
        "login",
        "access",
        "api",
        "timeout",
        "500",
        "403",
        "404",
        "performance",
        "slow",
        "outage",
        "down",
        "exception",
        "traceback",
        "integration",
        "webhook",
        "token",
        "authentication",
    }

    billing_score = sum(1 for kw in billing_keywords if kw in text)
    technical_score = sum(1 for kw in technical_keywords if kw in text)

    if billing_score > technical_score:
        suggested = "billing"
    elif technical_score > 0:
        suggested = "technical"
    else:
        suggested = "general"

    return {
        "billing_signals": billing_score,
        "technical_signals": technical_score,
        "suggested_category": suggested,
        "suggested_routing": f"{suggested}_queue",
    }


def create_agent() -> Agent:
    """Create the Classifier agent."""
    memory = MemoryPlugin()

    return Agent(
        name="Classifier",
        model="gpt-4o-mini",
        prompt="""You are a customer support ticket classifier. Your job is to read incoming
tickets and route them to the correct specialist team.

Categories:
- billing   — invoices, charges, refunds, subscriptions, pricing, cancellations
- technical — errors, bugs, crashes, login issues, API problems, outages
- general   — feature requests, general questions, account settings, feedback

Priority levels:
- high   — service down, payment failed, security issue, data loss
- medium — feature broken, billing discrepancy, access denied
- low    — general question, feature request, feedback

Process:
1. Call classify_ticket with the ticket subject and body to get keyword signals.
2. Use the signals combined with your understanding to make a final decision.
3. Output ONLY a JSON object (no other text):
   {
     "category": "billing|technical|general",
     "priority": "high|medium|low",
     "summary": "one-sentence summary of the issue",
     "routing": "billing_queue|technical_queue|general_queue"
   }

The `routing` value determines which specialist receives the ticket, so it
must exactly match one of: billing_queue, technical_queue, general_queue.""",
        tools=[classify_ticket],
        plugins=[memory],
    )
