"""
Specialist Agents

One factory function creates specialists for billing, technical, and general
support. Each has a focused prompt and a shared draft_response tool.
"""

from typing import Any, Dict, List

from daita import Agent
from daita.core.tools import tool
from daita.plugins import MemoryPlugin


@tool
async def draft_response(
    category: str,
    issue_summary: str,
    resolution: str,
    next_steps: List[str],
    escalate: bool = False,
) -> Dict[str, Any]:
    """
    Format a structured customer-facing response.

    Args:
        category: Ticket category (billing, technical, general).
        issue_summary: One-sentence description of the issue.
        resolution: The resolution or explanation provided.
        next_steps: List of action items for the customer (up to 5).
        escalate: True if this ticket needs human review.

    Returns:
        Formatted response dict with subject, body, and metadata.
    """
    subject = f"Re: {issue_summary}"

    steps_text = ""
    if next_steps:
        steps_text = "\n\nNext steps:\n" + "\n".join(
            f"  {i+1}. {step}" for i, step in enumerate(next_steps[:5])
        )

    escalation_note = ""
    if escalate:
        escalation_note = (
            "\n\n[ESCALATION FLAG] This ticket has been flagged for human review."
        )

    body = f"""{resolution}{steps_text}

If you have any further questions, please don't hesitate to reply to this message.

Best regards,
Support Team{escalation_note}"""

    capped_steps = next_steps[:5]

    return {
        "subject": subject,
        "body": body,
        "category": category,
        "escalate": escalate,
        "next_steps": capped_steps,
    }


_PROMPTS = {
    "billing": """You are a billing support specialist. You handle questions about
invoices, charges, refunds, subscriptions, and pricing.

When you receive a ticket:
1. Identify the specific billing issue (wrong charge, refund request, plan change, etc.)
2. Explain the relevant policy clearly and sympathetically
3. Call draft_response with:
   - issue_summary: one sentence describing the billing issue
   - resolution: clear explanation + what action will be taken
   - next_steps: concrete steps (e.g. "Check your invoice at billing.example.com")
   - escalate: true only if you cannot resolve it (e.g. requires manual refund approval)

Policies:
- Refunds are available within 30 days of charge for annual plans
- Monthly plans are non-refundable but can be cancelled any time
- Billing discrepancies are investigated within 2 business days""",

    "technical": """You are a technical support specialist. You handle bug reports,
errors, login issues, API problems, and performance issues.

When you receive a ticket:
1. Identify the specific technical problem and its likely cause
2. Provide a clear diagnosis and solution
3. Call draft_response with:
   - issue_summary: one sentence describing the technical problem
   - resolution: step-by-step fix or workaround
   - next_steps: actionable debugging steps or links to docs
   - escalate: true if this looks like a platform-side bug needing engineering

Troubleshooting approach:
- Ask for error messages, stack traces, and reproduction steps if not provided
- Check common causes first: auth tokens, rate limits, network issues
- Provide specific commands or code snippets when helpful""",

    "general": """You are a general support specialist. You handle feature requests,
general questions, account settings, and feedback.

When you receive a ticket:
1. Understand what the customer is asking or requesting
2. Answer clearly and helpfully, or acknowledge the feedback
3. Call draft_response with:
   - issue_summary: one sentence describing what the customer needs
   - resolution: direct, helpful answer or acknowledgement
   - next_steps: relevant documentation links or follow-up actions
   - escalate: false (general tickets rarely need escalation)

Tone: friendly, concise, solution-focused. Avoid jargon.""",
}


def create_specialist(category: str) -> Agent:
    """
    Create a specialist agent for the given support category.

    Args:
        category: One of "billing", "technical", or "general".
    """
    if category not in _PROMPTS:
        raise ValueError(f"Unknown category: {category!r}. Must be billing, technical, or general.")

    memory = MemoryPlugin()

    return Agent(
        name=f"{category.title()} Specialist",
        model="gpt-4o-mini",
        prompt=_PROMPTS[category],
        tools=[draft_response],
        plugins=[memory],
    )
