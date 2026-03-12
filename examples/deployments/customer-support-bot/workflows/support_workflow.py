"""
Customer Support Workflow

The Classifier reads incoming tickets and routes them via relay channels
to the appropriate Specialist agent:

  Classifier → billing_queue   → Billing Specialist
  Classifier → technical_queue → Technical Specialist
  Classifier → general_queue   → General Specialist

This demonstrates conditional routing — unlike the linear pattern in etl-pipeline,
the Classifier's output channel determines which downstream agent runs.
"""

from daita import Workflow


def create_workflow() -> Workflow:
    """Build the customer support routing workflow."""
    from agents.classifier import create_agent as create_classifier
    from agents.specialist import create_specialist

    workflow = Workflow("Customer Support")

    workflow.add_agent("classifier", create_classifier())
    workflow.add_agent("billing", create_specialist("billing"))
    workflow.add_agent("technical", create_specialist("technical"))
    workflow.add_agent("general", create_specialist("general"))

    # Conditional routing: classifier outputs to one of three channels
    workflow.connect("classifier", "billing_queue", "billing")
    workflow.connect("classifier", "technical_queue", "technical")
    workflow.connect("classifier", "general_queue", "general")

    return workflow


async def run_workflow(ticket: dict) -> dict:
    """
    Process a support ticket through the classify-and-route workflow.

    Args:
        ticket: Dict with keys: subject (str), body (str), customer_id (str).

    Returns:
        Status dict.
    """
    workflow = create_workflow()
    try:
        await workflow.start()

        prompt = (
            f"New support ticket:\n\n"
            f"Subject: {ticket.get('subject', '')}\n"
            f"Customer: {ticket.get('customer_id', 'unknown')}\n\n"
            f"{ticket.get('body', '')}\n\n"
            "Classify this ticket and route it to the appropriate queue."
        )

        await workflow.inject_data("classifier", prompt)
        return {"status": "success", "message": "Ticket routed"}
    finally:
        await workflow.stop()


if __name__ == "__main__":
    import asyncio

    async def main():
        result = await run_workflow(
            {
                "subject": "Cannot log in to my account",
                "body": "I keep getting a 403 error when I try to log in. "
                        "My password was recently reset but the issue persists.",
                "customer_id": "cust_12345",
            }
        )
        print(result)

    asyncio.run(main())
