"""
Customer Support Bot — Entry Point

Process a support ticket through the classify-and-route workflow.

Usage:
    python run.py                                      # Demo tickets
    python run.py "Subject" "Ticket body text here"   # One ticket

Requirements:
    OPENAI_API_KEY
"""

import asyncio
import os
import sys
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path if env_path.exists() else None)
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent))

DEMO_TICKETS = [
    {
        "subject": "Charged twice this month",
        "body": "Hi, I noticed two charges of $49 on my credit card this month. "
                "My subscription should only be $49/month. Please refund the duplicate.",
        "customer_id": "cust_001",
    },
    {
        "subject": "API returning 500 errors",
        "body": "Our integration keeps getting 500 errors on POST /api/v2/events. "
                "This started about 2 hours ago and is affecting all our users. "
                "Error: Internal Server Error",
        "customer_id": "cust_002",
    },
    {
        "subject": "How do I export my data?",
        "body": "I'd like to download all my data as a CSV. "
                "I looked in the settings but couldn't find an export option.",
        "customer_id": "cust_003",
    },
]


def check_environment():
    if not os.getenv("OPENAI_API_KEY"):
        print("Missing required environment variable:")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)


async def process_ticket(ticket: dict):
    from workflows.support_workflow import run_workflow

    print(f"\nTicket: {ticket['subject']}")
    print(f"Customer: {ticket['customer_id']}")
    print("-" * 50)
    result = await run_workflow(ticket)
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")


async def run_demo():
    print("=" * 60)
    print("CUSTOMER SUPPORT BOT")
    print("=" * 60)
    print(f"Processing {len(DEMO_TICKETS)} demo tickets\n")

    for ticket in DEMO_TICKETS:
        await process_ticket(ticket)

    print("\n" + "=" * 60)
    print("Done")


async def run_single(subject: str, body: str):
    await process_ticket(
        {"subject": subject, "body": body, "customer_id": "cli_user"}
    )


if __name__ == "__main__":
    check_environment()

    if len(sys.argv) == 1:
        asyncio.run(run_demo())
    elif len(sys.argv) >= 3:
        asyncio.run(run_single(sys.argv[1], " ".join(sys.argv[2:])))
    else:
        print("Usage:")
        print('  python run.py                           # Demo tickets')
        print('  python run.py "Subject" "Body text"     # One ticket')
        sys.exit(1)
