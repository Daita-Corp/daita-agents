"""
Slack Reporter — Manual Trigger

Runs the daily sales digest immediately, without waiting for the cron schedule.
Useful for testing the report before deploying.

Usage:
    python run.py          # Trigger digest now

Requirements:
    OPENAI_API_KEY
    DATABASE_URL
    SLACK_BOT_TOKEN
    SLACK_CHANNEL_ID       # Optional — defaults to #analytics
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


def check_environment():
    missing = []
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY=sk-...")
    if not os.getenv("DATABASE_URL"):
        missing.append("DATABASE_URL=postgresql://user:pass@host:5432/dbname")
    if not os.getenv("SLACK_BOT_TOKEN"):
        missing.append("SLACK_BOT_TOKEN=xoxb-...")
    if missing:
        print("Missing required environment variables:")
        for var in missing:
            print(f"  export {var}")
        sys.exit(1)


async def run():
    from agents.reporter_agent import create_agent

    channel = os.getenv("SLACK_CHANNEL_ID", "#analytics")
    print(f"Triggering daily digest -> {channel}")
    print("-" * 50)

    agent = create_agent()
    await agent.start()

    try:
        result = await agent.run(
            "Run the daily sales digest for today and post it to Slack."
        )
        print(result)
    finally:
        await agent.stop()


if __name__ == "__main__":
    check_environment()
    asyncio.run(run())
