"""
Database Health Monitor — Entry Point

Starts the database health investigation agent and runs one operator prompt.
Runtime-native monitoring should trigger this agent through runtime operations.

Usage:
    python run.py

Requirements:
    OPENAI_API_KEY   — OpenAI API key
    DATABASE_URL     — postgresql://user:pass@host:5432/dbname
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
    if missing:
        print("Missing required environment variables:")
        for var in missing:
            print(f"  export {var}")
        sys.exit(1)


async def run():
    from agents.monitor_agent import create_agent

    agent = create_agent()

    print("=" * 65)
    print("DATABASE HEALTH INVESTIGATOR")
    print("=" * 65)
    print(f"Database: {os.getenv('DATABASE_URL', '').split('@')[-1]}")
    print("Tools: slow queries, connection pressure, table bloat")
    print("=" * 65)
    print("Investigating current database health...\n")

    try:
        result = await agent.run(
            "Check the current PostgreSQL health. Use the diagnostic tools for "
            "slow queries, connection pressure, and table bloat, then summarize "
            "severity and recommended actions."
        )
        print(result)
    finally:
        await agent.stop()


if __name__ == "__main__":
    check_environment()
    asyncio.run(run())
