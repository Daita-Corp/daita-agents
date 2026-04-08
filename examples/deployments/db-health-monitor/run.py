"""
Database Health Monitor — Entry Point

Starts the agent with three watches that continuously poll PostgreSQL for
slow queries, connection pressure, and table bloat. The agent runs
indefinitely until interrupted with Ctrl-C.

Usage:
    python run.py                     # Run with default intervals
    python run.py --fast              # Shorter intervals for demo/testing

Requirements:
    OPENAI_API_KEY   — OpenAI API key
    DATABASE_URL     — postgresql://user:pass@host:5432/dbname
"""

import asyncio
import os
import signal
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


async def run(fast: bool = False):
    from agents.monitor_agent import create_agent

    if fast:
        agent = create_agent(
            slow_query_interval="10s",
            connection_interval="10s",
            bloat_interval="15s",
        )
    else:
        agent = create_agent()

    print("=" * 65)
    print("DATABASE HEALTH MONITOR")
    print("=" * 65)
    print(f"Database: {os.getenv('DATABASE_URL', '').split('@')[-1]}")
    print(
        f"Mode: {'fast (10-15s intervals)' if fast else 'standard (30s-5m intervals)'}"
    )
    print(f"Watches: slow_queries, connection_pressure, table_bloat")
    print("=" * 65)
    print("Monitoring... press Ctrl-C to stop.\n")

    await agent.start()

    # Wait until interrupted
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    try:
        await stop.wait()
    finally:
        print("\nShutting down...")
        await agent.stop()
        print("Stopped.")


if __name__ == "__main__":
    check_environment()
    asyncio.run(run(fast="--fast" in sys.argv))
