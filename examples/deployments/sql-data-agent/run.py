"""
SQL Data Agent — Entry Point

Ask natural language questions about your PostgreSQL database.

Usage:
    python run.py                          # Demo questions against DATABASE_URL
    python run.py "Your question here"     # One-shot question

Requirements:
    OPENAI_API_KEY   — OpenAI API key
    DATABASE_URL     — postgresql://user:pass@host:5432/dbname
"""

import asyncio
import os
import sys
import time
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


DEMO_QUESTIONS = [
    "What tables are in this database and what do they contain?",
    "Show me the top 10 customers by total order value.",
    "How many orders were placed in the last 30 days?",
    "Which products have never been ordered?",
    "What is the average order value by month for the past year?",
]


async def ask(agent, question: str) -> tuple:
    result = await agent.run(question, detailed=True)
    return (
        result.get("result", ""),
        result.get("processing_time_ms", 0),
        result.get("cost", 0),
    )


async def run_demo():
    from agents.sql_agent import create_agent

    print("=" * 65)
    print("SQL DATA AGENT")
    print("=" * 65)
    print(f"Database: {os.getenv('DATABASE_URL', '').split('@')[-1]}")
    print(f"Mode: Demo — {len(DEMO_QUESTIONS)} sample questions")
    print("=" * 65)

    agent = create_agent()
    await agent.start()

    total_cost = 0.0
    start = time.time()

    try:
        for i, question in enumerate(DEMO_QUESTIONS, 1):
            print(f"\nQ{i}: {question}")
            print("-" * 50)
            answer, ms, cost = await ask(agent, question)
            print(answer)
            print(f"\n[{ms:.0f}ms  ${cost:.4f}]")
            total_cost += cost
    finally:
        await agent.stop()

    elapsed = time.time() - start
    print("\n" + "=" * 65)
    print(f"Done  |  {elapsed:.1f}s  |  total cost ${total_cost:.4f}")
    print("=" * 65)


async def run_single(question: str):
    from agents.sql_agent import create_agent

    agent = create_agent()
    await agent.start()

    try:
        answer, ms, cost = await ask(agent, question)
        print(answer)
        print(f"\n[{ms:.0f}ms  ${cost:.4f}]")
    finally:
        await agent.stop()


if __name__ == "__main__":
    check_environment()

    if len(sys.argv) == 1:
        asyncio.run(run_demo())
    else:
        asyncio.run(run_single(" ".join(sys.argv[1:])))
