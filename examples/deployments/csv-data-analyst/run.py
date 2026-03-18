"""
CSV Data Analyst — Entry Point

Ask natural language questions about any CSV file.

Usage:
    python run.py                                    # Demo mode (sample_sales.csv)
    python run.py data/my_data.csv                   # Analyse your own file
    python run.py data/my_data.csv "Your question"   # One-shot question

Requirements:
    OPENAI_API_KEY environment variable
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def check_environment():
    if not os.getenv("OPENAI_API_KEY"):
        print("Missing required environment variable:")
        print("  export OPENAI_API_KEY=sk-...")
        print("\nGet your key at: https://platform.openai.com/api-keys")
        sys.exit(1)


DEMO_QUESTIONS = [
    "What are the top 5 products by total revenue?",
    "What is the total revenue broken down by region?",
    "Which category sells the most units on average per order?",
    "Show me all Electronics orders where revenue exceeded $4000.",
    "Which month in 2024 had the highest total sales?",
]


async def ask(agent, question: str, csv_path: str) -> str:
    """Send one question to the agent and return the answer."""
    full_question = f"File: {csv_path}\n\n{question}"
    result = await agent.run(full_question, detailed=True)
    return (
        result.get("result", ""),
        result.get("processing_time_ms", 0),
        result.get("cost", 0),
    )


async def run_demo(csv_path: str):
    from agents.csv_analyst import create_agent

    print("=" * 65)
    print("CSV DATA ANALYST")
    print("=" * 65)
    print(f"File: {csv_path}")
    print(f"Mode: Demo — running {len(DEMO_QUESTIONS)} sample questions")
    print("=" * 65)

    agent = create_agent(csv_path)
    await agent.start()

    total_cost = 0.0
    start = time.time()

    try:
        for i, question in enumerate(DEMO_QUESTIONS, 1):
            print(f"\nQ{i}: {question}")
            print("-" * 50)
            answer, ms, cost = await ask(agent, question, csv_path)
            print(answer)
            print(f"\n[{ms:.0f}ms  ${cost:.4f}]")
            total_cost += cost
    finally:
        await agent.stop()

    elapsed = time.time() - start
    print("\n" + "=" * 65)
    print(f"Done  |  {elapsed:.1f}s  |  total cost ${total_cost:.4f}")
    print("=" * 65)
    print("\nTo analyse your own file:")
    print('  python run.py path/to/your/data.csv "Your question here"')


async def run_single(csv_path: str, question: str):
    from agents.csv_analyst import create_agent

    agent = create_agent(csv_path)
    await agent.start()

    try:
        answer, ms, cost = await ask(agent, question, csv_path)
        print(answer)
        print(f"\n[{ms:.0f}ms  ${cost:.4f}]")
    finally:
        await agent.stop()


if __name__ == "__main__":
    check_environment()

    if len(sys.argv) == 1:
        # Demo mode
        csv_file = "data/sample_sales.csv"
        asyncio.run(run_demo(csv_file))

    elif len(sys.argv) == 2:
        # File only — run demo questions against that file
        csv_file = sys.argv[1]
        if not Path(csv_file).exists():
            print(f"File not found: {csv_file}")
            sys.exit(1)
        asyncio.run(run_demo(csv_file))

    else:
        # File + question
        csv_file = sys.argv[1]
        question = " ".join(sys.argv[2:])
        if not Path(csv_file).exists():
            print(f"File not found: {csv_file}")
            sys.exit(1)
        asyncio.run(run_single(csv_file, question))
