"""
Deep Research — Entry Point

Run a multi-agent deep research pipeline for any query.

Usage:
    python run.py "What are the latest breakthroughs in solid-state batteries?"

Requirements:
    OPENAI_API_KEY
    TAVILY_API_KEY    (free tier: 1000 searches/month at tavily.com)
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

DEMO_QUERY = "What are the latest breakthroughs in solid-state batteries?"


def check_environment():
    missing = []
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY=sk-...")
    if not os.getenv("TAVILY_API_KEY"):
        missing.append("TAVILY_API_KEY=tvly-...  (get free key at tavily.com)")
    if missing:
        print("Missing required environment variables:")
        for var in missing:
            print(f"  export {var}")
        sys.exit(1)


async def run(query: str):
    from workflows.research_workflow import run_workflow

    print("=" * 70)
    print("DEEP RESEARCH PIPELINE")
    print("=" * 70)
    print(f"Query: {query}")
    print()
    print("Pipeline: Orchestrator → Researcher → Analyst → Report Writer")
    print("=" * 70)
    print()

    start = time.time()
    result = await run_workflow(query)
    elapsed = time.time() - start

    print(f"Status : {result['status']}")
    print(f"Message: {result['message']}")
    print(f"Time   : {elapsed:.1f}s")
    print()
    print("The report is produced by the Report Writer agent at the end of the")
    print("pipeline. Check the workflow output for the full markdown report.")


if __name__ == "__main__":
    check_environment()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = DEMO_QUERY
        print(f"No query provided — using demo query:\n  {query}\n")

    asyncio.run(run(query))
