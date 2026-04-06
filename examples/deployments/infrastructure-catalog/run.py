"""
Infrastructure Catalog Agent — Entry Point

Discover and catalog data stores across AWS and GitHub, then ask questions
about your organization's data landscape.

Usage:
    python run.py                              # Demo — discover and summarize
    python run.py "What production databases do we have?"  # One-shot question

Requirements:
    OPENAI_API_KEY    — OpenAI API key
    AWS credentials   — via env vars, AWS profile, or IAM role
    GITHUB_TOKEN      — (optional) GitHub personal access token
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

    # AWS: check for at least one auth method
    has_aws = any([
        os.getenv("AWS_ACCESS_KEY_ID"),
        os.getenv("AWS_PROFILE"),
        os.getenv("AWS_ROLE_ARN"),
    ])
    if not has_aws:
        missing.append("AWS credentials (AWS_ACCESS_KEY_ID/AWS_PROFILE/AWS_ROLE_ARN)")

    if missing:
        print("Missing required environment variables:")
        for var in missing:
            print(f"  {var}")
        print("\nOptional:")
        print("  GITHUB_TOKEN     — enables GitHub repo scanning")
        print("  GITHUB_ORG       — GitHub org to scan (all repos)")
        print("  GITHUB_REPOS     — comma-separated owner/repo list")
        print("  AWS_REGIONS      — comma-separated (default: us-east-1)")
        print("  AWS_SERVICES     — comma-separated (default: rds,dynamodb,s3,elasticache,redshift)")
        sys.exit(1)


DEMO_QUESTIONS = [
    # Q1: Discover + batch-store in memory (tests batch remember, categories)
    "Discover all our data stores. Batch-store each discovery in memory with "
    "category='store' and appropriate importance (0.8 for production, 0.6 for "
    "staging, 0.4 for dev). Then give me a summary grouped by environment.",

    # Q2: Query from memory without re-scanning (tests recall, list_by_category)
    "What production databases do we have? Check memory first — don't re-scan "
    "unless memory is empty. Show me their types and regions.",

    # Q3: Store an org rule + check memory stats (tests pinned rules, stats)
    "Remember this org rule: 'All production databases must have encryption "
    "enabled and automated backups configured.' Store it with importance=0.9 "
    "and category='rule'. Then show me memory stats (list_memories with "
    "include_stats=True) so I can see what we've cataloged.",

    # Q4: Temporal query (tests since/before filtering)
    "What infrastructure discoveries have been made in the last 24 hours? "
    "Use recall with since='24h'.",

    # Q5: Cross-reference memory (tests recall + list_by_category together)
    "Are there any stores discovered from GitHub that look like they might "
    "have real credentials? Cross-reference with our stored rules.",
]


async def ask(agent, question: str) -> tuple:
    result = await agent.run(question, detailed=True)
    return (
        result.get("result", ""),
        result.get("processing_time_ms", 0),
        result.get("cost", 0),
    )


async def run_demo():
    from agents.catalog_agent import create_agent

    print("=" * 65)
    print("INFRASTRUCTURE CATALOG AGENT")
    print("=" * 65)
    regions = os.getenv("AWS_REGIONS", "us-east-1")
    print(f"AWS regions: {regions}")
    if os.getenv("GITHUB_ORG"):
        print(f"GitHub org: {os.getenv('GITHUB_ORG')}")
    elif os.getenv("GITHUB_REPOS"):
        print(f"GitHub repos: {os.getenv('GITHUB_REPOS')}")
    print(f"Mode: Demo — {len(DEMO_QUESTIONS)} questions")
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
    from agents.catalog_agent import create_agent

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
