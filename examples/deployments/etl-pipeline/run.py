"""
ETL Pipeline — Manual Trigger

Runs one ETL pass immediately, without waiting for the nightly cron schedule.

Usage:
    python run.py                # Process default batch (1000 records)
    python run.py 5000           # Process up to 5000 records

Requirements:
    OPENAI_API_KEY
    SOURCE_DATABASE_URL    postgresql://user:pass@host:5432/source_db
    DEST_DATABASE_URL      postgresql://user:pass@host:5432/dest_db
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
    if not os.getenv("SOURCE_DATABASE_URL"):
        missing.append("SOURCE_DATABASE_URL=postgresql://user:pass@host:5432/source_db")
    if not os.getenv("DEST_DATABASE_URL"):
        missing.append("DEST_DATABASE_URL=postgresql://user:pass@host:5432/dest_db")
    if missing:
        print("Missing required environment variables:")
        for var in missing:
            print(f"  export {var}")
        sys.exit(1)


async def run(batch_size: int = 1000):
    from workflows.etl_workflow import run_workflow

    print("=" * 60)
    print("ETL PIPELINE")
    print("=" * 60)
    print(f"Batch size : {batch_size}")
    print(f"Source     : {os.getenv('SOURCE_DATABASE_URL', '').split('@')[-1]}")
    print(f"Destination: {os.getenv('DEST_DATABASE_URL', '').split('@')[-1]}")
    print("=" * 60)

    start = time.time()
    result = await run_workflow(batch_size=batch_size)
    elapsed = time.time() - start

    print(f"\nStatus : {result['status']}")
    print(f"Message: {result['message']}")
    print(f"Time   : {elapsed:.1f}s")


if __name__ == "__main__":
    check_environment()

    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    asyncio.run(run(batch_size))
