"""
Entry point for the code review agent.

Usage:
    python run.py                          # Review data/sample.py (demo)
    python run.py path/to/file.py          # Review a specific file
    python run.py path/to/file.py "focus"  # Review with a specific focus
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_environment():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set.")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)


async def review(file_path: str, focus: str = ""):
    from agents.reviewer import create_agent

    agent = create_agent()
    await agent.start()

    prompt = f"Review the Python file at: {file_path}"
    if focus:
        prompt += f"\n\nFocus especially on: {focus}"

    try:
        result = await agent.run(prompt, detailed=True)
        answer = result.get("result", "")
        time_ms = result.get("processing_time_ms", 0)
        cost = result.get("cost", 0)

        print("\n" + "=" * 70)
        print("CODE REVIEW")
        print("=" * 70)
        print(answer)
        print("\n" + "-" * 70)
        print(f"Time: {time_ms}ms | Cost: ${cost:.4f}")
        print("-" * 70)
    finally:
        await agent.stop()


if __name__ == "__main__":
    check_environment()

    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "sample.py"
        )

    focus = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""

    asyncio.run(review(target, focus))
