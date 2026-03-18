"""
Database chat — interactive terminal session powered by Agent.from_db().

Connect to any database and ask questions in plain English.
The agent discovers the schema automatically and keeps the conversation
in context across turns.

Prerequisites:
    pip install 'daita-agents[postgresql]'   # or [mysql], [sqlite], etc.
    export DATABASE_URL=postgresql://user:pass@host:5432/mydb
    export OPENAI_API_KEY=sk-...             # or ANTHROPIC_API_KEY, etc.

Run:
    python examples/db_chat.py

Commands during chat:
    reset   — clear conversation history
    quit    — exit
"""

import asyncio
import os
import sys
import time


async def main():
    # ── Validate environment ──────────────────────────────────────────────────
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("Error: DATABASE_URL is not set.")
        print("  export DATABASE_URL=postgresql://user:pass@host:5432/mydb")
        sys.exit(1)

    # ── Connect and discover schema ───────────────────────────────────────────
    import daita

    print(f"Connecting to {db_url.split('@')[-1]} ...")
    agent = await daita.Agent.from_db(
        db_url,
        history=True,  # keeps prior Q&A in context so follow-up questions work
        read_only=True,
    )
    print("Ready. Type your question, 'reset' to clear history, or 'quit' to exit.\n")

    # ── Chat loop ─────────────────────────────────────────────────────────────
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue

        if question.lower() in {"quit", "exit", "q"}:
            break

        if question.lower() == "reset":
            from daita.agents.conversation import ConversationHistory

            agent._db_history = ConversationHistory()
            print("[History cleared]\n")
            continue

        t0 = time.time()
        answer = await agent.run(question)
        elapsed_ms = (time.time() - t0) * 1000

        print(f"\nAgent: {answer}")
        print(f"[{elapsed_ms:.0f}ms  |  {agent._db_history.turn_count} turn(s)]\n")


if __name__ == "__main__":
    asyncio.run(main())
