"""
Google Drive agent — search, read, and analyze files from Drive.

Attach the Google Drive plugin to an agent and ask questions in plain English.
The agent can search for files, read their contents (Google Docs, Sheets, Slides,
CSV, XLSX, DOCX, PDF, text), and reason across them.

Prerequisites:
    pip install 'daita-agents[google-drive]'
    export OPENAI_API_KEY=sk-...   # or ANTHROPIC_API_KEY, etc.

    Auth options (pick one):
      a) Service account  — set GOOGLE_CREDENTIALS_PATH=/path/to/service_account.json
      b) OAuth            — set GOOGLE_CREDENTIALS_PATH=/path/to/client_secrets.json
                            (browser will open on first run to authorize)
      c) Application Default Credentials — run `gcloud auth application-default login`
                            and omit GOOGLE_CREDENTIALS_PATH

Run:
    python examples/google_drive_agent.py

Example prompts:
    "Find my Q4 reports and summarize the key numbers"
    "What files did I modify this week?"
    "Read the project brief in My Drive and list the requirements"
    "Search for budget spreadsheets and tell me the totals"
"""

import asyncio
import os

from daita import Agent
from daita.plugins import google_drive


async def main():
    credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH")

    drive = google_drive(
        credentials_path=credentials_path,  # None → uses Application Default Credentials
        read_only=True,
    )

    agent = Agent(
        name="drive_assistant",
        model="gpt-4o-mini",
        prompt=(
            "You are a helpful assistant with access to the user's Google Drive. "
            "When asked about files or documents, search Drive first to find relevant files, "
            "then read their contents to answer accurately. "
            "Always mention the file name when referencing specific documents."
        ),
        tools=[drive],
    )

    async with drive:
        print("Google Drive agent ready.")
        print("Try: 'Find my latest reports' or 'What files did I modify this week?'\n")

        prompts = [
            "What files do I have in My Drive?",
            "Find any spreadsheets and summarize what's in the first one you find.",
        ]

        for prompt in prompts:
            print(f"You: {prompt}")
            result = await agent.run(prompt)
            print(f"Agent: {result}\n")


if __name__ == "__main__":
    asyncio.run(main())
