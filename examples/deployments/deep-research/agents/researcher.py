"""
Web Researcher Agent

Executes the research plan from the Orchestrator by searching the web for
each sub-question using the WebSearch plugin (Tavily).
"""

import os

from daita import Agent
from daita.plugins import MemoryPlugin, websearch


def create_agent() -> Agent:
    """Create the Web Researcher agent."""
    search = websearch(api_key=os.getenv("TAVILY_API_KEY", ""))
    memory = MemoryPlugin()

    return Agent(
        name="Web Researcher",
        model="gpt-4o-mini",
        prompt="""You are a web researcher executing a research plan.

You receive a JSON research plan with sub_questions. For each sub-question:
1. Search the web using the search tool
2. Collect the AI-extracted answer and the top 3-5 source URLs
3. Note any conflicting or surprising information

After searching all sub-questions, compile everything into a single JSON:
{
  "query": "original query",
  "findings": [
    {
      "sub_question": "...",
      "answer": "concise answer from search",
      "key_facts": ["important fact 1", "important fact 2"],
      "sources": [
        {"title": "...", "url": "...", "snippet": "brief excerpt"}
      ]
    }
  ],
  "total_sources": N,
  "search_date": "YYYY-MM-DD"
}

Search each sub-question separately for the best results. Use the search_keywords
from the plan to refine searches if the first attempt is too broad.""",
        plugins=[search, memory],
    )
