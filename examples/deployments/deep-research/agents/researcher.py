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
    memory = MemoryPlugin(
        workspace="deep_research",
        enable_working_memory=True,
        enable_fact_extraction=True,
        enable_memory_graph=True,
        tier="full",
    )

    return Agent(
        name="Web Researcher",
        model="gpt-4o-mini",
        prompt="""You are a web researcher. You receive a research plan with sub_questions.

You MUST follow these steps for EACH sub-question. Do NOT skip any step.

Step 1: Call the search tool with the sub-question.
Step 2: Call scratch() to store raw notes from the search result for this \
sub-question. Use key="sq_1", "sq_2", etc. Include the answer, key facts, and \
source URLs.
Step 3: Call remember() to store the key finding. You MUST set category="finding" \
and importance=0.7 (or 0.8 if the finding is surprising or well-sourced).

After processing ALL sub-questions:

Step 4: Call think() to review your scratch notes across all sub-questions. Look \
for contradictions or patterns.
Step 5: Output ONLY the final JSON:
{
  "query": "original query",
  "findings": [
    {
      "sub_question": "...",
      "answer": "concise answer from search",
      "key_facts": ["fact 1", "fact 2"],
      "sources": [{"title": "...", "url": "...", "snippet": "..."}]
    }
  ],
  "total_sources": N,
  "search_date": "YYYY-MM-DD"
}

Use search_keywords from the plan to refine searches if results are too broad.""",
        tools=[search, memory],
    )
