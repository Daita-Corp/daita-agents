"""
Orchestrator Agent

Plans a research strategy for a given query by decomposing it into
3-5 specific, targeted sub-questions that the Web Researcher will search for.
"""

import json
from typing import Any, Dict

from daita import Agent
from daita.core.tools import tool
from daita.plugins import MemoryPlugin


@tool
async def decompose_query(query: str) -> Dict[str, Any]:
    """
    Provide a structured framework for breaking a research query into sub-questions.

    Returns guidance and the expected output format so the LLM knows exactly
    what to produce.

    Args:
        query: The high-level research question.

    Returns:
        Decomposition framework with angles to consider.
    """
    return {
        "original_query": query,
        "research_angles": [
            "Background and definitions — what is this, and why does it matter?",
            "Current state of the art — what is the latest progress or evidence?",
            "Key players and applications — who is working on this, and how is it used?",
            "Challenges and limitations — what are the main obstacles or criticisms?",
            "Future outlook — what is expected or predicted in the near term?",
        ],
        "output_format": {
            "query": "original query",
            "sub_questions": ["3-5 specific, searchable questions"],
            "search_keywords": ["3-8 key terms for search"],
            "depth": "surface | moderate | deep",
        },
        "instructions": (
            "Produce 3-5 sub-questions based on the angles above, tailored to the "
            "specific query. Questions should be specific enough to search for directly. "
            "Output only valid JSON matching output_format."
        ),
    }


def create_agent() -> Agent:
    """Create the Orchestrator agent."""
    memory = MemoryPlugin()

    return Agent(
        name="Orchestrator",
        model="gpt-4o-mini",
        prompt="""You are a research orchestrator. Your job is to plan a structured \
research strategy for any query.

Process:
1. Call decompose_query with the research question to get the framework.
2. Write 3-5 specific, searchable sub-questions covering: background, current state,
   key players, challenges, and future outlook. Adapt to the specific topic.
3. Output ONLY valid JSON:
   {
     "query": "the original research query",
     "sub_questions": [
       "Specific searchable question 1",
       "Specific searchable question 2",
       "..."
     ],
     "search_keywords": ["key", "search", "terms"],
     "depth": "moderate"
   }

Quality bar for sub-questions:
- Specific enough to return useful search results
- Together they cover the full scope of the original query
- Ordered logically (background first, then specifics)""",
        tools=[decompose_query],
        plugins=[memory],
    )
