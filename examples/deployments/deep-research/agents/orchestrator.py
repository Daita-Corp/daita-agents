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
    memory = MemoryPlugin(
        workspace="deep_research",
        enable_working_memory=True,
        enable_memory_graph=True,
        tier="full",
    )

    return Agent(
        name="Orchestrator",
        model="gpt-4o-mini",
        prompt="""You are a research orchestrator. Plan a structured research strategy.

You MUST follow these steps in order. Do NOT skip any step.

Step 1: Call decompose_query(query) to get the research framework.
Step 2: Call scratch() to store your initial analysis of the query — note which \
angles are most important for this specific topic and why. Use key="planning_notes".
Step 3: Draft 3-5 specific, searchable sub-questions covering background, current \
state, key players, challenges, and future outlook.
Step 4: Call remember() to store the final plan. You MUST set \
category="research_plan" and importance=0.8. Pass a summary of the plan as content.
Step 5: Output ONLY the final JSON:
{
  "query": "the original research query",
  "sub_questions": ["3-5 specific, searchable questions"],
  "search_keywords": ["3-8 key terms for search"],
  "depth": "surface | moderate | deep"
}

Quality bar: sub-questions must be specific enough to return useful search results, \
cover the full scope, and be ordered logically (background first).""",
        tools=[decompose_query, memory],
    )
