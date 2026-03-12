"""
Analyst Agent

Synthesises raw research findings from the Web Researcher into structured
insights: cross-cutting themes, key facts, contradictions, and knowledge gaps.
"""

import json
from typing import Any, Dict

from daita import Agent
from daita.core.tools import tool
from daita.plugins import LineagePlugin, MemoryPlugin


@tool
async def extract_scope(findings_json: str) -> Dict[str, Any]:
    """
    Parse raw findings JSON and return a summary of the research scope.

    Args:
        findings_json: JSON string of research findings from the Web Researcher.

    Returns:
        Scope summary — query, sub-question count, total sources, analysis prompts.
    """
    try:
        findings = json.loads(findings_json)
    except (json.JSONDecodeError, TypeError) as e:
        return {"error": f"Invalid JSON: {e}"}

    total_sources = sum(len(f.get("sources", [])) for f in findings.get("findings", []))

    return {
        "query": findings.get("query", ""),
        "sub_question_count": len(findings.get("findings", [])),
        "total_sources": total_sources,
        "sub_questions": [f["sub_question"] for f in findings.get("findings", [])],
        "analysis_guidance": [
            "Identify consensus findings — what do multiple sources agree on?",
            "Spot contradictions — where do sources disagree?",
            "Extract key facts, figures, dates, and named entities",
            "Note knowledge gaps — important questions that the search did not answer",
            "Identify themes that cut across multiple sub-questions",
        ],
    }


def create_agent() -> Agent:
    """Create the Analyst agent."""
    memory = MemoryPlugin()
    lineage_plugin = LineagePlugin()

    return Agent(
        name="Analyst",
        model="gpt-4o-mini",
        prompt="""You are a research analyst synthesising web search findings.

Process:
1. Call extract_scope to understand the full scope of the research.
2. Analyse all findings holistically — look for patterns across sub-questions.
3. Output a synthesis JSON:
{
  "query": "...",
  "executive_summary": "2-3 sentences capturing the most important takeaways",
  "key_findings": [
    "Specific finding with supporting data point [source title]"
  ],
  "themes": {
    "theme_name": "description of the theme and supporting evidence"
  },
  "contradictions": [
    "Description of conflicting information between sources"
  ],
  "knowledge_gaps": [
    "Important aspect not covered by the research"
  ],
  "key_entities": {
    "entity_name": "why it is significant"
  },
  "sources": [
    {"title": "...", "url": "..."}
  ]
}

Be analytical, not just summarising. Aim to produce genuine insight — what
patterns emerge? What is surprising or counterintuitive? What remains uncertain?""",
        tools=[extract_scope],
        plugins=[memory, lineage_plugin],
    )
