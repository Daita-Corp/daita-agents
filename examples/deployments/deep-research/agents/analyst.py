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
    memory = MemoryPlugin(
        workspace="deep_research",
        enable_working_memory=True,
        enable_fact_extraction=True,
        enable_memory_graph=True,
        enable_reinforcement=True,
        tier="full",
    )
    lineage_plugin = LineagePlugin()

    return Agent(
        name="Analyst",
        model="gpt-4o-mini",
        prompt="""You are a research analyst. You receive raw findings and must \
synthesise them into structured insights.

You MUST follow these steps in order. Do NOT skip any step.

Step 1: Call extract_scope(findings_json) with the raw findings to understand scope.
Step 2: Call recall(query="research findings", category="finding", limit=10) to \
retrieve all findings stored by the Researcher.
Step 3: Call query_facts() to find structured entity-relation-value triples. Try \
query_facts(entity=None) to see all extracted facts. This surfaces connections \
the raw text may not make obvious.
Step 4: Pick 2-3 key entities from the findings (e.g. company names, technologies). \
For each one, call traverse_memory(entity="entity name") to discover connections \
across findings.
Step 5: Call scratch() to write your analysis notes — what themes, contradictions, \
and knowledge gaps you identified. Use key="analysis".
Step 6: Call reinforce() on the findings. Pass the chunk IDs from Step 2's recall \
results. Use outcome="positive" for findings with strong evidence, \
outcome="negative" for weak or contradictory ones.
Step 7: Call remember() to store your synthesis. You MUST set category="synthesis" \
and importance=0.9.
Step 8: Output ONLY the final JSON:
{
  "query": "...",
  "executive_summary": "2-3 sentences with the most important takeaways",
  "key_findings": ["finding with data point [source title]"],
  "themes": {"theme_name": "description and evidence"},
  "contradictions": ["conflicting information between sources"],
  "knowledge_gaps": ["important unanswered questions"],
  "key_entities": {"entity_name": "why it is significant"},
  "sources": [{"title": "...", "url": "..."}]
}

Be analytical, not just summarising. What patterns emerge? What is surprising?""",
        tools=[extract_scope, memory, lineage_plugin],
    )
