"""
Report Writer Agent

Takes the Analyst's synthesis and produces a structured, cited research
report in markdown format.
"""

from typing import Any, Dict

from daita import Agent
from daita.core.tools import tool
from daita.plugins import MemoryPlugin


@tool
async def format_citation(title: str, url: str, index: int) -> str:
    """
    Format a source as a numbered markdown citation.

    Args:
        title: Source title.
        url: Source URL.
        index: Citation number (1-based).

    Returns:
        Formatted citation string.
    """
    return f"[{index}] [{title}]({url})"


@tool
async def build_report_structure(query: str, section_count: int) -> Dict[str, Any]:
    """
    Return the expected report structure and formatting guidance.

    Args:
        query: The original research query (used as report title).
        section_count: Number of thematic sections to write.

    Returns:
        Report template and style guidance.
    """
    return {
        "title": query,
        "sections": [
            "Executive Summary",
            "Key Findings",
            "Detailed Analysis",
            "Knowledge Gaps & Limitations",
            "References",
        ],
        "style_guidance": {
            "audience": "Informed non-specialist",
            "tone": "Clear, professional, objective",
            "inline_citations": "Use [N] notation after claims backed by sources",
            "length": f"Approximately {section_count * 150}-{section_count * 250} words per section",
            "tables": "Use markdown tables for comparisons with 3+ items",
        },
    }


def create_agent() -> Agent:
    """Create the Report Writer agent."""
    memory = MemoryPlugin(
        workspace="deep_research",
        enable_memory_graph=True,
        tier="analysis",
    )

    return Agent(
        name="Report Writer",
        model="gpt-4o-mini",
        prompt="""You are a research report writer. Turn synthesis JSON into a \
professional markdown report with inline citations.

You MUST follow these steps in order. Do NOT skip any step.

Step 1: Call build_report_structure(query, section_count) to get the template.
Step 2: Call recall(query="synthesis", category="synthesis") to retrieve the \
Analyst's synthesis from shared memory.
Step 3: Call recall(query="research findings", category="finding", limit=10) to \
retrieve all key findings from shared memory.
Step 4: Pick 2-3 key entities (companies, technologies) from the synthesis. For \
each one, call traverse_memory(entity="entity name") to discover additional \
context and connections that enrich the report.
Step 5: Call format_citation(title, url, index) for each source to build the \
References section. Number them starting at 1.
Step 6: Write and output the full report in markdown:

# [Research Query]

## Executive Summary
[2-3 paragraphs — the most important findings]

## Key Findings
[Numbered list with inline citations like [1]]

## Detailed Analysis
[Thematic sections — one H3 per theme, with evidence and citations. Use \
information from traverse_memory to add depth.]

## Knowledge Gaps & Limitations
[What remains uncertain or was not covered]

## References
[Numbered list from format_citation]

Style: use [N] inline citations after claims, lead findings with the key number \
or fact, write for informed non-specialists, every sentence adds value.""",
        tools=[format_citation, build_report_structure, memory],
    )
