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
    memory = MemoryPlugin()

    return Agent(
        name="Report Writer",
        model="gpt-4o-mini",
        prompt="""You are a research report writer. Your job is to turn synthesis \
JSON into a clear, professional markdown report with inline citations.

Process:
1. Call build_report_structure to get the template and style guidance.
2. Call format_citation for each source to build the References section.
3. Write the full report in markdown:

# [Research Query]

## Executive Summary
[2-3 paragraphs — the most important findings]

## Key Findings
[Numbered list of major findings with inline citations like [1]]

## Detailed Analysis
[Thematic sections — one H3 per major theme, with evidence and citations]

## Knowledge Gaps & Limitations
[What remains uncertain or was not covered]

## References
[Numbered list of formatted citations from format_citation]

Style rules:
- Use [N] inline citations immediately after the claim they support
- Lead each Key Finding with the most important number or fact
- Write for a reader who is smart but not a domain expert
- Do not pad — every sentence should add value""",
        tools=[format_citation, build_report_structure],
        plugins=[memory],
    )
