"""
Deep Research Workflow

Four-agent linear pipeline:

  Orchestrator → (research_plan)  → Web Researcher
  Web Researcher → (raw_findings) → Analyst
  Analyst → (synthesis)           → Report Writer

The Orchestrator plans the research, the Web Researcher searches the web,
the Analyst synthesises the findings, and the Report Writer produces a
cited markdown report.
"""

from daita import Workflow


def create_workflow() -> Workflow:
    """Build the deep research pipeline."""
    from agents.orchestrator import create_agent as create_orchestrator
    from agents.researcher import create_agent as create_researcher
    from agents.analyst import create_agent as create_analyst
    from agents.report_writer import create_agent as create_report_writer

    workflow = Workflow("Deep Research")

    workflow.add_agent("orchestrator", create_orchestrator())
    workflow.add_agent("researcher", create_researcher())
    workflow.add_agent("analyst", create_analyst())
    workflow.add_agent("report_writer", create_report_writer())

    workflow.connect("orchestrator", "research_plan", "researcher")
    workflow.connect("researcher", "raw_findings", "analyst")
    workflow.connect("analyst", "synthesis", "report_writer")

    return workflow


async def run_workflow(query: str) -> dict:
    """
    Start a deep research pipeline for the given query.

    Args:
        query: The research question (e.g. "What are the latest breakthroughs
               in solid-state batteries?").
    """
    workflow = create_workflow()
    try:
        await workflow.start()
        await workflow.inject_data(
            "orchestrator",
            f"Research query: {query}\n\n"
            "Create a research plan with 3-5 targeted sub-questions.",
        )
        return {"status": "success", "message": f"Research pipeline started: {query}"}
    finally:
        await workflow.stop()


if __name__ == "__main__":
    import asyncio

    async def main():
        result = await run_workflow(
            "What are the latest breakthroughs in solid-state batteries?"
        )
        print(result)

    asyncio.run(main())
