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

import time


def _print_tool_usage(result: dict, agent, stage_elapsed: float) -> None:
    """Print which tools an agent called during its run, with timing."""
    print(f"  Stage time: {stage_elapsed:.1f}s")

    # Use agent's internal tool call history (has duration_ms)
    history = getattr(agent, "_tool_call_history", [])
    if not history:
        print(f"  Tools called: (none)")
        return

    # Aggregate calls per tool: count and total duration
    stats: dict[str, dict] = {}
    for tc in history:
        name = tc.get("name", "unknown")
        dur = tc.get("duration_ms", 0)
        if name not in stats:
            stats[name] = {"count": 0, "total_ms": 0}
        stats[name]["count"] += 1
        stats[name]["total_ms"] += dur

    parts = []
    for name, s in stats.items():
        label = f"{name}({s['count']}x)" if s["count"] > 1 else name
        total_s = s["total_ms"] / 1000
        parts.append(f"{label} {total_s:.1f}s")
    print(f"  Tools: {', '.join(parts)}")

    total_tool_ms = sum(s["total_ms"] for s in stats.values())
    llm_ms = (stage_elapsed * 1000) - total_tool_ms
    print(
        f"  LLM thinking: {llm_ms / 1000:.1f}s, tool execution: {total_tool_ms / 1000:.1f}s"
    )

    # LLM cost if available
    cost = result.get("cost", 0)
    tokens = result.get("tokens", {})
    if cost or tokens:
        prompt_t = tokens.get("prompt_tokens", 0)
        comp_t = tokens.get("completion_tokens", 0)
        print(
            f"  Tokens: {prompt_t + comp_t} ({prompt_t} prompt, {comp_t} completion), cost: ${cost:.4f}"
        )


async def run_workflow(query: str) -> dict:
    """
    Run a deep research pipeline for the given query.

    Each agent runs sequentially, passing its output as the next agent's input
    via receive_message(). All agents share a 'deep_research' memory workspace,
    so they can also recall findings stored by earlier agents.

    Args:
        query: The research question (e.g. "What are the latest breakthroughs
               in solid-state batteries?").

    Returns:
        dict with status, final report, and per-stage outputs.
    """
    from agents.orchestrator import create_agent as create_orchestrator
    from agents.researcher import create_agent as create_researcher
    from agents.analyst import create_agent as create_analyst
    from agents.report_writer import create_agent as create_report_writer

    orchestrator = create_orchestrator()
    researcher = create_researcher()
    analyst = create_analyst()
    report_writer = create_report_writer()

    agents = [orchestrator, researcher, analyst, report_writer]
    stages = {}

    try:
        # Stage 1: Orchestrator plans the research
        print("[1/4] Orchestrator — planning research strategy...")
        t0 = time.time()
        plan_result = await orchestrator.run(
            f"Research query: {query}\n\n"
            "Create a research plan with 3-5 targeted sub-questions.",
            detailed=True,
        )
        stages["research_plan"] = plan_result.get("result", "")
        _print_tool_usage(plan_result, orchestrator, time.time() - t0)
        print(f"  ✓ Research plan ready\n")

        # Stage 2: Web Researcher searches for each sub-question
        print("[2/4] Web Researcher — searching the web...")
        t0 = time.time()
        findings_result = await researcher.receive_message(
            data=stages["research_plan"],
            source_agent="Orchestrator",
            channel="research_plan",
        )
        stages["raw_findings"] = findings_result.get("result", "")
        _print_tool_usage(findings_result, researcher, time.time() - t0)
        print(f"  ✓ Raw findings collected\n")

        # Stage 3: Analyst synthesises findings
        print("[3/4] Analyst — synthesising findings...")
        t0 = time.time()
        synthesis_result = await analyst.receive_message(
            data=stages["raw_findings"],
            source_agent="Web Researcher",
            channel="raw_findings",
        )
        stages["synthesis"] = synthesis_result.get("result", "")
        _print_tool_usage(synthesis_result, analyst, time.time() - t0)
        print(f"  ✓ Synthesis complete\n")

        # Stage 4: Report Writer produces the final report
        print("[4/4] Report Writer — writing report...")
        t0 = time.time()
        report_result = await report_writer.receive_message(
            data=stages["synthesis"],
            source_agent="Analyst",
            channel="synthesis",
        )
        stages["report"] = report_result.get("result", "")
        _print_tool_usage(report_result, report_writer, time.time() - t0)
        print(f"  ✓ Report written\n")

    finally:
        # Stop all agents — flushes memory graph and triggers on_agent_stop
        for agent in agents:
            try:
                await agent.stop()
            except Exception:
                pass

    # Print the final report
    print("=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(stages.get("report", ""))

    return {
        "status": "success",
        "message": f"Research complete: {query}",
        "report": stages.get("report", ""),
        "stages": stages,
    }


if __name__ == "__main__":
    import asyncio

    async def main():
        result = await run_workflow(
            "What are the latest breakthroughs in solid-state batteries?"
        )
        print(f"\nStatus: {result['status']}")

    asyncio.run(main())
