"""Report rendering for eval results."""

from __future__ import annotations

from html import escape

from .models import EvalReport


def render_pretty(report: EvalReport) -> str:
    lines = [
        f"Daita Eval: {report.suite['name']}",
        f"Run: {report.run_id}  Agent: {report.agent.get('agent_name') or 'unknown'}  Model: {report.agent.get('model') or 'unknown'}",
        "",
        "Summary",
        f"  Cases:  {report.summary.cases_passed} passed / {report.summary.cases_failed} failed / {report.summary.cases_warned} warned",
        f"  Runs:   {report.summary.runs_passed} passed / {report.summary.runs_failed} failed",
        f"  Score:  {report.score * 100:.1f}%",
        f"  Cost:   ${report.summary.total_cost:.4f}",
        f"  Time:   {report.summary.total_latency_ms / 1000:.1f}s",
    ]
    if report.summary.judge_calls:
        lines.append(
            f"  Judge:  {report.summary.judge_calls} calls / "
            f"${report.summary.judge_cost:.4f} / "
            f"{report.summary.judge_latency_ms / 1000:.1f}s / "
            f"{report.summary.judge_tokens_total} tokens"
        )
    if report.artifact_path:
        lines.append(f"  Output: {report.artifact_path}")

    if report.cases:
        lines.extend(["", "Cases"])
        for case in report.cases:
            lines.extend(_render_case_lines(case))

    errors = [f for f in report.failures if f.severity == "error"]
    warnings = [f for f in report.failures if f.severity == "warning"]
    if errors:
        lines.extend(["", "Failures"])
        lines.extend(_render_failure_lines(errors))
    if warnings:
        lines.extend(["", "Warnings"])
        lines.extend(_render_failure_lines(warnings))
    return "\n".join(lines)


def render_markdown(report: EvalReport) -> str:
    lines = [
        f"# Daita Eval: {report.suite['name']}",
        "",
        f"- Run: `{report.run_id}`",
        f"- Status: `{report.status}`",
        f"- Score: `{report.score * 100:.1f}%`",
        f"- Cases: `{report.summary.cases_passed}` passed / `{report.summary.cases_failed}` failed / `{report.summary.cases_warned}` warned",
        f"- Runs: `{report.summary.runs_passed}` passed / `{report.summary.runs_failed}` failed",
        f"- Cost: `${report.summary.total_cost:.4f}`",
        f"- Time: `{report.summary.total_latency_ms / 1000:.1f}s`",
    ]
    if report.failures:
        lines.extend(["", "## Failures"])
        for failure in report.failures:
            lines.append(
                f"- `{failure.code}` in `{failure.case_id}`"
                f"{' / ' + failure.run_id if failure.run_id else ''}: {failure.message}"
            )
    return "\n".join(lines) + "\n"


def render_junit(report: EvalReport) -> str:
    tests = report.summary.cases_total
    failures = report.summary.cases_failed
    lines = [
        f'<testsuite name="{escape(str(report.suite["name"]))}" tests="{tests}" failures="{failures}">'
    ]
    for case in report.cases:
        lines.append(
            f'  <testcase classname="daita.eval" name="{escape(case.case_id)}">'
        )
        if case.status == "failed":
            messages = [
                f.message
                for f in report.failures
                if f.case_id == case.case_id and f.severity == "error"
            ]
            message = escape("; ".join(messages) or "case failed")
            lines.append(f'    <failure message="{message}">{message}</failure>')
        lines.append("  </testcase>")
    lines.append("</testsuite>")
    return "\n".join(lines) + "\n"


def _render_failure_lines(failures) -> list[str]:
    lines = []
    grouped = {}
    for failure in failures:
        grouped.setdefault((failure.case_id, failure.run_id), []).append(failure)
    for (case_id, run_id), items in grouped.items():
        heading = f"  FAIL {case_id}"
        if run_id:
            heading += f"  {run_id}"
        lines.append(heading)
        for failure in items:
            artifact = f" ({failure.artifact_path})" if failure.artifact_path else ""
            lines.append(f"    - {failure.code}: {failure.message}{artifact}")
    return lines


def _render_case_lines(case) -> list[str]:
    status = case.status.upper()
    run_summary = f"{case.runs_passed}/{case.runs_requested} runs"
    cost = sum(run.metrics.cost or 0 for run in case.runs)
    latency_ms = sum(run.metrics.latency_ms or 0 for run in case.runs)
    tokens = sum(run.metrics.tokens_total or 0 for run in case.runs)
    lines = [
        f"  {status} {case.case_id}  {run_summary}  "
        f"${cost:.4f}  {latency_ms / 1000:.1f}s  {tokens} tokens"
    ]
    if case.runs_requested > 1:
        lines.append(
            "    stability: "
            f"{case.stability.answer_variants} answer variants, "
            f"{case.stability.tool_sequence_variants} tool sequences"
        )
    tool_sequence = _first_tool_sequence(case)
    if tool_sequence:
        lines.append(f"    tools: {' -> '.join(tool_sequence)}")
    skill_summary = _span_summary(case, "skill")
    if skill_summary:
        lines.append(f"    skills: {skill_summary}")
    plugin_summary = _span_summary(case, "plugin")
    if plugin_summary:
        lines.append(f"    plugins: {plugin_summary}")
    judge_summary = _judge_summary(case)
    if judge_summary:
        lines.append(f"    judge: {judge_summary}")
    return lines


def _first_tool_sequence(case) -> list[str]:
    if not case.runs:
        return []
    return [call.name for call in case.runs[0].tool_calls]


def _judge_summary(case) -> str:
    judges = [judge for run in case.runs for judge in run.judges]
    if not judges:
        return ""
    passed = sum(1 for judge in judges if judge.passed)
    avg_score = sum(judge.score for judge in judges) / len(judges)
    return f"{passed}/{len(judges)} passed, avg score {avg_score:.2f}"


def _span_summary(case, kind: str) -> str:
    if not case.runs:
        return ""
    spans = [span for span in case.runs[0].execution_spans if span.kind == kind]
    if not spans:
        return ""
    items = []
    for span in spans[:5]:
        label = span.name
        if span.operation:
            label = f"{label}.{span.operation}"
        if span.latency_ms is not None:
            label = f"{label} {span.latency_ms:.0f}ms"
        elif span.status:
            label = f"{label} {span.status}"
        items.append(label)
    if len(spans) > 5:
        items.append(f"+{len(spans) - 5} more")
    return ", ".join(items)
