"""Async eval runner for runnable Daita targets."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .analysis import (
    RunEvidence,
    extract_run_evidence,
    metric_delta,
    metric_snapshot,
    preview_text,
    summarize_stability,
)
from .assertions import (
    evaluate_run_assertions,
    evaluate_stability_assertions,
    has_errors,
)
from .artifacts import ArtifactWriter
from .baselines import (
    compare_baseline as compare_report_to_baseline,
    record_baseline as record_report_baseline,
    write_baseline_comparison,
)
from .config import AgentConfig, EvalCaseConfig, EvalSuiteConfig
from .datasets import expand_cases
from .judges import JudgeEvaluation, evaluate_judge
from .models import (
    CaseResult,
    EvalFailure,
    EvalReport,
    ReportSummary,
    RunResult,
)


async def run_suite(
    config: EvalSuiteConfig,
    *,
    config_path: str | None = None,
    output_dir: str | Path | None = None,
    write_artifacts: bool = True,
    compare_baseline: bool = False,
    record_baseline: bool = False,
    baseline_path: str | Path | None = None,
) -> EvalReport:
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    cases = expand_cases(config, config_path=config_path)
    target = await load_target(config.agent)
    adapter = RunnableAdapter(target)

    agent_info = {
        "factory": config.agent.factory,
        "agent_id": getattr(target, "agent_id", None),
        "agent_name": config.agent.label or getattr(target, "name", None),
        "model": _get_model(target),
    }

    case_results: list[CaseResult] = []
    try:
        await adapter.start()
        for case in cases:
            case_results.append(await _run_case(case, config, adapter))
    finally:
        await adapter.stop()

    report = _build_report(
        config=config,
        run_id=run_id,
        agent_info=agent_info,
        cases=case_results,
        config_path=config_path,
    )

    if compare_baseline:
        report.baseline_comparison = compare_report_to_baseline(
            report, config, path=baseline_path
        )
        if (
            report.baseline_comparison
            and report.baseline_comparison["status"] == "failed"
        ):
            report.status = "failed"
            report.failures.append(
                EvalFailure(
                    case_id="baseline",
                    code="baseline_regression",
                    message="Baseline comparison detected regressions.",
                    severity="error",
                    observed=report.baseline_comparison.get("regressions", []),
                )
            )

    if write_artifacts:
        writer = ArtifactWriter(
            output_root=Path(output_dir or config.artifacts.output_dir),
            artifact_config=config.artifacts,
        )
        report.artifact_path = str(writer.write(report))
        write_baseline_comparison(report)

    if record_baseline:
        record_report_baseline(report, config, path=baseline_path)

    return report


async def load_target(config: AgentConfig) -> Any:
    if config.factory is None:
        raise ValueError("agent.factory is required")
    return await load_factory(config.factory, config.kwargs)


async def load_factory(factory_path: str, kwargs: dict[str, Any]) -> Any:
    module_name, sep, func_name = factory_path.partition(":")
    if not sep or not module_name or not func_name:
        raise ValueError("agent.factory must use 'module:function' format")
    module = importlib.import_module(module_name)
    factory = getattr(module, func_name)
    result = factory(**kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


class RunnableAdapter:
    """Small adapter for agents or custom objects with run/start/stop methods."""

    def __init__(self, target: Any):
        self.target = target

    async def start(self) -> None:
        start = getattr(self.target, "start", None)
        if start is not None:
            result = start()
            if inspect.isawaitable(result):
                await result

    async def stop(self) -> None:
        stop = getattr(self.target, "stop", None)
        if stop is not None:
            result = stop()
            if inspect.isawaitable(result):
                await result

    async def run(
        self,
        prompt: str,
        *,
        max_iterations: int,
        timeout_seconds: float | None,
    ) -> Any:
        run = getattr(self.target, "run_detailed", None) or getattr(
            self.target, "run", None
        )
        if run is None:
            raise TypeError(
                "Eval target must provide run_detailed(prompt, ...) or run(prompt, ...)"
            )

        kwargs: dict[str, Any] = {}
        if _accepts_kwarg(run, "max_iterations"):
            kwargs["max_iterations"] = max_iterations
        if _accepts_kwarg(run, "timeout_seconds"):
            kwargs["timeout_seconds"] = timeout_seconds

        before = metric_snapshot(self.target)
        start = time.perf_counter()
        result = run(prompt, **kwargs)
        if inspect.isawaitable(result):
            if timeout_seconds and not _accepts_kwarg(run, "timeout_seconds"):
                result = await asyncio.wait_for(result, timeout=timeout_seconds)
            else:
                result = await result
        latency_ms = (time.perf_counter() - start) * 1000
        after = metric_snapshot(self.target)
        return {
            "runtime_result": result,
            "latency_ms": latency_ms,
            "_eval_metric_delta": metric_delta(before, after),
        }


async def _run_case(
    case: EvalCaseConfig,
    suite: EvalSuiteConfig,
    adapter: RunnableAdapter,
) -> CaseResult:
    runs_requested = case.runs or suite.defaults.runs
    max_iterations = case.max_iterations or suite.defaults.max_iterations
    timeout_seconds = case.timeout_seconds or suite.defaults.timeout_seconds

    run_results: list[RunResult] = []
    evidence_runs: list[RunEvidence] = []
    for index in range(runs_requested):
        run_id = f"run-{index + 1:03d}"
        try:
            raw = await adapter.run(
                case.prompt,
                max_iterations=max_iterations,
                timeout_seconds=timeout_seconds,
            )
            evidence = extract_run_evidence(case.prompt, raw)
            assertions = evaluate_run_assertions(case, evidence, suite.defaults)
            judge_evaluation = await evaluate_judge(
                case,
                evidence,
                deterministic_failed=has_errors(assertions),
            )
            judges = []
            if judge_evaluation is not None:
                assertions.append(judge_evaluation.assertion)
                judges.append(judge_evaluation)
            status = "failed" if has_errors(assertions) else "passed"
            evidence_runs.append(evidence)
            run_results.append(
                _to_run_result(run_id, evidence, assertions, status, judges)
            )
        except Exception as exc:
            run_results.append(
                RunResult(
                    run_id=run_id,
                    status="failed",
                    prompt_hash="",
                    answer_hash="",
                    final_answer_preview="",
                    assertions=[],
                    errors=[str(exc)],
                )
            )

    stability = summarize_stability(evidence_runs)
    stability_assertions = evaluate_stability_assertions(case, stability)
    runs_passed = sum(1 for run in run_results if run.status == "passed")
    runs_failed = len(run_results) - runs_passed
    status = "failed" if runs_failed or has_errors(stability_assertions) else "passed"
    return CaseResult(
        case_id=case.id,
        status=status,
        runs_requested=runs_requested,
        runs_passed=runs_passed,
        runs_failed=runs_failed,
        pass_rule=case.pass_rule or suite.defaults.pass_rule,
        stability=stability,
        runs=run_results,
        assertions=stability_assertions,
    )


def _to_run_result(
    run_id: str,
    evidence: RunEvidence,
    assertions,
    status: str,
    judges: list[JudgeEvaluation] | None = None,
) -> RunResult:
    judge_results = []
    for index, judge in enumerate(judges or []):
        judge.result.input_artifact_path = f"judge-{index + 1:03d}-input.json"
        judge.result.output_artifact_path = f"judge-{index + 1:03d}-output.json"
        judge.result.input_payload = judge.input_payload
        judge.result.output_payload = judge.output_payload
        judge_results.append(judge.result)
    return RunResult(
        run_id=run_id,
        status=status,
        prompt_hash=evidence.prompt_hash,
        answer_hash=evidence.answer_hash,
        final_answer_preview=preview_text(evidence.answer),
        final_answer=evidence.answer,
        metrics=evidence.metrics,
        trace_id=evidence.trace_id,
        operation_id=evidence.operation_id,
        operation_status=evidence.operation_status,
        operation_type=evidence.operation_type,
        intent=evidence.intent,
        tasks=evidence.tasks,
        evidence=evidence.evidence,
        governance=evidence.governance,
        approvals=evidence.approvals,
        warnings=evidence.warnings,
        assertions=assertions,
        judges=judge_results,
    )


def _build_report(
    *,
    config: EvalSuiteConfig,
    run_id: str,
    agent_info: dict[str, Any],
    cases: list[CaseResult],
    config_path: str | None,
) -> EvalReport:
    summary = ReportSummary(
        cases_total=len(cases),
        cases_passed=sum(1 for c in cases if c.status == "passed"),
        cases_failed=sum(1 for c in cases if c.status == "failed"),
        cases_warned=sum(1 for c in cases if c.status == "warned"),
        runs_total=sum(len(c.runs) for c in cases),
        runs_passed=sum(c.runs_passed for c in cases),
        runs_failed=sum(c.runs_failed for c in cases),
        total_cost=sum(run.metrics.cost or 0 for case in cases for run in case.runs),
        total_latency_ms=sum(
            run.metrics.latency_ms or 0 for case in cases for run in case.runs
        ),
        judge_calls=sum(len(run.judges) for case in cases for run in case.runs),
        judge_cost=sum(
            judge.cost or 0
            for case in cases
            for run in case.runs
            for judge in run.judges
        ),
        judge_latency_ms=sum(
            judge.latency_ms or 0
            for case in cases
            for run in case.runs
            for judge in run.judges
        ),
        judge_tokens_total=sum(
            judge.tokens_total or 0
            for case in cases
            for run in case.runs
            for judge in run.judges
        ),
    )
    failures = _collect_failures(cases)
    status = "failed" if summary.cases_failed else "passed"
    score = summary.cases_passed / summary.cases_total if summary.cases_total else 0.0
    return EvalReport(
        run_id=run_id,
        suite={
            "name": config.name,
            "version": config.version,
            "config_path": config_path,
        },
        agent=agent_info,
        status=status,
        score=score,
        summary=summary,
        failures=failures,
        cases=cases,
    )


def _collect_failures(cases: list[CaseResult]) -> list[EvalFailure]:
    failures: list[EvalFailure] = []
    for case in cases:
        for run in case.runs:
            for assertion in run.assertions:
                if assertion.status == "failed":
                    failures.append(
                        EvalFailure(
                            case_id=case.case_id,
                            run_id=run.run_id,
                            code=assertion.code,
                            message=assertion.message,
                            severity=assertion.severity,
                            assertion_path=assertion.assertion_path,
                            observed=assertion.observed,
                            expected=assertion.expected,
                            artifact_path=f"cases/{case.case_id}/{run.run_id}.json",
                            fix_hints=assertion.fix_hints,
                            related_task_ids=assertion.related_task_ids,
                            related_evidence_ids=assertion.related_evidence_ids,
                            related_trace_ids=[run.trace_id] if run.trace_id else [],
                        )
                    )
            for error in run.errors:
                failures.append(
                    EvalFailure(
                        case_id=case.case_id,
                        run_id=run.run_id,
                        code="run_error",
                        message=error,
                        severity="error",
                        artifact_path=f"cases/{case.case_id}/{run.run_id}.json",
                    )
                )
        for assertion in case.assertions:
            if assertion.status == "failed":
                failures.append(
                    EvalFailure(
                        case_id=case.case_id,
                        code=assertion.code,
                        message=assertion.message,
                        severity=assertion.severity,
                        assertion_path=assertion.assertion_path,
                        observed=assertion.observed,
                        expected=assertion.expected,
                        artifact_path=f"cases/{case.case_id}/case.json",
                        fix_hints=assertion.fix_hints,
                    )
                )
    return failures


def _accepts_kwarg(func: Any, name: str) -> bool:
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind == p.VAR_KEYWORD for p in params.values())


def _get_model(target: Any) -> str | None:
    llm = getattr(target, "llm", None)
    return getattr(llm, "model", None) or getattr(llm, "model_name", None)
