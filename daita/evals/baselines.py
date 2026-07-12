"""Baseline recording and comparison for eval reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import EvalSuiteConfig
from .models import EvalReport


def default_baseline_path(config: EvalSuiteConfig) -> Path:
    return Path(".daita/evals/baselines") / f"{config.name}.json"


def record_baseline(
    report: EvalReport, config: EvalSuiteConfig, path: str | Path | None = None
) -> Path:
    target = Path(path or config.baselines.path or default_baseline_path(config))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report.model_dump_json(indent=2))
    return target


def compare_baseline(
    report: EvalReport,
    config: EvalSuiteConfig,
    path: str | Path | None = None,
) -> dict[str, Any] | None:
    target = Path(path or config.baselines.path or default_baseline_path(config))
    if not target.exists():
        return None

    baseline = EvalReport.model_validate_json(target.read_text())
    comparison = {
        "baseline_path": str(target),
        "status": "passed",
        "regressions": [],
        "score_delta": report.score - baseline.score,
        "cost_delta": report.summary.total_cost - baseline.summary.total_cost,
        "latency_ms_delta": report.summary.total_latency_ms
        - baseline.summary.total_latency_ms,
        "new_failures": _new_failures(baseline, report),
        "case_changes": _case_changes(baseline, report),
    }

    policy = config.baselines.fail_on
    if policy.new_failures and comparison["new_failures"]:
        _add_regression(comparison, "new_failures", comparison["new_failures"])
    if (
        policy.score_drop_gt is not None
        and comparison["score_delta"] < -policy.score_drop_gt
    ):
        _add_regression(comparison, "score_drop", comparison["score_delta"])
    if policy.cost_increase_pct_gt is not None:
        pct = _pct_increase(baseline.summary.total_cost, report.summary.total_cost)
        comparison["cost_increase_pct"] = pct
        if pct > policy.cost_increase_pct_gt:
            _add_regression(comparison, "cost_increase", pct)
    if policy.latency_increase_pct_gt is not None:
        pct = _pct_increase(
            baseline.summary.total_latency_ms, report.summary.total_latency_ms
        )
        comparison["latency_increase_pct"] = pct
        if pct > policy.latency_increase_pct_gt:
            _add_regression(comparison, "latency_increase", pct)
    if policy.capability_sequence_changed:
        changed = [
            change
            for change in comparison["case_changes"]
            if change.get("capability_sequence_changed")
        ]
        if changed:
            _add_regression(comparison, "capability_sequence_changed", changed)
    return comparison


def write_baseline_comparison(report: EvalReport) -> None:
    if not report.artifact_path or not report.baseline_comparison:
        return
    path = Path(report.artifact_path) / "baseline-comparison.json"
    path.write_text(json.dumps(report.baseline_comparison, indent=2, sort_keys=True))


def _new_failures(baseline: EvalReport, report: EvalReport) -> list[dict[str, Any]]:
    old = {
        (failure.case_id, failure.run_id, failure.code) for failure in baseline.failures
    }
    return [
        failure.model_dump(mode="json")
        for failure in report.failures
        if (failure.case_id, failure.run_id, failure.code) not in old
    ]


def _case_changes(baseline: EvalReport, report: EvalReport) -> list[dict[str, Any]]:
    baseline_cases = {case.case_id: case for case in baseline.cases}
    changes = []
    for case in report.cases:
        old = baseline_cases.get(case.case_id)
        if old is None:
            changes.append({"case_id": case.case_id, "change": "new_case"})
            continue
        old_capabilities = _capability_sequences(old)
        new_capabilities = _capability_sequences(case)
        changes.append(
            {
                "case_id": case.case_id,
                "status_before": old.status,
                "status_after": case.status,
                "capability_sequence_changed": old_capabilities != new_capabilities,
                "capability_sequences_before": old_capabilities,
                "capability_sequences_after": new_capabilities,
            }
        )
    return changes


def _capability_sequences(case) -> list[list[str]]:
    return [[task.capability_id for task in run.tasks] for run in case.runs]


def _pct_increase(old: float, new: float) -> float:
    if old <= 0:
        return 0.0 if new <= old else 100.0
    return ((new - old) / old) * 100


def _add_regression(comparison: dict[str, Any], code: str, observed: Any) -> None:
    comparison["status"] = "failed"
    comparison["regressions"].append({"code": code, "observed": observed})
