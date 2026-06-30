"""Shared benchmark orchestration and artifact writing for from_db scale tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Awaitable, Callable, Iterable

from daita.runtime import OperationStatus, TaskStatus

OperationFactory = Callable[[int], Awaitable[Any]]


@dataclass(frozen=True)
class ScaleBenchmarkParameters:
    concurrency: int
    operations: int
    scenario: str | None = None
    gates: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "concurrency": self.concurrency,
            "operations": self.operations,
            "scenario": self.scenario,
            "gates": dict(self.gates),
            **dict(self.extra),
        }


def artifact_output_dir(tmp_path: Path, suite: str) -> Path:
    root = Path(os.environ.get("DAITA_PERF_OUTPUT_DIR", tmp_path))
    return root / suite


async def run_scale_benchmark(
    *,
    suite: str,
    parameters: ScaleBenchmarkParameters,
    operation_factory: OperationFactory,
    output_dir: Path,
    environment: dict[str, Any] | None = None,
    artifact_name: str | None = None,
) -> dict[str, Any]:
    """Run operations concurrently and write a machine-readable JSON artifact."""

    started_at = _iso_now()
    started = time.perf_counter()
    semaphore = asyncio.Semaphore(parameters.concurrency)

    async def _run_one(index: int) -> dict[str, Any]:
        async with semaphore:
            return await measure_operation(index, operation_factory)

    operations = await asyncio.gather(
        *(_run_one(index) for index in range(parameters.operations))
    )
    elapsed_s = max(time.perf_counter() - started, 0.000001)
    artifact = {
        "suite": suite,
        "started_at": started_at,
        "finished_at": _iso_now(),
        "environment": {
            **default_environment_metadata(),
            **dict(environment or {}),
        },
        "parameters": parameters.to_dict(),
        "summary": summarize_operations(operations, elapsed_s),
        "operations": operations,
    }
    write_artifact(
        artifact,
        output_dir
        / (
            artifact_name
            or f"{suite}-{parameters.scenario or 'benchmark'}-c{parameters.concurrency}.json"
        ),
    )
    return artifact


async def measure_operation(
    index: int,
    operation_factory: OperationFactory,
) -> dict[str, Any]:
    started_at = _iso_now()
    started = time.perf_counter()
    try:
        result = await operation_factory(index)
        latency_ms = (time.perf_counter() - started) * 1000
        return operation_record(
            index=index,
            latency_ms=latency_ms,
            started_at=started_at,
            result=result,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark artifacts classify all errors
        latency_ms = (time.perf_counter() - started) * 1000
        return operation_record(
            index=index,
            latency_ms=latency_ms,
            started_at=started_at,
            error=exc,
        )


def operation_record(
    *,
    index: int,
    latency_ms: float,
    started_at: str,
    result: Any | None = None,
    error: BaseException | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if error is not None:
        return {
            "index": index,
            "started_at": started_at,
            "latency_ms": round(latency_ms, 3),
            "success": False,
            "status": "error",
            "operation_id": None,
            "capability_sequence": [],
            "task_count": 0,
            "evidence_count": 0,
            "llm": empty_llm_usage(),
            "error": classify_error(error),
            "metadata": dict(metadata or {}),
        }

    diagnostics = _diagnostics(result)
    tasks = _tasks_from_result(result)
    evidence = tuple(getattr(result, "evidence", ()) or ())
    status = getattr(result, "status", None)
    if isinstance(status, OperationStatus):
        status_value = status.value
    elif status is not None:
        status_value = str(status)
    else:
        status_value = "unknown"
    error_payload = diagnostics.get("error") if isinstance(diagnostics, dict) else None
    success = status_value == OperationStatus.SUCCEEDED.value and not error_payload
    return {
        "index": index,
        "started_at": started_at,
        "latency_ms": round(latency_ms, 3),
        "success": success,
        "status": status_value,
        "operation_id": getattr(result, "operation_id", None),
        "capability_sequence": capability_sequence(tasks),
        "task_count": _task_count(diagnostics, tasks),
        "evidence_count": len(evidence),
        "evidence_kinds": [str(getattr(item, "kind", "")) for item in evidence],
        "llm": llm_usage_from_result(result),
        "error": error_payload,
        "metadata": {
            **dict(metadata or {}),
            "warnings": list(getattr(result, "warnings", ()) or ()),
            "answer": getattr(result, "answer", None),
            "planned_sql": (
                diagnostics.get("execution", {}).get("planned_sql")
                if isinstance(diagnostics.get("execution"), dict)
                else None
            ),
            "planning_mode": (
                diagnostics.get("execution", {}).get("planning_mode")
                if isinstance(diagnostics.get("execution"), dict)
                else None
            ),
        },
    }


def operation_record_from_snapshot(
    *,
    index: int,
    latency_ms: float,
    started_at: str,
    snapshot: Any,
    error: BaseException | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if error is not None:
        return operation_record(
            index=index,
            latency_ms=latency_ms,
            started_at=started_at,
            error=error,
            metadata=metadata,
        )
    tasks = tuple(getattr(snapshot, "tasks", ()) or ())
    evidence = tuple(getattr(snapshot, "evidence", ()) or ())
    operation = getattr(snapshot, "operation", None)
    status = getattr(operation, "status", None)
    status_value = status.value if isinstance(status, OperationStatus) else str(status)
    return {
        "index": index,
        "started_at": started_at,
        "latency_ms": round(latency_ms, 3),
        "success": status is OperationStatus.SUCCEEDED,
        "status": status_value,
        "operation_id": getattr(operation, "id", None),
        "capability_sequence": capability_sequence(tasks),
        "task_count": len(tasks),
        "evidence_count": len(evidence),
        "evidence_kinds": [str(getattr(item, "kind", "")) for item in evidence],
        "llm": empty_llm_usage(),
        "error": None,
        "metadata": dict(metadata or {}),
    }


def summarize_operations(
    operations: Iterable[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    records = list(operations)
    latencies = [float(item["latency_ms"]) for item in records]
    successes = [item for item in records if item.get("success")]
    errors = [item for item in records if item.get("error")]
    total = len(records)
    total_llm_calls = sum(
        int(item.get("llm", {}).get("calls") or 0) for item in records
    )
    return {
        "success_rate": (len(successes) / total) if total else 0.0,
        "error_rate": (len(errors) / total) if total else 0.0,
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p95": percentile(latencies, 95),
        "latency_ms_p99": percentile(latencies, 99),
        "latency_ms_max": round(max(latencies), 3) if latencies else 0.0,
        "latency_ms_mean": round(statistics.fmean(latencies), 3) if latencies else 0.0,
        "throughput_ops_per_sec": round(total / elapsed_seconds, 3),
        "llm_calls_per_operation": round(total_llm_calls / total, 3) if total else 0.0,
        "task_count": sum(int(item.get("task_count") or 0) for item in records),
        "evidence_count": sum(int(item.get("evidence_count") or 0) for item in records),
        "error_classes": error_classes(records),
        "capability_sequences": _capability_sequence_counts(records),
    }


def assert_latency_gates(
    artifact: dict[str, Any],
    *,
    success_rate: float,
    p50_ms: float | None = None,
    p95_ms: float | None = None,
    p99_ms: float | None = None,
    llm_calls_per_operation: float | None = None,
) -> None:
    summary = artifact["summary"]
    assert summary["success_rate"] >= success_rate, summary
    if p50_ms is not None:
        assert summary["latency_ms_p50"] < p50_ms, summary
    if p95_ms is not None:
        assert summary["latency_ms_p95"] < p95_ms, summary
    if p99_ms is not None:
        assert summary["latency_ms_p99"] < p99_ms, summary
    if llm_calls_per_operation is not None:
        assert summary["llm_calls_per_operation"] <= llm_calls_per_operation, summary


def write_artifact(artifact: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, default=str), "utf-8"
    )
    print(f"[from_db_perf_artifact] {path}")
    return path


def validate_artifact_contract(artifact: dict[str, Any]) -> None:
    required_top = {
        "suite",
        "started_at",
        "environment",
        "parameters",
        "summary",
        "operations",
    }
    assert required_top <= set(artifact)
    required_environment = {"python", "platform", "model", "dataset"}
    assert required_environment <= set(artifact["environment"])
    required_summary = {
        "success_rate",
        "error_rate",
        "latency_ms_p50",
        "latency_ms_p95",
        "latency_ms_p99",
        "latency_ms_max",
        "throughput_ops_per_sec",
        "llm_calls_per_operation",
        "task_count",
        "evidence_count",
        "error_classes",
        "capability_sequences",
    }
    assert required_summary <= set(artifact["summary"])
    for operation in artifact["operations"]:
        required_operation = {
            "index",
            "started_at",
            "latency_ms",
            "success",
            "status",
            "operation_id",
            "capability_sequence",
            "task_count",
            "evidence_count",
            "llm",
            "error",
            "metadata",
        }
        assert required_operation <= set(operation)
        assert {"calls", "tokens", "estimated_cost_usd"} <= set(operation["llm"])


def default_environment_metadata(**extra: Any) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "model": os.environ.get("OPENAI_TEST_MODEL"),
        "dataset": None,
        "postgres_version": None,
        "daita_eval_postgres": os.environ.get("DAITA_EVAL_POSTGRES"),
        "daita_run_live_llm": os.environ.get("DAITA_RUN_LIVE_LLM"),
        **extra,
    }


def postgres_live_required() -> None:
    import pytest

    if os.environ.get("DAITA_EVAL_POSTGRES") != "1":
        pytest.skip(
            "Set DAITA_EVAL_POSTGRES=1 to run live PostgreSQL performance tests"
        )


def live_llm_required() -> None:
    import pytest

    if os.environ.get("DAITA_RUN_LIVE_LLM") != "1":
        pytest.skip("Set DAITA_RUN_LIVE_LLM=1 to run live LLM performance tests")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")


def capability_sequence(tasks: Iterable[Any]) -> list[str]:
    return [str(_task_value(task, "capability_id")) for task in tasks]


def task_status_counts(tasks: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        status = _task_value(task, "status")
        value = status.value if isinstance(status, TaskStatus) else str(status)
        counts[value] = counts.get(value, 0) + 1
    return counts


def llm_usage_from_result(result: Any) -> dict[str, Any]:
    usage = empty_llm_usage()
    diagnostics = _diagnostics(result)
    _merge_llm_usage(usage, diagnostics.get("llm"))
    _merge_llm_usage(usage, diagnostics.get("synthesis", {}).get("diagnostics"))
    execution = diagnostics.get("execution")
    if isinstance(execution, dict):
        _merge_llm_usage(usage, execution.get("query_plan", {}).get("llm"))
    for evidence in getattr(result, "evidence", ()) or ():
        payload = getattr(evidence, "payload", {}) or {}
        if isinstance(payload, dict):
            _merge_llm_usage(usage, payload.get("diagnostics"))
            _merge_llm_usage(usage, payload.get("llm"))
            if payload.get("diagnostics", {}).get("mode") == "llm":
                usage["calls"] += 1
    return usage


def empty_llm_usage() -> dict[str, Any]:
    return {"calls": 0, "tokens": {}, "estimated_cost_usd": None}


def classify_error(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def error_classes(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    classes: dict[str, int] = {}
    for record in records:
        error = record.get("error")
        if not error:
            continue
        name = str(
            error.get("type") if isinstance(error, dict) else type(error).__name__
        )
        classes[name] = classes.get(name, 0) + 1
    return classes


def percentile(values: list[float], percentile_rank: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    index = int(round((percentile_rank / 100) * (len(ordered) - 1)))
    return round(ordered[index], 3)


def _diagnostics(result: Any) -> dict[str, Any]:
    diagnostics = getattr(result, "diagnostics", {}) or {}
    return diagnostics if isinstance(diagnostics, dict) else {}


def _tasks_from_result(result: Any) -> tuple[Any, ...]:
    diagnostics = _diagnostics(result)
    execution = diagnostics.get("execution") if isinstance(diagnostics, dict) else None
    tasks = execution.get("tasks") if isinstance(execution, dict) else ()
    return tuple(tasks or ())


def _task_count(diagnostics: dict[str, Any], tasks: tuple[Any, ...]) -> int:
    execution = diagnostics.get("execution") if isinstance(diagnostics, dict) else None
    if isinstance(execution, dict) and execution.get("task_count") is not None:
        return int(execution["task_count"])
    return len(tasks)


def _task_value(task: Any, key: str) -> Any:
    if isinstance(task, dict):
        return task.get(key)
    return getattr(task, key, None)


def _merge_llm_usage(usage: dict[str, Any], candidate: Any) -> None:
    if not isinstance(candidate, dict):
        return
    mode = candidate.get("mode")
    if mode == "llm" or candidate.get("provider") or candidate.get("model"):
        usage["calls"] += 1
    tokens = candidate.get("tokens")
    if not isinstance(tokens, dict):
        tokens = {
            key: candidate.get(key)
            for key in (
                "input_tokens",
                "output_tokens",
                "prompt_tokens",
                "completion_tokens",
            )
            if candidate.get(key) is not None
        }
    if tokens:
        merged = dict(usage["tokens"])
        for key, value in tokens.items():
            if isinstance(value, int | float):
                merged[key] = merged.get(key, 0) + value
        usage["tokens"] = merged
    cost = candidate.get("estimated_cost_usd") or candidate.get("cost")
    if isinstance(cost, int | float):
        usage["estimated_cost_usd"] = (usage["estimated_cost_usd"] or 0) + cost


def _capability_sequence_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        sequence = " > ".join(record.get("capability_sequence") or ())
        counts[sequence] = counts.get(sequence, 0) + 1
    return counts


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()
