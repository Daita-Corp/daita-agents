"""Shared benchmark orchestration and artifact writing for from_db scale tests."""

from __future__ import annotations

import asyncio
import ast
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any, Awaitable, Callable, Iterable

from daita.runtime import OperationStatus, TaskStatus

OperationFactory = Callable[[int], Awaitable[Any]]

NEUTRAL_ARTIFACT_SCHEMA_NAME = "daita.from_db.operation-benchmark"
NEUTRAL_ARTIFACT_SCHEMA_VERSION = "1.0.0"
ARCHITECTURE_INVENTORY_SCHEMA_NAME = "daita.from_db.architecture-inventory"
ARCHITECTURE_INVENTORY_SCHEMA_VERSION = "1.0.0"

_HARNESS_FILES = (
    "tests/performance/from_db/scale_runner.py",
    "tests/performance/from_db/test_runtime_observability_contract.py",
    "tests/performance/from_db/test_postgres_large_schema_load_live.py",
    "tests/integration/evals/eval_from_db_factories.py",
    "tests/integration/evals/test_from_db_postgres_performance_live.py",
    "tests/integration/evals/test_from_db_postgres_quality_benchmark_live.py",
    "tests/integration/evals/test_from_db_postgres_wide_schema_live.py",
    "tests/integration/from_db/test_from_db_phase0_baseline_live.py",
)
_TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
    "total_tokens",
)


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


async def measure_agent_operation(
    agent: Any,
    prompt: str,
    *,
    measurement: dict[str, Any] | None = None,
    correctness_evaluator: Callable[[Any, Any], dict[str, Any]] | None = None,
    run_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one DB-agent operation and retain its authorized measurement surfaces."""

    provider = _agent_provider(agent)
    provider_before = _provider_metric_snapshot(provider)
    started = time.perf_counter()
    result = await agent.run_detailed(prompt, **dict(run_kwargs or {}))
    operation_latency_ms = (time.perf_counter() - started) * 1000
    snapshot = await agent.runtime.inspect_operation(result.operation_id)
    traces = _operation_traces(result, snapshot)
    provider_after = _provider_metric_snapshot(provider)
    resolved_measurement = dict(measurement or {})
    if correctness_evaluator is not None:
        try:
            resolved_measurement["correctness"] = correctness_evaluator(
                result, snapshot
            )
        except Exception as exc:  # noqa: BLE001 - record evaluator failures faithfully
            resolved_measurement["correctness"] = {
                "answer": {"passed": False, "error": classify_error(exc)},
                "sql": {"passed": False, "error": classify_error(exc)},
            }
    return {
        "_phase0_measurement": True,
        "runtime_result": result,
        "operation_snapshot": snapshot,
        "model_traces": traces,
        "provider_delta": _provider_metric_delta(provider_before, provider_after),
        "operation_latency_ms": operation_latency_ms,
        "measurement": resolved_measurement,
    }


def measured_agent_operation_factory(
    agent: Any,
    prompt: str,
    *,
    measurement: dict[str, Any] | None = None,
    run_kwargs: dict[str, Any] | None = None,
) -> OperationFactory:
    """Return a scale-runner operation factory with full DB measurement capture."""

    async def _run(index: int) -> dict[str, Any]:
        return await measure_agent_operation(
            agent,
            prompt,
            measurement={
                **dict(measurement or {}),
                "run_id": f"run-{index + 1:03d}",
            },
            run_kwargs=run_kwargs,
        )

    return _run


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
    resolved_environment = {
        **default_environment_metadata(),
        **dict(environment or {}),
    }
    benchmark_context = {
        "suite": suite,
        "scenario": parameters.scenario,
        "concurrency": parameters.concurrency,
        "environment": resolved_environment,
    }

    async def _run_one(index: int) -> dict[str, Any]:
        async with semaphore:
            return await measure_operation(
                index,
                operation_factory,
                benchmark_context=benchmark_context,
            )

    operations = await asyncio.gather(
        *(_run_one(index) for index in range(parameters.operations))
    )
    elapsed_s = max(time.perf_counter() - started, 0.000001)
    artifact = {
        "schema": {
            "name": NEUTRAL_ARTIFACT_SCHEMA_NAME,
            "version": NEUTRAL_ARTIFACT_SCHEMA_VERSION,
        },
        "suite": suite,
        "started_at": started_at,
        "finished_at": _iso_now(),
        "environment": resolved_environment,
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
    *,
    benchmark_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started_at = _iso_now()
    started = time.perf_counter()
    try:
        raw_result = await operation_factory(index)
        latency_ms = (time.perf_counter() - started) * 1000
        envelope = _measurement_envelope(raw_result)
        result = envelope.get("runtime_result", raw_result)
        return operation_record(
            index=index,
            latency_ms=float(envelope.get("operation_latency_ms") or latency_ms),
            started_at=started_at,
            result=result,
            snapshot=envelope.get("operation_snapshot"),
            model_traces=envelope.get("model_traces") or (),
            provider_delta=envelope.get("provider_delta"),
            metadata=envelope.get("metadata"),
            measurement={
                **dict(benchmark_context or {}),
                **dict(envelope.get("measurement") or {}),
            },
        )
    except Exception as exc:  # noqa: BLE001 - benchmark artifacts classify all errors
        latency_ms = (time.perf_counter() - started) * 1000
        return operation_record(
            index=index,
            latency_ms=latency_ms,
            started_at=started_at,
            error=exc,
            measurement=dict(benchmark_context or {}),
        )


def operation_record(
    *,
    index: int,
    latency_ms: float,
    started_at: str,
    result: Any | None = None,
    error: BaseException | None = None,
    metadata: dict[str, Any] | None = None,
    snapshot: Any | None = None,
    model_traces: Iterable[Any] = (),
    provider_delta: dict[str, Any] | None = None,
    measurement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    measurement = dict(measurement or {})
    environment = dict(measurement.get("environment") or {})
    scenario = measurement.get("scenario")
    run_id = measurement.get("run_id") or f"run-{index + 1:03d}"
    if error is not None:
        return {
            "schema_version": NEUTRAL_ARTIFACT_SCHEMA_VERSION,
            "index": index,
            "run_id": run_id,
            "scenario": scenario,
            "started_at": started_at,
            "latency_ms": round(latency_ms, 3),
            "latency": {
                "model_ms": None,
                "operation_ms": round(latency_ms, 3),
            },
            "success": False,
            "status": "error",
            "operation_id": None,
            "capability_sequence": [],
            "task_count": 0,
            "evidence_count": 0,
            "event_count": 0,
            "model_calls": [],
            "llm": empty_llm_usage(),
            "error": classify_error(error),
            "errors": {
                "framework": classify_error(error),
                "provider": None,
            },
            "provenance": _operation_provenance(environment, measurement),
            "correctness": _correctness_record(measurement),
            "metadata": dict(metadata or {}),
        }

    diagnostics = _diagnostics(result)
    tasks = tuple(getattr(snapshot, "tasks", ()) or ()) or _tasks_from_result(result)
    evidence = tuple(getattr(snapshot, "evidence", ()) or ()) or tuple(
        getattr(result, "evidence", ()) or ()
    )
    events = tuple(getattr(snapshot, "events", ()) or ())
    status = getattr(result, "status", None)
    if isinstance(status, OperationStatus):
        status_value = status.value
    elif status is not None:
        status_value = str(status)
    else:
        status_value = "unknown"
    error_payload = diagnostics.get("error") if isinstance(diagnostics, dict) else None
    success = status_value == OperationStatus.SUCCEEDED.value and not error_payload
    planner = diagnostics.get("planner") if isinstance(diagnostics, dict) else None
    model_call_telemetry = aggregate_model_calls(
        tasks=tasks,
        evidence=evidence,
        events=events,
        traces=model_traces,
        provider_delta=(
            provider_delta if int(measurement.get("concurrency") or 1) == 1 else None
        ),
    )
    reported_telemetry = _reported_result_telemetry(result)
    telemetry_discrepancies = _telemetry_discrepancies(
        reported_telemetry,
        model_call_telemetry["summary"],
    )
    sql = _sql_measurement(tasks, evidence, diagnostics)
    context_sizes = _context_size_measurement(result, evidence, model_call_telemetry)
    return {
        "schema_version": NEUTRAL_ARTIFACT_SCHEMA_VERSION,
        "index": index,
        "run_id": run_id,
        "scenario": scenario,
        "started_at": started_at,
        "latency_ms": round(latency_ms, 3),
        "latency": {
            "model_ms": model_call_telemetry["summary"]["model_latency_ms"],
            "operation_ms": round(latency_ms, 3),
        },
        "success": success,
        "status": status_value,
        "operation_id": getattr(result, "operation_id", None),
        "capability_sequence": capability_sequence(tasks),
        "task_sequence": [_task_measurement(task) for task in tasks],
        "task_count": _task_count(diagnostics, tasks),
        "evidence_count": len(evidence),
        "evidence_kinds": [str(getattr(item, "kind", "")) for item in evidence],
        "evidence_sequence": [_evidence_measurement(item) for item in evidence],
        "event_count": len(events),
        "event_types": [
            str(_enum_value(_record_value(item, "type"))) for item in events
        ],
        "model_calls": model_call_telemetry["calls"],
        "llm": _legacy_llm_summary(model_call_telemetry["summary"]),
        "model_call_summary": model_call_telemetry["summary"],
        "reported_result_telemetry": reported_telemetry,
        "telemetry_discrepancies": telemetry_discrepancies,
        "error": error_payload,
        "errors": {
            "framework": _classified_result_error(error_payload),
            "provider": _provider_error(model_call_telemetry["calls"]),
        },
        "provenance": _operation_provenance(environment, measurement),
        "correctness": _correctness_record(measurement),
        "sql": sql,
        "repair_count": _repair_count(tasks, evidence),
        "catalog": _catalog_measurement(tasks, evidence),
        "context_sizes": context_sizes,
        "truncation_redaction": _truncation_redaction_facts(evidence),
        "metadata": {
            **dict(metadata or {}),
            "warnings": list(getattr(result, "warnings", ()) or ()),
            "answer": getattr(result, "answer", None),
            "planned_sql": (
                diagnostics.get("execution", {}).get("planned_sql")
                if isinstance(diagnostics.get("execution"), dict)
                else None
            ),
            "planner_status": (
                planner.get("status") if isinstance(planner, dict) else None
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
    model_traces: Iterable[Any] = (),
    provider_delta: dict[str, Any] | None = None,
    measurement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if error is not None:
        return operation_record(
            index=index,
            latency_ms=latency_ms,
            started_at=started_at,
            error=error,
            metadata=metadata,
            measurement=measurement,
        )
    tasks = tuple(getattr(snapshot, "tasks", ()) or ())
    evidence = tuple(getattr(snapshot, "evidence", ()) or ())
    events = tuple(getattr(snapshot, "events", ()) or ())
    operation = getattr(snapshot, "operation", None)
    status = getattr(operation, "status", None)
    status_value = status.value if isinstance(status, OperationStatus) else str(status)
    model_call_telemetry = aggregate_model_calls(
        tasks=tasks,
        evidence=evidence,
        events=events,
        traces=model_traces,
        provider_delta=provider_delta,
    )
    return {
        "schema_version": NEUTRAL_ARTIFACT_SCHEMA_VERSION,
        "index": index,
        "run_id": dict(measurement or {}).get("run_id") or f"run-{index + 1:03d}",
        "scenario": dict(measurement or {}).get("scenario"),
        "started_at": started_at,
        "latency_ms": round(latency_ms, 3),
        "latency": {
            "model_ms": model_call_telemetry["summary"]["model_latency_ms"],
            "operation_ms": round(latency_ms, 3),
        },
        "success": status is OperationStatus.SUCCEEDED,
        "status": status_value,
        "operation_id": getattr(operation, "id", None),
        "capability_sequence": capability_sequence(tasks),
        "task_sequence": [_task_measurement(task) for task in tasks],
        "task_count": len(tasks),
        "evidence_count": len(evidence),
        "evidence_kinds": [str(getattr(item, "kind", "")) for item in evidence],
        "evidence_sequence": [_evidence_measurement(item) for item in evidence],
        "event_count": len(events),
        "event_types": [
            str(_enum_value(_record_value(item, "type"))) for item in events
        ],
        "model_calls": model_call_telemetry["calls"],
        "llm": _legacy_llm_summary(model_call_telemetry["summary"]),
        "model_call_summary": model_call_telemetry["summary"],
        "error": None,
        "errors": {
            "framework": None,
            "provider": _provider_error(model_call_telemetry["calls"]),
        },
        "provenance": _operation_provenance(
            dict(dict(measurement or {}).get("environment") or {}),
            dict(measurement or {}),
        ),
        "correctness": _correctness_record(dict(measurement or {})),
        "sql": _sql_measurement(tasks, evidence, {}),
        "repair_count": _repair_count(tasks, evidence),
        "catalog": _catalog_measurement(tasks, evidence),
        "context_sizes": _context_size_measurement(
            None, evidence, model_call_telemetry
        ),
        "truncation_redaction": _truncation_redaction_facts(evidence),
        "metadata": dict(metadata or {}),
    }


def aggregate_model_calls(
    *,
    tasks: Iterable[Any] = (),
    evidence: Iterable[Any] = (),
    events: Iterable[Any] = (),
    traces: Iterable[Any] = (),
    provider_delta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate all observable model calls without treating absence as zero."""

    task_records = [_record_mapping(item) for item in tasks]
    task_by_id = {str(item.get("id")): item for item in task_records if item.get("id")}
    diagnostic_calls: list[dict[str, Any]] = []
    for source_type, records in (("evidence", evidence), ("event", events)):
        for raw in records:
            record = _record_mapping(raw)
            diagnostic_calls.extend(
                _model_calls_from_record(
                    record,
                    source_type=source_type,
                    task_by_id=task_by_id,
                )
            )
    diagnostic_calls = _deduplicate_diagnostic_calls(diagnostic_calls)
    trace_calls = _model_calls_from_traces(traces)
    calls = _merge_trace_and_diagnostic_calls(trace_calls, diagnostic_calls)
    summary = _model_call_summary(calls, provider_delta=provider_delta)
    return {"calls": calls, "summary": summary}


def _model_calls_from_record(
    record: dict[str, Any],
    *,
    source_type: str,
    task_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    payload = _mapping(record.get("payload"))
    kind = str(record.get("kind") or record.get("type") or "")
    task_id = _optional_string(record.get("task_id"))
    task = task_by_id.get(task_id or "", {})
    capability_id = str(task.get("capability_id") or record.get("capability_id") or "")
    stage = _model_stage(kind=kind, capability_id=capability_id)
    turn = _optional_int(
        payload.get("turn")
        or _mapping(task.get("metadata")).get("planner_turn")
        or _mapping(task.get("metadata")).get("turn")
    )
    source = {
        "type": source_type,
        "id": record.get("id"),
        "kind": kind or None,
    }
    diagnostics = list(_known_model_diagnostics(kind, payload))
    if not diagnostics:
        diagnostics = list(_walk_model_diagnostics(payload))
    return [
        _normalized_model_call(
            candidate,
            stage=stage,
            turn=turn,
            task_id=task_id,
            source=source,
        )
        for candidate in diagnostics
    ]


def _known_model_diagnostics(
    kind: str, payload: dict[str, Any]
) -> Iterable[dict[str, Any]]:
    candidates: list[Any] = []
    if kind == "planner.decision":
        candidates.append(
            _mapping(
                _mapping(_mapping(payload.get("decision")).get("metadata")).get("llm")
            )
        )
    elif kind in {"query.plan.proposal", "query.plan.repair"}:
        candidates.extend([payload.get("planner_diagnostics"), payload.get("llm")])
    elif kind == "analysis.plan":
        candidates.append(_mapping(payload.get("diagnostics")).get("llm"))
    elif kind in {"analysis.synthesis", "answer.synthesis"}:
        candidates.append(payload.get("diagnostics"))
    for candidate in candidates:
        value = _mapping(candidate)
        if _looks_like_model_diagnostics(value):
            yield value


def _walk_model_diagnostics(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        nested_model_mapping = False
        for key, child in value.items():
            if key in {
                "llm",
                "planner_diagnostics",
                "synthesis_diagnostics",
                "model_diagnostics",
            }:
                candidate = _mapping(child)
                if _looks_like_model_diagnostics(candidate):
                    nested_model_mapping = True
                    yield candidate
            yield from _walk_model_diagnostics(child)
        if not nested_model_mapping and _looks_like_model_diagnostics(value):
            yield value
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_model_diagnostics(child)


def _looks_like_model_diagnostics(value: dict[str, Any]) -> bool:
    if not value:
        return False
    mode = str(value.get("mode") or "")
    if mode.startswith("deterministic") or value.get("provider") == "daita.db":
        return False
    if value.get("model") == "deterministic":
        return False
    identity = bool(value.get("provider") or value.get("model"))
    usage = bool(
        value.get("tokens")
        or any(value.get(key) is not None for key in (*_TOKEN_FIELDS, "latency_ms"))
        or value.get("estimated_cost_usd") is not None
        or value.get("estimated_cost") is not None
    )
    return mode == "llm" or (identity and usage)


def _normalized_model_call(
    diagnostics: dict[str, Any],
    *,
    stage: str,
    turn: int | None,
    task_id: str | None,
    source: dict[str, Any],
) -> dict[str, Any]:
    tokens = _mapping(diagnostics.get("tokens"))
    input_tokens = _optional_number(
        diagnostics.get("input_tokens")
        if diagnostics.get("input_tokens") is not None
        else tokens.get("input_tokens", tokens.get("prompt_tokens"))
    )
    output_tokens = _optional_number(
        diagnostics.get("output_tokens")
        if diagnostics.get("output_tokens") is not None
        else tokens.get("output_tokens", tokens.get("completion_tokens"))
    )
    total_tokens = _optional_number(
        diagnostics.get("total_tokens")
        if diagnostics.get("total_tokens") is not None
        else tokens.get("total_tokens")
    )
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    cost = diagnostics.get("estimated_cost_usd")
    if cost is None:
        cost = diagnostics.get("estimated_cost", diagnostics.get("cost"))
    call = {
        "call_id": _explicit_call_id(diagnostics),
        "stage": stage,
        "turn": turn,
        "task_id": task_id,
        "provider": _optional_string(diagnostics.get("provider")),
        "model": _optional_string(diagnostics.get("model")),
        "model_parameters": _safe_model_parameters(diagnostics),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": _optional_number(
            diagnostics.get("cached_input_tokens")
            if diagnostics.get("cached_input_tokens") is not None
            else tokens.get("cached_input_tokens")
        ),
        "reasoning_tokens": _optional_number(
            diagnostics.get("reasoning_tokens")
            if diagnostics.get("reasoning_tokens") is not None
            else tokens.get("reasoning_tokens")
        ),
        "total_tokens": total_tokens,
        "estimated_cost_usd": _optional_number(cost),
        "latency_ms": _optional_number(diagnostics.get("latency_ms")),
        "prompt_chars": _optional_number(diagnostics.get("prompt_chars")),
        "observation_chars": _optional_number(diagnostics.get("observation_chars")),
        "status": _optional_string(diagnostics.get("status")) or "observed",
        "sources": [source],
    }
    if call["call_id"] is None:
        call["call_id"] = "diagnostic:" + _stable_json_hash(
            {
                key: value
                for key, value in call.items()
                if key not in {"call_id", "sources", "status"}
            }
        )
    return call


def _model_calls_from_traces(traces: Iterable[Any]) -> list[dict[str, Any]]:
    records = [_record_mapping(item) for item in traces]
    by_span_id = {
        str(record.get("span_id")): record
        for record in records
        if record.get("span_id")
    }
    calls = []
    for record in records:
        if str(record.get("type") or "") != "llm_call":
            continue
        metadata = _mapping(record.get("metadata"))
        capability_id, task_id = _trace_runtime_ancestor(record, by_span_id)
        operation_name = str(record.get("operation") or "")
        provider = operation_name.removeprefix("llm_") or None
        calls.append(
            {
                "call_id": str(record.get("span_id") or "trace-unknown"),
                "stage": _model_stage(kind="", capability_id=capability_id),
                "turn": None,
                "task_id": task_id,
                "provider": provider,
                "model": _optional_string(metadata.get("model")),
                "model_parameters": {},
                **{key: None for key in _TOKEN_FIELDS},
                "estimated_cost_usd": None,
                "latency_ms": _optional_number(record.get("duration_ms")),
                "prompt_chars": (
                    len(str(record.get("input_preview")))
                    if record.get("input_preview") is not None
                    else None
                ),
                "observation_chars": None,
                "status": _optional_string(record.get("status")) or "unknown",
                "sources": [
                    {
                        "type": "trace",
                        "id": record.get("span_id"),
                        "kind": operation_name or None,
                    }
                ],
            }
        )
    return calls


def _trace_runtime_ancestor(
    record: dict[str, Any], by_span_id: dict[str, dict[str, Any]]
) -> tuple[str, str | None]:
    parent_id = _optional_string(record.get("parent_span_id"))
    for _ in range(12):
        if not parent_id:
            break
        parent = by_span_id.get(parent_id)
        if parent is None:
            break
        metadata = _mapping(parent.get("metadata"))
        capability_id = str(metadata.get("capability_id") or "")
        if capability_id:
            return capability_id, _optional_string(metadata.get("task_id"))
        parent_id = _optional_string(parent.get("parent_span_id"))
    return "", None


def _deduplicate_diagnostic_calls(
    calls: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    deduplicated: dict[str, dict[str, Any]] = {}
    for call in calls:
        key = str(call["call_id"])
        existing = deduplicated.get(key)
        if existing is None:
            deduplicated[key] = call
            continue
        existing["sources"] = _unique_mappings(
            [*existing.get("sources", ()), *call.get("sources", ())]
        )
    return list(deduplicated.values())


def _merge_trace_and_diagnostic_calls(
    trace_calls: list[dict[str, Any]], diagnostic_calls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not trace_calls:
        return diagnostic_calls
    unmatched = set(range(len(trace_calls)))
    merged = [dict(call) for call in trace_calls]
    for diagnostic in diagnostic_calls:
        candidates = [
            index
            for index in unmatched
            if merged[index].get("stage") == diagnostic.get("stage")
        ]
        if not candidates:
            candidates = [
                index
                for index in unmatched
                if not diagnostic.get("provider")
                or merged[index].get("provider") == diagnostic.get("provider")
            ]
        if not candidates:
            merged.append(diagnostic)
            continue
        index = min(
            candidates,
            key=lambda item: _latency_distance(merged[item], diagnostic),
        )
        unmatched.remove(index)
        trace = merged[index]
        for key, value in diagnostic.items():
            if key == "call_id":
                continue
            if key == "sources":
                trace[key] = _unique_mappings(
                    [*trace.get(key, ()), *diagnostic.get(key, ())]
                )
            elif value is not None and (trace.get(key) is None or key in _TOKEN_FIELDS):
                trace[key] = value
    return merged


def _model_call_summary(
    calls: list[dict[str, Any]],
    *,
    provider_delta: dict[str, Any] | None,
) -> dict[str, Any]:
    provider_delta = dict(provider_delta or {})
    provider_usable = bool(calls) and _optional_number(
        provider_delta.get("total_tokens")
    ) not in {None, 0}
    field_totals: dict[str, Any] = {}
    unattributed: dict[str, Any] = {}
    discrepancies: list[dict[str, Any]] = []
    for field in (*_TOKEN_FIELDS, "estimated_cost_usd"):
        known_values = [
            value
            for value in (_optional_number(call.get(field)) for call in calls)
            if value is not None
        ]
        call_sum = sum(known_values) if known_values else None
        provider_value = (
            _optional_number(provider_delta.get(field)) if provider_usable else None
        )
        if provider_value is not None:
            field_totals[field] = provider_value
            if call_sum is not None and provider_value > call_sum:
                unattributed[field] = provider_value - call_sum
            elif call_sum is not None and provider_value < call_sum:
                discrepancies.append(
                    {
                        "field": field,
                        "provider_delta": provider_value,
                        "persisted_call_sum": call_sum,
                    }
                )
        elif len(known_values) == len(calls) and calls:
            field_totals[field] = call_sum
        else:
            field_totals[field] = None
    latencies = [_optional_number(call.get("latency_ms")) for call in calls]
    model_latency = (
        sum(value for value in latencies if value is not None)
        if calls and all(value is not None for value in latencies)
        else None
    )
    return {
        "call_count": len(calls),
        **field_totals,
        "model_latency_ms": model_latency,
        "unattributed_usage": unattributed,
        "discrepancies": discrepancies,
        "all_calls_have_trace_identity": bool(calls)
        and all(
            any(source.get("type") == "trace" for source in call.get("sources", ()))
            for call in calls
        ),
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
    aggregate_model_usage = {
        field: _sum_operation_metric(records, field)
        for field in (*_TOKEN_FIELDS, "estimated_cost_usd", "model_latency_ms")
    }
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
        "model_usage": {
            "call_count": total_llm_calls,
            **aggregate_model_usage,
        },
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
        "schema",
        "suite",
        "started_at",
        "environment",
        "parameters",
        "summary",
        "operations",
    }
    assert required_top <= set(artifact)
    assert artifact["schema"] == {
        "name": NEUTRAL_ARTIFACT_SCHEMA_NAME,
        "version": NEUTRAL_ARTIFACT_SCHEMA_VERSION,
    }
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
            "schema_version",
            "index",
            "run_id",
            "scenario",
            "started_at",
            "latency_ms",
            "latency",
            "success",
            "status",
            "operation_id",
            "capability_sequence",
            "task_count",
            "evidence_count",
            "event_count",
            "model_calls",
            "llm",
            "error",
            "provenance",
            "correctness",
            "metadata",
        }
        assert required_operation <= set(operation)
        assert {"calls", "tokens", "estimated_cost_usd"} <= set(operation["llm"])


def default_environment_metadata(**extra: Any) -> dict[str, Any]:
    repository = Path.cwd()
    git = git_metadata(repository)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "model": os.environ.get("OPENAI_TEST_MODEL"),
        "provider": "openai" if os.environ.get("OPENAI_API_KEY") else None,
        "model_parameters": {"temperature": 0},
        "dataset": None,
        "source_git_sha": git["source_git_sha"],
        "measurement_harness_sha": measurement_harness_sha(repository),
        "measurement_harness_git_sha": git["source_git_sha"],
        "branch": git["branch"],
        "control_label": os.environ.get("DAITA_PHASE0_CONTROL_LABEL", "baseline"),
        "dirty": git["dirty"],
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
    planner = diagnostics.get("planner")
    if isinstance(planner, dict):
        _merge_llm_usage(usage, planner.get("diagnostics", {}).get("llm"))
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


def _measurement_envelope(value: Any) -> dict[str, Any]:
    if isinstance(value, dict) and value.get("_phase0_measurement") is True:
        return value
    return {}


def _agent_provider(agent: Any) -> Any | None:
    runtime = getattr(agent, "runtime", None)
    service = getattr(runtime, "db_llm_service", None)
    if service is None or not bool(getattr(service, "available", False)):
        return None
    return service.provider


def _provider_metric_snapshot(provider: Any | None) -> dict[str, Any]:
    if provider is None:
        return {}
    token_getter = getattr(provider, "get_accumulated_tokens", None)
    cost_getter = getattr(provider, "get_accumulated_cost", None)
    tokens = token_getter() if callable(token_getter) else {}
    return {
        "input_tokens": _optional_number(
            _mapping(tokens).get("input_tokens", _mapping(tokens).get("prompt_tokens"))
        ),
        "output_tokens": _optional_number(
            _mapping(tokens).get(
                "output_tokens", _mapping(tokens).get("completion_tokens")
            )
        ),
        "cached_input_tokens": _optional_number(
            _mapping(tokens).get("cached_input_tokens")
        ),
        "reasoning_tokens": _optional_number(_mapping(tokens).get("reasoning_tokens")),
        "total_tokens": _optional_number(_mapping(tokens).get("total_tokens")),
        "estimated_cost_usd": _optional_number(
            cost_getter() if callable(cost_getter) else None
        ),
    }


def _provider_metric_delta(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    result = {}
    for key in (*_TOKEN_FIELDS, "estimated_cost_usd"):
        left = _optional_number(before.get(key))
        right = _optional_number(after.get(key))
        result[key] = right - left if left is not None and right is not None else None
    return result


def _operation_traces(result: Any, snapshot: Any) -> list[dict[str, Any]]:
    trace_id = None
    diagnostics = _diagnostics(result)
    trace = _mapping(diagnostics.get("trace"))
    trace_id = trace.get("trace_id")
    if trace_id is None and snapshot is not None:
        operation = getattr(snapshot, "operation", None)
        trace_id = _mapping(
            _mapping(getattr(operation, "metadata", {})).get("trace")
        ).get("trace_id")
    if not trace_id:
        return []
    try:
        from daita.core.tracing import get_trace_manager

        manager = get_trace_manager()
        manager.flush(timeout_millis=5000)
        return [
            item
            for item in manager.get_recent_operations(limit=500)
            if item.get("trace_id") == trace_id
        ]
    except Exception:
        return []


def _model_stage(*, kind: str, capability_id: str) -> str:
    capability = str(capability_id or "")
    if capability == "db.query.plan":
        return "sql_planner"
    if capability == "db.query.repair":
        return "query_repair"
    if capability == "db.answer.synthesize":
        return "final_synthesis"
    if capability == "db.analysis.plan":
        return "analysis_planner"
    if capability == "db.analysis.summarize":
        return "analysis_summarizer"
    if ".memory." in capability:
        return "memory"
    if ".monitor." in capability or capability.startswith("monitor."):
        return "monitor"
    if kind == "planner.decision" or (not kind and not capability):
        return "outer_planner"
    if kind == "query.plan.repair":
        return "query_repair"
    if kind == "query.plan.proposal":
        return "sql_planner"
    if kind == "analysis.plan":
        return "analysis_planner"
    if kind == "analysis.synthesis":
        return "analysis_summarizer"
    if kind == "answer.synthesis":
        return "final_synthesis"
    if kind.startswith("db.memory.") or kind.startswith("memory."):
        return "memory"
    if kind.startswith("monitor."):
        return "monitor"
    return "unknown"


def _explicit_call_id(diagnostics: dict[str, Any]) -> str | None:
    for key in ("model_call_id", "call_id", "response_id", "request_id"):
        if diagnostics.get(key):
            return str(diagnostics[key])
    return None


def _safe_model_parameters(diagnostics: dict[str, Any]) -> dict[str, Any]:
    parameters = _mapping(diagnostics.get("model_parameters"))
    for key in ("temperature", "top_p", "max_tokens"):
        if diagnostics.get(key) is not None:
            parameters[key] = diagnostics[key]
    return parameters


def _latency_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_value = _optional_number(left.get("latency_ms"))
    right_value = _optional_number(right.get("latency_ms"))
    if left_value is None or right_value is None:
        return float("inf")
    return abs(left_value - right_value)


def _legacy_llm_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "calls": summary.get("call_count"),
        "tokens": {
            key: summary.get(key)
            for key in _TOKEN_FIELDS
            if summary.get(key) is not None
        },
        "estimated_cost_usd": summary.get("estimated_cost_usd"),
    }


def _reported_result_telemetry(result: Any) -> dict[str, Any] | None:
    try:
        telemetry = getattr(result, "telemetry")
    except Exception:
        return None
    return dict(telemetry) if isinstance(telemetry, dict) else None


def _telemetry_discrepancies(
    reported: dict[str, Any] | None, complete: dict[str, Any]
) -> list[dict[str, Any]]:
    if not reported:
        return []
    discrepancies = []
    for reported_key, complete_key in (
        ("llm_calls", "call_count"),
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("total_tokens", "total_tokens"),
        ("estimated_cost_usd", "estimated_cost_usd"),
    ):
        public_value = _optional_number(reported.get(reported_key))
        full_value = _optional_number(complete.get(complete_key))
        if (
            public_value is not None
            and full_value is not None
            and public_value != full_value
        ):
            discrepancies.append(
                {
                    "field": reported_key,
                    "reported_result": public_value,
                    "complete_operation": full_value,
                }
            )
    return discrepancies


def _task_measurement(task: Any) -> dict[str, Any]:
    return {
        "id": _record_value(task, "id"),
        "capability_id": _record_value(task, "capability_id"),
        "executor_id": _record_value(task, "executor_id"),
        "status": _enum_value(_record_value(task, "status")),
    }


def _evidence_measurement(evidence: Any) -> dict[str, Any]:
    return {
        "id": _record_value(evidence, "id"),
        "kind": _record_value(evidence, "kind"),
        "owner": _record_value(evidence, "owner"),
        "task_id": _record_value(evidence, "task_id"),
        "accepted": _record_value(evidence, "accepted"),
    }


def _sql_measurement(
    tasks: Iterable[Any], evidence: Iterable[Any], diagnostics: dict[str, Any]
) -> dict[str, Any]:
    planned = None
    executed = None
    execution = _mapping(diagnostics.get("execution"))
    if isinstance(execution.get("planned_sql"), str):
        planned = execution["planned_sql"]
    for item in evidence:
        record = _record_mapping(item)
        payload = _mapping(record.get("payload"))
        sql = _first_sql(payload)
        if not sql:
            continue
        if record.get("kind") == "query.result" and record.get("accepted", True):
            executed = sql
        elif record.get("kind") in {
            "query.plan.proposal",
            "query.plan.validation",
            "sql.validation",
        }:
            planned = sql
    if planned is None:
        for task in tasks:
            planned = _first_sql(_mapping(_record_value(task, "input"))) or planned
    return {
        "planned_fingerprint": _sql_fingerprint(planned),
        "executed_fingerprint": _sql_fingerprint(executed),
        "read_only": _sql_read_only(executed or planned),
        "planned_present": planned is not None,
        "executed_present": executed is not None,
    }


def _context_size_measurement(
    result: Any | None,
    evidence: Iterable[Any],
    model_call_telemetry: dict[str, Any],
) -> dict[str, Any]:
    sections: dict[str, int] = {}
    request = getattr(result, "request", None)
    prompt = getattr(request, "prompt", None)
    if isinstance(prompt, str):
        sections["user_prompt"] = len(prompt)
    observation_chars = 0
    for item in evidence:
        record = _record_mapping(item)
        payload = _mapping(record.get("payload"))
        kind = str(record.get("kind") or "")
        if kind == "planning.context":
            rendered = payload.get("rendered_context")
            if isinstance(rendered, str):
                sections["planning_context"] = sections.get(
                    "planning_context", 0
                ) + len(rendered)
        if kind == "planner.observation":
            observation_chars += len(json.dumps(payload, sort_keys=True, default=str))
    trace_prompt_chars = [
        _optional_number(call.get("prompt_chars"))
        for call in model_call_telemetry.get("calls", ())
    ]
    return {
        "prompt_context_chars_by_section": sections,
        "model_visible_observation_chars": observation_chars,
        "model_call_prompt_chars": trace_prompt_chars,
    }


def _repair_count(tasks: Iterable[Any], evidence: Iterable[Any]) -> int:
    repair_task_ids = {
        str(_record_value(task, "id"))
        for task in tasks
        if _record_value(task, "capability_id") == "db.query.repair"
    }
    repair_evidence_task_ids = {
        str(_record_value(item, "task_id"))
        for item in evidence
        if _record_value(item, "kind") == "query.plan.repair"
        and _record_value(item, "task_id")
    }
    return len(repair_task_ids | repair_evidence_task_ids)


def _catalog_measurement(
    tasks: Iterable[Any], evidence: Iterable[Any]
) -> dict[str, Any]:
    capabilities = [
        str(_record_value(task, "capability_id") or "")
        for task in tasks
        if str(_record_value(task, "capability_id") or "").startswith("catalog.")
        or _record_value(task, "capability_id")
        in {"db.schema.inspect", "db.column_values.profile"}
    ]
    cache_facts = []
    for item in evidence:
        payload = _mapping(_record_value(item, "payload"))
        for key in ("cache_hit", "cache_status", "freshness", "profile_freshness"):
            if key in payload:
                cache_facts.append(
                    {"kind": _record_value(item, "kind"), key: payload[key]}
                )
    return {
        "task_count": len(capabilities),
        "capability_sequence": capabilities,
        "cache_behavior": cache_facts,
    }


def _truncation_redaction_facts(evidence: Iterable[Any]) -> dict[str, Any]:
    truncated = []
    redacted = []
    for item in evidence:
        record = _record_mapping(item)
        payload = _mapping(record.get("payload"))
        if payload.get("truncated") is True or any(
            value is True for key, value in payload.items() if "truncat" in str(key)
        ):
            truncated.append(record.get("kind"))
        if payload.get("redacted") is True:
            redacted.append(record.get("kind"))
    return {
        "truncated": bool(truncated),
        "truncated_evidence_kinds": [str(item) for item in truncated],
        "redacted": bool(redacted),
        "redacted_evidence_kinds": [str(item) for item in redacted],
    }


def _operation_provenance(
    environment: dict[str, Any], measurement: dict[str, Any]
) -> dict[str, Any]:
    return {
        "source_git_sha": environment.get("source_git_sha"),
        "measurement_harness_sha": environment.get("measurement_harness_sha"),
        "measurement_harness_git_sha": environment.get("measurement_harness_git_sha"),
        "branch": environment.get("branch"),
        "control_label": measurement.get("control_label")
        or environment.get("control_label"),
        "dirty": environment.get("dirty"),
        "provider": measurement.get("provider") or environment.get("provider"),
        "model": measurement.get("model") or environment.get("model"),
        "model_parameters": measurement.get("model_parameters")
        or environment.get("model_parameters")
        or {},
        "database": measurement.get("database")
        or {
            "type": environment.get("database_type"),
            "version": environment.get("postgres_version"),
        },
        "fixture_revision": measurement.get("fixture_revision")
        or environment.get("fixture_revision"),
        "state": measurement.get("state"),
        "concurrency": measurement.get("concurrency"),
    }


def _correctness_record(measurement: dict[str, Any]) -> dict[str, Any]:
    correctness = _mapping(measurement.get("correctness"))
    return {
        "answer": correctness.get("answer")
        or {"passed": measurement.get("answer_correct")},
        "sql": correctness.get("sql") or {"passed": measurement.get("sql_correct")},
    }


def _classified_result_error(error: Any) -> dict[str, Any] | None:
    if not error:
        return None
    if isinstance(error, dict):
        return {
            "type": str(error.get("type") or error.get("error_type") or "RuntimeError"),
            "message": str(error.get("message") or error.get("error") or ""),
        }
    return {"type": type(error).__name__, "message": str(error)}


def _provider_error(calls: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    failed = [call for call in calls if call.get("status") in {"error", "failed"}]
    if not failed:
        return None
    return {
        "type": "ProviderCallError",
        "call_ids": [call["call_id"] for call in failed],
    }


def _sum_operation_metric(
    records: list[dict[str, Any]], field: str
) -> int | float | None:
    values = [
        _optional_number(_mapping(item.get("model_call_summary")).get(field))
        for item in records
    ]
    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def git_metadata(repository: Path) -> dict[str, Any]:
    def _git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ("git", *args),
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = _git("status", "--porcelain")
    branch = _git("branch", "--show-current") or "detached"
    return {
        "source_git_sha": _git("rev-parse", "HEAD"),
        "branch": branch,
        "dirty": bool(status),
    }


def measurement_harness_sha(repository: Path) -> str:
    digest = hashlib.sha256()
    included = False
    for relative in _HARNESS_FILES:
        path = repository / relative
        if not path.exists():
            continue
        included = True
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}" if included else "unknown"


def collect_architecture_inventory(
    repository: Path,
    *,
    harness_repository: Path | None = None,
) -> dict[str, Any]:
    """Collect the reproducible Phase 0 from_db architecture inventory."""

    repository = repository.resolve()
    harness_repository = (harness_repository or repository).resolve()
    source_root = repository / "daita" / "db"
    source_sections = {
        "daita/db": source_root,
        "daita/db/loop": source_root / "loop",
        "daita/db/runtime": source_root / "runtime",
        "daita/db/memory": source_root / "memory",
        "daita/db/monitor_commands": source_root / "monitor_commands",
        "daita/db/monitor_scheduler": source_root / "monitor_scheduler",
    }
    source_counts = {
        name: _python_tree_counts(path) for name, path in source_sections.items()
    }
    test_roots = (
        repository / "tests" / "unit" / "db",
        repository / "tests" / "integration" / "from_db",
        repository / "tests" / "integration" / "evals",
        repository / "tests" / "performance" / "from_db",
    )
    test_files = sorted({file for root in test_roots for file in root.rglob("*.py")})
    expected = {
        "daita_db_lines": 50271,
        "daita_db_modules": 127,
        "loop_lines": 7343,
        "runtime_lines": 18405,
        "memory_lines": 2922,
        "monitor_lines": 3710,
        "db_test_lines": 43597,
        "db_test_modules": 77,
        "planner_action_members": 27,
    }
    action_members = _enum_members(
        source_root / "planner_protocol.py", "DbPlannerActionKind"
    )
    actual = {
        "daita_db_lines": source_counts["daita/db"]["lines"],
        "daita_db_modules": source_counts["daita/db"]["modules"],
        "loop_lines": source_counts["daita/db/loop"]["lines"],
        "runtime_lines": source_counts["daita/db/runtime"]["lines"],
        "memory_lines": source_counts["daita/db/memory"]["lines"],
        "monitor_lines": source_counts["daita/db/monitor_commands"]["lines"]
        + source_counts["daita/db/monitor_scheduler"]["lines"],
        "db_test_lines": sum(_line_count(file) for file in test_files),
        "db_test_modules": len(test_files),
        "planner_action_members": len(action_members),
    }
    registry = _declarative_db_registry_inventory()
    source_files = sorted(source_root.rglob("*.py"))
    all_db_files = sorted({*source_files, *test_files})
    git = git_metadata(repository)
    return {
        "schema": {
            "name": ARCHITECTURE_INVENTORY_SCHEMA_NAME,
            "version": ARCHITECTURE_INVENTORY_SCHEMA_VERSION,
        },
        "generated_at": _iso_now(),
        "source_git_sha": git["source_git_sha"],
        "measurement_harness_sha": measurement_harness_sha(harness_repository),
        "repository_dirty": git["dirty"],
        "method": {
            "command": (
                "python -m tests.performance.from_db.scale_runner "
                "--repository <source-worktree> --harness-repository <harness-worktree> "
                "--inventory-output <path> --schema-output <path>"
            ),
            "line_count": "len(path.read_text(encoding='utf-8').splitlines())",
            "module_count": "recursive count of *.py files",
            "registry_profile": (
                "ExtensionRegistry(CatalogPlugin(auto_persist=False), "
                "SQLitePlugin(path=':memory:'), "
                "DbRuntimePlanningPlugin(llm_capable=True)) without setup/connect"
            ),
            "imports": "Python ast Import/ImportFrom edges between documented subsystem categories",
            "llm_owners": (
                "Python classes referencing generate_json or "
                "generate_synthesis_json model-service methods"
            ),
        },
        "python_source": source_counts,
        "db_tests": {
            "lines": actual["db_test_lines"],
            "modules": actual["db_test_modules"],
            "roots": [str(path.relative_to(repository)) for path in test_roots],
            "by_root": {
                str(path.relative_to(repository)): _python_tree_counts(path)
                for path in test_roots
            },
        },
        "planner_action_kind": {
            "members": action_members,
            "member_count": len(action_members),
        },
        "registered_runtime": registry,
        "db_specific_model_call_owners": _db_model_call_owners(
            source_files, repository
        ),
        "large_files": {
            "over_800_lines": _large_file_records(all_db_files, repository, 800),
            "over_1000_lines": _large_file_records(all_db_files, repository, 1000),
            "source_only_over_800_lines": _large_file_records(
                source_files, repository, 800
            ),
            "source_only_over_1000_lines": _large_file_records(
                source_files, repository, 1000
            ),
        },
        "subsystem_imports": _subsystem_import_inventory(source_files, repository),
        "expected_reconciliation": {
            key: {
                "expected": expected[key],
                "actual": actual[key],
                "delta": actual[key] - expected[key],
                "matches": actual[key] == expected[key],
            }
            for key in expected
        },
    }


def neutral_artifact_schema() -> dict[str, Any]:
    """Return the versioned neutral operation artifact JSON schema."""

    nullable_number = {"type": ["number", "null"]}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"urn:{NEUTRAL_ARTIFACT_SCHEMA_NAME}:{NEUTRAL_ARTIFACT_SCHEMA_VERSION}",
        "title": "Daita from_db neutral operation benchmark artifact",
        "type": "object",
        "required": [
            "schema",
            "suite",
            "environment",
            "parameters",
            "summary",
            "operations",
        ],
        "properties": {
            "schema": {
                "const": {
                    "name": NEUTRAL_ARTIFACT_SCHEMA_NAME,
                    "version": NEUTRAL_ARTIFACT_SCHEMA_VERSION,
                }
            },
            "suite": {"type": "string"},
            "environment": {"type": "object"},
            "parameters": {"type": "object"},
            "summary": {"type": "object"},
            "operations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "schema_version",
                        "run_id",
                        "scenario",
                        "provenance",
                        "success",
                        "status",
                        "correctness",
                        "sql",
                        "capability_sequence",
                        "task_sequence",
                        "evidence_sequence",
                        "event_count",
                        "model_calls",
                        "model_call_summary",
                        "latency",
                        "context_sizes",
                        "repair_count",
                        "catalog",
                        "errors",
                        "truncation_redaction",
                    ],
                    "properties": {
                        "schema_version": {"const": NEUTRAL_ARTIFACT_SCHEMA_VERSION},
                        "run_id": {"type": "string"},
                        "scenario": {"type": ["string", "null"]},
                        "provenance": {"type": "object"},
                        "success": {"type": "boolean"},
                        "status": {"type": "string"},
                        "correctness": {"type": "object"},
                        "sql": {"type": "object"},
                        "capability_sequence": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "task_sequence": {"type": "array"},
                        "evidence_sequence": {"type": "array"},
                        "event_count": {"type": "integer", "minimum": 0},
                        "model_calls": {"type": "array"},
                        "model_call_summary": {"type": "object"},
                        "latency": {
                            "type": "object",
                            "required": ["model_ms", "operation_ms"],
                            "properties": {
                                "model_ms": nullable_number,
                                "operation_ms": {"type": "number"},
                            },
                        },
                        "context_sizes": {"type": "object"},
                        "repair_count": {"type": "integer", "minimum": 0},
                        "catalog": {"type": "object"},
                        "errors": {"type": "object"},
                        "truncation_redaction": {"type": "object"},
                    },
                },
            },
        },
    }


def apply_eval_report_correctness(
    report: Any,
    *,
    output_root: Path | None = None,
    pytest_node_id: str | None = None,
) -> int:
    """Join existing eval assertion results into optional Phase 0 call artifacts."""

    root = output_root or (
        Path(os.environ["DAITA_PHASE0_OUTPUT_DIR"])
        if os.environ.get("DAITA_PHASE0_OUTPUT_DIR")
        else None
    )
    if root is None:
        return 0
    node_id = (
        pytest_node_id
        or os.environ.get("PYTEST_CURRENT_TEST", "unknown-test").split(" (", 1)[0]
    )
    safe_node_id = re_safe_path(node_id)
    capture_root = root / "neutral-eval-operations" / safe_node_id
    if not capture_root.exists():
        return 0
    by_prompt: dict[str, list[tuple[str, Any]]] = {}
    for case in getattr(report, "cases", ()):
        for run in getattr(case, "runs", ()):
            by_prompt.setdefault(str(run.prompt_hash), []).append((case.case_id, run))
    offsets: dict[str, int] = {}
    updated = 0
    for path in sorted(capture_root.rglob("*.json")):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        operations = artifact.get("operations") or []
        if len(operations) != 1:
            continue
        operation = operations[0]
        prompt_hash = str(_mapping(operation.get("metadata")).get("prompt_hash") or "")
        matches = by_prompt.get(prompt_hash) or []
        offset = offsets.get(prompt_hash, 0)
        if offset >= len(matches):
            continue
        case_id, run = matches[offset]
        offsets[prompt_hash] = offset + 1
        passed = getattr(run, "status", None) == "passed"
        assertions = [
            {
                "code": assertion.code,
                "status": assertion.status,
                "assertion_path": assertion.assertion_path,
            }
            for assertion in getattr(run, "assertions", ())
        ]
        operation["scenario"] = case_id
        operation["run_id"] = str(run.run_id)
        operation["correctness"] = {
            "answer": {
                "passed": passed,
                "evaluation_source": "daita.evals",
                "case_id": case_id,
                "assertions": assertions,
            },
            "sql": {
                "passed": passed,
                "evaluation_source": "daita.evals",
                "case_id": case_id,
                "assertions": assertions,
            },
        }
        artifact["parameters"]["scenario"] = case_id
        _write_json_file(path, artifact)
        updated += 1
    return updated


def re_safe_path(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "unknown-test"


def _python_tree_counts(path: Path) -> dict[str, int]:
    files = sorted(path.rglob("*.py")) if path.exists() else []
    return {"lines": sum(_line_count(file) for file in files), "modules": len(files)}


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _enum_members(path: Path, class_name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [
                target.id
                for child in node.body
                if isinstance(child, (ast.Assign, ast.AnnAssign))
                for target in (
                    child.targets if isinstance(child, ast.Assign) else [child.target]
                )
                if isinstance(target, ast.Name) and not target.id.startswith("_")
            ]
    return []


def _declarative_db_registry_inventory() -> dict[str, Any]:
    from daita.db.runtime.extensions.plugin import DbRuntimePlanningPlugin
    from daita.plugins.catalog import CatalogPlugin
    from daita.plugins.registry import ExtensionRegistry
    from daita.plugins.sqlite import SQLitePlugin

    registry = ExtensionRegistry()
    registry.register_many(
        (
            CatalogPlugin(auto_persist=False),
            SQLitePlugin(path=":memory:"),
            DbRuntimePlanningPlugin(llm_capable=True),
        )
    )
    capabilities = [
        {
            "id": item.id,
            "owner": item.owner,
            "executor": item.executor,
            "model_visible": item.model_visible,
            "runtime_only": item.runtime_only,
        }
        for item in registry.capabilities
    ]
    executors = [
        {
            "id": item.id,
            "capability_ids": sorted(item.capability_ids),
        }
        for item in registry.executors
    ]
    evidence_schemas = [
        {"kind": item.kind, "owner": item.owner} for item in registry.evidence_schemas
    ]
    tool_views = [
        {
            "name": item.name,
            "owner": registry.get_tool_view_owner(item.name),
            "capability_id": item.capability_id,
        }
        for item in registry.tool_views
    ]
    return {
        "profile": "sqlite+catalog+db_runtime_llm_capable",
        "plugin_ids": list(registry.plugin_ids),
        "capability_count": len(capabilities),
        "capabilities": capabilities,
        "db_runtime_capabilities": [
            item for item in capabilities if item["owner"] == "db_runtime"
        ],
        "executor_count": len(executors),
        "executors": executors,
        "db_runtime_executors": [
            item for item in executors if item["id"].startswith(("db.", "db_"))
        ],
        "evidence_schema_count": len(evidence_schemas),
        "evidence_schemas": evidence_schemas,
        "db_runtime_evidence_schemas": [
            item for item in evidence_schemas if item["owner"] == "db_runtime"
        ],
        "model_visible_capabilities": [
            item for item in capabilities if item["model_visible"]
        ],
        "model_visible_tool_view_count": len(tool_views),
        "model_visible_tool_views": tool_views,
    }


def _db_model_call_owners(files: Iterable[Path], repository: Path) -> dict[str, Any]:
    owners = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            calls = sorted(
                {
                    child.attr
                    for child in ast.walk(node)
                    if isinstance(child, ast.Attribute)
                    and child.attr in {"generate_json", "generate_synthesis_json"}
                }
            )
            if calls:
                owners.append(
                    {
                        "class": node.name,
                        "file": str(path.relative_to(repository)),
                        "line": node.lineno,
                        "model_methods": calls,
                    }
                )
    return {"count": len(owners), "owners": owners}


def _large_file_records(
    files: Iterable[Path], repository: Path, threshold: int
) -> list[dict[str, Any]]:
    return [
        {"file": str(path.relative_to(repository)), "lines": lines}
        for path in files
        if (lines := _line_count(path)) > threshold
    ]


def _subsystem_import_inventory(
    files: Iterable[Path], repository: Path
) -> dict[str, Any]:
    edges = []
    counts: dict[str, int] = {}
    for path in files:
        source_module = ".".join(path.relative_to(repository).with_suffix("").parts)
        source_subsystem = _subsystem(source_module)
        if source_subsystem is None:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            targets = _import_targets(node, source_module)
            for target in targets:
                target_subsystem = _subsystem(target)
                if target_subsystem is None or target_subsystem == source_subsystem:
                    continue
                edge = f"{source_subsystem}->{target_subsystem}"
                counts[edge] = counts.get(edge, 0) + 1
                edges.append(
                    {
                        "source_subsystem": source_subsystem,
                        "target_subsystem": target_subsystem,
                        "source_module": source_module,
                        "target_module": target,
                        "file": str(path.relative_to(repository)),
                        "line": node.lineno,
                    }
                )
    return {"edge_counts": dict(sorted(counts.items())), "imports": edges}


def _subsystem(module: str) -> str | None:
    if module.startswith("daita.db.loop"):
        return "loop"
    if module.startswith("daita.db.runtime.analysis") or module == "daita.db.analysis":
        return "analysis"
    if module.startswith("daita.db.memory") or any(
        token in module for token in ("runtime.memory", "extensions.memory")
    ):
        return "memory"
    if module.startswith(
        ("daita.db.monitor_commands", "daita.db.monitor_scheduler")
    ) or any(
        token in module
        for token in ("daita.db.monitors", "runtime.monitor", "extensions.monitor")
    ):
        return "monitor"
    if module.startswith("daita.db.runtime"):
        return "runtime"
    if module in {
        "daita.db.planning",
        "daita.db.planning_context",
        "daita.db.planner_protocol",
        "daita.db.llm_agent_planner",
        "daita.db.llm_planner",
        "daita.db.query_plan",
        "daita.db.plan_validation",
    }:
        return "planning"
    return None


def _import_targets(node: ast.AST, source_module: str) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if not isinstance(node, ast.ImportFrom):
        return []
    if node.level == 0:
        base = node.module or ""
    else:
        package = source_module.rpartition(".")[0].split(".")
        base_parts = package[: len(package) - node.level + 1]
        if node.module:
            base_parts.extend(node.module.split("."))
        base = ".".join(base_parts)
    if node.module:
        return [base]
    return [f"{base}.{alias.name}" for alias in node.names]


def _write_json_file(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--harness-repository", type=Path)
    parser.add_argument("--inventory-output", type=Path)
    parser.add_argument("--schema-output", type=Path)
    args = parser.parse_args()
    if args.inventory_output is None and args.schema_output is None:
        parser.error("at least one output option is required")
    if args.inventory_output is not None:
        _write_json_file(
            args.inventory_output,
            collect_architecture_inventory(
                args.repository,
                harness_repository=args.harness_repository,
            ),
        )
    if args.schema_output is not None:
        _write_json_file(args.schema_output, neutral_artifact_schema())


def _first_sql(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("sql", "selected_sql", "planned_sql", "query", "statement"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        for child in value.values():
            candidate = _first_sql(child)
            if candidate:
                return candidate
    elif isinstance(value, (list, tuple)):
        for child in value:
            candidate = _first_sql(child)
            if candidate:
                return candidate
    return None


def _sql_fingerprint(sql: str | None) -> str | None:
    if not sql:
        return None
    normalized = " ".join(sql.strip().rstrip(";").split()).lower()
    return "sha256:" + hashlib.sha256(normalized.encode()).hexdigest()


def _sql_read_only(sql: str | None) -> bool | None:
    if not sql:
        return None
    normalized = sql.lstrip().lower()
    return normalized.startswith(("select ", "select\n", "with ", "with\n"))


def _record_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
        return dict(converted) if isinstance(converted, dict) else {}
    return {
        key: getattr(value, key)
        for key in (
            "id",
            "kind",
            "type",
            "owner",
            "operation_id",
            "task_id",
            "capability_id",
            "executor_id",
            "status",
            "accepted",
            "payload",
            "input",
            "metadata",
        )
        if hasattr(value, key)
    }


def _record_value(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else getattr(value, key, None)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _optional_int(value: Any) -> int | None:
    number = _optional_number(value)
    return int(number) if number is not None else None


def _optional_number(value: Any) -> int | float | None:
    if value is None or value == "unknown" or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except (TypeError, ValueError):
        return None


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _stable_json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _unique_mappings(values: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for value in values:
        key = _stable_json_hash(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(value))
    return result


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    _main()
