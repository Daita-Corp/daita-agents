"""Non-live artifact contract tests for from_db production-scale benchmarks."""

from __future__ import annotations

import json

import pytest

from .scale_runner import (
    ScaleBenchmarkParameters,
    operation_record,
    run_scale_benchmark,
    validate_artifact_contract,
)

pytestmark = pytest.mark.performance


class _FakeResult:
    operation_id = "op-1"
    status = "succeeded"
    answer = "ok"
    warnings = ()
    evidence = (
        type("Evidence", (), {"kind": "sql.validation", "payload": {}})(),
        type("Evidence", (), {"kind": "query.result", "payload": {}})(),
    )
    diagnostics = {
        "execution": {
            "planned_sql": "SELECT COUNT(*) FROM customers",
            "task_count": 2,
            "tasks": [
                {"capability_id": "db.sql.validate"},
                {"capability_id": "db.sql.execute_read"},
            ],
        },
        "planner": {"status": "finished"},
        "synthesis": {"diagnostics": {"mode": "deterministic"}},
    }


async def test_scale_artifact_contains_required_summary_fields(tmp_path):
    artifact = await _sample_artifact(tmp_path)

    validate_artifact_contract(artifact)
    assert artifact["summary"]["success_rate"] == 1.0
    assert artifact["summary"]["throughput_ops_per_sec"] > 0


async def test_scale_artifact_contains_per_operation_capability_sequences(tmp_path):
    artifact = await _sample_artifact(tmp_path)

    operation = artifact["operations"][0]
    assert operation["capability_sequence"] == [
        "db.sql.validate",
        "db.sql.execute_read",
    ]
    assert artifact["summary"]["capability_sequences"]


async def test_scale_artifact_contains_latency_percentiles(tmp_path):
    artifact = await _sample_artifact(tmp_path)

    summary = artifact["summary"]
    assert {"latency_ms_p50", "latency_ms_p95", "latency_ms_p99"} <= set(summary)
    assert summary["latency_ms_p99"] >= summary["latency_ms_p50"]


async def test_scale_artifact_contains_error_classification(tmp_path):
    async def failing_operation(_index: int):
        raise ValueError("bad benchmark input")

    artifact = await run_scale_benchmark(
        suite="contract-error",
        parameters=ScaleBenchmarkParameters(concurrency=1, operations=1),
        operation_factory=failing_operation,
        output_dir=tmp_path,
        environment={"dataset": "contract"},
    )

    validate_artifact_contract(artifact)
    assert artifact["summary"]["error_classes"] == {"ValueError": 1}
    assert artifact["operations"][0]["error"]["type"] == "ValueError"


async def test_scale_artifact_contains_llm_and_db_metadata_when_available(tmp_path):
    artifact = await _sample_artifact(tmp_path)
    artifact_path = tmp_path / "contract.json"
    artifact_path.write_text(json.dumps(artifact), "utf-8")
    reloaded = json.loads(artifact_path.read_text("utf-8"))

    assert reloaded["environment"]["postgres_version"] == "PostgreSQL contract"
    assert reloaded["environment"]["model"] == "none"
    assert reloaded["operations"][0]["llm"] == {
        "calls": 0,
        "tokens": {},
        "estimated_cost_usd": None,
    }


def test_operation_record_accepts_explicit_errors_without_live_dependencies():
    record = operation_record(
        index=0,
        latency_ms=12.5,
        started_at="2026-01-01T00:00:00+00:00",
        error=RuntimeError("connection refused"),
    )

    assert record["success"] is False
    assert record["error"] == {
        "type": "RuntimeError",
        "message": "connection refused",
    }


async def _sample_artifact(tmp_path):
    async def operation(_index: int):
        return _FakeResult()

    return await run_scale_benchmark(
        suite="contract",
        parameters=ScaleBenchmarkParameters(
            concurrency=1,
            operations=3,
            scenario="observability",
        ),
        operation_factory=operation,
        output_dir=tmp_path,
        environment={
            "postgres_version": "PostgreSQL contract",
            "model": "none",
            "dataset": "contract",
        },
    )
