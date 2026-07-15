"""Non-live artifact contract tests for from_db production-scale benchmarks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .scale_runner import (
    ScaleBenchmarkParameters,
    aggregate_model_calls,
    neutral_artifact_schema,
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


def test_operation_record_measures_slim_repair_and_model_observation_size():
    snapshot = type(
        "Snapshot",
        (),
        {
            "tasks": (
                {
                    "id": "validate-1",
                    "capability_id": "db.sql.validate",
                    "metadata": {"query_attempt": 1},
                },
                {
                    "id": "validate-2",
                    "capability_id": "db.sql.validate",
                    "metadata": {"query_attempt": 2},
                },
            ),
            "evidence": (),
            "events": (
                {
                    "id": "llm-2",
                    "type": "llm.completed",
                    "payload": {
                        "turn": 2,
                        "slim_model_turn": {
                            "call_id": "op-1:sqlite-slim:2",
                            "mode": "llm",
                            "purpose": "operation",
                            "provider": "openai",
                            "model": "contract-model",
                            "model_parameters": {"temperature": 0},
                            "prompt_chars": 500,
                            "observation_chars": 321,
                            "latency_ms": 10,
                            "tokens": {
                                "prompt_tokens": 10,
                                "completion_tokens": 5,
                                "total_tokens": 15,
                            },
                        },
                    },
                },
            ),
        },
    )()

    record = operation_record(
        index=0,
        latency_ms=20,
        started_at="2026-01-01T00:00:00+00:00",
        result=_FakeResult(),
        snapshot=snapshot,
    )

    assert record["repair_count"] == 1
    assert record["context_sizes"]["model_visible_observation_chars"] == 321
    assert record["model_call_summary"]["call_count"] == 1
    assert record["model_calls"][0]["stage"] == "operation_selection"
    assert record["model_calls"][0]["model_parameters"] == {"temperature": 0}


def test_model_call_aggregation_counts_all_stages_and_deduplicates_sources():
    tasks = [
        {
            "id": "task-plan",
            "capability_id": "db.query.plan",
            "metadata": {"planner_turn": 1},
        },
        {
            "id": "task-synthesis",
            "capability_id": "db.answer.synthesize",
            "metadata": {"planner_turn": 2},
        },
    ]
    outer = _llm_diagnostics(
        input_tokens=100,
        output_tokens=20,
        cached_input_tokens=10,
        reasoning_tokens=2,
        latency_ms=10,
        cost=0.001,
    )
    inner = _llm_diagnostics(
        input_tokens=200,
        output_tokens=30,
        cached_input_tokens=15,
        reasoning_tokens=3,
        latency_ms=20,
        cost=0.002,
    )
    synthesis = _llm_diagnostics(
        input_tokens=300,
        output_tokens=40,
        cached_input_tokens=20,
        reasoning_tokens=4,
        latency_ms=30,
        cost=0.003,
    )
    evidence = [
        {
            "id": "outer-1",
            "kind": "planner.decision",
            "payload": {"turn": 1, "decision": {"metadata": {"llm": outer}}},
        },
        {
            "id": "inner-1",
            "kind": "query.plan.proposal",
            "task_id": "task-plan",
            "payload": {"planner_diagnostics": inner},
        },
        {
            "id": "synthesis-1",
            "kind": "answer.synthesis",
            "task_id": "task-synthesis",
            "payload": {"diagnostics": synthesis},
        },
    ]
    events = [
        {
            "id": "event-synthesis-1",
            "type": "diagnostic",
            "task_id": "task-synthesis",
            "payload": {"diagnostics": synthesis},
        }
    ]

    aggregate = aggregate_model_calls(
        tasks=tasks,
        evidence=evidence,
        events=events,
    )

    assert [call["stage"] for call in aggregate["calls"]] == [
        "outer_planner",
        "sql_planner",
        "final_synthesis",
    ]
    assert aggregate["summary"] == {
        "call_count": 3,
        "input_tokens": 600,
        "output_tokens": 90,
        "cached_input_tokens": 45,
        "reasoning_tokens": 9,
        "total_tokens": 690,
        "estimated_cost_usd": 0.006,
        "model_latency_ms": 60,
        "unattributed_usage": {},
        "discrepancies": [],
        "all_calls_have_trace_identity": False,
    }
    final_call = aggregate["calls"][-1]
    assert {source["type"] for source in final_call["sources"]} == {
        "evidence",
        "event",
    }


def test_model_call_aggregation_preserves_unknown_fields_and_deterministic_stages():
    evidence = [
        {
            "id": "outer-1",
            "kind": "planner.decision",
            "payload": {
                "turn": 1,
                "decision": {
                    "metadata": {
                        "llm": {
                            "mode": "llm",
                            "provider": "openai",
                            "model": "contract-model",
                        }
                    }
                },
            },
        },
        {
            "id": "deterministic-synthesis",
            "kind": "answer.synthesis",
            "payload": {
                "diagnostics": {
                    "mode": "deterministic_fallback",
                    "provider": "daita.db",
                    "model": "deterministic",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                }
            },
        },
    ]

    aggregate = aggregate_model_calls(evidence=evidence)

    assert aggregate["summary"]["call_count"] == 1
    assert aggregate["calls"][0]["stage"] == "outer_planner"
    for field in (
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "reasoning_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "model_latency_ms",
    ):
        assert aggregate["summary"][field] is None


def test_operation_record_exposes_synthesis_only_public_telemetry_discrepancy():
    result = type(
        "Result",
        (),
        {
            "operation_id": "op-telemetry",
            "status": "succeeded",
            "answer": "ok",
            "warnings": (),
            "diagnostics": {"execution": {"task_count": 0, "tasks": []}},
            "telemetry": {
                "llm_calls": 1,
                "input_tokens": 300,
                "output_tokens": 40,
                "total_tokens": 340,
                "estimated_cost_usd": 0.003,
            },
        },
    )()
    snapshot = type(
        "Snapshot",
        (),
        {
            "tasks": (),
            "events": (),
            "evidence": (
                type(
                    "Evidence",
                    (),
                    {
                        "id": "outer-1",
                        "kind": "planner.decision",
                        "task_id": None,
                        "payload": {
                            "turn": 1,
                            "decision": {
                                "metadata": {
                                    "llm": _llm_diagnostics(
                                        input_tokens=100,
                                        output_tokens=20,
                                        latency_ms=10,
                                        cost=0.001,
                                    )
                                }
                            },
                        },
                    },
                )(),
                type(
                    "Evidence",
                    (),
                    {
                        "id": "synthesis-1",
                        "kind": "answer.synthesis",
                        "task_id": None,
                        "payload": {
                            "diagnostics": _llm_diagnostics(
                                input_tokens=300,
                                output_tokens=40,
                                latency_ms=30,
                                cost=0.003,
                            )
                        },
                    },
                )(),
            ),
        },
    )()

    record = operation_record(
        index=0,
        latency_ms=75,
        started_at="2026-01-01T00:00:00+00:00",
        result=result,
        snapshot=snapshot,
    )

    assert record["model_call_summary"]["call_count"] == 2
    assert {item["field"] for item in record["telemetry_discrepancies"]} == {
        "llm_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "estimated_cost_usd",
    }
    assert record["latency"] == {"model_ms": 40, "operation_ms": 75}


def test_phase0_inventory_matches_documented_starting_counts():
    inventory_path = (
        Path.cwd()
        / ".daita/slim-experiment/phase0"
        / "b87df31873d33fffbf50498f5dc4d8892115e8f8"
        / "architecture_inventory.json"
    )
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))

    for key in (
        "daita_db_lines",
        "daita_db_modules",
        "loop_lines",
        "runtime_lines",
        "memory_lines",
        "monitor_lines",
        "planner_action_members",
    ):
        assert inventory["expected_reconciliation"][key]["matches"]
    assert inventory["expected_reconciliation"]["db_test_lines"]["matches"]
    assert inventory["db_specific_model_call_owners"]["count"] == 7
    assert inventory["registered_runtime"]["model_visible_tool_view_count"] == 5
    assert inventory["planner_action_kind"]["member_count"] == 27


def test_neutral_artifact_schema_is_versioned_and_requires_comparison_fields():
    schema = neutral_artifact_schema()
    operation = schema["properties"]["operations"]["items"]

    assert schema["$id"].endswith(":1.0.0")
    assert {
        "provenance",
        "correctness",
        "model_calls",
        "model_call_summary",
        "latency",
        "context_sizes",
        "truncation_redaction",
    } <= set(operation["required"])


async def test_from_db_eval_target_merges_public_answer_with_persisted_snapshot(
    monkeypatch,
):
    from dataclasses import dataclass, field

    from tests.integration.evals.eval_from_db_factories import FromDbEvalTarget

    monkeypatch.delenv("DAITA_PHASE0_OUTPUT_DIR", raising=False)

    class PersistedSnapshot:
        def to_dict(self):
            return {
                "operation": {"id": "db-op-contract", "status": "succeeded"},
                "tasks": [{"id": "task-1", "capability_id": "db.schema.inspect"}],
                "evidence": [{"id": "evidence-1", "kind": "query.result"}],
            }

    persisted_snapshot = PersistedSnapshot()

    class FakeRuntime:
        async def inspect_operation(self, operation_id):
            assert operation_id == "db-op-contract"
            return persisted_snapshot

    class FakeAgent:
        runtime = FakeRuntime()
        llm = None

        async def run_detailed(self, prompt, **kwargs):
            assert prompt == "count rows"
            assert kwargs == {"max_iterations": 8}
            return FakeResult()

    @dataclass
    class FakeResult:
        operation_id: str = "db-op-contract"
        answer: str = "The count is 4."
        diagnostics: dict = field(
            default_factory=lambda: {"execution": {"task_refs": []}}
        )
        evidence: list = field(default_factory=list)

    target = FromDbEvalTarget(FakeAgent(), name="contract")

    payload = await target.run_detailed("count rows", max_iterations=8)

    assert payload["answer"] == "The count is 4."
    assert payload["tasks"][0]["capability_id"] == "db.schema.inspect"
    assert payload["evidence"][0]["kind"] == "query.result"
    assert payload["diagnostics"]["execution"]["task_count"] == 1


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


def _llm_diagnostics(
    *,
    input_tokens,
    output_tokens,
    latency_ms,
    cost,
    cached_input_tokens=None,
    reasoning_tokens=None,
):
    tokens = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if cached_input_tokens is not None:
        tokens["cached_input_tokens"] = cached_input_tokens
    if reasoning_tokens is not None:
        tokens["reasoning_tokens"] = reasoning_tokens
    return {
        "mode": "llm",
        "provider": "openai",
        "model": "contract-model",
        "tokens": tokens,
        "latency_ms": latency_ms,
        "estimated_cost_usd": cost,
    }
