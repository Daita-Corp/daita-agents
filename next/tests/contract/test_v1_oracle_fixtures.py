from __future__ import annotations

import json
from pathlib import Path

NEXT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = NEXT_ROOT / "tests" / "fixtures" / "v1"
BASELINE_COMMIT = "b87df31873d33fffbf50498f5dc4d8892115e8f8"
PLAN_SHA256 = "403ad8c3030a126375759b57af4ebe767c6066352b2db158488669a28cc3f935"
FIXTURE_NAMES = {
    "loop_trajectories.json",
    "public_surface.json",
    "runtime_serialization.json",
    "sql_validation.json",
}


def _load(name: str) -> dict:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def test_oracle_fixture_set_is_canonical_and_baseline_anchored() -> None:
    fixture_paths = {path.name for path in FIXTURE_ROOT.glob("*.json")}
    assert fixture_paths == FIXTURE_NAMES

    for name in FIXTURE_NAMES:
        path = FIXTURE_ROOT / name
        payload = _load(name)
        assert path.read_text(encoding="utf-8") == (
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        assert payload["_meta"]["schema_version"] == 1
        assert payload["_meta"]["v1_baseline_commit"] == BASELINE_COMMIT
        assert payload["_meta"]["architecture_plan_sha256"] == PLAN_SHA256
        assert payload["_meta"]["source_refs"]


def test_public_surface_snapshot_is_complete() -> None:
    payload = _load("public_surface.json")

    assert payload["distribution"] == {
        "module_version": "1.0.0",
        "name": "daita-agents",
        "requires_python": ">=3.11",
        "version": "1.0.0",
    }
    assert len(payload["exports"]["daita"]) == 61
    assert len(payload["exports"]["daita.db"]) == 23
    assert len(payload["exports"]["daita.llm"]) == 13
    assert len(payload["optional_extras"]) == 44
    assert payload["providers"] == [
        "openai",
        "anthropic",
        "grok",
        "gemini",
        "ollama",
        "mock",
    ]
    from_db_names = [item["name"] for item in payload["signatures"]["daita.db.from_db"]]
    assert from_db_names == [
        "source",
        "name",
        "mode",
        "config",
        "source_options",
        "llm",
        "runtime",
        "memory",
        "catalog",
        "lineage",
        "quality",
        "history",
        "stateful",
        "plugins",
        "skills",
    ]


def test_runtime_snapshot_preserves_task_evidence_event_correlations() -> None:
    payload = _load("runtime_serialization.json")
    records = payload["records"]
    task = records["task_pending"]
    evidence = records["evidence"]
    events = records["events"]

    assert records["operation_running"]["status"] == "running"
    assert records["operation_succeeded"]["status"] == "succeeded"
    assert task["status"] == "pending"
    assert records["task_succeeded"]["status"] == "succeeded"
    assert evidence["accepted"] is True
    assert evidence["operation_id"] == task["operation_id"]
    assert evidence["task_id"] == task["id"]
    assert [event["type"] for event in events] == [
        "task.created",
        "executor.started",
        "evidence.accepted",
        "executor.completed",
    ]
    assert events[0]["task_id"] == task["id"]
    assert events[2]["evidence_id"] == evidence["id"]


def test_sql_snapshot_captures_valid_repair_and_write_facts() -> None:
    payload = _load("sql_validation.json")
    cases = {case["id"]: case for case in payload["cases"]}

    assert set(cases) == {
        "valid_bounded_read",
        "unknown_table",
        "missing_column",
        "write_classification",
        "empty_sql",
    }
    assert cases["valid_bounded_read"]["result"]["ok"] is True
    assert cases["valid_bounded_read"]["result"]["statement_facts"]["is_read"] is True
    assert cases["unknown_table"]["result"]["unknown_tables"] == ["customerz"]
    assert cases["unknown_table"]["result"]["do_not_retry_same_sql"] is True
    assert cases["missing_column"]["result"]["missing_columns"][0]["column"] == "state"
    write_facts = cases["write_classification"]["result"]["statement_facts"]
    assert write_facts["is_read"] is False
    assert write_facts["mutating_statement_classes"] == ["UPDATE"]
    assert cases["empty_sql"]["result"]["repair_required"] is True


def test_loop_trajectory_snapshot_encodes_required_progression() -> None:
    payload = _load("loop_trajectories.json")
    trajectories = {item["id"]: item for item in payload["trajectories"]}

    assert trajectories["text_only_completion"]["task_count"] == 0
    tool_steps = trajectories["tool_evidence_completion"]["steps"]
    assert tool_steps.index("task.persisted") < tool_steps.index("executor.invoked")
    assert tool_steps.index("executor.invoked") < tool_steps.index("evidence.accepted")
    assert tool_steps.index("evidence.accepted") < tool_steps.index(
        "observation.persisted"
    )
    assert trajectories["identical_failure_no_progress"]["terminal_reason"] == (
        "no_progress"
    )
    approval_steps = trajectories["approval_resume"]["steps"]
    assert "same_operation.resumed" in approval_steps
    assert "completed_tasks.skipped" in approval_steps
    assert "executor.invoked_once" in approval_steps
