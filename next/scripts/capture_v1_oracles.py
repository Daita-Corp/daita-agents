#!/usr/bin/env python3
"""Capture neutral v1 oracle fixtures in a v1-only interpreter.

This script is intentionally not imported or executed by the v2 test suite.
Run it from the repository root using the v1 development environment, then let
v2 tests consume only the serialized JSON files.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import tomllib
from typing import Any

NEXT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = NEXT_ROOT.parent
V1_PACKAGE_ROOT = (REPOSITORY_ROOT / "daita").resolve()
FIXTURE_ROOT = NEXT_ROOT / "tests" / "fixtures" / "v1"
BASELINE_COMMIT = "b87df31873d33fffbf50498f5dc4d8892115e8f8"
PLAN_SHA256 = "403ad8c3030a126375759b57af4ebe767c6066352b2db158488669a28cc3f935"


def _assert_v1_oracle_environment() -> Any:
    import daita

    package_path = Path(daita.__file__).resolve()
    try:
        package_path.relative_to(V1_PACKAGE_ROOT)
    except ValueError as exc:
        raise RuntimeError(
            f"capture must import root v1 from {V1_PACKAGE_ROOT}, got {package_path}"
        ) from exc

    unchanged = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            BASELINE_COMMIT,
            "--",
            "daita",
            "tests",
            "pyproject.toml",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    if unchanged.returncode != 0:
        raise RuntimeError("root v1 oracle differs from the recorded baseline")
    return daita


def _metadata(*source_refs: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "v1_baseline_commit": BASELINE_COMMIT,
        "architecture_plan_sha256": PLAN_SHA256,
        "capture_script": "next/scripts/capture_v1_oracles.py",
        "source_refs": list(source_refs),
    }


def _literal_all(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            return list(ast.literal_eval(node.value))
    raise RuntimeError(f"No literal __all__ in {path}")


def _signature(callable_object: Any) -> list[dict[str, Any]]:
    parameters: list[dict[str, Any]] = []
    for parameter in inspect.signature(callable_object).parameters.values():
        required = parameter.default is inspect.Parameter.empty
        parameters.append(
            {
                "name": parameter.name,
                "kind": parameter.kind.name,
                "required": required,
                "default_repr": None if required else repr(parameter.default),
            }
        )
    return parameters


def capture_public_surface(daita: Any) -> dict[str, Any]:
    import daita.db as db
    import daita.llm as llm
    from daita.agents.agent import Agent
    from daita.llm.factory import list_available_providers

    configuration = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    return {
        "_meta": _metadata(
            "daita/__init__.py",
            "daita/db/__init__.py",
            "daita/llm/__init__.py",
            "tests/unit/db/test_public_api.py",
        ),
        "distribution": {
            "name": configuration["project"]["name"],
            "version": configuration["project"]["version"],
            "module_version": daita.__version__,
            "requires_python": configuration["project"]["requires-python"],
        },
        "exports": {
            "daita": _literal_all(REPOSITORY_ROOT / "daita" / "__init__.py"),
            "daita.db": _literal_all(REPOSITORY_ROOT / "daita" / "db" / "__init__.py"),
            "daita.llm": _literal_all(
                REPOSITORY_ROOT / "daita" / "llm" / "__init__.py"
            ),
        },
        "signatures": {
            "Agent.__init__": _signature(Agent.__init__),
            "Agent.from_db": _signature(Agent.from_db),
            "Agent.run": _signature(Agent.run),
            "daita.db.from_db": _signature(db.from_db),
        },
        "providers": list_available_providers(),
        "optional_extras": sorted(configuration["project"]["optional-dependencies"]),
        "llm_export_count": len(llm.__all__),
    }


def capture_runtime_serialization() -> dict[str, Any]:
    from daita.runtime.primitives import (
        AccessMode,
        ApprovalStatus,
        Capability,
        Evidence,
        Operation,
        OperationStatus,
        PolicyDecision,
        PolicyEffect,
        RiskLevel,
        RuntimeEvent,
        RuntimeEventType,
        Task,
        TaskStatus,
    )

    capability = Capability(
        id="oracle.lookup",
        owner="oracle",
        description="Read one bounded oracle value.",
        domains=frozenset({"data"}),
        operation_types=frozenset({"data.query"}),
        access=AccessMode.READ,
        risk=RiskLevel.LOW,
        input_schema={"type": "object", "required": ["key"]},
        output_evidence=frozenset({"oracle.lookup.result"}),
        executor="oracle.lookup",
        model_visible=True,
        retry_safe=True,
        replay_safe=True,
        idempotent=True,
        side_effecting=False,
    )
    operation = Operation(
        id="op-oracle-1",
        operation_type="data.query",
        status=OperationStatus.RUNNING,
        request={"prompt": "Look up alpha."},
        required_evidence=frozenset({"oracle.lookup.result"}),
    )
    task = Task(
        id="task-oracle-1",
        operation_id=operation.id,
        capability_id=capability.id,
        executor_id=capability.executor,
        input={"key": "alpha"},
        status=TaskStatus.PENDING,
        required_evidence=frozenset({"oracle.lookup.result"}),
        metadata={"owner": capability.owner, "attempt": 1},
    )
    evidence = Evidence(
        id="evidence-oracle-1",
        kind="oracle.lookup.result",
        owner=capability.owner,
        operation_id=operation.id,
        task_id=task.id,
        schema_version="1",
        accepted=True,
        payload={"key": "alpha", "value": 7},
    )
    policy = PolicyDecision(
        policy_id="oracle.read",
        owner="oracle",
        effect=PolicyEffect.ALLOW,
        reason="Bounded attached-source read.",
        severity=RiskLevel.LOW,
        operation_id=operation.id,
    )
    events = (
        RuntimeEvent(
            id="event-1",
            type=RuntimeEventType.TASK_CREATED,
            operation_id=operation.id,
            task_id=task.id,
            capability_id=task.capability_id,
            executor_id=task.executor_id,
            message="Task persisted.",
            timestamp=1.0,
        ),
        RuntimeEvent(
            id="event-2",
            type=RuntimeEventType.EXECUTOR_STARTED,
            operation_id=operation.id,
            task_id=task.id,
            capability_id=task.capability_id,
            executor_id=task.executor_id,
            message="Executor started.",
            timestamp=2.0,
        ),
        RuntimeEvent(
            id="event-3",
            type=RuntimeEventType.EVIDENCE_ACCEPTED,
            operation_id=operation.id,
            task_id=task.id,
            capability_id=task.capability_id,
            executor_id=task.executor_id,
            evidence_id=evidence.id,
            message="Evidence accepted.",
            timestamp=3.0,
        ),
        RuntimeEvent(
            id="event-4",
            type=RuntimeEventType.EXECUTOR_COMPLETED,
            operation_id=operation.id,
            task_id=task.id,
            capability_id=task.capability_id,
            executor_id=task.executor_id,
            evidence_id=evidence.id,
            message="Executor completed.",
            timestamp=4.0,
        ),
    )
    return {
        "_meta": _metadata(
            "daita/runtime/primitives.py",
            "tests/unit/runtime/test_primitives.py",
            "tests/unit/runtime/test_kernel.py",
        ),
        "enum_values": {
            "access_mode": [value.value for value in AccessMode],
            "risk_level": [value.value for value in RiskLevel],
            "operation_status": [value.value for value in OperationStatus],
            "task_status": [value.value for value in TaskStatus],
            "approval_status": [value.value for value in ApprovalStatus],
            "policy_effect": [value.value for value in PolicyEffect],
            "runtime_event_type": [value.value for value in RuntimeEventType],
        },
        "records": {
            "capability": capability.to_dict(),
            "operation_running": operation.to_dict(),
            "operation_succeeded": replace(
                operation, status=OperationStatus.SUCCEEDED
            ).to_dict(),
            "task_pending": task.to_dict(),
            "task_succeeded": replace(task, status=TaskStatus.SUCCEEDED).to_dict(),
            "evidence": evidence.to_dict(),
            "policy": policy.to_dict(),
            "events": [event.to_dict() for event in events],
        },
        "required_order": [
            "task.persisted",
            "executor.started",
            "evidence.accepted",
            "task.terminal",
            "observation.persisted",
        ],
    }


def capture_sql_validation() -> dict[str, Any]:
    from daita.db.query_sql_validation import validate_sql_against_schema

    schema = {
        "database_type": "sqlite",
        "tables": [
            {
                "name": "customers",
                "columns": [
                    {"name": "id", "type": "integer", "is_primary_key": True},
                    {"name": "status", "type": "text"},
                    {"name": "total", "type": "real"},
                ],
            }
        ],
    }
    inputs = (
        (
            "valid_bounded_read",
            "SELECT id, status FROM customers WHERE status = 'complete' LIMIT 10",
        ),
        ("unknown_table", "SELECT id FROM customerz LIMIT 10"),
        ("missing_column", "SELECT state FROM customers LIMIT 10"),
        (
            "write_classification",
            "UPDATE customers SET status = 'complete' WHERE id = 1",
        ),
        ("empty_sql", ""),
    )
    cases = []
    for case_id, sql in inputs:
        cases.append(
            {
                "id": case_id,
                "sql": sql,
                "result": validate_sql_against_schema(sql, schema, dialect="sqlite"),
            }
        )
    return {
        "_meta": _metadata(
            "daita/db/sql_analysis.py",
            "daita/db/query_sql_validation.py",
            "tests/unit/db/test_plan_validation.py",
        ),
        "schema": schema,
        "cases": cases,
    }


def capture_loop_trajectories() -> dict[str, Any]:
    return {
        "_meta": _metadata(
            "daita/agents/chat/runtime.py",
            "tests/unit/agents/test_chat_runtime.py",
            "tests/unit/db/test_agent_loop_completion_targets.py",
            "tests/unit/db/test_agent_loop_phase2.py",
        ),
        "trajectories": [
            {
                "id": "text_only_completion",
                "steps": [
                    "operation.persisted",
                    "model.response.persisted",
                    "readiness.allowed",
                    "operation.succeeded",
                ],
                "task_count": 0,
                "terminal_reason": "completed",
            },
            {
                "id": "tool_evidence_completion",
                "steps": [
                    "model.tool_call.persisted",
                    "task.persisted",
                    "executor.invoked",
                    "evidence.accepted",
                    "observation.persisted",
                    "model.final_answer.persisted",
                    "readiness.allowed",
                    "operation.succeeded",
                ],
                "task_count": 1,
                "terminal_reason": "completed",
            },
            {
                "id": "validation_repair",
                "steps": [
                    "model.invalid_action.persisted",
                    "validation.failure_observed",
                    "model.repaired_action.persisted",
                    "task.persisted",
                    "evidence.accepted",
                    "readiness.allowed",
                    "operation.succeeded",
                ],
                "task_count": 1,
                "terminal_reason": "completed_after_repair",
            },
            {
                "id": "identical_failure_no_progress",
                "steps": [
                    "validation.failure_observed",
                    "identical_action.repeated",
                    "no_progress_budget.exhausted",
                    "operation.failed",
                ],
                "task_count": 0,
                "terminal_reason": "no_progress",
            },
            {
                "id": "approval_resume",
                "steps": [
                    "task.persisted",
                    "approval.persisted",
                    "operation.waiting_for_approval",
                    "approval.approved",
                    "same_operation.resumed",
                    "completed_tasks.skipped",
                    "executor.invoked_once",
                    "evidence.accepted",
                ],
                "task_count": 1,
                "terminal_reason": "resumed_once",
            },
        ],
    }


def capture_all(daita: Any) -> dict[str, dict[str, Any]]:
    return {
        "public_surface.json": capture_public_surface(daita),
        "runtime_serialization.json": capture_runtime_serialization(),
        "sql_validation.json": capture_sql_validation(),
        "loop_trajectories.json": capture_loop_trajectories(),
    }


def _render(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    arguments = parser.parse_args()

    daita = _assert_v1_oracle_environment()
    fixtures = capture_all(daita)
    if arguments.write:
        FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
        for name, payload in fixtures.items():
            (FIXTURE_ROOT / name).write_text(_render(payload), encoding="utf-8")
        return 0

    stale = [
        name
        for name, payload in fixtures.items()
        if not (FIXTURE_ROOT / name).exists()
        or (FIXTURE_ROOT / name).read_text(encoding="utf-8") != _render(payload)
    ]
    if stale:
        raise SystemExit(f"stale v1 oracle fixture(s): {', '.join(stale)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
