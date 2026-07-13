"""Bucket 3 live-gate artifact capture and terminal reporting."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from daita.db.runtime import DbRuntime
from tests.integration.from_db.live_production_helpers import (
    _redact_artifact_payload,
    classify_live_gate_failure,
    write_failure_artifacts,
)

_BUCKET3_FILES = frozenset(
    {
        "test_from_db_memory_live.py",
        "test_from_db_live_memory_contracts.py",
        "test_from_db_live_production_contracts.py",
    }
)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    setattr(item, f"rep_{report.when}", report)


@pytest.fixture(autouse=True)
def _capture_bucket3_failure_artifacts(request, monkeypatch, tmp_path_factory):
    if Path(str(request.fspath)).name not in _BUCKET3_FILES:
        yield
        return

    captures: dict[str, tuple[Any, Any | None]] = {}
    capability_captures: list[dict[str, Any]] = []
    original_run = DbRuntime.run
    original_execute_capability = DbRuntime.execute_capability

    async def capturing_run(runtime, *args, **kwargs):
        result = await original_run(runtime, *args, **kwargs)
        snapshot = None
        try:
            snapshot = await runtime.inspect_operation(result.operation_id)
        except Exception:  # noqa: BLE001 - artifact capture cannot alter the gate
            pass
        captures[result.operation_id] = (result, snapshot)
        return result

    async def capturing_execute_capability(runtime, capability_id, *args, **kwargs):
        evidence = await original_execute_capability(
            runtime,
            capability_id,
            *args,
            **kwargs,
        )
        capability_captures.append(
            {
                "capability_id": capability_id,
                "owner": kwargs.get("owner"),
                "operation_type": kwargs.get("operation_type"),
                "evidence": [item.to_dict() for item in evidence],
            }
        )
        return evidence

    monkeypatch.setattr(DbRuntime, "run", capturing_run)
    monkeypatch.setattr(
        DbRuntime,
        "execute_capability",
        capturing_execute_capability,
    )
    yield

    report = getattr(request.node, "rep_call", None)
    if report is None or not report.failed:
        return

    artifact_root = _artifact_root(request.config, tmp_path_factory)
    test_dir = artifact_root / _safe_test_id(request.node.nodeid)
    test_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for result, snapshot in captures.values():
        artifact_dir = write_failure_artifacts(
            test_dir,
            result=result,
            snapshot=snapshot,
        )
        written.append(str(artifact_dir))

    failure_text = str(report.longrepr)
    metadata = {
        "nodeid": request.node.nodeid,
        "classification": classify_live_gate_failure(failure_text),
        "failure": failure_text,
        "operation_artifacts": written,
        "capability_calls": capability_captures,
    }
    (test_dir / "failure.json").write_text(
        json.dumps(
            _redact_artifact_payload(metadata),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[Bucket 3 artifacts] {test_dir}")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not os.environ.get("DAITA_BUCKET3_ARTIFACT_ROOT"):
        return
    artifact_root = Path(os.environ["DAITA_BUCKET3_ARTIFACT_ROOT"]).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    stats = {
        outcome: len(terminalreporter.stats.get(outcome, ()))
        for outcome in ("passed", "failed", "skipped", "error")
    }
    classifications: dict[str, int] = {}
    for report in terminalreporter.stats.get("failed", ()):
        classification = classify_live_gate_failure(str(report.longrepr))
        classifications[classification] = classifications.get(classification, 0) + 1
    junit = getattr(config.option, "xmlpath", None)
    summary = {
        "exit_status": exitstatus,
        "counts": stats,
        "failure_classifications": classifications,
        "junit": str(Path(junit).resolve()) if junit else None,
        "artifact_root": str(artifact_root),
        "postgresql_status": os.environ.get(
            "DAITA_BUCKET3_POSTGRES_STATUS",
            "not_run_separate_required_gate",
        ),
    }
    (artifact_root / "pytest-session-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    terminalreporter.write_sep("=", "Bucket 3 release evidence")
    terminalreporter.write_line(f"JUnit: {summary['junit'] or 'not configured'}")
    terminalreporter.write_line(f"Artifacts: {artifact_root}")
    terminalreporter.write_line(f"PostgreSQL: {summary['postgresql_status']}")


def _artifact_root(config, tmp_path_factory) -> Path:
    configured = os.environ.get("DAITA_BUCKET3_ARTIFACT_ROOT")
    if configured:
        return Path(configured).resolve()
    return tmp_path_factory.getbasetemp() / "from_db_failure_artifacts"


def _safe_test_id(nodeid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", nodeid).strip("_")
