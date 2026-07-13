"""Run the repeatable Bucket 3 live release gate with durable evidence."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tests.integration.from_db.live_production_helpers import (
    _redact_artifact_payload,
    run_live_gate_preflight,
)

BUCKET3_FILES = (
    "tests/integration/from_db/test_from_db_memory_live.py",
    "tests/integration/from_db/test_from_db_live_memory_contracts.py",
    "tests/integration/from_db/test_from_db_live_production_contracts.py",
)
POSTGRES_TEST = (
    "tests/integration/from_db/test_from_db_live_production_contracts.py::"
    "test_live_postgres_simple_query_full_loop_contract"
)


def main() -> int:
    args = _parse_args()
    _require_project_virtualenv()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    gate_root = args.artifact_root.resolve() / run_id
    gate_root.mkdir(parents=True, exist_ok=False)

    preflight = asyncio.run(run_live_gate_preflight())
    _write_json(
        gate_root / "preflight.json",
        {"status": preflight.status, "detail": preflight.detail},
    )
    if not preflight.ready:
        print(f"Preflight: {preflight.status}")
        print(f"Artifacts: {gate_root}")
        print("PostgreSQL: not run (separate required gate)")
        return 2

    run_records = [
        _run_pytest_gate(gate_root, index=index) for index in range(1, args.runs + 1)
    ]
    postgres_record = _run_postgres_gate(gate_root) if args.include_postgresql else None
    consecutive = _consecutive_clean_runs(run_records)
    summary = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "model": os.environ.get("OPENAI_TEST_MODEL", "gpt-5.4-mini"),
        "repository_state": _repository_state(),
        "preflight": {"status": preflight.status, "detail": preflight.detail},
        "non_postgresql_runs": run_records,
        "consecutive_clean_runs": consecutive,
        "three_consecutive_clean_runs": consecutive >= 3,
        "postgresql": postgres_record
        or {"status": "not_run", "counts_toward_non_postgresql": False},
        "artifact_root": str(gate_root),
    }
    _write_json(gate_root / "gate-summary.json", summary)
    _print_summary(summary)
    return 0 if summary["three_consecutive_clean_runs"] else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPOSITORY / ".daita" / "release-gates" / "from_db_bucket3",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--include-postgresql", action="store_true")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    return args


def _require_project_virtualenv() -> None:
    expected = (REPOSITORY / ".venv").resolve()
    if Path(sys.prefix).resolve() != expected:
        raise SystemExit(
            "Run with the project interpreter: "
            ".venv/bin/python scripts/run_from_db_bucket3_gate.py"
        )


def _run_pytest_gate(gate_root: Path, *, index: int) -> dict[str, object]:
    run_root = gate_root / f"run-{index:02d}"
    artifact_root = run_root / "artifacts"
    base_temp = run_root / "pytest-temp"
    junit = run_root / "junit.xml"
    artifact_root.mkdir(parents=True)
    command = [
        sys.executable,
        "-m",
        "pytest",
        *BUCKET3_FILES,
        "-m",
        "not requires_db",
        "-q",
        "-rs",
        "-s",
        f"--basetemp={base_temp}",
        f"--junitxml={junit}",
    ]
    env = {
        **os.environ,
        "DAITA_RUN_LIVE_LLM": "1",
        "DAITA_BUCKET3_ARTIFACT_ROOT": str(artifact_root),
        "DAITA_BUCKET3_POSTGRES_STATUS": "not_run_separate_required_gate",
    }
    completed = subprocess.run(command, cwd=REPOSITORY, env=env, check=False)
    counts = _junit_counts(junit)
    clean = completed.returncode == 0 and counts.get("skipped", 0) == 0
    return {
        "index": index,
        "command": _display_command(command),
        "returncode": completed.returncode,
        "counts": counts,
        "junit": str(junit),
        "artifact_root": str(artifact_root),
        "counts_toward_consecutive_requirement": clean,
    }


def _run_postgres_gate(gate_root: Path) -> dict[str, object]:
    run_root = gate_root / "postgresql"
    artifact_root = run_root / "artifacts"
    base_temp = run_root / "pytest-temp"
    junit = run_root / "junit.xml"
    artifact_root.mkdir(parents=True)
    command = [
        sys.executable,
        "-m",
        "pytest",
        POSTGRES_TEST,
        "-q",
        "-rs",
        "-s",
        f"--basetemp={base_temp}",
        f"--junitxml={junit}",
    ]
    env = {
        **os.environ,
        "DAITA_RUN_LIVE_LLM": "1",
        "DAITA_BUCKET3_ARTIFACT_ROOT": str(artifact_root),
        "DAITA_BUCKET3_POSTGRES_STATUS": "running",
    }
    completed = subprocess.run(command, cwd=REPOSITORY, env=env, check=False)
    counts = _junit_counts(junit)
    status = (
        "passed"
        if completed.returncode == 0 and not counts.get("skipped")
        else "failed"
    )
    if counts.get("skipped"):
        status = "skipped_not_counted"
    return {
        "status": status,
        "command": _display_command(command),
        "returncode": completed.returncode,
        "counts": counts,
        "junit": str(junit),
        "artifact_root": str(artifact_root),
        "counts_toward_non_postgresql": False,
    }


def _junit_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {"passed": 0, "failed": 0, "skipped": 0, "errors": 1}
    root = ElementTree.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
    return {
        "passed": tests - failures - errors - skipped,
        "failed": failures,
        "skipped": skipped,
        "errors": errors,
    }


def _consecutive_clean_runs(records: list[dict[str, object]]) -> int:
    consecutive = 0
    for record in records:
        if record["counts_toward_consecutive_requirement"]:
            consecutive += 1
        else:
            consecutive = 0
    return consecutive


def _repository_state() -> str:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return f"{sha} ({'dirty worktree' if dirty else 'clean worktree'})"


def _display_command(command: list[str]) -> str:
    displayed = [
        ".venv/bin/python" if item == sys.executable else item for item in command
    ]
    return " ".join(displayed)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            _redact_artifact_payload(payload),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _print_summary(summary: dict[str, object]) -> None:
    print(f"Artifacts: {summary['artifact_root']}")
    for record in summary["non_postgresql_runs"]:
        print(
            f"Run {record['index']}: {record['counts']} "
            f"JUnit={record['junit']} Artifacts={record['artifact_root']}"
        )
    print(f"Three consecutive clean runs: {summary['three_consecutive_clean_runs']}")
    print(f"PostgreSQL: {summary['postgresql']['status']}")


if __name__ == "__main__":
    raise SystemExit(main())
