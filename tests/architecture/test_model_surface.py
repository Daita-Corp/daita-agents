"""The model-facing surface is snapshotted and gated on a live smoke record.

Scripted models replay old behaviour, so a changed tool description, schema,
guidance text or result shape passes every deterministic test. The snapshot
makes such a change deliberate; the record makes it reach a live smoke run on
the subscription before merge. See docs/TESTING.md, "Model-surface gate".
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from scripts import live_smoke
from tests.support.live_smoke_record import (
    CASE_PASSED,
    CASE_PASSED_ON_RERUN,
    REQUIRE_ENV,
    build_record,
    gate_failure,
    load_record,
    record_problems,
    require_recorded,
)
from tests.support.model_surface import (
    DIGEST_PATH,
    SNAPSHOT_PATH,
    UPDATE_COMMAND,
    committed_digest,
    compute_surface_text,
    describe_changes,
    digest_text,
)
from tests.support.paths import REPO_ROOT

DIGEST = "sha256:" + "a" * 64
OTHER_DIGEST = "sha256:" + "b" * 64


def test_model_surface_snapshot_is_current() -> None:
    current = compute_surface_text()
    committed = SNAPSHOT_PATH.read_text(encoding="utf-8")
    if committed != current:
        changes = "\n".join(
            f"  - {line}" for line in describe_changes(committed, current)
        )
        pytest.fail(
            "The model-facing surface changed:\n"
            f"{changes}\n"
            f"Review the diff, then run: {UPDATE_COMMAND}\n"
            "A changed surface also needs a live smoke run before merge.",
            pytrace=False,
        )
    assert committed_digest() == digest_text(
        committed
    ), f"{DIGEST_PATH.name} does not match the snapshot; run {UPDATE_COMMAND}"


def test_live_smoke_record_covers_the_current_surface() -> None:
    digest = committed_digest()
    assert digest is not None, f"missing {DIGEST_PATH.name}; run {UPDATE_COMMAND}"
    failure = gate_failure(
        load_record(),
        surface_digest=digest,
        expected_cases=live_smoke.node_ids(),
        require_recorded=require_recorded(),
    )
    if failure is not None:
        pytest.fail(failure, pytrace=False)


def _recorded(**changes: object) -> dict[str, object]:
    record = build_record(
        surface_digest=DIGEST,
        cases={"tests/live/a.py::test_a": CASE_PASSED},
        recorded_on=date(2026, 10, 1),
    )
    record.update(changes)
    return record


def _pending(digest: str = DIGEST) -> dict[str, object]:
    return {
        "status": "pending",
        "surface_digest": digest,
        "billing": "subscription",
        "recorded_on": None,
        "cases": {},
        "summary": {"passed": 0, "total": 0},
    }


def _gate(record: dict[str, object] | None, *, require: bool) -> str | None:
    return gate_failure(
        record,
        surface_digest=DIGEST,
        expected_cases=("tests/live/a.py::test_a",),
        require_recorded=require,
    )


def test_gate_blocks_missing_stale_and_malformed_records() -> None:
    assert "missing" in str(_gate(None, require=False))
    stale = _gate(_pending(OTHER_DIGEST), require=False)
    assert stale is not None and "surface changed since the last live smoke" in stale
    assert "malformed" in str(_gate(_recorded(billing="api"), require=False))
    assert "malformed" in str(_gate(_recorded(host="laptop"), require=False))
    assert "malformed" in str(
        _gate(_recorded(cases={"tests/live/a.py::test_a": "failed"}), require=False)
    )
    # A record for another smoke list does not cover the current one.
    other_list = gate_failure(
        _recorded(),
        surface_digest=DIGEST,
        expected_cases=("tests/live/a.py::test_a", "tests/live/b.py::test_b"),
        require_recorded=True,
    )
    assert (
        other_list is not None and "missing ['tests/live/b.py::test_b']" in other_list
    )


def test_pending_record_blocks_only_when_a_live_smoke_is_required() -> None:
    assert _gate(_pending(), require=False) is None
    pending = _gate(_pending(), require=True)
    assert pending is not None and REQUIRE_ENV in pending
    assert _gate(_recorded(), require=True) is None


def test_recorded_runs_count_reruns_as_passed() -> None:
    record = build_record(
        surface_digest=DIGEST,
        cases={"x::a": CASE_PASSED, "x::b": CASE_PASSED_ON_RERUN},
        recorded_on=date(2026, 10, 1),
    )
    assert record["summary"] == {"passed": 2, "total": 2}
    assert record_problems(record, expected_cases=("x::a", "x::b")) == []
    with pytest.raises(ValueError):
        build_record(
            surface_digest=DIGEST,
            cases={"x::a": "failed"},
            recorded_on=date(2026, 10, 1),
        )


def test_committed_record_is_well_formed() -> None:
    record = load_record()
    assert record is not None
    assert record_problems(record, expected_cases=live_smoke.node_ids()) == []


def test_live_smoke_refuses_without_subscription_billing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for value in (None, "api"):
        if value is None:
            monkeypatch.delenv(live_smoke.BILLING_ENV, raising=False)
        else:
            monkeypatch.setenv(live_smoke.BILLING_ENV, value)
        assert live_smoke.main(["run"]) == 2
        assert "never uses paid API billing" in capsys.readouterr().err


def test_case_environment_forces_subscription_and_strips_api_keys(
    tmp_path: Path,
) -> None:
    case = next(case for case in live_smoke.CASES if case.report_env)
    environment = live_smoke.case_environment(
        case,
        {"OPENAI_API_KEY": "unused", "DAITA_LIVE_BILLING": "api", "PATH": "/bin"},
        tmp_path,
    )
    assert "OPENAI_API_KEY" not in environment
    assert environment["DAITA_LIVE_BILLING"] == "subscription"
    assert environment["PATH"] == "/bin"
    for key, value in case.env.items():
        assert environment[key] == value
    for name in case.report_env:
        assert Path(environment[name]).is_relative_to(tmp_path)


@pytest.mark.parametrize(
    "attempts,expected,count",
    [
        (["passed"], CASE_PASSED, 1),
        (["failed", "passed"], CASE_PASSED_ON_RERUN, 2),
        (["timed_out", "passed"], CASE_PASSED_ON_RERUN, 2),
        (["failed", "failed"], "failed", 2),
        (["skipped"], "skipped", 1),
        (["not_collected"], "not_collected", 1),
    ],
)
def test_one_rerun_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    attempts: list[str],
    expected: str,
    count: int,
) -> None:
    calls: list[int] = []

    def attempt(case: object, number: int, base: object, evidence: object) -> str:
        calls.append(number)
        return attempts[number - 1]

    monkeypatch.setattr(live_smoke, "_attempt", attempt)
    assert live_smoke.run_case(live_smoke.CASES[0], {}, tmp_path) == expected
    assert calls == list(range(1, count + 1))


def test_junit_outcome_requires_exactly_the_selected_node(tmp_path: Path) -> None:
    node = "tests/live/x.py::test_case[param]"

    def write(body: str) -> Path:
        path = tmp_path / "result.xml"
        path.write_text(f"<testsuites><testsuite>{body}</testsuite></testsuites>")
        return path

    case = '<testcase classname="tests.live.x" name="test_case[param]"'
    assert live_smoke.junit_outcome(write(case + "/>"), node) == "passed"
    assert live_smoke.junit_outcome(write(case + "><failure/></testcase>"), node) == (
        "failed"
    )
    assert live_smoke.junit_outcome(write(case + "><error/></testcase>"), node) == (
        "failed"
    )
    assert live_smoke.junit_outcome(write(case + "><skipped/></testcase>"), node) == (
        "skipped"
    )
    assert live_smoke.junit_outcome(write(""), node) == "not_collected"
    assert live_smoke.junit_outcome(tmp_path / "absent.xml", node) == "not_collected"


def test_smoke_list_has_one_case_per_suite_and_every_node_collects() -> None:
    suites = [case.suite for case in live_smoke.CASES]
    nodes = live_smoke.node_ids()
    assert len(set(suites)) == len(suites)
    assert len(set(nodes)) == len(nodes)
    assert all(node.startswith("tests/live/") for node in nodes)
    assert all(
        any(key.startswith("DAITA_RUN_") for key in case.env)
        for case in live_smoke.CASES
    )
    # Collection resolves the exact parametrized IDs without running a case.
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("DAITA_") and not key.endswith("_API_KEY")
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-o",
            "addopts=--strict-markers",
            "-p",
            "no:cacheprovider",
            *nodes,
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    collected = {
        line.strip()
        for line in completed.stdout.splitlines()
        if line.startswith("tests/live/")
    }
    assert collected == set(nodes), completed.stdout
