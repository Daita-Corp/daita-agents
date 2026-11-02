"""The model-facing surface is snapshotted and gated on a live smoke record.

Scripted models replay old behaviour, so a changed tool description, schema,
guidance text or result shape passes every deterministic test. The snapshot
makes such a change deliberate; the record makes it reach a live smoke run on
the subscription before merge. See docs/TESTING.md, "Model-surface gate".
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from daita._json import canonical_json
from daita.llm.models import TextBlock, ToolResultBlock, model_tool_result
from scripts import live_smoke
from tests.support.model_surface import (
    DIGEST_PATH,
    SNAPSHOT_PATH,
    UPDATE_COMMAND,
    capture_requests,
    committed_digest,
    compute_surface_text,
    describe_changes,
    digest_text,
)
from tests.support.paths import REPO_ROOT

DIGEST = "sha256:" + "a" * 64
CASES = ("tests/live/a.py::test_a",)


def test_model_surface_snapshot_is_current() -> None:
    current = compute_surface_text()
    committed = SNAPSHOT_PATH.read_text(encoding="utf-8")
    if committed != current:
        changes = "\n".join(
            f"  - {line}" for line in describe_changes(committed, current)
        )
        pytest.fail(
            f"The model-facing surface changed:\n{changes}\n"
            f"Review the diff, then run: {UPDATE_COMMAND}\n"
            "A changed surface also needs a live smoke run before merge.",
            pytrace=False,
        )
    assert committed_digest() == digest_text(
        committed
    ), f"{DIGEST_PATH.name} does not match the snapshot; run {UPDATE_COMMAND}"


# A resource ID, a source ID, or a run-local handle as a JSON string value.
_IDENTITY_TEXT = re.compile(
    r'catalog-resource:sha256:|source:(?:workspace:)?sha256:|"[st][1-9][0-9]{0,4}"'
)


def test_model_never_sees_an_id_or_a_handle() -> None:
    """Sources are named by display name, tables and files by native name.

    Every tool definition, instruction paragraph and rendered tool result of
    the fixture run (including toolbox_inspect contracts and the catalog
    context) is free of IDs and handles; the resolver maps names back.
    """

    requests, definitions = capture_requests()
    shown: list[str] = [
        canonical_json({"d": item.description, "s": item.input_schema})
        for item in definitions
    ]
    for request in requests:
        shown.extend(
            canonical_json({"d": item.description, "s": item.input_schema})
            for item in request.tools
        )
        for message in request.messages:
            for block in message.content:
                if isinstance(block, TextBlock):
                    shown.append(block.text)
                elif isinstance(block, ToolResultBlock):
                    shown.append(
                        canonical_json(
                            model_tool_result(
                                block,
                                request.tool_result_refs.get(block.call_id),
                                request.scope_names,
                            )
                        )
                    )
    assert len(requests) >= 4 and any("toolbox_inspect" in text for text in shown)
    offending = [
        text[max(0, match.start() - 80) : match.end() + 20]
        for text in shown
        for match in _IDENTITY_TEXT.finditer(text)
    ]
    assert offending == []


def test_live_smoke_record_covers_the_current_surface() -> None:
    digest = committed_digest()
    assert digest is not None, f"missing {DIGEST_PATH.name}; run {UPDATE_COMMAND}"
    record = live_smoke.load_record()
    failure = live_smoke.gate_failure(
        record, surface_digest=digest, expected_cases=live_smoke.node_ids()
    )
    if failure is None:
        return
    stale = record is not None and "surface changed" in failure
    if stale and os.environ.get(live_smoke.REQUIRE_ENV) != "1":
        # Developers may change the surface locally; pull requests may not
        # merge it without a recorded live smoke run.
        pytest.skip(f"{failure}\n(enforced when {live_smoke.REQUIRE_ENV}=1)")
    pytest.fail(failure, pytrace=False)


def _record(**changes: object) -> dict[str, object]:
    record: dict[str, object] = {
        "surface_digest": DIGEST,
        "recorded_on": "2026-10-01",
        "cases": {CASES[0]: live_smoke.CASE_PASSED},
    }
    record.update(changes)
    return record


@pytest.mark.parametrize(
    "record,reason",
    [
        (None, "missing"),
        (_record(surface_digest="sha256:" + "b" * 64), "surface changed"),
        (_record(billing="api"), "malformed"),
        (_record(recorded_on="yesterday"), "malformed"),
        (_record(cases={CASES[0]: "failed"}), "malformed"),
        (_record(cases={"tests/live/other.py::test_b": "passed"}), "malformed"),
    ],
)
def test_gate_blocks_missing_stale_and_malformed_records(
    record: dict[str, object] | None, reason: str
) -> None:
    failure = live_smoke.gate_failure(
        record, surface_digest=DIGEST, expected_cases=CASES
    )
    assert failure is not None and reason in failure


def test_gate_accepts_a_passing_record_including_reruns() -> None:
    for outcome in (live_smoke.CASE_PASSED, live_smoke.CASE_PASSED_ON_RERUN):
        record = _record(cases={CASES[0]: outcome})
        assert (
            live_smoke.gate_failure(record, surface_digest=DIGEST, expected_cases=CASES)
            is None
        )


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
    case = live_smoke.CASES[0]
    environment = live_smoke.case_environment(
        case,
        {"OPENAI_API_KEY": "unused", "DAITA_LIVE_BILLING": "api", "PATH": "/bin"},
        tmp_path,
    )
    assert "OPENAI_API_KEY" not in environment
    assert environment["DAITA_LIVE_BILLING"] == "subscription"
    assert environment["PATH"] == "/bin"
    assert all(environment[key] == value for key, value in case.env.items())
    assert Path(environment["DAITA_DATAOPS_REPORT_DIR"]).is_relative_to(tmp_path)


@pytest.mark.parametrize(
    "attempts,expected,count",
    [
        (["passed"], live_smoke.CASE_PASSED, 1),
        (["failed", "passed"], live_smoke.CASE_PASSED_ON_RERUN, 2),
        (["timed_out", "passed"], live_smoke.CASE_PASSED_ON_RERUN, 2),
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


@pytest.mark.parametrize(
    "body,expected",
    [
        ("/>", "passed"),
        ("><failure/></testcase>", "failed"),
        ("><error/></testcase>", "failed"),
        ("><skipped/></testcase>", "skipped"),
        (None, "not_collected"),
    ],
)
def test_junit_outcome_requires_exactly_the_selected_node(
    tmp_path: Path, body: str | None, expected: str
) -> None:
    case = "" if body is None else f'<testcase name="test_case[param]"{body}'
    path = tmp_path / "result.xml"
    path.write_text(f"<testsuites><testsuite>{case}</testsuite></testsuites>")
    assert live_smoke.junit_outcome(path, "tests/live/x.py::test_case[param]") == (
        expected
    )


def test_smoke_list_has_one_case_per_suite_and_every_node_collects() -> None:
    suites = [case.suite for case in live_smoke.CASES]
    nodes = live_smoke.node_ids()
    assert len(set(suites)) == len(suites) and len(set(nodes)) == len(nodes)
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
