"""Run the fixed live smoke list on the subscription and record the result.

    .venv/bin/python -m tests.support.live_billing login     # once
    DAITA_LIVE_BILLING=subscription .venv/bin/python scripts/live_smoke.py run
    .venv/bin/python scripts/live_smoke.py list

``run`` executes one representative real-model case per live suite, each in
its own pytest process with that suite's documented authorization flags. It
refuses to start unless ``DAITA_LIVE_BILLING=subscription`` is set, and it
strips every ``*_API_KEY`` variable from the case environment, so no case can
fall back to paid API billing. A case that fails and then passes on its one
rerun counts as passed and is recorded as ``passed_on_rerun``. Only when every
case passes does it rewrite ``release/live-smoke.json`` with the current
model-surface digest. Evidence (JUnit and pytest output) stays in a private
temporary directory outside the repository. See docs/TESTING.md,
"Model-surface gate".
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

_ROOT = Path(__file__).resolve().parents[1]
for _entry in (str(_ROOT / "src"), str(_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from tests.support.live_smoke_record import (
    CASE_PASSED,
    CASE_PASSED_ON_RERUN,
    RECORD_PATH,
    build_record,
    record_text,
)
from tests.support.model_surface import (
    UPDATE_COMMAND,
    committed_digest,
    compute_surface_text,
    digest_text,
)

BILLING_ENV = "DAITA_LIVE_BILLING"
CASE_TIMEOUT_SECONDS = 20 * 60
DEFAULT_JOBS = 3


@dataclass(frozen=True)
class SmokeCase:
    """One live node with the suite flags it needs, per docs/TESTING.md."""

    suite: str
    node_id: str
    env: Mapping[str, str]
    # Developer-supplied resources that must already be configured; checked
    # for presence only, never read or printed.
    requires_env: tuple[str, ...] = ()
    # Cases sharing a resource never run concurrently.
    resource: str | None = None
    # Report-directory variables pointed at this run's private evidence.
    report_env: tuple[str, ...] = field(default=())


_MODEL = "openai:gpt-5.6-terra"
_PHASE_F = {
    "DAITA_RUN_LIVE_PHASE_F_MCP": "1",
    "DAITA_PHASE_F_LIVE_PROFILE": "user_flow",
}
_POSTGRES_LIVE = ("DAITA_LARGE_POSTGRES_WRITE_TESTER_PASSWORD",)

# The single smoke list. One representative case per live suite.
CASES: tuple[SmokeCase, ...] = (
    SmokeCase(
        "benchmarks/paraphrases",
        "tests/live/benchmarks/test_paraphrases.py::"
        "test_paraphrased_immediate_reads_do_not_create_jobs[quick-lookup]",
        {"DAITA_RUN_LIVE_STAGE_B_PARAPHRASE_BENCHMARK": "1"},
    ),
    SmokeCase(
        "benchmarks/catalog_scale",
        "tests/live/benchmarks/test_catalog_scale.py::"
        "test_immediate_target_survives_catalog_distractors[64]",
        {"DAITA_RUN_LIVE_STAGE_B_CATALOG_BENCHMARK": "1"},
    ),
    SmokeCase(
        "data_operations",
        "tests/live/data_operations/test_workflows.py::"
        f"test_live_analytical_correctness[{_MODEL}-0-join_grain-00]",
        {"DAITA_RUN_LIVE_DATA_OPERATIONS": "1"},
        report_env=("DAITA_DATAOPS_REPORT_DIR",),
    ),
    SmokeCase(
        "acceptance/l8_graph",
        "tests/live/acceptance/test_result_refs.py::"
        "test_l8_graph_task_complete_cites_its_read[0]",
        {"DAITA_RUN_LIVE_RESULT_REFS": "1"},
    ),
    SmokeCase(
        "mcp/reads",
        "tests/live/mcp/test_read_workflows.py::"
        "test_live_dependent_reads_discover_tools_and_bind_returned_account"
        "[0-2-2026-07-28-normal]",
        {"DAITA_RUN_LIVE_MCP_READS": "1"},
        report_env=("DAITA_LIVE_MCP_READ_REPORT_DIR",),
    ),
    SmokeCase(
        "mcp/actions",
        "tests/live/mcp/test_actions.py::"
        f"test_live_research_and_honest_action_evidence[{_MODEL}-0-success]",
        _PHASE_F,
        report_env=("DAITA_PHASE_F_LIVE_REPORT_DIR",),
    ),
    SmokeCase(
        "jobs/graph_planner",
        "tests/live/jobs/test_graph_planner.py::"
        "test_live_planner_uses_typed_replacement_without_production_cutover",
        {"DAITA_RUN_LIVE_GRAPH_PLANNER": "1"},
    ),
    SmokeCase(
        "routines/contract_authoring",
        "tests/live/routines/test_contract_authoring.py::"
        "test_model_authors_scheduled_contract_without_dispatch[0-familiar]",
        {**_PHASE_F, "DAITA_RUN_CONTRACT_QUALIFICATION": "1"},
        report_env=("DAITA_PHASE_F_LIVE_REPORT_DIR",),
    ),
    SmokeCase(
        "data/write_release",
        "tests/live/data/test_model_write_acceptance.py::"
        f"test_live_mixed_upsert_and_unchanged_repeat[{_MODEL}-0]",
        {"DAITA_RUN_POSTGRES_LIVE_LLM": "1"},
        requires_env=_POSTGRES_LIVE,
        resource="postgres-large",
        report_env=("DAITA_POSTGRES_LIVE_REPORT_DIR",),
    ),
    SmokeCase(
        "data/drift_during_approval",
        "tests/live/data/test_model_write_acceptance.py::"
        f"test_live_drift_during_approval_does_not_reapply[{_MODEL}-0]",
        {
            "DAITA_RUN_POSTGRES_LIVE_LLM": "1",
            "DAITA_POSTGRES_LIVE_PROFILE": "user_flow",
        },
        requires_env=_POSTGRES_LIVE,
        resource="postgres-large",
        report_env=("DAITA_POSTGRES_LIVE_REPORT_DIR",),
    ),
    SmokeCase(
        "workspace/edit",
        "tests/live/workspace/test_text_edit.py::"
        "test_live_model_reads_edits_approves_and_replaces_exact_bound_file",
        {"DAITA_RUN_LIVE_LOCAL_TEXT_EDIT": "1"},
    ),
    SmokeCase(
        "artifacts/lifecycle",
        "tests/live/artifacts/test_lifecycle.py::"
        "test_live_model_artifact_creation_reuse_and_deletion_survive_reopen",
        {"DAITA_RUN_LIVE_ARTIFACT_LIFECYCLE": "1"},
    ),
    SmokeCase(
        "analysis",
        "tests/live/analysis/test_public_journeys.py::"
        "test_real_model_dependent_cells_and_error_correction",
        {"DAITA_RUN_LIVE_ANALYSIS": "1"},
        resource="native-analysis",
        report_env=("DAITA_ANALYSIS_LIVE_REPORT_DIR",),
    ),
)


def node_ids() -> tuple[str, ...]:
    return tuple(case.node_id for case in CASES)


def case_environment(
    case: SmokeCase, base: Mapping[str, str], evidence: Path
) -> dict[str, str]:
    """The child environment: subscription billing, no API keys, suite flags."""

    environment = {
        key: value for key, value in base.items() if not key.endswith("_API_KEY")
    }
    environment.update(case.env)
    environment[BILLING_ENV] = "subscription"
    for name in case.report_env:
        environment.setdefault(name, str(evidence / "reports" / _slug(case)))
    return environment


def _slug(case: SmokeCase) -> str:
    return case.suite.replace("/", "-")


def junit_outcome(path: Path, node_id: str) -> str:
    """Return passed, failed, skipped or not_collected for the one node."""

    try:
        tree = ElementTree.parse(path)
    except (OSError, ElementTree.ParseError):
        return "not_collected"
    cases = list(tree.iter("testcase"))
    if len(cases) != 1:
        return "not_collected"
    (testcase,) = cases
    name = node_id.rsplit("::", 1)[1]
    if testcase.get("name") != name:
        return "not_collected"
    if testcase.find("failure") is not None or testcase.find("error") is not None:
        return "failed"
    if testcase.find("skipped") is not None:
        return "skipped"
    return "passed"


def _attempt(
    case: SmokeCase, attempt: int, base: Mapping[str, str], evidence: Path
) -> str:
    stem = evidence / f"{_slug(case)}-attempt-{attempt}"
    junit = stem.with_suffix(".xml")
    command = [
        sys.executable,
        "-m",
        "pytest",
        case.node_id,
        "-o",
        "addopts=--tb=short -q --strict-markers",
        "-p",
        "no:cacheprovider",
        f"--junitxml={junit}",
        "-o",
        "junit_family=xunit1",
    ]
    with stem.with_suffix(".log").open("wb") as log:
        try:
            completed = subprocess.run(
                command,
                cwd=_ROOT,
                env=case_environment(case, base, evidence),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=CASE_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "timed_out"
    outcome = junit_outcome(junit, case.node_id)
    if outcome == "passed" and completed.returncode != 0:
        return "failed"
    return outcome


def run_case(case: SmokeCase, base: Mapping[str, str], evidence: Path) -> str:
    """Run once and, on an ordinary failure, rerun exactly once."""

    first = _attempt(case, 1, base, evidence)
    if first == "passed":
        return CASE_PASSED
    if first in {"skipped", "not_collected"}:
        # A skip or a missing node is configuration, not flakiness.
        return first
    second = _attempt(case, 2, base, evidence)
    return CASE_PASSED_ON_RERUN if second == "passed" else first


def _run_all(
    cases: Sequence[SmokeCase], jobs: int, base: Mapping[str, str], evidence: Path
) -> dict[str, str]:
    locks: dict[str, threading.Lock] = {
        case.resource: threading.Lock() for case in cases if case.resource
    }
    printer = threading.Lock()

    def one(case: SmokeCase) -> tuple[str, str]:
        lock = locks.get(case.resource) if case.resource else None
        if lock is not None:
            lock.acquire()
        started = time.monotonic()
        try:
            outcome = run_case(case, base, evidence)
        finally:
            if lock is not None:
                lock.release()
        with printer:
            print(
                f"{outcome:>16}  {time.monotonic() - started:6.0f}s  {case.suite}",
                flush=True,
            )
        return case.node_id, outcome

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return dict(pool.map(one, cases))


def _list() -> int:
    for case in CASES:
        flags = " ".join(f"{key}={value}" for key, value in sorted(case.env.items()))
        print(f"{case.suite}\n  {case.node_id}\n  {flags}")
        if case.requires_env:
            print(f"  requires: {', '.join(case.requires_env)}")
    return 0


def _run(selected: Sequence[str], jobs: int) -> int:
    if os.environ.get(BILLING_ENV) != "subscription":
        print(
            f"Refusing to run: set {BILLING_ENV}=subscription. The live smoke "
            "never uses paid API billing.",
            file=sys.stderr,
        )
        return 2
    digest = digest_text(compute_surface_text())
    if committed_digest() != digest:
        print(
            "The committed model-surface snapshot is stale; run "
            f"{UPDATE_COMMAND} and commit it before the live smoke.",
            file=sys.stderr,
        )
        return 2
    cases = [
        case
        for case in CASES
        if not selected or any(item in case.node_id for item in selected)
    ]
    if not cases:
        print("No smoke case matches --case.", file=sys.stderr)
        return 2
    missing = sorted(
        {
            name
            for case in cases
            for name in case.requires_env
            if not os.environ.get(name)
        }
    )
    if missing:
        print(
            f"Configure the live fixtures first; unset: {', '.join(missing)}",
            file=sys.stderr,
        )
        return 2
    evidence = Path(tempfile.mkdtemp(prefix="daita-live-smoke-"))
    print(f"surface {digest}; private evidence in {evidence}", flush=True)
    outcomes = _run_all(cases, max(1, jobs), os.environ, evidence)
    failed = {
        node: outcome
        for node, outcome in outcomes.items()
        if outcome not in {CASE_PASSED, CASE_PASSED_ON_RERUN}
    }
    if failed:
        for node, outcome in sorted(failed.items()):
            print(f"FAILED ({outcome}): {node}", file=sys.stderr)
        print("Record not written.", file=sys.stderr)
        return 1
    if selected:
        print("All selected cases passed; a partial run does not write the record.")
        return 0
    record = build_record(
        surface_digest=digest, cases=outcomes, recorded_on=datetime.now(UTC).date()
    )
    RECORD_PATH.write_text(record_text(record), encoding="utf-8")
    print(f"All {len(outcomes)} cases passed; wrote {RECORD_PATH.relative_to(_ROOT)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="print the smoke cases and their flags")
    run = commands.add_parser("run", help="run the smoke and record a full pass")
    run.add_argument(
        "--case",
        action="append",
        default=[],
        help="run only cases whose node ID contains this text (never records)",
    )
    run.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    arguments = parser.parse_args(argv)
    if arguments.command == "list":
        return _list()
    return _run(arguments.case, arguments.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
