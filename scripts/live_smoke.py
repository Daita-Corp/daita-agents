"""Run the fixed live smoke list on the subscription and record the result.

    .venv/bin/python -m tests.support.live_billing login     # once
    DAITA_LIVE_BILLING=subscription .venv/bin/python scripts/live_smoke.py run
    .venv/bin/python scripts/live_smoke.py list

``run`` executes one representative real-model case per live suite, each in
its own pytest process with that suite's documented authorization flags. It
refuses to start unless ``DAITA_LIVE_BILLING=subscription`` is set and strips
every ``*_API_KEY`` variable from the case environment, so no case can fall
back to paid API billing. A failed case reruns once; a pass on rerun is
recorded as ``passed_on_rerun``. Only a full pass rewrites
``release/live-smoke.json`` with the current model-surface digest; the record
holds no machine, user, path or credential facts. Evidence stays in a private
temporary directory. See docs/TESTING.md, "Model-surface gate".
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from collections.abc import Collection, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

_ROOT = Path(__file__).resolve().parents[1]
for _entry in (str(_ROOT / "src"), str(_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from tests.support.model_surface import (
    UPDATE_COMMAND,
    committed_digest,
    compute_surface_text,
    digest_text,
)

BILLING_ENV = "DAITA_LIVE_BILLING"
REQUIRE_ENV = "DAITA_REQUIRE_LIVE_SMOKE"
RECORD_PATH = _ROOT / "release" / "live-smoke.json"
RUN_COMMAND = (
    "DAITA_LIVE_BILLING=subscription .venv/bin/python scripts/live_smoke.py run"
)
CASE_PASSED = "passed"
CASE_PASSED_ON_RERUN = "passed_on_rerun"
CASE_TIMEOUT_SECONDS = 20 * 60
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class SmokeCase:
    """One live node with the suite flags it needs, per docs/TESTING.md."""

    suite: str
    node_id: str
    env: Mapping[str, str]
    # Cases sharing a resource never run concurrently.
    resource: str | None = None


_MODEL = "openai:gpt-5.6-terra"
_PHASE_F = {
    "DAITA_RUN_LIVE_PHASE_F_MCP": "1",
    "DAITA_PHASE_F_LIVE_PROFILE": "user_flow",
}
_POSTGRES = {"DAITA_RUN_POSTGRES_LIVE_LLM": "1"}

# The single smoke list: one representative case per live suite.
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
    ),
    SmokeCase(
        "mcp/actions",
        "tests/live/mcp/test_actions.py::"
        f"test_live_research_and_honest_action_evidence[{_MODEL}-0-success]",
        _PHASE_F,
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
    ),
    SmokeCase(
        "data/write_release",
        "tests/live/data/test_model_write_acceptance.py::"
        f"test_live_mixed_upsert_and_unchanged_repeat[{_MODEL}-0]",
        _POSTGRES,
        resource="postgres-large",
    ),
    SmokeCase(
        "data/drift_during_approval",
        "tests/live/data/test_model_write_acceptance.py::"
        f"test_live_drift_during_approval_does_not_reapply[{_MODEL}-0]",
        {**_POSTGRES, "DAITA_POSTGRES_LIVE_PROFILE": "user_flow"},
        resource="postgres-large",
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
    ),
)
# Every suite's private report directory points into this run's evidence.
_REPORT_ENV = (
    "DAITA_ANALYSIS_LIVE_REPORT_DIR",
    "DAITA_DATAOPS_REPORT_DIR",
    "DAITA_LIVE_MCP_READ_REPORT_DIR",
    "DAITA_PHASE_F_LIVE_REPORT_DIR",
    "DAITA_POSTGRES_LIVE_REPORT_DIR",
)


def node_ids() -> tuple[str, ...]:
    return tuple(case.node_id for case in CASES)


# The committed record: {"surface_digest", "recorded_on", "cases"}.


def load_record() -> dict[str, Any] | None:
    try:
        value = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise TypeError("live smoke record must be a JSON object")
    return value


def record_problems(
    record: Mapping[str, object], *, expected_cases: Collection[str]
) -> list[str]:
    problems: list[str] = []
    if set(record) != {"surface_digest", "recorded_on", "cases"}:
        problems.append("keys must be exactly surface_digest, recorded_on, cases")
    digest = record.get("surface_digest")
    if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
        problems.append("surface_digest must be sha256:<64 hex>")
    try:
        date.fromisoformat(str(record.get("recorded_on")))
    except ValueError:
        problems.append("recorded_on must be YYYY-MM-DD")
    cases = record.get("cases")
    if not isinstance(cases, dict) or set(cases) != set(expected_cases):
        problems.append("cases must name exactly the smoke list")
    elif any(
        value not in {CASE_PASSED, CASE_PASSED_ON_RERUN} for value in cases.values()
    ):
        problems.append("every case must be passed or passed_on_rerun")
    return problems


def gate_failure(
    record: Mapping[str, object] | None,
    *,
    surface_digest: str,
    expected_cases: Collection[str],
) -> str | None:
    """Explain why the record does not cover this surface, or return None."""

    how = f"Run the live smoke on the subscription:\n{RUN_COMMAND}"
    if record is None:
        return f"{RECORD_PATH.name} is missing. {how}"
    problems = record_problems(record, expected_cases=expected_cases)
    if problems:
        return f"{RECORD_PATH.name} is malformed: {'; '.join(problems)}"
    if record["surface_digest"] != surface_digest:
        return (
            "The model-facing surface changed since the last live smoke run "
            f"(record {record['surface_digest']}, snapshot {surface_digest}). {how}"
        )
    return None


# Running the smoke.


def case_environment(
    case: SmokeCase, base: Mapping[str, str], evidence: Path
) -> dict[str, str]:
    """The child environment: subscription billing, no API keys, suite flags."""

    environment = {
        key: value for key, value in base.items() if not key.endswith("_API_KEY")
    }
    environment.update(case.env)
    environment[BILLING_ENV] = "subscription"
    for name in _REPORT_ENV:
        environment.setdefault(name, str(evidence / "reports" / _slug(case)))
    return environment


def _slug(case: SmokeCase) -> str:
    return case.suite.replace("/", "-")


def junit_outcome(path: Path, node_id: str) -> str:
    """Return passed, failed, skipped or not_collected for the one node."""

    try:
        cases = list(ElementTree.parse(path).iter("testcase"))
    except (OSError, ElementTree.ParseError):
        return "not_collected"
    if len(cases) != 1 or cases[0].get("name") != node_id.rsplit("::", 1)[1]:
        return "not_collected"
    if cases[0].find("failure") is not None or cases[0].find("error") is not None:
        return "failed"
    return "skipped" if cases[0].find("skipped") is not None else "passed"


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
    return "failed" if outcome == "passed" and completed.returncode else outcome


def run_case(case: SmokeCase, base: Mapping[str, str], evidence: Path) -> str:
    """Run once and, on an ordinary failure, rerun exactly once."""

    first = _attempt(case, 1, base, evidence)
    if first == "passed":
        return CASE_PASSED
    if first in {"skipped", "not_collected"}:
        # A skip or a missing node is configuration, not flakiness.
        return first
    return (
        CASE_PASSED_ON_RERUN if _attempt(case, 2, base, evidence) == "passed" else first
    )


def _run_all(
    cases: Sequence[SmokeCase], jobs: int, base: Mapping[str, str], evidence: Path
) -> dict[str, str]:
    locks = {case.resource: threading.Lock() for case in cases if case.resource}
    printer = threading.Lock()

    def one(case: SmokeCase) -> tuple[str, str]:
        lock = locks.get(case.resource) if case.resource else None
        with lock or threading.Lock():
            outcome = run_case(case, base, evidence)
        with printer:
            print(f"{outcome:>16}  {case.suite}", flush=True)
        return case.node_id, outcome

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        return dict(pool.map(one, cases))


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
            f"The model-surface snapshot is stale; run {UPDATE_COMMAND} first.",
            file=sys.stderr,
        )
        return 2
    cases = [c for c in CASES if not selected or any(s in c.node_id for s in selected)]
    if not cases:
        print("No smoke case matches --case.", file=sys.stderr)
        return 2
    # Agent homes refuse aliased paths, and macOS temp lives under a symlink.
    evidence = Path(tempfile.mkdtemp(prefix="daita-live-smoke-")).resolve()
    print(f"surface {digest}; private evidence in {evidence}", flush=True)
    outcomes = _run_all(cases, jobs, os.environ, evidence)
    failed = {
        node: outcome
        for node, outcome in outcomes.items()
        if outcome not in {CASE_PASSED, CASE_PASSED_ON_RERUN}
    }
    for node, outcome in sorted(failed.items()):
        print(f"FAILED ({outcome}): {node}", file=sys.stderr)
    if failed or selected:
        print("Record not written (failures or a partial --case run).")
        return 1 if failed else 0
    record = {
        "surface_digest": digest,
        "recorded_on": datetime.now(UTC).date().isoformat(),
        "cases": dict(sorted(outcomes.items())),
    }
    RECORD_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
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
    run.add_argument("--jobs", type=int, default=3)
    arguments = parser.parse_args(argv)
    if arguments.command == "list":
        for case in CASES:
            flags = " ".join(
                f"{key}={value}" for key, value in sorted(case.env.items())
            )
            print(f"{case.suite}\n  {case.node_id}\n  {flags}")
        return 0
    return _run(arguments.case, arguments.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
