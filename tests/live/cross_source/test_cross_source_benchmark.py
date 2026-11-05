"""Benchmark for questions that span PostgreSQL, SQLite and a folder.

Fixtures live in ``tests/fixtures/cross-source-benchmark`` and hold known answers
plus planted key mismatches (see README.md). The SQLite/folder oracle test runs
offline; the PostgreSQL oracle needs ``DAITA_CROSS_SOURCE_POSTGRES_URL``; the paid
Daita-vs-Claude Code comparison is separately gated. Importing this module
contacts nothing and resolves no credentials.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import re
import shutil
import sqlite3
import stat
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

from daita import (
    Agent,
    AgentEvent,
    FolderSource,
    LoopExitKind,
    LoopLimits,
    SQLiteSource,
)
from daita.evaluation import measure_observer_events
from daita.llm.models import FinishReason, ModelResponse, ToolCall
from daita.llm.providers.mock import MockModelProvider
from daita.security import SecretReference
from tests.support.live_comparison import (
    ANSWER_FORMAT,
    OUTCOMES,
    answer_record,
    claude_code_version,
    normalize,
    run_claude_code,
    system_summary,
)
from tests.support.paths import FIXTURES_ROOT, REPO_ROOT
from tests.support.workspace import workspace_for

_URL = "DAITA_CROSS_SOURCE_POSTGRES_URL"
_LIVE_GATE = "DAITA_RUN_LIVE_CROSS_SOURCE_COMPARISON"
_MODEL_ID = "DAITA_CROSS_SOURCE_MODEL_ID"
_REPEATS = "DAITA_CROSS_SOURCE_REPEATS"
_MAX_COST = "DAITA_CROSS_SOURCE_MAX_COST_USD"
_SYSTEMS = "DAITA_CROSS_SOURCE_SYSTEMS"
_REPORT_DIR = "DAITA_CROSS_SOURCE_REPORT_DIR"
_CLAUDE = "DAITA_CROSS_SOURCE_CLAUDE_EXECUTABLE"
_DATABASE = "daita_cross_source"
_READER = ("daita_cross_source_reader", "daita_cross_source_reader_password")
_FIXTURE = FIXTURES_ROOT / "cross-source-benchmark"
_CREDENTIAL = SecretReference.keychain("fixture:cross-source-benchmark")
_PASS_PERCENT = 90

_needs_postgres = pytest.mark.skipif(
    not os.environ.get(_URL),
    reason=(
        f"set {_URL} to a disposable {_DATABASE} database, for example after "
        "starting tests/fixtures/cross-source-benchmark/compose.yaml"
    ),
)


@dataclass(frozen=True)
class _Target:
    host: str
    port: int
    user: str
    password: str
    ssl_mode: str


def _target() -> _Target:
    url = urlsplit(os.environ[_URL])
    if url.scheme not in {"postgres", "postgresql"} or url.path != f"/{_DATABASE}":
        pytest.fail(f"{_URL} must be a postgresql:// URL for database {_DATABASE}")
    return _Target(
        host=url.hostname or "127.0.0.1",
        port=url.port or 5432,
        user=unquote(url.username or "postgres"),
        password=unquote(url.password or ""),
        ssl_mode=parse_qs(url.query).get("sslmode", ["disable"])[0],
    )


class _Secrets:
    def __init__(self, password: str) -> None:
        self._password = password

    async def resolve(self, reference: SecretReference) -> str:
        assert reference == _CREDENTIAL
        return self._password


class _OracleModel(MockModelProvider):
    """Issue one question's known-correct ``data_query`` call, then stop."""

    def __init__(self) -> None:
        super().__init__((), provider_id="mock:cross-source-oracle")

    def ask(self, arguments: Mapping[str, object]) -> None:
        self._script = (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(id="oracle", name="data_query", arguments=dict(arguments)),
                ),
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="Answered."),
        )
        self._cursor = 0


def _questions() -> list[dict[str, Any]]:
    return json.loads((_FIXTURE / "questions.json").read_text())


def _offline_questions() -> list[dict[str, Any]]:
    return [item for item in _questions() if "postgresql" not in item["sources"]]


def _fixture_sources(root: Path, name: str) -> tuple[Path, Path]:
    """A fresh SQLite CRM and a copy of the folder files under ``root``."""

    database = root / f"{name}-crm.sqlite"
    folder = root / f"{name}-files"
    root.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.executescript((_FIXTURE / "seed_sqlite.sql").read_text())
    shutil.copytree(_FIXTURE / "files", folder)
    return database, folder


def _read_only(database: Path, folder: Path) -> None:
    """Make Claude Code's copies unwritable; it has sqlite3 and duckdb shells."""

    database.chmod(stat.S_IREAD)
    for item in folder.iterdir():
        item.chmod(stat.S_IREAD)
    folder.chmod(stat.S_IREAD | stat.S_IEXEC)


async def _attach_sources(
    agent: Agent, database: Path, folder: Path, target: _Target | None
) -> None:
    if target is not None:
        await agent.attach_postgresql(
            host=target.host,
            port=target.port,
            database=_DATABASE,
            username=_READER[0],
            credential=_CREDENTIAL,
            schemas=("sales",),
            ssl_mode=target.ssl_mode,
            name="Sales database",
        )
    await agent.attach(SQLiteSource(database, name="CRM database"))
    await agent.attach(FolderSource(folder, name="Business files"))


def _combined_data(evidence: Sequence[tuple[ToolCall, Any]]) -> list[Mapping[str, Any]]:
    """The ``data`` of every successful ``data_query`` result, in call order."""

    return [
        block.output["data"]
        for call, block in evidence
        if call.name == "data_query"
        and block is not None
        and not block.is_error
        and isinstance(block.output.get("data"), Mapping)
    ]


def _first_value(rows: object) -> str | None:
    if not isinstance(rows, Sequence) or not rows:
        return None
    row = rows[0]
    if isinstance(row, Mapping):
        return normalize(next(iter(row.values()))) if row else None
    if isinstance(row, Sequence) and not isinstance(row, str) and row:
        return normalize(row[0])
    return None


def _joined_checks(results: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        check
        for data in results
        for check in data.get("join_checks") or ()
        if isinstance(check, Mapping)
    ]


def _reported_in_result(
    results: Sequence[Mapping[str, Any]], mismatch: Mapping[str, Any]
) -> bool:
    """A join check shows the planted counts (key or row level, either side
    first), shows the planted type difference, or says it could not measure."""

    planted = {
        tuple(sorted((item["left_unmatched"], item["right_unmatched"])))
        for item in (mismatch, mismatch.get("row_level") or mismatch)
    }
    for check in _joined_checks(results):
        left, right = check.get("left_unmatched"), check.get("right_unmatched")
        if left is None or right is None:
            return True  # The source could not supply the counts.
        if tuple(sorted((left, right))) in planted:
            return True
        if mismatch.get("near_matches") and any(
            (check.get(key) or 0) > 0
            for key in ("left_near_matches", "right_near_matches")
        ):
            return True
        types = {str(check.get("left_type")), str(check.get("right_type"))}
        if mismatch.get("types_differ") and len(types) == 2:
            return True
    return False


def _reported_in_answer(text: str, mismatch: Mapping[str, Any], outcome: str) -> bool:
    """The text mentions the plant, or the question's answer is the plant itself
    (``answer_is_mismatch``: q08 and q11 ask for the unmatched count) and is correct."""

    if mismatch.get("answer_is_mismatch") and outcome == "correct":
        return True
    return any(re.search(item, text, re.I) for item in mismatch["mentions"])


def _scrub(text: str, *secrets: str) -> str:
    for secret in secrets:
        text = text.replace(secret, "***") if secret else text
    return text


async def _ask_oracle(
    agent: Agent, model: _OracleModel, question: Mapping[str, Any]
) -> dict[str, Any]:
    model.ask(question["oracle"])
    result = await agent.run(question["question"])
    evidence = await agent.tool_evidence(result.run_id)
    results = _combined_data(evidence)
    errors = [
        block.output.get("error")
        for call, block in evidence
        if call.name == "data_query" and block is not None and block.is_error
    ]
    answer = _first_value(results[-1].get("rows")) if results else None
    mismatch = question.get("mismatch")
    return {
        "id": question["id"],
        "category": question["category"],
        "expected": question["expected"],
        "answer": answer,
        "outcome": "correct" if answer == question["expected"] else "wrong",
        "errors": errors,
        "join_checks": _joined_checks(results),
        "warnings": [item for data in results for item in data.get("warnings") or ()],
        "read_consistency": [data.get("read_consistency") for data in results],
        "reported_in_result": (
            _reported_in_result(results, mismatch) if mismatch else None
        ),
    }


def _oracle_problems(runs: Sequence[Mapping[str, Any]]) -> list[str]:
    problems = []
    for run in runs:
        if run["outcome"] != "correct":
            problems.append(
                f"{run['id']}: answer {run['answer']} errors {run['errors']}"
            )
        if run["reported_in_result"] is False:
            problems.append(f"{run['id']}: join_checks {run['join_checks']}")
        if run["reported_in_result"] and not run["warnings"]:
            problems.append(f"{run['id']}: no warnings for a planted mismatch")
    return problems


async def test_sqlite_and_folder_oracle_answers_are_correct(tmp_path: Path) -> None:
    questions = _offline_questions()
    assert len(questions) >= 4
    assert sum(1 for item in questions if "mismatch" in item) >= 2
    database, folder = _fixture_sources(tmp_path / "data", "oracle")
    model = _OracleModel()
    async with await Agent.create(
        "cross-source-oracle",
        root=tmp_path / "home",
        model=model,
        model_profile=model.model_profile,
        workspace=workspace_for(tmp_path / "home"),
    ) as agent:
        await _attach_sources(agent, database, folder, None)
        runs = [await _ask_oracle(agent, model, item) for item in questions]
    assert not _oracle_problems(runs), runs


# --- reports ---------------------------------------------------------------


def _report_directory() -> Path:
    configured = os.environ.get(_REPORT_DIR)
    path = (
        Path(configured).expanduser()
        if configured
        else REPO_ROOT / "live-evidence" / "cross-source"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _revision() -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.run(
            ("git", *args), cwd=REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "--short", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--", "src")),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
    }


def _write_report(
    report: Mapping[str, Any], markdown: str, stem: str, *secrets: str
) -> None:
    directory = _report_directory()
    encoded = json.dumps(report, indent=2, sort_keys=True, default=str)
    (directory / f"{stem}.json").write_text(_scrub(encoded, *secrets) + "\n")
    (directory / f"{stem}.md").write_text(_scrub(markdown, *secrets))
    print(f"\nreport written to {directory / stem}.md and .json")


async def _seed(target: _Target) -> Any:
    import asyncpg  # type: ignore[import-untyped]

    admin = await asyncpg.connect(
        host=target.host,
        port=target.port,
        user=target.user,
        password=target.password,
        database=_DATABASE,
        ssl=False if target.ssl_mode == "disable" else target.ssl_mode,
        timeout=10,
        command_timeout=120,
    )
    # One implicit transaction per blank-line chunk keeps DDL under the lock table.
    for chunk in re.split(r"\n\s*\n", (_FIXTURE / "seed.sql").read_text()):
        if re.sub(r"--[^\n]*", "", chunk).strip():
            await admin.execute(chunk)
    return admin


@_needs_postgres
@pytest.mark.integration
@pytest.mark.requires_db
async def test_full_oracle(tmp_path: Path) -> None:
    target = _target()
    admin = await _seed(target)
    await admin.close()
    questions = _questions()
    database, folder = _fixture_sources(tmp_path / "data", "oracle")
    model = _OracleModel()
    async with await Agent.create(
        "cross-source-full-oracle",
        root=tmp_path / "home",
        model=model,
        model_profile=model.model_profile,
        secret_provider=_Secrets(_READER[1]),
        workspace=workspace_for(tmp_path / "home"),
    ) as agent:
        await _attach_sources(agent, database, folder, target)
        runs = [await _ask_oracle(agent, model, item) for item in questions]
    correct = sum(1 for item in runs if item["outcome"] == "correct")
    report = {"environment": _revision(), "questions": runs, "correct": correct}
    lines = [
        "# Cross-source benchmark — oracle",
        "",
        "| Question | Outcome | Answer | Expected | Mismatch in join_checks |",
        "| --- | --- | ---: | ---: | --- |",
        *(
            f"| {r['id']} | {r['outcome']} | {r['answer']} | {r['expected']} | "
            f"{r['reported_in_result']} |"
            for r in runs
        ),
    ]
    _write_report(
        report,
        "\n".join(lines) + "\n",
        "cross-source-oracle",
        *_READER,
        target.password,
    )
    assert not _oracle_problems(runs), runs


# --- paid comparison ---------------------------------------------------------


def _calls(evidence: Sequence[tuple[ToolCall, Any]]) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    for call, block in evidence:
        error = None
        if block is None:
            error = "no_result"
        elif block.is_error:
            detail = block.output.get("error")
            error = str(detail.get("code")) if isinstance(detail, Mapping) else "error"
        arguments = call.arguments
        sql = " ; ".join(
            f"[{item.get('alias')} from {item.get('source_id')}] {item.get('sql')}"
            for item in cast(Sequence[object], arguments.get("inputs") or ())
            if isinstance(item, Mapping)
        )
        if sql:
            sql += f" => {arguments.get('sql')}"
        calls.append(
            {
                "tool": call.name,
                "sql": (sql or str(arguments.get("sql") or ""))[:2_000],
                "error": error,
            }
        )
    return calls


async def _run_daita(
    target: _Target,
    model_id: str,
    questions: Sequence[Mapping[str, Any]],
    root: Path,
    entry: dict[str, Any],
    repeat: int,
) -> None:
    """Answer every question with a fresh agent; append results to ``entry``."""

    from tests.support.live_billing import live_provider

    def api_key() -> str:
        value = os.environ.get("ANTHROPIC_API_KEY")
        if not value:
            pytest.fail("ANTHROPIC_API_KEY must be set")
        return value

    profile, provider = live_provider(
        model_id, api_key=api_key, max_output_tokens=4_096
    )
    database, folder = _fixture_sources(root / "data", "daita")
    events: list[AgentEvent] = []
    try:
        # Default configuration and limits: Daita as shipped.
        async with await Agent.create(
            "cross-source-comparison",
            root=root / "home",
            model=provider,
            model_profile=profile,
            secret_provider=_Secrets(_READER[1]),
            workspace=workspace_for(root / "home"),
            observer=events.append,
        ) as agent:
            await _attach_sources(agent, database, folder, target)
            for question in questions:
                before = len(events)
                started = time.perf_counter()
                async with asyncio.timeout(360):
                    result = await agent.run(question["question"] + ANSWER_FORMAT)
                seconds = time.perf_counter() - started
                if result.steps == 0 and result.kind is not LoopExitKind.COMPLETED:
                    pytest.fail(f"Daita could not run: {result.reason}")
                measured = measure_observer_events(tuple(events[before:]))
                evidence = await agent.tool_evidence(result.run_id)
                text = result.final_text or ""
                mismatch = question.get("mismatch")
                estimate = result.usage.cost_estimate
                entry["questions"].append(
                    {
                        "id": question["id"],
                        "category": question["category"],
                        "repeat": repeat,
                        **answer_record(
                            question["expected"],
                            text,
                            result.kind is LoopExitKind.COMPLETED,
                        ),
                        "reported_in_answer": (
                            _reported_in_answer(
                                text,
                                mismatch,
                                str(
                                    answer_record(
                                        question["expected"],
                                        text,
                                        result.kind is LoopExitKind.COMPLETED,
                                    )["outcome"]
                                ),
                            )
                            if mismatch
                            else None
                        ),
                        "reported_in_result": (
                            _reported_in_result(_combined_data(evidence), mismatch)
                            if mismatch
                            else None
                        ),
                        "run_seconds": round(seconds, 3),
                        "turns": result.steps,
                        "tool_calls": measured.tool_calls,
                        "input_tokens": result.usage.input_tokens,
                        "output_tokens": result.usage.output_tokens,
                        "total_tokens": result.usage.total_tokens,
                        "cost_usd": (
                            None
                            if estimate.amount_usd is None
                            or estimate.status.value != "complete"
                            else str(estimate.amount_usd)
                        ),
                        "exit": result.reason,
                        "provider_failure": str(result.provider_failure)[:400],
                        # What was in flight when the run ended: the last events.
                        "last_events": [
                            f"{event.kind.value} {dict(event.data)}"[:400]
                            for event in events[before:][-6:]
                        ],
                        "calls": _calls(evidence),
                    }
                )
    finally:
        await provider.close()


async def _run_claude(
    model: str,
    target: _Target,
    questions: Sequence[Mapping[str, Any]],
    root: Path,
    entry: dict[str, Any],
    repeat: int,
) -> None:
    database, folder = _fixture_sources(root / "data", "claude")
    _read_only(database, folder)
    try:
        await run_claude_code(
            model,
            questions,
            {item["id"]: item["expected"] for item in questions},
            root / "runs",
            entry,
            repeat,
            executable_env=_CLAUDE,
            preamble=(
                "Bash allows only psql, sqlite3, duckdb, ls, cat, head and echo: "
                "run one simple command at a time, with no shell variables, loops "
                "or cd; the PostgreSQL connection settings are already exported. "
                "Three data sources are available, all read-only. A PostgreSQL "
                "database: use psql, its connection settings are already in the "
                "environment (schema sales). A SQLite database file: use sqlite3 on "
                f"{database}. A folder of CSV and JSONL files: {folder}; query them "
                'with duckdb, for example duckdb -c "SELECT * FROM '
                f"read_csv('{folder}/<file>.csv')\". Answer from these sources: "
            ),
            environment={
                "PGHOST": target.host,
                "PGPORT": str(target.port),
                "PGDATABASE": _DATABASE,
                "PGUSER": _READER[0],
                "PGPASSWORD": _READER[1],
                "PGSSLMODE": target.ssl_mode,
            },
            allowed_tools=(
                "Bash(psql:*) Bash(sqlite3:*) Bash(duckdb:*) "
                "Bash(ls:*) Bash(cat:*) Bash(head:*) Bash(echo:*)"
            ),
            max_cost=os.environ.get(_MAX_COST, "0.50"),
        )
    finally:
        # Let pytest remove the read-only copies.
        folder.chmod(stat.S_IRWXU)
        for item in folder.iterdir():
            item.chmod(stat.S_IRUSR | stat.S_IWUSR)
        database.chmod(stat.S_IRUSR | stat.S_IWUSR)
    # Mismatch questions: did the final text mention the planted problem?
    by_id = {item["id"]: item for item in questions}
    for run in entry["questions"]:
        if run["repeat"] == repeat and "reported_in_answer" not in run:
            mismatch = by_id[run["id"]].get("mismatch")
            run["reported_in_answer"] = (
                _reported_in_answer(run["final_text"], mismatch, run["outcome"])
                if mismatch
                else None
            )


def _acceptance(report: Mapping[str, Any]) -> dict[str, Any]:
    systems = report["systems"]
    daita = systems.get("daita", {}).get("questions", [])
    correct = sum(1 for run in daita if run["outcome"] == "correct")
    percent = round(100 * correct / len(daita), 1) if daita else 0.0
    planted = [run for run in daita if run["reported_in_answer"] is not None]
    missed = [
        f"{run['id']}#{run['repeat']}"
        for run in planted
        if not (run["reported_in_result"] or run["reported_in_answer"])
    ]
    expected = len(report["question_ids"]) * report["repeats"]
    baseline = len(systems.get("claude_code", {}).get("questions", []))
    denied = [
        f"{run['id']}#{run['repeat']}"
        for run in systems.get("claude_code", {}).get("questions", [])
        if run.get("ended_on_denial")
    ]
    checks = {
        f"Daita correct at least {_PASS_PERCENT}%": (
            f"{correct}/{len(daita)} = {percent}%",
            bool(daita) and percent >= _PASS_PERCENT,
        ),
        "Every planted mismatch reported in every Daita run (in the result's "
        "join_checks or in the answer)": (
            f"{len(planted) - len(missed)}/{len(planted)}"
            + (f", missed {missed}" if missed else ""),
            bool(planted) and not missed,
        ),
        "Claude Code baseline recorded (psql, sqlite3, duckdb)": (
            f"{baseline}/{expected} runs"
            + (
                f", {len(denied)} ended on a denied tool call {denied}"
                if denied
                else ""
            ),
            baseline == expected and not denied,
        ),
    }
    return {
        name: {"result": result, "passed": passed}
        for name, (result, passed) in checks.items()
    }


def _system_extras(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    planted = [run for run in runs if run.get("reported_in_answer") is not None]
    known = [
        Decimal(str(run["cost_usd"])) for run in runs if run["cost_usd"] is not None
    ]
    return {
        "known_cost_usd": str(sum(known, Decimal(0))),
        "unknown_cost_runs": len(runs) - len(known),
        # Claude Code: denied/errored shell calls; Daita: tool calls with an error code.
        "runs_with_denied_calls": sum(1 for run in runs if run.get("denied_calls")),
        "runs_with_errored_calls": sum(
            1
            for run in runs
            if run.get("errored_calls")
            or any(call["error"] for call in run.get("calls") or ())
        ),
        "runs_ended_on_denial": sum(1 for run in runs if run.get("ended_on_denial")),
        "mismatch_runs": len(planted),
        "mismatch_reported_in_answer": sum(
            1 for run in planted if run["reported_in_answer"]
        ),
        "mismatch_reported_in_result": sum(
            1 for run in planted if run.get("reported_in_result")
        ),
    }


def _comparison_markdown(report: Mapping[str, Any]) -> str:
    meta, systems = report["environment"], report["systems"]
    lines = [
        "# Cross-source benchmark — Daita vs Claude Code",
        "",
        f"Commit `{meta['commit']}` ({meta['branch']}) · Claude Code "
        f"{report.get('claude_code_version')} · {meta['generated_at']} · "
        f"{report['repeats']} repeat(s) · {len(report['question_ids'])} questions",
        "",
        "## Acceptance",
        "",
        *(
            f"- {'PASS' if item['passed'] else 'FAIL'}: {name} — {item['result']}"
            for name, item in report["acceptance"].items()
        ),
        "",
        "| System | Model | Runs | Correct | Confident-wrong | No answer | Incomplete "
        "| Mismatch reported in answer | in result | Mean s | Turns | Tool calls "
        "| Tokens | Known cost USD (runs unpriced) | Runs with denied call "
        "| with errored call | Ended on denial |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, system in systems.items():
        s, o = system["summary"], system["summary"]["outcomes"]
        planted = s["mismatch_runs"]
        lines.append(
            f"| {name} | {system['model']} | {s['runs']} | "
            + " | ".join(str(o[item]) for item in OUTCOMES)
            + f" | {s['mismatch_reported_in_answer']}/{planted} | "
            f"{s['mismatch_reported_in_result']}/{planted} | {s['mean_run_seconds']} | "
            f"{s['total_turns']} | {s['total_tool_calls']} | {s['total_tokens']} | "
            f"{s['known_cost_usd']} ({s['unknown_cost_runs']}) | "
            f"{s['runs_with_denied_calls']} | {s['runs_with_errored_calls']} | "
            f"{s['runs_ended_on_denial']} |"
        )
    names = list(systems)
    lines += [
        "",
        "| Question | Category | "
        + " | ".join(f"{name} correct | {name} other outcomes" for name in names)
        + " |",
        "| --- | --- | " + " | ".join("---: | ---" for _ in names) + " |",
    ]
    for question in report["question_ids"]:
        cells = []
        for name in names:
            runs = [r for r in systems[name]["questions"] if r["id"] == question]
            correct = sum(1 for r in runs if r["outcome"] == "correct")
            others = ", ".join(
                sorted(r["outcome"] for r in runs if r["outcome"] != "correct")
            )
            cells.append(f"{correct}/{len(runs)} | {others or '—'}")
        lines.append(
            f"| {question} | {report['categories'][question]} | "
            + " | ".join(cells)
            + " |"
        )
    lines += ["", "## Run detail", ""]
    for name, system in systems.items():
        lines.append(f"### {name}")
        for run in system["questions"]:
            lines.append(
                f"- `{run['id']}` #{run['repeat']} {run['outcome']}: "
                f"{run['final_text'][-200:].strip()!r}"
            )
            for call in run.get("calls") or ():
                lines.append(f"  - {call['tool']} error={call['error']}: {call['sql']}")
            for event in run.get("last_events") or ():
                if run["exit"] != "completed":
                    lines.append(f"  - last event: {event}")
            for command in run.get("commands") or ():
                lines.append(f"  - shell: {command[:300]}")
        lines.append("")
    lines.append(
        "Scoring reads the requested last line (keyword rule, not a human label). "
        "A mismatch counts as reported in the answer when the final text matches the "
        "question's `mentions`, and in the result when a join check shows the "
        "planted counts or types or could not be measured."
    )
    return "\n".join(lines) + "\n"


@_needs_postgres
@pytest.mark.integration
@pytest.mark.requires_db
@pytest.mark.requires_llm
@pytest.mark.requires_network
@pytest.mark.skipif(
    os.environ.get(_LIVE_GATE) != "1",
    reason=f"set {_LIVE_GATE}=1 to authorize live model cost for the comparison",
)
async def test_daita_and_claude_code_answer_the_question_set(tmp_path: Path) -> None:
    model_id = os.environ.get(_MODEL_ID, "anthropic:claude-sonnet-5")
    provider, _, model = model_id.partition(":")
    if provider != "anthropic":
        pytest.fail(f"{_MODEL_ID} must be an anthropic: model (Claude Code runs it)")
    wanted = {
        item.strip()
        for item in os.environ.get(_SYSTEMS, "daita,claude_code").split(",")
    }
    systems = [name for name in ("daita", "claude_code") if name in wanted]
    if not systems:
        pytest.fail(f"{_SYSTEMS} must list daita and/or claude_code")
    repeats = int(os.environ.get(_REPEATS, "3"))
    if not 1 <= repeats <= 5:
        pytest.fail(f"{_REPEATS} must be from 1 through 5")
    target = _target()
    admin = await _seed(target)
    await admin.close()
    questions = _questions()
    limits = LoopLimits()
    report: dict[str, Any] = {
        "environment": _revision(),
        "claude_code_version": (
            claude_code_version(_CLAUDE) if "claude_code" in systems else None
        ),
        "repeats": repeats,
        "limits": {
            "daita": {
                "max_steps": limits.max_steps,
                "max_total_tokens": limits.max_total_tokens,
                "max_wall_time_seconds": limits.max_wall_time_seconds,
            },
            "claude_code": {"max_budget_usd": os.environ.get(_MAX_COST, "0.50")},
        },
        "question_ids": [item["id"] for item in questions],
        "categories": {item["id"]: item["category"] for item in questions},
        "systems": {
            name: {"model": model_id if name == "daita" else model, "questions": []}
            for name in systems
        },
    }
    try:
        # Repeats outermost so a mid-run failure leaves both systems comparable.
        for repeat in range(repeats):
            for name in systems:
                root = tmp_path / name / f"repeat-{repeat}"
                entry = report["systems"][name]
                if name == "daita":
                    await _run_daita(target, model_id, questions, root, entry, repeat)
                else:
                    await _run_claude(model, target, questions, root, entry, repeat)
    finally:
        # Paid results survive a later failure.
        for entry in report["systems"].values():
            entry["summary"] = system_summary(entry["questions"]) | _system_extras(
                entry["questions"]
            )
        report["acceptance"] = _acceptance(report)
        _write_report(
            report,
            _comparison_markdown(report),
            f"cross-source-comparison-{report['environment']['commit']}-{model}",
            *_READER,
            target.password,
        )
        for name, item in report["acceptance"].items():
            print(f"{'PASS' if item['passed'] else 'FAIL'}: {name} — {item['result']}")
    for name, system in report["systems"].items():
        assert len(system["questions"]) == len(questions) * repeats, name
