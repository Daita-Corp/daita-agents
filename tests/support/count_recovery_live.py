"""Bounded, opt-in real-model count campaigns; collection has no external I/O."""

from __future__ import annotations

import inspect
import json
import os
import re
import sqlite3
import subprocess
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, cast
from uuid import uuid4

from daita import Agent, LoopLimits, PostgreSQLSource, SQLiteSource, create_llm_provider
from daita._json import canonical_json
from daita.catalog.models import CatalogSchemaRequest, ResourceKind
from daita.llm._lifecycle import closing_stream
from daita.llm.errors import ModelProviderError
from daita.llm.models import (
    FinishReason,
    ModelRequest,
    ModelResponse,
    ModelStreamCompleted,
    ModelStreamEvent,
    ModelUsage,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.pricing import CostEstimate
from daita.llm.profiles import reviewed_model_profile
from daita.llm.protocols import provider_has_complete_pricing
from daita.loop.models import LoopExitKind, validate_completed_transcript
from daita.observation import PrivateDiagnosticSink
from daita.security import EnvironmentSecretProvider, SecretReference
from daita.storage.sqlite_codecs.transcripts import encode_loop_exit
from tests.support.count_recovery import (
    FIXTURE_DIGEST,
    MANIFEST,
    TOTALS,
    WINNER,
    quote,
    seed_sqlite,
)
from tests.support.job_benchmarks import RecordingProvider, live_provider
from tests.support.mcp_read_harness import ReadBudget, write_private_report
from tests.support.paths import REPO_ROOT

AUTHORIZATION = "DAITA_RUN_LIVE_COUNT_RECOVERY"
REPORT_ENV = "DAITA_COUNT_RECOVERY_REPORT_DIR"
REPEATS_ENV = "DAITA_COUNT_RECOVERY_REPEATS"
TOTAL_COST_ENV = "DAITA_COUNT_RECOVERY_TOTAL_COST_USD"
POSTGRES_GATE = "DAITA_RUN_COUNT_RECOVERY_POSTGRES"
CODEX_GATE = "DAITA_RUN_COUNT_RECOVERY_CODEX"
CURSOR_GATE = "DAITA_COUNT_RECOVERY_REQUIRE_CURSORS"
OPERATION_GATE = "DAITA_COUNT_RECOVERY_REQUIRE_OPERATIONS"
PARAPHRASES = (
    "Tell me which database has the most records in it.",
    "Compare the total number of rows in each available database and identify the larger one.",
    "Which of my databases holds more records? Give both totals and the winner.",
)
CASES = ("L1-a", "L1-b", "L1-c", "L2", "L3", "L4", "L5", "L6", "L7")


def score_operation_usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    requested = [r for r in records if r["call"]["name"] == "data_count_selection"]
    successful = [
        r
        for r in requested
        if isinstance(r.get("result"), dict)
        and r["result"].get("kind") == "data.count_operation"
    ]
    referenced = [r for r in successful if "selection_ref" in r["call"]["arguments"]]
    return {
        "requested": len(requested),
        "successful": len(successful),
        "successful_selection_references": len(referenced),
        "completed": sum(
            r["result"]["data"]["status"] == "completed" for r in successful
        ),
        "legacy_count_calls": sum(
            r["call"]["name"] == "data_count_rows" for r in records
        ),
        "adopted": bool(referenced) and len(referenced) == len(successful),
    }


def score_cursor_usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate requested handles from authenticated successful count results."""
    starts = []
    continuations = []
    successful_starts = []
    successful_continuations = []
    failures = []
    completed = []
    legacy = []
    for record in records:
        call = record["call"]
        if call["name"] != "data_count_rows":
            continue
        arguments = call["arguments"]
        result = record.get("result") or {}
        data = result.get("data") or {}
        if "selection_ref" in arguments:
            starts.append(call["id"])
            if data.get("selection_ref") == arguments["selection_ref"] and data.get(
                "progress"
            ):
                successful_starts.append(call["id"])
        elif "cursor" in arguments:
            continuations.append(call["id"])
            if data.get("selection_ref") and data.get("progress"):
                successful_continuations.append(call["id"])
        else:
            legacy.append(call["id"])
        if result.get("error") and (
            "selection_ref" in arguments or "cursor" in arguments
        ):
            failures.append({"call_id": call["id"], "code": result["error"]["code"]})
        if (
            data.get("progress")
            and data.get("next_cursor") is None
            and data["progress"]["remaining_resources"] == 0
        ):
            completed.append(
                {
                    "selection_ref": data["selection_ref"],
                    "source_id": data["source_id"],
                    "progress": data["progress"],
                }
            )
    return {
        "requested_selection_calls": starts,
        "requested_cursor_calls": continuations,
        "successful_selection_calls": successful_starts,
        "successful_cursor_calls": successful_continuations,
        "reference_failures": failures,
        "completed_selections": completed,
        "legacy_count_calls": legacy,
        "adopted": bool(successful_starts and successful_continuations),
    }


def repetitions() -> range:
    value = int(os.environ.get(REPEATS_ENV, "3"))
    if not 1 <= value <= 5:
        raise ValueError(f"{REPEATS_ENV} must be between 1 and 5")
    return range(value)


_ACTIVE_REPORT_DIRS: set[Path] = set()


def report_directory(current_report: Path) -> Path:
    raw = os.environ.get(REPORT_ENV)
    if not raw:
        raise ValueError(f"{REPORT_ENV} must name a fresh absolute private directory")
    path = Path(raw)
    if not path.is_absolute() or path.resolve().is_relative_to(REPO_ROOT.resolve()):
        raise ValueError(
            "count reports require an absolute path outside the repository"
        )
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.stat().st_mode & 0o077:
        raise ValueError("count report directory must have mode 0700")
    if path not in _ACTIVE_REPORT_DIRS:
        if any(item != current_report for item in path.iterdir()):
            raise ValueError("A fresh report directory is required for each campaign")
        _ACTIVE_REPORT_DIRS.add(path)
    return path


class BoundedContinuationProvider:
    """A recorded synthetic prefix followed by the actual priced/subscription provider.

    The API allowance decorator only narrows canonical paid admission. Graph-disabled
    host limits therefore do not remove paid-provider admission. Prefix responses are
    synthetic and explicitly excluded from unforced model-choice/efficiency evidence.
    """

    def __init__(self, delegate: RecordingProvider, allowance: Decimal | None):
        self.delegate = delegate
        self.allowance = allowance
        self.spent = Decimal(0)
        self.prefix: list[ModelResponse | Callable[[ModelRequest], ModelResponse]] = []
        self.seeded_requests: list[ModelRequest] = []
        self.narrowed: list[dict[str, object]] = []
        self.before_prefix: dict[int, Callable[[], Any]] = {}

    @property
    def provider_id(self):
        return self.delegate.provider_id

    def has_complete_pricing(self, request):
        return provider_has_complete_pricing(self.delegate, request)

    def supports_request_policy(self, request):
        return self.delegate.supports_request_policy(request)

    def _request(self, request: ModelRequest) -> ModelRequest:
        if self.allowance is None:
            return request
        remaining = self.allowance - self.spent
        if remaining <= 0:
            raise ValueError("paid continuation allowance exhausted before dispatch")
        ceiling = (
            min(remaining, request.max_estimated_cost_usd)
            if request.max_estimated_cost_usd is not None
            else remaining
        )
        self.narrowed.append(
            {
                "original_cost_ceiling": (
                    None
                    if request.max_estimated_cost_usd is None
                    else str(request.max_estimated_cost_usd)
                ),
                "narrowed_cost_ceiling": str(ceiling),
            }
        )
        return replace(request, max_estimated_cost_usd=ceiling)

    def _charge(self, usage: ModelUsage):
        if self.allowance is not None:
            if usage.cost_estimate.status.value != "complete":
                self.spent = max(self.spent, self.allowance)
            elif usage.cost_estimate.amount_usd is not None:
                self.spent += usage.cost_estimate.amount_usd

    async def _seed(self, request):
        if not self.prefix:
            return None
        index = len(self.seeded_requests)
        callback = self.before_prefix.get(index)
        if callback:
            value = callback()
            if inspect.isawaitable(value):
                await value
        self.seeded_requests.append(request)
        response = self.prefix.pop(0)
        return response(request) if callable(response) else response

    async def generate(self, request):
        seeded = await self._seed(request)
        if seeded is not None:
            return seeded
        try:
            response = await self.delegate.generate(self._request(request))
        except ModelProviderError as error:
            self._charge(error.usage)
            raise
        self._charge(response.usage)
        return response

    async def stream(self, request) -> AsyncIterator[ModelStreamEvent]:
        seeded = await self._seed(request)
        if seeded is not None:
            yield ModelStreamCompleted(seeded)
            return
        try:
            async with closing_stream(
                self.delegate.stream(self._request(request))
            ) as events:
                async for event in events:
                    if isinstance(event, ModelStreamCompleted):
                        self._charge(event.response.usage)
                    yield event
        except ModelProviderError as error:
            self._charge(error.usage)
            raise


def seeded(*calls: ToolCall) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=calls,
        usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal(0))),
        provider_id="synthetic:count_fault_prefix",
    )


def score_totals(answer: str | None, expected: dict[str, int]) -> dict[str, object]:
    """Conservative numerical pairing oracle; ambiguous parsing is unqualified."""
    text = (answer or "").replace("**", "").replace("`", "")
    table_totals: dict[str, list[int]] = {name: [] for name in expected}
    prose: list[str] = []
    total_column: int | None = None
    for line in text.splitlines():
        if "|" not in line:
            prose.append(line)
            total_column = None
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        headers = [
            index
            for index, cell in enumerate(cells)
            if re.search(r"\b(?:rows?|records?)\b", cell, re.I)
            and not re.search(r"\b(?:difference|delta|margin)\b", cell, re.I)
        ]
        if headers:
            total_column = headers[0] if len(headers) == 1 else None
            continue
        if all(re.fullmatch(r"[-: ]+", cell) for cell in cells):
            continue
        for name in expected:
            if not any(re.fullmatch(re.escape(name), cell, re.I) for cell in cells):
                continue
            index = total_column
            if index is None and len(cells) == 2:
                index = 1
            if index is not None and index < len(cells):
                value = re.fullmatch(
                    r"(\d[\d,]*)(?:\s+(?:rows|records))?", cells[index], re.I
                )
                if value:
                    table_totals[name].append(int(value[1].replace(",", "")))
    prose_text = "\n".join(prose)
    pairs: dict[str, list[int]] = {}
    for name in expected:
        matches = []
        source_pattern = r"\b(?:" + "|".join(re.escape(n) for n in expected) + r")\b"
        for line in prose_text.splitlines():
            labels = list(re.finditer(source_pattern, line, re.I))
            for index, label in enumerate(labels):
                if label[0].casefold() != name.casefold():
                    continue
                segment = line[
                    label.end() : (
                        labels[index + 1].start()
                        if index + 1 < len(labels)
                        else len(line)
                    )
                ]
                labelled = [
                    match[1]
                    for match in re.finditer(
                        r"(?<![\w])(\d[\d,]*)\s+(?:total\s+)?(?:rows|records)\b",
                        segment,
                        re.I,
                    )
                    if not re.search(
                        r"\b(?:by|difference|delta|margin|ahead|additional|subtotal|snapshots?)\b|\+",
                        segment[max(0, match.start() - 32) : match.start()],
                        re.I,
                    )
                ]
                direct = re.match(
                    r"\s*[:=\-]\s*(\d[\d,]*)(?=\s*(?:$|[.;]|(?:total\s+)?(?:rows|records)\b))",
                    segment,
                    re.I,
                )
                matches.extend(labelled or ([direct[1]] if direct else []))
        pairs[name] = list(
            dict.fromkeys(
                [
                    *table_totals[name],
                    *(int(value.replace(",", "")) for value in matches),
                ]
            )
        )
    numerical = all(pairs[name] == [total] for name, total in expected.items())
    winner = max(expected, key=lambda name: expected[name])
    winner_claim = any(
        re.search(pattern, text, re.I)
        for pattern in (
            rf"{winner}[^.\n]{{0,80}}(?:most|larger|highest|winner|more records|more rows)",
            rf"(?:most|larger|highest|winner)[^.\n]{{0,80}}{winner}",
        )
    )
    return {
        "parsed_totals": pairs,
        "numerical_correct": numerical,
        "winner_correct": bool(winner_claim),
        "ambiguous": any(len(v) != 1 for v in pairs.values()),
    }


def score_filtered_count(answer: str | None, expected: int) -> bool:
    text = (answer or "").replace("**", "").replace("`", "")
    values = re.findall(
        r"(?<![\w])(\d[\d,]*)\s+(?:matching\s+)?(?:rows|records)\b", text, re.I
    )
    values += re.findall(r"\b(?:filtered\s+)?count\s*[:=]\s*(\d[\d,]*)", text, re.I)
    if not values:
        text = re.sub(r"control_value\s*=\s*1\b", "", text, flags=re.I)
        values = re.findall(r"(?<![\w])\d[\d,]*", text)
    return {int(value.replace(",", "")) for value in values} == {expected}


def score_partial_totals(
    answer: str | None,
    observed_totals: dict[str, int],
    complete_sources: set[str],
    *,
    observed_resource_counts: dict[str, int] | None = None,
) -> dict[str, object]:
    """Verify lower-bound claims against measured evidence, never fixture totals."""
    text = answer or ""
    normalized = re.sub(r"\bat least\s+|≥\s*", "", text, flags=re.I)
    normalized = re.sub(r"\bbase[- ]table rows\b", "rows", normalized, flags=re.I)
    normalized = re.sub(r"\bsubtotal\b", "", normalized, flags=re.I)
    score = score_totals(normalized, observed_totals)
    acknowledges = bool(
        re.search(
            r"incomplete|partial|cannot|unable|remain(?:ing)?|lower bound|at least|counted so far|not (?:yet )?(?:available|completed)|≥",
            text,
            re.I,
        )
    )
    winner = max(observed_totals, key=lambda name: observed_totals[name])
    conclusive = all(
        name == winner or (name in complete_sources and observed_totals[winner] > total)
        for name, total in observed_totals.items()
    )
    claims_winner = bool(
        re.search(r"winner|most records|most rows|larger|highest", text, re.I)
    )
    score["honest_partial"] = (
        acknowledges
        and score["numerical_correct"]
        and not score["ambiguous"]
        and (not claims_winner or (conclusive and score["winner_correct"]))
    )
    score["winner_established_by_evidence"] = conclusive
    no_observations = (
        observed_resource_counts is not None
        and set(observed_resource_counts) == set(observed_totals)
        and all(value == 0 for value in observed_resource_counts.values())
        and not complete_sources
    )
    claims_any_winner = any(
        score_totals(text, {name: 0})["winner_correct"] for name in observed_totals
    )
    score["honest_no_observations"] = bool(
        no_observations
        and re.search(
            r"no\s+(?:table\s+)?(?:rows|counts|counting|observations)|not\s+(?:run|executed|counted)|before\s+(?:execution|dispatch)|did\s+not\s+(?:run|execute)",
            text,
            re.I,
        )
        and all(
            not values
            for values in cast(dict[str, list[int]], score["parsed_totals"]).values()
        )
        and not claims_any_winner
    )
    if no_observations:
        score["honest_partial"] = False
        score["winner_established_by_evidence"] = False
    return score


def summarize_campaign(
    reports: list[dict[str, Any]], *, expected_cases: int
) -> dict[str, object]:
    natural = [r for r in reports if r.get("case", "").startswith("L1")]
    measured = bool(natural) and all(
        r.get("usage_complete")
        and isinstance(r.get("tokens"), int)
        and isinstance(r.get("logical_requests"), int)
        for r in natural
    )
    tokens = [
        r["tokens"]
        for r in natural
        if r.get("usage_complete") and isinstance(r.get("tokens"), int)
    ]
    logical = [r["logical_requests"] for r in natural if "logical_requests" in r]
    complete = len(reports) == expected_cases and all(
        r.get("status") == "passed"
        and r.get("provider_kind") == "real"
        and r.get("usage_complete")
        and (r.get("case") in {"L2", "L6"} or r.get("coverage_correct") is True)
        for r in reports
    )
    if expected_cases % len(CASES) == 0:
        expected_repeats = expected_cases // len(CASES)
        complete = complete and all(
            {r.get("repetition") for r in reports if r.get("case") == case}
            == set(range(expected_repeats))
            for case in CASES
        )
    efficiency = (
        measured
        and bool(tokens)
        and median(tokens) <= 50_000
        and max(tokens) <= 100_000
        and median(logical) <= 8
    )
    return {
        "expected_cases": expected_cases,
        "recorded_cases": len(reports),
        "functional_qualified": complete,
        "efficiency_qualified": complete and efficiency,
        "usage_complete": measured,
        "median_tokens": median(tokens) if tokens else None,
        "max_tokens": max(tokens) if tokens else None,
        "median_logical_requests": median(logical) if logical else None,
        "live_qualified": complete and efficiency,
    }


def reserved_codex_tokens(report: dict[str, Any]) -> int:
    measured = int(report.get("tokens") or 0)
    return measured if report.get("usage_complete") else max(measured, 100_000)


async def postgres_sources(agent, prefix: str):
    """Dedicated loopback disposable DB only; read role must already exist."""
    import asyncpg  # type: ignore[import-untyped]

    password = os.environ["DAITA_COUNT_RECOVERY_POSTGRES_ADMIN_PASSWORD"]
    os.environ["DAITA_COUNT_RECOVERY_POSTGRES_READ_PASSWORD"]
    port = int(os.environ["DAITA_COUNT_RECOVERY_POSTGRES_PORT"])
    admin = await asyncpg.connect(
        host="127.0.0.1",
        port=port,
        database="daita_count_recovery_fixture",
        user="postgres",
        password=password,
        timeout=5,
        command_timeout=10,
    )
    schemas = []
    sources = {}
    try:
        assert (
            await admin.fetchval("SELECT current_database()")
            == "daita_count_recovery_fixture"
        )
        assert (
            await admin.fetchval(
                "SELECT NOT rolsuper FROM pg_roles WHERE rolname='daita_count_reader'"
            )
            is True
        )
        for ordinal, (name, tables) in enumerate(MANIFEST.items()):
            schema = f"count_{prefix}_{ordinal}"
            schemas.append(schema)
            await admin.execute(f"CREATE SCHEMA {quote(schema)}")
            for table in tables:
                identifier = quote(schema) + "." + quote(str(table["table"]))
                columns = ["record_id", "control_value"] + [
                    f"analytical_dimension_{i}"
                    for i in range(int(table["columns"]) - 2)
                ]
                await admin.execute(
                    f"CREATE TABLE {identifier} ("
                    + ",".join(quote(c) + " INTEGER" for c in columns)
                    + ")"
                )
                await admin.executemany(
                    f"INSERT INTO {identifier} (record_id, control_value) VALUES ($1,$2)",
                    [(i, i % 3) for i in range(int(table["rows"]))],
                )
                assert (
                    await admin.fetchval(f"SELECT count(*) FROM {identifier}")
                    == table["rows"]
                )
            await admin.execute(
                f"CREATE VIEW {quote(schema)}.distractor_view AS SELECT * FROM {quote(schema)}.records_02"
            )
            await admin.execute(
                f"GRANT USAGE ON SCHEMA {quote(schema)} TO daita_count_reader"
            )
            await admin.execute(
                f"GRANT SELECT ON ALL TABLES IN SCHEMA {quote(schema)} TO daita_count_reader"
            )
            for table in tables:
                identifier = schema + "." + quote(str(table["table"]))
                assert not await admin.fetchval(
                    "SELECT has_table_privilege('daita_count_reader', $1, 'INSERT,UPDATE,DELETE')",
                    identifier,
                )
            sources[name] = await agent.attach(
                PostgreSQLSource(
                    "127.0.0.1",
                    "daita_count_recovery_fixture",
                    "daita_count_reader",
                    credential=SecretReference.environment(
                        "DAITA_COUNT_RECOVERY_POSTGRES_READ_PASSWORD"
                    ),
                    port=port,
                    schemas=(schema,),
                    ssl_mode="disable",
                    name=name,
                    secret_provider=EnvironmentSecretProvider(),
                )
            )
    except BaseException:
        for schema in schemas:
            await admin.execute(f"DROP SCHEMA IF EXISTS {quote(schema)} CASCADE")
        await admin.close()
        raise
    return sources, admin, schemas


async def run_case(
    tmp_path: Path,
    *,
    case: str,
    model_id: str,
    repetition: int,
    transport: str,
    database: str,
    budget: ReadBudget | None,
    reports: list[dict[str, Any]],
    node: Any,
) -> None:
    uid = uuid4().hex
    # Bootstrap evidence is retained even if required configuration/setup fails.
    directory = Path(
        os.environ.get(
            REPORT_ENV, f"/private/tmp/daita-count-recovery-unconfigured-{uid}"
        )
    )
    if not directory.is_absolute() or directory.resolve().is_relative_to(
        REPO_ROOT.resolve()
    ):
        directory = Path("/private/tmp") / (
            "daita-count-recovery-invalid-report-" + uid
        )
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / f"{transport}-{database}-{case}-{repetition}-{uid}.json"
    report: dict[str, Any] = {
        "case": case,
        "model_id": model_id,
        "repetition": repetition,
        "transport": transport,
        "database": database,
        "status": "setup_incomplete",
        "fixture_digest": FIXTURE_DIGEST,
        "expected_totals": TOTALS,
        "expected_winner": WINNER,
        "journey": (
            "fault_seeded_real_model_continuation"
            if case in {"L3", "L4", "L6"}
            else "unscripted_real_model"
        ),
        "cleanup": "pending",
        "cursor_required": os.environ.get(CURSOR_GATE) == "1"
        and case not in {"L2", "L6"},
        "operation_required": os.environ.get(OPERATION_GATE) == "1"
        and case not in {"L2", "L6"},
        "attempt_gaps": "Physical retry diagnostics depend on provider boundary; missing is unavailable.",
    }
    node.user_properties.append(("count_recovery_evidence", str(path)))
    write_private_report(path, report)
    agent = None
    recorder = None
    admin = None
    schemas: list[str] = []
    result = None
    reserved = False
    started = perf_counter()
    try:
        directory = report_directory(path)
        if os.environ.get(OPERATION_GATE) == os.environ.get(CURSOR_GATE) == "1":
            raise ValueError(
                "Select either operation or sequential-cursor adoption, not both"
            )
        output_allowance = int(
            os.environ.get("DAITA_COUNT_RECOVERY_OUTPUT_TOKENS", "1536")
        )
        if not 256 <= output_allowance <= 8192:
            raise ValueError(
                "DAITA_COUNT_RECOVERY_OUTPUT_TOKENS must be between 256 and 8192"
            )
        if os.environ.get(AUTHORIZATION if transport == "api" else CODEX_GATE) != "1":
            raise ValueError("explicit transport authorization is required")
        if database == "postgresql" and os.environ.get(POSTGRES_GATE) != "1":
            raise ValueError("explicit PostgreSQL qualification gate is required")
        head = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                capture_output=True,
                check=True,
            )
            .stdout.decode()
            .strip()
        )
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
        ).stdout
        report.update(
            candidate_commit=head, candidate_diff_sha256=sha256(diff).hexdigest()
        )
        if transport == "api":
            if not os.environ.get(TOTAL_COST_ENV) or not os.environ.get(
                "DAITA_STAGE_B_BENCHMARK_MAX_COST_USD"
            ):
                raise ValueError("explicit total and per-run paid budgets are required")
            assert budget is not None
            budget.reserve()
            reserved = True
            profile, recorder = live_provider(
                model_id, max_output_tokens=output_allowance
            )
            allowance = budget.per_run
        else:
            if not model_id.startswith("codex:"):
                raise ValueError(
                    "Codex qualification must use the actual codex transport"
                )
            codex_profile = reviewed_model_profile(model_id)
            if codex_profile is None or not codex_profile.supports_tools:
                raise ValueError(
                    "Codex cohort requires an explicitly reviewed tool-capable profile"
                )
            profile = codex_profile
            prior = [r for r in reports if r["transport"] == "codex"]
            provider_failures = ReadBudget.provider_failure_codes
            if len(prior) >= 2 and all(
                r.get("terminal_reason") in provider_failures for r in prior[-2:]
            ):
                raise ValueError(
                    "Codex campaign stopped after two consecutive provider failures"
                )
            token_budget = int(os.environ["DAITA_COUNT_RECOVERY_CODEX_TOTAL_TOKENS"])
            used = sum(
                reserved_codex_tokens(r) for r in reports if r["transport"] == "codex"
            )
            if token_budget < used + 100_000:
                raise ValueError("Codex aggregate token reservation exhausted")
            recorder = RecordingProvider(
                create_llm_provider(
                    model_id,
                    subscription_credential=os.environ[
                        "DAITA_COUNT_RECOVERY_CODEX_CREDENTIAL"
                    ],
                    max_output_tokens=output_allowance,
                )
            )
            allowance = None
        report["provider_kind"] = (
            "mocked" if recorder.provider_id.startswith("mock:") else "real"
        )
        report["provider_id"] = recorder.provider_id
        provider = BoundedContinuationProvider(recorder, allowance)
        limits = LoopLimits(
            max_steps=4 if case == "L7" else 24,
            max_total_tokens=100_000,
            max_wall_time_seconds=300,
            max_estimated_cost_usd=(
                None if case in {"L3", "L4"} or transport == "codex" else allowance
            ),
        )
        report["settings"] = {
            "steps": limits.max_steps,
            "tokens": 100_000,
            "wall_seconds": 300,
            "output_allowance": output_allowance,
            "paid_ceiling": str(allowance) if allowance is not None else None,
        }
        mixed_fault = (
            case == "L4"
            and os.environ.get("DAITA_COUNT_RECOVERY_MIXED_SELECTOR_FAULT") == "1"
        )
        if mixed_fault:
            limits = replace(limits, max_steps=4)
            report["settings"]["steps"] = 4
        diagnostic_path = directory / f"{uid}.jsonl"
        report["diagnostics_path"] = str(diagnostic_path)
        sink = PrivateDiagnosticSink(diagnostic_path)
        agent = await Agent.create(
            "count-campaign-" + uid[:12],
            root=tmp_path / "homes",
            hosted=True,
            model=provider,
            model_profile=replace(
                profile,
                max_output_tokens=min(profile.max_output_tokens, output_allowance),
            ),
            limits=limits,
            observer=sink,
            secret_provider=EnvironmentSecretProvider(),
        )
        if database == "sqlite":
            paths = seed_sqlite(tmp_path)
            sources = {
                name: await agent.attach(SQLiteSource(p, name=name))
                for name, p in paths.items()
            }
        else:
            sources, admin, schemas = await postgres_sources(agent, uid[:12])
        resources = {
            name: [
                r
                for r in await agent.list_catalog_resources(source_id=s.id)
                if r.kind is ResourceKind.TABLE
            ]
            for name, s in sources.items()
        }
        expected_rows = {
            r.id: next(int(t["rows"]) for t in MANIFEST[name] if t["table"] == r.name)
            for name, items in resources.items()
            for r in items
        }
        full_bytes = compact_bytes = 0
        for name, items in resources.items():
            schema_request = CatalogSchemaRequest(
                agent.id, resource_ids=tuple(r.id for r in items), limit=50
            )
            full_bytes += len(
                canonical_json(
                    await agent._embedded._catalog_service.schema_slice(schema_request)
                ).encode()
            )
            compact_bytes += len(
                canonical_json(
                    await agent._embedded._catalog_service.schema_slice(
                        replace(schema_request, detail="identity")
                    )
                ).encode()
            )
        report["catalog_baseline"] = {
            "full_bytes": full_bytes,
            "identity_bytes": compact_bytes,
            "reduction": 1 - compact_bytes / full_bytes,
        }
        assert compact_bytes <= 0.4 * full_bytes
        scoped = ("Juniper",) if case in {"L5", "L6"} else tuple(sources)
        scope = tuple(sources[name].id for name in scoped)
        report["requested_scope"] = list(scoped)
        report["effective_resources"] = [
            r.id for name in scoped for r in resources[name]
        ]
        ids = tuple(r.id for r in resources["Juniper"])
        source_id = sources["Juniper"].id
        if case in {"L3", "L4"}:
            assert "start_graph_job" not in agent._embedded._capabilities.tool_names
            oversized = lambda call_id: ToolCall(
                call_id,
                "data_query",
                {
                    "source_id": source_id,
                    "resource_ids": ids,
                    "sql": "SELECT count(*) FROM records_00",
                },
            )
            provider.prefix = [seeded(oversized("oversized-1"))]
            if case == "L4":
                provider.prefix.append(seeded(oversized("oversized-2")))
            report["faults"] = ["oversized_pre_dispatch_query"] * len(provider.prefix)
            if mixed_fault:

                def mixed_response(request):
                    references = {
                        block.call_id: cast(Any, block.output["data"])["selection"][
                            "count_selection_ref"
                        ]
                        for message in request.messages
                        for block in message.content
                        if isinstance(block, ToolResultBlock)
                        and block.call_id in sources
                    }
                    return seeded(
                        *(
                            ToolCall(
                                f"oversized-{ordinal}",
                                "data_count_selection",
                                {
                                    "selection_ref": references["Juniper"],
                                    "source_id": source_id,
                                },
                            )
                            for ordinal in (1, 2)
                        )
                    )

                provider.prefix = [
                    seeded(
                        ToolCall(
                            "load-selection",
                            "toolbox_load",
                            {"tool_names": ("data_count_selection",)},
                        ),
                        *(
                            ToolCall(
                                name,
                                "catalog_schema",
                                {
                                    "source_id": source.id,
                                    "detail": "identity",
                                    "limit": 50,
                                },
                            )
                            for name, source in sources.items()
                        ),
                    ),
                    mixed_response,
                ]
                report["faults"] = ["mixed_selection_pre_dispatch"] * 2
        if case == "L6":
            provider.prefix = [
                seeded(
                    ToolCall(
                        "load-count",
                        "toolbox_load",
                        {"tool_names": ("data_count_rows",)},
                    )
                ),
                seeded(
                    ToolCall(
                        "prior-count",
                        "data_count_rows",
                        {"source_id": source_id, "resource_ids": ids[:4]},
                    )
                ),
                seeded(
                    ToolCall(
                        "stale-count",
                        "data_count_rows",
                        {"source_id": source_id, "resource_ids": ids[4:]},
                    )
                ),
            ]
            if database == "sqlite":

                def drift():
                    with sqlite3.connect(paths["Juniper"]) as connection:
                        connection.execute(
                            "ALTER TABLE records_00 ADD COLUMN changed INTEGER"
                        )

                provider.before_prefix[2] = drift
            else:
                assert admin is not None

                async def pg_drift():
                    changed = resources["Juniper"][4].name
                    await admin.execute(
                        f"ALTER TABLE {quote(schemas[0])}.{quote(changed)} ADD COLUMN changed INTEGER"
                    )

                provider.before_prefix[2] = pg_drift
            report["faults"] = ["catalog_structural_drift_before_count"]
        backend_calls: list[dict[str, Any]] = []

        class ObservedBackend:
            def __init__(self, delegate):
                self.delegate = delegate

            async def execute_read(self, **arguments):
                entry = {
                    "source_id": arguments["source_id"],
                    "sql": arguments["sql"],
                    "status": "started",
                }
                backend_calls.append(entry)
                try:
                    value = await self.delegate.execute_read(**arguments)
                    entry.update(status="completed", resource_ids=value.resource_ids)
                    return value
                except BaseException as error:
                    entry.update(
                        status="failed",
                        error_code=getattr(error, "error_code", type(error).__name__),
                    )
                    raise

        registry = agent._embedded._capabilities
        for capability_id in ("data.query", "data.count_rows"):
            _, raw_executor = registry.resolve_execution(capability_id)
            executor = cast(Any, raw_executor)
            owner = executor._query if capability_id == "data.count_rows" else executor
            owner._backends = {
                name: ObservedBackend(backend)
                for name, backend in owner._backends.items()
            }
        report["backend_invocations"] = backend_calls
        if case.startswith("L1"):
            prompt = PARAPHRASES["abc".index(case[-1])]
        elif case == "L2":
            prompt = "In Juniper records_02, how many rows have control_value equal to 1? Use current field evidence and give the exact filtered count."
        elif case == "L6":
            prompt = "Count the readable Juniper base table rows and give its total. Preserve genuine prior evidence and explain any missing coverage."
        else:
            prompt = PARAPHRASES[0]
        if case != "L2":
            prompt += " Sum base table rows once; give each available source total and the winner when comparison is possible. State the admitted scope and any coverage limitation."
        result = await agent.run(prompt, source_scope_ids=scope)
        transcript = await agent.transcript(result.run_id)
        report["result"] = json.loads(encode_loop_exit(result))
        assert all(call["source_id"] in scope for call in backend_calls)
        # Tool pairs contain only synthetic fixture data; opaque provider material is omitted.
        report["calls_results"] = [
            {
                "call": {
                    "id": c.id,
                    "name": c.name,
                    "arguments": json.loads(canonical_json(c.arguments)),
                },
                "result": None if r is None else json.loads(canonical_json(r.output)),
            }
            for c, r in transcript.tool_pairs
        ]
        report["cursor_usage"] = score_cursor_usage(report["calls_results"])
        report["operation_usage"] = score_operation_usage(report["calls_results"])
        real_call_ids = {
            call.id for response in recorder.responses for call in response.tool_calls
        }
        report["mixed_selector_calls"] = {
            "real_model": sum(
                pair["call"]["id"] in real_call_ids
                and pair["call"]["name"] == "data_count_selection"
                and "source_id" in pair["call"]["arguments"]
                and "selection_ref" in pair["call"]["arguments"]
                for pair in report["calls_results"]
            ),
            "fault_seeded": 2 if mixed_fault else 0,
        }
        for request in recorder.requests:
            for definition in request.tools:
                if definition.name == "data_count_selection":
                    assert set(
                        cast(
                            Mapping[str, object], definition.input_schema["properties"]
                        )
                    ) == {
                        "selection_ref",
                        "include_views",
                    }
                assert definition.name != "data_count_plan"
        children = [
            r for r in await agent.analysis_evidence(result.run_id) if r.kind == "child"
        ]
        report["child_observations"] = [
            {"evidence_id": r.evidence_id, "facts": json.loads(canonical_json(r.facts))}
            for r in children
        ]
        report["seeded_logical_requests"] = len(provider.seeded_requests)
        report["logical_requests"] = len(recorder.requests)
        report["narrowed_admission"] = provider.narrowed
        report["provider_timings"] = recorder.timings
        report["tokens"] = result.usage.total_tokens
        report["usage_complete"] = bool(recorder.timings) and all(
            t["usage_complete"] for t in recorder.timings
        )
        report["elapsed_seconds"] = perf_counter() - started
        report["terminal_reason"] = result.reason
        observed: dict[str, int] = {}
        operation_results = {
            c.id: json.loads(canonical_json(r.output["data"]))
            for c, r in transcript.tool_pairs
            if c.name == "data_count_selection" and r is not None and not r.is_error
        }
        for parent_id, summary in operation_results.items():
            related = [r for r in children if r.facts["parent_call_id"] == parent_id]
            measured = {}
            assert set(summary["evidence_call_ids"]) == {r.evidence_id for r in related}
            for record in related:
                facts = json.loads(canonical_json(record.facts))
                assert facts["capability_id"] == "data.count_rows"
                requested_ids = facts["arguments"]["resource_ids"]
                assert 1 <= len(requested_ids) <= 4
                assert all(
                    rid in report["effective_resources"] for rid in requested_ids
                )
                if facts["status"] != "succeeded":
                    continue
                data = facts["result"]["data"]
                assert [row["resource_id"] for row in data["counts"]] == requested_ids
                assert data["total_rows"] == sum(
                    expected_rows[rid] for rid in requested_ids
                )
                for row in data["counts"]:
                    assert row["rows"] == expected_rows[row["resource_id"]]
                    assert row["resource_id"] not in measured
                    measured[row["resource_id"]] = row["rows"]
            assert summary["counted_resources"] == len(measured)
            assert summary["observed_total_rows"] == sum(measured.values())
            assert (
                summary["counted_resources"]
                + summary["failed_resources"]
                + summary["interrupted_resources"]
                + summary["unexecuted_resources"]
                == summary["selected_resources"]
            )
            assert (summary["status"] == "completed") == (
                summary["counted_resources"] == summary["selected_resources"]
            )
            assert summary["total_rows"] == (
                sum(measured.values()) if summary["status"] == "completed" else None
            )
            observed.update(measured)
        count_pages = {
            call.id: json.loads(canonical_json(output.output["data"]))
            for call, output in transcript.tool_pairs
            if call.name == "data_count_rows"
            and output is not None
            and not output.is_error
        }
        catalog_column_payload = 0
        for call, output in transcript.tool_pairs:
            assert call.name != "start_graph_job"
            if call.name == "toolbox_search":
                assert "start_graph_job" not in str(call.arguments.get("query", ""))
            if call.name in {"data_query", "data_count_rows"}:
                requested = cast(
                    tuple[str, ...], call.arguments.get("resource_ids", ())
                )
                if output is not None and not output.is_error:
                    assert all(r in report["effective_resources"] for r in requested)
                    assert len(requested) <= 4
                    data = json.loads(canonical_json(output.output["data"]))
                    if call.name == "data_count_rows":
                        assert data["coverage"] == "complete_requested_batch"
                        resolved = tuple(row["resource_id"] for row in data["counts"])
                        assert (
                            len(resolved)
                            == len(set(resolved))
                            == data["counted_resources"]
                        )
                        assert 1 <= len(resolved) <= 4
                        assert all(
                            resource_id in report["effective_resources"]
                            for resource_id in resolved
                        )
                        assert data["total_rows"] == sum(
                            expected_rows[resource_id] for resource_id in resolved
                        )
                        if "progress" in data:
                            progress = data["progress"]
                            lineage = progress["evidence_call_ids"]
                            assert lineage[-1] == call.id and len(lineage) == len(
                                set(lineage)
                            )
                            rows = [
                                row
                                for key in lineage
                                for row in count_pages[key]["counts"]
                            ]
                            ledger = {row["resource_id"]: row["rows"] for row in rows}
                            assert (
                                len(rows)
                                == len(ledger)
                                == progress["counted_resources"]
                            )
                            assert progress["total_rows"] == sum(
                                expected_rows[key] for key in ledger
                            )
                            assert progress["remaining_resources"] == progress[
                                "selected_resources"
                            ] - len(ledger)
                            assert (data["next_cursor"] is None) == (
                                progress["remaining_resources"] == 0
                            )
                        for row in data["counts"]:
                            assert row["rows"] == expected_rows[row["resource_id"]]
                            observed[row["resource_id"]] = row["rows"]
                    elif (
                        len(requested) == 1
                        and re.search(r"COUNT\(\*\)", str(data["canonical_sql"]), re.I)
                        and not re.search(
                            r"\bWHERE|JOIN|GROUP|HAVING|LIMIT\b",
                            str(data["canonical_sql"]),
                            re.I,
                        )
                    ):
                        values = list(data["rows"][0].values())
                        if values == [expected_rows[requested[0]]]:
                            observed[requested[0]] = values[0]
            if (
                call.name == "catalog_schema"
                and output is not None
                and not output.is_error
            ):
                catalog_column_payload += sum(
                    int(len(r.get("columns", ())))
                    for r in json.loads(canonical_json(output.output["data"]))[
                        "resources"
                    ]
                )
        report["covered_resources"] = sorted(observed)
        report["coverage_correct"] = set(observed) == set(report["effective_resources"])
        report["catalog_columns_returned"] = catalog_column_payload
        report["result_bytes"] = sum(
            len(canonical_json(r.output).encode())
            for _, r in transcript.tool_pairs
            if r is not None
        )
        assert result.kind is LoopExitKind.COMPLETED, result.reason
        validate_completed_transcript(transcript, result)
        if case == "L2":
            filtered_expected = (int(MANIFEST["Juniper"][2]["rows"]) + 1) // 3
            report["score"] = {
                "expected_filtered_count": filtered_expected,
                "correct": score_filtered_count(result.final_text, filtered_expected),
            }
            assert report["score"]["correct"]
            control_id = next(
                r.id for r in resources["Juniper"] if r.name == "records_02"
            )
            control_evidence = []
            for call, output in transcript.tool_pairs:
                if (
                    call.name != "data_query"
                    or output is None
                    or output.is_error
                    or tuple(cast(tuple[str, ...], call.arguments["resource_ids"]))
                    != (control_id,)
                ):
                    continue
                data = json.loads(canonical_json(output.output["data"]))
                if "control_value" not in str(data["canonical_sql"]).lower():
                    continue
                rows = data["rows"]
                if len(rows) == 1 and list(rows[0].values()) == [filtered_expected]:
                    control_evidence.append(call.id)
                elif (
                    not data["truncated"]
                    and rows
                    and all("control_value" in row for row in rows)
                ):
                    if (
                        sum(row["control_value"] == 1 for row in rows)
                        == filtered_expected
                    ):
                        control_evidence.append(call.id)
            report["filtered_evidence_call_ids"] = control_evidence
            assert (
                control_evidence
            ), "Filtered answer needs authenticated field-dependent current evidence"
            assert catalog_column_payload > 0 or any(
                c.name == "catalog_inspect" and r is not None and not r.is_error
                for c, r in transcript.tool_pairs
            )
        elif case == "L6":
            failures = [r for c, r in transcript.tool_pairs if c.id == "stale-count"]
            assert failures and failures[0] is not None and failures[0].is_error
            assert re.search(
                r"missing|incomplete|stale|changed|cannot|unable",
                result.final_text or "",
                re.I,
            )
            assert str(TOTALS["Juniper"]) not in (result.final_text or "")
        else:
            expected = {name: TOTALS[name] for name in scoped}
            score = score_totals(result.final_text, expected)
            report["score"] = score
            incomplete = case == "L7" and not report["coverage_correct"]
            if incomplete:
                observed_totals = {
                    name: sum(
                        observed[r.id] for r in resources[name] if r.id in observed
                    )
                    for name in scoped
                }
                complete_sources = {
                    name
                    for name in scoped
                    if all(r.id in observed for r in resources[name])
                }
                report["partial_score"] = score_partial_totals(
                    result.final_text,
                    observed_totals,
                    complete_sources,
                    observed_resource_counts={
                        name: sum(r.id in observed for r in resources[name])
                        for name in scoped
                    },
                )
                assert (
                    report["partial_score"]["honest_partial"]
                    or report["partial_score"]["honest_no_observations"]
                ), report["partial_score"]
            else:
                assert score["numerical_correct"] and not score["ambiguous"], score
                assert report["coverage_correct"]
                if len(expected) > 1:
                    assert score["winner_correct"]
            if case == "L5":
                assert re.search(
                    r"scope|admitted|only|unavailable|cannot compare",
                    result.final_text or "",
                    re.I,
                )
        if case in {"L3", "L4"}:
            bad = [r for c, r in transcript.tool_pairs if c.id.startswith("oversized")]
            assert len(bad) == (2 if case == "L4" else 1)
            assert all(
                r is not None
                and json.loads(canonical_json(r.output))["error"]["details"][
                    "execution_disposition"
                ]
                == "not_dispatched"
                for r in bad
            )
            if case == "L4":
                assert any(
                    any(
                        call.name
                        in {"data_count_rows", "data_query", "data_count_selection"}
                        and len(
                            cast(
                                tuple[str, ...], call.arguments.get("resource_ids", ())
                            )
                        )
                        <= 4
                        for call in response.tool_calls
                    )
                    or (
                        response.text is not None
                        and re.search(
                            r"cannot|unavailable|blocked|unable", response.text, re.I
                        )
                    )
                    for response in recorder.responses[:2]
                ), "Correction or honest blocker is required within two real model responses"
                assert any(
                    "Repeated permanent failure" in str(request.messages)
                    for request in recorder.requests[:2]
                )
                assert all(
                    not any(
                        c.name == "data_query"
                        and c.arguments.get("resource_ids") == ids
                        for c in response.tool_calls
                    )
                    for response in recorder.responses[:2]
                )
        if case.startswith("L1"):
            assert (
                catalog_column_payload < 257
            ), "Full-column enumeration does not qualify compact discovery"
            assert report[
                "usage_complete"
            ], "Missing measurements cannot qualify efficiency"
            assert result.usage.total_tokens <= 100_000
        assert await agent.list_effects() == ()
        if report["cursor_required"]:
            assert report["cursor_usage"][
                "adopted"
            ], "Successful selection-reference and cursor usage is required; legacy calls do not qualify this interface"
        if report["operation_required"]:
            assert report["operation_usage"][
                "adopted"
            ], "Successful bounded operations using selection references are required"
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", failure_type=type(error).__name__)
        raise
    finally:
        errors = []
        try:
            if agent is not None:
                await agent.close()
        except BaseException as error:
            errors.append(type(error).__name__)
        try:
            if recorder is not None:
                await recorder.close()
        except BaseException as error:
            errors.append(type(error).__name__)
        if admin is not None:
            try:
                for schema in schemas:
                    await admin.execute(f"DROP SCHEMA {quote(schema)} CASCADE")
                await admin.close()
            except BaseException as error:
                errors.append(type(error).__name__)
        if reserved:
            assert budget is not None
            budget.settle(result)
            report["budget"] = {
                "total_usd": str(budget.total),
                "charged_usd": str(budget.charged),
                "incomplete_runs": budget.incomplete_runs,
                "stop_reason": budget.stop_reason,
            }
        report["cleanup"] = "complete" if not errors else errors
        if errors:
            report["status"] = "failed"
        report.setdefault("elapsed_seconds", perf_counter() - started)
        reports.append(report)
        write_private_report(path, report)
        if errors:
            raise RuntimeError("Count campaign cleanup failed: " + ", ".join(errors))
