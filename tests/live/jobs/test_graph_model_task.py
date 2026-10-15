"""Cost-bounded live smoke through the revision-2 public composition."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
from collections.abc import AsyncIterator
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, cast
from uuid import uuid4

import pytest

from daita import Agent, GraphState, LoopLimits, SQLiteSource, create_llm_provider
from daita._json import canonical_json
from daita.catalog.models import ResourceKind
from daita.domains.data import DATA_QUERY_EVIDENCE_KIND
from daita.jobs.graph.models import AttemptState, TaskExecutionKind, TaskRole
from daita.llm._lifecycle import closing_stream
from daita.llm.models import (
    FinishReason,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.pricing import CostEstimate
from daita.llm.profiles import reviewed_model_profile
from daita.loop.models import LoopExitKind, validate_completed_transcript
from daita.observation import PrivateDiagnosticSink
from daita.storage.home_migrations import CURRENT_HOME_REVISION
from daita.storage.sqlite_codecs.transcripts import encode_loop_exit
from tests.support.count_recovery import FIXTURE_DIGEST, MANIFEST, TOTALS, seed_sqlite
from tests.support.count_recovery_live import BoundedContinuationProvider
from tests.support.job_benchmarks import create_probe_home, live_provider
from tests.support.mcp_read_harness import ReadBudget, write_private_report
from tests.support.paths import REPO_ROOT
from tests.support.workspace import workspace_for

_AUTHORIZATION = "DAITA_RUN_LIVE_GRAPH_TASK"
_MODEL_ID = "DAITA_GRAPH_TASK_LIVE_MODEL_ID"
_MODEL_KEY = "DAITA_GRAPH_TASK_LIVE_LLM_API_KEY"
_MAX_COST = "DAITA_GRAPH_TASK_LIVE_MAX_COST_USD"
_DEFAULT_MODEL_ID = "openai:gpt-5.6-terra"
_API_PROVIDERS = frozenset({"anthropic", "gemini", "grok", "openai"})

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get(_AUTHORIZATION) != "1",
        reason=(
            f"set {_AUTHORIZATION}=1 only after explicitly authorizing one bounded "
            "live production graph smoke"
        ),
    ),
]

_COUNT_REPORT = "DAITA_GRAPH_COUNT_REPORT_DIR"
_COUNT_TOTAL = "DAITA_GRAPH_COUNT_TOTAL_COST_USD"
_COUNT_DIRECTORIES: set[Path] = set()


@pytest.fixture(scope="module")
def count_job_budget():
    # Dedicated qualification must fail on missing budgets, never skip as success.
    if not os.environ.get(_COUNT_TOTAL) or not os.environ.get(_MAX_COST):
        return None
    try:
        return ReadBudget(
            total=Decimal(os.environ[_COUNT_TOTAL]), per_run=_cost_limit()
        )
    except (ValueError, InvalidOperation):
        return None


@pytest.mark.parametrize("scenario", ["complete", "drift"])
async def test_live_count_job_uses_frozen_plan_and_retains_coverage(
    tmp_path: Path, request, scenario: str, count_job_budget, monkeypatch
) -> None:
    drift = scenario == "drift"
    uid = uuid4().hex
    raw = os.environ.get(_COUNT_REPORT)
    directory = Path(raw or f"/private/tmp/daita-graph-count-unconfigured-{uid}")
    if not directory.is_absolute() or directory.resolve().is_relative_to(REPO_ROOT):
        directory = Path(f"/private/tmp/daita-graph-count-invalid-report-{uid}")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / f"count-job-{scenario}-{uid}.json"
    request.node.user_properties.append(("count_job_evidence", str(path)))
    report: dict[str, Any] = {
        "status": "setup_incomplete",
        "case": scenario,
        "journey": (
            "real_model_admission_fault_seeded_framework_execution"
            if drift
            else "unscripted_real_model_admission_deterministic_execution"
        ),
        "fixture_digest": FIXTURE_DIGEST,
        "expected_totals": TOTALS,
        "expected_resources": 46,
        "cleanup": "pending",
        "qualification": "targeted_api_sqlite_durable_count_only",
    }
    write_private_report(path, report)
    agent = recorder = provider = foreground = inspection = None
    reserved = False
    drift_injected = False
    native_reads = []
    started = perf_counter()
    run_ids: set[str] = set()
    try:
        assert raw, f"{_COUNT_REPORT} must name a fresh private report directory"
        assert not directory.stat().st_mode & 0o077
        if directory not in _COUNT_DIRECTORIES:
            assert not any(
                item != path for item in directory.iterdir()
            ), "A fresh campaign directory is required"
            _COUNT_DIRECTORIES.add(directory)
        assert count_job_budget is not None, "Explicit aggregate budget is required"
        count_job_budget.reserve()
        reserved = True
        model_id = os.environ.get(_MODEL_ID, _DEFAULT_MODEL_ID)
        profile, recorder = live_provider(model_id, max_output_tokens=4096)
        report.update(
            model_id=model_id,
            provider_id=recorder.provider_id,
            provider_kind=(
                "mocked" if recorder.provider_id.startswith("mock:") else "real"
            ),
            candidate_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT
            )
            .decode()
            .strip(),
            candidate_diff_sha256=sha256(
                subprocess.check_output(
                    ["git", "diff", "--binary", "HEAD"], cwd=REPO_ROOT
                )
            ).hexdigest(),
            settings={
                "steps": 24,
                "tokens": 100_000,
                "wall_seconds": 300,
                "output_allowance": 4096,
                "whole_case_cost_ceiling": str(count_job_budget.per_run),
            },
        )
        paths = seed_sqlite(tmp_path)
        provider = BoundedContinuationProvider(recorder, count_job_budget.per_run)
        diagnostics = directory / f"{uid}.jsonl"
        report["diagnostics_path"] = str(diagnostics)
        agent = await Agent.create(
            "live-count-job-" + uid[:12],
            root=tmp_path / "homes",
            hosted=True,
            model=provider,
            model_profile=replace(profile, max_output_tokens=4096),
            limits=LoopLimits(
                max_steps=24,
                max_total_tokens=100_000,
                max_wall_time_seconds=300,
                max_estimated_cost_usd=count_job_budget.per_run,
            ),
            observer=PrivateDiagnosticSink(diagnostics),
        )
        sources = {
            name: await agent.attach(SQLiteSource(p, name=name))
            for name, p in paths.items()
        }
        expected_rows = {}
        for name, source in sources.items():
            for resource in await agent.list_catalog_resources(source_id=source.id):
                if resource.kind is ResourceKind.TABLE:
                    expected_rows[resource.id] = next(
                        int(t["rows"])
                        for t in MANIFEST[name]
                        if t["table"] == resource.name
                    )
        report["admitted_resources"] = sorted(expected_rows)
        original_admit = agent._embedded._job_owner.admit

        async def admit(admission):
            nonlocal drift_injected
            job = await original_admit(admission)
            if drift:
                with sqlite3.connect(paths["Meridian"]) as connection:
                    connection.execute(
                        "ALTER TABLE records_02 ADD COLUMN fixture_drift INTEGER"
                    )
                drift_injected = True
            return job

        monkeypatch.setattr(agent._embedded._job_owner, "admit", admit)
        from daita.adapters.sqlite_query import SQLiteQueryBackend

        original_read = SQLiteQueryBackend.execute_read

        async def read(self, **kwargs):
            native_reads.append(
                {"source_id": kwargs["source_id"], "started_at": perf_counter()}
            )
            return await original_read(self, **kwargs)

        monkeypatch.setattr(SQLiteQueryBackend, "execute_read", read)
        foreground = await agent.run(
            "Start exactly one durable job comparing all admitted Juniper and Meridian base-table row counts. "
            "Use their current identity selection references, deadline 300 seconds, and stop after the receipt.",
            source_scope_ids=tuple(s.id for s in sources.values()),
        )
        run_ids.add(foreground.run_id)
        foreground_requests = len(recorder.requests)
        assert foreground.kind is LoopExitKind.COMPLETED
        jobs = await agent.list_jobs()
        assert len(jobs) == 1, "Exactly one durable comparison must be admitted"
        async with asyncio.timeout(120):
            while True:
                inspection = await agent.inspect_job(jobs[0].job_id)
                assert inspection is not None
                if inspection.job.state in {
                    GraphState.SUCCEEDED,
                    GraphState.FAILED,
                    GraphState.CANCELLED,
                    GraphState.NEEDS_ATTENTION,
                }:
                    break
                await asyncio.sleep(0.1)
        assert all(
            t.execution_kind is TaskExecutionKind.INTERNAL_CAPABILITY
            for t in inspection.tasks
        )
        assert len(recorder.requests) == foreground_requests
        assert len(inspection.tasks) == 13
        report["count_worker_model_requests"] = 0
        report["finalizer_model_requests"] = 0
        observed = {}
        for batch_result in inspection.results:
            if batch_result.result_kind == "data.row_counts":
                assert 1 <= len(cast(Any, batch_result.payload["counts"])) <= 4
                for row in cast(Any, batch_result.payload["counts"]):
                    assert row["resource_id"] not in observed
                    assert row["rows"] == expected_rows[row["resource_id"]]
                    observed[row["resource_id"]] = row["rows"]
        report["coverage"] = {
            "observed_resources": len(observed),
            "observed_total_rows": sum(observed.values()),
            "complete": set(observed) == set(expected_rows),
        }
        report["native_reads"] = native_reads
        report["count_progress"] = inspection.count_progress
        transcript = await agent.transcript(foreground.run_id)
        report["admission_attempts"] = [
            {"tool_name": c.name, "is_error": None if r is None else r.is_error}
            for c, r in transcript.tool_pairs
            if c.name == "start_count_comparison"
        ]
        if drift:
            assert drift_injected
            assert inspection.job.state is GraphState.NEEDS_ATTENTION
            assert any(
                c.kind.value == "source_or_contract_drift" for c in inspection.controls
            )
            assert len(observed) < 46
            assert await agent.read_job_result(jobs[0].job_id) is None
            assert (
                await agent._embedded._store.list_graph_deliveries(
                    agent.id, job_id=jobs[0].job_id
                )
                == ()
            )
        else:
            assert inspection.job.state is GraphState.SUCCEEDED
            assert len(native_reads) == 12 and len(observed) == 46
            final_result = await agent.read_job_result(jobs[0].job_id)
            assert final_result is not None
            assert {
                r["source_id"]: r["total_rows"]
                for r in cast(Any, final_result.payload["sources"])
            } == {sources[name].id: total for name, total in TOTALS.items()}
            assert final_result.payload["winner_source_ids"] == (
                sources["Meridian"].id,
            )
            assert len(inspection.delivery_ids) == 1
        assert await agent.list_effects() == ()
        report["status"] = "passed"
    except BaseException as error:
        report.update(
            status="failed",
            failure_type=type(error).__name__,
            failure_message=str(error),
        )
        raise
    finally:
        successful = report["status"] == "passed"
        cleanup_errors = []
        if agent is not None:
            try:
                # A blocked drift case ends here; cleanup never replays the read.
                await agent.close()
                for job in await agent._embedded._store.list_graph_jobs(agent.id):
                    current = await agent._embedded._store.inspect_graph(
                        job.agent_id, job.job_id
                    )
                    if current is not None:
                        inspection = current
                        model_tasks = {
                            task.task_id
                            for task in current.tasks
                            if task.execution_kind is TaskExecutionKind.MODEL
                        }
                        run_ids.update(
                            a.run_id
                            for a in current.attempts
                            if a.task_id in model_tasks
                        )
                        report["job"] = {
                            "job_id": job.job_id,
                            "state": current.job.state.value,
                            "attempts": [
                                {
                                    "attempt_id": a.attempt_id,
                                    "task_id": a.task_id,
                                    "run_id": a.run_id,
                                    "state": a.state.value,
                                }
                                for a in current.attempts
                            ],
                            "accepted_results": [
                                {
                                    "task_id": r.task_id,
                                    "result_kind": r.result_kind,
                                    "summary": r.summary,
                                    "payload": json.loads(canonical_json(r.payload)),
                                }
                                for r in current.results
                            ],
                        }
            except BaseException as error:
                cleanup_errors.append(type(error).__name__)
        # Close drains machine runs before collecting exact terminal usage.
        results = []
        if agent is not None:
            try:
                for run_id in sorted(run_ids):
                    result = await agent._embedded._store.result(run_id)
                    transcript = await agent._embedded._store.load(run_id)
                    if result is not None:
                        if result.kind is LoopExitKind.COMPLETED:
                            validate_completed_transcript(transcript, result)
                        results.append(result)
                    report.setdefault("runs", []).append(
                        {
                            "run_id": run_id,
                            "result": (
                                None
                                if result is None
                                else json.loads(encode_loop_exit(result))
                            ),
                            "calls_results": [
                                {
                                    "call": {
                                        "id": c.id,
                                        "name": c.name,
                                        "arguments": json.loads(
                                            canonical_json(c.arguments)
                                        ),
                                    },
                                    "result": (
                                        None
                                        if r is None
                                        else json.loads(canonical_json(r.output))
                                    ),
                                }
                                for c, r in transcript.tool_pairs
                            ],
                        }
                    )
            except BaseException as error:
                cleanup_errors.append(type(error).__name__)
        if recorder is not None:
            try:
                await recorder.close()
            except BaseException as error:
                cleanup_errors.append(type(error).__name__)
            report["provider_timings"] = recorder.timings
            report["logical_requests"] = len(recorder.requests)
            report["tokens"] = sum(usage.total_tokens for usage in recorder.usages)
            report["usage_complete"] = bool(recorder.timings) and all(
                t["usage_complete"] for t in recorder.timings
            )
        if provider is not None:
            report["drift_injected"] = drift_injected
            report["narrowed_admission"] = provider.narrowed
            report["synthetic_foreground_requests"] = len(provider.seeded_requests)
        if reserved:
            # Settle once for the whole foreground+job reservation, without clipping.
            from daita.llm.models import ModelUsage
            from daita.llm.pricing import aggregate_cost_estimates

            aggregate = (
                replace(
                    results[0],
                    usage=ModelUsage(
                        input_tokens=sum(r.usage.input_tokens for r in results),
                        output_tokens=sum(r.usage.output_tokens for r in results),
                        cost_estimate=aggregate_cost_estimates(
                            tuple(r.usage.cost_estimate for r in results)
                        ),
                    ),
                )
                if results
                and report.get("usage_complete")
                and len(results) == len(run_ids)
                and sum(r.usage.total_tokens for r in results) == report.get("tokens")
                else None
            )
            count_job_budget.settle(aggregate)
            report["budget"] = {
                "total_usd": str(count_job_budget.total),
                "charged_usd": str(count_job_budget.charged),
                "incomplete_runs": count_job_budget.incomplete_runs,
            }
            report["usage_reconciled"] = bool(
                aggregate is not None
                and aggregate.usage.total_tokens == report.get("tokens")
            )
        report["cleanup"] = cleanup_errors or "complete"
        report["elapsed_seconds"] = perf_counter() - started
        if cleanup_errors or not report.get("usage_reconciled"):
            report["status"] = "failed"
        report["live_qualified"] = (
            report["status"] == "passed" and report.get("provider_kind") == "real"
        )
        report["usage_measurement"] = (
            "complete"
            if report.get("usage_complete")
            else "known_partial_or_unavailable"
        )
        write_private_report(path, report)
        if successful:
            assert not cleanup_errors, cleanup_errors
            assert report.get(
                "usage_reconciled"
            ), "Missing/unreconciled usage cannot qualify"


def _cost_limit() -> Decimal:
    raw = os.environ.get(_MAX_COST, "0.15")
    try:
        value = Decimal(raw)
    except InvalidOperation:
        pytest.fail(f"{_MAX_COST} must be a finite positive decimal")
    if not value.is_finite() or value <= 0:
        pytest.fail(f"{_MAX_COST} must be a finite positive decimal")
    return value


async def _wait_terminal(agent: Agent, job_id: str) -> None:
    deadline = asyncio.get_running_loop().time() + 120
    while asyncio.get_running_loop().time() < deadline:
        inspection = await agent.inspect_job(job_id)
        assert inspection is not None
        if inspection.job.state in {
            GraphState.SUCCEEDED,
            GraphState.FAILED,
            GraphState.CANCELLED,
            GraphState.NEEDS_ATTENTION,
        }:
            assert inspection.job.state is GraphState.SUCCEEDED
            return
        await asyncio.sleep(0.1)
    raise AssertionError("live production graph did not reach a terminal state")


async def test_live_model_task_uses_public_revision_2_composition(
    tmp_path: Path,
) -> None:
    model_id = os.environ.get(_MODEL_ID, _DEFAULT_MODEL_ID)
    provider_name = model_id.partition(":")[0]
    if provider_name not in _API_PROVIDERS:
        pytest.fail(f"{_MODEL_ID} must name an API-backed provider")
    profile = reviewed_model_profile(model_id)
    if profile is None or not profile.supports_tools:
        pytest.fail(f"{_MODEL_ID} must name a reviewed tool-capable model")
    api_key = os.environ.get(_MODEL_KEY)
    if api_key is None or not api_key.strip():
        pytest.skip(f"{_MODEL_KEY} is unavailable for the authorized live smoke")
    provider = create_llm_provider(
        model_id,
        api_key=api_key,
        max_output_tokens=min(profile.max_output_tokens, 2_048),
    )
    home = await create_probe_home(
        tmp_path,
        "live-production-graph",
        distractor_tables=3,
    )
    resources = tuple(sorted(home.resource_ids.items()))
    resource_ids = tuple(resource_id for _name, resource_id in resources)
    sql = "SELECT " + ", ".join(
        f"(SELECT COUNT(*) FROM {name}) AS count_{index}"
        for index, (name, _resource_id) in enumerate(resources)
    )
    agent = await Agent.open(
        home.name,
        root=home.root,
        model=provider,
        model_profile=profile,
        limits=LoopLimits(
            max_steps=8,
            max_total_tokens=40_000,
            max_estimated_cost_usd=_cost_limit(),
        ),
        workspace=workspace_for(home.root),
    )
    try:
        status = await Agent.inspect_home(home.name, root=home.root)
        assert status.found_revision == status.current_revision == CURRENT_HOME_REVISION
        foreground = await agent.run(
            "Start exactly one durable graph. Load start_graph_job, then call it "
            "with objective 'Run the exact admitted relational read', outcome_contract "
            f"{{'required_result_kind':'{DATA_QUERY_EVIDENCE_KIND}'}}, deadline_seconds "
            "300, and initial_task containing capability_id 'data.query', arguments "
            f"source_id={home.source_id!r}, resource_ids={resource_ids!r}, "
            f"sql={sql!r}, parameters=(). Use expected_result_contract "
            f"{{'result_kind':'{DATA_QUERY_EVIDENCE_KIND}'}} and retained_references "
            f"source_ids=({home.source_id!r},), resource_ids={resource_ids!r}, "
            "connector_binding_ids=(). After the durable receipt, stop.",
            source_scope_ids=(home.source_id,),
        )
        assert foreground.kind is LoopExitKind.COMPLETED
        jobs = await agent.list_jobs()
        assert len(jobs) == 1
        await _wait_terminal(agent, jobs[0].job_id)
        inspection = await agent.inspect_job(jobs[0].job_id)
        result = await agent.read_job_result(jobs[0].job_id)
        assert inspection is not None
        worker = next(task for task in inspection.tasks if task.role is TaskRole.WORKER)
        attempt = next(
            item for item in inspection.attempts if item.task_id == worker.task_id
        )
        assert worker.execution_kind is TaskExecutionKind.MODEL
        assert attempt.state is AttemptState.SUCCEEDED
        assert result is not None and result.result_kind == "graph.result_finalized"
        assert len(inspection.delivery_ids) == 1
    finally:
        await agent.close()
        await provider.close()
