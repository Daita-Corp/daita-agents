"""Reuse the existing paid-provider, budget, transport and evidence helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import httpx2 as httpx
import pytest

from daita.security import EnvironmentSecretProvider
from tests.support.data_operations import (
    AUTHORIZATION,
    REPORT_ENV,
    TOTAL_COST_ENV,
    AnalyticalCase,
    create_database,
    fixture_digest,
    repetitions,
    require_authorization,
)
from tests.support.job_benchmarks import (
    MAX_COST_ENV,
    configured_model_ids,
    live_provider,
)
from tests.support.mcp import MCPConformanceTransport
from tests.support.mcp_read_harness import (
    MeasuredMCPTransport,
    ReadBudget,
    evaluate_reads,
    positive_cost,
)
from tests.support.paths import REPO_ROOT
from tests.support.workspace import workspace_for


@pytest.fixture(autouse=True, scope="session")
def dataops_live_gate():
    if os.environ.get(AUTHORIZATION) != "1":
        pytest.skip(f"Set {AUTHORIZATION}=1 to authorize live data operations")


@pytest.fixture(params=configured_model_ids())
def model_id(request):
    return request.param


@pytest.fixture(params=repetitions())
def repetition(request):
    return request.param


@pytest.fixture(scope="session")
def dataops_budget():
    return ReadBudget(
        total=positive_cost(TOTAL_COST_ENV, "10.00"),
        per_run=positive_cost(MAX_COST_ENV, "0.50"),
    )


@pytest.fixture(scope="session")
def dataops_report_directory():
    require_authorization()
    directory = Path(
        os.environ.get(REPORT_ENV, f"/private/tmp/daita-dataops-{uuid4().hex}")
    ).expanduser()
    if not directory.is_absolute():
        pytest.fail(f"{REPORT_ENV} must be absolute")
    directory = directory.resolve()
    if directory.is_relative_to(REPO_ROOT.resolve()):
        pytest.fail("Live evidence must be retained outside the repository")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return directory


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    result = yield
    report = result.get_result()
    setattr(item, f"dataops_{report.when}_report", report)


@pytest.fixture
async def dataops(
    tmp_path, request, model_id, repetition, dataops_budget, dataops_report_directory
):
    require_authorization()
    report_path = dataops_report_directory / f"case-{uuid4().hex}.json"
    # Restrict the file before the existing reporter writes any transcript,
    # including when setup or runtime cleanup raises.
    report_path.touch(mode=0o600, exist_ok=False)
    report_path.write_text(
        json.dumps(
            {
                "case_id": request.node.nodeid,
                "model_id": model_id,
                "repetition": repetition,
                "status": "setup_incomplete",
            }
        )
        + "\n"
    )
    request.node.user_properties.append(("dataops_evidence", str(report_path)))
    case = getattr(request.node, "callspec", None)
    case = None if case is None else case.params.get("case")
    variant = case.variant if isinstance(case, AnalyticalCase) else 0
    database = tmp_path / "sales.sqlite"
    create_database(database, variant)
    service = MCPConformanceTransport()
    transport = MeasuredMCPTransport(httpx.MockTransport(service))
    async with evaluate_reads(
        tmp_path / "homes",
        transport=transport,
        budget=dataops_budget,
        report=report_path,
        case_id=request.node.nodeid,
        secrets=EnvironmentSecretProvider(),
        workspace=workspace_for(tmp_path / "workspace"),
        provider_pair=live_provider(model_id),
    ) as evaluation:
        source = await evaluation.agent.attach_sqlite(
            database, name="Settlement fixture"
        )
        resources = await evaluation.agent.list_catalog_resources(source_id=source.id)
        evaluation.setup.update(
            evaluation_scope="data_operations",
            service_kind="fixture",
            model_id=model_id,
            repetition=repetition,
            variant=variant,
            fixture_digest=fixture_digest(variant),
            source_id=source.id,
            resource_ids=[resource.id for resource in resources],
            database=str(database),
            results={},
            deferred_features={
                "executable_metrics": "not_implemented; advisory semantics tested",
            },
        )
        # The mutable fixture identity permits exact-schema drift tests through
        # the same SDK transport, without building another server framework.
        try:
            yield evaluation, service
        finally:
            report = getattr(request.node, "dataops_call_report", None)
            if report is not None and not report.passed:
                # Let the existing context manager write failed evidence as well
                # as settle/close every real provider and runtime owner.
                evaluation.setup["pytest_outcome"] = report.outcome
                evaluation.setup["assertion_failed"] = report.failed
    # The context manager cannot see a pytest assertion through a yield fixture.
    # Correct its outcome after cleanup, preserving the existing captured runs.
    report = getattr(request.node, "dataops_call_report", None)
    if report is not None and not report.passed:
        evaluation.report(
            report_path,
            case_id=request.node.nodeid,
            failure_type="unsupported" if report.skipped else "assertion_failed",
        )
    report_path.chmod(0o600)
