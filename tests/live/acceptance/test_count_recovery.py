"""Separate API/SQLite, API/PostgreSQL and actual Codex qualification cohorts.

No credentials are resolved and no service is contacted during collection.
Deselected/skipped cohorts are unperformed evidence, never qualification.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from tests.support.count_recovery_live import (
    AUTHORIZATION,
    CASES,
    CODEX_GATE,
    POSTGRES_GATE,
    REPORT_ENV,
    TOTAL_COST_ENV,
    repetitions,
    run_case,
    summarize_campaign,
)
from tests.support.job_benchmarks import MAX_COST_ENV, configured_model_ids
from tests.support.mcp_read_harness import (
    ReadBudget,
    positive_cost,
    write_private_report,
)

pytestmark = [pytest.mark.acceptance, pytest.mark.integration, pytest.mark.requires_llm]


@pytest.fixture(scope="module")
def campaign_budget():
    try:
        return ReadBudget(
            total=positive_cost(TOTAL_COST_ENV, "15"),
            per_run=positive_cost(MAX_COST_ENV, "0.50"),
        )
    except ValueError:
        return None  # run_case retains the missing/invalid-configuration failure report


@pytest.fixture(scope="module")
def campaign_reports():
    reports: list[dict[str, Any]] = []
    yield reports
    groups = defaultdict(list)
    for report in reports:
        groups[(report["model_id"], report["transport"], report["database"])].append(
            report
        )
    summaries = {
        ":".join(key): summarize_campaign(
            value, expected_cases=len(CASES) * len(repetitions())
        )
        for key, value in groups.items()
    }
    raw = os.environ.get(REPORT_ENV)
    if raw:
        directory = Path(raw)
        from tests.support.paths import REPO_ROOT

        if directory.is_absolute() and not directory.resolve().is_relative_to(
            REPO_ROOT.resolve()
        ):
            write_private_report(
                directory / "campaign-summary.json",
                {
                    "cohorts": summaries,
                    "unperformed_cohorts": {
                        "api_sqlite": os.environ.get(AUTHORIZATION) != "1",
                        "postgresql": os.environ.get(POSTGRES_GATE) != "1",
                        "codex": os.environ.get(CODEX_GATE) != "1",
                    },
                    "note": "Skipped, deselected, incomplete or missing-measurement runs never qualify. Compare only matched profiles, fixtures and transport settings.",
                },
            )


def serial_only():
    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.fail("Run count campaigns serially so reservations are shared")


@pytest.mark.skipif(
    os.environ.get(AUTHORIZATION) != "1",
    reason="API count/recovery campaign not authorized",
)
@pytest.mark.parametrize("model_id", configured_model_ids())
@pytest.mark.parametrize("repetition", repetitions())
@pytest.mark.parametrize("case", CASES)
async def test_api_sqlite_count_recovery(
    tmp_path, request, model_id, repetition, case, campaign_budget, campaign_reports
):
    serial_only()
    await run_case(
        tmp_path,
        case=case,
        model_id=model_id,
        repetition=repetition,
        transport="api",
        database="sqlite",
        budget=campaign_budget,
        reports=campaign_reports,
        node=request.node,
    )


@pytest.mark.requires_db
@pytest.mark.requires_network
@pytest.mark.skipif(
    os.environ.get(AUTHORIZATION) != "1" or os.environ.get(POSTGRES_GATE) != "1",
    reason="Separate PostgreSQL/API gates not authorized",
)
@pytest.mark.parametrize("model_id", configured_model_ids())
@pytest.mark.parametrize("repetition", repetitions())
@pytest.mark.parametrize("case", CASES)
async def test_api_postgresql_count_recovery(
    tmp_path, request, model_id, repetition, case, campaign_budget, campaign_reports
):
    serial_only()
    await run_case(
        tmp_path,
        case=case,
        model_id=model_id,
        repetition=repetition,
        transport="api",
        database="postgresql",
        budget=campaign_budget,
        reports=campaign_reports,
        node=request.node,
    )


@pytest.mark.requires_network
@pytest.mark.skipif(
    os.environ.get(CODEX_GATE) != "1",
    reason="Actual Codex transport cohort not authorized",
)
@pytest.mark.parametrize("repetition", repetitions())
@pytest.mark.parametrize("case", CASES)
async def test_codex_sqlite_count_recovery(
    tmp_path, request, repetition, case, campaign_reports
):
    serial_only()
    await run_case(
        tmp_path,
        case=case,
        model_id=os.environ.get(
            "DAITA_COUNT_RECOVERY_CODEX_MODEL_ID", "codex:unconfigured"
        ),
        repetition=repetition,
        transport="codex",
        database="sqlite",
        budget=None,
        reports=campaign_reports,
        node=request.node,
    )
