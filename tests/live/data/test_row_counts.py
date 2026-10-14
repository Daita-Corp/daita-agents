"""Real PostgreSQL count execution; no real-model qualification is claimed."""

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from daita import Agent
from daita.catalog.models import ResourceKind
from daita.llm.models import ToolCall
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import RunInput
from daita.security import EnvironmentSecretProvider
from tests.support.capability_runtime import execute_projected
from tests.support.count_recovery import MANIFEST, TOTALS, quote
from tests.support.count_recovery_live import POSTGRES_GATE, postgres_sources
from tests.support.mcp_read_harness import write_private_report
from tests.support.paths import REPO_ROOT

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_db,
    pytest.mark.skipif(
        os.environ.get(POSTGRES_GATE) != "1",
        reason="Native PostgreSQL count fixture not authorized",
    ),
]


async def test_real_postgresql_counts_exact_batches_and_rejects_drift(
    tmp_path, request
):
    directory = Path(os.environ["DAITA_COUNT_RECOVERY_REPORT_DIR"])
    assert directory.is_absolute() and not directory.resolve().is_relative_to(
        REPO_ROOT.resolve()
    )
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    assert directory.stat().st_mode & 0o077 == 0
    model = MockModelProvider(())
    agent = None
    admin = None
    schemas = []
    report: dict[str, Any] = {
        "qualification": "native_postgresql_only",
        "llm_qualified": False,
        "status": "started",
        "counts": [],
        "cleanup": "pending",
    }
    path = directory / ("native-postgresql-" + uuid4().hex + ".json")
    request.node.user_properties.append(("count_recovery_evidence", str(path)))
    write_private_report(path, report)
    try:
        agent = await Agent.create(
            "native-counts",
            root=tmp_path,
            hosted=True,
            model=model,
            model_profile=model.model_profile,
            secret_provider=EnvironmentSecretProvider(),
        )
        sources, admin, schemas = await postgres_sources(agent, uuid4().hex[:12])
        for name, source in sources.items():
            resources = [
                r
                for r in await agent.list_catalog_resources(source_id=source.id)
                if r.kind is ResourceKind.TABLE
            ]
            run = RunInput(
                "native-" + name,
                agent.id,
                "native adapter oracle",
                datetime.now(UTC),
                source_scope_ids=(source.id,),
            )
            observed = {}
            for offset in range(0, len(resources), 4):
                batch = resources[offset : offset + 4]
                call = ToolCall(
                    "count-" + str(offset),
                    "data_count_rows",
                    {
                        "source_id": source.id,
                        "resource_ids": tuple(r.id for r in batch),
                    },
                )
                result = (
                    await execute_projected(
                        agent._embedded._capability_runtime, run, (call,)
                    )
                ).ordered_results[0]
                assert not result.is_error, result.output
                data = result.output["data"]
                assert data["source_name"] == name
                assert data["coverage"] == "complete_requested_batch"
                for row in data["counts"]:
                    resource = next(r for r in batch if r.id == row["resource_id"])
                    expected = next(
                        t["rows"] for t in MANIFEST[name] if t["table"] == resource.name
                    )
                    assert (
                        row["rows"] == expected
                        and row["revision"] == resource.current_revision
                    )
                    observed[resource.id] = row["rows"]
                assert data["total_rows"] == sum(row["rows"] for row in data["counts"])
                report["counts"].append(data.to_dict())
            assert (
                len(observed) == len(MANIFEST[name])
                and sum(observed.values()) == TOTALS[name]
            )
            oversized = ToolCall(
                "oversized",
                "data_count_rows",
                {
                    "source_id": source.id,
                    "resource_ids": tuple(r.id for r in resources[:5]),
                },
            )
            rejected = (
                await execute_projected(
                    agent._embedded._capability_runtime, run, (oversized,)
                )
            ).ordered_results[0]
            assert rejected.is_error and "data" not in rejected.output
            target = resources[0]
            await admin.execute(
                f"ALTER TABLE {quote(schemas[0 if name == 'Juniper' else 1])}.{quote(target.name)} ADD COLUMN drift INTEGER"
            )
            stale = ToolCall(
                "stale",
                "data_count_rows",
                {"source_id": source.id, "resource_ids": (target.id,)},
            )
            failed = (
                await execute_projected(
                    agent._embedded._capability_runtime, run, (stale,)
                )
            ).ordered_results[0]
            assert failed.is_error and "data" not in failed.output
        assert not model.requests and await agent.list_effects() == ()
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", failure_type=type(error).__name__)
        raise
    finally:
        if agent is not None:
            await agent.close()
        if admin is not None:
            try:
                for schema in schemas:
                    await admin.execute(f"DROP SCHEMA {quote(schema)} CASCADE")
            finally:
                await admin.close()
        report["cleanup"] = "complete"
        write_private_report(path, report)
