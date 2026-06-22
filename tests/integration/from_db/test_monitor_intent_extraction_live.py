"""Live DB integration test for prompt-created monitor intent extraction.

Run:
    pytest tests/integration/from_db/test_monitor_intent_extraction_live.py \
        -m "requires_db and integration" -v -s

Set ``DAITA_MONITOR_INTENT_POSTGRES_URL`` to run against a dedicated external
Postgres test database. Without that, the test uses Docker Postgres when Docker
is available and falls back to a real SQLite database file otherwise.

This test intentionally does not require a live LLM. Prompt-created monitor
extraction is deterministic today; the live assertion here is that the
framework path works end-to-end against a real database schema and still flows
through proposal evidence, approval/resume, validation, and
``db.monitor.commit_create``.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pytest

from daita.agents.agent import Agent
from daita.db.runtime.extensions import HostedInAppMonitorDeliveryPlugin
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import (
    ApprovalStatus,
    HostRuntimeContext,
    OperationStatus,
    host_runtime_context,
)

from tests.integration._harness import start_container

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_db,
]

POSTGRES_IMAGE = "postgres:16-alpine"
POSTGRES_USER = "daita"
POSTGRES_PASSWORD = "daita_test_pw"
POSTGRES_DB = "daita_monitor_intent_live_test"
EXTERNAL_POSTGRES_URL_ENV = "DAITA_MONITOR_INTENT_POSTGRES_URL"

SEED_SQL = """
DROP TABLE IF EXISTS operations;

CREATE TABLE operations (
    id BIGSERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO operations (event_type, created_at) VALUES
    ('created', '2026-06-14T11:45:00Z'),
    ('updated', '2026-06-14T11:50:00Z');
"""

SQLITE_SEED_SQL = """
DROP TABLE IF EXISTS operations;

CREATE TABLE operations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL
);

INSERT INTO operations (event_type, created_at) VALUES
    ('created', '2026-06-14T11:45:00Z'),
    ('updated', '2026-06-14T11:50:00Z');
"""

PROMPT = (
    "I want you to create a monitor for the operations table. "
    "Notify me in app when a new record shows up in the table. "
    "Poll every 5 mins."
)

CREATE_CASES = (
    {
        "id": "conversational_preamble",
        "prompt": PROMPT,
        "name": "Operations New Rows",
        "monitor_id": "operations_new_rows",
        "target": "operations",
        "condition": "new_rows",
        "schedule": {"interval_seconds": 300},
        "delivery_kind": "in_app",
    },
    {
        "id": "direct_create",
        "prompt": (
            "Create a monitor for operations when new rows are added. "
            "Notify me in app every 5 minutes."
        ),
        "name": "Operations New Rows",
        "monitor_id": "operations_new_rows",
        "target": "operations",
        "condition": "new_rows",
        "schedule": {"interval_seconds": 300},
        "delivery_kind": "in_app",
    },
    {
        "id": "watch_prompt",
        "prompt": "Watch operations for new rows every 5 minutes. Notify me in app.",
        "name": "Operations New Rows",
        "monitor_id": "operations_new_rows",
        "target": "operations",
        "condition": "new_rows",
        "schedule": {"interval_seconds": 300},
        "delivery_kind": "in_app",
    },
    {
        "id": "explicit_name",
        "prompt": (
            'Create a monitor named "Operations Inserts" for operations when '
            "new rows are added. Notify me in app every 5 minutes."
        ),
        "name": "Operations Inserts",
        "monitor_id": "operations_inserts",
        "target": "operations",
        "condition": "new_rows",
        "schedule": {"interval_seconds": 300},
        "delivery_kind": "in_app",
    },
    {
        "id": "threshold",
        "prompt": (
            "Monitor operations every 5 minutes if operations exceed 10. "
            "Notify me in app."
        ),
        "name": "Operations Threshold",
        "monitor_id": "operations_threshold",
        "target": "operations",
        "condition": "threshold",
        "schedule": {"interval_seconds": 300},
        "delivery_kind": "in_app",
    },
    {
        "id": "hosted_default_delivery",
        "prompt": "Monitor operations every 5 minutes when new rows are added. Notify me.",
        "name": "Operations New Rows",
        "monitor_id": "operations_new_rows",
        "target": "operations",
        "condition": "new_rows",
        "schedule": {"interval_seconds": 300},
        "delivery_kind": "in_app",
    },
)


@pytest.fixture()
def live_db_source(tmp_path):
    external_url = os.environ.get(EXTERNAL_POSTGRES_URL_ENV)
    if external_url:
        asyncio.run(_seed_postgres(external_url))
        yield external_url
        return

    if not _docker_available():
        db_path = tmp_path / "monitor-intent-live.sqlite"
        asyncio.run(_seed_sqlite(db_path))
        yield str(db_path)
        return

    container = start_container(
        POSTGRES_IMAGE,
        container_port=5432,
        env={
            "POSTGRES_USER": POSTGRES_USER,
            "POSTGRES_PASSWORD": POSTGRES_PASSWORD,
            "POSTGRES_DB": POSTGRES_DB,
        },
        tag_prefix="daita-monitor-intent-pg",
    )
    url = (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{container.host}:{container.host_port}/{POSTGRES_DB}"
    )
    try:
        asyncio.run(_seed_postgres(url))
        yield url
    finally:
        container.remove()


async def test_live_db_prompt_monitor_create_uses_normalized_intent_and_resume(
    live_db_source,
):
    case = CREATE_CASES[0]
    result = await _create_prompt_monitor(live_db_source, case)

    assert result["created"].status in {
        OperationStatus.BLOCKED,
        OperationStatus.SUCCEEDED,
    }
    assert result["proposal_evidence"].accepted is True
    assert result["validation"]["accepted"] is True
    _assert_proposal_matches_case(result["proposal"], case)
    _assert_operation_and_approval_match_case(result, case)
    _assert_committed_monitor_matches_case(result, case)


@pytest.mark.parametrize(
    "case", CREATE_CASES, ids=[case["id"] for case in CREATE_CASES]
)
async def test_live_db_prompt_monitor_create_variants_extract_expected_intent(
    live_db_source,
    case,
):
    result = await _create_prompt_monitor(live_db_source, case)

    assert result["created"].status in {
        OperationStatus.BLOCKED,
        OperationStatus.SUCCEEDED,
    }
    assert result["proposal_evidence"].accepted is True
    assert result["validation"]["accepted"] is True
    _assert_proposal_matches_case(result["proposal"], case)
    _assert_operation_and_approval_match_case(result, case)
    _assert_committed_monitor_matches_case(result, case)


async def test_live_db_prompt_monitor_create_blocks_ambiguous_target(live_db_source):
    prompt = (
        "Create a monitor when new rows are added. Notify me in app every 5 minutes."
    )
    agent = await _agent_for_monitor_intent(live_db_source, "ambiguous_target")

    try:
        result = await agent.run_detailed(prompt)
        evidence = await agent.runtime.store.list_evidence(result.operation_id)
        proposal_evidence = _latest(evidence, "monitor.proposal")
        proposal = proposal_evidence.payload
    finally:
        await agent.stop()

    assert result.status is OperationStatus.BLOCKED
    assert proposal_evidence.accepted is False
    assert proposal["name"] == "DB Monitor"
    assert proposal["monitor_id"] == "new_rows_every_300_seconds"
    assert proposal["target_name"] == ""
    assert proposal["metadata"]["prompt"] == prompt
    assert "monitor.proposal_incomplete:source" in proposal["validation"]["errors"]
    assert "create_monitor_when_new_rows_added" not in proposal["monitor_id"]


async def _seed_postgres(url: str) -> None:
    asyncpg = pytest.importorskip(
        "asyncpg",
        reason="asyncpg required: pip install 'daita-agents[postgresql]'",
    )
    last_error: Exception | None = None
    for _attempt in range(60):
        try:
            connection = await asyncpg.connect(url, ssl=False)
            await connection.execute(SEED_SQL)
            await connection.close()
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            await asyncio.sleep(0.5)
    raise RuntimeError(f"Could not seed Postgres monitor intent test DB: {last_error}")


async def _agent_for_monitor_intent(source: str, case_id: str):
    delivery_plugin = HostedInAppMonitorDeliveryPlugin(
        service=lambda payload: {"id": f"live-test-{case_id}"}
    )
    with host_runtime_context(
        HostRuntimeContext(
            surface="web_app",
            delivery_defaults=("in_app",),
            runtime_extensions=(delivery_plugin,),
        )
    ):
        return await Agent.from_db(
            source,
            name=f"LiveMonitorIntentExtraction-{case_id}",
            cache_ttl=0,
        )


async def _create_prompt_monitor(source: str, case: dict[str, Any]) -> dict[str, Any]:
    agent = await _agent_for_monitor_intent(source, str(case["id"]))
    try:
        created = await agent.run_detailed(str(case["prompt"]))
        evidence = await agent.runtime.store.list_evidence(created.operation_id)
        proposal_evidence = _latest(evidence, "monitor.proposal")
        proposal = proposal_evidence.payload
        validation = proposal["validation"]
        operation = await agent.runtime.store.load_operation(created.operation_id)
        approvals = await agent.runtime.store.list_approval_requests(
            created.operation_id
        )
        if approvals:
            await agent.runtime.approval_channel.approve(approvals[0].approval_id)
            resumed = await agent.runtime.resume_operation(created.operation_id)
            final_status = resumed.operation.status
        else:
            resumed = None
            final_status = created.status
        monitors = await agent.list_monitors()
        committed_evidence = await agent.runtime.store.list_evidence(
            created.operation_id
        )
        return {
            "created": created,
            "proposal_evidence": proposal_evidence,
            "proposal": proposal,
            "validation": validation,
            "operation": operation,
            "approvals": approvals,
            "resumed": resumed,
            "final_status": final_status,
            "monitors": monitors,
            "definition": _latest(committed_evidence, "monitor.definition"),
        }
    finally:
        await agent.stop()


def _assert_proposal_matches_case(
    proposal: dict[str, Any],
    case: dict[str, Any],
) -> None:
    assert proposal["name"] == case["name"]
    assert proposal["monitor_id"] == case["monitor_id"]
    assert proposal["description"] == case["prompt"]
    assert proposal["target_type"] == "table"
    assert proposal["target_name"] == case["target"]
    assert proposal["source_scope"] == [case["target"]]
    assert proposal["schedule"] == case["schedule"]
    assert proposal["trigger"]["type"] == case["condition"]
    if case["condition"] == "new_rows":
        assert proposal["trigger"] == {
            "type": "new_rows",
            "path": "rows",
            "operator": "count_gt",
            "value": 0,
        }
    elif case["condition"] == "threshold":
        assert proposal["trigger"]["operator"] == "gt"
        assert proposal["trigger"]["value"] == 10
    delivery = proposal["action_plan"]["delivery_intent"]
    assert delivery["delivery_kind"] == case["delivery_kind"]
    assert delivery["target"] == {"type": "requesting_user"}
    assert delivery["payload_source"] == {"type": "monitor.report"}
    assert delivery["include_observed_rows"] is True
    assert delivery["template"] == "New rows were observed for the monitored table."
    assert proposal["metadata"]["prompt"] == case["prompt"]
    assert proposal["metadata"]["intent"]["target"]["name"] == case["target"]
    assert proposal["metadata"]["intent"]["condition"]["kind"] == case["condition"]
    assert proposal["metadata"]["intent"]["schedule"]["interval_seconds"] == 300
    assert (
        proposal["metadata"]["intent"]["delivery"]["delivery_kind"]
        == case["delivery_kind"]
    )
    assert proposal["metadata"]["extraction"]["extractor"] == "deterministic"
    assert proposal["observation_plan"]["kind"] == "planned_read"
    assert proposal["observation_plan"]["target_name"] == case["target"]
    assert f"select * from {case['target']}" in proposal["observation_plan"]["sql"]
    assert proposal["observation_plan"]["cursor"]["field"] == "created_at"
    validation = proposal["validation"]
    assert validation["required_capabilities"] == ["monitor.delivery.in_app"]
    assert validation["missing_capabilities"] == []
    assert validation["diagnostics"]["delivery_validation"]["accepted"] is True


def _assert_operation_and_approval_match_case(
    result: dict[str, Any],
    case: dict[str, Any],
) -> None:
    operation = result["operation"]
    approvals = result["approvals"]
    assert operation is not None
    assert operation.metadata["monitor_id"] == case["monitor_id"]
    assert operation.metadata["monitor_name"] == case["name"]
    if case["prompt"].lower().startswith(("create", "i want you to create")):
        assert len(approvals) == 1
        assert approvals[0].status is ApprovalStatus.PENDING
        assert approvals[0].proposed_action["operation_type"] == "monitor.create"
        assert approvals[0].proposed_action["monitor"]["id"] == case["monitor_id"]
        assert approvals[0].proposed_action["monitor"]["name"] == case["name"]
    else:
        assert approvals == []


def _assert_committed_monitor_matches_case(
    result: dict[str, Any],
    case: dict[str, Any],
) -> None:
    monitors = result["monitors"]
    assert result["final_status"] is OperationStatus.SUCCEEDED
    assert [monitor.id for monitor in monitors] == [case["monitor_id"]]
    assert monitors[0].name == case["name"]
    assert monitors[0].description == case["prompt"]
    assert monitors[0].schedule == case["schedule"]
    assert monitors[0].trigger["type"] == case["condition"]
    assert monitors[0].action_plan["delivery_intent"]["delivery_kind"] == (
        case["delivery_kind"]
    )
    assert result["definition"].payload["monitor"]["id"] == case["monitor_id"]
    assert result["definition"].payload["proposal_evidence_id"] == (
        result["proposal_evidence"].id
    )


async def _seed_sqlite(path: Path) -> None:
    plugin = SQLitePlugin(path=str(path))
    await plugin.execute_script(SQLITE_SEED_SQL)
    await plugin.disconnect()


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    return True


def _latest(evidence: tuple[Any, ...] | list[Any], kind: str) -> Any:
    matches = [item for item in evidence if item.kind == kind]
    assert matches, f"missing evidence kind {kind}"
    return matches[-1]
