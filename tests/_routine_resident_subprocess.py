from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from daita import (
    Agent,
    IntervalSchedule,
    MisfirePolicy,
    ReportingMode,
    SQLiteSource,
    ScheduledRoutineDraft,
)
from daita.hosting.resident import run_resident_host
from daita.llm.models import (
    FinishReason,
    ModelResponse,
    ModelSensitivity,
    ModelUsage,
)
from daita.llm.pricing import CostEstimate
from daita.llm.providers.mock import MockModelProvider
from daita.workspace import LocalWorkspace

INITIAL = datetime(2026, 8, 28, 12, tzinfo=UTC)
ADVANCED = INITIAL + timedelta(hours=1)


def response() -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        text="Current value is 7.",
        usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0"))),
    )


def inbox_count(state_path: Path) -> int:
    connection = sqlite3.connect(f"file:{state_path}?mode=ro", uri=True)
    try:
        row = connection.execute("SELECT COUNT(*) FROM conversation_inbox").fetchone()
        assert row is not None
        return int(row[0])
    finally:
        connection.close()


async def main(root: Path) -> None:
    workspace_path = root.parent / f".{root.name}-workspace"
    workspace_path.mkdir()
    workspace = LocalWorkspace(workspace_path)
    database = root / "current.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE current_value (value INTEGER NOT NULL)")
    connection.execute("INSERT INTO current_value VALUES (7)")
    connection.commit()
    connection.close()

    foreground = MockModelProvider(
        (response(),),
        provider_id="mock:resident-acceptance",
        complete_pricing=True,
    )
    agent = await Agent.create(
        "resident-acceptance",
        root=root,
        workspace=workspace,
        model=foreground,
        model_profile=foreground.model_profile,
        clock=lambda: INITIAL,
    )
    source = await agent.attach(SQLiteSource(database, name="Current"))
    resource = (await agent.list_catalog_resources(source_id=source.id))[0]
    origin = await agent.run("Read the current value before scheduling it.")
    draft = ScheduledRoutineDraft(
        origin_run_id=origin.run_id,
        title="Resident current value",
        authorized_instruction="Inspect the exact current_value resource and report it.",
        schedule=IntervalSchedule(3_600, ADVANCED),
        misfire_policy=MisfirePolicy.LATEST_ONLY,
        reporting_mode=ReportingMode.ALWAYS,
        precheck=None,
        allowed_source_ids=(source.id,),
        allowed_connector_binding_ids=(),
        allowed_resource_ids=(resource.id,),
        allowed_capability_ids=("catalog.inspect",),
        sensitivity_ceiling=ModelSensitivity.INTERNAL,
        eligible_model_routes=(foreground.provider_id,),
        per_run_max_tokens=1_000,
        per_run_max_cost_usd=Decimal("0"),
        cumulative_max_tokens=10_000,
        cumulative_max_cost_usd=Decimal("0"),
        cumulative_max_attempts=10,
        cumulative_max_occurrences=10,
        maximum_consecutive_failures=3,
        expires_at=INITIAL + timedelta(days=30),
    )
    await agent.create_routine(await agent.propose_routine(draft))
    state_path = agent.home / "state.db"
    assert await agent.inbox() == ()
    await agent.close()

    resident_model = MockModelProvider(
        (response(),),
        provider_id=foreground.provider_id,
        complete_pricing=True,
    )
    ready = asyncio.Event()
    stop = asyncio.Event()
    hosted = asyncio.create_task(
        run_resident_host(
            agent_name="resident-acceptance",
            root=root,
            workspace=workspace,
            clock=lambda: ADVANCED,
            model=resident_model,
            model_profile=resident_model.model_profile,
            stop_event=stop,
            on_ready=lambda _value: ready.set(),
        )
    )
    await ready.wait()
    for _ in range(5_000):
        if inbox_count(state_path) == 1:
            break
        await asyncio.sleep(0)
    assert inbox_count(state_path) == 1
    stop.set()
    await hosted

    restart_model = MockModelProvider(
        (),
        provider_id=foreground.provider_id,
        complete_pricing=True,
    )
    restarted_ready = asyncio.Event()
    restarted_stop = asyncio.Event()
    restarted = asyncio.create_task(
        run_resident_host(
            agent_name="resident-acceptance",
            root=root,
            workspace=workspace,
            clock=lambda: ADVANCED,
            model=restart_model,
            model_profile=restart_model.model_profile,
            stop_event=restarted_stop,
            on_ready=lambda _value: restarted_ready.set(),
        )
    )
    await restarted_ready.wait()
    for _ in range(100):
        await asyncio.sleep(0)
    restarted_stop.set()
    await restarted
    assert inbox_count(state_path) == 1
    print(json.dumps({"inbox_count": 1, "restart_model_calls": 0}))


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]).resolve()))
