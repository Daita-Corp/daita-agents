from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from daita import Agent
from daita.monitors import (
    IntervalSchedule,
    MonitorDefinition,
    MonitorScope,
    MonitorStatus,
)

NOW = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc)


async def test_public_monitor_lifecycle_survives_restart(tmp_path: Path) -> None:
    agent = await Agent.create("atlas", root=tmp_path, clock=lambda: NOW)
    definition = MonitorDefinition(
        name="Orders backlog",
        objective="Inspect the current orders backlog.",
        scope=MonitorScope(source_ids=("source-orders",)),
        schedule=IntervalSchedule(interval_seconds=300, anchor_at=NOW),
    )

    proposal = await agent.propose_monitor(
        "monitor-orders",
        definition,
        idempotency_key="monitor-create-1",
    )
    activated = await agent.confirm_monitor(
        proposal.id,
        candidate_hash=proposal.candidate_hash,
        actor_id="operator-1",
        reason="The scope and schedule are correct.",
    )
    assert activated.monitor.status is MonitorStatus.ENABLED
    assert await agent.list_monitors() == (activated.monitor,)
    assert await agent.inspect_monitor(activated.monitor.id) == activated

    paused = await agent.pause_monitor(
        activated.monitor.id,
        actor_id="operator-1",
        reason="Maintenance window.",
        idempotency_key="monitor-pause-1",
    )
    assert paused.monitor.status is MonitorStatus.PAUSED
    resumed = await agent.resume_monitor(
        activated.monitor.id,
        actor_id="operator-1",
        reason="Maintenance complete.",
        idempotency_key="monitor-resume-1",
    )
    assert resumed.monitor.status is MonitorStatus.ENABLED
    await agent.close()

    reopened = await Agent.open("atlas", root=tmp_path, clock=lambda: NOW)
    try:
        durable = await reopened.inspect_monitor(activated.monitor.id)
        assert durable.monitor == resumed.monitor
        deleted = await reopened.delete_monitor(
            activated.monitor.id,
            actor_id="operator-1",
            reason="Monitor retired.",
            idempotency_key="monitor-delete-1",
        )
        assert deleted.monitor.status is MonitorStatus.DELETED
        assert await reopened.list_monitors() == ()
        assert await reopened.list_monitors(include_deleted=True) == (deleted.monitor,)
    finally:
        await reopened.close()
