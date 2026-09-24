"""Trusted time and bounded model-facing routine timing."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from daita import Agent, ApprovalDecision
from daita.distribution import outcome_contract_projection
from daita.llm.models import FinishReason, ModelResponse, TextBlock, ToolCall
from daita.llm.providers.mock import MockModelProvider
from daita.routines.temporal import next_weekday_utc, system_user_timezone
from tests.support.capability_runtime import execute_projected
from tests.support.distribution import no_artifact_outcome_contract
from tests.support.workspace import workspace_for

NOW = datetime(2026, 9, 23, 16, 0, tzinfo=UTC)


def test_next_weekday_uses_local_day_and_rejects_dst_ambiguity() -> None:
    assert next_weekday_utc(
        now=NOW, timezone="America/Chicago", weekday=3, hour=17, minute=0
    ) == datetime(2026, 9, 23, 22, 0, tzinfo=UTC)
    assert next_weekday_utc(
        now=datetime(2026, 9, 23, 23, 0, tzinfo=UTC),
        timezone="America/Chicago",
        weekday=3,
        hour=17,
        minute=0,
    ) == datetime(2026, 9, 30, 22, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="ambiguous"):
        next_weekday_utc(
            now=datetime(2026, 10, 31, 12, tzinfo=UTC),
            timezone="America/Chicago",
            weekday=7,
            hour=1,
            minute=30,
        )
    with pytest.raises(ValueError, match="nonexistent"):
        next_weekday_utc(
            now=datetime(2026, 3, 7, 12, tzinfo=UTC),
            timezone="America/Chicago",
            weekday=7,
            hour=2,
            minute=30,
        )


def test_local_timezone_detection_requires_an_iana_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TZ", "America/Chicago")
    assert system_user_timezone() == "America/Chicago"


async def test_hosted_run_does_not_claim_the_host_timezone_as_the_users(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TZ", "America/Chicago")
    provider = MockModelProvider(
        (ModelResponse(finish_reason=FinishReason.STOP, text="Which timezone?"),),
        provider_id="mock:unknown-user-zone",
    )
    agent = await Agent.create(
        "hosted-timezone",
        root=tmp_path,
        hosted=True,
        model=provider,
        model_profile=provider.model_profile,
        clock=lambda: NOW,
    )
    try:
        await agent.run("Check again Wednesday at 5pm.")
        system_block = provider.requests[0].messages[0].content[0]
        assert isinstance(system_block, TextBlock)
        system_text = system_block.text
        assert "User-local IANA timezone is unknown" in system_text
        assert "detected user-local" not in system_text
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "schedule,seconds_field,expected_due",
    (
        (
            {"kind": "once", "after_seconds": 300},
            "after_seconds",
            NOW + timedelta(minutes=7),
        ),
        (
            {"kind": "interval", "interval_seconds": 300, "anchor_after_seconds": 300},
            "anchor_after_seconds",
            NOW + timedelta(minutes=7),
        ),
        (
            {
                "kind": "once_next_weekday",
                "timezone": "America/Chicago",
                "weekday": 3,
                "hour": 17,
                "minute": 0,
            },
            None,
            NOW + timedelta(hours=6),
        ),
        (
            {
                "kind": "interval",
                "interval_seconds": 3600,
                "anchor_at": (NOW - timedelta(seconds=100)).isoformat(),
            },
            None,
            NOW + timedelta(seconds=3500),
        ),
    ),
)
async def test_relative_schedule_starts_after_approval_and_uses_trusted_local_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schedule: dict[str, object],
    seconds_field: str | None,
    expected_due: datetime,
) -> None:
    import daita.context as context_module

    monkeypatch.setattr(
        context_module, "system_user_timezone", lambda: "America/Chicago"
    )
    current = [NOW]
    approvals = []

    def clock() -> datetime:
        return current[0]

    async def approve(request):
        approvals.append(request)
        current[0] += timedelta(minutes=2)
        return ApprovalDecision.APPROVE

    provider = MockModelProvider(
        (ModelResponse(finish_reason=FinishReason.STOP, text="Ready."),),
        provider_id="mock:relative-routine",
        complete_pricing=True,
    )
    agent = await Agent.create(
        "relative-routine",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=provider.model_profile,
        clock=clock,
        approval_handler=approve,
    )
    try:
        origin = await agent.run("Update me now and again in five minutes.")
        system_block = provider.requests[0].messages[0].content[0]
        assert isinstance(system_block, TextBlock)
        system_text = system_block.text
        assert "user-local Wednesday 2026-09-23 11:00:00 -0500" in system_text
        assert "America/Chicago" in system_text
        destinations = await agent.distribution_destinations(origin.conversation_id)
        arguments = {
            "title": "Five-minute follow-up",
            "authorized_instruction": "Read the source and report one update.",
            "schedule": schedule,
            "misfire_policy": "latest_only",
            "reporting_mode": "always",
            "allowed_source_ids": (),
            "allowed_resource_ids": (),
            "allowed_connector_binding_ids": (),
            "allowed_capability_ids": ("artifact.create_document",),
            "sensitivity_ceiling": "internal",
            "outcome_contract": outcome_contract_projection(
                no_artifact_outcome_contract()
            ),
            "distribution_destination_id": destinations[0].destination_id,
            "eligible_model_routes": (provider.provider_id,),
            "per_run_max_tokens": 1000,
            "per_run_max_cost_usd": "0.01",
            "cumulative_max_tokens": 1000,
            "cumulative_max_cost_usd": "0.01",
            "cumulative_max_attempts": 1,
            "cumulative_max_occurrences": 1,
            "maximum_consecutive_failures": 1,
            "expires_after_seconds": 86400,
            "skill_names": (),
            "run_immediately": False,
        }
        run = (await agent._embedded._store.load(origin.run_id)).run
        result = await execute_projected(
            agent._embedded._capability_runtime,
            run,
            (ToolCall("relative-once", "routine_create", arguments),),
        )
        assert not result.ordered_results[0].is_error, result.ordered_results[0].output
        assert len(approvals) == 1
        if seconds_field is not None:
            assert (
                approvals[0].arguments["timing_intent"]["schedule"][seconds_field]
                == 300
            )
        summary = (await agent.list_routines())[0]
        inspection = await agent.inspect_routine(summary.routine_id)
        assert inspection is not None
        assert inspection.routine.next_due_at == expected_due
        assert inspection.routine.expires_at == NOW + timedelta(minutes=2, days=1)
    finally:
        await agent.close()
