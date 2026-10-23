"""The routine owner: one intent shape, defaults, updates and controls."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import daita.routines.owner as owner_module
from daita import Agent, SQLiteSource
from daita._json import FrozenJsonObject
from daita.adapters.models import SourceRegistration
from daita.capabilities import AutomationEligibility, OperationalEffect
from daita.llm.models import FinishReason, ModelResponse
from daita.llm.providers.mock import MockModelProvider
from daita.routines.capabilities import (
    ROUTINE_CREATE_CAPABILITY_ID,
    ROUTINE_LIST_TOOL_NAME,
    routine_capability_declarations,
)
from daita.routines.models import (
    IntervalSchedule,
    OnceSchedule,
    RoutineControlAction,
    RoutineIntent,
    RoutineState,
)
from daita.routines.owner import RoutineError, RoutineOwner
from daita.scope import scope_refs
from daita.storage.sqlite import SQLiteStateStore
from tests.support.workspace import workspace_for

NOW = datetime(2026, 9, 23, 16, tzinfo=UTC)


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SQLiteStateStore]:
    opened = await SQLiteStateStore.open(tmp_path / "state.db")
    try:
        yield opened
    finally:
        await opened.close()


def _owner(
    store: SQLiteStateStore, clock: list[datetime] | None = None
) -> RoutineOwner:
    current = clock or [NOW]
    return RoutineOwner(agent_id="agent-1", store=store, clock=lambda: current[0])


async def _source(store: SQLiteStateStore, name: str) -> SourceRegistration:
    return await store.register_source(
        SourceRegistration.build(
            agent_id="agent-1",
            adapter_id="sqlite",
            native_identity=name,
            display_name=name,
            configuration={"path": f"/data/{name}.sqlite"},
            attached_at=NOW,
        )
    )


_INTENT = RoutineIntent(
    title="Hourly sales",
    prompt="Report the latest sales total.",
    schedule=IntervalSchedule(3_600, NOW + timedelta(hours=1)),
)


async def _create(owner: RoutineOwner, intent: RoutineIntent = _INTENT):
    prepared = await owner.prepare(
        intent, conversation_id="conversation-1", identity="create-1"
    )
    return await owner.save(prepared)


async def test_owner_admits_lists_inspects_and_controls_exact_agent_scope(
    store: SQLiteStateStore,
) -> None:
    owner = _owner(store)
    admitted = await _create(owner)
    assert admitted.next_due_at == NOW + timedelta(hours=1)
    assert (await owner.list())[0].routine_id == admitted.routine_id
    inspection = await owner.inspect(admitted.routine_id)
    assert inspection is not None and inspection.routine == admitted
    assert (
        await RoutineOwner(agent_id="agent-2", store=store, clock=lambda: NOW).inspect(
            admitted.routine_id
        )
        is None
    )

    paused = await owner.control(
        admitted.routine_id,
        expected_revision=admitted.revision,
        action=RoutineControlAction.PAUSE,
        authorized_control_call_id="call-pause",
    )
    assert paused.state is RoutineState.PAUSED
    assert paused.revision == 2


def test_routine_tools_are_static_interactive_management_capabilities(
    store: SQLiteStateStore,
) -> None:
    declarations = routine_capability_declarations(_owner(store))
    by_id = {item.id: item for item in declarations.capabilities}
    assert declarations.tool_views[0].name == ROUTINE_LIST_TOOL_NAME
    assert by_id[ROUTINE_CREATE_CAPABILITY_ID].operational_effect is (
        OperationalEffect.MANAGE_SCHEDULED_ROUTINE
    )
    assert all(
        item.automation_eligibility is AutomationEligibility.INTERACTIVE_ONLY
        for item in declarations.capabilities
    )


async def test_new_routine_takes_the_code_defaults(store: SQLiteStateStore) -> None:
    created = await _create(_owner(store))
    # Nothing but what was asked: no named sources means every current source.
    assert created.source_ids == ()
    assert created.state is RoutineState.ACTIVE
    assert created.consecutive_failures == created.occurrence_count == 0
    for missing in ("title", "prompt", "schedule"):
        values = {
            "title": _INTENT.title,
            "prompt": _INTENT.prompt,
            "schedule": _INTENT.schedule,
            missing: None,
        }
        with pytest.raises(RoutineError, match=f"needs {missing}") as refused:
            await _owner(store).prepare(
                RoutineIntent(**values),  # type: ignore[arg-type]
                conversation_id="conversation-1",
                identity=f"missing-{missing}",
            )
        assert refused.value.code == "routine_argument_missing"


async def test_update_keeps_omitted_fields_and_defaults_to_the_current_revision(
    store: SQLiteStateStore,
) -> None:
    sales = await _source(store, "sales")
    owner = _owner(store)
    created = await _create(
        owner,
        RoutineIntent(
            title=_INTENT.title,
            prompt=_INTENT.prompt,
            schedule=_INTENT.schedule,
            sources=("sales",),
        ),
    )
    assert created.source_ids == (sales.id,)
    later = OnceSchedule(NOW + timedelta(days=2))
    # Only the schedule changes; no expected revision means the current one.
    prepared = await owner.prepare(
        RoutineIntent(schedule=later),
        conversation_id="ignored-for-updates",
        identity=created.routine_id,
        routine_id=created.routine_id,
    )
    updated = await owner.save(prepared)
    assert updated.schedule == later
    assert updated.next_due_at == later.exact_at
    assert (updated.title, updated.authorized_instruction, updated.source_ids) == (
        created.title,
        created.authorized_instruction,
        created.source_ids,
    )
    assert updated.conversation_id == created.conversation_id
    assert updated.revision == created.revision + 1
    assert prepared.card["title"] == created.title
    assert prepared.card["sources"] == ("sales",)


async def test_stale_expected_revision_is_refused_and_saves_nothing(
    store: SQLiteStateStore,
) -> None:
    owner = _owner(store)
    created = await _create(owner)
    with pytest.raises(RoutineError) as stale:
        await owner.prepare(
            RoutineIntent(title="Renamed"),
            conversation_id="conversation-1",
            identity=created.routine_id,
            routine_id=created.routine_id,
            expected_revision=created.revision + 1,
        )
    assert stale.value.code == "routine_revision_changed"
    # A revision prepared against the current routine loses to a newer change.
    prepared = await owner.prepare(
        RoutineIntent(title="Renamed"),
        conversation_id="conversation-1",
        identity=created.routine_id,
        routine_id=created.routine_id,
    )
    await owner.control(
        created.routine_id,
        expected_revision=created.revision,
        action=RoutineControlAction.PAUSE,
        authorized_control_call_id="call-pause",
    )
    with pytest.raises(RoutineError) as raced:
        await owner.save(prepared)
    assert raced.value.code == "routine_revision_changed"
    current = await owner.inspect(created.routine_id)
    assert current is not None and current.routine.title == created.title


async def test_run_immediately_is_for_recurring_creation_only(
    store: SQLiteStateStore,
) -> None:
    owner = _owner(store)
    created = await _create(owner)
    with pytest.raises(RoutineError) as revision:
        await owner.prepare(
            RoutineIntent(run_immediately=True),
            conversation_id="conversation-1",
            identity=created.routine_id,
            routine_id=created.routine_id,
        )
    assert revision.value.code == "routine_immediate_creation_only"
    with pytest.raises(RoutineError, match="interval or calendar"):
        await owner.prepare(
            RoutineIntent(
                title="Once",
                prompt="Report once.",
                schedule=OnceSchedule(NOW + timedelta(hours=1)),
                run_immediately=True,
            ),
            conversation_id="conversation-1",
            identity="once-now",
        )


async def test_sources_resolve_by_name_or_id_and_unknown_names_are_refused(
    store: SQLiteStateStore,
) -> None:
    sales = await _source(store, "Sales")
    stock = await _source(store, "stock")
    owner = _owner(store)
    prepared = await owner.prepare(
        RoutineIntent(
            title=_INTENT.title,
            prompt=_INTENT.prompt,
            schedule=_INTENT.schedule,
            sources=("sales", stock.id),
        ),
        conversation_id="conversation-1",
        identity="named",
    )
    assert prepared.routine.source_ids == tuple(sorted((sales.id, stock.id)))
    with pytest.raises(RoutineError) as unknown:
        await owner.prepare(
            RoutineIntent(
                title=_INTENT.title,
                prompt=_INTENT.prompt,
                schedule=_INTENT.schedule,
                sources=("warehouse",),
            ),
            conversation_id="conversation-1",
            identity="unknown",
        )
    assert unknown.value.code == "routine_source_unknown"
    assert "Sales" in str(unknown.value.details)


async def test_card_shows_title_local_schedule_prompt_and_sources(
    store: SQLiteStateStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(owner_module, "system_user_timezone", lambda: "America/Chicago")
    owner = _owner(store)
    relative = await owner.prepare(
        RoutineIntent(
            title="Data quality note",
            prompt="Write a two-sentence note on why data quality matters.",
            schedule=FrozenJsonObject.from_mapping(
                {"kind": "once", "after_seconds": 480}
            ),
        ),
        conversation_id="conversation-1",
        identity="card",
    )
    assert relative.card.to_dict() == {
        "title": "Data quality note",
        "schedule": "Once at Wed 2026-09-23 11:08 CDT (timed from approval)",
        "prompt": "Write a two-sentence note on why data quality matters.",
        "sources": "All current sources",
    }
    recurring = await owner.prepare(
        RoutineIntent(
            title="Hourly sales",
            prompt="Report sales.",
            schedule=IntervalSchedule(3_600, NOW + timedelta(hours=1)),
            run_immediately=True,
        ),
        conversation_id="conversation-1",
        identity="card-recurring",
    )
    assert recurring.card["schedule"] == (
        "Now, then every 1 hour from Wed 2026-09-23 12:00 CDT"
    )


async def test_named_local_instant_does_not_jump_to_next_week_after_approval(
    store: SQLiteStateStore,
) -> None:
    clock = [datetime(2026, 9, 23, 22, 1, tzinfo=UTC)]
    owner = _owner(store, clock)
    named = RoutineIntent(
        title="Wednesday report",
        prompt="Report the week.",
        schedule=FrozenJsonObject.from_mapping(
            {
                "kind": "once_next_weekday",
                "timezone": "America/Chicago",
                "weekday": 3,
                "hour": 17,
                "minute": 0,
            }
        ),
    )
    # The card was shown at 16:59 local; approval landed after 17:00.
    shown_at = datetime(2026, 9, 23, 21, 59, tzinfo=UTC)
    card = await owner.prepare(
        named, conversation_id="conversation-1", identity="named", reference_at=shown_at
    )
    assert card.routine.schedule == OnceSchedule(datetime(2026, 9, 23, 22, tzinfo=UTC))
    with pytest.raises(RoutineError, match="passed before creation"):
        await owner.prepare(
            named,
            conversation_id="conversation-1",
            identity="named",
            reference_at=shown_at,
            relative_at=clock[0],
        )


async def test_resume_resets_the_failure_count(store: SQLiteStateStore) -> None:
    owner = _owner(store)
    created = await _create(owner)
    await store.transition_scheduled_routine(
        "agent-1",
        created.routine_id,
        expected_revision=created.revision,
        state=RoutineState.PAUSED,
        transitioned_at=NOW,
    )
    paused = await owner.inspect(created.routine_id)
    assert paused is not None
    resumed = await owner.control(
        created.routine_id,
        expected_revision=paused.routine.revision,
        action=RoutineControlAction.RESUME,
        authorized_control_call_id="call-resume",
    )
    assert resumed.state is RoutineState.ACTIVE
    assert resumed.consecutive_failures == 0


async def test_sources_accept_the_origin_runs_handles(tmp_path: Path) -> None:
    database = tmp_path / "crm.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE companies (name TEXT NOT NULL)")
    provider = MockModelProvider(
        (ModelResponse(finish_reason=FinishReason.STOP, text="Ready."),)
    )
    agent = await Agent.create(
        "routine-handles",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=provider.model_profile,
    )
    try:
        source = await agent.attach(SQLiteSource(database, name="CRM"))
        origin = await agent.run("Report companies weekly.")
        assert origin.conversation_id is not None
        transcript = await agent.transcript(origin.run_id)
        handle = scope_refs(transcript.run.resolved_source_scope).handles[source.id]
        owner = agent._embedded._routine_owner

        def intent(*sources: str) -> RoutineIntent:
            return RoutineIntent(
                title="Weekly companies",
                prompt="Report the companies.",
                schedule=_INTENT.schedule,
                sources=sources,
            )

        prepared = await owner.prepare(
            intent(handle),
            conversation_id=origin.conversation_id,
            identity="handles",
            origin_run_id=origin.run_id,
        )
        # The handle the model saw is stored as the full source id.
        assert prepared.routine.source_ids == (source.id,)
        for unknown, origin_run_id in (("s99", origin.run_id), (handle, None)):
            # An unissued handle, or any handle without the run that issued it.
            with pytest.raises(RoutineError) as refused:
                await owner.prepare(
                    intent(unknown),
                    conversation_id=origin.conversation_id,
                    identity="handles-unknown",
                    origin_run_id=origin_run_id,
                )
            assert refused.value.code == "routine_source_unknown"
    finally:
        await agent.close()
