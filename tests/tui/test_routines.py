"""Component-owned tests split from ``test_screens.py``."""

from __future__ import annotations

from tests.tui._support import (
    UTC,
    Button,
    ConfirmScreen,
    DaitaApp,
    ModelSensitivity,
    OptionList,
    RoutinesScreen,
    RoutineState,
    ScheduledRoutineInspection,
    ScheduledRoutineSummary,
    ScheduleKind,
    Static,
    datetime,
    replace,
    workspace_for,
)


def test_routine_approval_summary_shows_relative_and_local_timing_intent() -> None:
    import json

    from daita.tui.projection import approval_summary

    proposal = {
        "title": "Follow up",
        "revision": 1,
        "authorized_instruction": "Report once.",
        "schedule": {"kind": "once", "exact_at": "2026-09-23T22:00:00+00:00"},
        "misfire_policy": "latest_only",
    }
    relative = approval_summary(
        json.dumps(
            {
                "proposal": proposal,
                "timing_intent": {
                    "schedule": {"kind": "once", "after_seconds": 300},
                    "expires_after_seconds": 3600,
                },
                "timing_note": "The exact time is resolved after approval.",
            }
        ),
        "routines.create",
    )
    assert "Requested delay: 300 seconds after approval" in relative
    assert "Requested lifetime: 3600 seconds after approval" in relative
    assert "The exact time is resolved after approval" in relative

    local = approval_summary(
        json.dumps(
            {
                "proposal": proposal,
                "timing_intent": {
                    "schedule": {
                        "kind": "once_next_weekday",
                        "timezone": "America/Chicago",
                        "weekday": 3,
                        "hour": 17,
                        "minute": 0,
                    },
                },
            }
        ),
        "routines.create",
    )
    assert "next ISO weekday 3 at 17:00 in America/Chicago" in local


async def test_routines_command_opens_records_and_routes_create_through_agent_loop():
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    listing = await app.controller.dispatch_command("/routines")
    assert listing.kind == "screen"
    assert listing.screen == "routines"
    create = await app.controller.dispatch_command(
        "/routines create report the current invoice count every morning"
    )
    assert create.kind == "run"
    assert create.run_message is not None
    assert "routine management tools" in create.run_message
    update = await app.controller.dispatch_command(
        "/routines update routine-1 report the current invoice count every afternoon"
    )
    assert update.kind == "run"
    assert update.run_message is not None
    assert "exact current revision" in update.run_message


async def test_routines_screen_lists_authoritative_state_and_controls(monkeypatch):
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    observed = datetime(2026, 8, 28, 12, tzinfo=UTC)
    summary = ScheduledRoutineSummary(
        sensitivity_ceiling=ModelSensitivity.INTERNAL,
        routine_id="routine-ui",
        title="Invoice count",
        state=RoutineState.ACTIVE,
        schedule_kind=ScheduleKind.INTERVAL,
        next_due_at=observed,
        revision=1,
        occurrence_count=0,
        consecutive_failures=0,
    )
    current = summary
    controls: list[tuple[str, int, str]] = []

    def inspection() -> ScheduledRoutineInspection:
        from tests.routines._storage_support import routine_record

        routine = replace(
            routine_record(
                routine_id=current.routine_id,
                state=current.state,
                next_due_at=current.next_due_at,
            ),
            title=current.title,
            revision=current.revision,
        )
        return ScheduledRoutineInspection(routine, ())

    async def list_routines() -> tuple[ScheduledRoutineSummary, ...]:
        return (current,)

    async def inspect_routine(routine_id: str) -> ScheduledRoutineInspection | None:
        return inspection() if routine_id == current.routine_id else None

    async def control_routine(
        routine_id: str, *, expected_revision: int, action: str
    ) -> object:
        nonlocal current
        controls.append((routine_id, expected_revision, action))
        current = replace(
            current,
            state=RoutineState.PAUSED,
            revision=current.revision + 1,
            next_due_at=None,
        )
        return inspection().routine

    monkeypatch.setattr(app.controller, "list_routines", list_routines)
    monkeypatch.setattr(app.controller, "inspect_routine", inspect_routine)
    monkeypatch.setattr(app.controller, "control_routine", control_routine)

    async with app.run_test(size=(110, 36)) as pilot:
        await app.push_screen(RoutinesScreen())
        for _ in range(20):
            await pilot.pause(0.05)
            if "1 routine" in str(
                app.screen.query_one("#routines-summary", Static).content
            ):
                break
        manager = app.screen
        assert isinstance(manager, RoutinesScreen)
        assert manager.query_one("#routines-list", OptionList).option_count == 1
        assert "instruction_digest" in str(
            manager.query_one("#routines-detail", Static).content
        )
        assert await pilot.click("#routines-pause") is True
        for _ in range(20):
            await pilot.pause(0.05)
            if isinstance(app.screen, ConfirmScreen):
                break
        assert isinstance(app.screen, ConfirmScreen)
        await pilot.press("y")
        for _ in range(20):
            await pilot.pause(0.05)
            if app.screen is manager and controls:
                break
        assert controls == [("routine-ui", 1, "pause")]
        assert manager.query_one("#routines-resume", Button).disabled is False
        app.exit(0)
