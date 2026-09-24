"""Inspect and control scheduled routines from authoritative records."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Label, OptionList, Static
from textual.widgets.option_list import Option

from daita import (
    RoutineState,
    ScheduledRoutineInspection,
    ScheduledRoutineSummary,
)
from daita._json import FrozenJsonObject
from daita.routines.capabilities import (
    routine_inspection_projection,
)

from ..sanitization import safe_display, sanitize_terminal_text
from .confirm import ConfirmScreen


class RoutinesScreen(ModalScreen[None]):
    """Bounded lifecycle view over the public Agent routine surface."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Back", priority=True),
        Binding("r", "refresh", "Refresh", priority=True),
        Binding("i", "details", "Details", priority=True),
        Binding("x", "record", "Full record", priority=True),
        Binding("p", "pause", "Pause", priority=True),
        Binding("u", "resume", "Resume", priority=True),
        Binding("n", "run_now", "Run now", priority=True),
        Binding("d", "disable", "Disable", priority=True),
    ]

    def __init__(self, *, routine_id: str | None = None) -> None:
        super().__init__()
        self._routines: tuple[ScheduledRoutineSummary, ...] = ()
        self._target_routine_id = routine_id
        self._busy = False

    def compose(self) -> ComposeResult:
        with Vertical(id="routines-manager"):
            yield Label("Saved assignments", id="routines-title", markup=False)
            yield Static("Loading…", id="routines-summary", markup=False)
            yield OptionList(id="routines-list")
            with VerticalScroll(id="routines-detail-scroll"):
                yield Static("", id="routines-detail", markup=False)
            yield Static(
                "Assignments run while this agent is open and share the run lock with chat.",
                id="routines-help",
                markup=False,
            )
            with Horizontal(id="routines-actions"):
                yield Button("Refresh", id="routines-refresh")
                yield Button("Details", id="routines-details", variant="primary")
                yield Button("Full record", id="routines-record")
                yield Button("Pause", id="routines-toggle")
                yield Button("Run now", id="routines-run-now")
                yield Button("Disable", id="routines-disable", variant="error")
                yield Button("Close", id="routines-close")
            yield Static("", id="routines-error", markup=False)
            yield Footer()

    def on_mount(self) -> None:
        self._apply_responsive_layout()
        self.run_worker(
            self._handle("refresh"),
            name="routines-initial-load",
            group="routines-interaction",
            exclusive=True,
        )

    def on_resize(self) -> None:
        self._apply_responsive_layout()

    def _apply_responsive_layout(self) -> None:
        if self.is_mounted:
            self.set_class(
                self.size.width < 90 or self.size.height < 30,
                "-compact",
            )

    def action_close(self) -> None:
        if not self._busy:
            self.dismiss(None)

    def action_refresh(self) -> None:
        self._schedule("refresh")

    def action_details(self) -> None:
        self._schedule("details")

    def action_record(self) -> None:
        self._schedule("record")

    def action_pause(self) -> None:
        self._schedule("pause")

    def action_resume(self) -> None:
        self._schedule("resume")

    def action_run_now(self) -> None:
        self._schedule("run_now")

    def action_disable(self) -> None:
        self._schedule("disable")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "routines-close":
            self.action_close()
            return
        if button_id == "routines-toggle":
            summary = self._selected_summary()
            if summary is not None:
                self._schedule(
                    "pause" if summary.state is RoutineState.ACTIVE else "resume"
                )
            return
        action = {
            "routines-refresh": "refresh",
            "routines-details": "details",
            "routines-record": "record",
            "routines-run-now": "run_now",
            "routines-disable": "disable",
        }.get(button_id or "")
        if action is not None:
            self._schedule(action)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        del event
        self._schedule("details")

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        del event
        if not self._busy:
            summary = self._selected_summary()
            if summary is not None:
                self._render_overview(summary)
        self._update_actions()

    def _schedule(self, action: str) -> None:
        if self._busy:
            return
        self.run_worker(
            self._handle(action),
            name=f"routines-{action}",
            group="routines-interaction",
            exclusive=True,
        )

    async def _handle(self, action: str) -> None:
        self._busy = True
        self._set_error("")
        try:
            if action == "refresh":
                await self._refresh()
                return
            summary = self._selected_summary()
            if summary is None:
                raise ValueError("Select a routine first.")
            if action in {"details", "record"}:
                await self._show_inspection(
                    summary.routine_id, exact=action == "record"
                )
                return
            accepted = await self.app._await_modal(  # type: ignore[attr-defined]
                ConfirmScreen(
                    f"{action.replace('_', ' ').title()} routine "
                    f"{safe_display(summary.title, fallback='selected routine')}?"
                )
            )
            if not accepted:
                return
            await self.app.controller.control_routine(  # type: ignore[attr-defined]
                summary.routine_id,
                expected_revision=summary.revision,
                action=action,
            )
            await self._refresh(target_id=summary.routine_id)
        except (ValueError, RuntimeError, OSError) as error:
            self._set_error(
                sanitize_terminal_text(
                    str(error),
                    maximum=512,
                    preserve_lines=False,
                    fallback="Routine action failed.",
                )
            )
        finally:
            self._busy = False
            self._update_actions()

    async def _refresh(self, *, target_id: str | None = None) -> None:
        wanted = target_id or self._target_routine_id or self._selected_routine_id()
        routines = await self.app.controller.list_routines()  # type: ignore[attr-defined]
        self._routines = routines
        option_list = self.query_one("#routines-list", OptionList)
        option_list.clear_options()
        for summary in routines:
            option_list.add_option(
                Option(
                    Text(render_routine_list_label(summary)),
                    id=summary.routine_id,
                )
            )
        active = sum(item.state is RoutineState.ACTIVE for item in routines)
        self.query_one("#routines-summary", Static).update(
            f"{len(routines)} assignment{'s' if len(routines) != 1 else ''}  ·  "
            f"{active} active"
        )
        self._target_routine_id = None
        if routines:
            index = next(
                (
                    position
                    for position, item in enumerate(routines)
                    if item.routine_id == wanted
                ),
                0,
            )
            option_list.highlighted = index
            self._render_overview(routines[index])
        else:
            self.query_one("#routines-detail", Static).update(
                "No saved assignments yet. Use /routines create <instruction> in chat."
            )
        option_list.focus()
        self._update_actions()

    async def _show_inspection(self, routine_id: str, *, exact: bool) -> None:
        inspection = await self.app.controller.inspect_routine(  # type: ignore[attr-defined]
            routine_id
        )
        if inspection is None:
            raise ValueError("Routine no longer exists.")
        self.query_one("#routines-detail", Static).update(
            render_routine_inspection(inspection)
            if exact
            else render_routine_details(inspection)
        )

    def _selected_routine_id(self) -> str | None:
        listing = self.query_one("#routines-list", OptionList)
        if listing.highlighted is None:
            return None
        option = listing.get_option_at_index(listing.highlighted)
        return str(option.id) if option.id is not None else None

    def _selected_summary(self) -> ScheduledRoutineSummary | None:
        selected = self._selected_routine_id()
        return next(
            (item for item in self._routines if item.routine_id == selected), None
        )

    def _render_overview(self, summary: ScheduledRoutineSummary) -> None:
        self.query_one("#routines-detail", Static).update(
            render_routine_overview(summary)
        )

    def _update_actions(self) -> None:
        summary = self._selected_summary()
        state = None if summary is None else summary.state
        self.query_one("#routines-refresh", Button).disabled = self._busy
        self.query_one("#routines-details", Button).disabled = (
            self._busy or summary is None
        )
        self.query_one("#routines-record", Button).disabled = (
            self._busy or summary is None
        )
        toggle = self.query_one("#routines-toggle", Button)
        toggle.label = (
            "Resume"
            if state in {RoutineState.PAUSED, RoutineState.NEEDS_ATTENTION}
            else "Pause"
        )
        toggle.disabled = self._busy or state not in {
            RoutineState.ACTIVE,
            RoutineState.PAUSED,
            RoutineState.NEEDS_ATTENTION,
        }
        self.query_one("#routines-run-now", Button).disabled = (
            self._busy or state is not RoutineState.ACTIVE
        )
        self.query_one("#routines-disable", Button).disabled = self._busy or state in {
            None,
            RoutineState.COMPLETED,
            RoutineState.EXPIRED,
            RoutineState.DISABLED,
        }
        self.query_one("#routines-close", Button).disabled = self._busy

    def _set_error(self, message: str) -> None:
        error = self.query_one("#routines-error", Static)
        error.update(message)
        error.set_class(bool(message), "-visible")


def render_routine_list_label(summary: ScheduledRoutineSummary) -> str:
    state = summary.state.value.replace("_", " ").upper()
    title = safe_display(summary.title, fallback="Assignment", maximum=40)
    due = (
        "no next run"
        if summary.next_due_at is None
        else summary.next_due_at.strftime("%Y-%m-%d %H:%M UTC")
    )
    return f"{state:<16} {title}  ·  {due}"


def render_routine_overview(summary: ScheduledRoutineSummary) -> str:
    """Keep selection changes brief; exact contracts are an explicit action."""
    return "\n".join(
        (
            safe_display(summary.title, fallback="Assignment", maximum=128),
            f"State: {summary.state.value.replace('_', ' ')}  ·  revision {summary.revision}",
            f"Schedule: {summary.schedule_kind.value.replace('_', ' ')}",
            "Next run: "
            + (
                "none"
                if summary.next_due_at is None
                else summary.next_due_at.isoformat()
            ),
            f"Occurrences: {summary.occurrence_count}  ·  consecutive failures: {summary.consecutive_failures}",
            "Sensitivity ceiling: " + summary.sensitivity_ceiling.value,
            "ID: " + safe_display(summary.routine_id, fallback="routine", maximum=256),
            "\nChoose Details for the instruction and recent runs, or Full record for exact contracts and evidence.",
        )
    )


def _schedule_text(schedule: Mapping[str, object]) -> str:
    kind = schedule.get("kind")
    if kind == "once":
        return f"Once at {schedule.get('exact_at')}"
    if kind == "interval":
        return (
            f"Every {schedule.get('interval_seconds')} seconds "
            f"from {schedule.get('anchor_at')}"
        )
    if kind == "calendar":
        return (
            f"Calendar at {schedule.get('hour')}:{str(schedule.get('minute')).zfill(2)} "
            f"in {schedule.get('timezone')} · {schedule.get('day_selector')}"
        )
    return "Unknown schedule"


def render_routine_details(inspection: ScheduledRoutineInspection) -> str:
    """Readable lifecycle detail from the routine domain's current projection."""
    projection = routine_inspection_projection(inspection)
    routine = projection["routine"]
    assert isinstance(routine, dict)
    schedule = routine["schedule"]
    assert isinstance(schedule, dict)
    instruction = sanitize_terminal_text(
        routine["authorized_instruction"],
        maximum=1200,
        preserve_lines=True,
        fallback="(empty instruction)",
    )
    lines = [
        safe_display(routine["title"], fallback="Assignment", maximum=128),
        f"State: {routine['state']}  ·  revision {inspection.routine.revision}",
        f"ID: {safe_display(routine['routine_id'], fallback='routine', maximum=256)}",
        "",
        "Schedule",
        _schedule_text(schedule),
        f"Next run: {routine['next_due_at'] or 'none'}  ·  missed slots: {routine['misfire_policy']}",
        f"Expires: {routine['expires_at']}",
        "",
        "Instruction",
        instruction,
        "",
        "Limits and access",
        f"Per run: {routine['per_run_max_tokens']} tokens · ${routine['per_run_max_cost_usd']} estimated model cost",
        f"Total: {routine['cumulative_max_tokens']} tokens · ${routine['cumulative_max_cost_usd']} · {routine['cumulative_max_attempts']} attempts · {routine['cumulative_max_occurrences']} occurrences",
        f"Used: {routine['charged_tokens']} tokens · ${routine['charged_cost_usd']} · {routine['attempt_count']} attempts",
        f"Scope: {len(routine['allowed_source_ids'])} sources · {len(routine['allowed_resource_ids'])} resources · {len(routine['allowed_connector_binding_ids'])} connectors · {len(routine['allowed_capability_ids'])} capabilities",
        f"Sensitivity ceiling: {routine['sensitivity_ceiling']}",
        "",
        "Recent runs",
    ]
    occurrences = projection["recent_occurrences"]
    assert isinstance(occurrences, tuple)
    if not occurrences:
        lines.append("No occurrences yet.")
    for item in occurrences:
        assert isinstance(item, dict)
        failure = "" if item["failure_code"] is None else f" · {item['failure_code']}"
        lines.append(
            f"{item['scheduled_for']} · {item['disposition']}"
            f"{failure} · {len(item['delivery_ids'])} deliveries · "
            f"{len(item['effect_receipt_ids'])} effect receipts"
        )
    lines.append("\nFull record shows exact bindings, receipt IDs and delivery IDs.")
    return sanitize_terminal_text(
        "\n".join(lines),
        maximum=8000,
        preserve_lines=True,
        fallback="Assignment details unavailable.",
    )


def render_routine_inspection(inspection: ScheduledRoutineInspection) -> Text:
    """Expose the exact current projection only when the user requests it."""
    projection = routine_inspection_projection(inspection)
    exact = json.dumps(
        FrozenJsonObject.from_mapping(projection).to_dict(),
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    return Text("Full record\n" + exact)


__all__ = [
    "RoutinesScreen",
    "render_routine_details",
    "render_routine_inspection",
    "render_routine_overview",
]
