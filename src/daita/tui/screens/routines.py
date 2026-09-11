"""Inspect and control scheduled routines from authoritative records."""

from __future__ import annotations

import json

from daita._json import FrozenJsonObject
from daita.routines.capabilities import (
    routine_inspection_projection,
    routine_projection,
)
from ..projection import approval_summary

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Label, OptionList, Static
from textual.widgets.option_list import Option

from daita import (
    RoutineState,
    ScheduledRoutineInspection,
    ScheduledRoutineSummary,
)

from ..sanitization import safe_display, sanitize_terminal_text
from .confirm import ConfirmScreen


class RoutinesScreen(ModalScreen[None]):
    """Bounded lifecycle view over the public Agent routine surface."""

    BINDINGS = [
        Binding("escape", "close", "Back", priority=True),
        Binding("r", "refresh", "Refresh", priority=True),
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
                "Host open in this TUI. Queued assignments share the run lock with chat. "
                "Exit this TUI before starting daita host --agent <name>; stop that host "
                "before reopening here. No progress while every host is closed.",
                id="routines-help",
                markup=False,
            )
            with Horizontal(id="routines-actions"):
                yield Button("Refresh", id="routines-refresh")
                yield Button("Pause", id="routines-pause")
                yield Button("Resume", id="routines-resume")
                yield Button("Run now", id="routines-run-now", variant="primary")
                yield Button("Disable", id="routines-disable", variant="error")
                yield Button("Close", id="routines-close")
            yield Static("", id="routines-error", markup=False)
            yield Footer()

    def on_mount(self) -> None:
        self.run_worker(
            self._handle("refresh"),
            name="routines-initial-load",
            group="routines-interaction",
            exclusive=True,
        )

    def action_close(self) -> None:
        if not self._busy:
            self.dismiss(None)

    def action_refresh(self) -> None:
        self._schedule("refresh")

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
        action = {
            "routines-refresh": "refresh",
            "routines-pause": "pause",
            "routines-resume": "resume",
            "routines-run-now": "run_now",
            "routines-disable": "disable",
        }.get(button_id or "")
        if action is not None:
            self._schedule(action)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        del event
        self._schedule("inspect")

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        del event
        if not self._busy:
            self._schedule("inspect")

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
            if action == "inspect":
                await self._show_inspection(summary.routine_id)
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
        routines = await self.app.controller.list_routines()  # type: ignore[attr-defined]
        self._routines = routines
        option_list = self.query_one("#routines-list", OptionList)
        option_list.clear_options()
        for summary in routines:
            due = (
                "—" if summary.next_due_at is None else summary.next_due_at.isoformat()
            )
            option_list.add_option(
                Option(
                    Text.assemble(
                        (safe_display(summary.title, fallback="Routine"), "bold"),
                        f"  {summary.state.value}  next {due}",
                    ),
                    id=summary.routine_id,
                )
            )
        self.query_one("#routines-summary", Static).update(
            f"{len(routines)} routine{'s' if len(routines) != 1 else ''}"
        )
        wanted = target_id or self._target_routine_id
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
            await self._show_inspection(routines[index].routine_id)
        else:
            self.query_one("#routines-detail", Static).update(
                "No scheduled routines. Use /routines create <instruction> in chat."
            )
        self._update_actions()

    async def _show_inspection(self, routine_id: str) -> None:
        inspection = await self.app.controller.inspect_routine(  # type: ignore[attr-defined]
            routine_id
        )
        if inspection is None:
            raise ValueError("Routine no longer exists.")
        self.query_one("#routines-detail", Static).update(
            render_routine_inspection(inspection)
        )

    def _selected_summary(self) -> ScheduledRoutineSummary | None:
        index = self.query_one("#routines-list", OptionList).highlighted
        if index is None or not 0 <= index < len(self._routines):
            return None
        return self._routines[index]

    def _update_actions(self) -> None:
        summary = self._selected_summary()
        state = None if summary is None else summary.state
        self.query_one("#routines-pause", Button).disabled = (
            self._busy or state is not RoutineState.ACTIVE
        )
        self.query_one("#routines-resume", Button).disabled = (
            self._busy
            or state
            not in {
                RoutineState.PAUSED,
                RoutineState.NEEDS_ATTENTION,
            }
        )
        self.query_one("#routines-run-now", Button).disabled = (
            self._busy or state is not RoutineState.ACTIVE
        )
        self.query_one("#routines-disable", Button).disabled = self._busy or state in {
            None,
            RoutineState.COMPLETED,
            RoutineState.EXPIRED,
            RoutineState.DISABLED,
        }

    def _set_error(self, message: str) -> None:
        self.query_one("#routines-error", Static).update(message)


def render_routine_inspection(inspection: ScheduledRoutineInspection) -> Text:
    projection = routine_inspection_projection(inspection)
    routine = routine_projection(inspection.routine)
    exact = json.dumps(
        FrozenJsonObject.from_mapping(projection).to_dict(),
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    lines = [
        approval_summary(
            json.dumps({"proposal": FrozenJsonObject.from_mapping(routine).to_dict()}),
            "routines.inspect",
        )
    ]
    for item in inspection.recent_occurrences:
        lines.append(
            f"{item.scheduled_for.isoformat()} · {item.disposition.value} · "
            f"{item.failure_code or 'no failure recorded'} · occurrence {item.occurrence_id} · "
            f"run {item.terminal_run_id or item.reserved_run_id or 'not started'}"
        )
    lines.append(
        "Receipt references and artifact/delivery evidence are in the exact details below. Use /effects to inspect or resolve uncertainty."
    )
    lines.append(exact)
    return Text("\n".join(lines))


__all__ = ["RoutinesScreen", "render_routine_inspection"]
