"""List background jobs and show one: checklist, sub-runs, question, answer."""

from __future__ import annotations

from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from daita.jobs import ChecklistStatus, DelegationState, Job, JobState

from ..sanitization import safe_display, sanitize_terminal_text
from .confirm import ConfirmScreen

_LIVE = frozenset({JobState.RUNNING, JobState.WAITING_FOR_YOU})
_LIMIT_FAILURES = frozenset({"job_run_limit_reached", "job_token_limit_reached"})
_ITEM_MARKS = {
    ChecklistStatus.PENDING: "[ ]",
    ChecklistStatus.IN_PROGRESS: "[~]",
    ChecklistStatus.DONE: "[x]",
    ChecklistStatus.CANCELLED: "[-]",
}


class JobsScreen(ModalScreen[None]):
    """The agent's background jobs; answer or cancel the selected one."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Back", priority=True),
        Binding("r", "refresh", "Refresh"),
        Binding("c", "cancel_job", "Cancel"),
    ]

    def __init__(self, *, job_id: str | None = None) -> None:
        super().__init__()
        self._target_job_id = job_id
        self._jobs: tuple[Job, ...] = ()
        self._busy = False

    def compose(self) -> ComposeResult:
        with Vertical(id="jobs-manager"):
            yield Label("Background jobs", id="jobs-title", markup=False)
            yield Static("Loading…", id="jobs-summary", markup=False)
            yield Static("", id="jobs-notice", markup=False)
            yield OptionList(id="jobs-list")
            with VerticalScroll(id="jobs-detail-scroll"):
                yield Static("", id="jobs-detail", markup=False)
            yield Input(placeholder="Answer the job's question", id="jobs-answer")
            yield Static(
                "Jobs keep going in the host and ask you when they need you.",
                id="jobs-help",
                markup=False,
            )
            with Horizontal(id="jobs-actions"):
                yield Button("Answer", id="jobs-answer-send", variant="primary")
                yield Button("Refresh", id="jobs-refresh")
                yield Button("Cancel job", id="jobs-cancel")
                yield Button("Close", id="jobs-close")
            yield Static("", id="jobs-error", markup=False)
            yield Footer()

    def on_mount(self) -> None:
        self._schedule("refresh")

    def action_close(self) -> None:
        if not self._busy:
            self.dismiss(None)

    def action_refresh(self) -> None:
        self._schedule("refresh")

    def action_cancel_job(self) -> None:
        self._schedule("cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        actions = {
            "jobs-answer-send": "answer",
            "jobs-refresh": "refresh",
            "jobs-cancel": "cancel",
        }
        if event.button.id == "jobs-close":
            self.action_close()
        elif (action := actions.get(event.button.id or "")) is not None:
            self._schedule(action)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "jobs-answer":
            self._schedule("answer")

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        del event
        self._render_selected()

    def _schedule(self, action: str) -> None:
        if not self._busy:
            self.run_worker(
                self._handle(action),
                name=f"jobs-{action}",
                group="jobs-interaction",
                exclusive=True,
            )

    async def _handle(self, action: str) -> None:
        self._set_busy(True)
        self.query_one("#jobs-error", Static).update("")
        controller = self.app.controller  # type: ignore[attr-defined]
        try:
            job = self._selected()
            if action == "answer":
                if job is None:
                    raise ValueError("Select a job first.")
                answer = self.query_one("#jobs-answer", Input)
                if not answer.value.strip():
                    raise ValueError("Type an answer first.")
                await controller.answer_job(job.job_id, answer.value)
                answer.value = ""
                self._notice("Answered; the job continues.")
            elif action == "cancel":
                if job is None or job.state not in _LIVE:
                    raise ValueError("Select a running or waiting job first.")
                accepted = await self.app._await_modal(  # type: ignore[attr-defined]
                    ConfirmScreen(
                        "Cancel this job?\n"
                        + safe_display(job.objective, fallback="job", maximum=128)
                        + "\n\nIts runs stop now; this cannot be undone."
                    )
                )
                if not accepted:
                    return
                await controller.cancel_job(job.job_id)
                self._notice("Job cancelled.")
            await self._load()
        except (ValueError, RuntimeError, OSError) as error:
            self.query_one("#jobs-error", Static).update(
                sanitize_terminal_text(
                    str(error),
                    maximum=512,
                    preserve_lines=False,
                    fallback="Job action failed.",
                )
            )
        finally:
            if self.is_mounted:
                self._set_busy(False)

    async def _load(self) -> None:
        selected = self._selected_id() or self._target_job_id
        self._jobs = await self.app.controller.list_jobs()  # type: ignore[attr-defined]
        listing = self.query_one("#jobs-list", OptionList)
        listing.clear_options()
        for job in self._jobs:
            listing.add_option(Option(Text(_list_label(job)), id=job.job_id))
        if self._jobs:
            listing.highlighted = next(
                (i for i, job in enumerate(self._jobs) if job.job_id == selected), 0
            )
            listing.focus()
        live = sum(job.state in _LIVE for job in self._jobs)
        waiting = sum(job.state is JobState.WAITING_FOR_YOU for job in self._jobs)
        self.query_one("#jobs-summary", Static).update(
            f"{len(self._jobs)} jobs  ·  {live} live  ·  {waiting} waiting for you"
        )
        self._render_selected()

    def _notice(self, text: str) -> None:
        self.query_one("#jobs-notice", Static).update(text)

    def _selected_id(self) -> str | None:
        listing = self.query_one("#jobs-list", OptionList)
        if listing.highlighted is None:
            return None
        option = listing.get_option_at_index(listing.highlighted)
        return None if option.id is None else str(option.id)

    def _selected(self) -> Job | None:
        selected = self._selected_id()
        return next((job for job in self._jobs if job.job_id == selected), None)

    def _render_selected(self) -> None:
        if not self.is_mounted:
            return
        job = self._selected()
        self.query_one("#jobs-detail", Static).update(
            "No background jobs yet. Ask the agent to start one for long or "
            "multi-part work."
            if job is None
            else render_job(job)
        )
        self._update_actions()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._update_actions()

    def _update_actions(self) -> None:
        if not self.is_mounted:
            return
        job = self._selected()
        answerable = job is not None and (
            job.state is JobState.WAITING_FOR_YOU
            or (job.state is JobState.FAILED and job.failure_code in _LIMIT_FAILURES)
        )
        self.query_one("#jobs-answer", Input).disabled = self._busy or not answerable
        self.query_one("#jobs-answer-send", Button).disabled = (
            self._busy or not answerable
        )
        self.query_one("#jobs-cancel", Button).disabled = (
            self._busy or job is None or job.state not in _LIVE
        )
        self.query_one("#jobs-refresh", Button).disabled = self._busy
        self.query_one("#jobs-close", Button).disabled = self._busy


def _clean(value: str, maximum: int, *, lines: bool = False) -> str:
    return sanitize_terminal_text(
        value, maximum=maximum, preserve_lines=lines, fallback=""
    )


def _list_label(job: Job) -> str:
    done, total = job.progress
    progress = f" · {done} of {total} done" if total else ""
    return _clean(
        f"{job.state.value.replace('_', ' ').upper():<16} "
        f"{' '.join(job.objective.split())[:80]}{progress}",
        512,
    )


def render_job(job: Job) -> str:
    """One job for the terminal; every model-written value is sanitized."""

    done, total = job.progress
    lines = [
        _clean(job.objective, 2_000, lines=True),
        "",
        f"State: {job.state.value.replace('_', ' ')}"
        + (f" ({_clean(job.failure_code, 128)})" if job.failure_code else "")
        + f"  ·  {done} of {total} done  ·  run {job.run_count} of {job.run_limit}",
        f"Updated: {job.updated_at.strftime('%Y-%m-%d %H:%M UTC')}",
    ]
    if job.question:
        lines += ["", "Question for you:", _clean(job.question, 2_000, lines=True)]
    elif job.state is JobState.FAILED and job.failure_code in _LIMIT_FAILURES:
        lines += ["", "The job reached its limits. Answer to raise them and continue."]
    if job.checklist:
        lines += ["", "Checklist:"]
        for item in job.checklist:
            after = f"  (after {', '.join(item.after)})" if item.after else ""
            lines.append(
                f"  {_ITEM_MARKS[item.status]} {_clean(item.title, 200)}"
                f"{_clean(after, 300)}"
            )
    if job.delegations:
        lines += ["", "Sub-runs:"]
        for delegation in job.delegations[-12:]:
            mark = "…" if delegation.state is DelegationState.RUNNING else "·"
            lines.append(
                f"  {mark} {delegation.state.value}: {_clean(delegation.goal, 200)}"
            )
            if delegation.summary:
                lines.append("    " + _clean(delegation.summary, 400))
    if job.answers:
        lines += ["", "Your answers:"]
        lines += ["  " + _clean(answer.text, 400) for answer in job.answers[-5:]]
    if job.final_summary:
        lines += ["", "Result:", _clean(job.final_summary, 8_000, lines=True)]
    return "\n".join(lines)


__all__ = ["JobsScreen", "render_job"]
