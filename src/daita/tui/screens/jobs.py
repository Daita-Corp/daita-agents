"""List, inspect, cancel, and render durable jobs in the terminal UI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import ClassVar
from uuid import uuid4

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from daita.jobs import (
    ControlKind,
    ControlState,
    GraphBoardProjection,
    GraphInspection,
    GraphJob,
    GraphState,
    GraphTimelinePage,
    TaskControl,
    TaskResult,
)

from ..projection import bounded_json_text
from ..sanitization import safe_display, sanitize_terminal_text
from .confirm import ConfirmScreen

_CANCELABLE_STATES = frozenset(
    {
        GraphState.QUEUED,
        GraphState.ACTIVE,
        GraphState.BLOCKED,
        GraphState.NEEDS_ATTENTION,
    }
)
_CANCELLATION_STATES = frozenset({GraphState.CANCEL_REQUESTED, GraphState.CANCELLED})


class JobsScreen(ModalScreen[None]):
    """List and operate on jobs owned by the current agent."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Back", priority=True),
        Binding("r", "refresh", "Refresh", priority=True),
        Binding("d", "details", "Details", priority=True),
        Binding("b", "board", "Board", priority=True),
        Binding("t", "timeline", "Timeline", priority=True),
        Binding("k", "controls", "Controls", priority=True),
        Binding("o", "results", "Results", priority=True),
        Binding("c", "cancel_job", "Cancel", priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._jobs: tuple[GraphJob, ...] = ()
        self._notice = ""
        self._busy = False

    def compose(self) -> ComposeResult:
        with Vertical(id="jobs-manager"):
            yield Label("Durable jobs", id="jobs-title", markup=False)
            yield Static("Loading…", id="jobs-summary", markup=False)
            yield Static(self._notice, id="jobs-notice", markup=False)
            yield OptionList(id="jobs-list")
            with VerticalScroll(id="jobs-detail-scroll"):
                yield Static("", id="jobs-detail", markup=False)
            yield Static(
                "Jobs run only while this agent is open. Lifecycle actions do not use the model.",
                id="jobs-help",
                markup=False,
            )
            with Horizontal(id="jobs-actions"):
                yield Button("Refresh", id="jobs-refresh")
                yield Button("Details", id="jobs-details", variant="primary")
                yield Button("Board", id="jobs-board")
                yield Button("Timeline", id="jobs-timeline")
                yield Button("Controls", id="jobs-controls")
                yield Button("Results", id="jobs-results")
                yield Button("Cancel job", id="jobs-cancel")
                yield Button("Close", id="jobs-close")
            yield Static("", id="jobs-error", markup=False)
            yield Footer()

    def on_mount(self) -> None:
        self.run_worker(
            self._load_initial(),
            name="jobs-initial-load",
            group="jobs-interaction",
            exclusive=True,
        )

    def action_close(self) -> None:
        if not self._busy:
            self.dismiss(None)

    def action_refresh(self) -> None:
        self._schedule("refresh")

    def action_details(self) -> None:
        self._schedule("details")

    def action_results(self) -> None:
        self._schedule("results")

    def action_board(self) -> None:
        self._schedule("board")

    def action_timeline(self) -> None:
        self._schedule("timeline")

    def action_controls(self) -> None:
        self._schedule("controls")

    def action_cancel_job(self) -> None:
        self._schedule("cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "jobs-close":
            self.action_close()
            return
        actions = {
            "jobs-refresh": "refresh",
            "jobs-details": "details",
            "jobs-board": "board",
            "jobs-timeline": "timeline",
            "jobs-controls": "controls",
            "jobs-results": "results",
            "jobs-cancel": "cancel",
        }
        action = actions.get(button_id or "")
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
            self._handle_action(action),
            name=f"jobs-{action}",
            group="jobs-interaction",
            exclusive=True,
        )

    async def _load_initial(self) -> None:
        self._set_busy(True)
        try:
            await self._load_jobs()
            self.query_one("#jobs-list", OptionList).focus()
            summary = self._selected_summary()
            if summary is not None:
                self._render_overview(summary)
        except (ValueError, RuntimeError, OSError) as error:
            self._show_error(error)
        finally:
            if self.is_mounted:
                self._set_busy(False)

    async def _handle_action(self, action: str) -> None:
        self._set_busy(True)
        self.query_one("#jobs-error", Static).update("")
        try:
            if action == "refresh":
                await self._load_jobs()
                self._notice = "Job statuses refreshed."
                self.query_one("#jobs-notice", Static).update(self._notice)
                summary = self._selected_summary()
                if summary is not None:
                    self._render_overview(summary)
                return
            summary = self._selected_summary()
            if summary is None:
                raise ValueError("Select a job first.")
            if action == "details":
                await self._show_details(summary.job_id)
            elif action == "board":
                await self._show_board(summary.job_id)
            elif action == "timeline":
                await self._show_timeline(summary.job_id)
            elif action == "controls":
                await self._show_controls(summary.job_id)
            elif action == "results":
                await self._show_results(summary.job_id)
            elif action == "cancel":
                await self._cancel(summary.job_id)
        except (ValueError, RuntimeError, OSError) as error:
            self._show_error(error)
        finally:
            if self.is_mounted:
                self._set_busy(False)

    async def _load_jobs(self) -> None:
        selected = self._selected_job_id()
        self._jobs = await self.app.controller.list_jobs()  # type: ignore[attr-defined]
        listing = self.query_one("#jobs-list", OptionList)
        listing.clear_options()
        for summary in self._jobs:
            listing.add_option(
                Option(Text(self._list_label(summary)), id=summary.job_id)
            )
        if self._jobs:
            selected_index = next(
                (
                    index
                    for index, summary in enumerate(self._jobs)
                    if summary.job_id == selected
                ),
                0,
            )
            listing.highlighted = selected_index
        self.query_one("#jobs-summary", Static).update(self._summary_text())
        if not self._jobs:
            self.query_one("#jobs-detail", Static).update(
                "No durable jobs yet. Ask the agent to start a data profile when one is needed."
            )
        self._update_actions()

    async def _show_details(self, job_id: str) -> None:
        inspection = await self.app.controller.inspect_job(job_id)  # type: ignore[attr-defined]
        if inspection is None:
            raise ValueError("No durable job with that ID belongs to this agent.")
        self.query_one("#jobs-detail", Static).update(render_job_inspection(inspection))

    async def _show_results(self, job_id: str) -> None:
        result = await self.app.controller.read_job_result(job_id)  # type: ignore[attr-defined]
        if result is None:
            inspection = await self.app.controller.inspect_job(job_id)  # type: ignore[attr-defined]
            if inspection is None:
                raise ValueError("No durable job with that ID belongs to this agent.")
            raise ValueError(
                "Results are not available while this job is "
                + inspection.job.state.value
                + "."
            )
        self.query_one("#jobs-detail", Static).update(render_job_result(result))

    async def _show_board(self, job_id: str) -> None:
        board = await self.app.controller.job_board(job_id)  # type: ignore[attr-defined]
        if board is None:
            raise ValueError("No durable job with that ID belongs to this agent.")
        self.query_one("#jobs-detail", Static).update(render_job_board(board))

    async def _show_timeline(self, job_id: str) -> None:
        page = await self.app.controller.job_timeline(  # type: ignore[attr-defined]
            job_id,
            after_event_id=0,
            limit=100,
        )
        if page is None:
            raise ValueError("No durable job with that ID belongs to this agent.")
        self.query_one("#jobs-detail", Static).update(render_job_timeline(page))

    async def _show_controls(self, job_id: str) -> None:
        inspection = await self.app.controller.inspect_job(job_id)  # type: ignore[attr-defined]
        if inspection is None:
            raise ValueError("No durable job with that ID belongs to this agent.")
        if not any(item.state is ControlState.OPEN for item in inspection.controls):
            raise ValueError("This job has no open controls.")
        decision = await self.app._await_modal(  # type: ignore[attr-defined]
            JobControlScreen(inspection)
        )
        if decision is None:
            return
        await self._apply_control_decision(inspection, decision)
        await self._load_jobs()
        refreshed = await self.app.controller.inspect_job(job_id)  # type: ignore[attr-defined]
        if refreshed is not None:
            self.query_one("#jobs-detail", Static).update(
                render_job_inspection(refreshed)
            )

    async def _apply_control_decision(
        self,
        inspection: GraphInspection,
        decision: JobControlDecision,
    ) -> None:
        controller = self.app.controller  # type: ignore[attr-defined]
        key = "tui-" + uuid4().hex
        if decision.action == "answer":
            assert decision.answer is not None
            await controller.answer_job_input(
                inspection.job.job_id,
                decision.task_id,
                decision.control_id,
                answer=decision.answer,
                idempotency_key=key,
            )
        elif decision.action == "accept_review":
            await controller.accept_job_review(
                inspection.job.job_id,
                decision.task_id,
                decision.control_id,
                rationale=decision.text,
                idempotency_key=key,
            )
        elif decision.action == "request_changes":
            await controller.request_job_review_changes(
                inspection.job.job_id,
                decision.task_id,
                decision.control_id,
                rationale=decision.text,
                replacement_guidance=decision.text,
                idempotency_key=key,
            )
        elif decision.action == "retry":
            await controller.retry_job_control(
                inspection.job.job_id,
                decision.task_id,
                decision.control_id,
                advisory_note=decision.text,
                idempotency_key=key,
            )
        elif decision.action == "replace":
            await controller.replace_job_task(
                inspection.job.job_id,
                decision.task_id,
                advisory_note=decision.text,
                idempotency_key=key,
                expected_revision=inspection.graph.revision,
            )
        elif decision.action == "reject":
            await controller.reject_job_control(
                inspection.job.job_id,
                decision.task_id,
                decision.control_id,
                reason=decision.text,
                idempotency_key=key,
            )
        else:  # pragma: no cover - closed by JobControlScreen
            raise ValueError("Unsupported graph control decision.")
        self._notice = "Graph control applied through the job owner."
        self.query_one("#jobs-notice", Static).update(self._notice)

    async def _cancel(self, job_id: str) -> None:
        inspection = await self.app.controller.inspect_job(job_id)  # type: ignore[attr-defined]
        if inspection is None:
            raise ValueError("No durable job with that ID belongs to this agent.")
        if inspection.job.state not in _CANCELABLE_STATES:
            raise ValueError(
                "This job is "
                + inspection.job.state.value
                + " and cannot be cancelled."
            )
        accepted = await self.app._await_modal(  # type: ignore[attr-defined]
            ConfirmScreen(
                "Cancel durable job "
                + safe_display(job_id, fallback="job", maximum=256)
                + "?\n"
                + safe_display(
                    inspection.job.specification.objective,
                    fallback="job",
                    maximum=128,
                )
                + " · "
                + inspection.job.state.value
                + "\n\nCancellation is requested immediately and cannot be undone."
            )
        )
        if not accepted:
            return
        updated = await self.app.controller.cancel_job(job_id)  # type: ignore[attr-defined]
        if updated is None:
            raise ValueError("The job no longer exists within this agent boundary.")
        state = updated.state
        if state in _CANCELLATION_STATES:
            self._notice = f"Cancellation requested · {job_id} · {state.value}"
        else:
            self._notice = (
                f"Job became {state.value} before cancellation was applied · {job_id}"
            )
        self.query_one("#jobs-notice", Static).update(self._notice)
        await self._load_jobs()
        inspection = await self.app.controller.inspect_job(job_id)  # type: ignore[attr-defined]
        if inspection is not None:
            self.query_one("#jobs-detail", Static).update(
                render_job_inspection(inspection)
            )

    def _selected_job_id(self) -> str | None:
        listing = self.query_one("#jobs-list", OptionList)
        if listing.highlighted is None:
            return None
        option = listing.get_option_at_index(listing.highlighted)
        return str(option.id) if option.id is not None else None

    def _selected_summary(self) -> GraphJob | None:
        selected = self._selected_job_id()
        if selected is None:
            return None
        return next((item for item in self._jobs if item.job_id == selected), None)

    def _render_overview(self, summary: GraphJob) -> None:
        self.query_one("#jobs-detail", Static).update(render_job_summary(summary))

    def _summary_text(self) -> str:
        active = sum(
            item.state
            in {GraphState.QUEUED, GraphState.ACTIVE, GraphState.CANCEL_REQUESTED}
            for item in self._jobs
        )
        results = sum(item.terminal_result_id is not None for item in self._jobs)
        noun = "job" if len(self._jobs) == 1 else "jobs"
        return f"{len(self._jobs)} {noun}  ·  {active} active  ·  {results} results"

    @staticmethod
    def _list_label(summary: GraphJob) -> str:
        status = summary.state.value.replace("_", " ").upper()
        short_id = (
            summary.job_id if len(summary.job_id) <= 20 else "…" + summary.job_id[-19:]
        )
        result = " · result" if summary.terminal_result_id is not None else ""
        return sanitize_terminal_text(
            f"{status:<16} graph · {short_id} · "
            f"{summary.updated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}{result}",
            maximum=512,
            preserve_lines=False,
            fallback="job",
        )

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._update_actions()

    def _update_actions(self) -> None:
        if not self.is_mounted:
            return
        summary = self._selected_summary()
        self.query_one("#jobs-refresh", Button).disabled = self._busy
        self.query_one("#jobs-details", Button).disabled = self._busy or summary is None
        self.query_one("#jobs-board", Button).disabled = self._busy or summary is None
        self.query_one("#jobs-timeline", Button).disabled = (
            self._busy or summary is None
        )
        self.query_one("#jobs-controls", Button).disabled = (
            self._busy or summary is None
        )
        self.query_one("#jobs-results", Button).disabled = (
            self._busy or summary is None or summary.terminal_result_id is None
        )
        self.query_one("#jobs-cancel", Button).disabled = (
            self._busy or summary is None or summary.state not in _CANCELABLE_STATES
        )
        self.query_one("#jobs-close", Button).disabled = self._busy

    def _show_error(self, error: Exception) -> None:
        self.query_one("#jobs-error", Static).update(
            sanitize_terminal_text(
                str(error),
                maximum=512,
                preserve_lines=False,
                fallback="Job action failed.",
            )
        )


@dataclass(frozen=True, slots=True)
class JobControlDecision:
    action: str
    task_id: str
    control_id: str
    text: str
    answer: dict[str, object] | None = None


class JobControlScreen(ModalScreen[JobControlDecision | None]):
    """Collect one typed human outcome for an exact open graph control."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Back", priority=True)
    ]

    def __init__(self, inspection: GraphInspection) -> None:
        super().__init__()
        self._inspection = inspection
        self._controls = tuple(
            item for item in inspection.controls if item.state is ControlState.OPEN
        )

    def compose(self) -> ComposeResult:
        with Vertical(id="job-controls", classes="modal-panel"):
            yield Label("Graph controls", id="jobs-title", markup=False)
            yield Static(
                "Select an exact control. Text is advisory; the owner validates the typed outcome and fences stale state.",
                id="job-controls-help",
                markup=False,
            )
            yield OptionList(
                *(
                    Option(
                        Text(
                            f"{control.kind.value} · {control.task_id} · {control.control_id}"
                        ),
                        id=control.control_id,
                    )
                    for control in self._controls
                ),
                id="job-controls-list",
            )
            yield Static("", id="job-control-detail", markup=False)
            yield Input(
                placeholder="Rationale, guidance, or a JSON object for requested input",
                id="job-control-value",
            )
            with Horizontal(id="job-control-actions"):
                yield Button("Apply", id="job-control-submit", variant="primary")
                yield Button(
                    "Request changes",
                    id="job-control-changes",
                    variant="warning",
                )
                yield Button("Reject", id="job-control-reject", variant="error")
                yield Button("Close", id="job-control-close")
            yield Static("", id="job-control-error", markup=False)

    def on_mount(self) -> None:
        listing = self.query_one("#job-controls-list", OptionList)
        if self._controls:
            listing.highlighted = 0
            listing.focus()
            self._render_selected()
        self._update_buttons()

    def action_close(self) -> None:
        self.dismiss(None)

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        del event
        self._render_selected()
        self._update_buttons()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "job-control-close":
            self.dismiss(None)
        elif button_id == "job-control-submit":
            self._submit_primary()
        elif button_id == "job-control-changes":
            self._submit("request_changes")
        elif button_id == "job-control-reject":
            self._submit("reject")

    async def on_input_submitted(self, _event: Input.Submitted) -> None:
        self._submit_primary()

    def _selected(self) -> TaskControl | None:
        listing = self.query_one("#job-controls-list", OptionList)
        if listing.highlighted is None:
            return None
        option = listing.get_option_at_index(listing.highlighted)
        selected = None if option.id is None else str(option.id)
        return next(
            (item for item in self._controls if item.control_id == selected), None
        )

    def _submit_primary(self) -> None:
        control = self._selected()
        if control is None:
            self._error("Select a control first.")
            return
        if control.kind is ControlKind.REVIEW_REQUESTED:
            self._submit("accept_review")
        elif control.kind is ControlKind.NEEDS_INPUT:
            self._submit("answer")
        elif control.kind is ControlKind.CHANGES_REQUESTED:
            self._submit("replace")
        elif control.kind in {
            ControlKind.NEEDS_AUTHORIZATION,
            ControlKind.EFFECT_UNCERTAIN,
        }:
            self._error("This control requires rejection or a separately admitted job.")
        else:
            self._submit("retry")

    def _submit(self, action: str) -> None:
        control = self._selected()
        if control is None:
            self._error("Select a control first.")
            return
        text = self.query_one("#job-control-value", Input).value.strip()
        if not text or len(text.encode("utf-8")) > 4096:
            self._error("Enter a bounded rationale, answer, or advisory note.")
            return
        answer: dict[str, object] | None = None
        if action == "answer":
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                self._error("Requested input must be a JSON object.")
                return
            if not isinstance(decoded, dict) or not all(
                isinstance(key, str) for key in decoded
            ):
                self._error("Requested input must be a JSON object.")
                return
            answer = decoded
        self.dismiss(
            JobControlDecision(
                action=action,
                task_id=control.task_id,
                control_id=control.control_id,
                text=text,
                answer=answer,
            )
        )

    def _render_selected(self) -> None:
        control = self._selected()
        if control is None:
            return
        payload = bounded_json_text(dict(control.payload))
        self.query_one("#job-control-detail", Static).update(
            sanitize_terminal_text(
                f"{control.kind.value} · task {control.task_id}\n"
                f"Created: {control.created_at.isoformat()}\n"
                f"Digest: {control.payload_digest}\n\n{payload}",
                maximum=16_000,
                preserve_lines=True,
                fallback="Control details unavailable.",
            )
        )

    def _update_buttons(self) -> None:
        if not self.is_mounted:
            return
        control = self._selected()
        submit = self.query_one("#job-control-submit", Button)
        changes = self.query_one("#job-control-changes", Button)
        reject = self.query_one("#job-control-reject", Button)
        if control is None:
            submit.disabled = changes.disabled = reject.disabled = True
            return
        labels = {
            ControlKind.REVIEW_REQUESTED: "Accept candidate",
            ControlKind.NEEDS_INPUT: "Submit input",
            ControlKind.CHANGES_REQUESTED: "Create replacement",
        }
        submit.label = labels.get(control.kind, "Resolve and retry")
        submit.disabled = control.kind in {
            ControlKind.NEEDS_AUTHORIZATION,
            ControlKind.EFFECT_UNCERTAIN,
        }
        changes.disabled = control.kind is not ControlKind.REVIEW_REQUESTED
        reject.disabled = control.kind is ControlKind.REVIEW_REQUESTED

    def _error(self, message: str) -> None:
        self.query_one("#job-control-error", Static).update(message)


def render_job_summary(job: GraphJob) -> str:
    """Render one bounded current graph-job summary."""

    return "\n".join(
        (
            "Job " + safe_display(job.job_id, fallback="job", maximum=256),
            "graph · " + job.state.value,
            f"Created: {job.created_at.isoformat()}",
            f"Updated: {job.updated_at.isoformat()}",
            f"Deadline: {job.deadline_at.isoformat()}",
            "Origin conversation: "
            + safe_display(job.conversation_id, fallback="unknown", maximum=256),
            (
                f"Sources: {len(job.specification.authority.source_ids)} · "
                f"Resources: {len(job.specification.authority.resource_ids)}"
            ),
            "Result: "
            + ("available" if job.terminal_result_id is not None else "not available"),
            "\nChoose Details for graph tasks or Results for validated output.",
        )
    )


def render_job_inspection(inspection: GraphInspection) -> str:
    """Render bounded lifecycle facts for one exact owned graph."""

    return render_graph_inspection(inspection)


def render_graph_inspection(inspection: GraphInspection) -> str:
    """Render bounded read-only current graph state."""

    lines = [
        "Graph " + safe_display(inspection.job.job_id, fallback="job", maximum=256),
        "State: " + inspection.job.state.value,
        (
            f"Topology: {inspection.graph.task_count} tasks · "
            f"{inspection.graph.edge_count} dependencies · revision "
            f"{inspection.graph.revision}"
        ),
        f"Active attempts: {inspection.graph.active_attempt_count}",
        "",
        "Tasks",
    ]
    attempts_by_task = {
        task.task_id: tuple(
            item for item in inspection.attempts if item.task_id == task.task_id
        )
        for task in inspection.tasks
    }
    for task in inspection.tasks:
        attempts = attempts_by_task[task.task_id]
        lines.append(
            f"{task.role.value} · {task.state.value} · "
            f"{safe_display(task.task_id, fallback='task', maximum=256)} · "
            f"{len(attempts)} attempt(s)"
        )
        for attempt in attempts:
            detail = (
                f"  {attempt.ordinal}. {attempt.state.value} · fence "
                f"{attempt.fencing_epoch}"
            )
            if attempt.error_code is not None:
                detail += " · " + safe_display(
                    attempt.error_code,
                    fallback="attempt failed",
                    maximum=256,
                )
            lines.append(detail)
    lines.extend(("", "Dependencies"))
    if not inspection.dependencies:
        lines.append("No dependencies.")
    for edge in inspection.dependencies:
        lines.append(
            f"{edge.upstream_task_id} -> {edge.downstream_task_id} · "
            f"{edge.edge_kind.value}"
        )
    lines.extend(("", "Controls"))
    if not inspection.controls:
        lines.append("No controls.")
    for control in inspection.controls:
        lines.append(
            f"{control.kind.value} · {control.state.value} · "
            f"{control.task_id} · {control.control_id}"
        )
    lines.extend(("", "Checkpoints"))
    if not inspection.checkpoints:
        lines.append("No checkpoints.")
    for checkpoint in inspection.checkpoints:
        lines.append(
            f"{checkpoint.task_id} · {checkpoint.attempt_id} · "
            f"{checkpoint.checkpoint_id} · {checkpoint.ordinal} · "
            f"{safe_display(checkpoint.milestone, fallback='checkpoint', maximum=256)}"
        )
    lines.extend(("", "Budgets"))
    for ledger in inspection.budget_ledgers:
        owner = "root" if ledger.task_id is None else ledger.task_id
        lines.append(
            f"{owner} · {ledger.dimension}: {ledger.settled} settled + "
            f"{ledger.reserved} reserved / {ledger.ceiling}"
        )
    lines.extend(
        (
            "",
            f"Accepted results: {len(inspection.results)}",
            f"Artifacts: {sum(len(item.artifact_ids) for item in inspection.results)}",
            f"Deliveries: {len(inspection.delivery_ids)}",
            f"Recent events: {len(inspection.events)}",
        )
    )
    if inspection.results:
        lines.extend(("", "Accepted task results"))
        for result in inspection.results:
            artifacts = ", ".join(result.artifact_ids) or "no artifacts"
            lines.append(
                f"{result.task_id} · {result.result_id} · {result.result_kind} · "
                f"{artifacts}"
            )
    return sanitize_terminal_text(
        "\n".join(lines),
        maximum=32_768,
        preserve_lines=True,
        fallback="Graph details unavailable.",
    )


def render_job_board(board: GraphBoardProjection) -> str:
    """Render the public bounded Kanban and dependency projection."""

    lines = [
        f"Board {board.job_id}",
        f"State: {board.graph_state.value} · revision {board.graph_revision}",
        "",
        "Kanban",
    ]
    for column in board.columns:
        lines.append(f"{column.name}: " + (", ".join(column.task_ids) or "empty"))
    lines.extend(("", "Dependencies"))
    if not board.dependencies:
        lines.append("No dependencies.")
    for edge in board.dependencies:
        marker = "satisfied" if edge.satisfied else "waiting"
        lines.append(
            f"{edge.upstream_task_id} -> {edge.downstream_task_id} · "
            f"{edge.edge_kind} · {marker}"
        )
    lines.extend(("", "Diagnostics"))
    lines.append("deadlocked: " + ("yes" if board.diagnostics.deadlocked else "no"))
    if board.diagnostics.deadlock_reason is not None:
        lines.append("reason: " + board.diagnostics.deadlock_reason)
    lines.append(f"blockers: {len(board.diagnostics.blockers)}")
    lines.append(f"exhausted budgets: {len(board.diagnostics.exhausted_budgets)}")
    return sanitize_terminal_text(
        "\n".join(lines),
        maximum=32_768,
        preserve_lines=True,
        fallback="Graph board unavailable.",
    )


def render_job_timeline(page: GraphTimelinePage) -> str:
    """Render one stable cursor page without replaying events as state."""

    lines = [
        f"Timeline {page.job_id}",
        f"Current state: {page.graph_state.value} · revision {page.graph_revision}",
        f"Next cursor: {page.next_cursor if page.next_cursor is not None else 'end'}",
        "",
    ]
    if not page.events:
        lines.append("No events on this page.")
    for event in page.events:
        task = "" if event.task_id is None else f" · task {event.task_id}"
        lines.append(
            f"{event.event_id} · {event.created_at.isoformat()} · "
            f"{event.kind}{task}"
        )
    lines.extend(("", "Authoritative diagnostics"))
    lines.append(f"blockers: {len(page.diagnostics.blockers)}")
    lines.append("deadlocked: " + ("yes" if page.diagnostics.deadlocked else "no"))
    if page.diagnostics.deadlock_reason is not None:
        lines.append("reason: " + page.diagnostics.deadlock_reason)
    return sanitize_terminal_text(
        "\n".join(lines),
        maximum=32_768,
        preserve_lines=True,
        fallback="Graph timeline unavailable.",
    )


def render_job_result(result: TaskResult) -> str:
    """Render one bounded authenticated finalizer result."""

    lines = [
        "Result for " + safe_display(result.job_id, fallback="job", maximum=256),
        "Result ID: " + safe_display(result.result_id, fallback="result", maximum=256),
        f"Completed: {result.completed_at.isoformat()}",
        "Sensitivity: " + result.sensitivity.value,
        "Kind: " + safe_display(result.result_kind, fallback="result", maximum=256),
        "",
        "Summary: " + safe_display(result.summary, fallback="result", maximum=4096),
        "",
        "Payload",
        bounded_json_text(dict(result.payload)),
        "",
        "Provenance",
        bounded_json_text(dict(result.provenance)),
        "",
        f"Artifacts ({len(result.artifact_ids)})",
    ]
    if not result.artifact_ids:
        lines.append("No artifacts were produced.")
    for artifact_id in result.artifact_ids:
        lines.append(safe_display(artifact_id, fallback="artifact", maximum=256))
    return sanitize_terminal_text(
        "\n".join(lines),
        maximum=40_000,
        preserve_lines=True,
        fallback="Job result unavailable.",
    )


__all__ = [
    "JobControlDecision",
    "JobControlScreen",
    "JobsScreen",
    "render_graph_inspection",
    "render_job_board",
    "render_job_inspection",
    "render_job_result",
    "render_job_summary",
    "render_job_timeline",
]
