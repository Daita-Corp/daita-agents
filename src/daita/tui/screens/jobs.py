"""List, inspect, cancel, and render durable jobs in the terminal UI."""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Label, OptionList, Static
from textual.widgets.option_list import Option

from daita import GraphInspection, GraphJob, GraphState, TaskResult

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

    BINDINGS = [
        Binding("escape", "close", "Back", priority=True),
        Binding("r", "refresh", "Refresh", priority=True),
        Binding("d", "details", "Details", priority=True),
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
            f"Sources: {len(job.specification.authority.source_ids)} · "
            f"Resources: {len(job.specification.authority.resource_ids)}",
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
        f"Topology: {inspection.graph.task_count} tasks · "
        f"{inspection.graph.edge_count} dependencies · revision "
        f"{inspection.graph.revision}",
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
    return sanitize_terminal_text(
        "\n".join(lines),
        maximum=32_768,
        preserve_lines=True,
        fallback="Graph details unavailable.",
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
    "JobsScreen",
    "render_graph_inspection",
    "render_job_inspection",
    "render_job_result",
    "render_job_summary",
]
