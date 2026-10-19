"""Read-only navigation projections for the chat command center."""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Button, ContentSwitcher, OptionList, Static
from textual.widgets.option_list import Option

from daita import GraphState, RoutineState, ScheduledRoutineSummary
from daita.artifacts import ArtifactRef
from daita.distribution import DeliveryView
from daita.jobs import GraphJob

from ..sanitization import safe_display

ACTIVE_JOB_STATES = frozenset(
    {
        GraphState.QUEUED,
        GraphState.ACTIVE,
        GraphState.BLOCKED,
        GraphState.NEEDS_ATTENTION,
        GraphState.CANCEL_REQUESTED,
    }
)


class CommandCenterNavigate(Message):
    """Open an existing owner screen, optionally selecting an exact item."""

    def __init__(self, section: str, item_id: str | None = None) -> None:
        super().__init__()
        self.section = section
        self.item_id = item_id


class ProjectionList(OptionList):
    """Keep keyboard selection stable as bounded controller rows change."""

    def __init__(self, section: str, empty: str) -> None:
        super().__init__(id=f"command-{section}")
        self.section = section
        self._empty = empty
        self._rows: tuple[tuple[str, str], ...] | None = None

    def update_rows(self, rows: tuple[tuple[str, str], ...]) -> None:
        if rows == self._rows:
            return
        selected = (
            self.get_option_at_index(self.highlighted).id
            if self.highlighted is not None
            else None
        )
        self._rows = rows
        self.clear_options()
        self.add_options(
            [
                Option(Text(label, no_wrap=True, overflow="ellipsis"), id=identity)
                for identity, label in rows
            ]
            or [Option(Text(self._empty, style="dim"), disabled=True)]
        )
        self.highlighted = (
            next((i for i, row in enumerate(rows) if row[0] == selected), 0)
            if rows
            else None
        )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        if event.option.id is not None:
            self.post_message(CommandCenterNavigate(self.section, event.option.id))


class CommandSections(ContentSwitcher):
    """One read-only section at a time, with owner screens for deeper controls.

    A future evidence view can add a page here using the same controller reads
    and exact-item navigation, without changing execution or approval owners.
    """

    def __init__(self) -> None:
        super().__init__(initial="section-sources", id="command-sections")

    def compose(self) -> ComposeResult:
        with Vertical(id="section-sources", classes="command-section"):
            yield Button(
                "Manage sources ›", id="open-sources", classes="section-action"
            )
            yield Static(
                "Connected sources available for questions. Sources are optional.",
                classes="section-description",
                markup=False,
            )
            yield ProjectionList(
                "sources", "No connected sources. Open Manage sources to add one."
            )
        with Vertical(id="section-work", classes="command-section"):
            yield Button(
                "Active jobs · View all ›", id="open-jobs", classes="section-action"
            )
            yield ProjectionList("jobs", "No active jobs")
            yield Button(
                "Upcoming routines · Manage ›",
                id="open-routines",
                classes="section-action",
            )
            yield ProjectionList("routines", "No upcoming routines")
            yield Static(
                "Scheduled work progresses while this host is open.",
                classes="section-description",
                markup=False,
            )
        with Vertical(id="section-inbox", classes="command-section"):
            yield Button("Review inbox ›", id="open-inbox", classes="section-action")
            yield Static(
                "Unread results from background work.",
                classes="section-description",
                markup=False,
            )
            yield ProjectionList("inbox", "You're all caught up. No unread results.")
        with Vertical(id="section-artifacts", classes="command-section"):
            yield Button(
                "Manage artifacts ›", id="open-artifacts", classes="section-action"
            )
            yield Static(
                "Recent stored files. Open a file to preview, save, or delete it.",
                classes="section-description",
                markup=False,
            )
            yield ProjectionList("artifacts", "No stored artifacts")

    def update_sources(self, sources: tuple[Any, ...]) -> None:
        self.query_one("#command-sources", ProjectionList).update_rows(
            tuple(
                (item.id, safe_display(item.display_name, maximum=80))
                for item in sources
                if item.active
            )
        )

    def update_jobs(self, jobs: tuple[GraphJob, ...]) -> None:
        self.query_one("#command-jobs", ProjectionList).update_rows(
            tuple(
                (
                    item.job_id,
                    f"{item.state.value} · "
                    + safe_display(item.specification.objective, maximum=80),
                )
                for item in jobs
                if item.state in ACTIVE_JOB_STATES
            )
        )

    def update_routines(self, routines: tuple[ScheduledRoutineSummary, ...]) -> None:
        upcoming = sorted(
            (
                item
                for item in routines
                if item.state is RoutineState.ACTIVE and item.next_due_at is not None
            ),
            key=lambda item: (item.next_due_at, item.routine_id),
        )
        self.query_one("#command-routines", ProjectionList).update_rows(
            tuple(
                (
                    item.routine_id,
                    item.next_due_at.astimezone().strftime("%m/%d %H:%M")
                    + " · "
                    + safe_display(item.title, maximum=80),
                )
                for item in upcoming
                if item.next_due_at is not None
            )
        )

    def update_inbox(self, items: tuple[DeliveryView, ...]) -> None:
        # The existing inbox read returns only unacknowledged deliveries.
        self.query_one("#command-inbox", ProjectionList).update_rows(
            tuple(
                (
                    item.delivery_id,
                    f"{item.conclusion_state.value} · "
                    + safe_display(
                        item.conclusion_preview, fallback="Report", maximum=100
                    ),
                )
                for item in items
            )
        )

    def update_artifacts(self, items: tuple[ArtifactRef, ...]) -> None:
        self.query_one("#command-artifacts", ProjectionList).update_rows(
            tuple((item.artifact_id, safe_display(item.filename)) for item in items)
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(
            CommandCenterNavigate((event.button.id or "").removeprefix("open-"))
        )
