"""Read-only navigation projections for the chat command center."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, ContentSwitcher, OptionList, Static
from textual.widgets.option_list import Option

from daita import WorkBoard, WorkCard, WorkColumn, WorkKind
from daita.artifacts import ArtifactRef

from ..models import source_type_label
from ..sanitization import safe_display

_WORK_COLUMNS = (
    (WorkColumn.SCHEDULED, "Scheduled"),
    (WorkColumn.RUNNING, "Running"),
    (WorkColumn.NEEDS_YOU, "Needs you"),
    (WorkColumn.DONE, "Done"),
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
        self.details: dict[str, Text] = {}

    def update_rows(
        self,
        rows: tuple[tuple[str, str], ...],
        details: dict[str, Text] | None = None,
    ) -> None:
        self.details = details or {}
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


def _elapsed(delta: timedelta) -> str:
    seconds = max(0, int(delta.total_seconds()))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {seconds:02d}s"


def _ago(delta: timedelta) -> str:
    seconds = max(0, int(delta.total_seconds()))
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86_400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86_400}d ago"


def work_card_label(card: WorkCard, now: datetime) -> Text:
    """Two sanitized lines: the title, then status, timing and summary."""

    details = [card.status.replace("_", " ")]
    if card.column is WorkColumn.RUNNING and card.started_at is not None:
        details.append(_elapsed(now - card.started_at))
    elif card.column is WorkColumn.SCHEDULED and card.due_at is not None:
        details.append("due " + card.due_at.astimezone().strftime("%m/%d %H:%M"))
    elif card.column is WorkColumn.DONE:
        details.append("finished " + _ago(now - card.updated_at))
    elif card.kind is not WorkKind.ROUTINE:
        details.append(_ago(now - card.updated_at))
    summary = safe_display(card.summary, fallback="", maximum=240)
    if summary:
        details.append(summary)
    label = Text(safe_display(card.title, fallback="Work", maximum=120))
    label.append("\n" + " · ".join(details), style="dim")
    return label


def _detail(title: str, rows: tuple[tuple[str, str], ...], hint: str) -> Text:
    text = Text(safe_display(title, fallback="item", maximum=120), style="bold")
    text.append("\n")
    for label, value in rows:
        if value:
            text.append(f"\n{label.ljust(10)}", style="dim")
            text.append(safe_display(value, fallback="", maximum=240))
    text.append(f"\n\n{hint}", style="dim")
    return text


def source_detail(source: Any, type_label: str) -> Text:
    return _detail(
        source.display_name,
        (
            ("Type", type_label),
            ("Connected", source.attached_at.astimezone().strftime("%Y-%m-%d %H:%M")),
            ("About", getattr(source, "summary", "")),
            ("Use when", getattr(source, "when_to_use", "")),
        ),
        "Enter opens tables, relationships, permissions and settings",
    )


def artifact_detail(item: ArtifactRef) -> Text:
    size = item.byte_size
    size_text = f"{size:,} B" if size < 1_024 else f"{size / 1_024:,.1f} KiB"
    return _detail(
        item.filename,
        (
            ("Type", item.media_type),
            ("Size", size_text),
            ("Created", item.created_at.astimezone().strftime("%Y-%m-%d %H:%M")),
        ),
        "Enter opens preview, save and delete",
    )


def _work_target(card: WorkCard) -> tuple[str, str | None]:
    if card.kind is WorkKind.JOB:
        return "jobs", card.id
    if card.kind is WorkKind.ROUTINE:
        return "routines", card.id
    if card.delivery_id is not None:
        return "result", card.delivery_id
    return "routines", card.routine_id


class WorkBoardView(Widget):
    """Four columns of live and recent work; enter opens and d dismisses."""

    BINDINGS = [
        Binding("left", "move(-1)", "Previous column", show=False),
        Binding("right", "move(1)", "Next column", show=False),
        Binding("d", "dismiss_card", "Dismiss"),
    ]

    class DismissRequested(Message):
        """Ask the app to dismiss one finished or failed routine result."""

        def __init__(self, delivery_id: str) -> None:
            super().__init__()
            self.delivery_id = delivery_id

    def __init__(self) -> None:
        super().__init__(id="work-board")
        self._cards: dict[str, WorkCard] = {}
        self._column = 0

    def compose(self) -> ComposeResult:
        for column, label in _WORK_COLUMNS:
            with Vertical(classes="work-column"):
                yield Static(
                    label.upper(),
                    id=f"work-header-{column.value}",
                    classes="work-column-header",
                    markup=False,
                )
                yield OptionList(id=f"work-{column.value}", classes="work-cards")

    def on_mount(self) -> None:
        self.query_one("#work-header-needs_you", Static).add_class("-attention")
        self.on_resize()

    def on_resize(self) -> None:
        # Four columns need room; narrower terminals get a 2x2 grid.
        self.set_class(self.app.size.width < 100, "-narrow")

    def update_board(self, board: WorkBoard) -> None:
        """Replace every column, keeping each column's selected card."""

        self._cards = {}
        for column, label in _WORK_COLUMNS:
            cards = board.columns[column]
            hidden = board.truncated[column]
            self.query_one(f"#work-header-{column.value}", Static).update(
                f"{label.upper()} · {len(cards) + hidden}"
            )
            listing = self._listing(column)
            selected = self._highlighted_key(listing)
            keys = tuple(f"{card.kind.value}:{card.id}" for card in cards)
            self._cards.update(zip(keys, cards, strict=True))
            listing.clear_options()
            listing.add_options(
                [
                    Option(work_card_label(card, board.generated_at), id=key)
                    for key, card in zip(keys, cards, strict=True)
                ]
                or [Option(Text("Nothing here", style="dim"), disabled=True)]
            )
            if hidden:
                listing.add_option(
                    Option(Text(f"+{hidden} more", style="dim"), disabled=True)
                )
            listing.highlighted = (
                keys.index(selected) if selected in keys else 0 if keys else None
            )
        self.refresh_bindings()

    def focus_board(self) -> None:
        self._listing(_WORK_COLUMNS[self._column][0]).focus()

    def selected_card(self) -> WorkCard | None:
        listing = self._listing(_WORK_COLUMNS[self._column][0])
        key = self._highlighted_key(listing)
        return None if key is None else self._cards.get(key)

    def action_move(self, delta: int) -> None:
        self._column = max(0, min(len(_WORK_COLUMNS) - 1, self._column + delta))
        self.focus_board()

    def action_dismiss_card(self) -> None:
        card = self.selected_card()
        if card is not None and card.delivery_id is not None:
            self.post_message(self.DismissRequested(card.delivery_id))

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "dismiss_card":
            card = self.selected_card()
            return card is not None and card.delivery_id is not None
        return True

    def on_descendant_focus(self, event: Any) -> None:
        for index, (column, _label) in enumerate(_WORK_COLUMNS):
            if event.widget is self._listing(column):
                self._column = index
        self.refresh_bindings()

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        event.stop()
        self.refresh_bindings()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        card = self._cards.get(event.option.id or "")
        if card is not None:
            self.post_message(CommandCenterNavigate(*_work_target(card)))

    def _listing(self, column: WorkColumn) -> OptionList:
        return self.query_one(f"#work-{column.value}", OptionList)

    @staticmethod
    def _highlighted_key(listing: OptionList) -> str | None:
        if listing.highlighted is None:
            return None
        return listing.get_option_at_index(listing.highlighted).id


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
            with Horizontal(classes="section-body"):
                yield ProjectionList(
                    "sources", "No connected sources. Open Manage sources to add one."
                )
                yield Static("", id="detail-sources", classes="section-detail")
        with Vertical(id="section-work", classes="command-section"):
            with Horizontal(id="work-actions"):
                yield Button("All jobs ›", id="open-jobs", classes="section-action")
                yield Button(
                    "All routines ›", id="open-routines", classes="section-action"
                )
            yield WorkBoardView()
            yield Static(
                "Routines run while Daita is open here, or while "
                "`daita host --agent <name>` runs after you close it.",
                classes="section-description",
                markup=False,
            )
        with Vertical(id="section-artifacts", classes="command-section"):
            yield Button(
                "Manage artifacts ›", id="open-artifacts", classes="section-action"
            )
            yield Static(
                "Recent stored files. Open a file to preview, save, or delete it.",
                classes="section-description",
                markup=False,
            )
            with Horizontal(classes="section-body"):
                yield ProjectionList("artifacts", "No stored artifacts")
                yield Static("", id="detail-artifacts", classes="section-detail")

    def on_mount(self) -> None:
        self.on_resize()

    def on_resize(self) -> None:
        # Lists sit beside their details only where both stay readable.
        self.set_class(self.app.size.width >= 100, "-wide")

    def update_sources(
        self, sources: tuple[Any, ...], source_types: tuple[Any, ...] = ()
    ) -> None:
        active = tuple(item for item in sources if item.active)
        self._update_list(
            "sources",
            tuple(
                (item.id, safe_display(item.display_name, maximum=80))
                for item in active
            ),
            {
                item.id: source_detail(
                    item, source_type_label(source_types, item.adapter_id)
                )
                for item in active
            },
        )

    def update_work(self, board: WorkBoard) -> None:
        self.query_one(WorkBoardView).update_board(board)

    def update_artifacts(self, items: tuple[ArtifactRef, ...]) -> None:
        self._update_list(
            "artifacts",
            tuple((item.artifact_id, safe_display(item.filename)) for item in items),
            {item.artifact_id: artifact_detail(item) for item in items},
        )

    def _update_list(
        self,
        section: str,
        rows: tuple[tuple[str, str], ...],
        details: dict[str, Text],
    ) -> None:
        listing = self.query_one(f"#command-{section}", ProjectionList)
        listing.update_rows(rows, details)
        self._show_detail(listing)

    def _show_detail(self, listing: ProjectionList) -> None:
        option = (
            listing.get_option_at_index(listing.highlighted)
            if listing.highlighted is not None
            else None
        )
        detail = listing.details.get(option.id or "") if option else None
        self.query_one(f"#detail-{listing.section}", Static).update(detail or "")

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        if isinstance(event.option_list, ProjectionList):
            event.stop()
            self._show_detail(event.option_list)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(
            CommandCenterNavigate((event.button.id or "").removeprefix("open-"))
        )
