"""Implement the ready-agent chat screen and interactive run flow."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from rich.cells import cell_len
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import ActiveBinding, Binding
from textual.containers import Center, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, OptionList, Static, Tab, Tabs, TextArea
from textual.widgets.option_list import Option

from daita import ApprovalDecision, ApprovalRequest

from ..commands import NEXT_STEPS
from ..models import TranscriptBlock
from ..widgets.approval import ApprovalPanel
from ..widgets.command_center import (
    CommandCenterNavigate,
    CommandSections,
    WorkBoardView,
)
from ..widgets.composer import (
    CompletionPopup,
    Composer,
    ComposerCompletionAccepted,
    ComposerCompletionDismissed,
    ComposerCompletionMoved,
)
from ..widgets.status import ActivityBar, NoticeBar, StatusBar
from ..widgets.transcript import TranscriptView
from .help import HelpScreen

if TYPE_CHECKING:
    from ..app import DaitaApp

# Four-row Daita "D", one row per line of the empty-chat welcome text.
_WELCOME_LOGO = (
    "⣿⣿⣿⠿⠿⠿⣷⣦⡀   ",
    "⣿⣿⣿    ⢹⣿   ",
    "⣿⣿⣿    ⣸⣿   ",
    "⣿⣿⣿⣶⣶⣶⡿⠟⠁   ",
)


def welcome_text(
    *, model: str | None = None, sources: int = 0, catalog_empty: bool = False
) -> Text:
    """The empty-chat greeting with a setup checklist beside the logo."""

    model_line = (
        ("✓", "Model", model) if model else ("○", "Model", "choose one with /model")
    )
    if not sources:
        data_line = ("○", "Data", "connect with /sources, or ask about local files")
    elif catalog_empty:
        data_line = ("○", "Data", "0 tables found · open /sources and choose Edit")
    else:
        noun = "source" if sources == 1 else "sources"
        data_line = ("✓", "Data", f"{sources} {noun} connected")
    lines: tuple[tuple[str, str], ...] = (
        ("What would you like to know?", "bold"),
        ("Ask about your data or local files.", ""),
        ("Type / for commands or @ to choose a source.", "dim"),
        ("", ""),
    )
    welcome = Text()
    for index, (logo, (line, style)) in enumerate(
        zip(_WELCOME_LOGO, lines, strict=True)
    ):
        if index:
            welcome.append("\n")
        welcome.append(logo, style="bold")
        welcome.append(line, style=style)
    for mark, label, detail in (model_line, data_line):
        welcome.append("\n" + " " * cell_len(_WELCOME_LOGO[0]))
        welcome.append(mark, style="green" if mark == "✓" else "bold")
        welcome.append(f" {label.ljust(6)}", style="bold")
        welcome.append(detail, style="dim")
    return welcome


class ChatScreen(Screen[None]):
    AUTO_FOCUS = "#composer"

    BINDINGS = [
        Binding("ctrl+1", "section('chat')", show=False, priority=True),
        Binding("ctrl+2", "section('sources')", show=False, priority=True),
        Binding("ctrl+3", "section('work')", show=False, priority=True),
        Binding("ctrl+4", "section('artifacts')", show=False, priority=True),
        Binding("f6", "focus_tabs", "Tabs", priority=True),
        Binding("f1", "help", "Help", priority=True),
        Binding("escape", "focus_conversation", "Chat", show=False),
        Binding("ctrl+o", "toggle_tools", "Steps", priority=True),
        Binding("ctrl+end", "follow", "Latest", priority=True),
        Binding("ctrl+home", "start", "Start", priority=True),
        Binding("ctrl+c", "copy_or_cancel", "Copy / Cancel"),
        Binding("pageup", "page_up", "Page up", priority=True),
        Binding("pagedown", "page_down", "Page down", priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._accepted_completion_text: str | None = None
        self.section = "chat"

    def compose(self) -> ComposeResult:
        with Vertical(id="chat"):
            navigation = Tabs(
                *(
                    Tab(label, id=f"tab-{section}")
                    for section, label in (
                        ("chat", "Chat"),
                        ("sources", "Sources"),
                        ("work", "Work"),
                        ("artifacts", "Artifacts"),
                    )
                ),
                active="tab-chat",
                id="main-tabs",
            )
            yield StatusBar(navigation)
            with Vertical(id="main-view"):
                with Vertical(id="conversation-pane"):
                    with Vertical(id="welcome-region"):
                        yield Static(welcome_text(), id="welcome", markup=False)
                        with Center(id="suggestions-row"):
                            yield OptionList(id="suggestions")
                    yield TranscriptView()
                    yield ApprovalPanel()
                yield CommandSections()
            with Vertical(id="chat-bottom"):
                with Horizontal(id="next-steps"):
                    yield Static("Next", id="next-steps-label", markup=False)
                    for command, label in NEXT_STEPS:
                        yield Button(
                            label,
                            id=f"next-{command.removeprefix('/')}",
                            compact=True,
                            tooltip=command,
                        )
                yield ActivityBar()
                yield NoticeBar()
                yield CompletionPopup(id="completions")
                yield Composer()
            yield Footer(show_command_palette=False, compact=True)

    @property
    def active_bindings(self) -> dict[str, ActiveBinding]:
        # Retain every shortcut while showing only the current task's hints.
        bindings = super().active_bindings
        leading = (
            ("y", "n", "escape")
            if "y" in bindings
            else (
                ("enter", "ctrl+o", "f6", "f1")
                if self.section == "chat"
                else (
                    ("enter", "d", "escape", "f1")
                    if self.section == "work"
                    else ("enter", "escape", "f1")
                )
            )
        )
        return {
            key: value._replace(binding=replace(value.binding, show=key in leading))
            for key in (*leading, *bindings)
            if (value := bindings.get(key)) is not None
        }

    def on_mount(self) -> None:
        self.query_one("#next-steps").display = False
        self.query_one(CompletionPopup).display = False
        self.query_one(TranscriptView).display = False
        self._show_section("chat")
        self._apply_layout()

    def on_resize(self) -> None:
        if self.is_mounted:
            self._apply_layout()

    def on_screen_resume(self) -> None:
        # Resume follows mount, on first show and after every dialog closes.
        self.daita_app.chat_resumed.set()

    def _apply_layout(self) -> None:
        self.set_class(self.app.size.width < 80, "-narrow")
        # Size the strip to its labels so identity can use the remaining space.
        # Below 80 columns, Tabs keeps its normal horizontal keyboard scrolling.
        navigation = self.query_one(Tabs)
        navigation.styles.width = sum(
            cell_len(tab.label_text) + 2 for tab in navigation.query(Tab)
        )

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        event.stop()
        self.action_section((event.tab.id or "tab-chat").removeprefix("tab-"))

    def action_section(self, section: str) -> None:
        if self.query_one(ApprovalPanel).active:
            self.query_one(Tabs).active = "tab-chat"
            self.action_focus_conversation()
            return
        self._show_section(section)
        if section != "chat" and self.daita_app.controller.agent is not None:
            self.run_worker(
                self.daita_app.refresh_background_status(notify_new=False),
                name="section-refresh",
            )

    def _show_section(self, section: str) -> None:
        self.section = section
        self.query_one(Tabs).active = f"tab-{section}"
        is_chat = section == "chat"
        self.query_one("#conversation-pane").display = is_chat
        sections = self.query_one(CommandSections)
        sections.display = not is_chat
        if not is_chat:
            sections.current = f"section-{section}"
            self.dismiss_completions()
        self.query_one(Composer).display = is_chat
        if is_chat and self.composer().text:
            self.run_worker(self._refresh_completions(), name="completion-refresh")
        if not self.query_one(Tabs).has_focus:
            self.focus_section()
        self.refresh_bindings()

    def focus_section(self) -> None:
        if self.section == "chat":
            self.query_one(Composer).focus()
        elif self.section == "work":
            self.query_one(WorkBoardView).focus_board()
        else:
            self.query_one(f"#command-{self.section}", OptionList).focus()

    def action_focus_tabs(self) -> None:
        if self.query_one(ApprovalPanel).active:
            self.action_focus_conversation()
        else:
            self.query_one(Tabs).focus()

    def action_help(self) -> None:
        if self.query_one(ApprovalPanel).active:
            self.action_focus_conversation()
            return
        self.app.push_screen(HelpScreen())

    def action_focus_conversation(self) -> None:
        if self.section != "chat":
            self._show_section("chat")
        panel = self.query_one(ApprovalPanel)
        if panel.active:
            target = (
                "#approval-inline-no"
                if panel.query_one("#approval-inline-no").display
                else "#approval-inline-close"
            )
            panel.query_one(target, Button).focus()
        else:
            self.query_one(Composer).focus()

    def show_next_steps(self, visible: bool) -> None:
        self.query_one("#next-steps").display = visible

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if not button_id.startswith("next-"):
            return
        event.stop()
        self.run_worker(
            self.daita_app._handle_command("/" + button_id.removeprefix("next-")),
            name="next-step",
            group="foreground-interaction",
            exclusive=True,
        )

    def on_command_center_navigate(self, event: CommandCenterNavigate) -> None:
        event.stop()
        if self.query_one(ApprovalPanel).active:
            self.action_focus_conversation()
            return
        self.run_worker(
            self.daita_app.open_command_center_item(event.section, event.item_id),
            name="command-center-navigation",
            group="foreground-interaction",
            exclusive=True,
        )

    def on_work_board_view_dismiss_requested(
        self, event: WorkBoardView.DismissRequested
    ) -> None:
        event.stop()
        if self.query_one(ApprovalPanel).active:
            self.action_focus_conversation()
            return
        self.run_worker(
            self.daita_app.dismiss_work_result(event.delivery_id),
            name="work-dismiss",
            group="foreground-interaction",
            exclusive=True,
        )

    async def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area is self.query_one(Composer):
            await self._refresh_completions()

    async def _refresh_completions(self) -> None:
        if self.section != "chat" or self.query_one(ApprovalPanel).active:
            return
        composer = self.query_one(Composer)
        prefix = composer.text
        if self._accepted_completion_text == prefix:
            self._accepted_completion_text = None
            self.dismiss_completions()
            return
        self._accepted_completion_text = None
        matches = await self.daita_app.completion_matches(prefix)
        if (
            composer.text != prefix
            or self.section != "chat"
            or self.query_one(ApprovalPanel).active
        ):
            return
        popup = self.query_one(CompletionPopup)
        popup.update_matches(matches)
        composer.completion_active = bool(matches)

    def on_composer_completion_moved(self, event: ComposerCompletionMoved) -> None:
        self.query_one(CompletionPopup).move_highlight(event.delta)
        event.stop()

    def on_composer_completion_accepted(
        self, event: ComposerCompletionAccepted
    ) -> None:
        self._accept_completion(self.query_one(CompletionPopup).selected_insertion())
        event.stop()

    def on_composer_completion_dismissed(
        self, event: ComposerCompletionDismissed
    ) -> None:
        self.dismiss_completions()
        event.stop()

    def set_welcome(
        self,
        *,
        model: str | None,
        sources: int,
        catalog_empty: bool,
        suggestions: tuple[str, ...],
    ) -> None:
        self.query_one("#welcome", Static).update(
            welcome_text(model=model, sources=sources, catalog_empty=catalog_empty)
        )
        listing = self.query_one("#suggestions", OptionList)
        listing.clear_options()
        listing.add_options(
            [
                Option(Text.assemble(("› ", "bold"), suggestion), id=suggestion)
                for suggestion in suggestions
            ]
        )
        listing.display = bool(suggestions)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "suggestions":
            event.stop()
            composer = self.query_one(Composer)
            composer.load_text(event.option.id or "")
            composer.move_cursor((0, len(composer.text)))
            composer.focus()
            return
        if event.option_list.id != "completion-list":
            return
        popup = self.query_one(CompletionPopup)
        self._accept_completion(popup.insertion_at(event.option_index))
        event.stop()

    def _accept_completion(self, insertion: str | None) -> None:
        if insertion is None:
            return
        composer = self.query_one(Composer)
        self._accepted_completion_text = insertion
        composer.load_text(insertion)
        composer.move_cursor((0, len(insertion)))
        self.dismiss_completions()
        composer.focus()

    def dismiss_completions(self) -> None:
        self.query_one(CompletionPopup).dismiss()
        self.query_one(Composer).completion_active = False

    @property
    def daita_app(self) -> DaitaApp:
        return self.app  # type: ignore[return-value]

    def action_toggle_tools(self) -> None:
        self.query_one(TranscriptView).toggle_tools()

    def action_follow(self) -> None:
        self.query_one(TranscriptView).follow_latest()

    def action_start(self) -> None:
        transcript = self.query_one(TranscriptView)
        transcript.following = False
        transcript.scroll_home(animate=False)

    def action_page_up(self) -> None:
        transcript = self.query_one(TranscriptView)
        transcript.following = False
        transcript.scroll_page_up()

    def action_page_down(self) -> None:
        self.query_one(TranscriptView).scroll_page_down()

    async def action_copy_or_cancel(self) -> None:
        await self.daita_app.copy_or_cancel()

    def set_status(
        self,
        *,
        agent: str,
        model: str,
        source: str,
        state: str,
        host: str = "connected",
        context_used: int | None = None,
        context_total: int | None = None,
        scheduled: int = 0,
        running: int = 0,
        needs_you: int = 0,
        active_reports: int = 0,
        source_count: int = 0,
        too_small: bool = False,
    ) -> None:
        self.query_one(StatusBar).update_status(
            agent=agent,
            model=model,
            source=source,
            host=host,
            state=state,
            context_used=context_used,
            context_total=context_total,
            scheduled=scheduled,
            running=running,
            needs_you=needs_you,
            active_reports=active_reports,
            source_count=source_count,
            too_small=too_small,
        )
        for section, label, count in (
            ("sources", "Sources", source_count),
            ("work", "Work", needs_you),
        ):
            self.query_one(f"#tab-{section}", Tab).label = (
                f"{label} {count}" if count else label
            )
        self.query_one("#tab-work").tooltip = (
            f"{running} running · {scheduled} scheduled · {needs_you} need you"
        )
        self._apply_layout()

    def show_notice(self, message: str) -> None:
        self.query_one(NoticeBar).show(message)

    def clear_notice(self) -> None:
        self.query_one(NoticeBar).clear()

    def set_blocks(self, blocks: tuple[TranscriptBlock, ...]) -> None:
        transcript = self.query_one(TranscriptView)
        transcript.set_blocks(blocks)
        self._show_conversation(bool(blocks))

    def append_block(self, block: TranscriptBlock) -> None:
        self._show_conversation(True)
        self.query_one(TranscriptView).append_block(block)

    def replace_partial(self, identity: str, text: str) -> None:
        self._show_conversation(True)
        self.query_one(TranscriptView).replace_partial(identity, text)

    def remove_block(self, identity: str) -> None:
        transcript = self.query_one(TranscriptView)
        transcript.remove_block(identity)
        self._show_conversation(not transcript.is_empty)

    def set_activity(self, stage: str, *, restart: bool = False) -> None:
        activity = self.query_one(ActivityBar)
        if restart or not activity.display:
            activity.start(stage)
        else:
            activity.update_stage(stage)

    def clear_activity(self) -> None:
        self.query_one(ActivityBar).stop()

    def step_started(self, call_id: str, label: str) -> None:
        self.query_one(ActivityBar).step_started(call_id, label)

    def step_finished(
        self, call_id: str, *, success: bool, duration_ms: object = None
    ) -> None:
        self.query_one(ActivityBar).step_finished(
            call_id, success=success, duration_ms=duration_ms
        )

    async def request_approval(
        self,
        request: ApprovalRequest,
    ) -> ApprovalDecision | None:
        self._show_section("chat")
        self._show_conversation(True)
        tabs = self.query_one(Tabs)
        composer = self.query_one(Composer)
        was_disabled = composer.disabled
        tabs.disabled = composer.disabled = True
        try:
            return await self.query_one(ApprovalPanel).request(request)
        finally:
            tabs.disabled = False
            composer.disabled = was_disabled
            if not was_disabled:
                self.action_focus_conversation()

    def _show_conversation(self, has_blocks: bool) -> None:
        self.query_one("#welcome-region", Vertical).display = not has_blocks
        self.query_one("#welcome").display = not has_blocks
        self.query_one("#suggestions").display = (
            not has_blocks
            and self.query_one("#suggestions", OptionList).option_count > 0
        )
        self.query_one(TranscriptView).display = has_blocks

    def set_submitting(self, submitting: bool) -> None:
        if self.query_one(ApprovalPanel).active and not submitting:
            return
        if submitting:
            self.dismiss_completions()
        composer = self.query_one(Composer)
        was_disabled = composer.disabled
        composer.set_submitting(submitting)
        if not submitting and was_disabled and self.section == "chat":
            self.action_focus_conversation()

    def composer(self) -> Composer:
        return self.query_one(Composer)
