"""Implement the ready-agent chat screen and interactive run flow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import ActiveBinding, Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, OptionList, TextArea

from daita import ApprovalDecision, ApprovalRequest

from ..models import TranscriptBlock
from ..widgets.approval import ApprovalPanel
from ..widgets.command_center import (
    CommandCenterNavigate,
    CommandSidebar,
    ResultsDrawer,
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
from ..widgets.welcome import WelcomeView

if TYPE_CHECKING:
    from ..app import DaitaApp


class ChatScreen(Screen[None]):
    BINDINGS = [
        Binding("ctrl+b", "toggle_sidebar", "Sidebar", priority=True),
        Binding("ctrl+r", "toggle_drawer", "Drawer", priority=True),
        Binding("escape", "focus_conversation", "Chat", show=False),
        Binding("ctrl+o", "toggle_tools", "Tools", priority=True),
        Binding("ctrl+end", "follow", "Latest", priority=True),
        Binding("ctrl+home", "start", "Start", priority=True),
        Binding("ctrl+c", "copy_or_cancel", "Copy / Cancel"),
        Binding("pageup", "page_up", "Page up", priority=True),
        Binding("pagedown", "page_down", "Page down", priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._accepted_completion_text: str | None = None
        self._sidebar_open = True
        self._drawer_open = False

    def compose(self) -> ComposeResult:
        with Vertical(id="chat"):
            yield StatusBar()
            with Horizontal(id="command-panes"):
                yield CommandSidebar()
                with Vertical(id="conversation-pane"):
                    with Vertical(id="welcome-region"):
                        yield WelcomeView(id="welcome")
                    yield TranscriptView()
                    yield ApprovalPanel()
                yield ResultsDrawer()
            with Vertical(id="chat-bottom"):
                yield ActivityBar()
                yield NoticeBar()
                yield CompletionPopup(id="completions")
                yield Composer()
            yield Footer(show_command_palette=False, compact=True)

    @property
    def active_bindings(self) -> dict[str, ActiveBinding]:
        # Keep pane hints visible at 80 columns even when the composer has focus.
        # Approval decisions retain the first positions during inline review.
        bindings = super().active_bindings
        leading = ("y", "n", "escape") if "y" in bindings else ("ctrl+b", "ctrl+r")
        return {key: bindings[key] for key in (*leading, *bindings) if key in bindings}

    def on_mount(self) -> None:
        self.query_one(Composer).focus()
        self.query_one(CompletionPopup).display = False
        self.query_one(TranscriptView).display = False
        self._apply_pane_layout()

    def on_resize(self) -> None:
        if self.is_mounted:
            self._apply_pane_layout()

    def _apply_pane_layout(self) -> None:
        sidebar = self.query_one(CommandSidebar)
        drawer = self.query_one(ResultsDrawer)
        roomy = self.app.size.width >= 80 and self.app.size.height >= 24
        drawer.display = roomy and self._drawer_open
        # At 80 columns a drawer leaves 48 columns for chat and a four-column rail.
        collapsed = not (roomy and self._sidebar_open) or (
            drawer.display and self.app.size.width < 100
        )
        sidebar.set_class(collapsed, "-collapsed")
        self.query_one("#sidebar-content").display = not collapsed
        self.query_one("#sidebar-rail").display = collapsed
        self.set_class(self.app.size.width < 80, "-narrow")
        focused = self.focused
        if focused is not None and (
            (collapsed and focused in sidebar.query("#sidebar-content *"))
            or (not drawer.display and focused in drawer.query("*"))
        ):
            self.action_focus_conversation()

    def action_toggle_sidebar(self) -> None:
        self._sidebar_open = not self._sidebar_open
        self._apply_pane_layout()
        self.action_focus_conversation()

    def action_toggle_drawer(self) -> None:
        self._drawer_open = not self._drawer_open
        self._apply_pane_layout()
        self.action_focus_conversation()
        if self._drawer_open:
            self.run_worker(
                self.daita_app.refresh_background_status(notify_new=False),
                name="drawer-refresh",
            )

    def action_focus_conversation(self) -> None:
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

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "sidebar-expand":
            self.action_toggle_sidebar()
            event.stop()

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

    async def on_text_area_changed(self, event: TextArea.Changed) -> None:
        composer = self.query_one(Composer)
        if event.text_area is not composer:
            return
        prefix = composer.text
        if self._accepted_completion_text == prefix:
            self._accepted_completion_text = None
            self.dismiss_completions()
            return
        self._accepted_completion_text = None
        matches = await self.daita_app.completion_matches(prefix)
        if composer.text != prefix:
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

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
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
        context_used: int | None = None,
        context_total: int | None = None,
        active_jobs: int = 0,
        saved_routines: int = 0,
        active_reports: int = 0,
        inbox_items: int = 0,
        source_count: int = 0,
        too_small: bool = False,
    ) -> None:
        self.query_one(StatusBar).update_status(
            agent=agent,
            model=model,
            source=source,
            state=state,
            context_used=context_used,
            context_total=context_total,
            active_jobs=active_jobs,
            saved_routines=saved_routines,
            active_reports=active_reports,
            inbox_items=inbox_items,
            source_count=source_count,
            too_small=too_small,
        )
        self.query_one(WelcomeView).update_identity(
            agent=agent,
            model=model,
            source=source,
        )

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

    async def request_approval(
        self,
        request: ApprovalRequest,
    ) -> ApprovalDecision | None:
        self._show_conversation(True)
        sidebar = self.query_one(CommandSidebar)
        drawer = self.query_one(ResultsDrawer)
        composer = self.query_one(Composer)
        was_disabled = composer.disabled
        sidebar.disabled = drawer.disabled = composer.disabled = True
        try:
            return await self.query_one(ApprovalPanel).request(request)
        finally:
            sidebar.disabled = drawer.disabled = False
            composer.disabled = was_disabled
            if not was_disabled:
                self.action_focus_conversation()

    def _show_conversation(self, has_blocks: bool) -> None:
        self.query_one("#welcome-region", Vertical).display = not has_blocks
        self.query_one(WelcomeView).display = not has_blocks
        self.query_one(TranscriptView).display = has_blocks

    def set_submitting(self, submitting: bool) -> None:
        if self.query_one(ApprovalPanel).active and not submitting:
            return
        if submitting:
            self.dismiss_completions()
        composer = self.query_one(Composer)
        was_disabled = composer.disabled
        composer.set_submitting(submitting)
        if not submitting and was_disabled:
            self.action_focus_conversation()

    def composer(self) -> Composer:
        return self.query_one(Composer)
