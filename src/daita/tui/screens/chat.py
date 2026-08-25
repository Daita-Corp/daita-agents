"""Implement the ready-agent chat screen and interactive run flow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, OptionList, TextArea

from daita import ApprovalDecision, ApprovalRequest

from ..models import TranscriptBlock
from ..widgets.approval import ApprovalPanel
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

    def compose(self) -> ComposeResult:
        with Vertical(id="chat"):
            yield StatusBar()
            with Vertical(id="welcome-region"):
                yield WelcomeView(id="welcome")
            yield TranscriptView()
            yield ApprovalPanel()
            yield ActivityBar()
            yield NoticeBar()
            yield CompletionPopup(id="completions")
            yield Composer()
            yield Footer()

    def on_mount(self) -> None:
        self.query_one(Composer).focus()
        self.query_one(CompletionPopup).display = False
        self.query_one(TranscriptView).display = False

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
        active_reports: int = 0,
        inbox_items: int = 0,
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
            active_reports=active_reports,
            inbox_items=inbox_items,
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
        return await self.query_one(ApprovalPanel).request(request)

    def _show_conversation(self, has_blocks: bool) -> None:
        self.query_one("#welcome-region", Vertical).display = not has_blocks
        self.query_one(WelcomeView).display = not has_blocks
        self.query_one(TranscriptView).display = has_blocks

    def set_submitting(self, submitting: bool) -> None:
        if submitting:
            self.dismiss_completions()
        composer = self.query_one(Composer)
        composer.set_submitting(submitting)
        if not submitting:
            composer.focus()

    def composer(self) -> Composer:
        return self.query_one(Composer)
