"""Pure and Pilot tests for the Textual interactive presentation."""

from __future__ import annotations

from _workspace_support import workspace_for

import asyncio
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.geometry import Offset
from textual.selection import Selection
from textual.widgets import (
    Button,
    Footer,
    Input,
    OptionList,
    Select,
    Static,
    Tree,
)
from textual.widgets._collapsible import CollapsibleTitle

from daita import (
    Agent,
    AgentEvent,
    AgentEventKind,
    ApprovalDecision,
    ApprovalRequest,
    DeliveryState,
    DeliverySubject,
    DeliverySubjectKind,
    InboxItem,
    JobStatus,
    LoopExit,
    LoopExitKind,
    SQLiteSource,
)
from daita._json import FrozenJsonObject
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    ModelSensitivity,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.security import CredentialSession, SecretReference, SecretResolutionError
from daita.tui.app import DaitaApp, _run_failure_notice
from daita.tui.clipboard import (
    MAX_CLIPBOARD_UTF8_BYTES,
    ClipboardResult,
    clipboard_mechanism,
    deliver_clipboard,
    osc52_sequence,
)
from daita.tui.commands import (
    SLASH_COMMAND_COMPLETIONS,
    learning_invocation_message,
    parse_postgresql_connection_url,
    parse_source_override,
)
from daita.tui.models import (
    MAX_COMPOSER_CHARACTERS,
    MIN_READY_ROWS,
    MIN_USABLE_COLUMNS,
    PickerOption,
    ToolCardDetails,
    ToolCardState,
    TranscriptBlock,
    UserInputError,
)
from daita.tui.observer import ObserverEvent
from daita.tui.projection import (
    approval_review_document,
    project_tool_details,
    redact_presentation_value,
)
from daita.tui.sanitization import sanitize_terminal_text
from daita.tui.screens.catalog import CatalogScreen
from daita.tui.screens.chat import ChatScreen
from daita.tui.screens.confirm import ConfirmScreen
from daita.tui.screens.editing import ReviewCostScreen
from daita.tui.screens.inbox import InboxScreen, render_inbox_item
from daita.tui.screens.jobs import JobsScreen
from daita.tui.screens.onboarding import (
    AgentCreateScreen,
    ModelSetupScreen,
    SourceSetupScreen,
)
from daita.tui.screens.permissions import PermissionsScreen
from daita.tui.screens.selection import SelectionScreen
from daita.tui.screens.source_edit import SourceEditScreen
from daita.tui.widgets.approval import ApprovalPanel
from daita.tui.widgets.composer import CompletionPopup, Composer
from daita.tui.widgets.status import (
    ActivityBar,
    context_window_text,
    format_token_count,
)
from daita.tui.widgets.tool_card import ToolCard
from daita.tui.widgets.transcript import TranscriptView
from daita.tui.widgets.welcome import WelcomeView


def test_sanitize_strips_terminal_controls_and_bounds_text():
    assert (
        sanitize_terminal_text(
            "ok\x1b[31msecret\x07",
            maximum=32,
            preserve_lines=False,
            fallback="x",
        )
        == "ok?[31msecret?"
    )
    assert sanitize_terminal_text(
        "a" * 20,
        maximum=10,
        preserve_lines=False,
        fallback="x",
    ).endswith("...")


def test_source_override_and_learning_parse():
    assert parse_source_override("hello") is None
    assert parse_source_override("@sales how many") == ("sales", "how many")
    assert parse_source_override('@"north west" total') == ("north west", "total")


def test_run_timeout_notice_explains_bounded_stop_and_retained_results():
    notice = _run_failure_notice(
        LoopExit(
            run_id="run-timeout",
            conversation_id="conversation-timeout",
            kind=LoopExitKind.FAILED,
            reason="timeout",
            created_at=datetime.now(UTC),
        )
    )

    assert "timed out after bounded retries" in notice
    assert "completed tool results remain available" in notice
    with pytest.raises(ValueError, match="usage"):
        learning_invocation_message("/learn")
    assert learning_invocation_message("Remember this") is None
    taught = learning_invocation_message("/learn keep the fiscal year")
    assert taught is not None
    assert "keep the fiscal year" in taught


def test_postgresql_url_and_redaction():
    host, port, database, username, password, ssl = parse_postgresql_connection_url(
        "postgresql://reader:p@db.example/app?sslmode=require"
    )
    assert (host, port, database, username, password, ssl) == (
        "db.example",
        5432,
        "app",
        "reader",
        "p",
        "require",
    )
    assert redact_presentation_value({"password": "x", "ok": 1}) == {
        "password": "[redacted]",
        "ok": 1,
    }


def test_tool_projection_and_approval_document():
    details = project_tool_details(
        ToolCall(id="c1", name="data_query_sqlite", arguments={"sql": "SELECT 1"}),
        ToolResultBlock(
            call_id="c1",
            output={
                "kind": "table",
                "data": {
                    "columns": ["n"],
                    "rows": [{"n": 1}],
                    "canonical_sql": "SELECT 1",
                },
            },
        ),
    )
    assert details.table is not None
    assert details.table.recorded_rows == 1
    document, reviewable = approval_review_document(
        tool_name="tool",
        capability_id="cap",
        arguments_text='{\n  "name": "safe"\n}',
    )
    assert reviewable and document is not None
    _secret_doc, secret_ok = approval_review_document(
        tool_name="tool",
        capability_id="cap",
        arguments_text='{"password": "x"}',
    )
    assert secret_ok is False
    assert approval_review_document(
        tool_name="tool",
        capability_id="cap",
        arguments_text=None,
    ) == (None, False)


def test_clipboard_mechanism_and_osc52_bounds():
    assert clipboard_mechanism(platform="darwin", environ={}) == "pbcopy"
    assert clipboard_mechanism(platform="linux", environ={}) == "osc52"
    assert (
        clipboard_mechanism(
            platform="darwin",
            environ={"SSH_TTY": "/dev/pts/1"},
        )
        == "osc52"
    )
    payload = b"hello"
    assert "52;c;" in osc52_sequence(payload, tmux=False)
    with pytest.raises(ValueError):
        osc52_sequence(b"x" * (MAX_CLIPBOARD_UTF8_BYTES + 1), tmux=False)


async def test_clipboard_reports_empty_and_oversize_truthfully():
    assert await deliver_clipboard("") == ClipboardResult(
        "failure",
        "none",
        "Copy failed: selection is empty.",
    )
    huge = "é" * (MAX_CLIPBOARD_UTF8_BYTES // 2 + 1)
    result = await deliver_clipboard(huge)
    assert result.status == "failure"
    assert "64 KiB" in result.message


async def test_app_mounts_create_screen_and_exits_on_cancel(tmp_path: Path):
    app = DaitaApp(root=tmp_path, workspace=workspace_for(tmp_path))
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        assert isinstance(app.screen, AgentCreateScreen)
        await pilot.press("escape")
        await pilot.pause()
    assert app.return_value == 0


async def test_agent_picker_create_button_routes_to_existing_creation_flow(
    tmp_path: Path,
):
    first = await Agent.create(
        "existing-one", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    await first.close()
    second = await Agent.create(
        "existing-two", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    await second.close()

    app = DaitaApp(root=tmp_path, workspace=workspace_for(tmp_path))
    async with app.run_test(size=(90, 28)) as pilot:
        for _ in range(20):
            await pilot.pause(0.05)
            if isinstance(app.screen, SelectionScreen):
                break
        picker = app.screen
        assert isinstance(picker, SelectionScreen)
        create = picker.query_one("#picker-secondary", Button)
        assert str(create.label) == "Create new agent"
        assert await pilot.click(create) is True

        for _ in range(20):
            await pilot.pause(0.05)
            if isinstance(app.screen, AgentCreateScreen):
                break
        assert isinstance(app.screen, AgentCreateScreen)

        await pilot.press("escape")
        for _ in range(20):
            await pilot.pause(0.05)
            if isinstance(app.screen, SelectionScreen):
                break
        assert isinstance(app.screen, SelectionScreen)
        assert await pilot.click("#picker-secondary") is True

        for _ in range(20):
            await pilot.pause(0.05)
            if isinstance(app.screen, AgentCreateScreen):
                break
        creation = app.screen
        assert isinstance(creation, AgentCreateScreen)
        creation.query_one("#agent-name", Input).value = "created-from-picker"
        assert await pilot.click("#create-agent") is True

        for _ in range(40):
            await pilot.pause(0.05)
            if isinstance(app.screen, ChatScreen):
                break
        assert isinstance(app.screen, ChatScreen)
        assert app.controller.require_agent().name == "created-from-picker"
        app.exit(0)

    assert await Agent.list(root=tmp_path) == (
        "created-from-picker",
        "existing-one",
        "existing-two",
    )


async def test_app_resize_and_too_small_screen(tmp_path: Path):
    app = DaitaApp(root=tmp_path, workspace=workspace_for(tmp_path))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.resize_terminal(MIN_USABLE_COLUMNS - 1, MIN_READY_ROWS - 1)
        await pilot.pause()
        assert app.size.width == MIN_USABLE_COLUMNS - 1
        app.exit(0)


async def test_agent_home_is_available_without_model_source_or_catalog(tmp_path: Path):
    opened = await Agent.create(
        "home-without-setup", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    app = DaitaApp(
        root=tmp_path, start_bootstrap=False, workspace=workspace_for(tmp_path)
    )
    app.controller.agent = opened
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            await app._ensure_ready()
            await pilot.pause()
            assert isinstance(app.screen, ChatScreen)
            notice = str(app.screen.query_one("#notice-bar", Static).content)
            assert "no model · use /model" in notice
            assert "Files:" in notice
            assert "Run source: none connected (a source is optional)" in notice
            assert "no source · use /source add" not in notice
            assert app.screen.query_one(Composer).disabled is False
            app.exit(0)
    finally:
        await opened.close()


def test_daita_theme_uses_the_official_terminal_palette():
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    theme = app.get_theme("daita")
    assert app.theme == "daita"
    assert theme is not None
    assert theme.primary == "#ACFD21"
    assert theme.background == "#000000"
    assert theme.surface == "#0D0D0D"
    assert theme.panel == "#111111"
    assert theme.boost == "#191C1F"
    assert theme.foreground == "#FFFFFF"
    assert theme.error == "#DE3535"
    assert theme.variables["block-cursor-background"] == "#ACFD21"


def test_context_window_copy_is_compact_exact_and_warns_near_capacity():
    assert format_token_count(999) == "999"
    assert format_token_count(8_500) == "8.5K"
    assert format_token_count(32_000) == "32K"
    assert format_token_count(1_050_000) == "1.1M"
    assert context_window_text(None, 32_000).plain == "ctx — / 32K"
    healthy = context_window_text(8_500, 32_000)
    warning = context_window_text(25_000, 32_000)
    critical = context_window_text(30_000, 32_000)
    assert healthy.plain == "ctx 8.5K / 32K"
    assert healthy.style == "#ACFD21"
    assert warning.style == "#FBBF24"
    assert critical.style == "#DE3535"
    with pytest.raises(ValueError):
        format_token_count(-1)


async def test_boot_and_empty_chat_show_the_responsive_daita_welcome(tmp_path: Path):
    boot = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    async with boot.run_test(size=(80, 24)):
        welcome = boot.query_one("#boot", WelcomeView)
        assert "DAITA  1.0.0" in str(welcome.content)
        assert "█████       ███" in str(welcome.content)
        assert "████████████▄" in str(welcome.content)
        assert "Your persistent data agent" in str(welcome.content)
        assert "Starting your workspace" in str(welcome.content)
        boot.exit(0)

    opened = await Agent.create(
        "welcome-agent", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    app = DaitaApp(
        root=tmp_path, start_bootstrap=False, workspace=workspace_for(tmp_path)
    )
    app.controller.agent = opened
    try:
        async with app.run_test(size=(110, 28)) as pilot:
            await app._show_chat()
            await pilot.pause()
            chat_welcome = app.screen.query_one("#welcome", WelcomeView)
            welcome_region = app.screen.query_one("#welcome-region")
            transcript = app.screen.query_one(TranscriptView)
            composer = app.screen.query_one(Composer)
            footer = app.screen.query_one(Footer)
            assert chat_welcome.display is True
            assert welcome_region.display is True
            assert transcript.display is False
            assert footer.region.y == composer.region.y + composer.region.height + 1
            assert "welcome-agent" in str(chat_welcome.content)
            assert "⣾⣿⣿" in str(chat_welcome.content)
            assert "Type / for commands" in str(chat_welcome.content)

            chat = app.chat()
            assert chat is not None
            chat.append_block(TranscriptBlock("user", "first", "hello"))
            await pilot.pause()
            assert chat_welcome.display is False
            assert welcome_region.display is False
            assert transcript.display is True
            app.exit(0)
    finally:
        await opened.close()


async def test_live_activity_and_exact_model_context_update_from_observation(
    tmp_path: Path,
):
    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="unused",
            ),
        )
    )
    profile = _mock_profile(provider)
    opened = await Agent.create(
        "observed-agent",
        root=tmp_path,
        model=provider,
        model_profile=profile,
        workspace=workspace_for(tmp_path),
    )
    app = DaitaApp(
        root=tmp_path, start_bootstrap=False, workspace=workspace_for(tmp_path)
    )
    app.controller.agent = opened
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            await app._show_chat()
            await pilot.pause()
            chat = app.chat()
            assert chat is not None
            activity = chat.query_one(ActivityBar)
            context = chat.query_one("#context-window", Static)
            assert "ctx — / 32K" in str(context.content)

            chat.set_activity("Thinking", restart=True)
            delta = AgentEvent(
                kind=AgentEventKind.MODEL_TEXT_DELTA,
                occurred_at=datetime.now(UTC),
                run_id="run-live",
                conversation_id="conversation-live",
                data=FrozenJsonObject.from_mapping(
                    {"model_call_index": 1, "text": "Streaming answer"}
                ),
            )
            await app.on_observer_event(ObserverEvent(delta))
            assert activity.display is True
            assert "Writing answer" in str(activity.content)
            assert "Streaming answer" in chat.query_one(TranscriptView).copy_text()

            completed = AgentEvent(
                kind=AgentEventKind.MODEL_COMPLETED,
                occurred_at=datetime.now(UTC),
                run_id="run-live",
                conversation_id="conversation-live",
                data=FrozenJsonObject.from_mapping(
                    {
                        "provider_id": provider.provider_id,
                        "model_call_index": 1,
                        "has_text": True,
                        "has_tool_calls": False,
                        "duration_ms": 12,
                        "input_tokens": 8_500,
                        "context_input_tokens": 8_500,
                        "output_tokens": 50,
                    }
                ),
            )
            await app.on_observer_event(ObserverEvent(completed))
            await pilot.pause()
            assert "ctx 8.5K / 32K" in str(context.content)

            tool_completed = AgentEvent(
                kind=AgentEventKind.TOOL_COMPLETED,
                occurred_at=datetime.now(UTC),
                run_id="run-live",
                conversation_id="conversation-live",
                data=FrozenJsonObject.from_mapping(
                    {
                        "tool_name": "data_query_postgresql",
                        "capability_id": "data.postgresql.query",
                        "call_id": "call-live",
                        "duration_ms": 10,
                        "is_error": False,
                        "error_code": None,
                    }
                ),
            )
            await app.on_observer_event(ObserverEvent(tool_completed))
            assert "Processing results" in str(activity.content)

            run_completed = AgentEvent(
                kind=AgentEventKind.RUN_COMPLETED,
                occurred_at=datetime.now(UTC),
                run_id="run-live",
                conversation_id="conversation-live",
                data=FrozenJsonObject.from_mapping({"exit_kind": "completed"}),
            )
            await app.on_observer_event(ObserverEvent(run_completed))
            assert activity.display is False
            app.exit(0)
    finally:
        await opened.close()


async def test_typed_command_palette_navigates_and_inserts_without_submitting(
    tmp_path: Path,
):
    opened = await Agent.create(
        "palette-agent", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    app = DaitaApp(
        root=tmp_path, start_bootstrap=False, workspace=workspace_for(tmp_path)
    )
    app.controller.agent = opened
    try:
        async with app.run_test(size=(90, 28)) as pilot:
            await app._show_chat()
            await pilot.pause()
            composer = app.screen.query_one(Composer)
            popup = app.screen.query_one(CompletionPopup)
            listing = popup.query_one(OptionList)
            composer.focus()
            assert composer.styles.border_top[0] == "round"
            assert composer.styles.border_right[0] == "round"

            await pilot.press("/")
            await pilot.pause()
            assert popup.display is True
            assert popup.styles.border_left[0] == "solid"
            assert popup.styles.background.hex == "#111111"
            assert (
                listing.get_component_styles(
                    "option-list--option-highlighted"
                ).background.hex
                == "#343434"
            )
            assert listing.highlighted == 0
            assert popup.selected_insertion() == "/model"

            await pilot.press("down", "enter")
            await pilot.pause()
            assert composer.text == "/sources"
            assert popup.display is False
            assert app._run_task is None

            await pilot.press("enter")
            await pilot.pause()
            assert composer.text == ""
            assert app._run_task is None

            await pilot.press("/", "s", "o", "u")
            await pilot.pause()
            assert popup.display is True
            assert {shown for _insert, shown, _description in popup.matches} >= {
                "/source",
                "/sources",
            }

            await pilot.press("escape")
            await pilot.pause()
            assert popup.display is False
            assert composer.text == "/sou"
            await pilot.press("escape", "escape")
            await pilot.pause()
            assert composer.text == ""

            await pilot.resize_terminal(MIN_USABLE_COLUMNS, MIN_READY_ROWS)
            await pilot.pause()
            composer.focus()
            await pilot.press("/")
            await pilot.pause()
            assert popup.display is True
            assert composer.region.height >= 3
            assert listing.region.height >= 1
            assert popup.region.y < composer.region.y
            app.exit(0)
    finally:
        await opened.close()


async def test_skill_and_source_completions_are_plain_and_selectable(tmp_path: Path):
    database = tmp_path / "completion.sqlite"
    _create_sqlite_source(database, "records")
    opened = await Agent.create(
        "completion-agent", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    await opened.attach(SQLiteSource(database, name="North [bold]"))
    await opened.save_skill(
        "audit",
        "Review [bold red]recorded totals[/]",
        "Check each recorded total against its source.",
    )
    app = DaitaApp(
        root=tmp_path, start_bootstrap=False, workspace=workspace_for(tmp_path)
    )
    app.controller.agent = opened
    try:
        async with app.run_test(size=(90, 28)) as pilot:
            await app._show_chat()
            await pilot.pause()
            composer = app.screen.query_one(Composer)
            popup = app.screen.query_one(CompletionPopup)
            listing = popup.query_one(OptionList)
            composer.focus()

            await pilot.press("/", "a", "u")
            await pilot.pause()
            assert popup.matches == (
                (
                    "/audit ",
                    "/audit",
                    "Review [bold red]recorded totals[/]",
                ),
            )
            option = listing.get_option_at_index(0)
            assert isinstance(option.prompt, Text)
            assert "[bold red]recorded totals[/]" in option.prompt.plain
            await pilot.press("tab")
            await pilot.pause()
            assert composer.text == "/audit "
            assert popup.display is False

            composer.clear()
            await pilot.press("@")
            await pilot.pause()
            assert popup.display is True
            assert popup.matches[0][1] == "@North [bold]"
            listing.action_select()
            await pilot.pause()
            assert composer.text == '@"North [bold]" '
            assert popup.display is False
            app.exit(0)
    finally:
        await opened.close()


async def test_selection_screen_filters_without_reordering():
    class Harness(App[tuple[str, ...] | None]):
        def compose(self) -> ComposeResult:
            yield Static("host")

        def on_mount(self) -> None:
            self.run_worker(self._present(), exclusive=True)

        async def _present(self) -> None:
            result = await self.push_screen_wait(
                SelectionScreen(
                    title="Pick",
                    options=(
                        PickerOption("a", "Alpha"),
                        PickerOption("b", "Beta"),
                        PickerOption("c", "Gamma"),
                    ),
                )
            )
            self.exit(result)

    app = Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, SelectionScreen)
        picker.query_one("#picker-options", OptionList).highlighted = 1
        picker.action_confirm()
        await pilot.pause()
    assert app.return_value == ("b",)


async def test_selection_screen_secondary_action_is_distinct_from_options():
    class Harness(App[tuple[str, ...] | None]):
        def compose(self) -> ComposeResult:
            yield Static("host")

        def on_mount(self) -> None:
            self.run_worker(self._present(), exclusive=True)

        async def _present(self) -> None:
            result = await self.push_screen_wait(
                SelectionScreen(
                    title="Pick",
                    options=(PickerOption("existing", "Existing"),),
                    secondary_action=PickerOption("create", "Create new"),
                )
            )
            self.exit(result)

    with pytest.raises(ValueError, match="must differ"):
        SelectionScreen(
            title="Invalid",
            options=(PickerOption("same", "Existing"),),
            secondary_action=PickerOption("same", "Create new"),
        )

    app = Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, SelectionScreen)
        assert await pilot.click("#picker-secondary") is True
        await pilot.pause()
    assert app.return_value == ("create",)


async def test_multi_selection_shows_literal_marks_and_continue_button():
    class Harness(App[tuple[str, ...] | None]):
        def compose(self) -> ComposeResult:
            yield Static("host")

        def on_mount(self) -> None:
            self.run_worker(self._present(), exclusive=True)

        async def _present(self) -> None:
            result = await self.push_screen_wait(
                SelectionScreen(
                    title="Select PostgreSQL schemas",
                    options=(
                        PickerOption("core", "core", "contains tables"),
                        PickerOption("public", "public", "empty"),
                    ),
                    multi=True,
                    initial_selected=("core",),
                )
            )
            self.exit(result)

    app = Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, SelectionScreen)
        listing = picker.query_one("#picker-options", OptionList)
        selected = listing.get_option_at_index(0)
        unselected = listing.get_option_at_index(1)
        assert isinstance(selected.prompt, Text)
        assert selected.prompt.plain == "[x] core — contains tables"
        assert isinstance(unselected.prompt, Text)
        assert unselected.prompt.plain == "[ ] public — empty"
        button = picker.query_one("#picker-confirm", Button)
        assert str(button.label) == "Continue"
        assert await pilot.click(button) is True
        await pilot.pause()
    assert app.return_value == ("core",)


def _tui_inbox_item(
    delivery_id: str = "delivery-stage-c",
    *,
    report: str = "The durable profile completed successfully.",
) -> InboxItem:
    observed = datetime(2026, 8, 23, 14, 5, tzinfo=UTC)
    return InboxItem(
        delivery_id=delivery_id,
        agent_id="agent-inbox",
        conversation_id="conversation-inbox",
        subject=DeliverySubject(
            kind=DeliverySubjectKind.STANDALONE_FOLLOWUP,
            subject_id="followup-inbox",
        ),
        resulting_run_id="run-followup",
        grant_id="grant-followup",
        logical_key="standalone_followup:followup-inbox:conclusion",
        conclusion_digest="sha256:" + "1" * 64,
        payload={
            "subject": {
                "kind": "standalone_followup",
                "subject_id": "followup-inbox",
            },
            "job_id": "job-profile",
            "run_id": "run-followup",
            "outcome": "completed",
            "reason": "completed",
            "report_digest": "sha256:" + "2" * 64,
            "report_preview": report,
            "report_truncated": False,
            "evidence_digest": "sha256:" + "3" * 64,
        },
        sensitivity=ModelSensitivity.INTERNAL,
        destination="conversation_inbox:conversation-inbox",
        destination_sensitivity_ceiling=ModelSensitivity.INTERNAL,
        state=DeliveryState.AVAILABLE,
        created_at=observed,
        updated_at=observed,
    )


def _tui_job_summary(
    job_id: str,
    status: JobStatus,
    *,
    result_available: bool,
) -> SimpleNamespace:
    observed = datetime(2026, 8, 23, 14, 0, tzinfo=UTC)
    return SimpleNamespace(
        job_id=job_id,
        origin_conversation_id="conversation-jobs",
        job_kind="data_profile",
        status=status,
        execution_mode=SimpleNamespace(value="daita"),
        source_ids=("source-one",),
        resource_ids=("resource-one",),
        sensitivity=SimpleNamespace(value="internal"),
        created_at=observed,
        updated_at=observed,
        result_available=result_available,
    )


def _tui_job_inspection(summary: SimpleNamespace) -> SimpleNamespace:
    observed = datetime(2026, 8, 23, 14, 0, tzinfo=UTC)
    return SimpleNamespace(
        summary=summary,
        origin_run_id="run-jobs",
        specification_digest="sha256:" + "1" * 64,
        execution_capability_id="jobs.data_profile.execute",
        execution_contract_digest="sha256:" + "2" * 64,
        desired_state=SimpleNamespace(
            value=(
                "cancel"
                if summary.status in {JobStatus.CANCEL_REQUESTED, JobStatus.CANCELLED}
                else "run"
            )
        ),
        deadline_at=observed,
        attempts=(
            SimpleNamespace(
                number=1,
                fencing_epoch=1,
                status=SimpleNamespace(value="claimed"),
                claimed_at=observed,
                completed_at=None,
                error_code=None,
                external_intents=(),
                external_observations=(),
            ),
        ),
        cancel_requested_at=(
            observed
            if summary.status in {JobStatus.CANCEL_REQUESTED, JobStatus.CANCELLED}
            else None
        ),
        terminal_at=(observed if summary.status is JobStatus.CANCELLED else None),
        failure_code=None,
        external_executor=None,
    )


async def test_jobs_commands_route_without_model_calls(monkeypatch):
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    running = _tui_job_summary("job-running", JobStatus.RUNNING, result_available=False)
    succeeded = _tui_job_summary(
        "job-succeeded", JobStatus.SUCCEEDED, result_available=True
    )

    async def inspect_job(job_id: str) -> object | None:
        if job_id == running.job_id:
            return _tui_job_inspection(running)
        if job_id == succeeded.job_id:
            return _tui_job_inspection(succeeded)
        return None

    monkeypatch.setattr(app.controller, "inspect_job", inspect_job)

    assert {
        insertion
        for insertion, _display, _description in SLASH_COMMAND_COMPLETIONS
        if insertion.startswith("/jobs")
    } == {"/jobs", "/jobs inspect ", "/jobs results ", "/jobs cancel "}

    listed = await app.controller.dispatch_command("/jobs")
    assert listed.kind == "screen"
    assert listed.screen == "jobs"

    inspected = await app.controller.dispatch_command("/jobs inspect job-running")
    assert inspected.screen == "jobs"
    assert inspected.payload == {"job_id": "job-running", "view": "inspect"}

    results = await app.controller.dispatch_command("/jobs results job-succeeded")
    assert results.screen == "jobs"
    assert results.payload == {"job_id": "job-succeeded", "view": "results"}

    cancellation = await app.controller.dispatch_command("/jobs cancel job-running")
    assert cancellation.kind == "confirm"
    assert cancellation.screen == "confirm_cancel_job"
    assert cancellation.payload == {"job_id": "job-running"}
    assert "data_profile · running" in cancellation.message

    terminal = await app.controller.dispatch_command("/jobs cancel job-succeeded")
    assert terminal.kind == "notice"
    assert "succeeded and cannot be cancelled" in terminal.message

    malformed = await app.controller.dispatch_command("/jobs retry job-running")
    assert malformed.kind == "notice"
    assert malformed.message.startswith("Usage: /jobs")

    with pytest.raises(UserInputError, match="belongs to this agent"):
        await app.controller.dispatch_command("/jobs cancel job-missing")


async def test_inbox_command_routes_without_a_model_call():
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))

    assert (
        "/inbox",
        "/inbox",
        "Inspect and acknowledge completed background reports",
    ) in SLASH_COMMAND_COMPLETIONS
    outcome = await app.controller.dispatch_command("/inbox")
    assert outcome.kind == "screen"
    assert outcome.screen == "inbox"
    malformed = await app.controller.dispatch_command("/inbox extra")
    assert malformed.kind == "notice"
    assert malformed.message == "Usage: /inbox"


async def test_inbox_screen_inspects_sanitizes_and_acknowledges(monkeypatch):
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    item = _tui_inbox_item(report="Ready\x1b[31m @everyone")
    items = [item]
    acknowledgments: list[str] = []

    async def list_inbox() -> tuple[InboxItem, ...]:
        return tuple(items)

    async def acknowledge_inbox(delivery_id: str) -> InboxItem | None:
        acknowledgments.append(delivery_id)
        items.clear()
        return item

    monkeypatch.setattr(app.controller, "list_inbox", list_inbox)
    monkeypatch.setattr(app.controller, "acknowledge_inbox", acknowledge_inbox)

    async with app.run_test(size=(110, 36)) as pilot:
        await app.push_screen(InboxScreen())
        for _ in range(20):
            await pilot.pause(0.05)
            if "1 unacknowledged result" in str(
                app.screen.query_one("#inbox-summary", Static).content
            ):
                break
        manager = app.screen
        assert isinstance(manager, InboxScreen)
        listing = manager.query_one("#inbox-list", OptionList)
        assert listing.option_count == 1
        assert listing.has_focus is True
        detail = str(manager.query_one("#inbox-detail", Static).content)
        assert "Ready?[31m @everyone" in detail
        assert "\x1b" not in detail
        assert "Result run: run-followup" in detail
        assert manager.query_one("#inbox-acknowledge", Button).disabled is False

        assert await pilot.click("#inbox-acknowledge") is True
        for _ in range(20):
            await pilot.pause(0.05)
            if "0 unacknowledged results" in str(
                manager.query_one("#inbox-summary", Static).content
            ):
                break
        assert acknowledgments == [item.delivery_id]
        assert listing.option_count == 0
        assert "never reruns reasoning" in str(
            manager.query_one("#inbox-help", Static).content
        )
        app.exit(0)


def test_inbox_rendering_withholds_blocked_reports_and_marks_bounded_previews():
    available = _tui_inbox_item(report="bounded preview")
    truncated = replace(
        available,
        payload={**dict(available.payload), "report_truncated": True},
    )
    assert "Preview truncated" in render_inbox_item(truncated)

    blocked = replace(
        available,
        state=DeliveryState.BLOCKED,
        payload={**dict(available.payload), "report_preview": None},
        destination_sensitivity_ceiling=ModelSensitivity.PUBLIC,
        terminal_error="delivery_sensitivity_ineligible",
    )
    rendered = render_inbox_item(blocked)
    assert "bounded preview" not in rendered
    assert "Preview withheld" in rendered
    assert "delivery_sensitivity_ineligible" in rendered


async def test_background_status_notifies_once_and_remains_outside_transcript(
    tmp_path: Path,
    monkeypatch,
):
    opened = await Agent.create(
        "inbox-status-agent", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    app = DaitaApp(
        root=tmp_path, start_bootstrap=False, workspace=workspace_for(tmp_path)
    )
    app.controller.agent = opened
    running = _tui_job_summary(
        "job-running-status", JobStatus.RUNNING, result_available=False
    )
    current_inbox: list[InboxItem] = []
    notifications: list[tuple[str, str | None]] = []

    async def list_jobs() -> tuple[object, ...]:
        return (running,)

    async def list_inbox() -> tuple[InboxItem, ...]:
        return tuple(current_inbox)

    def notify(message: str, *, title: str | None = None, **_kwargs: object) -> None:
        notifications.append((message, title))

    monkeypatch.setattr(app.controller, "list_jobs", list_jobs)
    monkeypatch.setattr(app.controller, "list_inbox", list_inbox)
    monkeypatch.setattr(app, "notify", notify)
    try:
        async with app.run_test(size=(110, 32)) as pilot:
            await app._show_chat()
            await pilot.pause()
            status = app.screen.query_one("#background-status", Static)
            assert "jobs 1" in str(status.content)
            assert status.display is True

            current_inbox.append(_tui_inbox_item())
            await app.refresh_background_status(notify_new=True)
            await pilot.pause()
            assert "jobs 1" in str(status.content)
            assert "inbox 1" in str(status.content)
            assert notifications == [
                ("1 background report is ready. Open /inbox to review.", "Inbox")
            ]
            await app.refresh_background_status(notify_new=True)
            assert len(notifications) == 1
            assert app.screen.query_one(TranscriptView).is_empty
            app.exit(0)
    finally:
        await opened.close()


async def test_machine_origin_observations_do_not_project_into_foreground_chat(
    tmp_path: Path,
):
    opened = await Agent.create(
        "origin-isolation-agent", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    app = DaitaApp(
        root=tmp_path, start_bootstrap=False, workspace=workspace_for(tmp_path)
    )
    app.controller.agent = opened
    observed = datetime.now(UTC)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            await app._show_chat()
            await pilot.pause()
            chat = app.chat()
            assert chat is not None
            transcript = chat.query_one(TranscriptView)
            context = chat.query_one("#context-window", Static)

            await app.on_observer_event(
                ObserverEvent(
                    AgentEvent(
                        kind=AgentEventKind.RUN_STARTED,
                        occurred_at=observed,
                        run_id="run-autonomous",
                        conversation_id="conversation-origin",
                        data=FrozenJsonObject.from_mapping({"agent_id": opened.id}),
                        run_origin="job_event",
                    )
                )
            )
            assert "reporting 1" in str(
                chat.query_one("#background-status", Static).content
            )

            await app.on_observer_event(
                ObserverEvent(
                    AgentEvent(
                        kind=AgentEventKind.MODEL_TEXT_DELTA,
                        occurred_at=observed,
                        run_id="run-autonomous",
                        conversation_id="conversation-origin",
                        data=FrozenJsonObject.from_mapping(
                            {"model_call_index": 1, "text": "Hidden report draft"}
                        ),
                        run_origin="job_event",
                    )
                )
            )
            await app.on_observer_event(
                ObserverEvent(
                    AgentEvent(
                        kind=AgentEventKind.MODEL_COMPLETED,
                        occurred_at=observed,
                        run_id="run-autonomous",
                        conversation_id="conversation-origin",
                        data=FrozenJsonObject.from_mapping(
                            {
                                "provider_id": "mock:scripted",
                                "model_call_index": 1,
                                "context_input_tokens": 9_999,
                            }
                        ),
                        run_origin="job_event",
                    )
                )
            )
            assert transcript.is_empty
            assert "Hidden report draft" not in transcript.copy_text()
            assert "ctx —" in str(context.content)
            assert chat.query_one(ActivityBar).display is False

            await app.on_observer_event(
                ObserverEvent(
                    AgentEvent(
                        kind=AgentEventKind.RUN_COMPLETED,
                        occurred_at=observed,
                        run_id="run-autonomous",
                        conversation_id="conversation-origin",
                        data=FrozenJsonObject.from_mapping({"exit_kind": "completed"}),
                        run_origin="job_event",
                    )
                )
            )
            assert "reporting" not in str(
                chat.query_one("#background-status", Static).content
            )
            app.exit(0)
    finally:
        await opened.close()


async def test_jobs_manager_lists_inspects_reads_cancels_and_refreshes(monkeypatch):
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    running = _tui_job_summary(
        "job-running-0123456789", JobStatus.RUNNING, result_available=False
    )
    succeeded = _tui_job_summary(
        "job-succeeded-0123456789", JobStatus.SUCCEEDED, result_available=True
    )
    jobs = [running, succeeded]
    list_calls = 0
    cancel_calls: list[str] = []

    async def list_jobs() -> tuple[object, ...]:
        nonlocal list_calls
        list_calls += 1
        return tuple(jobs)

    async def inspect_job(job_id: str) -> object | None:
        return next(
            (
                _tui_job_inspection(summary)
                for summary in jobs
                if summary.job_id == job_id
            ),
            None,
        )

    async def read_job_result(job_id: str) -> object | None:
        if job_id != succeeded.job_id:
            return None
        observed = datetime(2026, 8, 23, 14, 1, tzinfo=UTC)
        return SimpleNamespace(
            job_id=job_id,
            result_id="result-profile",
            summary=FrozenJsonObject.from_mapping({"profiled_resources": 1}),
            sensitivity=SimpleNamespace(value="internal"),
            provenance=FrozenJsonObject.from_mapping(
                {"authority": "job_owner_agent_scope"}
            ),
            artifact_refs=(),
            completed_at=observed,
        )

    async def cancel_job(job_id: str) -> object | None:
        cancel_calls.append(job_id)
        if job_id != running.job_id:
            return None
        cancelled = _tui_job_summary(
            running.job_id, JobStatus.CANCEL_REQUESTED, result_available=False
        )
        jobs[0] = cancelled
        return _tui_job_inspection(cancelled)

    monkeypatch.setattr(app.controller, "list_jobs", list_jobs)
    monkeypatch.setattr(app.controller, "inspect_job", inspect_job)
    monkeypatch.setattr(app.controller, "read_job_result", read_job_result)
    monkeypatch.setattr(app.controller, "cancel_job", cancel_job)

    async with app.run_test(size=(110, 36)) as pilot:
        await app.push_screen(JobsScreen())
        for _ in range(20):
            await pilot.pause(0.05)
            if "2 jobs" in str(app.screen.query_one("#jobs-summary", Static).content):
                break
        manager = app.screen
        assert isinstance(manager, JobsScreen)
        panel = manager.query_one("#jobs-manager")
        assert panel.styles.border_left[0] == "solid"
        assert panel.styles.background.hex == "#111111"
        listing = manager.query_one("#jobs-list", OptionList)
        assert listing.option_count == 2
        assert listing.has_focus is True
        first_prompt = listing.get_option_at_index(0).prompt
        assert isinstance(first_prompt, Text)
        assert "RUNNING" in first_prompt.plain
        assert manager.query_one("#jobs-cancel", Button).disabled is False
        assert manager.query_one("#jobs-results", Button).disabled is True

        assert await pilot.click("#jobs-details") is True
        await pilot.pause()
        assert "Lifecycle" in str(manager.query_one("#jobs-detail", Static).content)

        assert await pilot.click("#jobs-cancel") is True
        for _ in range(20):
            await pilot.pause(0.05)
            if isinstance(app.screen, ConfirmScreen):
                break
        assert isinstance(app.screen, ConfirmScreen)
        await pilot.press("y")
        for _ in range(20):
            await pilot.pause(0.05)
            if app.screen is manager and cancel_calls:
                break
        assert app.screen is manager
        assert cancel_calls == [running.job_id]
        assert "Cancellation requested" in str(
            manager.query_one("#jobs-notice", Static).content
        )
        assert manager.query_one("#jobs-cancel", Button).disabled is True

        listing.highlighted = 1
        await pilot.pause()
        assert manager.query_one("#jobs-results", Button).disabled is False
        assert await pilot.click("#jobs-results") is True
        await pilot.pause()
        result_text = str(manager.query_one("#jobs-detail", Static).content)
        assert "profiled_resources" in result_text
        assert "Artifacts (0)" in result_text

        assert await pilot.click("#jobs-refresh") is True
        await pilot.pause()
        assert list_calls >= 3
        assert "Job statuses refreshed" in str(
            manager.query_one("#jobs-notice", Static).content
        )
        assert await pilot.click("#jobs-close") is True
        app.exit(0)


async def test_direct_jobs_cancel_command_confirms_and_opens_updated_manager(
    monkeypatch,
):
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    running = _tui_job_summary(
        "job-direct-cancel", JobStatus.RUNNING, result_available=False
    )
    current = running
    cancel_calls: list[str] = []

    async def list_jobs() -> tuple[object, ...]:
        return (current,)

    async def inspect_job(job_id: str) -> object | None:
        return _tui_job_inspection(current) if job_id == current.job_id else None

    async def cancel_job(job_id: str) -> object | None:
        nonlocal current
        cancel_calls.append(job_id)
        current = _tui_job_summary(
            job_id, JobStatus.CANCEL_REQUESTED, result_available=False
        )
        return _tui_job_inspection(current)

    async def skill_invocation_message(_message: str) -> None:
        return None

    monkeypatch.setattr(app.controller, "list_jobs", list_jobs)
    monkeypatch.setattr(app.controller, "inspect_job", inspect_job)
    monkeypatch.setattr(app.controller, "cancel_job", cancel_job)
    monkeypatch.setattr(
        app.controller, "skill_invocation_message", skill_invocation_message
    )

    async with app.run_test(size=(110, 36)) as pilot:
        await app.push_screen(ChatScreen())
        composer = app.screen.query_one(Composer)
        composer.load_text(f"/jobs cancel {running.job_id}")
        composer.action_submit()
        for _ in range(20):
            await pilot.pause(0.05)
            if isinstance(app.screen, ConfirmScreen):
                break
        assert isinstance(app.screen, ConfirmScreen)
        assert "data_profile · running" in str(
            app.screen.query_one("#confirm-message").render()
        )
        await pilot.press("y")
        for _ in range(20):
            await pilot.pause(0.05)
            if isinstance(app.screen, JobsScreen):
                break
        manager = app.screen
        assert isinstance(manager, JobsScreen)
        assert cancel_calls == [running.job_id]
        assert "Cancellation requested" in str(
            manager.query_one("#jobs-notice", Static).content
        )
        app.exit(0)


async def test_source_permissions_picker_remains_interactive_after_command_submit(
    tmp_path: Path,
):
    database = tmp_path / "permissions.sqlite"
    _create_sqlite_source(database, "records")
    opened = await Agent.create(
        "permissions-picker", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    source = await opened.attach(SQLiteSource(database, name="Records"))
    app = DaitaApp(
        root=tmp_path, start_bootstrap=False, workspace=workspace_for(tmp_path)
    )
    app.controller.agent = opened
    try:
        async with app.run_test(size=(90, 28)) as pilot:
            await app._show_chat()
            composer = app.screen.query_one(Composer)
            composer.load_text("/source permissions")
            composer.action_submit()
            await pilot.pause()

            permissions = app.screen
            assert isinstance(permissions, PermissionsScreen)
            assert await pilot.click("#perm-source") is True
            await pilot.pause()

            picker = app.screen
            assert isinstance(picker, SelectionScreen)
            panel = picker.query_one("#picker")
            assert panel.styles.border_left[0] == "solid"
            assert panel.styles.background.hex == "#111111"
            picker_filter = picker.query_one("#picker-filter", Input)
            assert picker_filter.styles.border_left[0] == ""
            assert picker_filter.styles.border_bottom[0] == "solid"
            assert picker.query_one("#picker-title").styles.color.hex == "#FFFFFF"
            listing = picker.query_one("#picker-options", OptionList)
            assert listing.styles.border_left[0] == ""
            assert listing.styles.background.hex == "#111111"
            assert (
                listing.get_component_styles(
                    "option-list--option-highlighted"
                ).background.hex
                == "#343434"
            )
            assert await pilot.click(listing, offset=(2, 0)) is True
            await pilot.pause()

            assert app.screen is permissions
            assert isinstance(permissions, PermissionsScreen)
            assert permissions._source_id == source.id
            assert "Records" in str(permissions.query_one("#perm-body", Static).content)
            app.exit(0)
    finally:
        await opened.close()


async def test_source_permissions_configures_exact_postgresql_update_scope():
    read_scope = SimpleNamespace(mode=SimpleNamespace(value="all"), resource_ids=())
    initial_state = SimpleNamespace(
        read_scope=read_scope,
        postgresql_update_scopes=(),
    )
    tickets = SimpleNamespace(
        resource_id="resource-tickets",
        display_name="support.tickets",
        resource_kind="table",
        eligible_assignment_columns=("priority", "ticket_status"),
        postgresql_update_eligible=True,
        requires_advanced_column_selection=False,
    )
    inspection = SimpleNamespace(
        source_id="source-postgresql",
        source_display_name="Support PostgreSQL",
        adapter_id="postgresql",
        catalog_generation="sync-one",
        state=initial_state,
        resources=(tickets,),
    )
    preview_calls: list[dict[str, object]] = []
    apply_calls: list[dict[str, object]] = []

    async def inspect_source_permissions(source_id: str):
        assert source_id == inspection.source_id
        return inspection

    async def preview_source_permissions(**kwargs: object):
        preview_calls.append(kwargs)
        updates = kwargs["postgresql_update_scopes"]
        assert isinstance(updates, dict)
        scopes = tuple(
            SimpleNamespace(
                resource_id=resource_id,
                allowed_assignment_columns=tuple(columns),
            )
            for resource_id, columns in updates.items()
        )
        after = SimpleNamespace(
            read_scope=SimpleNamespace(
                mode=SimpleNamespace(value=kwargs["read_mode"]),
                resource_ids=kwargs["read_resource_ids"],
            ),
            postgresql_update_scopes=scopes,
        )
        return SimpleNamespace(
            source_id=inspection.source_id,
            catalog_generation=inspection.catalog_generation,
            before=initial_state,
            after=after,
            confirmation_fingerprint="sha256:" + "1" * 64,
        )

    async def apply_source_permissions(**kwargs: object):
        apply_calls.append(kwargs)
        return inspection

    def choose_single(app: DaitaApp, identity: str) -> None:
        picker = app.screen
        assert isinstance(picker, SelectionScreen)
        listing = picker.query_one("#picker-options", OptionList)
        listing.highlighted = next(
            index
            for index in range(listing.option_count)
            if str(listing.get_option_at_index(index).id) == identity
        )
        picker.action_confirm()

    def choose_multi(app: DaitaApp, identity: str) -> None:
        picker = app.screen
        assert isinstance(picker, SelectionScreen)
        listing = picker.query_one("#picker-options", OptionList)
        listing.highlighted = next(
            index
            for index in range(listing.option_count)
            if str(listing.get_option_at_index(index).id) == identity
        )
        picker.action_toggle_selected()
        picker.action_confirm()

    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    app.controller.inspect_source_permissions = inspect_source_permissions  # type: ignore[method-assign]
    app.controller.preview_source_permissions = preview_source_permissions  # type: ignore[method-assign]
    app.controller.apply_source_permissions = apply_source_permissions  # type: ignore[method-assign]
    async with app.run_test(size=(100, 32)) as pilot:
        await app.push_screen(PermissionsScreen(source_id=inspection.source_id))
        permissions = app.screen
        assert isinstance(permissions, PermissionsScreen)

        assert await pilot.click("#perm-update") is True
        await pilot.pause()
        choose_single(app, "selected")
        await pilot.pause()
        choose_multi(app, tickets.resource_id)
        await pilot.pause()
        choose_single(app, "advanced")
        await pilot.pause()
        choose_multi(app, "priority")
        await pilot.pause()

        assert app.screen is permissions
        assert preview_calls == [
            {
                "source_id": inspection.source_id,
                "read_mode": "all",
                "read_resource_ids": (),
                "postgresql_update_scopes": {
                    tickets.resource_id: ("priority",),
                },
            }
        ]
        body = str(permissions.query_one("#perm-body", Static).content)
        assert "PostgreSQL update tables: 0 → 1" in body
        assert "support.tickets: priority" in body

        assert await pilot.click("#perm-apply") is True
        await pilot.pause()
        assert apply_calls == [
            {
                "source_id": inspection.source_id,
                "confirmation_fingerprint": "sha256:" + "1" * 64,
            }
        ]
        app.exit(0)


async def test_approval_approve_deny_cancel_and_unreviewable():
    request = ApprovalRequest(
        run_id="run-1",
        call_id="call-1",
        tool_name="data_update_postgresql",
        capability_id="data.postgresql.update",
        arguments=FrozenJsonObject.from_mapping({"name": "safe"}),
        reason="update one row",
    )
    secret = ApprovalRequest(
        run_id="run-2",
        call_id="call-2",
        tool_name="data_update_postgresql",
        capability_id="data.postgresql.update",
        arguments=FrozenJsonObject.from_mapping({"password": "hidden-secret"}),
        reason="secret shaped",
    )
    oversized = ApprovalRequest(
        run_id="run-3",
        call_id="call-3",
        tool_name="data_update_postgresql",
        capability_id="data.postgresql.update",
        arguments=FrozenJsonObject.from_mapping({"blob": "x" * (70 * 1024)}),
        reason="too big",
    )

    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    async with app.run_test(size=(80, 24)) as pilot:
        await app.push_screen(ChatScreen())
        chat = app.screen
        assert isinstance(chat, ChatScreen)
        panel = chat.query_one(ApprovalPanel)

        approved = asyncio.create_task(app.handle_approval(request))
        await pilot.pause()
        assert app.screen is chat
        assert panel.display is True
        assert panel.styles.border_left[0] == "solid"
        assert panel.region.y < chat.query_one(Composer).region.y
        await pilot.press("y")
        await pilot.pause()
        assert await approved is ApprovalDecision.APPROVE
        assert panel.display is False

        denied = asyncio.create_task(app.handle_approval(request))
        await pilot.pause()
        await pilot.press("n")
        await pilot.pause()
        assert await denied is ApprovalDecision.DENY

        cancelled = asyncio.create_task(app.handle_approval(request))
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        with pytest.raises(asyncio.CancelledError):
            await cancelled

        hidden = asyncio.create_task(app.handle_approval(secret))
        await pilot.pause()
        assert panel.query_one("#approval-inline-unreviewable").display is True
        panel.action_cancel()
        await pilot.pause()
        with pytest.raises(asyncio.CancelledError):
            await hidden

        hidden_oversize = asyncio.create_task(app.handle_approval(oversized))
        await pilot.pause()
        assert panel.query_one("#approval-inline-unreviewable").display is True
        panel.action_cancel()
        await pilot.pause()
        with pytest.raises(asyncio.CancelledError):
            await hidden_oversize
        app.exit(0)

    too_small = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    async with too_small.run_test(size=(40, 10)):
        await too_small.push_screen(ChatScreen())
        with pytest.raises(RuntimeError, match="too small"):
            await too_small.handle_approval(request)
        too_small.exit(0)


async def test_destructive_confirm_requires_explicit_yes():
    class Harness(App[bool]):
        def compose(self) -> ComposeResult:
            yield Static("host")

        def on_mount(self) -> None:
            self.run_worker(self._present(), exclusive=True)

        async def _present(self) -> None:
            result: bool = await self.push_screen_wait(ConfirmScreen("Delete?"))
            self.exit(result)

    declined = Harness()
    async with declined.run_test() as pilot:
        await pilot.pause()
        await pilot.press("n")
        await pilot.pause()
    assert declined.return_value is False

    accepted = Harness()
    async with accepted.run_test() as pilot:
        await pilot.pause()
        await pilot.press("y")
        await pilot.pause()
    assert accepted.return_value is True


def _mock_profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=32_000,
        max_output_tokens=1_024,
        supports_tools=True,
        supports_parallel_tools=True,
        supports_streaming=True,
    )


async def test_composer_submit_runs_agent_once(tmp_path: Path):
    from datetime import UTC, datetime

    from daita import LoopExit, LoopExitKind
    from daita.tui.screens.chat import ChatScreen

    opened = await Agent.create(
        "ready", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    calls: list[str] = []

    async def fake_run(message: str, **_kwargs: object) -> LoopExit:
        calls.append(message)
        return LoopExit(
            run_id="run-test",
            conversation_id="conv-test",
            kind=LoopExitKind.COMPLETED,
            reason="done",
            created_at=datetime.now(UTC),
            final_text="one answer",
        )

    opened.run = fake_run  # type: ignore[method-assign]
    app = DaitaApp(
        root=tmp_path, start_bootstrap=False, workspace=workspace_for(tmp_path)
    )
    app.controller.agent = opened
    try:
        async with app.run_test(size=(80, 24)) as pilot:
            await app._show_chat()
            await pilot.pause()
            assert isinstance(app.screen, ChatScreen)
            await app.submit_composer("what is ready?")
            assert app._run_task is not None
            await app._run_task
            await pilot.pause()
            assert calls == ["what is ready?"]
            assert MAX_COMPOSER_CHARACTERS == 16_384
            app.exit(0)
    finally:
        await opened.close()


async def test_chat_accumulates_and_resumes_the_full_conversation_transcript(
    tmp_path: Path,
):
    provider = MockModelProvider(
        (
            ModelResponse(finish_reason=FinishReason.STOP, text="First answer"),
            ModelResponse(finish_reason=FinishReason.STOP, text="Second answer"),
        )
    )
    opened = await Agent.create(
        "conversation-transcript",
        root=tmp_path,
        model=provider,
        model_profile=_mock_profile(provider),
        workspace=workspace_for(tmp_path),
    )
    app = DaitaApp(
        root=tmp_path, start_bootstrap=False, workspace=workspace_for(tmp_path)
    )
    app.controller.agent = opened
    try:
        async with app.run_test(size=(80, 18)) as pilot:
            await app._show_chat()
            await app.submit_composer("First question")
            assert app._run_task is not None
            await app._run_task
            await pilot.pause()
            conversation_id = app.controller.conversation_id
            assert conversation_id is not None

            await app.submit_composer("Second question")
            assert app._run_task is not None
            await app._run_task
            await pilot.pause()

            transcript = app.screen.query_one(TranscriptView)
            rendered = transcript.copy_text()
            assert rendered.index("First question") < rendered.index("First answer")
            assert rendered.index("First answer") < rendered.index("Second question")
            assert rendered.index("Second question") < rendered.index("Second answer")

            await app._handle_command("/new")
            await pilot.pause()
            assert "First question" not in transcript.copy_text()
            assert "Second answer" not in transcript.copy_text()

            await app._handle_command(f"/resume {conversation_id}")
            await pilot.pause()
            resumed = transcript.copy_text()
            assert resumed.index("First question") < resumed.index("First answer")
            assert resumed.index("First answer") < resumed.index("Second question")
            assert resumed.index("Second question") < resumed.index("Second answer")
            app.exit(0)
    finally:
        await opened.close()


async def test_transcript_history_supports_review_and_latest_navigation():
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    async with app.run_test(size=(60, 14)) as pilot:
        await app.push_screen(ChatScreen())
        chat = app.chat()
        assert chat is not None
        chat.set_blocks(
            tuple(
                TranscriptBlock(
                    "assistant" if index % 2 else "user",
                    f"history-{index}",
                    f"Turn {index}\n" + "detail\n" * 3,
                )
                for index in range(12)
            )
        )
        transcript = chat.query_one(TranscriptView)
        await pilot.pause()
        transcript.follow_latest()
        await pilot.pause()
        bottom = transcript.scroll_y
        assert transcript.max_scroll_y > 0
        assert bottom == transcript.max_scroll_y

        await pilot.press("pageup")
        await pilot.pause()
        assert transcript.scroll_y < bottom
        assert transcript.following is False

        await pilot.press("ctrl+home")
        await pilot.pause()
        assert transcript.scroll_y == 0

        await pilot.press("ctrl+end")
        await pilot.pause()
        assert transcript.scroll_y == transcript.max_scroll_y
        assert transcript.following is True
        app.exit(0)


async def test_ctrl_o_toggles_tool_calls_without_exiting_chat():
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    async with app.run_test(size=(80, 24)) as pilot:
        await app.push_screen(ChatScreen())
        chat = app.screen
        assert isinstance(chat, ChatScreen)
        chat.set_blocks(
            (
                TranscriptBlock("user", "user-one", "Inspect the records"),
                TranscriptBlock(
                    "tool",
                    "tool-one",
                    tool_card=ToolCardState(
                        run_id="run-one",
                        call_id="call-one",
                        capability_id="data.query",
                        label="Query records",
                        state="done",
                        details=ToolCardDetails(summary="Returned one record"),
                    ),
                ),
                TranscriptBlock("assistant", "assistant-one", "Done"),
            )
        )
        composer = chat.query_one(Composer)
        composer.focus()
        cards = list(chat.query(ToolCard))
        assert len(cards) == 1
        assert cards[0].display is False

        await pilot.press("ctrl+o")
        await pilot.pause()
        assert app.screen is chat
        assert cards[0].display is True
        assert composer.has_focus is True
        title = cards[0].query_one(CollapsibleTitle)
        assert (title.styles.color.r, title.styles.color.g, title.styles.color.b) != (
            0,
            0,
            0,
        )
        title.focus()
        await pilot.press("enter")
        await pilot.pause()
        detail = cards[0].query_one("#tool-detail-call-one", Static)
        assert "Returned one record" in str(detail.content)
        assert (
            detail.styles.color.r,
            detail.styles.color.g,
            detail.styles.color.b,
        ) != (
            0,
            0,
            0,
        )

        await pilot.press("ctrl+o")
        await pilot.pause()
        assert app.screen is chat
        assert cards[0].display is False
        app.exit(0)


async def test_tui_credential_session_survives_internal_agent_reopen(tmp_path: Path):
    class _Keychain:
        def __init__(self) -> None:
            self.values: dict[str, str] = {}
            self.resolve_calls = 0

        async def resolve(self, reference):
            self.resolve_calls += 1
            return self.values[reference.name]

        async def set(self, reference, value):
            self.values[reference.name] = value

        async def delete(self, reference):
            self.values.pop(reference.name, None)

    keychain = _Keychain()
    reference = SecretReference.keychain("agent:postgresql:session-test")
    keychain.values[reference.name] = "session-secret"
    app = DaitaApp(
        root=tmp_path,
        keychain=keychain,
        start_bootstrap=False,
        workspace=workspace_for(tmp_path),
    )
    session = app.controller.keychain
    assert isinstance(session, CredentialSession)
    assert await session.resolve(reference) == "session-secret"
    assert keychain.resolve_calls == 1

    created = await app.controller.create_agent(
        "credential-session",
        observer=None,
        approval_handler=None,
    )
    assert created._embedded._keychain is session

    reopened = await app.controller.reopen_agent(
        observer=None,
        approval_handler=None,
    )
    assert reopened._embedded._keychain is session
    assert await session.resolve(reference) == "session-secret"
    assert keychain.resolve_calls == 1
    await app.controller.close()
    with pytest.raises(SecretResolutionError, match="credential session is closed"):
        await session.resolve(reference)


async def test_tui_preloads_active_credentials_before_accepting_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class _Keychain:
        def __init__(self) -> None:
            self.values: dict[str, str] = {}
            self.resolve_calls: list[SecretReference] = []

        async def resolve(self, reference):
            self.resolve_calls.append(reference)
            return self.values[reference.name]

        async def set(self, reference, value):
            self.values[reference.name] = value

        async def delete(self, reference):
            self.values.pop(reference.name, None)

    model_reference = SecretReference.keychain("agent:codex:active")
    database_reference = SecretReference.keychain("agent:postgresql:active")
    inactive_reference = SecretReference.keychain("agent:postgresql:inactive")
    keychain = _Keychain()
    keychain.values.update(
        {
            model_reference.name: "model-secret",
            database_reference.name: "database-secret",
            inactive_reference.name: "inactive-secret",
        }
    )

    class _OpenedAgent:
        model_route = SimpleNamespace(
            candidates=(SimpleNamespace(secret_reference=model_reference),)
        )

        async def list_sources(self):
            return (
                SimpleNamespace(
                    active=True,
                    configuration={"credential_ref": database_reference.to_uri()},
                ),
                SimpleNamespace(
                    active=False,
                    configuration={"credential_ref": inactive_reference.to_uri()},
                ),
            )

        async def close(self):
            return None

    opened = _OpenedAgent()

    async def open_agent(*args, **kwargs):
        return opened

    monkeypatch.setattr("daita.tui.controller.Agent.open", open_agent)
    app = DaitaApp(
        root=tmp_path,
        keychain=keychain,
        start_bootstrap=False,
        workspace=workspace_for(tmp_path),
    )

    assert (
        await app.controller.open_agent(
            "credential-preload",
            observer=None,
            approval_handler=None,
        )
        is opened
    )
    assert keychain.resolve_calls == [model_reference, database_reference]

    session = app.controller.keychain
    assert await session.resolve(model_reference) == "model-secret"
    assert await session.resolve(database_reference) == "database-secret"
    assert keychain.resolve_calls == [model_reference, database_reference]
    await app.controller.close()


async def test_one_interactive_run_path_is_unique():
    source = Path("src/daita/tui").read_text(encoding="utf-8") if False else None
    text = ""
    for path in Path("src/daita/tui").rglob("*.py"):
        text += path.read_text(encoding="utf-8")
    assert text.count("agent.run(") == 1


async def test_review_cost_modal_validates_before_dismissal():
    class Harness(App[str | None]):
        def compose(self) -> ComposeResult:
            yield Static("host")

        def on_mount(self) -> None:
            self.run_worker(self._present(), exclusive=True)

        async def _present(self) -> None:
            self.exit(await self.push_screen_wait(ReviewCostScreen()))

    app = Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        field = app.screen.query_one("#review-cost-value", Input)
        field.value = "not-money"
        screen_after = app.screen
        assert isinstance(screen_after, ReviewCostScreen)
        screen_after._submit()
        await pilot.pause()
        assert isinstance(app.screen, ReviewCostScreen)
        field.value = "0.15"
        screen = app.screen
        assert isinstance(screen, ReviewCostScreen)
        screen._submit()
        await pilot.pause()
    assert app.return_value == "0.15"


def test_external_editor_runs_only_inside_textual_suspend():
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    events: list[str] = []

    @contextmanager
    def suspended() -> Iterator[None]:
        events.append("suspend")
        try:
            yield
        finally:
            events.append("resume")

    app.suspend = suspended  # type: ignore[method-assign]

    def edit(seed: str) -> str:
        events.append(f"edit:{seed}")
        return seed + " changed"

    app.controller.edit_document = edit  # type: ignore[method-assign]
    assert app._edit_document("memory") == "memory changed"
    assert events == ["suspend", "edit:memory", "resume"]


async def test_skill_memory_and_candidate_editor_flows_use_public_controller(
    tmp_path: Path,
):
    opened = await Agent.create(
        "editors", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    app = DaitaApp(
        root=tmp_path, start_bootstrap=False, workspace=workspace_for(tmp_path)
    )
    app.controller.agent = opened
    try:
        app._edit_document = lambda seed: (  # type: ignore[method-assign]
            "# audit\n\nUse for audit work.\n\n"
            "## Instructions\n\nCheck every recorded total.\n"
        )
        await app._create_skill("audit")
        skill = await opened.read_skill("audit")
        assert skill is not None
        assert skill.description == "Use for audit work."

        app._edit_document = lambda seed: seed + "Remember fiscal calendars.\n"  # type: ignore[method-assign]
        await app._edit_memory_target("memory")
        assert "Remember fiscal calendars." in await opened.read_memory()

        documents: list[tuple[str, str]] = []

        async def candidate_document(candidate_id: str) -> str:
            assert candidate_id == "candidate-1"
            return '{"statement": "old"}\n'

        async def save_candidate(candidate_id: str, text: str) -> None:
            documents.append((candidate_id, text))

        app.controller.candidate_editor_document = candidate_document  # type: ignore[method-assign]
        app.controller.save_candidate_document = save_candidate  # type: ignore[method-assign]
        app._edit_document = lambda seed: '{"statement": "new"}\n'  # type: ignore[method-assign]
        await app._edit_candidate("candidate-1")
        assert documents == [("candidate-1", '{"statement": "new"}\n')]
    finally:
        await opened.close()


def _create_sqlite_source(path: Path, table: str) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(f'CREATE TABLE "{table}" (id INTEGER PRIMARY KEY)')
        connection.commit()
    finally:
        connection.close()


async def test_source_edit_screen_reviews_and_switches_atomically(tmp_path: Path):
    current_path = tmp_path / "current.sqlite"
    edited_path = tmp_path / "edited.sqlite"
    _create_sqlite_source(current_path, "records")
    _create_sqlite_source(edited_path, "records")
    opened = await Agent.create(
        "source-editor", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    await opened.attach(SQLiteSource(current_path, name="Warehouse"))
    app = DaitaApp(
        root=tmp_path, start_bootstrap=False, workspace=workspace_for(tmp_path)
    )
    app.controller.agent = opened
    try:
        async with app.run_test(size=(100, 34)) as pilot:
            await app._show_chat()
            await pilot.pause()
            command_task = asyncio.create_task(
                app._open_command_screen("source_edit", {})
            )
            await pilot.pause()
            assert isinstance(app.screen, SourceEditScreen)
            app.screen.query_one("#edit-source-path", Input).value = str(edited_path)
            apply_button = app.screen.query_one("#edit-source-apply", Button)
            apply_button.scroll_visible(animate=False)
            await pilot.pause()
            assert await pilot.click(apply_button, offset=(2, 1)) is True
            for _ in range(20):
                await pilot.pause(0.05)
                if isinstance(app.screen, ConfirmScreen):
                    break
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            await command_task
            active = await opened.active_source()
            assert active is not None
            assert active.configuration["path"] == str(edited_path)
            app.exit(0)
    finally:
        await opened.close()


async def test_postgresql_source_edit_probes_and_selects_schemas(monkeypatch):
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    source = SimpleNamespace(
        id="source-postgresql",
        adapter_id="postgresql",
        display_name="Warehouse",
        configuration={
            "host": "127.0.0.1",
            "port": 5432,
            "database": "fixture",
            "username": "reader",
            "schemas": ["public"],
            "ssl_mode": "disable",
            "credential_ref": "keychain:test-postgresql-edit",
        },
    )
    edited: dict[str, object] = {}

    async def active_source() -> object:
        return source

    async def probe_postgresql_source(*_args: object, **_kwargs: object) -> object:
        return SimpleNamespace(
            schemas=(
                SimpleNamespace(name="public", has_base_tables=False),
                SimpleNamespace(name="core", has_base_tables=True),
                SimpleNamespace(name="sales", has_base_tables=True),
            ),
            truncated=False,
        )

    async def edit_source_connection(*_args: object, **kwargs: object) -> object:
        edited.update(kwargs)
        return SimpleNamespace(source=source)

    monkeypatch.setattr(app.controller, "active_source", active_source)
    monkeypatch.setattr(
        app.controller, "probe_postgresql_source", probe_postgresql_source
    )
    monkeypatch.setattr(
        app.controller, "edit_source_connection", edit_source_connection
    )

    async with app.run_test(size=(100, 34)) as pilot:
        modal_task = asyncio.create_task(app._await_modal(SourceEditScreen()))
        await pilot.pause()
        edit = app.screen
        assert isinstance(edit, SourceEditScreen)
        apply_button = edit.query_one("#edit-source-apply", Button)
        apply_button.scroll_visible(animate=False)
        await pilot.pause()
        assert await pilot.click(apply_button, offset=(2, 1)) is True
        for _ in range(20):
            await pilot.pause(0.05)
            if isinstance(app.screen, SelectionScreen):
                break
        picker = app.screen
        assert isinstance(picker, SelectionScreen)
        assert picker._selected == {"core", "sales"}
        picker.action_confirm()

        for _ in range(20):
            await pilot.pause(0.05)
            if modal_task.done():
                break
        assert await modal_task is True
        assert edited["schemas"] == ("core", "sales")
        app.exit(0)


async def test_source_edit_rejects_a_zero_resource_preview_without_confirmation(
    monkeypatch,
):
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))

    async def active_source() -> None:
        return None

    async def list_sources() -> tuple[object, ...]:
        return ()

    monkeypatch.setattr(app.controller, "active_source", active_source)
    monkeypatch.setattr(app.controller, "list_sources", list_sources)

    async with app.run_test(size=(100, 34)) as pilot:
        await app.push_screen(SourceEditScreen())
        await pilot.pause()
        edit = app.screen
        assert isinstance(edit, SourceEditScreen)
        accepted = await edit._confirm_preview(SimpleNamespace(resource_count=0))
        assert accepted is False
        assert app.screen is edit
        assert "no catalogable tables" in str(
            edit.query_one("#source-edit-error").render()
        )
        app.exit(0)


async def test_catalog_command_opens_grouped_named_resource_tree(tmp_path: Path):
    first_path = tmp_path / "first.sqlite"
    second_path = tmp_path / "second.sqlite"
    _create_sqlite_source(first_path, "orders")
    _create_sqlite_source(second_path, "tickets")
    opened = await Agent.create(
        "catalog-browser", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    first = await opened.attach(SQLiteSource(first_path, name="Sales"))
    await opened.attach(SQLiteSource(second_path, name="Support"))
    await opened.select_source(first.id)
    app = DaitaApp(
        root=tmp_path, start_bootstrap=False, workspace=workspace_for(tmp_path)
    )
    app.controller.agent = opened
    try:
        async with app.run_test(size=(100, 34)) as pilot:
            await app._show_chat()
            await pilot.pause()
            composer = app.screen.query_one(Composer)
            composer.load_text("/catalog")
            composer.action_submit()
            await pilot.pause()

            assert isinstance(app.screen, CatalogScreen)
            summary = app.screen.query_one("#catalog-summary", Static)
            assert "2 sources  ·  2 resources  ·  0 relationships" in str(
                summary.content
            )
            tree = app.screen.query_one("#catalog-tree", Tree)
            browser = app.screen.query_one("#catalog-browser")
            assert browser.styles.border_left[0] == "solid"
            assert browser.styles.background.hex == "#111111"
            assert tree.styles.border_left[0] == ""
            assert tree.styles.border_top[0] == "solid"
            assert tree.styles.background.hex == "#111111"
            assert tree.get_component_styles("tree--cursor").background.hex == "#343434"
            assert app.screen.query_one("#catalog-title").styles.color.hex == "#FFFFFF"
            assert tree.has_focus is True
            assert tree.cursor_line == 0
            source_labels = tuple(str(node.label) for node in tree.root.children)
            assert source_labels[0] == "● Sales  SQLite · 1 resource  current"
            assert source_labels[1] == "Support  SQLite · 1 resource"
            resource_labels = tuple(
                str(node.label)
                for source_node in tree.root.children
                for node in source_node.children
            )
            assert resource_labels == ("main.orders  table", "main.tickets  table")
            assert "resource" not in resource_labels

            await pilot.press("down")
            assert tree.cursor_line == 1
            await pilot.press("up")
            assert tree.cursor_line == 0
            first_source = tree.root.children[0]
            assert first_source.is_expanded is True
            await pilot.press("enter")
            assert first_source.is_expanded is False
            await pilot.press("enter")
            assert first_source.is_expanded is True
            assert await pilot.click(tree, offset=(2, 1)) is True
            await pilot.pause()
            assert first_source.is_expanded is False
            assert await pilot.click(tree, offset=(6, 1)) is True
            await pilot.pause()
            assert first_source.is_expanded is True

            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, ChatScreen)

            composer = app.screen.query_one(Composer)
            composer.load_text("/catalog")
            composer.action_submit()
            await pilot.pause()
            assert isinstance(app.screen, CatalogScreen)
            assert await pilot.click("FooterKey") is True
            await pilot.pause()
            assert isinstance(app.screen, ChatScreen)

            composer = app.screen.query_one(Composer)
            composer.load_text(f"/source refresh {first.id}")
            composer.action_submit()
            await pilot.pause()
            assert isinstance(app.screen, CatalogScreen)
            refresh_notice = app.screen.query_one("#catalog-notice", Static)
            assert str(refresh_notice.content) == (
                "Catalog refresh succeeded · Sales · 1 resource"
            )
            assert refresh_notice.has_class("-warning") is False
            app.exit(0)
    finally:
        await opened.close()


async def test_empty_catalog_refresh_opens_catalog_without_an_onboarding_loop(
    tmp_path: Path,
):
    database = tmp_path / "empty.sqlite"
    sqlite3.connect(database).close()
    opened = await Agent.create(
        "empty-catalog-browser", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    source = await opened.attach(SQLiteSource(database, name="Empty source"))
    app = DaitaApp(
        root=tmp_path, start_bootstrap=False, workspace=workspace_for(tmp_path)
    )
    app.controller.agent = opened
    try:
        async with app.run_test(size=(100, 34)) as pilot:
            await app._show_chat()
            await pilot.pause()
            composer = app.screen.query_one(Composer)
            composer.load_text(f"/source refresh {source.id}")
            composer.action_submit()
            for _ in range(20):
                await pilot.pause(0.05)
                if isinstance(app.screen, CatalogScreen):
                    break

            assert isinstance(app.screen, CatalogScreen)
            refresh_notice = app.screen.query_one("#catalog-notice", Static)
            assert str(refresh_notice.content) == (
                "Catalog refresh completed, but found no resources · Empty source · "
                "use /source edit to review its schemas or path"
            )
            assert refresh_notice.has_class("-warning") is True
            tree = app.screen.query_one("#catalog-tree", Tree)
            source_node = tree.root.children[0]
            assert "0 resources" in str(source_node.label)
            assert str(source_node.children[0].label) == "No current resources"
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, ChatScreen)
            assert "Catalog refresh completed, but found no resources" in str(
                app.screen.query_one("#notice-bar", Static).content
            )
            app.exit(0)
    finally:
        await opened.close()


async def test_model_setup_uses_codex_device_login_without_api_key():
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    configured: dict[str, object] = {}
    verification: list[tuple[str, str]] = []
    authentication_started = asyncio.Event()
    authorization_release = asyncio.Event()

    app.controller.model_requires_explicit_limits = (  # type: ignore[method-assign]
        lambda **_kwargs: False
    )

    async def authenticate(**kwargs: object) -> str:
        on_verification = kwargs["on_verification"]
        on_progress = kwargs["on_progress"]
        assert callable(on_verification)
        assert callable(on_progress)
        prompt = SimpleNamespace(
            verification_url="https://auth.openai.com/codex/device",
            user_code="ABCD-EFGH",
        )
        on_verification(prompt)
        verification.append((prompt.verification_url, prompt.user_code))
        on_progress("Waiting for ChatGPT authorization")
        authentication_started.set()
        await authorization_release.wait()
        return "opaque-subscription-credential"

    async def configure(**kwargs: object) -> None:
        configured.update(kwargs)

    app.controller.authenticate_model_subscription = authenticate  # type: ignore[method-assign]
    app.controller.configure_model = configure  # type: ignore[method-assign]

    async with app.run_test(size=(90, 30)) as pilot:
        modal_task = asyncio.create_task(app._await_modal(ModelSetupScreen()))
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ModelSetupScreen)
        panel = screen.query_one("#onboard")
        assert panel.styles.border_left[0] == "solid"
        assert panel.styles.background.hex == "#111111"
        assert screen.query_one("#onboard-title").styles.color.hex == "#FFFFFF"
        assert screen.query_one("#model-help").styles.color.hex == "#FFFFFF99"
        model_id = screen.query_one("#model-id", Input)
        assert model_id.styles.border_left[0] == ""
        assert model_id.styles.border_bottom[0] == "solid"
        assert model_id.styles.background.hex == "#111111"
        choose_provider = screen.query_one("#choose-provider", Button)
        assert choose_provider.styles.border_left[0] == "solid"
        assert choose_provider.styles.background.hex in {"#181818", "#303030"}
        assert screen.query_one(Footer).styles.background.hex == "#111111"
        screen._provider = "codex"
        screen._model = "gpt-5.6-sol"
        screen.query_one("#model-id", Input).value = "gpt-5.6-sol"
        screen.query_one("#model-secret", Input).value = "must-not-be-used"
        assert await pilot.click("#save-model") is True
        await asyncio.wait_for(authentication_started.wait(), timeout=5)
        await asyncio.wait_for(pilot.pause(), timeout=5)
        auth_help = str(screen.query_one("#model-help", Static).content)
        assert "Waiting for ChatGPT authorization" in auth_help
        assert "https://auth.openai.com/codex/device" in auth_help
        assert "ABCD-EFGH" in auth_help
        authorization_release.set()
        assert await asyncio.wait_for(modal_task, timeout=5) is True
        app.exit(0)

    assert configured["provider"] == "codex"
    assert configured["model"] == "gpt-5.6-sol"
    assert configured["api_key"] is None
    assert configured["subscription_credential"] == "opaque-subscription-credential"
    assert verification == [("https://auth.openai.com/codex/device", "ABCD-EFGH")]


async def test_model_setup_provider_and_model_pickers_do_not_block_each_other():
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))

    async with app.run_test(size=(90, 30)) as pilot:
        await app.push_screen(ModelSetupScreen())
        await pilot.pause()
        setup = app.screen
        assert isinstance(setup, ModelSetupScreen)

        assert await pilot.click("#choose-provider") is True
        await asyncio.wait_for(pilot.pause(), timeout=5)
        provider_picker = app.screen
        assert isinstance(provider_picker, SelectionScreen)
        provider_options = provider_picker.query_one("#picker-options", OptionList)
        assert await pilot.click(provider_options, offset=(2, 0)) is True

        await asyncio.wait_for(pilot.pause(), timeout=5)
        model_picker = app.screen
        assert isinstance(model_picker, SelectionScreen)
        model_options = model_picker.query_one("#picker-options", OptionList)
        assert await pilot.click(model_options, offset=(2, 0)) is True

        await asyncio.wait_for(pilot.pause(), timeout=5)
        assert app.screen is setup
        assert isinstance(setup, ModelSetupScreen)
        assert setup._provider == "openai"
        assert setup._model == "gpt-5.6-sol"
        assert setup.query_one("#model-id", Input).value == "gpt-5.6-sol"
        app.exit(0)


async def test_source_setup_matches_the_muted_onboarding_treatment():
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))

    async with app.run_test(size=(100, 32)) as pilot:
        await app.push_screen(SourceSetupScreen())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SourceSetupScreen)

        panel = screen.query_one("#onboard")
        assert panel.styles.border_left[0] == "solid"
        assert panel.styles.background.hex == "#111111"
        assert screen.query_one("#onboard-title").styles.color.hex == "#FFFFFF"

        source_name = screen.query_one("#source-name", Input)
        assert source_name.styles.border_left[0] == ""
        assert source_name.styles.border_bottom[0] == "solid"
        assert source_name.styles.background.hex == "#111111"

        source_type = screen.query_one("#source-type", Select)
        select_current = source_type.query_one("SelectCurrent")
        assert select_current.styles.border_left[0] == ""
        assert select_current.styles.border_bottom[0] == "solid"
        assert select_current.styles.background.hex == "#111111"
        await pilot.click("#source-type")
        await pilot.pause()
        assert source_type.expanded is True
        select_overlay = source_type.query_one("SelectOverlay")
        assert select_overlay.styles.border_left[0] == "solid"
        assert select_overlay.styles.background.hex == "#111111"
        assert (
            select_overlay.get_component_styles(
                "option-list--option-highlighted"
            ).background.hex
            == "#343434"
        )

        attach = screen.query_one("#attach-source", Button)
        assert attach.styles.border_left[0] == "solid"
        assert attach.styles.background.hex in {"#181818", "#303030"}
        footer = screen.query_one(Footer)
        assert footer.styles.background.hex == "#111111"
        assert attach.region.bottom <= footer.region.y
        app.exit(0)


async def test_remaining_control_screens_share_the_muted_minimal_treatment():
    create_app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    async with create_app.run_test(size=(100, 32)) as pilot:
        await create_app.push_screen(AgentCreateScreen())
        await pilot.pause()
        panel = create_app.screen.query_one("#onboard")
        assert panel.styles.border_left[0] == "solid"
        assert panel.styles.background.hex == "#111111"
        assert panel.region.height < create_app.size.height
        name = create_app.screen.query_one("#agent-name", Input)
        assert name.styles.border_left[0] == ""
        assert name.styles.border_bottom[0] == "solid"
        create = create_app.screen.query_one("#create-agent", Button)
        assert create.styles.background.hex in {"#181818", "#303030"}
        create_app.exit(0)

    permissions_app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    async with permissions_app.run_test(size=(100, 32)) as pilot:
        await permissions_app.push_screen(PermissionsScreen())
        await pilot.pause()
        panel = permissions_app.screen.query_one("#permissions")
        assert panel.styles.border_left[0] == "solid"
        assert panel.styles.background.hex == "#111111"
        assert (
            permissions_app.screen.query_one("#perm-help").styles.color.hex
            == "#FFFFFF99"
        )
        apply = permissions_app.screen.query_one("#perm-apply", Button)
        assert apply.styles.background.hex in {"#181818", "#303030"}
        permissions_app.exit(0)

    confirm_app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    async with confirm_app.run_test(size=(100, 32)) as pilot:
        await confirm_app.push_screen(ConfirmScreen("Apply this change?"))
        await pilot.pause()
        panel = confirm_app.screen.query_one("#confirm")
        actions = confirm_app.screen.query_one("#confirm-actions")
        assert panel.styles.border_left[0] == "solid"
        assert panel.styles.background.hex == "#111111"
        assert panel.region.width <= 88
        assert panel.region.height <= 10
        assert actions.region.height == 3
        confirm_app.exit(0)


async def test_source_setup_accepts_a_successfully_attached_empty_catalog(monkeypatch):
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    attach_count = 0

    async def attach_sqlite(_path: Path, *, name: str | None) -> object:
        nonlocal attach_count
        attach_count += 1
        assert name == "Fixture"
        return SimpleNamespace(id=f"source-{attach_count}")

    monkeypatch.setattr(app.controller, "attach_sqlite", attach_sqlite)

    async with app.run_test(size=(100, 32)) as pilot:
        modal_task = asyncio.create_task(app._await_modal(SourceSetupScreen()))
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SourceSetupScreen)
        screen.query_one("#source-type", Select).value = "sqlite"
        screen.query_one("#source-name", Input).value = "Fixture"
        screen.query_one("#source-path", Input).value = "/fixture.sqlite"

        assert await pilot.click("#attach-source", offset=(2, 1)) is True
        for _ in range(20):
            await pilot.pause(0.05)
            if modal_task.done():
                break
        assert attach_count == 1
        assert modal_task.done() is True
        assert await modal_task is True
        app.exit(0)


async def test_postgresql_setup_probes_and_preselects_schemas_with_tables(monkeypatch):
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    credential = SecretReference.keychain("test-postgresql-probe")
    attached: dict[str, object] = {}
    deleted: list[SecretReference] = []

    async def store_password(_password: str) -> SecretReference:
        return credential

    async def probe_postgresql(**_kwargs: object) -> object:
        return SimpleNamespace(
            schemas=(
                SimpleNamespace(name="public", has_base_tables=False),
                SimpleNamespace(name="core", has_base_tables=True),
                SimpleNamespace(name="sales", has_base_tables=True),
            ),
            truncated=False,
        )

    async def attach_postgresql(**kwargs: object) -> object:
        attached.update(kwargs)
        return SimpleNamespace(id="source-postgresql")

    async def delete_password(reference: SecretReference) -> None:
        deleted.append(reference)

    monkeypatch.setattr(app.controller, "store_postgresql_password", store_password)
    monkeypatch.setattr(app.controller, "probe_postgresql", probe_postgresql)
    monkeypatch.setattr(app.controller, "attach_postgresql", attach_postgresql)
    monkeypatch.setattr(app.controller, "delete_postgresql_password", delete_password)

    async with app.run_test(size=(100, 32)) as pilot:
        modal_task = asyncio.create_task(app._await_modal(SourceSetupScreen()))
        await pilot.pause()
        setup = app.screen
        assert isinstance(setup, SourceSetupScreen)
        setup.query_one("#source-type", Select).value = "postgresql"
        setup.query_one("#pg-host", Input).value = "127.0.0.1"
        setup.query_one("#pg-database", Input).value = "fixture"
        setup.query_one("#pg-username", Input).value = "reader"
        setup.query_one("#pg-password", Input).value = "secret"
        setup.query_one("#pg-ssl", Select).value = "disable"

        assert await pilot.click("#attach-source", offset=(2, 1)) is True
        for _ in range(20):
            await pilot.pause(0.05)
            if isinstance(app.screen, SelectionScreen):
                break
        picker = app.screen
        assert isinstance(picker, SelectionScreen)
        assert picker._selected == {"core", "sales"}
        picker.action_confirm()

        for _ in range(20):
            await pilot.pause(0.05)
            if modal_task.done():
                break
        assert modal_task.done() is True
        assert await modal_task is True
        assert attached["schemas"] == ("core", "sales")
        assert deleted == []
        app.exit(0)


async def test_copy_uses_native_wrap_independent_text_selection(monkeypatch):
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    copied: list[str] = []

    async def deliver(text: str) -> ClipboardResult:
        copied.append(text)
        return ClipboardResult("success", "test", "Copied selection.")

    monkeypatch.setattr("daita.tui.app.deliver_clipboard", deliver)
    async with app.run_test(size=(80, 24)) as pilot:
        await app.push_screen(ChatScreen())
        chat = app.chat()
        assert chat is not None
        chat.append_block(TranscriptBlock("user", "selection-user", "hello world"))
        await pilot.pause()
        widget = chat.query_one(".transcript-user", Static)
        chat.selections = {widget: Selection.from_offsets(Offset(0, 0), Offset(5, 0))}
        await app.copy_or_cancel()
        assert copied == ["hello"]
        app.exit(0)


async def test_active_run_cancellation_does_not_retry_agent(tmp_path: Path):
    opened = await Agent.create(
        "cancel-once", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    calls: list[str] = []
    started = asyncio.Event()

    async def wait_forever(message: str, **_kwargs: object):
        calls.append(message)
        started.set()
        await asyncio.Event().wait()

    opened.run = wait_forever  # type: ignore[method-assign]
    app = DaitaApp(
        root=tmp_path, start_bootstrap=False, workspace=workspace_for(tmp_path)
    )
    app.controller.agent = opened
    try:
        async with app.run_test(size=(80, 24)) as pilot:
            await app._show_chat()
            await app.submit_composer("cancel this run")
            await asyncio.wait_for(started.wait(), timeout=1)
            await app.copy_or_cancel()
            await pilot.pause()
            assert calls == ["cancel this run"]
            assert app._run_task is not None and app._run_task.cancelled()
            chat = app.chat()
            assert chat is not None
            notice = chat.query_one("#notice-bar", Static)
            assert "cancelled" in str(notice.content).lower()
            app.exit(0)
    finally:
        await opened.close()


async def test_approval_presentation_failure_is_not_converted_to_denial():
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    request = ApprovalRequest(
        run_id="run-failure",
        call_id="call-failure",
        tool_name="data_update_postgresql",
        capability_id="data.postgresql.update",
        arguments=FrozenJsonObject.from_mapping({"name": "safe"}),
        reason="review failure",
    )

    async def fail_to_present(_request: ApprovalRequest) -> ApprovalDecision | None:
        raise RuntimeError("approval renderer failed")

    async with app.run_test(size=(80, 24)):
        await app.push_screen(ChatScreen())
        chat = app.screen
        assert isinstance(chat, ChatScreen)
        chat.request_approval = fail_to_present  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="approval renderer failed"):
            await app.handle_approval(request)
        app.exit(0)


@pytest.mark.skipif(os.name != "posix", reason="real PTY certification is POSIX-only")
def test_real_pty_normal_exit_restores_alternate_screen(tmp_path: Path):
    import fcntl
    import pty
    import select
    import struct
    import subprocess
    import sys
    import termios
    import time

    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=str(Path("src").resolve()),
        TERM="xterm-256color",
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "daita.cli",
            "--root",
            str(tmp_path),
        ],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=environment,
        close_fds=True,
        start_new_session=True,
    )
    os.close(slave)
    output = bytearray()
    escape_sent = False
    deadline = time.monotonic() + 10
    try:
        while time.monotonic() < deadline and process.poll() is None:
            readable, _, _ = select.select([master], [], [], 0.1)
            if readable:
                try:
                    chunk = os.read(master, 65_536)
                except OSError:
                    break
                if not chunk:
                    break
                output.extend(chunk)
            if not escape_sent and b"\x1b[?1049h" in output:
                os.write(master, b"\x1b")
                escape_sent = True
        return_code = process.wait(timeout=3)
        while True:
            readable, _, _ = select.select([master], [], [], 0)
            if not readable:
                break
            try:
                chunk = os.read(master, 65_536)
            except OSError:
                break
            if not chunk:
                break
            output.extend(chunk)
    finally:
        os.close(master)
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    assert escape_sent
    assert return_code == 0
    assert b"\x1b[?1049h" in output
    assert b"\x1b[?1049l" in output
