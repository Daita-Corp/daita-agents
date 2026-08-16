"""Pure and Pilot tests for the Textual interactive presentation."""

from __future__ import annotations

import asyncio
import os
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.geometry import Offset
from textual.selection import Selection
from textual.widgets import Button, Input, OptionList, Static

from daita import (
    Agent,
    AgentEvent,
    AgentEventKind,
    ApprovalDecision,
    ApprovalRequest,
    SQLiteSource,
)
from daita._json import FrozenJsonObject
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.tui.app import DaitaApp
from daita.tui.clipboard import (
    MAX_CLIPBOARD_UTF8_BYTES,
    ClipboardResult,
    clipboard_mechanism,
    deliver_clipboard,
    osc52_sequence,
)
from daita.tui.commands import (
    learning_invocation_message,
    parse_postgresql_connection_url,
    parse_source_override,
)
from daita.tui.models import (
    MAX_COMPOSER_CHARACTERS,
    MIN_READY_ROWS,
    MIN_USABLE_COLUMNS,
    TranscriptBlock,
)
from daita.tui.projection import (
    approval_review_document,
    project_tool_details,
    redact_presentation_value,
)
from daita.tui.sanitization import sanitize_terminal_text
from daita.tui.screens.approval import ApprovalScreen
from daita.tui.screens.chat import ChatScreen
from daita.tui.screens.confirm import ConfirmScreen
from daita.tui.screens.editing import ReviewCostScreen
from daita.tui.screens.onboarding import AgentCreateScreen, ModelSetupScreen
from daita.tui.screens.selection import SelectionScreen
from daita.tui.screens.source_edit import SourceEditScreen
from daita.tui.models import PickerOption
from daita.tui.observer import ObserverEvent
from daita.tui.widgets.composer import Composer, CompletionPopup
from daita.tui.widgets.status import (
    ActivityBar,
    context_window_text,
    format_token_count,
)
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
    app = DaitaApp(root=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        assert isinstance(app.screen, AgentCreateScreen)
        await pilot.press("escape")
        await pilot.pause()
    assert app.return_value == 0


async def test_app_resize_and_too_small_screen(tmp_path: Path):
    app = DaitaApp(root=tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.resize_terminal(MIN_USABLE_COLUMNS - 1, MIN_READY_ROWS - 1)
        await pilot.pause()
        assert app.size.width == MIN_USABLE_COLUMNS - 1
        app.exit(0)


def test_daita_theme_uses_the_official_terminal_palette():
    app = DaitaApp(start_bootstrap=False)
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
    boot = DaitaApp(start_bootstrap=False)
    async with boot.run_test(size=(80, 24)):
        welcome = boot.query_one("#boot", WelcomeView)
        assert "DAITA" in str(welcome.content)
        assert "Starting your workspace" in str(welcome.content)
        boot.exit(0)

    opened = await Agent.create("welcome-agent", root=tmp_path)
    app = DaitaApp(root=tmp_path, start_bootstrap=False)
    app.controller.agent = opened
    try:
        async with app.run_test(size=(90, 28)) as pilot:
            await app._show_chat()
            await pilot.pause()
            chat_welcome = app.screen.query_one("#welcome", WelcomeView)
            transcript = app.screen.query_one(TranscriptView)
            assert chat_welcome.display is True
            assert transcript.display is False
            assert "welcome-agent" in str(chat_welcome.content)
            assert "Type / for commands" in str(chat_welcome.content)

            chat = app.chat()
            assert chat is not None
            chat.append_block(TranscriptBlock("user", "first", "hello"))
            await pilot.pause()
            assert chat_welcome.display is False
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
    )
    app = DaitaApp(root=tmp_path, start_bootstrap=False)
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
    opened = await Agent.create("palette-agent", root=tmp_path)
    app = DaitaApp(root=tmp_path, start_bootstrap=False)
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
    opened = await Agent.create("completion-agent", root=tmp_path)
    await opened.attach(SQLiteSource(database, name="North [bold]"))
    await opened.save_skill(
        "audit",
        "Review [bold red]recorded totals[/]",
        "Check each recorded total against its source.",
    )
    app = DaitaApp(root=tmp_path, start_bootstrap=False)
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

    class Harness(App[ApprovalDecision | None]):
        def __init__(self, target: ApprovalRequest) -> None:
            super().__init__()
            self._target = target

        def compose(self) -> ComposeResult:
            yield Static("host")

        def on_mount(self) -> None:
            self.run_worker(self._present(), exclusive=True)

        async def _present(self) -> None:
            self.exit(await self.push_screen_wait(ApprovalScreen(self._target)))

    approved = Harness(request)
    async with approved.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("y")
        await pilot.pause()
    assert approved.return_value is ApprovalDecision.APPROVE

    denied = Harness(request)
    async with denied.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("n")
        await pilot.pause()
    assert denied.return_value is ApprovalDecision.DENY

    cancelled = Harness(request)
    async with cancelled.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
    assert cancelled.return_value is None

    hidden = Harness(secret)
    async with hidden.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert hidden.screen.query_one("#approval-unreviewable")
        screen = hidden.screen
        assert isinstance(screen, ApprovalScreen)
        screen.action_cancel()
        await pilot.pause()
    assert hidden.return_value is None

    hidden_oversize = Harness(oversized)
    async with hidden_oversize.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert hidden_oversize.screen.query_one("#approval-unreviewable")
        screen = hidden_oversize.screen
        assert isinstance(screen, ApprovalScreen)
        screen.action_cancel()
        await pilot.pause()
    assert hidden_oversize.return_value is None

    too_small = Harness(request)
    async with too_small.run_test(size=(40, 10)) as pilot:
        await pilot.pause()
    assert too_small.return_value is None


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

    opened = await Agent.create("ready", root=tmp_path)
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
    app = DaitaApp(root=tmp_path, start_bootstrap=False)
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
    app = DaitaApp(start_bootstrap=False)
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
    opened = await Agent.create("editors", root=tmp_path)
    app = DaitaApp(root=tmp_path, start_bootstrap=False)
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
    opened = await Agent.create("source-editor", root=tmp_path)
    await opened.attach(SQLiteSource(current_path, name="Warehouse"))
    app = DaitaApp(root=tmp_path, start_bootstrap=False)
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
            apply_task = asyncio.create_task(app.screen._apply())
            for _ in range(20):
                await pilot.pause(0.05)
                if isinstance(app.screen, ConfirmScreen):
                    break
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            await apply_task
            await command_task
            active = await opened.active_source()
            assert active is not None
            assert active.configuration["path"] == str(edited_path)
            app.exit(0)
    finally:
        await opened.close()


async def test_model_setup_uses_codex_device_login_without_api_key():
    app = DaitaApp(start_bootstrap=False)
    configured: dict[str, object] = {}
    verification: list[tuple[str, str]] = []

    app.controller.model_requires_explicit_limits = (  # type: ignore[method-assign]
        lambda **_kwargs: False
    )

    async def authenticate(**kwargs: object) -> str:
        on_verification = kwargs["on_verification"]
        assert callable(on_verification)
        prompt = SimpleNamespace(
            verification_url="https://auth.openai.com/codex/device",
            user_code="ABCD-EFGH",
        )
        on_verification(prompt)
        verification.append((prompt.verification_url, prompt.user_code))
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
        screen._provider = "codex"
        screen._model = "gpt-5.6-sol"
        screen.query_one("#model-id", Input).value = "gpt-5.6-sol"
        screen.query_one("#model-secret", Input).value = "must-not-be-used"
        await screen.on_button_pressed(
            Button.Pressed(screen.query_one("#save-model", Button))
        )
        assert await modal_task is True
        app.exit(0)

    assert configured["provider"] == "codex"
    assert configured["model"] == "gpt-5.6-sol"
    assert configured["api_key"] is None
    assert configured["subscription_credential"] == "opaque-subscription-credential"
    assert verification == [("https://auth.openai.com/codex/device", "ABCD-EFGH")]


async def test_copy_uses_native_wrap_independent_text_selection(monkeypatch):
    app = DaitaApp(start_bootstrap=False)
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
    opened = await Agent.create("cancel-once", root=tmp_path)
    calls: list[str] = []
    started = asyncio.Event()

    async def wait_forever(message: str, **_kwargs: object):
        calls.append(message)
        started.set()
        await asyncio.Event().wait()

    opened.run = wait_forever  # type: ignore[method-assign]
    app = DaitaApp(root=tmp_path, start_bootstrap=False)
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
    app = DaitaApp(start_bootstrap=False)
    request = ApprovalRequest(
        run_id="run-failure",
        call_id="call-failure",
        tool_name="data_update_postgresql",
        capability_id="data.postgresql.update",
        arguments=FrozenJsonObject.from_mapping({"name": "safe"}),
        reason="review failure",
    )

    async def fail_to_present(_screen: object) -> object:
        raise RuntimeError("approval renderer failed")

    app._await_modal = fail_to_present  # type: ignore[assignment]
    async with app.run_test(size=(80, 24)):
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
