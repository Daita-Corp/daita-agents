"""Component-owned tests split from ``test_screens.py``."""

from __future__ import annotations

from daita import AgentConfig, MCPToolSelection
from daita.adapters.mcp import SDKMCPClientFactory
from daita.llm.models import FinishReason, ModelResponse, ToolCall
from daita.llm.providers.mock import MockModelProvider
from tests.support.mcp import MemoryKeychain, conformance_identities, mock_transport
from tests.tui._support import (
    MIN_READY_ROWS,
    MIN_USABLE_COLUMNS,
    Agent,
    AgentCreateScreen,
    Button,
    ChatScreen,
    Composer,
    ConfirmScreen,
    CredentialSession,
    DaitaApp,
    Footer,
    Input,
    ModelSetupScreen,
    OptionList,
    Path,
    PermissionsScreen,
    SecretReference,
    SecretResolutionError,
    Select,
    SelectionScreen,
    SimpleNamespace,
    SourceSetupScreen,
    Static,
    TranscriptBlock,
    TranscriptView,
    WelcomeView,
    asyncio,
    parse_postgresql_connection_url,
    pytest,
    redact_presentation_value,
    workspace_for,
)


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
        await pilot.pause()
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


async def test_agent_picker_enters_delete_mode_then_confirms_typed_name(
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
        await pilot.pause()
        buttons = tuple(picker.query_one("#picker-actions").query(Button))
        assert [str(button.label) for button in buttons] == [
            "Create new agent",
            "Delete agent",
        ]
        assert len({button.region.y for button in buttons}) == 1

        assert await pilot.click("#picker-tertiary") is True
        for _ in range(20):
            await pilot.pause(0.05)
            if isinstance(app.screen, SelectionScreen) and app.screen is not picker:
                break
        delete_picker = app.screen
        assert isinstance(delete_picker, SelectionScreen)
        assert "Select an agent to delete" in str(
            delete_picker.query_one("#picker-title").render()
        )
        listing = delete_picker.query_one("#picker-options", OptionList)
        assert await pilot.click(listing, offset=(2, 1)) is True

        for _ in range(20):
            await pilot.pause(0.05)
            if isinstance(app.screen, ConfirmScreen):
                break
        confirmation = app.screen
        assert isinstance(confirmation, ConfirmScreen)
        assert "existing-two" in str(
            confirmation.query_one("#confirm-message").render()
        )
        assert "local files are not modified" in str(
            confirmation.query_one("#confirm-message").render()
        )
        confirmation_input = confirmation.query_one("#confirm-input", Input)
        confirmation_input.value = "not-an-agent"
        confirmation_input.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen is confirmation
        assert "did not match" in str(
            confirmation.query_one("#confirm-message").render()
        )
        assert await Agent.list(root=tmp_path) == ("existing-one", "existing-two")

        confirmation_input.value = "existing-two"
        await pilot.press("enter")

        for _ in range(40):
            await pilot.pause(0.05)
            if isinstance(app.screen, SelectionScreen):
                break
        assert isinstance(app.screen, SelectionScreen)
        assert await Agent.list(root=tmp_path) == ("existing-one",)

        assert await pilot.click("#picker-tertiary") is True
        for _ in range(20):
            await pilot.pause(0.05)
            if (
                isinstance(app.screen, SelectionScreen)
                and app.screen is not picker
                and "to delete" in str(app.screen.query_one("#picker-title").render())
            ):
                break
        final_delete_picker = app.screen
        assert isinstance(final_delete_picker, SelectionScreen)
        listing = final_delete_picker.query_one("#picker-options", OptionList)
        assert await pilot.click(listing, offset=(2, 0)) is True

        for _ in range(20):
            await pilot.pause(0.05)
            if isinstance(app.screen, ConfirmScreen):
                break
        delete_confirmation = app.screen
        assert isinstance(delete_confirmation, ConfirmScreen)
        confirmation_input = delete_confirmation.query_one("#confirm-input", Input)
        confirmation_input.value = "existing-one"
        confirmation_input.focus()
        await pilot.press("enter")

        for _ in range(40):
            await pilot.pause(0.05)
            if isinstance(app.screen, AgentCreateScreen):
                break
        assert isinstance(app.screen, AgentCreateScreen)
        assert await Agent.list(root=tmp_path) == ()
        await pilot.press("escape")
        await pilot.pause()

    assert app.return_value == 0


@pytest.mark.parametrize("failure", ("malformed_config", "mcp_schema"))
async def test_agent_picker_recovers_incompatible_model_settings_for_replacement(
    tmp_path: Path,
    failure,
):
    keychain = MemoryKeychain()
    legacy = await Agent.create(
        "legacy-model",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        keychain=keychain,
        model_validator=MockModelProvider(
            (
                ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(
                        ToolCall("validation", "daita_validate_tool_support", {}),
                    ),
                ),
            ),
            provider_id="gemini:fixture",
        ),
    )
    if failure == "mcp_schema":
        await legacy.configure_model(
            provider="gemini",
            model="fixture",
            api_key="fixture-key",
            context_window_tokens=8192,
            max_output_tokens=1024,
        )
    await legacy.close()
    other = await Agent.create(
        "other-agent", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    await other.close()
    config_path = tmp_path / "agents" / "legacy-model" / "config.json"
    if failure == "malformed_config":
        config_path.write_text("{}", encoding="utf-8")
    else:
        identity, _ = conformance_identities()
        identity.tool("lookup")["inputSchema"] = {
            "type": "object",
            "properties": {"query": {"type": ["string", "null"]}},
        }
        # Reproduce an existing home admitted before model compatibility checks.
        legacy = await Agent.open(
            "legacy-model",
            root=tmp_path,
            workspace=workspace_for(tmp_path),
            config=AgentConfig(),
            keychain=keychain,
            mcp_client_factory=SDKMCPClientFactory(
                http_transport=mock_transport(identity)
            ),
        )
        try:
            await legacy.attach_mcp_server(
                endpoint=identity.endpoint,
                selections=(MCPToolSelection("lookup"),),
            )
        finally:
            await legacy.close()
    before = config_path.read_bytes()

    app = DaitaApp(root=tmp_path, workspace=workspace_for(tmp_path), keychain=keychain)
    async with app.run_test(size=(90, 28)) as pilot:
        # Screen selection precedes its mount handler populating the options.
        # Wait for the actual picker contents, not only the active screen type.
        for _ in range(20):
            await pilot.pause(0.05)
            if (
                isinstance(app.screen, SelectionScreen)
                and app.screen.is_mounted
                and app.screen.query_one("#picker-options", OptionList).option_count
                == 2
            ):
                break
        picker = app.screen
        assert isinstance(picker, SelectionScreen)
        listing = picker.query_one("#picker-options", OptionList)
        assert {
            str(listing.get_option_at_index(index).id)
            for index in range(listing.option_count)
        } == {"legacy-model", "other-agent"}
        listing.highlighted = next(
            index
            for index in range(listing.option_count)
            if str(listing.get_option_at_index(index).id) == "legacy-model"
        )
        picker.action_confirm()

        for _ in range(40):
            await pilot.pause(0.05)
            if isinstance(app.screen, ModelSetupScreen):
                break
        setup = app.screen
        assert app._startup_error is None
        assert isinstance(setup, ModelSetupScreen)
        assert "no longer compatible" in str(
            setup.query_one("#model-help", Static).content
        )
        assert app.controller.require_agent().name == "legacy-model"
        assert app.controller.require_agent().model_profile is None
        assert config_path.read_bytes() == before

        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, ChatScreen)
        assert (
            await app.controller.dispatch_command("/mcp")
        ).screen == "mcp_management"
        if failure == "mcp_schema":
            statuses = await app.controller.list_mcp_servers()
            assert len(statuses) == 1
            assert statuses[0].active_in_runtime
        app.exit(0)

    assert config_path.read_bytes() == before


async def test_chat_preserves_input_while_agent_reopens(tmp_path: Path, monkeypatch):
    opened = await Agent.create(
        "reopening", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    app = DaitaApp(
        root=tmp_path, start_bootstrap=False, workspace=workspace_for(tmp_path)
    )
    app.controller.agent = opened
    entered, release = asyncio.Event(), asyncio.Event()
    original_open = Agent.open

    async def gated_open(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original_open(*args, **kwargs)

    monkeypatch.setattr(Agent, "open", gated_open)
    try:
        async with app.run_test(size=(90, 28)) as pilot:
            await app._show_chat()
            reopening = asyncio.create_task(
                app.controller.reopen_agent(observer=None, approval_handler=None)
            )
            try:
                await asyncio.wait_for(entered.wait(), 5)
                composer = app.screen.query_one(Composer)
                composer.load_text("/mcp")
                await app.submit_composer(composer.text)
                assert composer.text == "/mcp"
                assert app._exception is None
            finally:
                release.set()
                await asyncio.wait_for(reopening, 5)
            assert app.controller.require_agent().name == "reopening"
            app.exit(0)
    finally:
        release.set()
        await app.controller.close()


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
            assert "Files:" not in notice
            assert "Sources:" not in notice
            assert "no source · use /source add" not in notice
            assert app.screen.query_one(Composer).disabled is False
            app.exit(0)
    finally:
        await opened.close()


@pytest.mark.integration
@pytest.mark.parametrize("no_color", [False, True])
async def test_terminal_defaults_preserve_markdown_and_focus_without_rgb_paint(
    monkeypatch, no_color: bool
):
    from io import StringIO

    from rich.color import ColorType
    from rich.console import Console
    from textual.widgets import Markdown, Tabs

    if no_color:
        monkeypatch.setenv("NO_COLOR", "1")
    else:
        monkeypatch.delenv("NO_COLOR", raising=False)
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    async with app.run_test(size=(100, 32)) as pilot:
        try:
            await app.push_screen(ChatScreen())
            chat = app.screen
            assert isinstance(chat, ChatScreen)
            chat.set_blocks(
                (
                    TranscriptBlock(
                        "assistant",
                        "native-markdown",
                        "# Report\n\n**Revenue** and `count`\n\n"
                        "- First item\n- Second item\n\n"
                        "| Region | Total |\n| --- | --- |\n| West | 14 |\n\n"
                        "```python\ncount = 14\nprint(count)\n```",
                    ),
                )
            )
            await pilot.pause()
            assert app.native_ansi_color
            assert app.no_color is no_color
            assert chat.styles.color.ansi == -1
            assert chat.styles.background.ansi == -1
            markdown = chat.query_one(Markdown)
            assert markdown.query("MarkdownH1")
            assert markdown.query("MarkdownBulletList")
            assert markdown.query("MarkdownTable")
            assert markdown.query("MarkdownFence")
            composer = chat.query_one(Composer)
            assert composer.get_component_styles("text-area--cursor").text_style.reverse
            composer.load_text("draft")
            await pilot.press("f6")
            assert chat.query_one(Tabs).has_focus
            selected = chat.query_one("Tab.-active")
            assert selected.styles.text_style.bold
            assert selected.styles.text_style.underline
            assert selected.styles.text_style.reverse
            assert composer.text == "draft"
            # Render the actual screen to ANSI, independent of screenshot palettes.
            output = StringIO()
            console = Console(
                file=output,
                force_terminal=True,
                color_system="truecolor",
                no_color=False,
            )
            frame = chat._compositor.render_update(full=True, simplify=True)
            assert frame is not None
            segments = list(console.render(frame))
            for segment in segments:
                if segment.style:
                    for color in (segment.style.color, segment.style.bgcolor):
                        if color:
                            assert color.type in (ColorType.DEFAULT, ColorType.STANDARD)
                            if no_color:
                                assert color.type == ColorType.DEFAULT
            console.print(frame)
            ansi = output.getvalue()
            assert "38;2;" not in ansi
            assert "48;2;" not in ansi
            assert "Revenue" in ansi
            assert "West" in ansi
            assert "print" in ansi
            assert any(segment.style and segment.style.reverse for segment in segments)
        finally:
            app.exit(0)


async def test_wide_boot_swirls_dots_into_the_braille_logo():
    boot = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    async with boot.run_test(size=(140, 28)) as pilot:
        welcome = boot.query_one("#boot", WelcomeView)
        await pilot.pause()
        assert "⣿⣿⣿⣿⣿⣿⡿⠛⠛⠛⠛⠛⠛⢿⣿⣿⣿⣿⣿⣿" not in str(welcome.content)
        await asyncio.wait_for(welcome.intro_finished(), timeout=5)
        await pilot.pause()
        assert "⣿⣿⣿⣿⣿⣿⡿⠛⠛⠛⠛⠛⠛⢿⣿⣿⣿⣿⣿⣿" in str(welcome.content)
        assert "Starting your workspace" in str(welcome.content)
        boot.exit(0)


async def test_boot_and_empty_chat_show_the_responsive_daita_welcome(tmp_path: Path):
    boot = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    async with boot.run_test(size=(80, 24)):
        welcome = boot.query_one("#boot", WelcomeView)
        from daita import __version__

        assert f"DAITA  {__version__}" in str(welcome.content)
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
        async with app.run_test(size=(140, 28)) as pilot:
            await app._show_chat()
            await pilot.pause()
            chat_welcome = app.screen.query_one("#welcome", Static)
            welcome_region = app.screen.query_one("#welcome-region")
            transcript = app.screen.query_one(TranscriptView)
            composer = app.screen.query_one(Composer)
            footer = app.screen.query_one(Footer)
            assert chat_welcome.display is True
            assert welcome_region.display is True
            assert transcript.display is False
            assert footer.region.y == composer.region.y + composer.region.height + 1
            assert "What would you like to know?" in str(chat_welcome.content)
            assert "Ask about your data or local files." in str(chat_welcome.content)
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


@pytest.mark.parametrize(
    "login_mode", ("first_login", "reuse", "reconnect", "reuse_failure")
)
async def test_model_setup_uses_codex_device_login_without_api_key(login_mode):
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    configured: dict[str, object] = {}
    verification: list[tuple[str, str]] = []
    authentication_started = asyncio.Event()
    authorization_release = asyncio.Event()
    if login_mode != "first_login":

        class _OpenedAgent:
            model_route = SimpleNamespace(
                candidates=(
                    SimpleNamespace(
                        provider_id="codex:gpt-5.6-sol",
                        secret_reference=SecretReference.keychain("saved-subscription"),
                    ),
                )
            )

            async def close(self):
                return None

        app.controller.agent = _OpenedAgent()  # type: ignore[assignment]

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
        if (
            login_mode == "reuse_failure"
            and kwargs.get("subscription_credential") is None
        ):
            raise ValueError("The saved subscription login was rejected.")
        configured.update(kwargs)

    app.controller.authenticate_model_subscription = authenticate  # type: ignore[method-assign]
    app.controller.configure_model = configure  # type: ignore[method-assign]

    async with app.run_test(size=(90, 30)) as pilot:
        try:
            modal_task = asyncio.create_task(app._await_modal(ModelSetupScreen()))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ModelSetupScreen)
            panel = screen.query_one("#onboard")
            assert panel.styles.border_left[0] == "solid"
            assert panel.styles.background.ansi == -1
            assert screen.query_one("#onboard-title").styles.color.ansi == -1
            assert screen.query_one("#model-help").styles.color.ansi == -1
            model_id = screen.query_one("#model-id", Input)
            assert model_id.display is False
            assert screen.query_one("#model-provider-id", Input).display is False
            assert screen.query_one("#model-secret", Input).display is False
            assert screen.query_one("#model-base-url", Input).display is False
            assert screen.query_one("#model-context", Input).display is False
            assert screen.query_one("#model-output", Input).display is False
            assert screen.query_one("#save-model", Button).display is False
            assert model_id.styles.border_left[0] == ""
            assert model_id.styles.border_bottom[0] == "solid"
            assert model_id.styles.background.ansi == -1
            choose_provider = screen.query_one("#choose-provider", Button)
            assert choose_provider.styles.border_left[0] == "solid"
            assert choose_provider.styles.background.ansi == -1
            assert screen.query_one(Footer).styles.background.ansi == -1
            screen._apply_provider_selection("codex", "gpt-5.6-sol")
            if login_mode != "first_login":
                screen.query_one("#model-id", Input).value = "gpt-5.6-terra"
            await pilot.pause()
            assert screen.query_one("#model-secret", Input).display is False
            assert screen.query_one("#save-model", Button).display is True
            screen.query_one("#model-secret", Input).value = "must-not-be-used"
            screen.query_one("#model-base-url", Input).value = (
                "https://must-not-be-used"
            )
            screen.query_one("#model-context", Input).value = "1"
            screen.query_one("#model-output", Input).value = "1"
            assert screen.query_one("#model-reconnect", Button).display == (
                login_mode != "first_login"
            )
            assert await pilot.click(
                "#model-reconnect" if login_mode == "reconnect" else "#save-model"
            )
            if login_mode == "reuse_failure":
                workers = tuple(
                    worker for worker in app.workers if worker.node is screen
                )
                if workers:
                    await asyncio.wait_for(
                        app.workers.wait_for_complete(workers), timeout=5
                    )
                await pilot.pause()
                assert app.screen is screen
                assert not configured
                assert not verification
                assert not screen.query_one("#model-reconnect", Button).disabled
                assert await pilot.click("#model-reconnect")
            if login_mode != "reuse":
                await asyncio.wait_for(authentication_started.wait(), timeout=5)
                await asyncio.wait_for(pilot.pause(), timeout=5)
                auth_help = str(screen.query_one("#model-help", Static).content)
                assert "Waiting for ChatGPT authorization" in auth_help
                assert "https://auth.openai.com/codex/device" in auth_help
                assert "ABCD-EFGH" in auth_help
                authorization_release.set()
            assert await asyncio.wait_for(modal_task, timeout=5) is True
            app.exit(0)

        finally:
            app.exit(0)
    assert configured["provider"] == "codex"
    assert configured["model"] == (
        "gpt-5.6-sol" if login_mode == "first_login" else "gpt-5.6-terra"
    )
    assert configured["api_key"] is None
    assert configured["subscription_credential"] == (
        None if login_mode == "reuse" else "opaque-subscription-credential"
    )
    assert configured["base_url"] is None
    assert configured["context_window_tokens"] is None
    assert configured["max_output_tokens"] is None
    assert verification == (
        []
        if login_mode == "reuse"
        else [("https://auth.openai.com/codex/device", "ABCD-EFGH")]
    )


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
        assert setup._model == "gpt-6-astra"
        assert setup.query_one("#model-id", Input).value == "gpt-6-astra"
        assert "OpenAI API" in str(setup.query_one("#choose-provider", Button).label)
        assert setup.query_one("#model-provider-id", Input).display is False
        assert setup.query_one("#model-secret", Input).display is True
        assert setup.query_one("#model-base-url", Input).display is False
        assert setup.query_one("#model-context", Input).display is False
        assert setup.query_one("#model-output", Input).display is False
        assert setup.query_one("#save-model", Button).display is True
        app.exit(0)


async def test_model_setup_reveals_only_custom_provider_fields_when_selected():
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))

    async with app.run_test(size=(90, 30)) as pilot:
        await app.push_screen(ModelSetupScreen())
        await pilot.pause()
        setup = app.screen
        assert isinstance(setup, ModelSetupScreen)

        setup._apply_provider_selection("custom", None)
        await pilot.pause()
        assert setup.query_one("#model-provider-id", Input).display is True
        assert setup.query_one("#model-id", Input).display is True
        assert setup.query_one("#model-secret", Input).display is True
        assert setup.query_one("#model-base-url", Input).display is True
        assert setup.query_one("#model-context", Input).display is False
        assert setup.query_one("#model-output", Input).display is False

        setup.query_one("#model-id", Input).value = "private-model"
        await pilot.pause()
        assert setup.query_one("#model-context", Input).display is True
        assert setup.query_one("#model-output", Input).display is True
        app.exit(0)


async def test_source_setup_uses_terminal_defaults_and_visible_selection():
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))

    async with app.run_test(size=(100, 32)) as pilot:
        try:
            await app.push_screen(SourceSetupScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SourceSetupScreen)

            panel = screen.query_one("#onboard")
            assert panel.styles.border_left[0] == "solid"
            assert panel.styles.background.ansi == -1
            assert screen.query_one("#onboard-title").styles.color.ansi == -1

            source_name = screen.query_one("#source-name", Input)
            assert source_name.styles.border_left[0] == ""
            assert source_name.styles.border_bottom[0] == "solid"
            assert source_name.styles.background.ansi == -1

            source_type = screen.query_one("#source-type", Select)
            select_current = source_type.query_one("SelectCurrent")
            assert select_current.styles.border_left[0] == ""
            assert source_type.has_focus
            assert select_current.styles.border_bottom[0] == "heavy"
            assert select_current.styles.background.ansi == -1
            await pilot.click("#source-type")
            await pilot.pause()
            assert source_type.expanded is True
            select_overlay = source_type.query_one("SelectOverlay")
            assert select_overlay.styles.border_left[0] == "solid"
            assert select_overlay.styles.background.ansi == -1
            assert (
                select_overlay.get_component_styles(
                    "option-list--option-highlighted"
                ).text_style.reverse
                is True
            )

            attach = screen.query_one("#attach-source", Button)
            assert attach.styles.border_left[0] == "solid"
            assert attach.styles.background.ansi == -1
            footer = screen.query_one(Footer)
            assert footer.styles.background.ansi == -1
            assert attach.region.bottom <= footer.region.y
            app.exit(0)

        finally:
            app.exit(0)


async def test_control_screens_use_terminal_defaults_and_retain_dialog_geometry():
    create_app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    async with create_app.run_test(size=(100, 32)) as pilot:
        try:
            await create_app.push_screen(AgentCreateScreen())
            await pilot.pause()
            panel = create_app.screen.query_one("#onboard")
            assert panel.styles.border_left[0] == "solid"
            assert panel.styles.background.ansi == -1
            assert panel.region.height < create_app.size.height
            name = create_app.screen.query_one("#agent-name", Input)
            assert name.styles.border_left[0] == ""
            assert name.has_focus
            assert name.styles.border_bottom[0] == "heavy"
            create = create_app.screen.query_one("#create-agent", Button)
            assert create.styles.background.ansi == -1
            create_app.exit(0)

        finally:
            create_app.exit(0)
    permissions_app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    async with permissions_app.run_test(size=(100, 32)) as pilot:
        try:
            await permissions_app.push_screen(PermissionsScreen())
            await pilot.pause()
            panel = permissions_app.screen.query_one("#permissions")
            assert panel.styles.border_left[0] == "solid"
            assert panel.styles.background.ansi == -1
            assert (
                permissions_app.screen.query_one("#perm-help").styles.color.ansi == -1
            )
            apply = permissions_app.screen.query_one("#perm-apply", Button)
            assert apply.styles.background.ansi == -1
            permissions_app.exit(0)

        finally:
            permissions_app.exit(0)
    confirm_app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    async with confirm_app.run_test(size=(100, 32)) as pilot:
        try:
            await confirm_app.push_screen(ConfirmScreen("Apply this change?"))
            await pilot.pause()
            panel = confirm_app.screen.query_one("#confirm")
            actions = confirm_app.screen.query_one("#confirm-actions")
            assert panel.styles.border_left[0] == "solid"
            assert panel.styles.background.ansi == -1
            assert panel.region.width <= 88
            assert panel.region.height <= 10
            assert actions.region.height == 3
            confirm_app.exit(0)

        finally:
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


async def test_source_setup_starts_with_unfilled_labeled_fields():
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    async with app.run_test(size=(100, 32)) as pilot:
        await app.push_screen(SourceSetupScreen())
        await pilot.pause()
        setup = app.screen
        assert isinstance(setup, SourceSetupScreen)

        field_ids = (
            "source-name",
            "source-path",
            "pg-host",
            "pg-port",
            "pg-database",
            "pg-username",
            "pg-password",
            "pg-schemas",
        )
        assert all(
            setup.query_one(f"#{field_id}", Input).value == "" for field_id in field_ids
        )
        assert setup.query_one("#source-type", Select).value is Select.NULL
        assert setup.query_one("#pg-ssl", Select).value is Select.NULL
        assert setup.query_one("#pg-port", Input).placeholder == "Port"
        assert setup.query_one("#pg-schemas", Input).placeholder == (
            "Schemas (comma-separated)"
        )
        app.exit(0)


async def test_postgresql_url_honors_an_explicit_ssl_selection(monkeypatch):
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    credential = SecretReference.keychain("test-postgresql-url-ssl")
    probed: dict[str, object] = {}

    async def store_password(_password: str) -> SecretReference:
        return credential

    async def probe_postgresql(**kwargs: object) -> object:
        probed.update(kwargs)
        return SimpleNamespace(
            schemas=(SimpleNamespace(name="analytics", has_base_tables=True),),
            truncated=False,
        )

    async def attach_postgresql(**_kwargs: object) -> object:
        return SimpleNamespace(id="source-postgresql")

    monkeypatch.setattr(app.controller, "store_postgresql_password", store_password)
    monkeypatch.setattr(app.controller, "probe_postgresql", probe_postgresql)
    monkeypatch.setattr(app.controller, "attach_postgresql", attach_postgresql)

    async with app.run_test(size=(100, 32)) as pilot:
        modal_task = asyncio.create_task(app._await_modal(SourceSetupScreen()))
        await pilot.pause()
        setup = app.screen
        assert isinstance(setup, SourceSetupScreen)
        setup.query_one("#source-type", Select).value = "postgresql"
        setup.query_one("#source-path", Input).value = (
            "postgresql://postgres:fixture_admin_password@127.0.0.1:55432/"
            "daita_fixture"
        )
        setup.query_one("#pg-schemas", Input).value = "analytics"
        setup.query_one("#pg-ssl", Select).value = "disable"

        assert await pilot.click("#attach-source", offset=(2, 1)) is True
        for _ in range(20):
            await pilot.pause(0.05)
            if isinstance(app.screen, SelectionScreen):
                break
        picker = app.screen
        assert isinstance(picker, SelectionScreen)
        picker.action_confirm()
        for _ in range(20):
            await pilot.pause(0.05)
            if modal_task.done():
                break

        assert await modal_task is True
        assert probed["host"] == "127.0.0.1"
        assert probed["port"] == 55432
        assert probed["database"] == "daita_fixture"
        assert probed["username"] == "postgres"
        assert probed["ssl_mode"] == "disable"
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
