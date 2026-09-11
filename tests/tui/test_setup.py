"""Component-owned tests split from ``test_screens.py``."""

from __future__ import annotations

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
            assert "Sources: none connected (sources are optional)" in notice
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


async def test_boot_and_empty_chat_show_the_responsive_daita_welcome(tmp_path: Path):
    boot = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    async with boot.run_test(size=(80, 24)):
        welcome = boot.query_one("#boot", WelcomeView)
        assert "DAITA  1.0.1" in str(welcome.content)
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
