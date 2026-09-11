"""Component-owned tests split from ``test_screens.py``."""

from __future__ import annotations

from tests.tui._support import (
    BUILTIN_SLASH_COMMAND_ROOTS,
    BUILTIN_SLASH_COMMANDS,
    MIN_READY_ROWS,
    MIN_USABLE_COLUMNS,
    SLASH_COMMAND_COMPLETIONS,
    Agent,
    App,
    Button,
    CompletionPopup,
    Composer,
    ComposeResult,
    DaitaApp,
    OptionList,
    Path,
    PickerOption,
    SelectionScreen,
    SQLiteSource,
    Static,
    Text,
    _create_sqlite_source,
    parse_source_override,
    pytest,
    workspace_for,
)


def test_source_override_and_learning_parse():
    assert parse_source_override("hello") is None
    assert parse_source_override("@sales how many") == ("sales", "how many")
    assert parse_source_override('@"north west" total') == ("north west", "total")


def test_every_advertised_slash_root_is_the_recognized_builtin_set():
    advertised_roots = frozenset(
        display.split(maxsplit=1)[0]
        for _insertion, display, _description in SLASH_COMMAND_COMPLETIONS
    )

    assert BUILTIN_SLASH_COMMAND_ROOTS == advertised_roots
    assert BUILTIN_SLASH_COMMANDS == advertised_roots


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
