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
    CatalogScreen,
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

    assert advertised_roots < BUILTIN_SLASH_COMMAND_ROOTS
    assert BUILTIN_SLASH_COMMANDS == BUILTIN_SLASH_COMMAND_ROOTS
    assert BUILTIN_SLASH_COMMAND_ROOTS - advertised_roots == {
        "/catalog",
        "/source",
    }


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
            try:
                await app._show_chat()
                await pilot.pause()
                composer = app.screen.query_one(Composer)
                popup = app.screen.query_one(CompletionPopup)
                listing = popup.query_one(OptionList)
                composer.focus()
                assert composer.styles.border_top[0] == "heavy"
                assert composer.styles.border_right[0] == "heavy"

                await pilot.press("/")
                await pilot.pause()
                assert popup.display is True
                await pilot.press("ctrl+3")
                await pilot.pause()
                assert popup.display is False
                assert composer.text == "/"
                await pilot.press("ctrl+1")
                await pilot.pause()
                assert popup.display is True
                assert popup.styles.border_left[0] == "solid"
                assert popup.styles.background.ansi == -1
                assert (
                    listing.get_component_styles(
                        "option-list--option-highlighted"
                    ).text_style.reverse
                    is True
                )
                assert listing.highlighted == 0
                assert popup.selected_insertion() == "/save"
                assert listing.option_count == len(SLASH_COMMAND_COMPLETIONS)

                for _ in range(len(SLASH_COMMAND_COMPLETIONS) - 1):
                    popup.move_highlight(1)
                assert popup.selected_insertion() == "/agent delete"
                assert listing.scroll_offset.y > 0
                for _ in range(len(SLASH_COMMAND_COMPLETIONS) - 1):
                    popup.move_highlight(-1)
                assert popup.selected_insertion() == "/save"

                # Answer actions lead the menu; /sources opens the Data group.
                await pilot.press("down", "down", "enter")
                await pilot.pause()
                assert composer.text == "/sources"
                assert popup.display is False
                assert app._run_task is None

                await pilot.press("enter")
                await pilot.pause()
                assert isinstance(app.screen, CatalogScreen)
                await pilot.press("escape")
                await pilot.pause()
                assert composer.text == ""
                assert app._run_task is None

                await pilot.press("/", "s", "o", "u")
                await pilot.pause()
                assert popup.display is True
                assert {shown for _insert, shown, _description in popup.matches} == {
                    "/sources"
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


async def test_selection_screen_tertiary_action_does_not_require_a_selection():
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
                        PickerOption("first", "First"),
                        PickerOption("second", "Second"),
                    ),
                    secondary_action=PickerOption("create", "Create new"),
                    tertiary_action=PickerOption("delete", "Delete"),
                )
            )
            self.exit(result)

    with pytest.raises(ValueError, match="tertiary action identity must differ"):
        SelectionScreen(
            title="Invalid",
            options=(PickerOption("same", "Existing"),),
            tertiary_action=PickerOption("same", "Delete"),
        )
    with pytest.raises(ValueError, match="picker action identities must be distinct"):
        SelectionScreen(
            title="Invalid",
            options=(PickerOption("existing", "Existing"),),
            secondary_action=PickerOption("action", "Create"),
            tertiary_action=PickerOption("action", "Delete"),
        )

    app = Harness()
    async with app.run_test(size=(90, 28)) as pilot:
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, SelectionScreen)
        buttons = tuple(picker.query_one("#picker-actions").query(Button))
        assert [str(button.label) for button in buttons] == [
            "Select",
            "Create new",
            "Delete",
        ]
        assert len({button.region.y for button in buttons}) == 1
        assert await pilot.click("#picker-tertiary") is True
        await pilot.pause()
    assert app.return_value == ("delete",)


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


def test_menu_groups_label_every_row_and_starter_questions_name_real_tables():
    from daita.tui.commands import (
        COMMAND_GROUPS,
        completion_group,
        suggested_questions,
    )

    for group, commands in COMMAND_GROUPS:
        assert all(completion_group(display) == group for _i, display, _d in commands)
    assert completion_group("@sales") == "Source"
    assert completion_group("/weekly-report") == "Skill"

    assert suggested_questions(("orders", "customers")) == (
        "What is in orders?",
        "Show the 10 most recent rows of orders",
        "How does orders relate to customers?",
    )
    assert suggested_questions(("orders",))[-1] == (
        "Show the 10 most recent rows of orders"
    )
    assert all("folder" in question for question in suggested_questions(()))
