"""Component-owned tests split from ``test_screens.py``."""

from __future__ import annotations

from tests.tui._support import (
    Agent,
    Button,
    CatalogScreen,
    ChatScreen,
    Composer,
    ConfirmScreen,
    DaitaApp,
    Input,
    OptionList,
    Path,
    PermissionsScreen,
    SelectionScreen,
    SimpleNamespace,
    SourceEditScreen,
    SQLiteSource,
    Static,
    Tree,
    _create_sqlite_source,
    asyncio,
    sqlite3,
    workspace_for,
)


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


async def test_source_permissions_configures_exact_relational_write_scope():
    read_scope = SimpleNamespace(mode=SimpleNamespace(value="all"), resource_ids=())
    initial_state = SimpleNamespace(
        read_scope=read_scope,
        relational_write_scopes=(),
    )
    tickets = SimpleNamespace(
        resource_id="resource-tickets",
        key_columns=("ticket_id",),
        display_name="support.tickets",
        resource_kind="table",
        eligible_assignment_columns=("priority", "ticket_status"),
        relational_update_eligible=True,
        upsert_conflict_keys=(),
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
        updates = kwargs["relational_write_scopes"]
        assert isinstance(updates, dict)
        scopes = tuple(
            SimpleNamespace(
                resource_id=resource_id,
                **columns,
                constraints=lambda columns=columns: dict(columns),
            )
            for resource_id, columns in updates.items()
        )
        after = SimpleNamespace(
            read_scope=SimpleNamespace(
                mode=SimpleNamespace(value=kwargs["read_mode"]),
                resource_ids=kwargs["read_resource_ids"],
            ),
            relational_write_scopes=scopes,
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

        assert await pilot.click("#perm-write") is True
        await pilot.pause()
        choose_single(app, tickets.resource_id)
        await pilot.pause()
        choose_single(app, "update")
        await pilot.pause()
        choose_multi(app, "priority")
        await pilot.pause()

        assert app.screen is permissions
        assert preview_calls == [
            {
                "source_id": inspection.source_id,
                "read_mode": "all",
                "read_resource_ids": (),
                "relational_write_scopes": {
                    tickets.resource_id: {
                        "allowed_operations": ("update",),
                        "allowed_insert_columns": (),
                        "allowed_update_columns": ("priority",),
                        "key_columns": ("ticket_id",),
                        "generated_identity_columns": (),
                        "max_rows": 100,
                    },
                },
            }
        ]
        body = str(permissions.query_one("#perm-body", Static).content)
        assert "Before → after" in body
        assert '"table": "support.tickets"' in body
        assert '"max_rows": 100' in body

        assert await pilot.click("#perm-apply") is True
        await pilot.pause()
        assert apply_calls == [
            {
                "source_id": inspection.source_id,
                "confirmation_fingerprint": "sha256:" + "1" * 64,
            }
        ]
        app.exit(0)


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
            (active,) = tuple(
                item for item in await opened.list_sources() if item.active
            )
            assert active.configuration["path"] == str(edited_path)
            app.exit(0)
    finally:
        await opened.close()


async def test_postgresql_source_edit_probes_and_selects_schemas(monkeypatch):
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    source = SimpleNamespace(
        id="source-postgresql",
        active=True,
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

    async def list_sources() -> tuple[object, ...]:
        return (source,)

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

    monkeypatch.setattr(app.controller, "list_sources", list_sources)
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

    async def list_sources() -> tuple[object, ...]:
        return ()

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
            assert source_labels[0] == "Sales  SQLite · 1 resource"
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


async def test_catalog_notice_and_tree_are_ready_before_mount_completes():
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    entered, release = asyncio.Event(), asyncio.Event()

    class DelayedCatalog(CatalogScreen):
        async def on_mount(self) -> None:  # type: ignore[override]
            entered.set()
            await release.wait()

    screen = DelayedCatalog(
        summary=SimpleNamespace(
            active_source_count=1, resource_count=0, relationship_count=0
        ),
        sources=(
            SimpleNamespace(
                id="source-empty", display_name="Empty source", adapter_id="sqlite"
            ),
        ),
        resources=(),
        notice="No current resources",
        notice_warning=True,
    )
    async with app.run_test(size=(100, 34)):
        mounted = asyncio.ensure_future(app.push_screen(screen))
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
            assert (
                str(screen.query_one("#catalog-notice", Static).content)
                == "No current resources"
            )
            tree = screen.query_one("#catalog-tree", Tree)
            assert len(tree.root.children) == 1
            assert "0 resources" in str(tree.root.children[0].label)
            assert (
                str(tree.root.children[0].children[0].label) == "No current resources"
            )
        finally:
            release.set()
            await mounted
            app.exit(0)


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
