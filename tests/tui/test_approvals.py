"""Component-owned tests split from ``test_screens.py``."""

from __future__ import annotations

from tests.tui._support import (
    CAPABILITY_LABELS,
    App,
    ApprovalDecision,
    ApprovalPanel,
    ApprovalRequest,
    ChatScreen,
    Composer,
    ComposeResult,
    ConfirmScreen,
    DaitaApp,
    FrozenJsonObject,
    Input,
    ReviewCostScreen,
    Static,
    ToolCall,
    ToolResultBlock,
    approval_review_document,
    asyncio,
    project_tool_details,
    pytest,
    workspace_for,
)


def test_tool_projection_and_approval_document():
    assert CAPABILITY_LABELS["toolbox_search"] == "Search toolboxes"
    assert CAPABILITY_LABELS["toolbox_load"] == "Load selected tools"
    assert CAPABILITY_LABELS["file_search"] == "Search workspace files"
    assert CAPABILITY_LABELS["file_read"] == "Read workspace file"
    assert CAPABILITY_LABELS["file_query"] == "Query workspace data"
    assert CAPABILITY_LABELS["artifact_edit_text"] == "Prepare workspace edit"
    assert CAPABILITY_LABELS["artifact_save_local"] == "Save artifact locally"
    details = project_tool_details(
        ToolCall(
            id="c1",
            name="data_query",
            arguments={
                "source_id": "source-1",
                "resource_ids": ("resource-1",),
                "sql": "SELECT 1",
            },
        ),
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
        reason="Replace the exact unchanged workspace file config.yaml?",
    )
    assert reviewable and document is not None
    assert "Change: Replace the exact unchanged workspace file config.yaml?" in document
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


async def test_approval_approve_deny_cancel_and_unreviewable():
    request = ApprovalRequest(
        run_id="run-1",
        call_id="call-1",
        tool_name="data_update_rows",
        capability_id="data.update_rows",
        arguments=FrozenJsonObject.from_mapping({"name": "safe"}),
        reason="update one row",
    )
    secret = ApprovalRequest(
        run_id="run-2",
        call_id="call-2",
        tool_name="data_update_rows",
        capability_id="data.update_rows",
        arguments=FrozenJsonObject.from_mapping({"password": "hidden-secret"}),
        reason="secret shaped",
    )
    oversized = ApprovalRequest(
        run_id="run-3",
        call_id="call-3",
        tool_name="data_update_rows",
        capability_id="data.update_rows",
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
