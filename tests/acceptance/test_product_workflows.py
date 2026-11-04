"""Component-owned tests split from ``test_product_integration.py``."""

from __future__ import annotations

from daita import connect
from daita.tui.screens.selection import SelectionScreen
from tests.support.host import StartHost, host_workspace, short_root
from tests.support.product_workflows import (
    ActionFixture,
    DaitaApp,
    Input,
    OnceSchedule,
    OutcomeState,
    PermissionsScreen,
    Static,
    ToolCall,
    approval_review_document,
    choose,
    create_fixture,
    pytestmark as _pytestmark,
    render_routine_inspection,
    replace,
    response,
    timedelta,
)
from tests.tui._support import settle_until

pytestmark = _pytestmark


async def test_guided_write_authoring_applies_exact_preview(
    monkeypatch, start_host: StartHost
):
    # The fixture builds the agent in process; the host then owns it with the
    # same model and clock, and the app connects to it.
    root = short_root()
    (
        built,
        model,
        _db,
        _research,
        _binding,
        resource,
        constraints,
        _batch,
        clock,
        _approvals,
    ) = await create_fixture(root, monkeypatch)
    await built.close()
    harness = await start_host(
        root=root,
        agent_options={
            "model": model,
            "model_profile": model.model_profile,
            "clock": lambda: clock[0],
        },
    )
    agent = await connect("native-acceptance", cwd=harness.work(), root=root)
    app = DaitaApp(root=root, start_bootstrap=False, workspace=host_workspace(harness))
    app.controller.agent = agent
    try:
        inspection = await agent.inspect_source_permissions(resource.source_id)
        choice = next(
            item for item in inspection.resources if item.resource_id == resource.id
        )
        assert "domain" in choice.eligible_insert_columns
        before = inspection.state
        async with app.run_test(size=(110, 38)) as pilot:
            await app.push_screen(PermissionsScreen(source_id=resource.source_id))
            manager = app.screen
            assert isinstance(manager, PermissionsScreen)
            manager.query_one("#perm-max-rows", Input).value = "7"

            async def next_picker(previous):
                await settle_until(
                    pilot,
                    lambda: isinstance(app.screen, SelectionScreen)
                    and app.screen is not previous,
                )

            assert await pilot.click("#perm-write")
            await settle_until(pilot, lambda: isinstance(app.screen, SelectionScreen))
            for identities, multi in (
                ((resource.id,), False),
                (("insert", "update"), True),
                (("domain", "name", "evidence_url"), True),
            ):
                picker = app.screen
                choose(app, identities, multi=multi)
                await next_picker(picker)
            choose(app, ("name", "evidence_url"), multi=True)
            await settle_until(pilot, lambda: manager._preview is not None)
            assert app.screen is manager
            assert isinstance(manager, PermissionsScreen)
            preview = manager._preview
            assert preview is not None and manager._reviewable
            assert (
                await agent.inspect_source_permissions(resource.source_id)
            ).state == before
            assert preview.after.relational_write_scopes[0].max_rows == 7
            assert preview.after.relational_write_scopes[0].allowed_operations == (
                "insert",
                "update",
            )
            body = str(manager.query_one("#perm-body", Static).content)
            assert '"table": "public.companies"' in body and '"max_rows": 7' in body
            assert await pilot.click("#perm-apply")
            await settle_until(pilot, lambda: app.screen is not manager)
            assert (
                await agent.inspect_source_permissions(resource.source_id)
            ).state == preview.after
            app.exit(0)
    finally:
        await agent.close()


async def test_once_mcp_research_creates_sourced_briefing_and_inbox_while_host_open(
    tmp_path,
):
    fixture = await ActionFixture(tmp_path).start()
    try:
        intent = replace(
            await fixture.draft(),
            title="Competitor briefing",
            prompt="Research competitors using the admitted tool and create a sourced briefing with coverage limits.",
            schedule=OnceSchedule(fixture.clock + timedelta(hours=1)),
            run_immediately=False,
        )
        routine = await fixture.agent.create_routine(
            intent,
            conversation_id=fixture.origin.conversation_id,
            confirmation_handler=fixture.approve,
        )
        assert len(fixture.approvals) == 1
        request = fixture.approvals[0]
        exact = request.render_arguments_for_review()
        document, reviewable = approval_review_document(
            tool_name=request.tool_name,
            capability_id=request.capability_id,
            arguments_text=exact,
            reason=request.reason,
        )
        assert reviewable and exact is not None and document is not None
        assert document.endswith(exact)
        assert "Competitor briefing" in document
        assert fixture.server.calls == []
        await fixture.agent.close()
        fixture.clock += timedelta(hours=1)
        assert fixture.server.calls == []  # No host means no scheduled progress.
        fixture.model.steps = [
            response(
                ToolCall(
                    "load",
                    "toolbox_load",
                    {
                        "tool_names": (
                            fixture.research.local_name,
                            "artifact_create",
                        )
                    },
                )
            ),
            response(
                ToolCall(
                    "research", fixture.research.local_name, {"query": "competitors"}
                )
            ),
            response(
                ToolCall(
                    "briefing",
                    "artifact_create",
                    {
                        "from": "document",
                        "format": "markdown",
                        "content": "# Competitor briefing\n[Research source](https://research.test/report) reports improved status. Coverage is limited; researched claims are not exhaustive.",
                    },
                )
            ),
            response(text="Created a sourced briefing; coverage is limited."),
        ]
        await fixture.reopen()
        delivery = await fixture.delivery()
        assert delivery is not None
        assert delivery.delivery.outcome.conclusion_state is OutcomeState.SUCCEEDED
        assert len(delivery.delivery.outcome.artifact_references) == 1
        assert [name for name, _ in fixture.server.calls] == ["research"]
        assert await fixture.agent.list_sources() == ()
        assert await fixture.agent.list_effects() == ()
        inspection = await fixture.agent.inspect_routine(routine.routine_id)
        assert inspection is not None
        rendered = render_routine_inspection(inspection).plain
        assert "completed" in rendered and "delivery_ids" in rendered
    finally:
        await fixture.agent.close()
