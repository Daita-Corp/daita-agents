"""Component-owned tests split from ``test_toolbox_contracts.py``."""

from __future__ import annotations

from tests.capabilities._toolbox_support import (
    NOW,
    Agent,
    MockModelProvider,
    ModelProfile,
    RunInput,
    TextBlock,
    _load,
    canonical_json,
    pytest,
    replace,
    workspace_for,
)


@pytest.mark.parametrize(
    "routes,cost",
    [
        ((), None),
        (("mock:one",), None),
        (("mock:beta", "mock:alpha"), "0.25"),
        (tuple(f"mock:{index:03d}" + "x" * 240 for index in range(64)), "0.25"),
    ],
)
async def test_routine_authoring_choices_are_complete_frozen_and_mandatory(
    tmp_path, monkeypatch, routes, cost
):
    from decimal import Decimal

    from daita.llm.errors import ContextWindowExceeded

    agent = await Agent.create(
        "authoring-facts",
        root=tmp_path,
        model=MockModelProvider((), provider_id="mock:facts"),
        model_profile=ModelProfile(
            id="mock:facts",
            context_window_tokens=64000,
            max_output_tokens=2000,
            supports_tools=True,
        ),
        workspace=workspace_for(tmp_path),
    )
    try:
        runtime = agent._embedded._capability_runtime
        builder = agent._embedded._context_builder
        owner = agent._embedded._routine_owner
        assert builder is not None
        monkeypatch.setattr(owner, "_eligible_model_routes", routes)
        monkeypatch.setattr(
            owner, "_maximum_per_run_cost_usd", None if cost is None else Decimal(cost)
        )
        run = RunInput(
            id="facts-run",
            agent_id=agent.id,
            message="Prepare a recurring briefing.",
            created_at=NOW,
            conversation_id="facts-conversation",
        )
        messages = (run.start_message(),)
        catalog = await runtime.prepare_run(run)
        snapshot = await builder.prepare(run, messages, catalog)
        facts = snapshot.routine_authoring_facts
        assert facts is not None
        assert facts["eligible_model_routes"] == routes
        assert facts["maximum_per_run_cost_usd"] == cost
        monkeypatch.setattr(owner, "_eligible_model_routes", ("mock:later",))
        result, loaded_messages, projection = await _load(
            runtime, run, catalog, messages, ("routine_create",), call_id="facts-load"
        )
        assert not result.is_error, result
        request = builder.project(
            snapshot, loaded_messages, step=2, tool_context=projection
        )
        system = "\n".join(
            block.text
            for block in request.messages[0].content
            if isinstance(block, TextBlock)
        )
        assert canonical_json(facts) in system
        assert "mock:later" not in system
        assert "No current-model alias exists" in system
        # Facts are mandatory when authoring is callable; they cannot silently
        # disappear under hard context pressure or turn into optional discovery.
        with pytest.raises(ContextWindowExceeded):
            builder.project(
                replace(
                    snapshot,
                    profile=replace(
                        snapshot.profile,
                        context_window_tokens=1500,
                        max_output_tokens=500,
                    ),
                ),
                loaded_messages,
                step=2,
                tool_context=projection,
            )
    finally:
        await agent.close()
