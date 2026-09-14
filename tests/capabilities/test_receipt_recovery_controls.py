"""Component-owned tests split from ``test_product_workflows.py``."""

from __future__ import annotations

from tests.support.product_workflows import (
    ActionFixture,
    Agent,
    EffectResolutionDecision,
    cli,
    patch,
    pytest,
    pytestmark as _pytestmark,
    workspace_for,
)

pytestmark = _pytestmark


@pytest.mark.parametrize("decision", tuple(EffectResolutionDecision))
async def test_receipt_recovery_controls_preserve_observation_and_dispatch_nothing(
    tmp_path, decision
):
    fixture = await ActionFixture(tmp_path).start()
    try:
        fixture.transport.mode = "disconnect"
        fixture.script()
        await fixture.agent.run("Send one reviewed test notification.")
        receipt = (await fixture.agent.list_effects(unresolved_only=True))[0]
        calls = list(fixture.server.calls)
        note = (
            "Investigated the remote system; record this exact decision without retry."
        )
        await fixture.agent.close()
        original_open = Agent.open

        async def opened(name, **kwargs):
            return await original_open(name, **{**fixture.kwargs(), **kwargs})

        args = cli.build_parser().parse_args(
            [
                "--root",
                str(fixture.root),
                "--workspace",
                str(workspace_for(tmp_path).root),
                "effects",
                "resolve",
                "mcp-actions",
                receipt.receipt_id,
                "--expected-digest",
                receipt.receipt_digest,
                "--decision",
                decision.value,
                "--note",
                note,
            ]
        )
        with (
            patch.object(Agent, "open", side_effect=opened),
            patch("builtins.input", return_value="y"),
        ):
            result = await cli._execute(args)
        assert isinstance(result, dict)
        assert isinstance(result["resolution"], dict)
        assert result["receipt_digest"] == receipt.receipt_digest
        assert result["resolution"]["decision"] == decision.value
        await fixture.reopen()
        resolved = await fixture.agent.inspect_effect(receipt.receipt_id)
        assert resolved is not None and resolved.resolution is not None
        assert resolved.receipt_digest == receipt.receipt_digest
        assert resolved.outcome == receipt.outcome
        assert resolved.resolution.decision is decision
        assert fixture.server.calls == calls
    finally:
        await fixture.agent.close()
