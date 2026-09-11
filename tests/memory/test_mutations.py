"""Component-owned tests split from ``test_approval.py``."""

from __future__ import annotations

from tests.support.approval_learning import (
    EAGER_LIMITS,
    Agent,
    ApprovalDecision,
    ApprovalRequest,
    FrozenJsonObject,
    MockModelProvider,
    _call,
    _memory_call,
    _profile,
    _stop,
    _system_text,
    _tool_results,
    workspace_for,
)


async def test_explicit_correction_is_one_approved_foreground_memory_write(tmp_path):
    content = "Revenue means paid invoice subtotal, excluding voided invoices."
    approvals: list[ApprovalRequest] = []

    async def approve(request):
        approvals.append(request)
        return ApprovalDecision.APPROVE

    provider = MockModelProvider(
        (_call(_memory_call(content=content)), _stop("I saved the correction."))
    )
    agent = await Agent.create(
        "learn-correction",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
        approval_handler=approve,
        workspace=workspace_for(tmp_path),
    )
    try:
        result = await agent.run(
            "Correction: revenue means paid invoice subtotal. Remember this."
        )
        assert result.final_text == "I saved the correction."
        assert await agent.read_memory() == content
        assert len(approvals) == 1
        assert approvals[0].arguments["content"] == content
        assert len(provider.logical_requests) == 2
        assert content not in _system_text(provider.logical_requests[1])
        first, second = provider.logical_requests
        assert (
            first.sensitivity_provenance["static_context_sha256"]
            == second.sensitivity_provenance["static_context_sha256"]
        )
        assert first.tools == second.tools
        assert (
            f"Remaining model requests including this one: {EAGER_LIMITS.max_steps - provider.requests.index(first)}."
            in _system_text(first)
        )
        assert (
            f"Remaining model requests including this one: {EAGER_LIMITS.max_steps - provider.requests.index(second)}."
            in _system_text(second)
        )
        transcript = await agent.transcript(result.run_id)
        assert second.messages[1:] == transcript.messages[:-1]
        tool_result = _tool_results(provider)[0]
        assert tool_result.output["data"] == FrozenJsonObject.from_mapping(
            {"target": "memory", "replaced": True}
        )
        prompt = _system_text(provider.logical_requests[0])
        assert "explicit durable definitions/preferences/corrections" in prompt
        assert "ordinary text ends run" in prompt
        assert "Approval card alone confirms" in prompt
        assert "never ask typed approval" in prompt
    finally:
        await agent.close()


async def test_weak_learning_signal_stays_in_transcript_without_a_write(tmp_path):
    approvals: list[ApprovalRequest] = []

    async def approve(request):
        approvals.append(request)
        return ApprovalDecision.APPROVE

    provider = MockModelProvider((_stop("That is the current result."),))
    agent = await Agent.create(
        "weak-learning-signal",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
        approval_handler=approve,
        workspace=workspace_for(tmp_path),
    )
    try:
        result = await agent.run("Revenue happened to be 42 in this one result.")
        assert result.final_text == "That is the current result."
        assert await agent.read_memory() == ""
        assert await agent.read_user_profile() == ""
        assert await agent.list_skills() == ()
        assert approvals == []
        prompt = _system_text(provider.requests[0])
        assert "inference/one-offs are weak" in prompt
        assert "Never learn raw results" in prompt
        assert "Approval card alone confirms" in prompt
        assert "never ask typed approval" in prompt
    finally:
        await agent.close()
