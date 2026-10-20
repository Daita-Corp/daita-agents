"""Component-owned tests split from ``test_approval.py``."""

from __future__ import annotations

import pytest

from daita.capabilities import CapabilityInputError
from daita.llm.models import ToolCall
from daita.memory.capabilities import apply_memory_edit
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
        assert second.messages[1:-1] == transcript.messages[:-1]
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


def test_memory_edits_change_one_entry_and_reject_unclear_targets():
    assert apply_memory_edit("", "add", content=" FY starts in February. ") == (
        "FY starts in February."
    )
    assert apply_memory_edit("A\n", "add", content="B") == "A\nB"
    assert apply_memory_edit("A\nB\nC", "replace", old_text="B", content="D") == (
        "A\nD\nC"
    )
    assert apply_memory_edit("A\nB\nC", "remove", old_text="B") == "A\nC"
    assert apply_memory_edit("A\nB", "remove", old_text="B") == "A\n"
    assert apply_memory_edit("Keep A B", "remove", old_text=" B") == "Keep A"
    for current, action, arguments, code in (
        ("FY\nOther", "add", {"content": "FY"}, "memory_entry_exists"),
        ("A", "replace", {"old_text": "Z", "content": "B"}, "memory_text_not_found"),
        ("A A", "remove", {"old_text": "A"}, "memory_text_ambiguous"),
        ("A", "replace", {"old_text": "A", "content": " "}, "memory_invalid_content"),
    ):
        with pytest.raises(CapabilityInputError) as raised:
            apply_memory_edit(current, action, **arguments)
        assert raised.value.code == code


async def test_memory_replace_approves_only_the_edit_and_keeps_other_entries(
    tmp_path,
):
    approvals: list[ApprovalRequest] = []

    async def approve(request):
        approvals.append(request)
        return ApprovalDecision.APPROVE

    edit = ToolCall(
        id="edit",
        name="memory_set",
        arguments={
            "target": "memory",
            "action": "replace",
            "old_text": "Revenue excludes voids.",
            "content": "Revenue excludes voids and refunds.",
        },
    )
    provider = MockModelProvider((_call(edit), _stop("Updated.")))
    agent = await Agent.create(
        "edit-one-entry",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
        approval_handler=approve,
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.set_memory("FY starts in February.\nRevenue excludes voids.")
        await agent.run("Correction: revenue also excludes refunds.")
        assert await agent.read_memory() == (
            "FY starts in February.\nRevenue excludes voids and refunds."
        )
        assert len(approvals) == 1
        assert "FY starts in February." not in str(approvals[0].arguments)
    finally:
        await agent.close()
