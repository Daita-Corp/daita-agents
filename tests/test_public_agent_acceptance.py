from __future__ import annotations

from collections.abc import Iterable, Mapping

from daita import (
    Agent,
    AgentEvent,
    AgentEventKind,
    ApprovalDecision,
    ApprovalRequest,
    ConversationRun,
    Skill,
    SkillSummary,
)
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import LoopLimits, ToolProjectionMode

EAGER_LIMITS = LoopLimits(tool_projection_mode=ToolProjectionMode.EAGER)


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=32_000,
        max_output_tokens=2_000,
        supports_tools=True,
        supports_parallel_tools=True,
    )


def _stop(text: str) -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text)


def _call(call_id: str, name: str, arguments: dict[str, object]) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
    )


def _texts(request: ModelRequest, *, roles: Iterable[MessageRole]) -> tuple[str, ...]:
    selected = frozenset(roles)
    return tuple(
        block.text
        for message in request.messages
        if message.role in selected
        for block in message.content
        if isinstance(block, TextBlock)
    )


def _tool_results(request: ModelRequest) -> tuple[ToolResultBlock, ...]:
    return tuple(
        block
        for message in request.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    )


async def test_completed_mvp_public_agent_journey_survives_cold_reopen(tmp_path):
    initial_memory = "ALPHA_KNOWLEDGE: revenue excludes voided invoices."
    updated_memory = "OMEGA_KNOWLEDGE: revenue uses paid invoice date."
    skill_description = "Apply the approved monthly revenue procedure."
    skill_instructions = "SKILL_SECRET: group paid invoices by UTC month."
    first_question = "Use the monthly procedure for ACCOUNT_DATA_SENTINEL."
    first_answer = "I used the saved monthly procedure."
    follow_question = "Now use the same procedure for EMEA."
    follow_answer = "I reused the prior procedure for EMEA."
    update_question = "Remember the corrected revenue definition."
    update_answer = "I saved the approved correction."
    cold_question = "Apply that correction to the same analysis."
    cold_answer = "The persisted context and conversation are available."

    provider = MockModelProvider(
        (
            _call("view-monthly", "skill_view", {"name": "monthly-revenue"}),
            _stop(first_answer),
            _stop(follow_answer),
            _call(
                "replace-memory",
                "memory_set",
                {"target": "memory", "content": updated_memory},
            ),
            _stop(update_answer),
        )
    )
    events: list[AgentEvent] = []
    approvals: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.APPROVE

    agent = await Agent.create(
        "public-journey",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
        approval_handler=approve,
        observer=events.append,
    )
    try:
        await agent.set_memory(initial_memory)
        assert await agent.save_skill(
            "monthly-revenue",
            skill_description,
            skill_instructions,
        )
        assert await agent.list_skills() == (
            SkillSummary("monthly-revenue", skill_description),
        )

        first = await agent.run(first_question)
        conversation_id = first.conversation_id
        assert conversation_id
        assert first.final_text == first_answer

        first_request = provider.requests[0]
        first_system = "\n".join(_texts(first_request, roles=(MessageRole.SYSTEM,)))
        assert initial_memory in first_system
        assert skill_description in first_system
        assert skill_instructions not in first_system
        viewed = _tool_results(provider.requests[1])
        assert len(viewed) == 1
        viewed_data = viewed[0].output["data"]
        assert isinstance(viewed_data, Mapping)
        assert viewed_data["instructions"] == skill_instructions

        follow = await agent.run(
            follow_question,
            conversation_id=conversation_id,
        )
        assert follow.final_text == follow_answer
        follow_context = _texts(
            provider.requests[2],
            roles=(MessageRole.USER, MessageRole.ASSISTANT),
        )
        assert first_question in follow_context
        assert first_answer in follow_context
        assert follow_question in follow_context

        follow_transcript = await agent.transcript(follow.run_id)
        follow_transcript_text = tuple(
            block.text
            for message in follow_transcript.messages
            for block in message.content
            if isinstance(block, TextBlock)
        )
        assert follow_transcript_text == (follow_question, follow_answer)

        updated = await agent.run(
            update_question,
            conversation_id=conversation_id,
        )
        assert updated.final_text == update_answer
        assert len(approvals) == 1
        assert approvals[0].tool_name == "memory_set"
        assert dict(approvals[0].arguments) == {
            "target": "memory",
            "content": updated_memory,
        }
        replacement_results = tuple(
            result
            for result in _tool_results(provider.requests[4])
            if result.call_id == "replace-memory"
        )
        assert len(replacement_results) == 1
        assert replacement_results[0].is_error is False
        replacement_data = replacement_results[0].output["data"]
        assert isinstance(replacement_data, Mapping)
        assert replacement_data["target"] == "memory"
        assert replacement_data["replaced"] is True
        assert await agent.read_memory() == updated_memory
        provider.assert_consumed()
    finally:
        await agent.close()

    reopened_provider = MockModelProvider((_stop(cold_answer),))
    reopened = await Agent.open(
        "public-journey",
        root=tmp_path,
        model=reopened_provider,
        model_profile=_profile(reopened_provider),
        approval_handler=approve,
        observer=events.append,
    )
    try:
        assert await reopened.read_memory() == updated_memory
        assert await reopened.read_skill("monthly-revenue") == Skill(
            "monthly-revenue",
            skill_description,
            skill_instructions,
        )

        cold = await reopened.run(
            cold_question,
            conversation_id=conversation_id,
        )
        assert cold.final_text == cold_answer
        cold_context = _texts(
            reopened_provider.requests[0],
            roles=(MessageRole.USER, MessageRole.ASSISTANT),
        )
        expected_order = (
            first_question,
            first_answer,
            follow_question,
            follow_answer,
            update_question,
            update_answer,
            cold_question,
        )
        assert tuple(item for item in cold_context if item in expected_order) == (
            expected_order
        )
        cold_system = "\n".join(
            _texts(reopened_provider.requests[0], roles=(MessageRole.SYSTEM,))
        )
        assert updated_memory in cold_system
        assert skill_description in cold_system
        assert skill_instructions not in cold_system

        runs = await reopened.conversation_runs(conversation_id)
        assert all(isinstance(run, ConversationRun) for run in runs)
        assert tuple(run.turn_index for run in runs) == (0, 1, 2, 3)
        assert tuple(run.transcript.run.id for run in runs) == (
            first.run_id,
            follow.run_id,
            updated.run_id,
            cold.run_id,
        )
        assert all(run.result is not None for run in runs)
        assert tuple(len(run.transcript.messages) for run in runs) == (4, 2, 4, 2)
        assert all(
            run.transcript.run.conversation_id == conversation_id for run in runs
        )
        reopened_provider.assert_consumed()
    finally:
        await reopened.close()

    assert tuple(event.kind for event in events) == (
        AgentEventKind.RUN_STARTED,
        AgentEventKind.MODEL_COMPLETED,
        AgentEventKind.TOOL_STARTED,
        AgentEventKind.TOOL_COMPLETED,
        AgentEventKind.MODEL_COMPLETED,
        AgentEventKind.RUN_COMPLETED,
        AgentEventKind.RUN_STARTED,
        AgentEventKind.MODEL_COMPLETED,
        AgentEventKind.RUN_COMPLETED,
        AgentEventKind.RUN_STARTED,
        AgentEventKind.MODEL_COMPLETED,
        AgentEventKind.TOOL_STARTED,
        AgentEventKind.APPROVAL_REQUESTED,
        AgentEventKind.APPROVAL_DECIDED,
        AgentEventKind.TOOL_COMPLETED,
        AgentEventKind.MODEL_COMPLETED,
        AgentEventKind.RUN_COMPLETED,
        AgentEventKind.RUN_STARTED,
        AgentEventKind.MODEL_COMPLETED,
        AgentEventKind.RUN_COMPLETED,
    )
    rendered_events = repr(events)
    for raw_value in (
        initial_memory,
        updated_memory,
        skill_instructions,
        first_question,
        follow_question,
        update_question,
        cold_question,
    ):
        assert raw_value not in rendered_events
