import asyncio

import pytest
from _toolbox_model_support import ToolboxAwareMockModelProvider as MockModelProvider

from daita import Agent
from daita.capabilities import ApprovalDecision
from daita.llm.models import FinishReason, ModelResponse, ModelSensitivity, ToolCall
from daita.llm.routing import ModelProviderRegistration, ModelRouter, RetryPolicy
from daita.loop.models import LoopExitKind
from daita.memory import MemoryStore
from daita.skills import SkillStore


async def test_imported_advisory_text_requires_conservative_classification(tmp_path):
    (tmp_path / "MEMORY.md").write_text("Private imported note")
    store = MemoryStore(tmp_path, asyncio.Lock())
    assert await store.read_context() == (
        "Private imported note",
        "",
        ModelSensitivity.RESTRICTED,
    )
    await store.set_memory("Explicit public note", sensitivity=ModelSensitivity.PUBLIC)
    await store.close()
    reopened = MemoryStore(tmp_path, asyncio.Lock())
    assert (await reopened.read_context())[2] is ModelSensitivity.PUBLIC
    await reopened.close()

    directory = tmp_path / "skills" / "imported"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        "# imported\n\nPrivate procedure\n\n## Instructions\n\nUse the private method.\n"
    )
    skills = SkillStore(tmp_path, asyncio.Lock())
    imported = await skills.read_skill("imported")
    assert imported is not None
    assert imported.sensitivity is ModelSensitivity.RESTRICTED
    await skills.close()


@pytest.mark.parametrize("target", ("memory", "skill"))
async def test_model_retained_advice_keeps_full_request_floor_across_reopen(
    tmp_path, target
):
    call = (
        ToolCall(
            "retain",
            "memory_set",
            {
                "target": "memory",
                "content": "Use the confidential accounting convention.",
            },
        )
        if target == "memory"
        else ToolCall(
            "retain",
            "skill_save",
            {
                "name": "accounting",
                "description": "Accounting convention",
                "instructions": "Use the confidential accounting convention.",
            },
        )
    )
    provider = MockModelProvider(
        (
            ModelResponse(finish_reason=FinishReason.TOOL_CALLS, tool_calls=(call,)),
            ModelResponse(finish_reason=FinishReason.STOP, text="Saved."),
        )
    )

    async def approve(request):
        return ApprovalDecision.APPROVE

    agent = await Agent.create(
        "advisory-floor",
        root=tmp_path,
        hosted=True,
        model=provider,
        model_profile=provider.model_profile,
        approval_handler=approve,
    )
    await agent.set_user_profile(
        "Confidential accounting convention.", sensitivity=ModelSensitivity.CONFIDENTIAL
    )
    result = await agent.learn("Save this convention for future use.")
    assert result.kind is LoopExitKind.COMPLETED
    assert provider.requests and all(
        request.sensitivity is ModelSensitivity.CONFIDENTIAL
        for request in provider.requests
    )
    if target == "memory":
        assert (await agent._embedded._memory_store.read_context())[
            2
        ] is ModelSensitivity.CONFIDENTIAL
    else:
        skill = await agent.read_skill("accounting")
        assert skill is not None and skill.sensitivity is ModelSensitivity.CONFIDENTIAL
        _, digest = await agent._embedded._skill_store.read_skill_with_digest(
            "accounting"
        )
        retained = await agent._embedded._skill_store.retain_current_skill(
            "accounting", "sha256:" + digest
        )
        assert retained.sensitivity is ModelSensitivity.CONFIDENTIAL
    await agent.set_user_profile("", sensitivity=ModelSensitivity.PUBLIC)
    await agent.close()

    public = MockModelProvider(
        (ModelResponse(finish_reason=FinishReason.STOP, text="Must not run"),)
    )
    router = ModelRouter(
        (
            ModelProviderRegistration(
                provider=public,
                profile=public.model_profile,
                allowed_sensitivities=frozenset({ModelSensitivity.PUBLIC}),
            ),
        ),
        retry_policy=RetryPolicy(
            max_attempts_per_candidate=1, max_total_attempts=1, backoff_seconds=0
        ),
    )
    reopened = await Agent.open(
        "advisory-floor",
        root=tmp_path,
        hosted=True,
        model=router,
        model_profile=router.model_profile,
    )
    try:
        result = await reopened.run("A fresh conversation")
        assert result.reason == "model_route_ineligible"
        assert result.sensitivity is ModelSensitivity.CONFIDENTIAL
        assert public.requests == ()
    finally:
        await reopened.close()


async def test_model_document_and_later_artifact_read_keep_full_request_classification(
    tmp_path,
):
    from daita.artifacts.models import ArtifactAuthorship
    from daita.llm.models import ToolResultBlock

    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "document",
                        "artifact_create_document",
                        {
                            "format": "markdown",
                            "content": "Confidential analysis from the current request.",
                        },
                    ),
                ),
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP, text="The document was created."
            ),
        )
    )
    agent = await Agent.create(
        "artifact-floor",
        root=tmp_path,
        hosted=True,
        model=provider,
        model_profile=provider.model_profile,
    )
    try:
        await agent.set_user_profile(
            "Confidential background.", sensitivity=ModelSensitivity.CONFIDENTIAL
        )
        created = await agent.run("Write a short analysis document.")
        assert created.kind is LoopExitKind.COMPLETED and len(created.artifacts) == 1
        artifact = created.artifacts[0]
        assert artifact.sensitivity.value == "confidential"
        assert (
            artifact.provenance.authorship is ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS
        )
        await agent.set_user_profile("", sensitivity=ModelSensitivity.PUBLIC)
        provider.replace_script(
            (
                ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(
                        ToolCall(
                            "read-stored",
                            "artifact_read",
                            {"artifact_id": artifact.artifact_id},
                        ),
                    ),
                ),
                ModelResponse(
                    finish_reason=FinishReason.STOP,
                    text="The classified document was read.",
                ),
            )
        )
        read = await agent.run("Read this exact stored artifact in a new conversation.")
        transcript = await agent.transcript(read.run_id)
        result = next(
            block
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock) and block.call_id == "read-stored"
        )
        assert not result.is_error
        assert result.sensitivity is ModelSensitivity.CONFIDENTIAL
        assert read.sensitivity is ModelSensitivity.CONFIDENTIAL
    finally:
        await agent.close()
