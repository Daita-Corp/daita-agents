from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import stat
from typing import cast

import pytest

from daita import Agent
from daita.agent import AgentNotConfiguredError, SessionOperationActiveError
from daita.llm.models import ModelProfile
from daita.llm.providers.mock import MockModelProvider
from daita.memory import (
    MemoryInspectionRequest,
    MemoryListRequest,
    MemoryRecallRequest,
    MemoryScope,
)
from daita.operations.models import AgentTrigger, TriggerKind
from daita.operations.runtime import OperationRuntime
from daita.sessions import Session
from daita.skills import SkillNotFoundError

NOW = datetime(2026, 7, 18, 18, 0, tzinfo=timezone.utc)


def _profile(
    *,
    id: str = "mock:scripted",
    supports_tools: bool = True,
    supports_parallel_tools: bool = False,
    supports_structured_output: bool = False,
    supports_streaming: bool = False,
    supports_reasoning: bool = False,
    supports_vision: bool = False,
    supports_documents: bool = False,
    supports_prompt_caching: bool = False,
    supports_native_continuation: bool = False,
    input_cost_per_million_usd: Decimal | None = None,
    output_cost_per_million_usd: Decimal | None = None,
    data_routing_classification: str = "standard",
    available: bool = True,
    healthy: bool = True,
) -> ModelProfile:
    return ModelProfile(
        id=id,
        context_window_tokens=32_768,
        max_output_tokens=4_096,
        supports_tools=supports_tools,
        supports_parallel_tools=supports_parallel_tools,
        supports_structured_output=supports_structured_output,
        supports_streaming=supports_streaming,
        supports_reasoning=supports_reasoning,
        supports_vision=supports_vision,
        supports_documents=supports_documents,
        supports_prompt_caching=supports_prompt_caching,
        supports_native_continuation=supports_native_continuation,
        input_cost_per_million_usd=input_cost_per_million_usd,
        output_cost_per_million_usd=output_cost_per_million_usd,
        data_routing_classification=data_routing_classification,
        available=available,
        healthy=healthy,
    )


@pytest.mark.parametrize(
    "case",
    (
        "profile_without_model",
        "missing",
        "wrong_type",
        "provider_mismatch",
        "unavailable",
        "unhealthy",
        "no_tools",
    ),
)
async def test_model_profile_configuration_fails_before_home_mutation(
    tmp_path: Path,
    case: str,
) -> None:
    model = None if case == "profile_without_model" else MockModelProvider(())
    profile: object = _profile()
    expected_error: type[BaseException] = AgentNotConfiguredError
    if case == "missing":
        profile = None
    elif case == "wrong_type":
        profile = object()
        expected_error = TypeError
    elif case == "provider_mismatch":
        profile = _profile(id="mock:other")
    elif case == "unavailable":
        profile = _profile(available=False)
    elif case == "unhealthy":
        profile = _profile(healthy=False)
    elif case == "no_tools":
        profile = _profile(supports_tools=False)

    with pytest.raises(expected_error):
        await Agent.create(
            "atlas",
            root=tmp_path,
            model=model,
            model_profile=cast(ModelProfile, profile),
        )

    assert tuple(tmp_path.iterdir()) == ()


async def test_model_profile_binding_is_exact_restart_stable_and_drift_closed(
    tmp_path: Path,
) -> None:
    profile = _profile(
        supports_parallel_tools=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_reasoning=True,
        supports_vision=True,
        supports_documents=True,
        supports_prompt_caching=True,
        supports_native_continuation=True,
        input_cost_per_million_usd=Decimal("1.25"),
        output_cost_per_million_usd=Decimal("5.50"),
        data_routing_classification="restricted",
    )
    created_provider = MockModelProvider(())
    created = await Agent.create(
        "atlas",
        root=tmp_path,
        model=created_provider,
        model_profile=profile,
        clock=lambda: NOW,
    )
    assert created.model_profile == profile
    await created.close()

    reopened_provider = MockModelProvider(())
    reopened = await Agent.open(
        "atlas",
        root=tmp_path,
        model=reopened_provider,
        clock=lambda: NOW,
    )
    assert reopened.model_profile == profile
    assert reopened_provider.requests == ()
    await reopened.close()

    inspection_only = await Agent.open("atlas", root=tmp_path, clock=lambda: NOW)
    assert inspection_only.model_profile == profile
    await inspection_only.close()

    provider_drift = MockModelProvider((), provider_id="mock:other")
    with pytest.raises(AgentNotConfiguredError, match="provider differs"):
        await Agent.open("atlas", root=tmp_path, model=provider_drift)
    assert provider_drift.requests == ()

    static_drift = MockModelProvider(())
    with pytest.raises(AgentNotConfiguredError, match="profile differs"):
        await Agent.open(
            "atlas",
            root=tmp_path,
            model=static_drift,
            model_profile=replace(profile, max_output_tokens=2_048),
        )
    assert static_drift.requests == ()

    final_reopen = await Agent.open("atlas", root=tmp_path)
    assert final_reopen.model_profile == profile
    await final_reopen.close()


async def test_first_configured_open_binds_required_profile(tmp_path: Path) -> None:
    unconfigured = await Agent.create("atlas", root=tmp_path, clock=lambda: NOW)
    assert unconfigured.model_profile is None
    await unconfigured.close()

    missing_provider = MockModelProvider(())
    with pytest.raises(AgentNotConfiguredError, match="first configured open"):
        await Agent.open("atlas", root=tmp_path, model=missing_provider)
    assert missing_provider.requests == ()

    profile = _profile()
    configured = await Agent.open(
        "atlas",
        root=tmp_path,
        model=MockModelProvider(()),
        model_profile=profile,
    )
    assert configured.model_profile == profile
    await configured.close()

    reopened = await Agent.open(
        "atlas",
        root=tmp_path,
        model=MockModelProvider(()),
    )
    assert reopened.model_profile == profile
    await reopened.close()


async def test_default_composition_creates_private_empty_skills_directory(
    tmp_path: Path,
) -> None:
    agent = await Agent.create("atlas", root=tmp_path, clock=lambda: NOW)
    try:
        skills_root = agent.home / "skills"
        metadata = skills_root.lstat()
        assert skills_root.is_dir()
        assert not skills_root.is_symlink()
        assert stat.S_IMODE(metadata.st_mode) == 0o700
        assert tuple(skills_root.iterdir()) == ()
        assert await agent.list_skills() == ()
        assert await agent.refresh_skills() == ()
    finally:
        await agent.close()


async def test_public_memory_and_skill_facades_remain_agent_scoped(
    tmp_path: Path,
) -> None:
    alpha = await Agent.create("alpha", root=tmp_path, clock=lambda: NOW)
    beta = await Agent.create("beta", root=tmp_path, clock=lambda: NOW)
    try:
        foreign_scope = MemoryScope(agent_id=beta.id)
        with pytest.raises(ValueError, match="another agent"):
            await alpha.recall_memory(
                MemoryRecallRequest(
                    query="completed status",
                    scope=foreign_scope,
                )
            )
        with pytest.raises(ValueError, match="another agent"):
            await alpha.list_memories(MemoryListRequest(scope=foreign_scope))
        with pytest.raises(ValueError, match="another agent"):
            await alpha.inspect_memory(
                MemoryInspectionRequest(
                    agent_id=beta.id,
                    memory_id="memory:foreign",
                )
            )

        own_recall = await alpha.recall_memory(
            MemoryRecallRequest(
                query="completed status",
                scope=MemoryScope(agent_id=alpha.id),
            )
        )
        assert own_recall.hits == ()

        skill_directory = beta.home / "skills" / "private-procedure"
        skill_directory.mkdir(mode=0o700)
        (skill_directory / "SKILL.md").write_text(
            "+++\n"
            'name = "private-procedure"\n'
            'version = "1.0.0"\n'
            'description = "A beta-only procedure."\n'
            'activation_mode = "explicit"\n'
            "+++\n"
            "Use only capabilities already exposed by the current agent.\n",
            encoding="utf-8",
        )
        discovered = await beta.refresh_skills()
        assert len(discovered) == 1
        assert discovered[0].agent_id == beta.id
        inspection = await beta.activate_skill(
            discovered[0].skill_id,
            discovered[0].version_id,
            expected_active_version_id=None,
            actor_id="user:beta",
            reason="Enable beta's explicit procedure.",
        )
        assert inspection.skill.agent_id == beta.id
        assert inspection.index.active_version_id == discovered[0].version_id

        assert await alpha.list_skills() == ()
        with pytest.raises(SkillNotFoundError):
            await alpha.inspect_skill(discovered[0].skill_id)
    finally:
        await beta.close()
        await alpha.close()


async def test_new_session_message_fails_before_creating_a_second_operation(
    tmp_path: Path,
) -> None:
    provider = MockModelProvider(())
    agent = await Agent.create(
        "atlas",
        root=tmp_path,
        model=provider,
        model_profile=_profile(),
        clock=lambda: NOW,
    )
    try:
        store = agent._embedded._store
        await store.create_session(
            Session(
                id="session-active",
                agent_id=agent.id,
                title="Active work",
                created_at=NOW,
                updated_at=NOW,
            )
        )
        runtime = OperationRuntime(store=store, clock=lambda: NOW)
        existing = await runtime.begin(
            AgentTrigger(
                id="trigger-active",
                agent_id=agent.id,
                kind=TriggerKind.USER,
                source_id="user:session-active",
                session_id="session-active",
                payload={"message": "Existing active objective"},
                created_at=NOW,
            )
        )

        with pytest.raises(SessionOperationActiveError) as raised:
            await agent.run("Do something else", session_id="session-active")

        assert raised.value.code == "session_operation_active"
        assert raised.value.operation_id == existing.operation.id
        assert provider.requests == ()
        nonterminal = await store.load_nonterminal(agent.id)
        assert tuple(item.snapshot.operation.id for item in nonterminal) == (
            existing.operation.id,
        )
    finally:
        await agent.close()
