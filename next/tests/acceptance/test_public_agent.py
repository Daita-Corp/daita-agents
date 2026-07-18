from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import threading
import tomllib

import pytest

from daita import Agent
from daita.agent import (
    AgentHomeError,
    AgentIdentityMismatchError,
    AgentNameError,
    AgentNotConfiguredError,
    HostActiveError,
)
from daita.hosting import embedded as embedded_owner
from daita.identity import AgentIdentity
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolDefinition,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import LoopExitKind, Readiness, Turn
from daita.operations.checkpoints import OperationSnapshot
from daita.operations.models import (
    ActionProposal,
    AgentTrigger,
    Evidence,
    Observation,
    TriggerKind,
)
from daita.operations.runtime import OperationRuntime
from daita.operations.store import InvalidOperationCheckpointError
from daita.sessions import Session

NOW = datetime(2026, 7, 18, 18, 0, tzinfo=timezone.utc)


class TextContext:
    async def build(
        self,
        operation: OperationSnapshot,
        turn: Turn,
        tools: tuple[ToolDefinition, ...],
    ) -> ModelRequest:
        assert tools == ()
        message = operation.trigger.payload["message"]
        assert isinstance(message, str)
        return ModelRequest(
            operation_id=operation.operation.id,
            turn_id=turn.id,
            messages=(
                CanonicalMessage(
                    agent_id=operation.operation.agent_id,
                    operation_id=operation.operation.id,
                    session_id=operation.operation.session_id,
                    turn_id=turn.id,
                    role=MessageRole.USER,
                    content=(TextBlock(message),),
                ),
            ),
        )


class TextDomain:
    def tool_views(self, operation: OperationSnapshot) -> tuple[ToolDefinition, ...]:
        return ()

    async def validate_action(
        self, call: ToolCall, operation: OperationSnapshot
    ) -> ActionProposal:
        raise AssertionError("text-only domain has no actions")

    async def project_observation(self, evidence: Evidence) -> Observation:
        raise AssertionError("text-only domain has no observations")

    async def evaluate_final_answer(
        self, text: str, operation: OperationSnapshot
    ) -> Readiness:
        return Readiness(
            allowed=True,
            code="ready",
            message="Text is ready.",
            evaluated_at=NOW,
        )


def _provider(*texts: str) -> MockModelProvider:
    return MockModelProvider(
        tuple(
            ModelResponse(text=text, finish_reason=FinishReason.STOP) for text in texts
        )
    )


async def test_create_reopen_identity_manifest_and_default_v1_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    v1 = tmp_path / ".daita"
    v1.mkdir()
    sentinel = v1 / "sentinel"
    sentinel.write_text("v1", encoding="utf-8")

    agent = await Agent.create("atlas", clock=lambda: NOW)
    agent_id = agent.id
    assert agent.home == tmp_path / ".daita-next" / "agents" / "atlas"
    manifest = tomllib.loads((agent.home / "agent.toml").read_text())
    assert manifest == {
        "manifest_version": 1,
        "agent_id": agent_id,
        "display_name": "atlas",
        "state_path": "state.db",
        "state_schema_generation": 2,
        "created_at": "2026-07-18T18:00:00+00:00",
    }
    assert sentinel.read_text(encoding="utf-8") == "v1"
    await agent.close()

    reopened = await Agent.open("atlas", clock=lambda: NOW)
    assert reopened.id == agent_id
    assert reopened.name == "atlas"
    await reopened.close()
    assert sentinel.read_text(encoding="utf-8") == "v1"


@pytest.mark.parametrize("name", ("../atlas", "a/b", ".", "", "atlas.toml"))
async def test_agent_name_cannot_escape_its_isolated_home(
    tmp_path: Path, name: str
) -> None:
    with pytest.raises(AgentNameError):
        await Agent.create(name, root=tmp_path)


async def test_manifest_database_mismatch_fails_closed(tmp_path: Path) -> None:
    agent = await Agent.create("atlas", root=tmp_path, clock=lambda: NOW)
    home = agent.home
    await agent.close()

    manifest_path = home / "agent.toml"
    manifest_path.write_text(
        manifest_path.read_text().replace(agent.id, "agent-forged"),
        encoding="utf-8",
    )
    with pytest.raises(AgentIdentityMismatchError):
        await Agent.open("atlas", root=tmp_path)

    manifest_path.write_text(
        manifest_path.read_text().replace("agent-forged", agent.id),
        encoding="utf-8",
    )
    with sqlite3.connect(home / "state.db") as connection:
        connection.execute("UPDATE agents SET display_name = 'forged'")
    with pytest.raises(AgentIdentityMismatchError):
        await Agent.open("atlas", root=tmp_path)


async def test_state_database_symlink_is_never_opened(tmp_path: Path) -> None:
    agent = await Agent.create("atlas", root=tmp_path, clock=lambda: NOW)
    home = agent.home
    await agent.close()
    real_state = home / "real-state.db"
    (home / "state.db").rename(real_state)
    (home / "state.db").symlink_to(real_state)

    with pytest.raises(AgentIdentityMismatchError, match="symlink"):
        await Agent.open("atlas", root=tmp_path)


async def test_embedded_instances_share_one_per_agent_writer_lock(
    tmp_path: Path,
) -> None:
    first = await Agent.create("atlas", root=tmp_path)
    with pytest.raises(HostActiveError) as raised:
        await Agent.open("atlas", root=tmp_path)
    assert raised.value.code == "host_active"
    await first.close()

    reopened = await Agent.open("atlas", root=tmp_path)
    await reopened.close()


async def test_concurrent_create_has_one_admitted_writer(tmp_path: Path) -> None:
    outcomes = await asyncio.gather(
        Agent.create("atlas", root=tmp_path),
        Agent.create("atlas", root=tmp_path),
        return_exceptions=True,
    )
    winners = [outcome for outcome in outcomes if isinstance(outcome, Agent)]
    losers = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    assert len(winners) == 1
    assert len(losers) == 1
    assert isinstance(losers[0], HostActiveError)
    await winners[0].close()


async def test_agent_home_rejects_v1_root_and_symlink_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    v1 = tmp_path / ".daita"
    v1.mkdir()
    with pytest.raises(AgentHomeError, match="v1"):
        await Agent.create("atlas", root=v1)
    with pytest.raises(AgentHomeError, match="v1"):
        await Agent.create("atlas", root=v1 / "nested")

    real_root = tmp_path / "real"
    real_root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(AgentHomeError, match="symlink"):
        await Agent.create("atlas", root=alias / "nested")
    with pytest.raises(AgentHomeError, match="alias"):
        await Agent.create("atlas", root=tmp_path / "parent" / ".." / "state")


async def test_invalid_embedded_composition_writes_no_agent_home(
    tmp_path: Path,
) -> None:
    with pytest.raises(AgentNotConfiguredError):
        await Agent.create("atlas", root=tmp_path, model=_provider("unused"))
    assert not (tmp_path / "agents" / "atlas").exists()


async def test_cancelled_admission_releases_returned_writer_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_admit = embedded_owner._admit_agent_home
    admitted = threading.Event()
    release = threading.Event()

    def delayed_admit(
        name: str, root: str | Path | None, create: bool
    ) -> tuple[Path, embedded_owner._WriterLock]:
        result = real_admit(name, root, create)
        admitted.set()
        assert release.wait(timeout=5)
        return result

    monkeypatch.setattr(embedded_owner, "_admit_agent_home", delayed_admit)
    task = asyncio.create_task(Agent.create("atlas", root=tmp_path))
    assert await asyncio.to_thread(admitted.wait, 5)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    monkeypatch.setattr(embedded_owner, "_admit_agent_home", real_admit)
    retry = await Agent.create("atlas", root=tmp_path)
    await retry.close()


async def test_manifest_failure_cleans_only_new_bootstrap_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_write = embedded_owner._write_manifest

    def fail_manifest(home: Path, identity: AgentIdentity) -> None:
        raise OSError("injected manifest failure")

    monkeypatch.setattr(embedded_owner, "_write_manifest", fail_manifest)
    with pytest.raises(OSError, match="injected manifest failure"):
        await Agent.create("atlas", root=tmp_path)
    home = tmp_path / "agents" / "atlas"
    assert not (home / "agent.toml").exists()
    assert not (home / "state.db").exists()

    monkeypatch.setattr(embedded_owner, "_write_manifest", real_write)
    retry = await Agent.create("atlas", root=tmp_path)
    await retry.close()


async def test_manifest_publication_cancellation_leaves_reopenable_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_write = embedded_owner._write_manifest
    published = threading.Event()
    release = threading.Event()

    def delayed_manifest(home: Path, identity: AgentIdentity) -> None:
        real_write(home, identity)
        published.set()
        assert release.wait(timeout=5)

    monkeypatch.setattr(embedded_owner, "_write_manifest", delayed_manifest)
    task = asyncio.create_task(Agent.create("atlas", root=tmp_path))
    assert await asyncio.to_thread(published.wait, 5)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    monkeypatch.setattr(embedded_owner, "_write_manifest", real_write)
    reopened = await Agent.open("atlas", root=tmp_path)
    await reopened.close()


async def test_authoritative_home_requires_session_and_updates_monotonically(
    tmp_path: Path,
) -> None:
    agent = await Agent.create("atlas", root=tmp_path, clock=lambda: NOW)
    store = agent._embedded._store
    earlier = NOW - timedelta(hours=1)
    runtime = OperationRuntime(store=store, clock=lambda: earlier)
    missing_trigger = AgentTrigger(
        id="trigger-missing-session",
        agent_id=agent.id,
        kind=TriggerKind.USER,
        source_id="user:missing",
        session_id="missing-session",
        payload={"message": "must fail closed"},
        created_at=earlier,
    )
    with pytest.raises(
        InvalidOperationCheckpointError,
        match="requires a persisted session",
    ):
        await runtime.begin(missing_trigger)

    await store.create_session(
        Session(
            id="session-a",
            agent_id=agent.id,
            title="Stable timestamp",
            created_at=NOW,
            updated_at=NOW,
        )
    )
    await runtime.begin(
        AgentTrigger(
            id="trigger-earlier-operation",
            agent_id=agent.id,
            kind=TriggerKind.USER,
            source_id="user:session-a",
            session_id="session-a",
            payload={"message": "clock moved backward"},
            created_at=earlier,
        )
    )
    transcript = await store.load_session(agent.id, "session-a")
    assert transcript is not None
    assert transcript.session.updated_at == NOW
    await agent.close()


async def test_run_inspect_resume_and_session_transcripts_survive_reopen(
    tmp_path: Path,
) -> None:
    first = await Agent.create(
        "atlas",
        root=tmp_path,
        model=_provider("first answer", "isolated answer", "stateless answer"),
        context_builder=TextContext(),
        domain=TextDomain(),
        clock=lambda: NOW,
    )
    first_result = await first.run("first question", session_id="session-a")
    isolated_result = await first.run("isolated question", session_id="session-b")
    stateless_result = await first.run("stateless question")
    assert first_result.kind is LoopExitKind.COMPLETED
    inspected = await first.inspect(first_result.operation_id)
    assert inspected.operation.final_text == "first answer"
    assert (
        await first.inspect(stateless_result.operation_id)
    ).operation.session_id is None
    assert await first.resume(first_result.operation_id) == first_result
    await first.close()

    reopened = await Agent.open(
        "atlas",
        root=tmp_path,
        model=_provider("follow-up answer"),
        context_builder=TextContext(),
        domain=TextDomain(),
        clock=lambda: NOW,
    )
    transcript_a = await reopened.transcript("session-a")
    transcript_b = await reopened.transcript("session-b")
    assert [message.role for message in transcript_a.messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert [
        block.text for message in transcript_a.messages for block in message.content
    ] == [
        "first question",
        "first answer",
    ]
    assert [
        block.text for message in transcript_b.messages for block in message.content
    ] == [
        "isolated question",
        "isolated answer",
    ]
    assert isolated_result.operation_id not in transcript_a.operation_ids
    assert stateless_result.operation_id not in transcript_a.operation_ids
    assert stateless_result.operation_id not in transcript_b.operation_ids

    follow_up = await reopened.run("follow up", session_id="session-a")
    assert follow_up.final_text == "follow-up answer"
    transcript_a = await reopened.transcript("session-a")
    assert len(transcript_a.operation_ids) == 2
    assert len(transcript_a.messages) == 4
    assert [
        block.text for message in transcript_a.messages for block in message.content
    ] == [
        "first question",
        "first answer",
        "follow up",
        "follow-up answer",
    ]
    await reopened.close()
