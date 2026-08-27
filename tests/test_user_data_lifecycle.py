from __future__ import annotations

import json
import sqlite3
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest
from _workspace_support import workspace_for

import daita.hosting.embedded as embedded
from daita import Agent, SQLiteSource, cli
from daita.adapters.models import SourceRegistration
from daita.agent import AgentHomeError
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    ToolCall,
)
from daita.llm.providers.mock import MockModelProvider
from daita.security import KeychainSecretProvider, SecretReference
from daita.skills import Skill
from daita.tui.controller import PresentationController

NOW = datetime(2026, 7, 29, tzinfo=UTC)


class _Keychain:
    def __init__(self, *, fail_delete: bool = False) -> None:
        self.values: dict[str, str] = {}
        self.deleted: list[str] = []
        self.fail_delete = fail_delete

    async def resolve(self, reference: SecretReference) -> str:
        return self.values[reference.name]

    async def set(self, reference: SecretReference, value: str) -> None:
        self.values[reference.name] = value

    async def delete(self, reference: SecretReference) -> None:
        if self.fail_delete:
            raise RuntimeError("keychain unavailable")
        self.deleted.append(reference.name)
        self.values.pop(reference.name, None)


class _MissingKeyringEntry:
    def get_password(self, service_name: str, username: str) -> None:
        del service_name, username
        return None

    def set_password(
        self,
        service_name: str,
        username: str,
        password: str,
    ) -> None:
        raise AssertionError((service_name, username, password))

    def delete_password(self, service_name: str, username: str) -> None:
        del service_name, username
        raise RuntimeError("entry is already absent")


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=32_000,
        max_output_tokens=1_000,
        supports_tools=True,
    )


def _stop(text: str) -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text)


def _validation_response(provider_id: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCall(
                id="validation",
                name="daita_validate_tool_support",
                arguments={},
            ),
        ),
        provider_id=provider_id,
    )


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE records (id INTEGER PRIMARY KEY)")


async def test_agent_state_root_and_database_are_owner_only(tmp_path: Path) -> None:
    root = tmp_path / "state-root"
    root.mkdir(mode=0o755)

    agent = await Agent.create("atlas", root=root, workspace=workspace_for(root))
    try:
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
        assert stat.S_IMODE(agent.home.stat().st_mode) == 0o700
        assert stat.S_IMODE((agent.home / "state.db").stat().st_mode) == 0o600
    finally:
        await agent.close()


async def test_clear_conversations_removes_transcript_derived_state_only(
    tmp_path: Path,
) -> None:
    provider = MockModelProvider((_stop("answer"),), provider_id="mock:lifecycle")
    agent = await Agent.create(
        "atlas",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.set_memory("Approved durable memory.")
        await agent.save_skill(
            "approved-skill",
            "An approved durable skill.",
            "Use the current catalog before querying.",
        )
        result = await agent.run("Remember this transcript only.")
        state_path = agent.home / "state.db"
        with sqlite3.connect(state_path) as connection:
            connection.execute(
                """INSERT INTO learning_candidates(agent_id, id, data)
                   VALUES (?, ?, ?)""",
                (agent.id, "candidate-sensitive", "sensitive candidate"),
            )
            connection.execute(
                "INSERT INTO metadata(key, data) VALUES (?, ?)",
                (f"learning_review_stamps:{agent.id}", "sensitive stamps"),
            )

        assert await agent.clear_conversations() == 1
        assert not await agent.conversation_exists(result.conversation_id)
        with pytest.raises(KeyError, match="unknown run"):
            await agent.transcript(result.run_id)
        assert await agent.read_memory() == "Approved durable memory."
        assert await agent.read_skill("approved-skill") == Skill(
            "approved-skill",
            "An approved durable skill.",
            "Use the current catalog before querying.",
        )
        with sqlite3.connect(state_path) as connection:
            assert connection.execute("SELECT COUNT(*) FROM runs").fetchone() == (0,)
            assert connection.execute("SELECT COUNT(*) FROM messages").fetchone() == (
                0,
            )
            assert connection.execute(
                "SELECT COUNT(*) FROM learning_candidates"
            ).fetchone() == (0,)
            assert connection.execute(
                "SELECT COUNT(*) FROM metadata WHERE key LIKE 'learning_review_stamps:%'"
            ).fetchone() == (0,)
    finally:
        await agent.close()


async def test_detach_deletes_only_an_owned_postgresql_credential(
    tmp_path: Path,
) -> None:
    keychain = _Keychain()
    agent = await Agent.create(
        "atlas", root=tmp_path, keychain=keychain, workspace=workspace_for(tmp_path)
    )
    try:
        reference = SecretReference.keychain(
            embedded._credential_account(agent.id, "postgresql", "credential-fixed")
        )
        keychain.values[reference.name] = "secret"
        registration = SourceRegistration.build(
            agent_id=agent.id,
            adapter_id="postgresql",
            native_identity="postgresql:test",
            display_name="Warehouse",
            configuration={"credential_ref": reference.to_uri()},
            attached_at=NOW,
        )
        await agent._embedded._store.register_source(registration)

        detached = await agent.detach(registration.id)

        assert not detached.active
        assert keychain.deleted == [reference.name]
        assert reference.name not in keychain.values
    finally:
        await agent.close()


async def test_agent_delete_removes_home_and_all_owned_credentials(
    tmp_path: Path,
) -> None:
    keychain = _Keychain()
    agent = await Agent.create(
        "atlas", root=tmp_path, keychain=keychain, workspace=workspace_for(tmp_path)
    )
    await agent.close()
    validator = MockModelProvider(
        (_validation_response("openai:test-model"),),
        provider_id="openai:test-model",
    )
    configured = await Agent.open(
        "atlas",
        root=tmp_path,
        keychain=keychain,
        model_validator=validator,
        workspace=workspace_for(tmp_path),
    )
    route = await configured.configure_model(
        provider="openai",
        model="test-model",
        api_key="secret",
        context_window_tokens=8_192,
        max_output_tokens=1_024,
    )
    reference = route.candidates[0].secret_reference
    assert reference is not None
    home = configured.home
    await configured.close()
    config_path = home / "config.json"
    document = json.loads(config_path.read_text(encoding="utf-8"))
    document["model_route"]["candidates"][0]["profile"]["healthy"] = False
    config_path.write_text(json.dumps(document), encoding="utf-8")

    await Agent.delete("atlas", root=tmp_path, keychain=keychain)

    assert not home.exists()
    assert await Agent.list(root=tmp_path) == ()
    assert keychain.deleted == [reference.name]
    assert keychain.values == {}


async def test_agent_delete_preserves_home_when_credential_cleanup_fails(
    tmp_path: Path,
) -> None:
    keychain = _Keychain()
    agent = await Agent.create(
        "atlas", root=tmp_path, keychain=keychain, workspace=workspace_for(tmp_path)
    )
    await agent.close()
    validator = MockModelProvider(
        (_validation_response("openai:test-model"),),
        provider_id="openai:test-model",
    )
    configured = await Agent.open(
        "atlas",
        root=tmp_path,
        keychain=keychain,
        model_validator=validator,
        workspace=workspace_for(tmp_path),
    )
    await configured.configure_model(
        provider="openai",
        model="test-model",
        api_key="secret",
        context_window_tokens=8_192,
        max_output_tokens=1_024,
    )
    home = configured.home
    await configured.close()
    keychain.fail_delete = True

    with pytest.raises(AgentHomeError, match="home was preserved"):
        await Agent.delete("atlas", root=tmp_path, keychain=keychain)

    assert home.is_dir()
    assert await Agent.list(root=tmp_path) == ("atlas",)


async def test_agent_delete_respects_the_active_writer_lock(tmp_path: Path) -> None:
    agent = await Agent.create(
        "atlas", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        with pytest.raises(AgentHomeError, match="host_active"):
            await Agent.delete("atlas", root=tmp_path)
        assert agent.home.is_dir()
    finally:
        await agent.close()


async def test_keychain_delete_is_idempotent_when_the_entry_is_already_absent() -> None:
    keychain = KeychainSecretProvider(client=_MissingKeyringEntry())

    await keychain.delete(SecretReference.keychain("already-absent"))


async def test_terminal_lifecycle_commands_require_confirmation(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data.sqlite"
    _database(database)
    provider = MockModelProvider((_stop("answer"),), provider_id="mock:lifecycle")
    agent = await Agent.create(
        "atlas",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        workspace=workspace_for(tmp_path),
    )
    source = await agent.attach(SQLiteSource(database, name="Warehouse"))
    result = await agent.run("Create history.")
    controller = PresentationController(
        root=tmp_path, workspace=workspace_for(tmp_path)
    )
    controller.agent = agent
    controller.conversation_id = result.conversation_id

    clear = await controller.dispatch_command("/conversation clear")
    assert clear.kind == "confirm"
    assert await agent.conversation_exists(result.conversation_id)

    detach = await controller.dispatch_command(f"/source detach {source.id}")
    assert detach.kind == "confirm"
    assert (await agent.resolve_source(source.id)).active

    delete = await controller.dispatch_command("/agent delete")
    assert delete.kind == "confirm"
    assert agent.home.is_dir()
    await agent.close()


async def test_terminal_can_clear_history_detach_source_and_delete_agent(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data.sqlite"
    _database(database)
    provider = MockModelProvider((_stop("answer"),), provider_id="mock:lifecycle")
    agent = await Agent.create(
        "atlas",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        workspace=workspace_for(tmp_path),
    )
    source = await agent.attach(SQLiteSource(database, name="Warehouse"))
    result = await agent.run("Create history.")
    controller = PresentationController(
        root=tmp_path, workspace=workspace_for(tmp_path)
    )
    controller.agent = agent
    controller.conversation_id = result.conversation_id

    cleared = await controller.clear_conversations()
    assert cleared.conversation_id is None
    assert not await agent.conversation_exists(result.conversation_id)

    await controller.detach_source(source.id)
    controller.conversation_id = None
    assert not (await agent.list_sources())[0].active

    await controller.delete_open_agent()
    assert controller.agent is None
    assert not (tmp_path / "agents" / "atlas").exists()


async def test_headless_lifecycle_commands_require_and_honor_yes(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data.sqlite"
    _database(database)
    agent = await Agent.create(
        "atlas", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    source = await agent.attach(SQLiteSource(database, name="Warehouse"))
    await agent.close()
    parser = cli.build_parser()

    for arguments, message in (
        (
            ["--root", str(tmp_path), "detach", "atlas", source.id],
            "detach requires --yes",
        ),
        (
            ["--root", str(tmp_path), "conversations", "clear", "atlas"],
            "conversations clear requires --yes",
        ),
        (
            ["--root", str(tmp_path), "delete", "atlas"],
            "delete requires --yes",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            await cli._execute(parser.parse_args(arguments))

    detached = await cli._execute(
        parser.parse_args(
            [
                "--root",
                str(tmp_path),
                "detach",
                "atlas",
                source.id,
                "--yes",
            ]
        )
    )
    assert detached == {
        "source_id": source.id,
        "name": "Warehouse",
        "detached": True,
    }
    assert await cli._execute(
        parser.parse_args(
            [
                "--root",
                str(tmp_path),
                "conversations",
                "clear",
                "atlas",
                "--yes",
            ]
        )
    ) == {"name": "atlas", "cleared_runs": 0}
    assert await cli._execute(
        parser.parse_args(["--root", str(tmp_path), "delete", "atlas", "--yes"])
    ) == {"name": "atlas", "deleted": True}
    assert not (tmp_path / "agents" / "atlas").exists()
