from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sqlite3

import pytest

from daita import Agent, SQLiteSource
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
from daita import cli, terminal
import daita.hosting.embedded as embedded

NOW = datetime(2026, 7, 29, tzinfo=timezone.utc)


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


async def test_clear_conversations_removes_transcript_derived_state_only(
    tmp_path: Path,
) -> None:
    provider = MockModelProvider((_stop("answer"),), provider_id="mock:lifecycle")
    agent = await Agent.create(
        "atlas",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
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
    agent = await Agent.create("atlas", root=tmp_path, keychain=keychain)
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
    agent = await Agent.create("atlas", root=tmp_path, keychain=keychain)
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
    agent = await Agent.create("atlas", root=tmp_path, keychain=keychain)
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
    agent = await Agent.create("atlas", root=tmp_path)
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
    )
    source = await agent.attach(SQLiteSource(database, name="Warehouse"))
    result = await agent.run("Create history.")
    output = io.StringIO()

    agent, conversation_id, action = await terminal._handle_local_command(
        "/conversation clear",
        agent=agent,
        root=tmp_path,
        input_stream=io.StringIO("n\n"),
        output_stream=output,
        hidden_input=lambda _prompt: "",
        keychain=None,
        model_validator=None,
        approval_handler=None,
        conversation_id=result.conversation_id,
        validated=True,
    )
    assert conversation_id == result.conversation_id
    assert action is None
    assert await agent.conversation_exists(result.conversation_id)

    agent, conversation_id, action = await terminal._handle_local_command(
        f"/source detach {source.id}",
        agent=agent,
        root=tmp_path,
        input_stream=io.StringIO("n\n"),
        output_stream=output,
        hidden_input=lambda _prompt: "",
        keychain=None,
        model_validator=None,
        approval_handler=None,
        conversation_id=conversation_id,
        validated=True,
    )
    assert action is None
    assert (await agent.resolve_source(source.id)).active

    agent, conversation_id, action = await terminal._handle_local_command(
        "/agent delete",
        agent=agent,
        root=tmp_path,
        input_stream=io.StringIO("wrong-name\n"),
        output_stream=output,
        hidden_input=lambda _prompt: "",
        keychain=None,
        model_validator=None,
        approval_handler=None,
        conversation_id=conversation_id,
        validated=True,
    )
    assert action is None
    assert agent.home.is_dir()
    assert "was not changed" in output.getvalue()
    assert "was not detached" in output.getvalue()
    assert "was not deleted" in output.getvalue()
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
    )
    source = await agent.attach(SQLiteSource(database, name="Warehouse"))
    result = await agent.run("Create history.")
    output = io.StringIO()

    agent, conversation_id, action = await terminal._handle_local_command(
        "/conversation clear",
        agent=agent,
        root=tmp_path,
        input_stream=io.StringIO("y\n"),
        output_stream=output,
        hidden_input=lambda _prompt: "",
        keychain=None,
        model_validator=None,
        approval_handler=None,
        conversation_id=result.conversation_id,
        validated=True,
    )
    assert conversation_id is None
    assert action is None
    assert not await agent.conversation_exists(result.conversation_id)

    agent, conversation_id, action = await terminal._handle_local_command(
        f"/source detach {source.id}",
        agent=agent,
        root=tmp_path,
        input_stream=io.StringIO("y\n"),
        output_stream=output,
        hidden_input=lambda _prompt: "",
        keychain=None,
        model_validator=None,
        approval_handler=None,
        conversation_id=conversation_id,
        validated=True,
    )
    assert conversation_id is None
    assert action == "sources"
    assert not (await agent.list_sources())[0].active

    agent, conversation_id, action = await terminal._handle_local_command(
        "/agent delete",
        agent=agent,
        root=tmp_path,
        input_stream=io.StringIO("atlas\n"),
        output_stream=output,
        hidden_input=lambda _prompt: "",
        keychain=None,
        model_validator=None,
        approval_handler=None,
        conversation_id=conversation_id,
        validated=True,
    )
    assert conversation_id is None
    assert action == "deleted"
    assert not (tmp_path / "agents" / "atlas").exists()


async def test_headless_lifecycle_commands_require_and_honor_yes(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data.sqlite"
    _database(database)
    agent = await Agent.create("atlas", root=tmp_path)
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
