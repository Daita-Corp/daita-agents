from __future__ import annotations

import asyncio
import io
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from daita import Agent, SQLiteSource, terminal
from daita.agent import PostgreSQLProbeResult, PostgreSQLSourceError
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    TextBlock,
    ToolCall,
)
from daita.llm.providers.mock import MockModelProvider
from daita.security import SecretReference, SecretResolutionError
from daita.terminal import run_terminal_application


class _Keychain:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.events: list[tuple[str, str]] = []

    async def resolve(self, reference: SecretReference) -> str:
        return self.values[reference.name]

    async def set(self, reference: SecretReference, value: str) -> None:
        self.events.append(("set", reference.name))
        self.values[reference.name] = value

    async def delete(self, reference: SecretReference) -> None:
        self.events.append(("delete", reference.name))
        self.values.pop(reference.name, None)


def _provider() -> MockModelProvider:
    return MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="validation-call",
                        name="daita_validate_tool_support",
                        arguments={},
                    ),
                ),
                provider_id="openai:test-model",
            ),
        ),
        provider_id="openai:test-model",
    )


async def _configured_agent(root: Path, keychain: _Keychain) -> Agent:
    provider = _provider()
    agent = await Agent.create(
        "atlas",
        root=root,
        model_validator=provider,
        keychain=keychain,
    )
    await agent.configure_model(
        provider="openai",
        model="test-model",
        api_key="model-secret",
        context_window_tokens=8_192,
        max_output_tokens=256,
    )
    await agent.close()
    return await Agent.open("atlas", root=root, keychain=keychain)


@pytest.mark.parametrize("source_kind", ("sqlite", "directory"))
async def test_terminal_onboards_local_sources_and_renders_polished_ready_screen(
    tmp_path: Path,
    source_kind: str,
):
    keychain = _Keychain()
    agent = await _configured_agent(tmp_path, keychain)
    await agent.close()
    if source_kind == "sqlite":
        source_path = tmp_path / "fixture.sqlite"
        with sqlite3.connect(source_path) as connection:
            connection.execute("CREATE TABLE records (id INTEGER PRIMARY KEY)")
        source_input = f"1\n{source_path}\nFixture SQLite\n"
    else:
        source_path = tmp_path / "files"
        source_path.mkdir()
        (source_path / "records.csv").write_text("id,name\n1,Ada\n", encoding="utf-8")
        source_input = f"2\n{source_path}\nFixture files\n"
    output = io.StringIO()

    result = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(source_input),
        output_stream=output,
        hidden_input=lambda prompt: (_ for _ in ()).throw(
            AssertionError(f"unexpected hidden prompt: {prompt}")
        ),
        keychain=keychain,
    )

    assert result == 0
    text = output.getvalue()
    assert "Discovering tables and relationships" in text
    assert "Catalog ready" in text
    assert "Stage 2 status" not in text
    assert "Stage 4 status" not in text
    assert "Agent     atlas" in text
    assert "Model     OpenAI API · test-model · configured" in text
    assert "Connections  1" in text
    assert "Fixture SQLite" not in text
    assert "Fixture files" not in text
    expected_resources = "1 resource" if source_kind == "sqlite" else "2 resources"
    assert "Catalog   " in text and expected_resources in text
    assert "Status" in text and "Ready" in text
    assert "Conversation  new" in text
    assert "You › " in text
    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    try:
        sources = await reopened.list_sources()
        assert len(sources) == 1
        assert sources[0].display_name in {"Fixture SQLite", "Fixture files"}
    finally:
        await reopened.close()


async def test_terminal_postgresql_probe_selects_only_requested_schemas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keychain = _Keychain()
    agent = await _configured_agent(tmp_path, keychain)
    await agent.close()
    attached: list[dict[str, Any]] = []

    async def fake_probe(self: Agent, **kwargs: Any) -> PostgreSQLProbeResult:
        del self, kwargs
        return PostgreSQLProbeResult.build(
            (
                ("analytics", True),
                ("empty", False),
                ("reporting", True),
            ),
            truncated=True,
        )

    async def fake_attach(self: Agent, **kwargs: Any):
        del self
        attached.append(kwargs)
        return SimpleNamespace(display_name=kwargs["name"])

    monkeypatch.setattr(Agent, "probe_postgresql", fake_probe)
    monkeypatch.setattr(Agent, "attach_postgresql", fake_attach)
    output = io.StringIO()

    result = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(
            "3\n2\nWarehouse\ndb.example.test\n5432\nwarehouse\nreader\n\n1,3\n"
        ),
        output_stream=output,
        hidden_input=lambda prompt: "database-secret",
        keychain=keychain,
    )

    assert result == 0
    assert len(attached) == 1
    assert attached[0]["schemas"] == ("analytics", "reporting")
    reference = attached[0]["credential"]
    assert isinstance(reference, SecretReference)
    assert keychain.values[reference.name] == "database-secret"
    text = output.getvalue()
    assert "database-secret" not in text
    assert "analytics, reporting" in text
    assert "Connection validated" in text
    assert "more than 100 schemas exist; showing the first 100" in text
    assert "Schemas selected" in text
    assert "Discovering tables and relationships" in text
    assert "Stage 4 status" not in text


def test_postgresql_connection_url_parser_normalizes_supported_fields():
    connection = terminal._parse_postgresql_connection_url(
        "postgresql://reader%24:p%40ss%2Fword@[2001:db8::1]:6543/"
        "warehouse?sslmode=verify-full"
    )

    assert connection == (
        "2001:db8::1",
        6543,
        "warehouse",
        "reader$",
        "p@ss/word",
        "verify-full",
    )
    assert terminal._parse_postgresql_connection_url(
        "postgres://reader@db.example.test/warehouse"
    ) == (
        "db.example.test",
        5432,
        "warehouse",
        "reader",
        None,
        "require",
    )


@pytest.mark.parametrize(
    "connection_url",
    (
        "mysql://reader:secret@db.example.test/warehouse",
        "postgresql://reader:secret@db.example.test/",
        "postgresql://reader:secret@db.example.test:0/warehouse",
        "postgresql://reader:secret@db.example.test:99999/warehouse",
        "postgresql://reader:secret@one.example,two.example/warehouse",
        "postgresql://reader:secret@db.example.test/warehouse#fragment",
        "postgresql://reader:secret@db.example.test/warehouse?connect_timeout=10",
        "postgresql://reader:secret@db.example.test/warehouse?sslmode=invalid",
        "postgresql://reader:secret@db.example.test/warehouse"
        "?sslmode=require&sslmode=disable",
        "postgresql://reader:%ZZ@db.example.test/warehouse",
        "postgresql://reader:p%FF@db.example.test/warehouse",
    ),
)
def test_postgresql_connection_url_parser_rejects_unsupported_input(
    connection_url: str,
):
    with pytest.raises(ValueError, match="PostgreSQL connection URL is invalid"):
        terminal._parse_postgresql_connection_url(connection_url)


def test_postgresql_connection_url_prompt_retries_without_echoing_input():
    invalid_url = (
        "postgresql://reader:do-not-echo@db.example.test/warehouse"
        "?connect_timeout=10"
    )
    valid_url = "postgresql://reader@db.example.test/warehouse"
    answers = iter((invalid_url, valid_url))
    output = io.StringIO()

    connection = terminal._read_postgresql_connection_url(
        lambda prompt: next(answers),
        output,
    )

    assert connection == (
        "db.example.test",
        5432,
        "warehouse",
        "reader",
        None,
        "require",
    )
    assert output.getvalue() == terminal._POSTGRESQL_CONNECTION_URL_ERROR + "\n"
    assert invalid_url not in output.getvalue()


async def test_terminal_postgresql_connection_url_uses_existing_secure_attach_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keychain = _Keychain()
    agent = await _configured_agent(tmp_path, keychain)
    await agent.close()
    probes: list[dict[str, Any]] = []
    attached: list[dict[str, Any]] = []

    async def fake_probe(self: Agent, **kwargs: Any) -> PostgreSQLProbeResult:
        del self
        probes.append(kwargs)
        return PostgreSQLProbeResult.build(
            (
                ("analytics", True),
                ("reporting", True),
            )
        )

    async def fake_attach(self: Agent, **kwargs: Any):
        del self
        attached.append(kwargs)
        return SimpleNamespace(display_name=kwargs["name"])

    monkeypatch.setattr(Agent, "probe_postgresql", fake_probe)
    monkeypatch.setattr(Agent, "attach_postgresql", fake_attach)
    output = io.StringIO()
    connection_url = (
        "postgresql://reader%24:p%40ss%2Fword@db.example.test:6543/"
        "warehouse?sslmode=verify-full"
    )
    hidden_prompts: list[str] = []

    def hidden_input(prompt: str) -> str:
        hidden_prompts.append(prompt)
        return connection_url

    result = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO("3\n1\nWarehouse\n1,2\n"),
        output_stream=output,
        hidden_input=hidden_input,
        keychain=keychain,
    )

    assert result == 0
    assert hidden_prompts == ["Connection URL: "]
    assert len(probes) == 1
    assert len(attached) == 1
    for arguments in (probes[0], attached[0]):
        assert arguments["host"] == "db.example.test"
        assert arguments["port"] == 6543
        assert arguments["database"] == "warehouse"
        assert arguments["username"] == "reader$"
        assert arguments["ssl_mode"] == "verify-full"
    assert attached[0]["schemas"] == ("analytics", "reporting")
    assert attached[0]["name"] == "Warehouse"
    reference = attached[0]["credential"]
    assert isinstance(reference, SecretReference)
    assert probes[0]["credential"] == reference
    assert keychain.values[reference.name] == "p@ss/word"
    text = output.getvalue()
    assert connection_url not in text
    assert "p@ss/word" not in text
    assert "Connection URL" in text
    assert "Connection validated" in text


async def test_terminal_postgresql_url_without_password_prompts_for_password(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keychain = _Keychain()
    agent = await _configured_agent(tmp_path, keychain)
    await agent.close()
    attached: list[dict[str, Any]] = []

    async def fake_probe(self: Agent, **kwargs: Any) -> PostgreSQLProbeResult:
        del self, kwargs
        return PostgreSQLProbeResult.build((("analytics", True),))

    async def fake_attach(self: Agent, **kwargs: Any):
        del self
        attached.append(kwargs)
        return SimpleNamespace(display_name=kwargs["name"])

    monkeypatch.setattr(Agent, "probe_postgresql", fake_probe)
    monkeypatch.setattr(Agent, "attach_postgresql", fake_attach)
    answers = iter(
        (
            "postgresql://reader@db.example.test/warehouse",
            "separate-password",
        )
    )
    hidden_prompts: list[str] = []

    def hidden_input(prompt: str) -> str:
        hidden_prompts.append(prompt)
        return next(answers)

    result = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO("3\n1\nWarehouse\n1\n"),
        output_stream=io.StringIO(),
        hidden_input=hidden_input,
        keychain=keychain,
    )

    assert result == 0
    assert hidden_prompts == ["Connection URL: ", "Password: "]
    assert len(attached) == 1
    reference = attached[0]["credential"]
    assert keychain.values[reference.name] == "separate-password"


async def test_enhanced_postgresql_schema_toggling_attaches_stable_names_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keychain = _Keychain()
    agent: Agent | None = await _configured_agent(tmp_path, keychain)
    assert agent is not None
    attached: list[dict[str, Any]] = []

    async def fake_probe(self: Agent, **kwargs: Any) -> PostgreSQLProbeResult:
        del self, kwargs
        return PostgreSQLProbeResult.build(
            (
                ("analytics", True),
                ("empty", False),
                ("reporting", True),
            )
        )

    async def fake_attach(self: Agent, **kwargs: Any):
        del self
        attached.append(kwargs)
        return SimpleNamespace(display_name=kwargs["name"])

    monkeypatch.setattr(Agent, "probe_postgresql", fake_probe)
    monkeypatch.setattr(Agent, "attach_postgresql", fake_attach)
    output = io.StringIO()

    try:
        with create_pipe_input() as pipe:
            pipe.send_text("\x1b[B\r \x1b[B\x1b[B \r")
            await terminal._onboard_postgresql(
                agent,
                input_stream=io.StringIO(
                    "Warehouse\ndb.example.test\n5432\nwarehouse\nreader\n\n"
                ),
                output_stream=output,
                hidden_input=lambda prompt: "database-secret",
                selection_input=pipe,
                selection_output=DummyOutput(),
            )

        assert len(attached) == 1
        assert attached[0]["schemas"] == ("analytics", "reporting")
        assert "empty" not in attached[0]["schemas"]
        assert "information_schema" not in attached[0]["schemas"]
        assert "pg_catalog" not in attached[0]["schemas"]
        reference = attached[0]["credential"]
        assert keychain.values[reference.name] == "database-secret"
        assert "Schemas selected: analytics, reporting" in output.getvalue()
        assert "Space toggle" not in output.getvalue()

        database = tmp_path / "transcript-check.sqlite"
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE records (id INTEGER PRIMARY KEY)")
        await agent.attach(SQLiteSource(database, name="Transcript check"))
        await agent.close()
        agent = None

        provider = _provider()
        profile = ModelProfile(
            id=provider.provider_id,
            context_window_tokens=8_192,
            max_output_tokens=256,
            supports_tools=True,
        )
        agent = await Agent.open(
            "atlas",
            root=tmp_path,
            model=provider,
            model_profile=profile,
            keychain=keychain,
        )
        result = await agent.run("Check transcript isolation")
        transcript = await agent.transcript(result.run_id)
        transcript_text = "\n".join(
            block.text
            for message in transcript.messages
            for block in message.content
            if isinstance(block, TextBlock)
        )
        assert "Select one or more schemas" not in transcript_text
        assert "Space toggle" not in transcript_text
        assert "Schemas (comma-separated numbers)" not in transcript_text
    finally:
        if agent is not None:
            await agent.close()


@pytest.mark.parametrize(
    ("cancel_key", "expected_code"),
    (
        ("\x1b", 0),
        ("\x03", 130),
        ("\x04", 0),
    ),
)
async def test_enhanced_postgresql_schema_cancellation_cleans_up_and_allows_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel_key: str,
    expected_code: int,
):
    keychain = _Keychain()
    agent = await _configured_agent(tmp_path, keychain)
    await agent.close()

    async def fake_probe(self: Agent, **kwargs: Any) -> PostgreSQLProbeResult:
        del self, kwargs
        return PostgreSQLProbeResult.build(
            (
                ("analytics", True),
                ("reporting", True),
            )
        )

    async def forbidden_attach(self: Agent, **kwargs: Any):
        raise AssertionError((self, kwargs))

    monkeypatch.setattr(Agent, "probe_postgresql", fake_probe)
    monkeypatch.setattr(Agent, "attach_postgresql", forbidden_attach)
    output = io.StringIO()

    with create_pipe_input() as pipe:
        task = asyncio.create_task(
            run_terminal_application(
                root=tmp_path,
                input_stream=io.StringIO(
                    "Warehouse\ndb.example.test\n5432\nwarehouse\nreader\n\n"
                ),
                output_stream=output,
                hidden_input=lambda prompt: "database-secret",
                keychain=keychain,
                selection_input=pipe,
                selection_output=DummyOutput(),
            )
        )
        pipe.send_text("\x1b[B\x1b[B\r\x1b[B\r")
        for _ in range(100):
            if "Connection validated" in output.getvalue():
                break
            await asyncio.sleep(0.01)
        assert "Connection validated" in output.getvalue()
        pipe.send_text(cancel_key)
        result = await task

    assert result == expected_code
    assert all("postgresql" not in account for account in keychain.values)
    assert [
        action for action, account in keychain.events if "postgresql" in account
    ] == ["set", "delete"]

    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    try:
        assert await reopened.list_sources() == ()
        assert await reopened.list_catalog_resources() == ()
    finally:
        await reopened.close()

    database = tmp_path / f"clean-{ord(cancel_key)}.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE records (id INTEGER PRIMARY KEY)")
    clean_result = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(f"1\n{database}\nClean retry\n"),
        output_stream=io.StringIO(),
        hidden_input=lambda prompt: (_ for _ in ()).throw(
            AssertionError(f"unexpected hidden prompt: {prompt}")
        ),
        keychain=keychain,
    )

    assert clean_result == 0
    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    try:
        sources = await reopened.list_sources()
        assert len(sources) == 1
        assert sources[0].display_name == "Clean retry"
    finally:
        await reopened.close()


async def test_terminal_postgresql_probe_failure_cleans_secret_and_persists_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keychain = _Keychain()
    agent = await _configured_agent(tmp_path, keychain)
    await agent.close()
    diagnostic = "server says password=database-secret"

    async def failed_probe(self: Agent, **kwargs: Any) -> PostgreSQLProbeResult:
        del self, kwargs
        error = PostgreSQLSourceError(
            "postgresql_connect_failed",
            "PostgreSQL source could not be opened.",
        )
        error.add_note(diagnostic)
        raise error

    monkeypatch.setattr(Agent, "probe_postgresql", failed_probe)
    output = io.StringIO()
    connection_url = "postgresql://reader:database-secret@db.example.test/warehouse"

    result = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO("3\n1\nWarehouse\n"),
        output_stream=output,
        hidden_input=lambda prompt: connection_url,
        keychain=keychain,
    )

    assert result == 0
    assert "Could not connect to PostgreSQL" in output.getvalue()
    assert diagnostic not in output.getvalue()
    assert "database-secret" not in output.getvalue()
    assert connection_url not in output.getvalue()
    assert all("postgresql" not in account for account in keychain.values)
    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    try:
        assert await reopened.list_sources() == ()
        assert await reopened.list_catalog_resources() == ()
    finally:
        await reopened.close()


async def test_terminal_postgresql_selection_eof_cleans_secret_and_releases_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keychain = _Keychain()
    agent = await _configured_agent(tmp_path, keychain)
    await agent.close()

    async def fake_probe(self: Agent, **kwargs: Any) -> PostgreSQLProbeResult:
        del self, kwargs
        return PostgreSQLProbeResult.build((("analytics", True),))

    monkeypatch.setattr(Agent, "probe_postgresql", fake_probe)

    result = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(
            "3\n2\nWarehouse\ndb.example.test\n5432\nwarehouse\nreader\n\n"
        ),
        output_stream=io.StringIO(),
        hidden_input=lambda prompt: "database-secret",
        keychain=keychain,
    )

    assert result == 0
    assert all("postgresql" not in account for account in keychain.values)
    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    await reopened.close()


async def test_terminal_postgresql_hidden_password_interrupt_stores_nothing(
    tmp_path: Path,
):
    keychain = _Keychain()
    agent = await _configured_agent(tmp_path, keychain)
    await agent.close()

    result = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(
            "3\n2\nWarehouse\ndb.example.test\n5432\nwarehouse\nreader\n\n"
        ),
        output_stream=io.StringIO(),
        hidden_input=lambda prompt: (_ for _ in ()).throw(KeyboardInterrupt),
        keychain=keychain,
    )

    assert result == 130
    assert all("postgresql" not in account for account in keychain.values)
    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    await reopened.close()


async def test_terminal_normalizes_database_keychain_failure_without_raw_diagnostics(
    tmp_path: Path,
):
    class FailingKeychain(_Keychain):
        fail = False

        async def set(self, reference: SecretReference, value: str) -> None:
            if self.fail:
                raise SecretResolutionError(
                    "secret_provider_unavailable",
                    "raw database keychain diagnostic containing database-secret",
                )
            await super().set(reference, value)

    keychain = FailingKeychain()
    agent = await _configured_agent(tmp_path, keychain)
    await agent.close()
    keychain.fail = True
    output = io.StringIO()

    result = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(
            "3\n2\nWarehouse\ndb.example.test\n5432\nwarehouse\nreader\n\n"
        ),
        output_stream=output,
        hidden_input=lambda prompt: "database-secret",
        keychain=keychain,
    )

    assert result == 0
    assert (
        "The database password could not be saved to the OS keychain. "
        "Check keychain access and retry."
    ) in output.getvalue()
    assert "raw database keychain diagnostic" not in output.getvalue()
    assert "database-secret" not in output.getvalue()


async def test_terminal_skips_source_onboarding_when_an_active_source_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keychain = _Keychain()
    agent = await _configured_agent(tmp_path, keychain)
    source_path = tmp_path / "existing.sqlite"
    with sqlite3.connect(source_path) as connection:
        connection.execute("CREATE TABLE records (id INTEGER PRIMARY KEY)")
    await agent.attach(SQLiteSource(source_path, name="Existing"))
    await agent.close()

    async def forbidden_probe(self: Agent, **kwargs: Any):
        raise AssertionError((self, kwargs))

    monkeypatch.setattr(Agent, "probe_postgresql", forbidden_probe)
    output = io.StringIO()

    result = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(""),
        output_stream=output,
        hidden_input=lambda prompt: (_ for _ in ()).throw(
            AssertionError(f"unexpected hidden prompt: {prompt}")
        ),
        keychain=keychain,
    )

    assert result == 0
    assert "Select a data source" not in output.getvalue()
    assert "Connections  1" in output.getvalue()
    assert "Existing" not in output.getvalue()
    assert "Stage 4 status" not in output.getvalue()
    assert "Status" in output.getvalue() and "Ready" in output.getvalue()


async def test_empty_attachment_enters_repair_and_never_reports_ready(tmp_path: Path):
    keychain = _Keychain()
    agent = await _configured_agent(tmp_path, keychain)
    await agent.close()
    database = tmp_path / "empty.sqlite"
    with sqlite3.connect(database):
        pass
    output = io.StringIO()

    result = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(f"1\n{database}\nEmpty source\n2\n"),
        output_stream=output,
        hidden_input=lambda prompt: (_ for _ in ()).throw(
            AssertionError(f"unexpected hidden prompt: {prompt}")
        ),
        keychain=keychain,
    )

    assert result == 0
    text = output.getvalue()
    assert "Stage 4 status" not in text
    assert "0 tables" in text
    assert "\nNot ready\n" in text
    assert "No supported tables or resources were discovered" in text
    assert "Add or retry a supported source" in text
    assert "You › " not in text
    assert "Status  Ready" not in text


async def test_empty_repair_can_add_another_source_and_recompute_readiness(
    tmp_path: Path,
):
    keychain = _Keychain()
    agent = await _configured_agent(tmp_path, keychain)
    await agent.close()
    empty = tmp_path / "empty.sqlite"
    ready = tmp_path / "ready.sqlite"
    with sqlite3.connect(empty):
        pass
    with sqlite3.connect(ready) as connection:
        connection.execute("CREATE TABLE records (id INTEGER PRIMARY KEY)")
    output = io.StringIO()

    result = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(f"1\n{empty}\nEmpty\n1\n1\n{ready}\nReady source\n"),
        output_stream=output,
        hidden_input=lambda prompt: (_ for _ in ()).throw(
            AssertionError(f"unexpected hidden prompt: {prompt}")
        ),
        keychain=keychain,
    )

    assert result == 0
    text = output.getvalue()
    assert "Not ready" in text
    assert "Connections  2" in text
    assert "Catalog   1 resource" in text
    assert "Status" in text and "Ready" in text


async def test_stage_four_sanitizes_and_bounds_untrusted_source_labels(tmp_path: Path):
    keychain = _Keychain()
    agent = await _configured_agent(tmp_path, keychain)
    database = tmp_path / "unsafe.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE records (id INTEGER PRIMARY KEY)")
    unsafe = "unsafe\x1b[31m\u202e" + ("x" * 300)
    await agent.attach(SQLiteSource(database, name=unsafe))
    await agent.close()
    output = io.StringIO()

    result = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(""),
        output_stream=output,
        hidden_input=lambda prompt: (_ for _ in ()).throw(
            AssertionError(f"unexpected hidden prompt: {prompt}")
        ),
        keychain=keychain,
    )

    assert result == 0
    text = output.getvalue()
    assert "\x1b" not in text
    assert "\u202e" not in text
    assert "x" * 129 not in text
    assert "Status" in text and "Ready" in text


async def test_repair_eof_releases_lock_and_commits_no_extra_catalog_truth(
    tmp_path: Path,
):
    keychain = _Keychain()
    agent = await _configured_agent(tmp_path, keychain)
    database = tmp_path / "empty-existing.sqlite"
    with sqlite3.connect(database):
        pass
    await agent.attach(SQLiteSource(database, name="Empty existing"))
    before = await agent.catalog_summary()
    await agent.close()

    result = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(""),
        output_stream=io.StringIO(),
        hidden_input=lambda prompt: (_ for _ in ()).throw(
            AssertionError(f"unexpected hidden prompt: {prompt}")
        ),
        keychain=keychain,
    )

    assert result == 0
    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    try:
        assert await reopened.catalog_summary() == before
        assert len(await reopened.list_sources()) == 1
    finally:
        await reopened.close()
