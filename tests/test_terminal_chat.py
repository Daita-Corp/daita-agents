from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, TextIO, cast

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from daita import Agent, ApprovalDecision, ApprovalRequest, Skill, SQLiteSource
from daita import terminal, terminal_tui
from daita._json import FrozenJsonObject
from daita.llm.errors import ModelProviderError, ProviderErrorCode
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import LoopExitKind
from daita.security import SecretReference
from daita.terminal import run_terminal_application
import daita.hosting.embedded as embedded


class _Keychain:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def resolve(self, reference: SecretReference) -> str:
        return self.values[reference.name]

    async def set(self, reference: SecretReference, value: str) -> None:
        self.values[reference.name] = value

    async def delete(self, reference: SecretReference) -> None:
        self.values.pop(reference.name, None)


def _stop(text: str, *, provider_id: str = "openai:test-model") -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        text=text,
        provider_id=provider_id,
    )


def _validation_response(provider_id: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCall(
                id="validation-call",
                name="daita_validate_tool_support",
                arguments={},
            ),
        ),
        provider_id=provider_id,
    )


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=8_192,
        max_output_tokens=256,
        supports_tools=True,
    )


def _request_text(request: ModelRequest) -> tuple[str, ...]:
    return tuple(
        block.text
        for message in request.messages
        if message.role is not MessageRole.SYSTEM
        for block in message.content
        if isinstance(block, TextBlock)
    )


def _database(path: Path, *tables: str) -> None:
    with sqlite3.connect(path) as connection:
        for table in tables:
            connection.execute(
                f'CREATE TABLE "{table}" (id INTEGER PRIMARY KEY)'  # noqa: S608
            )


async def _ready_agent(
    root: Path,
    keychain: _Keychain,
    *,
    name: str = "atlas",
    source_name: str = "Ready SQLite",
    table: str = "records",
) -> tuple[Path, str]:
    database = root / f"{name}.sqlite"
    _database(database, table)
    validator = MockModelProvider(
        (_validation_response("openai:test-model"),),
        provider_id="openai:test-model",
    )
    agent = await Agent.create(
        name,
        root=root,
        keychain=keychain,
        model_validator=validator,
    )
    await agent.configure_model(
        provider="openai",
        model="test-model",
        api_key="model-secret",
        context_window_tokens=128_000,
        max_output_tokens=4_096,
    )
    registration = await agent.attach(SQLiteSource(database, name=source_name))
    await agent.close()
    return database, registration.id


def _install_provider(
    monkeypatch: pytest.MonkeyPatch,
    provider: MockModelProvider | Any,
) -> None:
    monkeypatch.setattr(
        embedded,
        "create_model_route_provider",
        lambda route, *, secret_provider=None: provider,
    )


async def test_ready_agent_enters_chat_and_explicitly_continues_one_conversation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keychain = _Keychain()
    await _ready_agent(tmp_path, keychain)
    provider = MockModelProvider(
        (_stop("first answer"), _stop("follow-up answer")),
        provider_id="openai:test-model",
    )
    _install_provider(monkeypatch, provider)
    output = io.StringIO()

    code = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO("first question\nfollow-up question\n/exit\n"),
        output_stream=output,
        keychain=keychain,
    )

    assert code == 0
    text = output.getvalue()
    assert "Status" in text and "Ready" in text
    assert text.count("Agent") == 1
    assert "atlas" in text
    assert text.count("OpenAI · test-model · configured") == 1
    assert "provider health was not checked this launch" not in text
    assert "Stage 2 status" not in text
    assert "Stage 4 status" not in text
    assert "Conversation" in text
    assert "new" in text
    assert "You › " in text
    assert "first answer" in text
    assert "follow-up answer" in text
    assert _request_text(provider.requests[0]) == ("first question",)
    assert _request_text(provider.requests[1]) == (
        "first question",
        "first answer",
        "follow-up question",
    )


async def test_ready_tui_does_not_print_redundant_startup_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keychain = _Keychain()
    await _ready_agent(tmp_path, keychain)
    _install_provider(
        monkeypatch,
        MockModelProvider((), provider_id="openai:test-model"),
    )
    entered: list[terminal_tui.TerminalViewState] = []

    async def fake_tui(
        state: terminal_tui.TerminalViewState,
        **kwargs: Any,
    ) -> terminal_tui.TerminalApplicationResult:
        del kwargs
        entered.append(state)
        return terminal_tui.TerminalApplicationResult(None, "exit")

    monkeypatch.setattr(terminal_tui, "run_terminal_tui", fake_tui)
    output = io.StringIO()

    code = await run_terminal_application(
        root=tmp_path,
        agent_name="atlas",
        input_stream=io.StringIO(),
        output_stream=output,
        keychain=keychain,
        tui_input=object(),
        tui_output=object(),
    )

    assert code == 0
    assert output.getvalue() == ""
    assert len(entered) == 1
    assert entered[0].agent_label == "atlas"
    assert entered[0].model_label == "test-model"
    assert entered[0].source_summary == "Ready SQLite"
    assert entered[0].context_capacity_tokens == 123_904


async def test_model_answers_preserve_lines_and_neutralize_terminal_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keychain = _Keychain()
    await _ready_agent(tmp_path, keychain)
    answer = (
        "Summary:\n"
        "- first\n"
        "- second\n"
        "ANSI \x1b[2J OSC \x1b]0;unsafe\x07 "
        "rewrite\rhidden bidi\u202e café 東京"
    )
    provider = MockModelProvider(
        (_stop(answer),),
        provider_id="openai:test-model",
    )
    _install_provider(monkeypatch, provider)
    output = io.StringIO()

    code = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO("show a summary\n/exit\n"),
        output_stream=output,
        keychain=keychain,
    )

    assert code == 0
    text = output.getvalue()
    assert "Summary:\n- first\n- second\n" in text
    assert "café 東京" in text
    assert "\x1b" not in text
    assert "\x07" not in text
    assert "\r" not in text
    assert "\u202e" not in text
    assert "ANSI ?[2J OSC ?]0;unsafe? rewrite?hidden bidi?" in text


def test_plain_terminal_model_answer_is_not_silently_truncated():
    answer = "ANSWER_START\n" + ("x" * 20_000) + "\nANSWER_END"

    rendered = terminal._render_model_answer(answer, maximum=None)

    assert rendered == answer


async def test_exact_approval_json_is_complete_reversible_and_terminal_safe():
    prefix = 'Quoted "line"\ncontrols:\x1b[2J\x00\u202e'
    instructions = prefix + "x" * (12_000 - len(prefix))
    arguments = FrozenJsonObject.from_mapping(
        {
            "description": "Monthly revenue",
            "instructions": instructions,
            "name": "monthly-revenue",
        }
    )
    request = ApprovalRequest(
        run_id="run-exact",
        call_id="call-exact",
        tool_name="skill_save",
        capability_id="skills.write",
        arguments=arguments,
        reason="write",
    )
    original_arguments = request.arguments
    output = io.StringIO()

    decision = await terminal._prompt_for_exact_approval(
        request,
        input_stream=io.StringIO("y\n"),
        output_stream=output,
    )

    assert decision is ApprovalDecision.APPROVE
    assert request.arguments is original_arguments
    text = output.getvalue()
    rendered = text.split("Arguments:\n", 1)[1].split(
        "\nApprove this exact change once?",
        1,
    )[0]
    assert json.loads(rendered) == request.arguments.to_dict()
    assert len(json.loads(rendered)["instructions"]) == 12_000
    assert "\\n" in rendered
    assert "\\u001b" in rendered
    assert "\\u0000" in rendered
    assert "\\u202e" in rendered
    assert "\x1b" not in rendered
    assert "\x00" not in rendered
    assert "\u202e" not in rendered


async def test_unreviewable_approval_is_denied_before_prompting(
    monkeypatch: pytest.MonkeyPatch,
):
    request = ApprovalRequest(
        run_id="run-oversized",
        call_id="call-oversized",
        tool_name="future_write",
        capability_id="future.write",
        arguments=FrozenJsonObject.from_mapping(
            {"content": "x" * (terminal._MAX_APPROVAL_DOCUMENT_CHARACTERS + 1)}
        ),
        reason="write",
    )
    output = io.StringIO()
    input_stream = io.StringIO()

    def forbidden_readline(size: int = -1, /) -> str:
        raise AssertionError((size, "approval prompt must not be read"))

    monkeypatch.setattr(input_stream, "readline", forbidden_readline)

    decision = await terminal._prompt_for_exact_approval(
        request,
        input_stream=input_stream,
        output_stream=output,
    )

    assert decision is ApprovalDecision.DENY
    assert "exact arguments exceed the terminal review bound" in output.getvalue()
    assert "Approve this exact change once?" not in output.getvalue()


@pytest.mark.parametrize(
    ("command", "confirmation", "setup_name"),
    (
        ("/source add\n", "", "_onboard_source"),
        ("/model\n", "y\n", "_onboard_model"),
    ),
)
async def test_live_chat_escape_from_nested_setup_returns_to_chat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    confirmation: str,
    setup_name: str,
):
    keychain = _Keychain()
    await _ready_agent(tmp_path, keychain)
    provider = MockModelProvider((), provider_id="openai:test-model")
    _install_provider(monkeypatch, provider)

    async def cancelled(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise terminal.SelectionCancelled

    monkeypatch.setattr(terminal, setup_name, cancelled)
    output = io.StringIO()

    code = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(command + confirmation + "/status\n/exit\n"),
        output_stream=output,
        keychain=keychain,
    )

    assert code == 0
    text = output.getvalue()
    assert "cancelled; returning to chat" in text
    assert "Setup cancelled." not in text
    assert text.count("You › ") >= 3
    assert provider.requests == ()


async def test_live_chat_schema_escape_removes_temporary_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keychain = _Keychain()
    await _ready_agent(tmp_path, keychain)
    agent = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    before_credentials = dict(keychain.values)

    async def fake_probe(self: Agent, **kwargs: Any):
        del self, kwargs
        return terminal.PostgreSQLProbeResult.build((("analytics", True),))

    async def forbidden_attach(self: Agent, **kwargs: Any):
        raise AssertionError((self, kwargs))

    monkeypatch.setattr(Agent, "probe_postgresql", fake_probe)
    monkeypatch.setattr(Agent, "attach_postgresql", forbidden_attach)
    output = io.StringIO()
    try:
        with create_pipe_input() as pipe:
            command = asyncio.create_task(
                terminal._handle_local_command(
                    "/source add",
                    agent=agent,
                    root=tmp_path,
                    input_stream=io.StringIO(
                        "Warehouse\ndb.example.test\n5432\nwarehouse\nreader\n\n"
                    ),
                    output_stream=output,
                    hidden_input=lambda prompt: "temporary-database-secret",
                    keychain=keychain,
                    model_validator=None,
                    approval_handler=None,
                    conversation_id="conversation-existing",
                    validated=False,
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
            pipe.send_text("\x1b")
            same_agent, conversation_id, action = await command

        assert same_agent is agent
        assert conversation_id == "conversation-existing"
        assert action is None
        assert keychain.values == before_credentials
        assert "Source setup cancelled; returning to chat." in output.getvalue()
        assert "temporary-database-secret" not in output.getvalue()
    finally:
        await agent.close()


async def test_empty_catalog_stays_in_repair_and_never_enters_chat(
    tmp_path: Path,
):
    keychain = _Keychain()
    database, _ = await _ready_agent(tmp_path, keychain)
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TABLE records")
    agent = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    source = (await agent.list_sources())[0]
    await agent.refresh_source(source.id)
    await agent.close()
    output = io.StringIO()

    code = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO("2\n"),
        output_stream=output,
        keychain=keychain,
    )

    assert code == 0
    assert "\nNot ready\n" in output.getvalue()
    assert "You › " not in output.getvalue()


async def test_new_and_resume_change_only_terminal_conversation_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keychain = _Keychain()
    await _ready_agent(tmp_path, keychain)
    provider = MockModelProvider(
        (_stop("seed answer"), _stop("continued answer"), _stop("new answer")),
        provider_id="openai:test-model",
    )
    _install_provider(monkeypatch, provider)
    first_output = io.StringIO()
    await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO("seed question\n/exit\n"),
        output_stream=first_output,
        keychain=keychain,
    )
    match = re.search(
        r"Conversation  (conversation-[A-Za-z0-9]+)", first_output.getvalue()
    )
    assert match is not None
    conversation_id = match.group(1)

    other_provider = MockModelProvider(
        (_stop("other answer", provider_id="mock:other"),),
        provider_id="mock:other",
    )
    other = await Agent.create(
        "other",
        root=tmp_path,
        model=other_provider,
        model_profile=_profile(other_provider),
    )
    cross_agent_id = (await other.run("other question")).conversation_id
    await other.close()

    second_output = io.StringIO()
    await run_terminal_application(
        root=tmp_path,
        agent_name="atlas",
        input_stream=io.StringIO(
            f"/resume {cross_agent_id}\n"
            f"/resume {conversation_id}\n"
            "continued question\n"
            "/new\n"
            "new question\n"
            "/exit\n"
        ),
        output_stream=second_output,
        keychain=keychain,
    )

    text = second_output.getvalue()
    assert "Cannot resume conversation: unknown conversation for this agent" in text
    assert f"Conversation  {conversation_id}" in text
    assert _request_text(provider.requests[1]) == (
        "seed question",
        "seed answer",
        "continued question",
    )
    assert _request_text(provider.requests[2]) == ("new question",)

    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    try:
        assert await reopened.conversation_exists(conversation_id) is True
        assert await reopened.conversation_exists(cross_agent_id) is False
    finally:
        await reopened.close()


async def test_local_status_commands_are_bounded_secret_free_and_never_modeled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keychain = _Keychain()
    unsafe = "unsafe\x1b[31m\u202e" + ("x" * 300)
    await _ready_agent(
        tmp_path,
        keychain,
        source_name=unsafe,
        table="resource_" + ("z" * 220),
    )
    provider = MockModelProvider((), provider_id="openai:test-model")
    _install_provider(monkeypatch, provider)
    output = io.StringIO()

    code = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(
            "/catalog\n/sources\n/settings\n/help\n/memory\n/user\n/skills\n/exit\n"
        ),
        output_stream=output,
        keychain=keychain,
    )

    assert code == 0
    text = output.getvalue()
    assert "Catalog preview" in text
    assert "Sources" in text
    assert "Settings" in text
    assert "Commands" in text
    assert "Wheel or Page Up/Page Down review" in text
    assert "Esc Esc clear input" in text
    assert "Animated status shows the active tool" in text
    assert "Approvals accept only Y or N" in text
    assert "copy request was sent, not that it succeeded" in text
    assert "terminal's bypass modifier (often Shift)" in text
    assert "\x1b" not in text
    assert "\u202e" not in text
    assert "x" * 129 not in text
    assert "z" * 129 not in text
    assert "model-secret" not in text
    assert "credential_ref" not in text
    assert "keychain://" not in text
    assert all(account not in text for account in keychain.values)
    assert provider.requests == ()


async def test_source_add_recomputes_catalog_before_returning_to_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keychain = _Keychain()
    await _ready_agent(tmp_path, keychain)
    second = tmp_path / "second.sqlite"
    _database(second, "second_table")
    provider = MockModelProvider((), provider_id="openai:test-model")
    _install_provider(monkeypatch, provider)
    output = io.StringIO()

    code = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(
            f"/source add\n1\n{second}\nSecond source\n/sources\n/exit\n"
        ),
        output_stream=output,
        keychain=keychain,
    )

    assert code == 0
    text = output.getvalue()
    assert "Second source" in text
    assert "2 resources" in text
    assert text.count("You › ") >= 2
    assert provider.requests == ()


async def test_source_use_and_one_question_override_are_local_and_persisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keychain = _Keychain()
    await _ready_agent(tmp_path, keychain)
    second = tmp_path / "second.sqlite"
    _database(second, "second_table")
    setup = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    try:
        second_source = await setup.attach(SQLiteSource(second, name="Second source"))
    finally:
        await setup.close()
    provider = MockModelProvider(
        (
            _stop("first answer"),
            _stop("second answer"),
            _stop("override answer"),
        ),
        provider_id="openai:test-model",
    )
    _install_provider(monkeypatch, provider)
    output = io.StringIO()

    code = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(
            "first question\n"
            "/source use Second source\n"
            "/sources\n"
            "second question\n"
            '@"Ready SQLite" override question\n'
            "/exit\n"
        ),
        output_stream=output,
        keychain=keychain,
    )

    assert code == 0
    text = output.getvalue()
    assert "Source  Second source" in text
    assert "Started a new conversation to keep source context isolated." in text
    assert "● Second source" in text
    assert "○ Ready SQLite" in text
    assert _request_text(provider.requests[0]) == ("first question",)
    assert _request_text(provider.requests[1]) == ("second question",)
    assert _request_text(provider.requests[2]) == ("override question",)
    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    try:
        assert await reopened.active_source() == second_source
    finally:
        await reopened.close()


async def test_source_refresh_commits_new_truth_before_the_next_model_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keychain = _Keychain()
    database, source_id = await _ready_agent(tmp_path, keychain)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE refreshed_table (id INTEGER PRIMARY KEY)")
    provider = MockModelProvider(
        (_stop("grounded answer"),),
        provider_id="openai:test-model",
    )
    _install_provider(monkeypatch, provider)
    output = io.StringIO()

    code = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(
            f"/source refresh {source_id}\n"
            "/catalog\n"
            "What is in refreshed_table?\n"
            "/exit\n"
        ),
        output_stream=output,
        keychain=keychain,
    )

    assert code == 0
    assert "refreshed_table" in output.getvalue()
    assert "refreshed_table" in repr(provider.requests[0])


async def test_empty_refresh_returns_to_catalog_repair_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keychain = _Keychain()
    database, source_id = await _ready_agent(tmp_path, keychain)
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TABLE records")
    provider = MockModelProvider((), provider_id="openai:test-model")
    _install_provider(monkeypatch, provider)
    output = io.StringIO()

    code = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(f"/source refresh {source_id}\n2\n"),
        output_stream=output,
        keychain=keychain,
    )

    assert code == 0
    text = output.getvalue()
    assert text.count("You › ") == 1
    assert "\nNot ready\n" in text
    assert "Add or retry a supported source" in text
    assert provider.requests == ()


async def test_failed_source_refresh_preserves_previous_committed_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keychain = _Keychain()
    database, source_id = await _ready_agent(tmp_path, keychain)
    before_agent = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    before = await before_agent.catalog_summary()
    await before_agent.close()
    database.unlink()
    provider = MockModelProvider((), provider_id="openai:test-model")
    _install_provider(monkeypatch, provider)
    output = io.StringIO()

    code = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(f"/source refresh {source_id}\n/catalog\n/exit\n"),
        output_stream=output,
        keychain=keychain,
    )

    assert code == 0
    assert "without replacing committed catalog truth" in output.getvalue()
    assert "records" in output.getvalue()
    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    try:
        assert await reopened.catalog_summary() == before
    finally:
        await reopened.close()


async def test_model_change_reopens_and_uses_replacement_on_the_next_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keychain = _Keychain()
    await _ready_agent(tmp_path, keychain)
    old_provider = MockModelProvider((), provider_id="openai:test-model")
    new_provider = MockModelProvider(
        (_stop("replacement answer", provider_id="openai:new-model"),),
        provider_id="openai:new-model",
    )

    def provider_for(route: Any, *, secret_provider: Any = None):
        del secret_provider
        provider_id = route.candidates[0].provider_id
        return new_provider if provider_id == "openai:new-model" else old_provider

    monkeypatch.setattr(embedded, "create_model_route_provider", provider_for)
    validator = MockModelProvider(
        (_validation_response("openai:new-model"),),
        provider_id="openai:new-model",
    )
    output = io.StringIO()

    code = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(
            "/model\n"
            "y\n"
            "1\n"
            "4\n"
            "new-model\n"
            "128000\n"
            "4096\n"
            "question for replacement\n"
            "/exit\n"
        ),
        output_stream=output,
        hidden_input=lambda prompt: "replacement-secret",
        keychain=keychain,
        model_validator=validator,
    )

    assert code == 0
    assert "replacement answer" in output.getvalue()
    assert old_provider.requests == ()
    assert _request_text(new_provider.requests[0]) == ("question for replacement",)
    assert "replacement-secret" not in output.getvalue()


async def test_failed_model_change_keeps_previous_committed_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keychain = _Keychain()
    await _ready_agent(tmp_path, keychain)
    old_provider = MockModelProvider((), provider_id="openai:test-model")
    _install_provider(monkeypatch, old_provider)
    validator = MockModelProvider(
        (ModelProviderError(ProviderErrorCode.AUTHENTICATION_ERROR),),
        provider_id="openai:rejected-model",
    )
    output = io.StringIO()

    code = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO("/model\ny\n1\n4\nrejected-model\n128000\n4096\n"),
        output_stream=output,
        hidden_input=lambda prompt: "rejected-secret",
        keychain=keychain,
        model_validator=validator,
    )

    assert code == 0
    assert "API key was rejected" in output.getvalue()
    assert "rejected-secret" not in output.getvalue()
    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    try:
        assert reopened.model_route is not None
        assert reopened.model_route.candidates[0].provider_id == "openai:test-model"
    finally:
        await reopened.close()


async def test_approval_and_knowledge_commands_reuse_existing_local_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keychain = _Keychain()
    await _ready_agent(tmp_path, keychain)
    agent = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    await agent.set_user_profile("Initial user profile")
    await agent.save_skill("monthly-revenue", "Description", "Instructions")
    await agent.close()
    proposed = "Revenue uses paid invoice date."
    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="memory_set",
                        arguments={"target": "memory", "content": proposed},
                    ),
                ),
            ),
            _stop("saved"),
        ),
        provider_id="openai:test-model",
    )
    _install_provider(monkeypatch, provider)
    output = io.StringIO()

    code = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(
            "remember this\n"
            "y\n"
            "/memory\n"
            "/user\n"
            "/skills\n"
            "/skills show monthly-revenue\n"
            "/exit\n"
        ),
        output_stream=output,
        keychain=keychain,
    )

    assert code == 0
    text = output.getvalue()
    assert "Approve this exact change once?" in text
    assert proposed in text
    assert "Initial user profile" in text
    assert "Instructions" in text
    conversation_match = re.search(
        r"Conversation  (conversation-[A-Za-z0-9]+)",
        text,
    )
    assert conversation_match is not None
    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    try:
        assert await reopened.read_memory() == proposed
        transcript = await reopened.transcript(
            (await reopened.conversation_runs(conversation_match.group(1)))[
                0
            ].transcript.run.id
        )
        assert all(
            not (isinstance(block, TextBlock) and block.text.startswith(("/", "y")))
            for message in transcript.messages
            for block in message.content
        )
    finally:
        await reopened.close()


async def test_skills_create_opens_canonical_template_and_saves_without_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keychain = _Keychain()
    await _ready_agent(tmp_path, keychain)
    provider = MockModelProvider((), provider_id="openai:test-model")
    _install_provider(monkeypatch, provider)
    edited_document = (
        "# customer-health-investigation\n\n"
        "Investigate customer health using the attached warehouse.\n\n"
        "## Instructions\n\n"
        "Start with account scope, then compare support and usage signals.\n"
    )
    opened: list[str] = []

    def edit_document(seed: str, *, agent_home: Path) -> str:
        assert agent_home.name == "atlas"
        opened.append(seed)
        return edited_document

    monkeypatch.setattr(terminal, "_edit_document", edit_document)
    output = io.StringIO()

    code = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(
            "/skills create customer-health-investigation\n/skills\n/exit\n"
        ),
        output_stream=output,
        keychain=keychain,
    )

    assert code == 0
    assert provider.requests == ()
    assert opened == [
        "# customer-health-investigation\n\n"
        "Describe when the agent should use this skill.\n\n"
        "## Instructions\n\n"
        "Write the reusable procedure here.\n"
    ]
    assert "Invoke it with /customer-health-investigation" in output.getvalue()
    assert "/customer-health-investigation:" in output.getvalue()
    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    try:
        skill = await reopened.read_skill("customer-health-investigation")
        assert skill is not None
        assert skill.description == (
            "Investigate customer health using the attached warehouse."
        )
    finally:
        await reopened.close()


async def test_skills_create_without_name_runs_guided_creation_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keychain = _Keychain()
    await _ready_agent(tmp_path, keychain)
    provider = MockModelProvider((), provider_id="openai:test-model")
    _install_provider(monkeypatch, provider)
    output = io.StringIO()

    code = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(
            "/skills create\n"
            "customer-health-investigation\n"
            "Investigate customer health using cross-schema evidence.\n"
            "Inspect the health snapshot first.\n"
            "Validate it against orders, billing, and support.\n"
            ".\n"
            "/exit\n"
        ),
        output_stream=output,
        keychain=keychain,
    )

    assert code == 0
    assert provider.requests == ()
    text = output.getvalue()
    assert "Name: " in text
    assert "Description: " in text
    assert "finish with a single . on its own line" in text
    assert "Invoke it with /customer-health-investigation" in text
    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    try:
        skill = await reopened.read_skill("customer-health-investigation")
        assert skill == Skill(
            "customer-health-investigation",
            "Investigate customer health using cross-schema evidence.",
            (
                "Inspect the health snapshot first.\n"
                "Validate it against orders, billing, and support."
            ),
        )
    finally:
        await reopened.close()


async def test_skills_create_reopens_invalid_draft_without_losing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    agent = await Agent.create("draft-retry", root=tmp_path)
    invalid = (
        "# customer-health-investigation\n\n"
        "Investigate customer health.\n\n"
        "## Instructions\n\n"
        "Missing the required final newline."
    )
    valid = f"{invalid}\n"
    opened: list[str] = []

    def edit_document(seed: str, *, agent_home: Path) -> str:
        del agent_home
        opened.append(seed)
        return invalid if len(opened) == 1 else valid

    monkeypatch.setattr(terminal, "_edit_document", edit_document)
    output = io.StringIO()
    try:
        created = await terminal._create_skill(
            agent,
            "customer-health-investigation",
            input_stream=io.StringIO("\n"),
            output_stream=output,
        )
        skill = await agent.read_skill("customer-health-investigation")
    finally:
        await agent.close()

    assert created is True
    assert opened[1] == invalid
    assert "Skill document is invalid" in output.getvalue()
    assert skill is not None


async def test_learn_command_routes_material_into_one_ordinary_foreground_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keychain = _Keychain()
    await _ready_agent(tmp_path, keychain)
    material = "Prefer a table followed by a two-sentence summary."
    provider = MockModelProvider(
        (_stop("I can propose that preference for approval."),),
        provider_id="openai:test-model",
    )
    _install_provider(monkeypatch, provider)
    output = io.StringIO()

    code = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(f"/learn {material}\n/exit\n"),
        output_stream=output,
        keychain=keychain,
    )

    assert code == 0
    routed = _request_text(provider.requests[0])
    assert len(routed) == 1
    assert routed[0].startswith("Treat the following as an explicit teaching request.")
    assert "Call the smallest fitting learning tool immediately" in routed[0]
    assert "approval card is the only confirmation" in routed[0]
    assert "never ask the user for a typed approval phrase" in routed[0]
    assert routed[0].endswith(material)
    assert terminal._learning_invocation_message("Remember this naturally.") is None
    with pytest.raises(ValueError, match=r"usage: /learn <material>"):
        terminal._learning_invocation_message("/learn")


@pytest.mark.parametrize(
    "invocation",
    (
        "/customer-health-investigation investigate account 42",
        "/skills use customer-health-investigation investigate account 42",
    ),
)
async def test_skill_slash_invocation_is_an_exact_ordinary_run_that_loads_skill_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invocation: str,
):
    keychain = _Keychain()
    await _ready_agent(tmp_path, keychain)
    agent = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    await agent.save_skill(
        "customer-health-investigation",
        "Investigate customer health.",
        "Compare account, support, and usage evidence.",
    )
    await agent.close()
    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="skill-load",
                        name="skill_view",
                        arguments={"name": "customer-health-investigation"},
                    ),
                ),
                provider_id="openai:test-model",
            ),
            _stop("Investigation complete."),
        ),
        provider_id="openai:test-model",
    )
    _install_provider(monkeypatch, provider)
    output = io.StringIO()

    code = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(f"{invocation}\n/exit\n"),
        output_stream=output,
        keychain=keychain,
    )

    assert code == 0
    assert _request_text(provider.requests[0]) == (invocation,)
    system_text = cast(TextBlock, provider.requests[0].messages[0].content[0]).text
    assert "only tool call in the first assistant step" in system_text
    assert provider.requests[1].messages[-1].role is MessageRole.TOOL
    conversation_match = re.search(
        r"Conversation  (conversation-[A-Za-z0-9]+)",
        output.getvalue(),
    )
    assert conversation_match is not None
    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    try:
        run = (await reopened.conversation_runs(conversation_match.group(1)))[0]
        transcript = await reopened.transcript(run.transcript.run.id)
        first = cast(TextBlock, transcript.messages[0].content[0])
        assert first.text == invocation
    finally:
        await reopened.close()


async def test_builtin_command_wins_skill_alias_and_explicit_fallback_remains_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keychain = _Keychain()
    await _ready_agent(tmp_path, keychain)
    agent = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    try:
        await agent.save_skill("status", "Custom status procedure.", "Inspect status.")
        assert await terminal._skill_invocation_message(agent, "/status") is None
        fallback = "/skills use status inspect customer 42"
        assert await terminal._skill_invocation_message(agent, fallback) == fallback
    finally:
        await agent.close()


class _CancelledProvider:
    provider_id = "openai:test-model"

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return True

    async def generate(self, request: ModelRequest) -> ModelResponse:
        raise asyncio.CancelledError


class _ScriptedInput:
    def __init__(self, events: list[str | BaseException]) -> None:
        self.events = events

    def readline(self, size: int = -1, /) -> str:
        del size
        event = self.events.pop(0)
        if isinstance(event, BaseException):
            raise event
        return event


async def test_ctrl_c_during_approval_cancels_exact_run_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keychain = _Keychain()
    await _ready_agent(tmp_path, keychain)
    proposed = "This interrupted write must not persist."
    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="memory_set",
                        arguments={"target": "memory", "content": proposed},
                    ),
                ),
            ),
        ),
        provider_id="openai:test-model",
    )
    _install_provider(monkeypatch, provider)
    monkeypatch.setattr(embedded, "_new_id", lambda prefix: f"{prefix}-approval")
    output = io.StringIO()

    code = await run_terminal_application(
        root=tmp_path,
        input_stream=cast(
            TextIO,
            _ScriptedInput(["remember this\n", KeyboardInterrupt(), "/exit\n"]),
        ),
        output_stream=output,
        keychain=keychain,
    )

    assert code == 0
    assert "Run interrupted" in output.getvalue()
    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    try:
        assert await reopened.read_memory() == ""
        records = await reopened.conversation_runs("conversation-approval")
        assert records[0].result is not None
        assert records[0].result.kind is LoopExitKind.INTERRUPTED
    finally:
        await reopened.close()


async def test_cancelled_run_is_interrupted_and_returns_to_prompt_without_lock_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keychain = _Keychain()
    await _ready_agent(tmp_path, keychain)
    _install_provider(monkeypatch, _CancelledProvider())
    monkeypatch.setattr(embedded, "_new_id", lambda prefix: f"{prefix}-fixed")
    output = io.StringIO()

    code = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO("cancel this run\n/exit\n"),
        output_stream=output,
        keychain=keychain,
    )

    assert code == 0
    assert "Run interrupted" in output.getvalue()
    assert output.getvalue().count("You › ") == 2
    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    try:
        records = await reopened.conversation_runs("conversation-fixed")
        assert records[0].result is not None
        assert records[0].result.kind is LoopExitKind.INTERRUPTED
        assert records[0].result.reason == "cancelled"
    finally:
        await reopened.close()


@pytest.mark.parametrize("terminal_input", ("", "/exit\n"))
async def test_eof_and_exit_release_writer_lock_and_print_bounded_resume_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_input: str,
):
    keychain = _Keychain()
    await _ready_agent(tmp_path, keychain)
    responses = (_stop("answer"),) if not terminal_input else ()
    provider = MockModelProvider(responses, provider_id="openai:test-model")
    _install_provider(monkeypatch, provider)
    output = io.StringIO()
    input_text = "question\n" if not terminal_input else terminal_input

    code = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(input_text),
        output_stream=output,
        keychain=keychain,
    )

    assert code == 0
    text = output.getvalue()
    assert len(text) < 20_000
    if not terminal_input:
        assert "Resume conversation" in text
    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    await reopened.close()
