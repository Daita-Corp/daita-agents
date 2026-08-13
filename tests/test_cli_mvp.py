from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import re
import shlex
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from daita import (
    Agent,
    PostgreSQLSource,
    Skill,
    SkillSummary,
    cli,
)
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    TextBlock,
    ToolCall,
)
from daita.llm.pricing import (
    CostBasis,
    CostEstimate,
    provider_reported_cost_estimate,
)
from daita.llm.providers.mock import MockModelProvider
from daita.security import SecretReference


class _TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def _invoke(
    argv: list[str],
    *,
    stdin: str = "",
    tty: bool = False,
) -> tuple[int, str, str]:
    input_stream = _TTYBuffer(stdin) if tty else io.StringIO(stdin)
    stdout = _TTYBuffer() if tty else io.StringIO()
    stderr = _TTYBuffer() if tty else io.StringIO()
    with (
        patch.object(sys, "stdin", input_stream),
        redirect_stdout(stdout),
        redirect_stderr(stderr),
    ):
        try:
            code = cli.main(argv)
        except SystemExit as error:
            code = error.code if isinstance(error.code, int) else 1
    return code, stdout.getvalue(), stderr.getvalue()


def _json_lines(text: str) -> tuple[dict[str, object], ...]:
    return tuple(json.loads(line) for line in text.splitlines() if line.strip())


def _subcommands(
    parser: argparse.ArgumentParser,
) -> dict[str, argparse.ArgumentParser]:
    action = next(
        item for item in parser._actions if isinstance(item, argparse._SubParsersAction)
    )
    return dict(action.choices)


def _surface(
    parser: argparse.ArgumentParser,
) -> tuple[tuple[str, ...], frozenset[str]]:
    positionals: list[str] = []
    options: set[str] = set()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            continue
        if action.option_strings:
            options.update(action.option_strings)
        else:
            positionals.append(action.dest)
    return tuple(positionals), frozenset(options)


def _stop(text: str, *, usage: ModelUsage | None = None) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        text=text,
        usage=usage or ModelUsage(),
    )


def _call(name: str, arguments: dict[str, object]) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(id="call-1", name=name, arguments=arguments),),
    )


def _request_text(request: ModelRequest) -> tuple[str, ...]:
    return tuple(
        block.text
        for message in request.messages
        if message.role is not MessageRole.SYSTEM
        for block in message.content
        if isinstance(block, TextBlock)
    )


async def _create_agent(root: Path, name: str) -> None:
    agent = await Agent.create(name, root=root)
    await agent.close()


async def _open_and_close_agent(root: Path, name: str) -> None:
    agent = await Agent.open(name, root=root)
    await agent.close()


async def _knowledge(root: Path, name: str) -> tuple[str, Skill | None]:
    agent = await Agent.open(name, root=root)
    try:
        return await agent.read_memory(), await agent.read_skill("monthly-revenue")
    finally:
        await agent.close()


async def _complete_knowledge(
    root: Path,
    name: str,
) -> tuple[str, str, tuple[object, ...]]:
    agent = await Agent.open(name, root=root)
    try:
        return (
            await agent.read_memory(),
            await agent.read_user_profile(),
            await agent.list_skills(),
        )
    finally:
        await agent.close()


async def _seed_knowledge(root: Path, name: str) -> None:
    agent = await Agent.open(name, root=root)
    try:
        await agent.set_memory("Initial memory")
        await agent.set_user_profile("Initial user profile")
        await agent.save_skill(
            "monthly-revenue",
            "Initial description",
            "Initial instructions",
        )
    finally:
        await agent.close()


def _replacement_editor(
    directory: Path,
    replacement: str,
    *,
    name: str,
) -> str:
    replacement_path = directory / f"{name}-replacement.txt"
    replacement_path.write_text(replacement, encoding="utf-8")
    editor_path = directory / f"{name}-editor.py"
    editor_path.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "target = Path(sys.argv[-1])\n"
        "replacement = Path(sys.argv[1]).read_text(encoding='utf-8')\n"
        "target.write_text(replacement, encoding='utf-8')\n",
        encoding="utf-8",
    )
    return shlex.join((sys.executable, str(editor_path), str(replacement_path)))


def test_current_parser_keeps_the_existing_one_shot_surface_green():
    parser = cli.build_parser()
    commands = _subcommands(parser)

    assert _surface(parser) == (
        (),
        frozenset({"-h", "--help", "--version", "--root", "--agent"}),
    )
    assert {"create", "attach", "sources", "run"} <= set(commands)
    assert _surface(commands["create"]) == (
        ("name",),
        frozenset({"-h", "--help"}),
    )
    assert _surface(commands["attach"]) == (
        ("name", "kind", "path"),
        frozenset(
            {
                "-h",
                "--help",
                "--host",
                "--port",
                "--database",
                "--username",
                "--password-env",
                "--schema",
                "--ssl-mode",
                "--source-name",
            }
        ),
    )
    assert _surface(commands["sources"]) == (
        ("name",),
        frozenset({"-h", "--help"}),
    )


def test_zero_argument_cli_dispatches_to_the_terminal_application():
    with patch.object(
        cli,
        "run_terminal_application",
        new=AsyncMock(return_value=0),
    ) as run_terminal:
        code, stdout, stderr = _invoke([], tty=True)

    assert code == 0
    assert stdout == ""
    assert stderr == ""
    run_terminal.assert_awaited_once()
    call = run_terminal.await_args
    assert call is not None
    assert call.kwargs["root"] is None
    assert call.kwargs["agent_name"] is None


def test_zero_argument_cli_rejects_non_tty_before_terminal_dispatch():
    with patch.object(
        cli,
        "run_terminal_application",
        new=AsyncMock(return_value=0),
    ) as run_terminal:
        code, stdout, stderr = _invoke([])

    assert code == 1
    assert stdout == ""
    assert _json_lines(stderr) == (
        {"error": "daita requires interactive stdin, stdout, and stderr"},
    )
    run_terminal.assert_not_awaited()


def test_zero_agents_prompts_for_and_creates_one_agent():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()

        code, stdout, stderr = _invoke(
            ["--root", str(root)],
            stdin="atlas\n",
            tty=True,
        )

        assert code == 0
        assert stderr == ""
        assert "Agent name: " in stdout
        assert "Select a model provider" in stdout
        assert "Agent     atlas" not in stdout
        assert "Stage 2 status" not in stdout
        asyncio.run(_open_and_close_agent(root, "atlas"))


def test_one_agent_is_selected_automatically_and_its_lock_is_released():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "solo"))

        code, stdout, stderr = _invoke(["--root", str(root)], tty=True)

        assert code == 0
        assert stderr == ""
        assert "Select an agent" not in stdout
        assert "Select a model provider" in stdout
        assert "Agent     solo" not in stdout
        asyncio.run(_open_and_close_agent(root, "solo"))


def test_multiple_agents_render_a_deterministic_numbered_picker():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "zulu"))
        asyncio.run(_create_agent(root, "alpha"))

        code, stdout, stderr = _invoke(
            ["--root", str(root)],
            stdin="2\n",
            tty=True,
        )

        assert code == 0
        assert stderr == ""
        assert "1. alpha\n2. zulu\n3. Create a new agent" in stdout
        assert "Select a model provider" in stdout
        assert "Agent     zulu" not in stdout


def test_multiple_agent_picker_can_create_another_agent():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "alpha"))
        asyncio.run(_create_agent(root, "beta"))

        code, stdout, stderr = _invoke(
            ["--root", str(root)],
            stdin="3\ngamma\n",
            tty=True,
        )

        assert code == 0
        assert stderr == ""
        assert "3. Create a new agent" in stdout
        assert "Select a model provider" in stdout
        assert "Agent     gamma" not in stdout
        assert asyncio.run(Agent.list(root=root)) == ("alpha", "beta", "gamma")


def test_agent_option_selects_the_exact_requested_agent():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "alpha"))
        asyncio.run(_create_agent(root, "atlas"))

        code, stdout, stderr = _invoke(
            ["--root", str(root), "--agent", "atlas"],
            tty=True,
        )

        assert code == 0
        assert stderr == ""
        assert "Select an agent" not in stdout
        assert "Select a model provider" in stdout
        assert "Agent     atlas" not in stdout


def test_unknown_agent_option_reports_an_actionable_error():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()

        code, stdout, stderr = _invoke(
            ["--root", str(root), "--agent", "missing"],
            tty=True,
        )

        assert code == 1
        assert stdout == ""
        error = _json_lines(stderr)[0]["error"]
        assert isinstance(error, str)
        assert "missing" in error
        assert "without --agent" in error


def test_agent_listing_omits_malformed_incomplete_and_symlinked_homes(
    tmp_path: Path,
):
    asyncio.run(_create_agent(tmp_path, "valid"))
    agents_root = tmp_path / "agents"
    incomplete = agents_root / "incomplete"
    incomplete.mkdir()
    malformed = agents_root / "malformed"
    malformed.mkdir()
    (malformed / "agent.toml").write_text("not = [valid", encoding="utf-8")
    (malformed / "state.db").touch()
    (agents_root / "linked").symlink_to(
        agents_root / "valid",
        target_is_directory=True,
    )

    assert asyncio.run(Agent.list(root=tmp_path)) == ("valid",)


@pytest.mark.parametrize(
    ("input_stream", "expected_code", "expected_message"),
    (
        (_TTYBuffer(""), 0, "Setup cancelled."),
        (None, 130, "Setup interrupted."),
    ),
)
def test_selection_eof_and_ctrl_c_exit_without_partial_agent_creation(
    tmp_path: Path,
    input_stream: _TTYBuffer | None,
    expected_code: int,
    expected_message: str,
):
    class _InterruptingInput:
        def isatty(self) -> bool:
            return True

        def readline(self, size: int = -1, /) -> str:
            del size
            raise KeyboardInterrupt

    stream = _InterruptingInput() if input_stream is None else input_stream
    stdout = _TTYBuffer()
    stderr = _TTYBuffer()
    with (
        patch.object(sys, "stdin", stream),
        redirect_stdout(stdout),
        redirect_stderr(stderr),
    ):
        code = cli.main(["--root", str(tmp_path)])

    assert code == expected_code
    assert expected_message in stdout.getvalue()
    assert stderr.getvalue() == ""
    assert asyncio.run(Agent.list(root=tmp_path)) == ()


def test_attach_parser_builds_postgresql_source_with_secret_reference():
    args = cli.build_parser().parse_args(
        [
            "attach",
            "atlas",
            "postgresql",
            "--host",
            "127.0.0.1",
            "--port",
            "55432",
            "--database",
            "daita_fixture",
            "--username",
            "daita_reader",
            "--password-env",
            "DAITA_FIXTURE_POSTGRES_PASSWORD",
            "--schema",
            "analytics",
            "--ssl-mode",
            "disable",
            "--source-name",
            "Fixture PostgreSQL",
        ]
    )

    source = cli._source_from_attach_args(args)

    assert isinstance(source, PostgreSQLSource)
    assert source.host == "127.0.0.1"
    assert source.port == 55432
    assert source.database == "daita_fixture"
    assert source.username == "daita_reader"
    assert source.credential == SecretReference.environment(
        "DAITA_FIXTURE_POSTGRES_PASSWORD"
    )
    assert source.schemas == ("analytics",)
    assert source.ssl_mode == "disable"
    assert source.name == "Fixture PostgreSQL"


def test_attach_postgresql_reports_missing_connection_fields():
    args = cli.build_parser().parse_args(["attach", "atlas", "postgresql"])

    with pytest.raises(
        ValueError,
        match=r"attach postgresql requires --host, --database, --username",
    ):
        cli._source_from_attach_args(args)


def test_current_run_writes_exactly_one_terminal_json_record_to_stdout():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        create_code, create_stdout, create_stderr = _invoke(
            ["--root", str(root), "create", "runner"]
        )
        assert create_code == 0
        assert len(_json_lines(create_stdout)) == 1
        assert create_stderr == ""

        provider = MockModelProvider((_stop("bounded answer"),))
        with patch.object(cli, "create_llm_provider", return_value=provider):
            code, stdout, stderr = _invoke(
                [
                    "--root",
                    str(root),
                    "run",
                    "runner",
                    "bounded question",
                    "--model",
                    "mock:scripted",
                ]
            )

    records = _json_lines(stdout)
    assert code == 0
    assert len(records) == 1
    assert stderr == ""
    assert records[0]["status"] == "completed"
    assert records[0]["text"] == "bounded answer"
    assert {"run_id", "status", "reason", "text", "steps"} <= set(records[0])
    provider.assert_consumed()


def test_current_diagnostics_use_stderr_and_leave_stdout_empty():
    with tempfile.TemporaryDirectory() as directory:
        with patch.object(
            cli,
            "create_llm_provider",
            side_effect=ImportError("install the mock extra"),
        ):
            code, stdout, stderr = _invoke(
                [
                    "--root",
                    directory,
                    "run",
                    "runner",
                    "question",
                    "--model",
                    "mock:scripted",
                ]
            )

    assert code == 1
    assert stdout == ""
    assert _json_lines(stderr) == ({"error": "install the mock extra"},)


def test_direct_cli_rejects_unreviewed_models_without_provider_owned_facts():
    class UnprofiledProvider:
        provider_id = "custom:unknown"

        def supports_request_policy(self, request: ModelRequest) -> bool:
            return True

        async def generate(self, request: ModelRequest) -> ModelResponse:
            raise AssertionError("an unadmitted provider must not be called")

    with patch.object(
        cli,
        "create_llm_provider",
        return_value=UnprofiledProvider(),
    ):
        with pytest.raises(
            ValueError,
            match="interactive terminal.*prove tool support.*hard token limits",
        ):
            cli._model_configuration(
                "custom:unknown",
                base_url="https://models.invalid/v1",
            )


def test_direct_cli_uses_provider_owned_profile_without_generic_defaults():
    provider = MockModelProvider(())

    with patch.object(cli, "create_llm_provider", return_value=provider):
        configured_provider, profile = cli._model_configuration("mock:scripted")

    assert configured_provider is provider
    assert profile == provider.model_profile
    assert profile.context_window_tokens == 128_000
    assert profile.max_output_tokens == 2_048
    assert not profile.supports_parallel_tools


def test_direct_cli_requires_explicit_limit_overrides_as_a_pair():
    with pytest.raises(
        ValueError,
        match="--context-window and --max-output must be supplied together",
    ):
        cli._model_configuration("openai:gpt-5.6-sol", context_window=4_096)


def test_direct_cli_limits_cannot_exceed_provider_owned_profile():
    provider = MockModelProvider(())

    with patch.object(cli, "create_llm_provider", return_value=provider):
        with pytest.raises(
            ValueError,
            match="cannot exceed.*provider-owned model profile",
        ):
            cli._model_configuration(
                "mock:scripted",
                context_window=128_001,
                max_output=2_048,
            )


def test_future_cli_1_parser_adds_only_explicit_run_continuation_flags():
    run = _subcommands(cli.build_parser())["run"]
    assert _surface(run) == (
        ("name", "message"),
        frozenset(
            {
                "-h",
                "--help",
                "--model",
                "--base-url",
                "--context-window",
                "--max-output",
                "--conversation-id",
                "--events-jsonl",
            }
        ),
    )


def test_future_cli_1_run_returns_and_cold_continues_an_explicit_conversation():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "continuation"))
        first_provider = MockModelProvider((_stop("first answer"),))
        second_provider = MockModelProvider((_stop("follow-up answer"),))
        with patch.object(
            cli,
            "create_llm_provider",
            side_effect=(first_provider, second_provider),
        ):
            first_code, first_stdout, first_stderr = _invoke(
                [
                    "--root",
                    str(root),
                    "run",
                    "continuation",
                    "first question",
                    "--model",
                    "mock:scripted",
                ]
            )
            first_result = _json_lines(first_stdout)[0]

            assert first_code == 0
            assert first_stderr == ""
            assert isinstance(first_result.get("conversation_id"), str)
            conversation_id = str(first_result["conversation_id"])

            second_code, second_stdout, second_stderr = _invoke(
                [
                    "--root",
                    str(root),
                    "run",
                    "continuation",
                    "follow-up question",
                    "--model",
                    "mock:scripted",
                    "--conversation-id",
                    conversation_id,
                ]
            )

    second_result = _json_lines(second_stdout)[0]
    assert second_code == 0
    assert second_stderr == ""
    assert second_result["conversation_id"] == conversation_id
    assert _request_text(second_provider.requests[0]) == (
        "first question",
        "first answer",
        "follow-up question",
    )
    first_provider.assert_consumed()
    second_provider.assert_consumed()


def test_future_cli_1_requested_event_jsonl_uses_stderr_only():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "events"))
        provider = MockModelProvider((_stop("event answer"),))
        with patch.object(cli, "create_llm_provider", return_value=provider):
            code, stdout, stderr = _invoke(
                [
                    "--root",
                    str(root),
                    "run",
                    "events",
                    "event question",
                    "--model",
                    "mock:scripted",
                    "--events-jsonl",
                ]
            )

    assert code == 0
    result_records = _json_lines(stdout)
    event_records = _json_lines(stderr)
    assert len(result_records) == 1
    assert isinstance(result_records[0].get("conversation_id"), str)
    assert tuple(record["kind"] for record in event_records) == (
        "run.started",
        "model.completed",
        "run.completed",
    )
    assert all("run_id" in record for record in event_records)
    assert all("conversation_id" in record for record in event_records)
    assert all("data" in record for record in event_records)
    provider.assert_consumed()


def test_future_cli_2_parser_adds_the_bounded_chat_surface():
    commands = _subcommands(cli.build_parser())
    assert "chat" in commands
    assert _surface(commands["chat"]) == (
        ("name",),
        frozenset({"-h", "--help", "--model", "--conversation"}),
    )


def test_future_cli_2_chat_keeps_one_explicit_in_process_conversation():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "chat-agent"))
        provider = MockModelProvider(
            (_stop("first chat answer"), _stop("follow-up chat answer"))
        )
        with patch.object(cli, "create_llm_provider", return_value=provider):
            code, stdout, stderr = _invoke(
                [
                    "--root",
                    str(root),
                    "chat",
                    "chat-agent",
                    "--model",
                    "mock:scripted",
                ],
                stdin="first chat question\nfollow-up chat question\n/exit\n",
                tty=True,
            )

    assert code == 0
    assert stderr == ""
    assert "first chat answer" in stdout
    assert "follow-up chat answer" in stdout
    assert "Resume with:" in stdout
    assert "--conversation" in stdout
    assert _request_text(provider.requests[1]) == (
        "first chat question",
        "first chat answer",
        "follow-up chat question",
    )
    provider.assert_consumed()


def test_cli_2_chat_rejects_non_tty_streams_before_provider_or_agent_open():
    with (
        patch.object(cli, "create_llm_provider") as provider_factory,
        patch.object(cli.Agent, "open", new_callable=AsyncMock) as agent_open,
    ):
        code, stdout, stderr = _invoke(
            ["chat", "not-opened", "--model", "mock:scripted"]
        )

    assert code == 1
    assert stdout == ""
    assert _json_lines(stderr) == (
        {
            "error": (
                "chat requires an interactive terminal on stdin, stdout, and stderr"
            )
        },
    )
    provider_factory.assert_not_called()
    agent_open.assert_not_awaited()


def test_cli_2_slash_commands_are_local_and_bounded():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "local-commands"))
        provider = MockModelProvider(())
        with patch.object(cli, "create_llm_provider", return_value=provider):
            code, stdout, stderr = _invoke(
                [
                    "--root",
                    str(root),
                    "chat",
                    "local-commands",
                    "--model",
                    "mock:scripted",
                ],
                stdin=(
                    "/help\n"
                    "/status\n"
                    "/conversation\n"
                    "/sources\n"
                    "/resume\n"
                    "/not-a-command\n"
                    "/exit\n"
                ),
                tty=True,
            )

    assert code == 0
    assert "Commands:" in stdout
    assert "This process: 0 turns, 0 steps, 0 tokens" in stdout
    assert "No conversation was created." in stdout
    assert "Usage: /resume <conversation-id>" in stderr
    assert "Unknown command. Type /help for commands." in stderr
    assert provider.requests == ()
    provider.assert_consumed()


@pytest.mark.parametrize(
    ("estimate", "rendered"),
    (
        (
            CostEstimate.complete(
                Decimal("0.02"),
                basis=CostBasis.PUBLIC_LIST,
                rate_schedule_id="public:test",
            ),
            "$0.02 estimated at public list rates",
        ),
        (
            CostEstimate.partial(
                Decimal("0.01"),
                code="unpriced_attempt",
                basis=CostBasis.PUBLIC_LIST,
                rate_schedule_id="public:test",
            ),
            "≥$0.01 estimated; some attempts were unpriced",
        ),
        (
            CostEstimate.unavailable(),
            "cost unavailable",
        ),
        (
            provider_reported_cost_estimate(
                Decimal("0"),
                currency="USD",
                unit="request",
            ),
            "provider API charge $0; local compute not estimated",
        ),
    ),
)
def test_cli_2_chat_renders_every_cost_state(estimate, rendered):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "cost-states"))
        provider = MockModelProvider(
            (
                _stop(
                    "answer",
                    usage=ModelUsage(cost_estimate=estimate),
                ),
            )
        )
        with patch.object(cli, "create_llm_provider", return_value=provider):
            code, stdout, stderr = _invoke(
                [
                    "--root",
                    str(root),
                    "chat",
                    "cost-states",
                    "--model",
                    "mock:scripted",
                ],
                stdin="question\n/exit\n",
                tty=True,
            )

    assert code == 0
    assert stderr == ""
    assert rendered in stdout
    if estimate.amount_usd is None:
        assert "$0" not in stdout


def test_cli_2_new_clears_only_the_in_process_conversation():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "new-command"))
        provider = MockModelProvider((_stop("first answer"), _stop("second answer")))
        with patch.object(cli, "create_llm_provider", return_value=provider):
            code, stdout, stderr = _invoke(
                [
                    "--root",
                    str(root),
                    "chat",
                    "new-command",
                    "--model",
                    "mock:scripted",
                ],
                stdin="first question\n/new\nsecond question\n/exit\n",
                tty=True,
            )

    created = re.findall(r"Conversation: (conversation-[A-Za-z0-9]+)", stdout)
    assert code == 0
    assert stderr == ""
    assert len(created) == 2
    assert created[0] != created[1]
    assert _request_text(provider.requests[0]) == ("first question",)
    assert _request_text(provider.requests[1]) == ("second question",)
    assert f"--conversation {created[1]}" in stdout
    provider.assert_consumed()


def test_cli_2_resume_validates_publicly_before_selecting_a_conversation():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "resume-command"))
        seed_provider = MockModelProvider((_stop("seed answer"),))
        chat_provider = MockModelProvider((_stop("continued answer"),))
        with patch.object(
            cli,
            "create_llm_provider",
            side_effect=(seed_provider, chat_provider),
        ):
            seed_code, seed_stdout, seed_stderr = _invoke(
                [
                    "--root",
                    str(root),
                    "run",
                    "resume-command",
                    "seed question",
                    "--model",
                    "mock:scripted",
                ]
            )
            conversation_id = str(_json_lines(seed_stdout)[0]["conversation_id"])
            code, stdout, stderr = _invoke(
                [
                    "--root",
                    str(root),
                    "chat",
                    "resume-command",
                    "--model",
                    "mock:scripted",
                ],
                stdin=(
                    f"/resume unknown-valid-id\n"
                    f"/resume {conversation_id}\n"
                    "continued question\n"
                    "/exit\n"
                ),
                tty=True,
            )

    assert seed_code == 0
    assert seed_stderr == ""
    assert code == 0
    assert "Cannot resume conversation: unknown conversation for this agent" in stderr
    assert f"Conversation: {conversation_id}" in stdout
    assert _request_text(chat_provider.requests[0]) == (
        "seed question",
        "seed answer",
        "continued question",
    )
    seed_provider.assert_consumed()
    chat_provider.assert_consumed()


def test_cli_2_eof_closes_agent_and_prints_exact_shell_safe_resume_command():
    with tempfile.TemporaryDirectory() as directory:
        root = (Path(directory) / "state root").resolve()
        root.mkdir()
        asyncio.run(_create_agent(root, "eof-agent"))
        provider = MockModelProvider((_stop("answer before EOF"),))
        with patch.object(cli, "create_llm_provider", return_value=provider):
            code, stdout, stderr = _invoke(
                [
                    "--root",
                    str(root),
                    "chat",
                    "eof-agent",
                    "--model",
                    "mock:scripted",
                ],
                stdin="question before EOF\n",
                tty=True,
            )

        asyncio.run(_open_and_close_agent(root, "eof-agent"))

    conversation_match = re.search(r"Conversation: (conversation-[A-Za-z0-9]+)", stdout)
    assert conversation_match is not None
    conversation_id = conversation_match.group(1)
    expected = shlex.join(
        (
            "daita",
            "--root",
            str(root),
            "chat",
            "eof-agent",
            "--model",
            "mock:scripted",
            "--conversation",
            conversation_id,
        )
    )
    assert code == 0
    assert stderr == ""
    assert stdout.endswith(f"Resume with:\n{expected}\n")
    provider.assert_consumed()


def test_cli_2_ctrl_c_closes_agent_without_creating_conversation():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "interrupt-agent"))
        provider = MockModelProvider(())
        with (
            patch.object(cli, "create_llm_provider", return_value=provider),
            patch("builtins.input", side_effect=KeyboardInterrupt),
        ):
            code, stdout, stderr = _invoke(
                [
                    "--root",
                    str(root),
                    "chat",
                    "interrupt-agent",
                    "--model",
                    "mock:scripted",
                ],
                tty=True,
            )

        asyncio.run(_open_and_close_agent(root, "interrupt-agent"))

    assert code == 130
    assert stderr == ""
    assert "No conversation was created." in stdout
    assert "Resume with:" not in stdout
    assert provider.requests == ()
    provider.assert_consumed()


def test_future_cli_3_chat_uses_the_existing_exact_once_approval_path():
    proposed = "Revenue uses paid invoice date in UTC."
    arguments: dict[str, object] = {"target": "memory", "content": proposed}
    provider = MockModelProvider(
        (
            _call("memory_set", arguments),
            _stop("saved once"),
            _stop("stage-two-no-handler fallback"),
        )
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "approval-agent"))
        with patch.object(cli, "create_llm_provider", return_value=provider):
            code, stdout, stderr = _invoke(
                [
                    "--root",
                    str(root),
                    "chat",
                    "approval-agent",
                    "--model",
                    "mock:scripted",
                ],
                stdin="remember the corrected definition\n  Y  \n/exit\n",
                tty=True,
            )

        memory, _ = asyncio.run(_knowledge(root, "approval-agent"))

    assert code == 0
    assert stderr == ""
    assert "memory_set" in stdout
    assert "memory.set" in stdout
    assert proposed in stdout
    assert (
        json.dumps(
            arguments,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        in stdout
    )
    assert stdout.count("Approve this exact change once? [y/n]") == 1
    assert memory == proposed
    assert len(provider.requests) == 2


@pytest.mark.parametrize(
    ("approval_input", "prompt_count"),
    (
        ("n\n/exit\n", 1),
        ("\nn\n/exit\n", 2),
        ("yes\nn\n/exit\n", 2),
        ("a\nn\n/exit\n", 2),
    ),
    ids=("deny", "blank", "unrecognized", "old-tui-approve"),
)
def test_cli_3_chat_requires_explicit_y_or_n(
    approval_input: str,
    prompt_count: int,
):
    proposed = "This must not be stored."
    provider = MockModelProvider(
        (
            _call("memory_set", {"target": "memory", "content": proposed}),
            _stop("the proposal was not saved"),
        )
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "default-deny"))
        with patch.object(cli, "create_llm_provider", return_value=provider):
            code, stdout, stderr = _invoke(
                [
                    "--root",
                    str(root),
                    "chat",
                    "default-deny",
                    "--model",
                    "mock:scripted",
                ],
                stdin=f"remember this\n{approval_input}",
                tty=True,
            )

        memory, _ = asyncio.run(_knowledge(root, "default-deny"))

    assert code == 0
    assert stderr == ""
    assert stdout.count("Approve this exact change once? [y/n]") == prompt_count
    assert stdout.count("Enter y to approve or n to deny.") == prompt_count - 1
    assert memory == ""
    provider.assert_consumed()


def test_cli_3_chat_decides_each_exact_request_independently():
    first = "Keep the first approved definition."
    second = "Do not replace it with this definition."
    provider = MockModelProvider(
        (
            _call("memory_set", {"target": "memory", "content": first}),
            _stop("first proposal handled"),
            _call("memory_set", {"target": "memory", "content": second}),
            _stop("second proposal handled"),
        )
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "independent-approvals"))
        with patch.object(cli, "create_llm_provider", return_value=provider):
            code, stdout, stderr = _invoke(
                [
                    "--root",
                    str(root),
                    "chat",
                    "independent-approvals",
                    "--model",
                    "mock:scripted",
                ],
                stdin="first proposal\ny\nsecond proposal\nn\n/exit\n",
                tty=True,
            )

        memory, _ = asyncio.run(_knowledge(root, "independent-approvals"))

    assert code == 0
    assert stderr == ""
    assert stdout.count("Approve this exact change once? [y/n]") == 2
    assert first in stdout
    assert second in stdout
    assert memory == first
    provider.assert_consumed()


def test_cli_3_ctrl_c_during_approval_never_writes_and_releases_the_lock():
    proposed = "An interrupted proposal must not be stored."
    provider = MockModelProvider(
        (_call("memory_set", {"target": "memory", "content": proposed}),)
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "approval-interrupt"))
        with (
            patch.object(cli, "create_llm_provider", return_value=provider),
            patch(
                "builtins.input",
                side_effect=("remember this", KeyboardInterrupt),
            ),
        ):
            code, stdout, stderr = _invoke(
                [
                    "--root",
                    str(root),
                    "chat",
                    "approval-interrupt",
                    "--model",
                    "mock:scripted",
                ],
                tty=True,
            )

        memory, _ = asyncio.run(_knowledge(root, "approval-interrupt"))
        asyncio.run(_open_and_close_agent(root, "approval-interrupt"))

    assert code == 130
    assert stderr == "Chat interrupted.\n"
    assert proposed in stdout
    assert memory == ""
    provider.assert_consumed()


def test_cli_3_run_still_installs_no_approval_handler():
    proposed = "A one-shot run must not prompt or write."
    provider = MockModelProvider(
        (
            _call("memory_set", {"target": "memory", "content": proposed}),
            _stop("approval remains required"),
        )
    )
    opened_with: list[dict[str, Any]] = []
    original_open = cli.Agent.open

    async def open_spy(*args: Any, **kwargs: Any) -> Agent:
        opened_with.append(kwargs)
        return await original_open(*args, **kwargs)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "non-interactive-run"))
        with (
            patch.object(cli, "create_llm_provider", return_value=provider),
            patch.object(cli.Agent, "open", side_effect=open_spy),
        ):
            code, stdout, stderr = _invoke(
                [
                    "--root",
                    str(root),
                    "run",
                    "non-interactive-run",
                    "remember this",
                    "--model",
                    "mock:scripted",
                ]
            )

        memory, _ = asyncio.run(_knowledge(root, "non-interactive-run"))

    assert code == 0
    assert stderr == ""
    assert "Approve this exact change once?" not in stdout
    assert len(opened_with) == 1
    assert "approval_handler" not in opened_with[0]
    assert memory == ""
    provider.assert_consumed()


def test_cli_3_y_and_n_remain_ordinary_messages_outside_approval():
    provider = MockModelProvider((_stop("first answer"), _stop("second answer")))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "ordinary-y-n"))
        with patch.object(cli, "create_llm_provider", return_value=provider):
            code, stdout, stderr = _invoke(
                [
                    "--root",
                    str(root),
                    "chat",
                    "ordinary-y-n",
                    "--model",
                    "mock:scripted",
                ],
                stdin="y\nn\n/exit\n",
                tty=True,
            )

    assert code == 0
    assert stderr == ""
    assert "Approve this exact change once?" not in stdout
    assert _request_text(provider.requests[0]) == ("y",)
    assert _request_text(provider.requests[1]) == (
        "y",
        "first answer",
        "n",
    )
    provider.assert_consumed()


def test_cli_parser_keeps_direct_knowledge_and_confirmed_lifecycle_commands():
    commands = _subcommands(cli.build_parser())
    assert set(commands) == {
        "artifacts",
        "create",
        "attach",
        "sources",
        "postgresql-update-readiness",
        "detach",
        "conversations",
        "export-location",
        "delete",
        "run",
        "chat",
        "memory",
        "skills",
    }
    assert _surface(commands["detach"]) == (
        ("name", "source_id"),
        frozenset({"-h", "--help", "--yes"}),
    )
    assert _surface(commands["delete"]) == (
        ("name",),
        frozenset({"-h", "--help", "--yes"}),
    )
    assert _surface(commands["postgresql-update-readiness"])[0] == (
        "name",
        "source_id",
        "resource_id",
    )
    conversations = _subcommands(commands["conversations"])
    assert set(conversations) == {"clear"}
    assert _surface(conversations["clear"]) == (
        ("name",),
        frozenset({"-h", "--help", "--yes"}),
    )

    memory = _subcommands(commands["memory"])
    assert set(memory) == {
        "accept-candidate",
        "clear-rejected",
        "edit",
        "edit-candidate",
        "inspect",
        "list-candidates",
        "read",
        "reject-candidate",
        "review",
        "set",
        "show-candidate",
    }
    assert _surface(memory["read"]) == (
        ("name",),
        frozenset({"-h", "--help", "--target"}),
    )
    assert _surface(memory["edit"]) == (
        ("name",),
        frozenset({"-h", "--help", "--target"}),
    )
    assert _surface(memory["set"]) == (
        ("name",),
        frozenset({"-h", "--help", "--target", "--file"}),
    )
    assert _surface(memory["inspect"]) == (
        ("name",),
        frozenset({"-h", "--help"}),
    )

    skills = _subcommands(commands["skills"])
    assert set(skills) == {"list", "show", "edit", "save", "delete"}
    assert _surface(skills["list"]) == (
        ("name",),
        frozenset({"-h", "--help"}),
    )
    for command in ("show", "edit", "delete"):
        assert _surface(skills[command]) == (
            ("name", "skill_name"),
            frozenset({"-h", "--help"}),
        )
    assert _surface(skills["save"]) == (
        ("name", "skill_name"),
        frozenset({"-h", "--help", "--description", "--instructions-file"}),
    )


def test_future_cli_4_direct_knowledge_commands_survive_reopen():
    memory_text = "Revenue excludes voided invoices."
    instructions = "Group paid invoices by UTC month."
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "knowledge-agent"))
        memory_file = root / "memory-input.txt"
        skill_file = root / "skill-input.txt"
        memory_file.write_text(memory_text, encoding="utf-8")
        skill_file.write_text(instructions, encoding="utf-8")

        memory_code, _, memory_stderr = _invoke(
            [
                "--root",
                str(root),
                "memory",
                "set",
                "knowledge-agent",
                "--target",
                "memory",
                "--file",
                str(memory_file),
            ]
        )
        skill_code, _, skill_stderr = _invoke(
            [
                "--root",
                str(root),
                "skills",
                "save",
                "knowledge-agent",
                "monthly-revenue",
                "--description",
                "Monthly revenue procedure.",
                "--instructions-file",
                str(skill_file),
            ]
        )
        inspect_code, inspect_stdout, inspect_stderr = _invoke(
            [
                "--root",
                str(root),
                "memory",
                "inspect",
                "knowledge-agent",
            ]
        )
        memory, skill = asyncio.run(_knowledge(root, "knowledge-agent"))

    assert memory_code == 0
    assert memory_stderr == ""
    assert skill_code == 0
    assert skill_stderr == ""
    assert inspect_code == 0
    assert inspect_stderr == ""
    assert _json_lines(inspect_stdout) == (
        {"annotations": [], "candidates": [], "global_memory": memory_text},
    )
    assert memory == memory_text
    assert skill is not None
    assert skill.name == "monthly-revenue"
    assert skill.description == "Monthly revenue procedure."
    assert skill.instructions == instructions


def test_cli_4_memory_file_stdin_and_reads_preserve_complete_utf8_content():
    memory_text = "Revenue uses café prices.\n\n- Keep € values exact."
    user_text = "Prefer concise answers.\nUnicode: λ"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "memory-io"))
        memory_file = root / "memory.txt"
        memory_file.write_text(memory_text, encoding="utf-8")

        memory_code, _, memory_stderr = _invoke(
            [
                "--root",
                str(root),
                "memory",
                "set",
                "memory-io",
                "--target",
                "memory",
                "--file",
                str(memory_file),
            ]
        )
        user_code, _, user_stderr = _invoke(
            [
                "--root",
                str(root),
                "memory",
                "set",
                "memory-io",
                "--target",
                "user",
                "--file",
                "-",
            ],
            stdin=user_text,
        )
        read_memory_code, read_memory, read_memory_stderr = _invoke(
            ["--root", str(root), "memory", "read", "memory-io"]
        )
        read_user_code, read_user, read_user_stderr = _invoke(
            [
                "--root",
                str(root),
                "memory",
                "read",
                "memory-io",
                "--target",
                "user",
            ]
        )
        persisted = asyncio.run(_complete_knowledge(root, "memory-io"))

    assert (memory_code, user_code, read_memory_code, read_user_code) == (0, 0, 0, 0)
    assert memory_stderr == user_stderr == ""
    assert read_memory_stderr == read_user_stderr == ""
    assert _json_lines(read_memory)[0]["content"] == memory_text
    assert _json_lines(read_user)[0]["content"] == user_text
    assert persisted[:2] == (memory_text, user_text)


def test_cli_4_skill_list_show_save_stdin_and_delete_use_complete_content():
    instructions = "Group paid invoices.\n\n```sql\nSELECT 'λ';\n```"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "skill-io"))
        save_code, save_stdout, save_stderr = _invoke(
            [
                "--root",
                str(root),
                "skills",
                "save",
                "skill-io",
                "monthly-revenue",
                "--description",
                "Monthly revenue procedure.",
                "--instructions-file",
                "-",
            ],
            stdin=instructions,
        )
        list_code, list_stdout, list_stderr = _invoke(
            ["--root", str(root), "skills", "list", "skill-io"]
        )
        show_code, show_stdout, show_stderr = _invoke(
            [
                "--root",
                str(root),
                "skills",
                "show",
                "skill-io",
                "monthly-revenue",
            ]
        )
        delete_code, delete_stdout, delete_stderr = _invoke(
            [
                "--root",
                str(root),
                "skills",
                "delete",
                "skill-io",
                "monthly-revenue",
            ]
        )
        persisted = asyncio.run(_complete_knowledge(root, "skill-io"))

    assert (save_code, list_code, show_code, delete_code) == (0, 0, 0, 0)
    assert save_stderr == list_stderr == show_stderr == delete_stderr == ""
    assert _json_lines(save_stdout)[0]["changed"] is True
    assert _json_lines(list_stdout) == (
        [
            {
                "description": "Monthly revenue procedure.",
                "name": "monthly-revenue",
            }
        ],
    )
    assert _json_lines(show_stdout)[0]["instructions"] == instructions
    assert _json_lines(delete_stdout)[0]["deleted"] is True
    assert persisted[2] == ()


def test_cli_4_editor_success_updates_memory_user_and_exact_skill_document():
    edited_memory = "Edited memory"
    edited_user = "Edited user profile"
    edited_skill = (
        "# monthly-revenue\n\nEdited description\n\n"
        "## Instructions\n\nEdited instructions\n"
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "editor-success"))
        asyncio.run(_seed_knowledge(root, "editor-success"))

        editor = _replacement_editor(root, edited_memory, name="memory")
        with patch.dict(os.environ, {"EDITOR": editor}, clear=True):
            memory_result = _invoke(
                ["--root", str(root), "memory", "edit", "editor-success"]
            )
        editor = _replacement_editor(root, edited_user, name="user")
        with patch.dict(os.environ, {"EDITOR": editor}, clear=True):
            user_result = _invoke(
                [
                    "--root",
                    str(root),
                    "memory",
                    "edit",
                    "editor-success",
                    "--target",
                    "user",
                ]
            )
        editor = _replacement_editor(root, edited_skill, name="skill")
        with patch.dict(os.environ, {"EDITOR": editor}, clear=True):
            skill_result = _invoke(
                [
                    "--root",
                    str(root),
                    "skills",
                    "edit",
                    "editor-success",
                    "monthly-revenue",
                ]
            )
        memory, user, _ = asyncio.run(_complete_knowledge(root, "editor-success"))
        _, skill = asyncio.run(_knowledge(root, "editor-success"))

    assert tuple(
        result[0] for result in (memory_result, user_result, skill_result)
    ) == (
        0,
        0,
        0,
    )
    assert tuple(
        result[2] for result in (memory_result, user_result, skill_result)
    ) == (
        "",
        "",
        "",
    )
    assert (memory, user) == (edited_memory, edited_user)
    assert skill == Skill(
        "monthly-revenue",
        "Edited description",
        "Edited instructions",
    )


@pytest.mark.parametrize(
    ("editor", "expected"),
    [
        (None, "$EDITOR is not set"),
        ("   ", "$EDITOR is not set"),
        ("'unterminated", "$EDITOR is malformed"),
        ("daita-editor-that-does-not-exist", "$EDITOR command is unavailable"),
        (
            shlex.join((sys.executable, "-c", "import sys; sys.exit(7)")),
            "$EDITOR exited with status 7",
        ),
    ],
)
def test_cli_4_editor_configuration_failures_never_write_and_release_lock(
    editor: str | None,
    expected: str,
):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "editor-failure"))
        asyncio.run(_seed_knowledge(root, "editor-failure"))
        environment = {} if editor is None else {"EDITOR": editor}
        with patch.dict(os.environ, environment, clear=True):
            code, stdout, stderr = _invoke(
                ["--root", str(root), "memory", "edit", "editor-failure"]
            )
        knowledge = asyncio.run(_complete_knowledge(root, "editor-failure"))
        asyncio.run(_open_and_close_agent(root, "editor-failure"))

    assert code == 1
    assert stdout == ""
    assert expected in stderr
    assert knowledge[:2] == ("Initial memory", "Initial user profile")


def test_cli_4_editor_interruption_never_writes_and_releases_lock():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "editor-interrupt"))
        asyncio.run(_seed_knowledge(root, "editor-interrupt"))
        with (
            patch.dict(os.environ, {"EDITOR": "unused-editor"}, clear=True),
            patch.object(cli.subprocess, "run", side_effect=KeyboardInterrupt),
        ):
            code, stdout, stderr = _invoke(
                ["--root", str(root), "memory", "edit", "editor-interrupt"]
            )
        knowledge = asyncio.run(_complete_knowledge(root, "editor-interrupt"))
        asyncio.run(_open_and_close_agent(root, "editor-interrupt"))

    assert code == 130
    assert stdout == ""
    assert stderr == "Chat interrupted.\n"
    assert knowledge[:2] == ("Initial memory", "Initial user profile")


def test_cli_4_malformed_skill_edit_never_partially_writes_and_releases_lock():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "malformed-edit"))
        asyncio.run(_seed_knowledge(root, "malformed-edit"))
        editor = _replacement_editor(
            root,
            "# wrong-name\n\nChanged\n\n## Instructions\n\nChanged\n",
            name="malformed",
        )
        with patch.dict(os.environ, {"EDITOR": editor}, clear=True):
            code, stdout, stderr = _invoke(
                [
                    "--root",
                    str(root),
                    "skills",
                    "edit",
                    "malformed-edit",
                    "monthly-revenue",
                ]
            )
        _, skill = asyncio.run(_knowledge(root, "malformed-edit"))
        asyncio.run(_open_and_close_agent(root, "malformed-edit"))

    assert code == 1
    assert stdout == ""
    assert "edited skill must keep the exact" in stderr
    assert skill == Skill(
        "monthly-revenue",
        "Initial description",
        "Initial instructions",
    )


def test_cli_4_file_and_public_validation_failures_preserve_prior_state():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "invalid-input"))
        asyncio.run(_seed_knowledge(root, "invalid-input"))
        invalid_utf8 = root / "invalid-utf8.txt"
        invalid_utf8.write_bytes(b"valid-prefix\xff")
        instructions = root / "instructions.txt"
        instructions.write_text("Valid instructions", encoding="utf-8")

        failures = (
            _invoke(
                [
                    "--root",
                    str(root),
                    "memory",
                    "set",
                    "invalid-input",
                    "--target",
                    "memory",
                    "--file",
                    str(invalid_utf8),
                ]
            ),
            _invoke(
                [
                    "--root",
                    str(root),
                    "memory",
                    "set",
                    "invalid-input",
                    "--target",
                    "memory",
                    "--file",
                    str(root / "missing.txt"),
                ]
            ),
            _invoke(
                [
                    "--root",
                    str(root),
                    "memory",
                    "set",
                    "invalid-input",
                    "--target",
                    "memory",
                    "--file",
                    "-",
                ],
                stdin="x" * 2_201,
            ),
            _invoke(
                [
                    "--root",
                    str(root),
                    "skills",
                    "save",
                    "invalid-input",
                    "INVALID_NAME",
                    "--description",
                    "Invalid skill",
                    "--instructions-file",
                    str(instructions),
                ]
            ),
            _invoke(
                [
                    "--root",
                    str(root),
                    "skills",
                    "save",
                    "invalid-input",
                    "monthly-revenue",
                    "--description",
                    "Changed description",
                    "--instructions-file",
                    "-",
                ],
                stdin="x" * 12_001,
            ),
        )
        knowledge = asyncio.run(_complete_knowledge(root, "invalid-input"))
        asyncio.run(_open_and_close_agent(root, "invalid-input"))

    assert all(code == 1 and stdout == "" for code, stdout, _ in failures)
    assert "codec can't decode" in failures[0][2]
    assert "No such file" in failures[1][2]
    assert "2200 character limit" in failures[2][2]
    assert "must match [a-z][a-z0-9-]{0,63}" in failures[3][2]
    assert "12000 character limit" in failures[4][2]
    assert knowledge[0] == "Initial memory"
    assert knowledge[2] == (SkillSummary("monthly-revenue", "Initial description"),)


def test_cli_4_chat_knowledge_commands_are_local_and_preserve_conversation():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "chat-knowledge"))
        asyncio.run(_seed_knowledge(root, "chat-knowledge"))
        seed_provider = MockModelProvider((_stop("seed answer"),))
        local_provider = MockModelProvider(())
        with patch.object(
            cli,
            "create_llm_provider",
            side_effect=(seed_provider, local_provider),
        ):
            seed_code, seed_stdout, _ = _invoke(
                [
                    "--root",
                    str(root),
                    "run",
                    "chat-knowledge",
                    "seed question",
                    "--model",
                    "mock:scripted",
                ]
            )
            conversation_id = str(_json_lines(seed_stdout)[0]["conversation_id"])
            code, stdout, stderr = _invoke(
                [
                    "--root",
                    str(root),
                    "chat",
                    "chat-knowledge",
                    "--model",
                    "mock:scripted",
                    "--conversation",
                    conversation_id,
                ],
                stdin=(
                    "/memory\n/user\n/skills\n"
                    "/skills show monthly-revenue\n/status\n/exit\n"
                ),
                tty=True,
            )

    assert seed_code == code == 0
    assert stderr == ""
    assert "Initial memory" in stdout
    assert "Initial user profile" in stdout
    assert "Initial instructions" in stdout
    assert f"Conversation: {conversation_id}" in stdout
    assert "This process: 0 turns, 0 steps, 0 tokens" in stdout
    assert local_provider.requests == ()
    seed_provider.assert_consumed()
    local_provider.assert_consumed()


def test_cli_4_chat_skill_alias_and_fallback_are_ordinary_model_runs():
    for invocation in (
        "/monthly-revenue investigate June",
        "/skills use monthly-revenue investigate June",
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            asyncio.run(_create_agent(root, "chat-invocation"))
            asyncio.run(_seed_knowledge(root, "chat-invocation"))
            provider = MockModelProvider(
                (
                    _call("skill_view", {"name": "monthly-revenue"}),
                    _stop("investigation complete"),
                )
            )
            with patch.object(cli, "create_llm_provider", return_value=provider):
                code, stdout, stderr = _invoke(
                    [
                        "--root",
                        str(root),
                        "chat",
                        "chat-invocation",
                        "--model",
                        "mock:scripted",
                    ],
                    stdin=f"{invocation}\n/exit\n",
                    tty=True,
                )

        assert code == 0
        assert stderr == ""
        assert _request_text(provider.requests[0]) == (invocation,)
        assert "investigation complete" in stdout
        provider.assert_consumed()


def test_cli_4_chat_learn_routes_to_an_ordinary_foreground_run():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "chat-learn"))
        material = "Always give me the summary before the detail."
        provider = MockModelProvider((_stop("I can propose that preference."),))
        with patch.object(cli, "create_llm_provider", return_value=provider):
            code, stdout, stderr = _invoke(
                [
                    "--root",
                    str(root),
                    "chat",
                    "chat-learn",
                    "--model",
                    "mock:scripted",
                ],
                stdin=f"/learn {material}\n/exit\n",
                tty=True,
            )

    assert code == 0
    assert stderr == ""
    routed = _request_text(provider.requests[0])
    assert len(routed) == 1
    assert routed[0].startswith("Treat the following as an explicit teaching request.")
    assert "approval card is the only confirmation" in routed[0]
    assert "never ask the user for a typed approval phrase" in routed[0]
    assert routed[0].endswith(material)
    assert "I can propose that preference." in stdout
    provider.assert_consumed()


def test_cli_4_chat_skill_create_is_local_and_create_only():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "chat-create"))
        editor_path = root / "skill-editor.py"
        editor_path.write_text(
            "from pathlib import Path\n"
            "import sys\n"
            "path = Path(sys.argv[-1])\n"
            "path.write_text(\n"
            "    '# customer-health-investigation\\n\\n'\n"
            "    'Investigate customer health.\\n\\n'\n"
            "    '## Instructions\\n\\n'\n"
            "    'Compare account, support, and usage signals.\\n',\n"
            "    encoding='utf-8',\n"
            ")\n",
            encoding="utf-8",
        )
        editor = shlex.join((sys.executable, str(editor_path)))
        provider = MockModelProvider(())
        with (
            patch.dict(os.environ, {"EDITOR": editor}, clear=True),
            patch.object(cli, "create_llm_provider", return_value=provider),
        ):
            code, stdout, stderr = _invoke(
                [
                    "--root",
                    str(root),
                    "chat",
                    "chat-create",
                    "--model",
                    "mock:scripted",
                ],
                stdin="/skills create customer-health-investigation\n/exit\n",
                tty=True,
            )
        knowledge = asyncio.run(_complete_knowledge(root, "chat-create"))

    assert code == 0
    assert stderr == ""
    assert "Invoke it with /customer-health-investigation" in stdout
    assert (
        SkillSummary(
            "customer-health-investigation",
            "Investigate customer health.",
        )
        in knowledge[2]
    )
    assert provider.requests == ()
    provider.assert_consumed()


def test_cli_4_chat_skill_create_without_name_runs_guided_flow():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "chat-create-wizard"))
        provider = MockModelProvider(())
        with patch.object(cli, "create_llm_provider", return_value=provider):
            code, stdout, stderr = _invoke(
                [
                    "--root",
                    str(root),
                    "chat",
                    "chat-create-wizard",
                    "--model",
                    "mock:scripted",
                ],
                stdin=(
                    "/skills create\n"
                    "customer-health-investigation\n"
                    "Investigate customer health.\n"
                    "Inspect the aggregate snapshot.\n"
                    "Validate it against current evidence.\n"
                    ".\n"
                    "/exit\n"
                ),
                tty=True,
            )
        knowledge = asyncio.run(_complete_knowledge(root, "chat-create-wizard"))

    assert code == 0
    assert stderr == ""
    assert "Name:" in stdout
    assert "Description:" in stdout
    assert "finish with a single . on its own line" in stdout
    assert (
        SkillSummary(
            "customer-health-investigation",
            "Investigate customer health.",
        )
        in knowledge[2]
    )
    assert provider.requests == ()
    provider.assert_consumed()


def test_cli_4_ordinary_knowledge_words_stay_messages_and_local_commands_keep_totals():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "ordinary-knowledge"))
        asyncio.run(_seed_knowledge(root, "ordinary-knowledge"))
        provider = MockModelProvider((_stop("ordinary answer"),))
        with patch.object(cli, "create_llm_provider", return_value=provider):
            code, stdout, stderr = _invoke(
                [
                    "--root",
                    str(root),
                    "chat",
                    "ordinary-knowledge",
                    "--model",
                    "mock:scripted",
                ],
                stdin="memory\n/memory\n/status\n/exit\n",
                tty=True,
            )

    assert code == 0
    assert stderr == ""
    assert _request_text(provider.requests[0]) == ("memory",)
    assert len(provider.requests) == 1
    assert "Initial memory" in stdout
    assert "This process: 1 turns, 1 steps" in stdout
    provider.assert_consumed()


def test_cli_4_chat_skill_delete_defaults_to_no_and_requires_explicit_yes():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "chat-delete"))
        asyncio.run(_seed_knowledge(root, "chat-delete"))
        provider = MockModelProvider(())
        with patch.object(cli, "create_llm_provider", return_value=provider):
            code, stdout, stderr = _invoke(
                [
                    "--root",
                    str(root),
                    "chat",
                    "chat-delete",
                    "--model",
                    "mock:scripted",
                ],
                stdin=(
                    "/skills delete monthly-revenue\n\n"
                    "/skills show monthly-revenue\n"
                    "/skills delete monthly-revenue\ny\n/exit\n"
                ),
                tty=True,
            )
        _, skill = asyncio.run(_knowledge(root, "chat-delete"))

    assert code == 0
    assert stderr == ""
    assert "Deletion cancelled." in stdout
    assert "Initial instructions" in stdout
    assert "Skill 'monthly-revenue' deleted." in stdout
    assert skill is None
    assert provider.requests == ()
    provider.assert_consumed()


def test_cli_4_chat_editor_commands_are_local_and_use_public_mutations():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "chat-editor"))
        asyncio.run(_seed_knowledge(root, "chat-editor"))
        editor_path = root / "selective-editor.py"
        editor_path.write_text(
            "from pathlib import Path\n"
            "import sys\n"
            "path = Path(sys.argv[-1])\n"
            "text = path.read_text(encoding='utf-8')\n"
            "if text.startswith('# monthly-revenue\\n'):\n"
            "    replacement = '# monthly-revenue\\n\\nChat description\\n\\n## "
            "Instructions\\n\\nChat instructions\\n'\n"
            "elif text == 'Initial memory':\n"
            "    replacement = 'Chat memory'\n"
            "else:\n"
            "    replacement = 'Chat user profile'\n"
            "path.write_text(replacement, encoding='utf-8')\n",
            encoding="utf-8",
        )
        editor = shlex.join((sys.executable, str(editor_path)))
        provider = MockModelProvider(())
        with (
            patch.dict(os.environ, {"EDITOR": editor}, clear=True),
            patch.object(cli, "create_llm_provider", return_value=provider),
        ):
            code, stdout, stderr = _invoke(
                [
                    "--root",
                    str(root),
                    "chat",
                    "chat-editor",
                    "--model",
                    "mock:scripted",
                ],
                stdin=(
                    "/memory edit\n/user edit\n" "/skills edit monthly-revenue\n/exit\n"
                ),
                tty=True,
            )
        memory, user, _ = asyncio.run(_complete_knowledge(root, "chat-editor"))
        _, skill = asyncio.run(_knowledge(root, "chat-editor"))

    assert code == 0
    assert stderr == ""
    assert "Memory updated." in stdout
    assert "User updated." in stdout
    assert "Skill 'monthly-revenue' updated." in stdout
    assert (memory, user) == ("Chat memory", "Chat user profile")
    assert skill == Skill(
        "monthly-revenue",
        "Chat description",
        "Chat instructions",
    )
    assert provider.requests == ()
    provider.assert_consumed()


def test_cli_4_shell_mutations_delegate_through_public_agent_methods_only():
    class FakeAgent:
        def __init__(self) -> None:
            self.set_memory = AsyncMock()
            self.save_skill = AsyncMock(return_value=True)
            self.close = AsyncMock()

    fake = FakeAgent()
    open_agent = AsyncMock(return_value=fake)
    with patch.object(cli.Agent, "open", open_agent):
        memory_result = _invoke(
            [
                "memory",
                "set",
                "public-only",
                "--target",
                "memory",
                "--file",
                "-",
            ],
            stdin="Public memory",
        )
        skill_result = _invoke(
            [
                "skills",
                "save",
                "public-only",
                "public-skill",
                "--description",
                "Public description",
                "--instructions-file",
                "-",
            ],
            stdin="Public instructions",
        )

    assert memory_result[0] == skill_result[0] == 0
    fake.set_memory.assert_awaited_once_with("Public memory")
    fake.save_skill.assert_awaited_once_with(
        "public-skill",
        "Public description",
        "Public instructions",
    )
    assert fake.close.await_count == 2
