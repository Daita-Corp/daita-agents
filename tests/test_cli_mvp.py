from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import shlex
import stat
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from _workspace_support import workspace_for

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
    agent = await Agent.create(name, root=root, workspace=workspace_for(root))
    await agent.close()


async def _open_and_close_agent(root: Path, name: str) -> None:
    agent = await Agent.open(name, root=root, workspace=workspace_for(root))
    await agent.close()


async def _knowledge(root: Path, name: str) -> tuple[str, Skill | None]:
    agent = await Agent.open(name, root=root, workspace=workspace_for(root))
    try:
        return await agent.read_memory(), await agent.read_skill("monthly-revenue")
    finally:
        await agent.close()


async def _complete_knowledge(
    root: Path,
    name: str,
) -> tuple[str, str, tuple[object, ...]]:
    agent = await Agent.open(name, root=root, workspace=workspace_for(root))
    try:
        return (
            await agent.read_memory(),
            await agent.read_user_profile(),
            await agent.list_skills(),
        )
    finally:
        await agent.close()


async def _seed_knowledge(root: Path, name: str) -> None:
    agent = await Agent.open(name, root=root, workspace=workspace_for(root))
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
        frozenset(
            {
                "-h",
                "--help",
                "--version",
                "--root",
                "--agent",
                "--workspace",
                "--workspace-sensitivity",
            }
        ),
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


def test_workspace_resolution_prefers_explicit_then_safe_cwd_and_preserves_sensitivity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    explicit_root = tmp_path / "explicit"
    explicit_root.mkdir()
    safe_cwd = tmp_path / "project"
    safe_cwd.mkdir()

    explicit_args = cli.build_parser().parse_args(
        [
            "--root",
            str(state_root),
            "--workspace",
            str(explicit_root),
            "--workspace-sensitivity",
            "confidential",
            "run",
            "agent",
            "question",
        ]
    )
    explicit = cli._resolve_cli_workspace(explicit_args)
    assert explicit.root == explicit_root.resolve()
    assert explicit.sensitivity.value == "confidential"

    monkeypatch.chdir(safe_cwd)
    cwd_args = cli.build_parser().parse_args(
        ["--root", str(state_root), "run", "agent", "question"]
    )
    inferred = cli._resolve_cli_workspace(cwd_args)
    assert inferred.root == safe_cwd.resolve()
    assert inferred.sensitivity.value == "internal"


def test_workspace_resolution_creates_user_only_conventional_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_home = tmp_path / "home"
    user_home.mkdir()
    monkeypatch.chdir(user_home)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: user_home))
    args = cli.build_parser().parse_args(["run", "agent", "question"])

    workspace = cli._resolve_cli_workspace(args)

    assert workspace.root == user_home / "Daita Workspace"
    assert workspace.root.is_dir()
    assert stat.S_IMODE(workspace.root.stat().st_mode) == 0o700


def test_delete_is_state_only_and_does_not_resolve_a_workspace() -> None:
    args = cli.build_parser().parse_args(["delete", "gone", "--yes"])
    with (
        patch.object(
            cli,
            "_resolve_cli_workspace",
            side_effect=AssertionError("delete must not resolve workspace authority"),
        ),
        patch.object(Agent, "delete", new=AsyncMock()) as deleted,
    ):
        result = asyncio.run(cli._execute(args))

    assert result == {"name": "gone", "deleted": True}
    deleted.assert_awaited_once_with("gone", root=None)


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


def test_zero_argument_cli_starts_textual_without_preapp_prompts():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        with patch.object(
            cli,
            "run_terminal_application",
            new=AsyncMock(return_value=0),
        ) as run_terminal:
            code, stdout, stderr = _invoke(
                ["--root", str(root)],
                stdin="atlas\n",
                tty=True,
            )

        assert code == 0
        assert stdout == ""
        assert stderr == ""
        run_terminal.assert_awaited_once()
        await_args = run_terminal.await_args
        assert await_args is not None
        assert await_args.kwargs["root"] == root
        assert await_args.kwargs["agent_name"] is None
        assert asyncio.run(Agent.list(root=root)) == ()


def test_agent_option_is_passed_to_the_textual_app():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        asyncio.run(_create_agent(root, "alpha"))
        asyncio.run(_create_agent(root, "atlas"))
        with patch.object(
            cli,
            "run_terminal_application",
            new=AsyncMock(return_value=0),
        ) as run_terminal:
            code, stdout, stderr = _invoke(
                ["--root", str(root), "--agent", "atlas"],
                tty=True,
            )

        assert code == 0
        assert stdout == ""
        assert stderr == ""
        await_args = run_terminal.await_args
        assert await_args is not None
        assert await_args.kwargs["agent_name"] == "atlas"


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


def test_chat_subcommand_is_a_strict_alias_of_the_textual_app():
    with patch.object(
        cli,
        "run_terminal_application",
        new=AsyncMock(return_value=0),
    ) as run_terminal:
        code, stdout, stderr = _invoke(
            ["chat", "atlas", "--model", "mock:scripted"],
            tty=True,
        )

    assert code == 0
    assert stdout == ""
    assert stderr == ""
    run_terminal.assert_awaited_once()
    await_args = run_terminal.await_args
    assert await_args is not None
    assert await_args.kwargs["agent_name"] == "atlas"


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


def test_run_without_model_opens_the_persisted_route_without_injection():
    result = SimpleNamespace(
        run_id="run-persisted",
        conversation_id="conversation-persisted",
        kind=SimpleNamespace(value="completed"),
        reason="completed",
        final_text="persisted answer",
        steps=1,
        artifacts=(),
        artifact_deliveries=(),
    )
    agent = AsyncMock()
    agent.run.return_value = result
    arguments = cli.build_parser().parse_args(["run", "runner", "use persisted route"])

    with (
        patch.object(Agent, "open", new=AsyncMock(return_value=agent)) as opened,
        patch.object(
            cli,
            "_model_configuration",
            side_effect=AssertionError("default run must not construct an override"),
        ),
    ):
        record = asyncio.run(cli._execute(arguments))

    assert isinstance(record, dict)
    assert record["text"] == "persisted answer"
    opened.assert_awaited_once_with(
        "runner",
        workspace=cli._resolve_cli_workspace(arguments),
        root=None,
        observer=None,
    )
    agent.run.assert_awaited_once_with(
        "use persisted route",
        conversation_id=None,
    )
    agent.close.assert_awaited_once()


def test_run_override_options_require_an_explicit_invocation_local_model():
    arguments = cli.build_parser().parse_args(
        ["run", "runner", "question", "--base-url", "https://models.invalid"]
    )

    with pytest.raises(ValueError, match="require --model"):
        asyncio.run(cli._execute(arguments))


def test_run_without_model_reports_missing_persisted_configuration():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        create_code, _stdout, _stderr = _invoke(
            ["--root", str(root), "create", "unconfigured-runner"]
        )
        assert create_code == 0
        code, stdout, stderr = _invoke(
            [
                "--root",
                str(root),
                "run",
                "unconfigured-runner",
                "question",
            ]
        )

    assert code == 1
    assert stdout == ""
    assert "agent execution requires a model" in stderr.lower()


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
                "--files-only",
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
        "host",
        "memory",
        "mcp",
        "skills",
        "routines",
        "inbox",
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

    assert _surface(commands["host"]) == (
        (),
        frozenset({"-h", "--help", "--agent"}),
    )
    routines = _subcommands(commands["routines"])
    assert set(routines) == {
        "list",
        "inspect",
        "create",
        "promote",
        "update",
        "pause",
        "resume",
        "run-now",
        "disable",
    }
    assert _surface(routines["list"]) == (
        ("name",),
        frozenset({"-h", "--help", "--state"}),
    )
    assert _surface(routines["inspect"]) == (
        ("name", "routine_id"),
        frozenset({"-h", "--help"}),
    )
    inbox = _subcommands(commands["inbox"])
    assert set(inbox) == {"destinations", "list", "inspect", "acknowledge"}
    assert _surface(inbox["destinations"]) == (
        ("name", "conversation_id"),
        frozenset({"-h", "--help", "--sensitivity-ceiling"}),
    )
    assert _surface(inbox["list"]) == (
        ("name",),
        frozenset(
            {
                "-h",
                "--help",
                "--conversation-id",
                "--include-acknowledged",
                "--limit",
            }
        ),
    )
    for command in ("inspect", "acknowledge"):
        assert _surface(inbox[command]) == (
            ("name", "delivery_id"),
            frozenset({"-h", "--help"}),
        )


def test_cli_routine_list_uses_public_agent_surface(tmp_path: Path) -> None:
    now = datetime(2026, 8, 28, 12, tzinfo=UTC)
    fake = SimpleNamespace(
        list_routines=AsyncMock(
            return_value=(
                SimpleNamespace(
                    routine_id="routine-1",
                    title="Current value",
                    state=SimpleNamespace(value="active"),
                    schedule_kind=SimpleNamespace(value="interval"),
                    next_due_at=now,
                    revision=2,
                    occurrence_count=1,
                    consecutive_failures=0,
                ),
            )
        ),
        close=AsyncMock(),
    )
    args = cli.build_parser().parse_args(
        ["--root", str(tmp_path), "routines", "list", "atlas"]
    )
    with patch.object(Agent, "open", new=AsyncMock(return_value=fake)):
        result = asyncio.run(cli._execute(args))
    assert result == [
        {
            "routine_id": "routine-1",
            "title": "Current value",
            "state": "active",
            "schedule_kind": "interval",
            "next_due_at": now.isoformat(),
            "revision": 2,
            "occurrence_count": 1,
            "consecutive_failures": 0,
        }
    ]
    fake.list_routines.assert_awaited_once()
    fake.close.assert_awaited_once()


def test_cli_host_dispatches_one_resident_composition(tmp_path: Path) -> None:
    args = cli.build_parser().parse_args(
        ["--root", str(tmp_path), "host", "--agent", "atlas"]
    )
    with patch.object(cli, "run_resident_host", new=AsyncMock()) as hosted:
        result = asyncio.run(cli._execute(args))
    assert result == {"agent": "atlas", "host": "stopped"}
    hosted.assert_awaited_once()
    await_args = hosted.await_args
    assert await_args is not None
    assert await_args.kwargs["agent_name"] == "atlas"


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
