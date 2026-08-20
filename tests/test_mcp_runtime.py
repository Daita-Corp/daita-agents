from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime

import httpx
import pytest
from textual.widgets import Button, Input, OptionList, Static

from daita import cli
from daita import (
    Agent,
    MCPAdmissionError,
    MCPAuthentication,
    MCPBindingState,
    MCPToolSelection,
)
from daita.adapters.mcp import StreamableHTTPMCPClientFactory
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolResultBlock,
    ModelSensitivity,
)
from daita.security import SecretReference
from daita.tui.commands import SLASH_COMMAND_COMPLETIONS
from daita.tui.app import DaitaApp
from daita.tui.controller import PresentationController
from daita.tui.models import UserInputError
from daita.tui.screens.chat import ChatScreen
from daita.tui.screens.confirm import ConfirmScreen
from daita.tui.screens.mcp import (
    MCPManagementScreen,
    MCPSetupScreen,
    generated_mcp_aliases,
)
from daita.tui.screens.selection import SelectionScreen
from daita.tui.widgets.composer import Composer
from _mcp_fixtures import (
    MCPConformanceTransport,
    MCPFixtureIdentity,
    MappingSecretProvider,
    conformance_identities,
    mock_transport,
)

NOW = datetime(2026, 8, 19, 12, 0, tzinfo=UTC)


class _MCPBatchProvider:
    provider_id = "mock:mcp-batch"

    def __init__(
        self,
        calls: tuple[ToolCall, ...],
        *,
        block_first_response: bool = False,
    ) -> None:
        self.model_profile = ModelProfile(
            id=self.provider_id,
            context_window_tokens=128_000,
            max_output_tokens=8_192,
            supports_tools=True,
            supports_parallel_tools=True,
        )
        self.calls = calls
        self.requests: list[ModelRequest] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        if not block_first_response:
            self.release.set()

    def supports_request_policy(self, request: ModelRequest) -> bool:
        del request
        return True

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            self.started.set()
            await self.release.wait()
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=self.calls,
            )
        return ModelResponse(finish_reason=FinishReason.STOP, text="done")


class _MCPSequenceProvider:
    provider_id = "mock:mcp-sequence"

    def __init__(self, calls: tuple[tuple[ToolCall, ...], ...]) -> None:
        self.model_profile = ModelProfile(
            id=self.provider_id,
            context_window_tokens=128_000,
            max_output_tokens=8_192,
            supports_tools=True,
            supports_parallel_tools=True,
        )
        self.calls = calls
        self.requests: list[ModelRequest] = []

    def supports_request_policy(self, request: ModelRequest) -> bool:
        del request
        return True

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        index = len(self.requests) - 1
        if index < len(self.calls):
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=self.calls[index],
            )
        return ModelResponse(finish_reason=FinishReason.STOP, text="done")


class _BlockingCallTimeInspection:
    def __init__(self, identity) -> None:
        self._transport = MCPConformanceTransport(identity)
        self._list_count = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if payload.get("method") == "tools/list":
            self._list_count += 1
            if self._list_count == 3:
                self.started.set()
                await self.release.wait()
        return await self._transport(request)


def _error_code(result: ToolResultBlock) -> str:
    error = result.output["error"]
    assert isinstance(error, Mapping)
    code = error["code"]
    assert isinstance(code, str)
    return code


async def _attach_two_bindings(tmp_path):
    alpha, beta = conformance_identities()
    secrets = MappingSecretProvider({"env:BETA_TOKEN": "fixture-beta-secret"})
    factory = StreamableHTTPMCPClientFactory(http_transport=mock_transport(alpha, beta))
    agent = await Agent.create(
        "mcp-multi-binding",
        root=tmp_path,
        clock=lambda: NOW,
        secret_provider=secrets,
        mcp_client_factory=factory,
    )
    inspection = await agent.inspect_mcp_server(endpoint=alpha.endpoint)
    assert inspection.server_name == "fixture-alpha"
    assert await agent.list_mcp_servers() == ()

    alpha_status = await agent.attach_mcp_server(
        endpoint=alpha.endpoint,
        selections=(
            MCPToolSelection(
                remote_name="lookup",
                local_alias="alpha_lookup",
                description="Read the admitted alpha fixture value.",
            ),
        ),
    )
    beta_status = await agent.attach_mcp_server(
        endpoint=beta.endpoint,
        authentication=MCPAuthentication.bearer(
            SecretReference.environment("BETA_TOKEN")
        ),
        selections=(
            MCPToolSelection(
                remote_name="lookup",
                local_alias="beta_lookup",
                description="Read the admitted beta fixture value.",
            ),
        ),
    )
    assert alpha_status.reopen_required
    assert beta_status.reopen_required
    assert alpha_status.binding.tools[0].local_name != (
        beta_status.binding.tools[0].local_name
    )
    assert {tool.remote_name for tool in alpha_status.binding.tools} == {"lookup"}

    with sqlite3.connect(agent.home / "state.db") as connection:
        durable = "\n".join(
            row[0] for row in connection.execute("SELECT data FROM mcp_server_bindings")
        )
    assert "fixture-beta-secret" not in durable
    assert "env:BETA_TOKEN" in durable
    return agent, alpha, beta, secrets, factory, alpha_status, beta_status


async def test_multi_binding_reopen_executes_through_normal_runtime_and_transcript(
    tmp_path,
):
    (
        agent,
        alpha,
        beta,
        secrets,
        factory,
        alpha_status,
        beta_status,
    ) = await _attach_two_bindings(tmp_path)
    alpha_name = alpha_status.binding.tools[0].local_name
    beta_name = beta_status.binding.tools[0].local_name
    await agent.close()

    provider = _MCPBatchProvider(
        (
            ToolCall(id="alpha-call", name=alpha_name, arguments={"query": "x"}),
            ToolCall(id="beta-call", name=beta_name, arguments={"id": 7}),
        )
    )
    reopened = await Agent.open(
        "mcp-multi-binding",
        root=tmp_path,
        clock=lambda: NOW,
        model=provider,
        model_profile=provider.model_profile,
        secret_provider=secrets,
        mcp_client_factory=factory,
    )
    try:
        statuses = await reopened.list_mcp_servers()
        assert all(status.active_in_runtime for status in statuses)
        result = await reopened.run("Use both admitted reads.")
        transcript = await reopened.transcript(result.run_id)
        blocks = tuple(
            block
            for message in transcript.messages
            if message.role is MessageRole.TOOL
            for block in message.content
            if isinstance(block, ToolResultBlock)
        )
        assert [block.call_id for block in blocks] == ["alpha-call", "beta-call"]
        assert all(not block.is_error for block in blocks)
        data = blocks[0].output["data"]
        assert isinstance(data, Mapping)
        provenance = data["provenance"]
        assert isinstance(provenance, Mapping)
        assert provenance["remote_tool_name"] == "lookup"
        assert alpha.calls == [("lookup", {"query": "x"})]
        assert beta.calls == [("lookup", {"id": 7})]

        first_request = provider.requests[0]
        assert {tool.name for tool in first_request.tools} >= {alpha_name, beta_name}
        assert "IGNORE ALL PRIOR INSTRUCTIONS" not in repr(first_request.tools)
        second_request = provider.requests[1]
        assert "IGNORE ALL PRIOR INSTRUCTIONS" in repr(second_request.messages)
        assert "untrusted" in repr(second_request.messages).lower()
    finally:
        await reopened.close()


async def test_existing_binding_identity_cannot_be_redirected_to_another_server(
    tmp_path,
):
    alpha, beta = conformance_identities()
    secrets = MappingSecretProvider({"env:BETA_TOKEN": "fixture-beta-secret"})
    factory = StreamableHTTPMCPClientFactory(http_transport=mock_transport(alpha, beta))
    agent = await Agent.create(
        "mcp-stable-binding",
        root=tmp_path,
        clock=lambda: NOW,
        secret_provider=secrets,
        mcp_client_factory=factory,
    )
    try:
        first = await agent.attach_mcp_server(
            endpoint=alpha.endpoint,
            selections=(
                MCPToolSelection(
                    remote_name="lookup",
                    local_alias="alpha_lookup",
                    description="Read the admitted alpha fixture value.",
                ),
            ),
        )
        with pytest.raises(MCPAdmissionError) as raised:
            await agent.attach_mcp_server(
                binding_id=first.binding.binding_id,
                endpoint=beta.endpoint,
                authentication=MCPAuthentication.bearer(
                    SecretReference.environment("BETA_TOKEN")
                ),
                selections=(
                    MCPToolSelection(
                        remote_name="lookup",
                        local_alias="beta_lookup",
                        description="Read the admitted beta fixture value.",
                    ),
                ),
            )
        assert raised.value.code == "mcp_binding_remote_changed"
        (stored,) = await agent.list_mcp_servers()
        assert stored.binding.endpoint == alpha.endpoint
        assert stored.binding.revision == first.binding.revision
    finally:
        await agent.close()


async def test_cli_and_tui_expose_bounded_mcp_administration(tmp_path, monkeypatch):
    alpha, _beta = conformance_identities()
    factory = StreamableHTTPMCPClientFactory(http_transport=mock_transport(alpha))
    agent = await Agent.create(
        "mcp-administration-surface",
        root=tmp_path,
        clock=lambda: NOW,
        mcp_client_factory=factory,
    )

    async def open_agent(*args, **kwargs):
        del args, kwargs
        return agent

    monkeypatch.setattr(cli.Agent, "open", open_agent)
    arguments = cli.build_parser().parse_args(
        (
            "--root",
            str(tmp_path),
            "mcp",
            "attach",
            "mcp-administration-surface",
            alpha.endpoint,
            "--tool",
            "lookup",
            "lookup",
            "Read an admitted fixture value.",
            "internal",
        )
    )
    mapping = await cli._execute(arguments)
    monkeypatch.undo()
    assert isinstance(mapping, Mapping)
    assert mapping["endpoint"] == alpha.endpoint
    assert mapping["reopen_required"] is True
    assert "authentication" not in repr(mapping).lower()
    binding_id = mapping["binding_id"]
    assert isinstance(binding_id, str)

    reopened = await Agent.open(
        "mcp-administration-surface",
        root=tmp_path,
        clock=lambda: NOW,
        mcp_client_factory=factory,
    )
    controller = PresentationController(root=tmp_path)
    controller.agent = reopened
    try:
        shown = await controller.dispatch_command("/mcp")
        assert shown.kind == "screen"
        assert shown.screen == "mcp_management"
        details = await controller.dispatch_command("/mcp status")
        assert details.kind == "notice"
        assert binding_id in details.message
        assert "restart required" not in details.message
        assert "ready" in details.message
        confirmation = await controller.dispatch_command(f"/mcp revoke {binding_id}")
        assert confirmation.kind == "confirm"
        assert confirmation.screen == "confirm_revoke_mcp"
        assert any(
            display == "/mcp"
            for _insertion, display, _description in SLASH_COMMAND_COMPLETIONS
        )
    finally:
        await reopened.close()


def _guided_mcp_identity() -> MCPFixtureIdentity:
    return MCPFixtureIdentity(
        host="guided-mcp.fixture.test",
        server_name="Context Fixture",
        server_version="4.0.1",
        protocol_version="2025-11-25",
        tools=[
            {
                "name": "query-docs",
                "description": "Remote text is not trusted as a tool description.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "resolve-library-id",
                "inputSchema": {
                    "type": "object",
                    "properties": {"library": {"type": "string"}},
                    "required": ["library"],
                    "additionalProperties": False,
                },
            },
        ],
        results={
            "query-docs": {"content": [{"type": "text", "text": "docs"}]},
            "resolve-library-id": {"content": [{"type": "text", "text": "library-id"}]},
        },
    )


def test_guided_mcp_aliases_are_safe_stable_and_collision_free():
    aliases = generated_mcp_aliases(
        (
            "resolve-library-id",
            "resolve_library_id",
            "123.lookup",
            "資料",
            "a" * 80,
            "a" * 79 + "-b",
        )
    )
    assert aliases[:4] == (
        "resolve_library_id",
        "resolve_library_id_2",
        "tool_123_lookup",
        "remote_tool",
    )
    assert len(aliases) == len(set(aliases))
    assert all(len(alias) <= 40 for alias in aliases)
    assert all(
        alias[0].isalpha() and alias.replace("_", "").isalnum() for alias in aliases
    )


async def test_mcp_management_groups_legacy_bindings_by_server(tmp_path):
    identity = _guided_mcp_identity()
    factory = StreamableHTTPMCPClientFactory(http_transport=mock_transport(identity))
    agent = await Agent.create(
        "mcp-grouped-management",
        root=tmp_path,
        clock=lambda: NOW,
        mcp_client_factory=factory,
    )
    first = await agent.attach_mcp_server(
        endpoint=identity.endpoint,
        selections=(
            MCPToolSelection(
                remote_name="query-docs",
                local_alias="query_docs",
                description="Read explicitly admitted MCP documentation.",
            ),
        ),
    )
    second = await agent.attach_mcp_server(
        endpoint=identity.endpoint,
        selections=(
            MCPToolSelection(
                remote_name="resolve-library-id",
                local_alias="resolve_library_id",
                description="Read an explicitly admitted MCP library identifier.",
            ),
        ),
    )
    await agent.close()
    reopened = await Agent.open(
        "mcp-grouped-management",
        root=tmp_path,
        clock=lambda: NOW,
        mcp_client_factory=factory,
    )
    app = DaitaApp(root=tmp_path, start_bootstrap=False)
    app.controller.agent = reopened
    async with app.run_test(size=(100, 34)) as pilot:
        await app._show_chat()
        composer = app.screen.query_one(Composer)
        composer.load_text("/mcp")
        composer.action_submit()
        await pilot.pause()

        assert isinstance(app.screen, MCPManagementScreen)
        summary = str(app.screen.query_one("#mcp-summary", Static).content)
        body = str(app.screen.query_one("#mcp-body", Static).content)
        assert summary == "1 server  ·  2 tools"
        assert "Context Fixture 4.0.1  ·  Ready" in body
        assert "query-docs" in body
        assert "resolve-library-id" in body
        assert first.binding.binding_id not in body
        assert second.binding.binding_id not in body
        assert "active  active" not in body
        app.exit(0)


async def test_guided_mcp_setup_attaches_one_multi_tool_binding_and_activates(
    tmp_path,
):
    identity = _guided_mcp_identity()
    factory = StreamableHTTPMCPClientFactory(http_transport=mock_transport(identity))
    opened = await Agent.create(
        "mcp-guided-setup",
        root=tmp_path,
        clock=lambda: NOW,
        mcp_client_factory=factory,
    )
    app = DaitaApp(root=tmp_path, start_bootstrap=False)
    app.controller.agent = opened
    reopen_calls: list[bool] = []

    async def reopen_with_fixture(*, observer, approval_handler):
        reopen_calls.append(observer is not None and approval_handler is not None)
        await app.controller.close_agent()
        reopened = await Agent.open(
            "mcp-guided-setup",
            root=tmp_path,
            clock=lambda: NOW,
            mcp_client_factory=factory,
            observer=observer,
            approval_handler=approval_handler,
        )
        app.controller.agent = reopened
        return reopened

    app.controller.reopen_agent = reopen_with_fixture  # type: ignore[method-assign]
    async with app.run_test(size=(104, 38)) as pilot:
        await app._show_chat()
        command_task = asyncio.create_task(app._open_command_screen("mcp_setup", {}))
        await pilot.pause()

        assert isinstance(app.screen, MCPSetupScreen)
        app.screen.query_one("#mcp-endpoint", Input).value = identity.endpoint
        assert await pilot.click("#mcp-inspect") is True
        await pilot.pause()
        inspection_text = str(
            app.screen.query_one("#mcp-inspection-body", Static).content
        )
        assert "Context Fixture 4.0.1" in inspection_text
        assert "Supported tools: 2" in inspection_text

        assert await pilot.click("#mcp-select") is True
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, SelectionScreen)
        listing = picker.query_one("#picker-options", OptionList)
        listing.highlighted = 0
        picker.action_toggle_selected()
        listing.highlighted = 1
        picker.action_toggle_selected()
        picker.action_confirm()
        await pilot.pause()

        assert isinstance(app.screen, MCPSetupScreen)
        review = str(app.screen.query_one("#mcp-inspection-body", Static).content)
        assert "Alias: query_docs" in review
        assert "Alias: resolve_library_id" in review
        assert "Result sensitivity: internal" in review
        assert "Remote descriptions and annotations are untrusted" in review

        assert await pilot.click("#mcp-attach") is True
        await pilot.pause()
        attestation = app.screen
        assert isinstance(attestation, ConfirmScreen)
        await pilot.press("y")
        for _ in range(100):
            await pilot.pause(0.05)
            if isinstance(app.screen, ConfirmScreen) and app.screen is not attestation:
                break
        activation = app.screen
        assert isinstance(activation, ConfirmScreen)
        assert activation is not attestation
        await pilot.press("y")
        await command_task

        assert isinstance(app.screen, ChatScreen)
        statuses = await app.controller.list_mcp_servers()
        assert len(statuses) == 1
        assert statuses[0].active_in_runtime
        assert reopen_calls == [True]
        assert {tool.remote_name for tool in statuses[0].binding.tools} == {
            "query-docs",
            "resolve-library-id",
        }
        app.exit(0)


async def test_mcp_management_refresh_and_revoke_do_not_require_typed_ids(tmp_path):
    identity = _guided_mcp_identity()
    factory = StreamableHTTPMCPClientFactory(http_transport=mock_transport(identity))
    agent = await Agent.create(
        "mcp-guided-actions",
        root=tmp_path,
        clock=lambda: NOW,
        mcp_client_factory=factory,
    )
    attached = await agent.attach_mcp_server(
        endpoint=identity.endpoint,
        selections=(
            MCPToolSelection(
                remote_name="query-docs",
                local_alias="query_docs",
                description="Read explicitly admitted MCP documentation.",
            ),
            MCPToolSelection(
                remote_name="resolve-library-id",
                local_alias="resolve_library_id",
                description="Read an explicitly admitted MCP library identifier.",
            ),
        ),
    )
    await agent.close()
    reopened = await Agent.open(
        "mcp-guided-actions",
        root=tmp_path,
        clock=lambda: NOW,
        mcp_client_factory=factory,
    )
    app = DaitaApp(root=tmp_path, start_bootstrap=False)
    app.controller.agent = reopened
    async with app.run_test(size=(104, 38)) as pilot:
        await app._show_chat()
        command_task = asyncio.create_task(
            app._open_command_screen("mcp_management", {})
        )
        await pilot.pause()
        assert isinstance(app.screen, MCPManagementScreen)
        assert app.screen.query_one("#mcp-restart", Button).disabled is True

        assert await pilot.click("#mcp-refresh") is True
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, SelectionScreen)
        listing = picker.query_one("#picker-options", OptionList)
        prompt = str(listing.get_option_at_index(0).prompt)
        assert "Context Fixture" in prompt
        assert "query-docs" in prompt
        assert attached.binding.binding_id not in prompt
        listing.highlighted = 0
        picker.action_confirm()
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        await pilot.press("n")
        await pilot.pause()

        assert isinstance(app.screen, MCPManagementScreen)
        (refreshed,) = await app.controller.list_mcp_servers()
        assert refreshed.reopen_required
        assert app.screen.query_one("#mcp-restart", Button).disabled is False

        assert await pilot.click("#mcp-revoke") is True
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, SelectionScreen)
        picker.query_one("#picker-options", OptionList).highlighted = 0
        picker.action_confirm()
        await pilot.pause()
        confirmation = app.screen
        assert isinstance(confirmation, ConfirmScreen)
        message = str(confirmation.query_one("#confirm-message", Static).content)
        assert "Context Fixture" in message
        assert "query-docs" in message
        assert attached.binding.binding_id not in message
        await pilot.press("y")
        await pilot.pause()

        assert isinstance(app.screen, MCPManagementScreen)
        (revoked,) = await app.controller.list_mcp_servers()
        assert revoked.binding.state is MCPBindingState.REVOKED
        assert "Revoked" in str(app.screen.query_one("#mcp-body", Static).content)
        assert attached.binding.binding_id not in str(
            app.screen.query_one("#mcp-help", Static).content
        )
        assert app.screen.query_one("#mcp-restart", Button).disabled is True
        app.screen.action_close()
        await command_task
        app.exit(0)


async def test_tui_attach_exposes_bounded_schema_rejection_reason(tmp_path):
    identity = MCPFixtureIdentity(
        host="unsupported-tui.fixture.test",
        server_name="unsupported-tui-fixture",
        server_version="1",
        protocol_version="2025-11-25",
        tools=[
            {
                "name": "unsupported",
                "inputSchema": {
                    "type": "object",
                    "properties": {"value": {"$ref": "https://invalid/schema"}},
                },
            }
        ],
        results={},
    )
    factory = StreamableHTTPMCPClientFactory(http_transport=mock_transport(identity))
    agent = await Agent.create(
        "mcp-tui-schema-reason",
        root=tmp_path,
        clock=lambda: NOW,
        mcp_client_factory=factory,
    )
    controller = PresentationController(root=tmp_path)
    controller.agent = agent
    try:
        with pytest.raises(
            UserInputError,
            match=r"Cannot attach MCP tool: unsupported schema keyword: \$ref",
        ):
            await controller.dispatch_command(
                f"/mcp attach {identity.endpoint} unsupported unsupported"
            )
        assert await agent.list_mcp_servers() == ()
    finally:
        await agent.close()


async def test_revocation_after_frozen_context_blocks_io_and_is_binding_isolated(
    tmp_path,
):
    (
        agent,
        alpha,
        beta,
        secrets,
        factory,
        alpha_status,
        beta_status,
    ) = await _attach_two_bindings(tmp_path)
    await agent.close()
    alpha_name = alpha_status.binding.tools[0].local_name
    beta_name = beta_status.binding.tools[0].local_name
    provider = _MCPBatchProvider(
        (
            ToolCall(id="revoked-call", name=alpha_name, arguments={"query": "x"}),
            ToolCall(id="sibling-call", name=beta_name, arguments={"id": 9}),
        ),
        block_first_response=True,
    )
    reopened = await Agent.open(
        "mcp-multi-binding",
        root=tmp_path,
        clock=lambda: NOW,
        model=provider,
        model_profile=provider.model_profile,
        secret_provider=secrets,
        mcp_client_factory=factory,
    )
    try:
        run = asyncio.create_task(reopened.run("Prepare both calls."))
        await provider.started.wait()
        before_alpha_calls = len(alpha.calls)
        revoked = await reopened.revoke_mcp_server(alpha_status.binding.binding_id)
        assert revoked.binding.state is MCPBindingState.REVOKED
        provider.release.set()
        exit_record = await run
        transcript = await reopened.transcript(exit_record.run_id)
        blocks = tuple(
            block
            for message in transcript.messages
            if message.role is MessageRole.TOOL
            for block in message.content
            if isinstance(block, ToolResultBlock)
        )
        assert [block.call_id for block in blocks] == ["revoked-call", "sibling-call"]
        assert _error_code(blocks[0]) == "tool_not_available"
        assert not blocks[1].is_error
        assert len(alpha.calls) == before_alpha_calls
        assert beta.calls[-1] == ("lookup", {"id": 9})
        statuses = await reopened.list_mcp_servers()
        by_id = {status.binding.binding_id: status for status in statuses}
        assert by_id[beta_status.binding.binding_id].active_in_runtime
    finally:
        provider.release.set()
        await reopened.close()


async def test_revocation_serializes_with_call_time_inspection(tmp_path):
    alpha, _beta = conformance_identities()
    transport = _BlockingCallTimeInspection(alpha)
    factory = StreamableHTTPMCPClientFactory(
        http_transport=httpx.MockTransport(transport)
    )
    agent = await Agent.create(
        "mcp-revocation-linearization",
        root=tmp_path,
        clock=lambda: NOW,
        mcp_client_factory=factory,
    )
    status = await agent.attach_mcp_server(
        endpoint=alpha.endpoint,
        selections=(
            MCPToolSelection(
                remote_name="lookup",
                local_alias="lookup",
                description="Read the admitted fixture value.",
            ),
        ),
    )
    await agent.close()
    provider = _MCPBatchProvider(
        (
            ToolCall(
                id="in-flight",
                name=status.binding.tools[0].local_name,
                arguments={"query": "x"},
            ),
        )
    )
    reopened = await Agent.open(
        "mcp-revocation-linearization",
        root=tmp_path,
        clock=lambda: NOW,
        model=provider,
        model_profile=provider.model_profile,
        mcp_client_factory=factory,
    )
    try:
        run = asyncio.create_task(reopened.run("Run the admitted call."))
        await asyncio.wait_for(transport.started.wait(), timeout=1)
        revoke = asyncio.create_task(
            reopened.revoke_mcp_server(status.binding.binding_id)
        )
        await asyncio.sleep(0)
        (during_call,) = await reopened.list_mcp_servers()
        assert during_call.binding.state is MCPBindingState.ACTIVE
        assert not revoke.done()

        transport.release.set()
        await asyncio.wait_for(run, timeout=1)
        revoked = await asyncio.wait_for(revoke, timeout=1)
        assert alpha.calls == [("lookup", {"query": "x"})]
        assert revoked.binding.state is MCPBindingState.REVOKED
        calls_after_revoke = tuple(alpha.calls)
        await asyncio.sleep(0)
        assert tuple(alpha.calls) == calls_after_revoke
    finally:
        transport.release.set()
        await reopened.close()


async def test_schema_drift_is_unavailable_until_explicit_refresh_and_reopen(tmp_path):
    (
        agent,
        _alpha,
        beta,
        secrets,
        factory,
        _alpha_status,
        beta_status,
    ) = await _attach_two_bindings(tmp_path)
    original_schema = dict(beta.tool("lookup")["inputSchema"])
    await agent.close()
    beta.tool("lookup")["inputSchema"] = {
        "type": "object",
        "properties": {"changed": {"type": "string"}},
        "required": ["changed"],
    }

    drifted = await Agent.open(
        "mcp-multi-binding",
        root=tmp_path,
        clock=lambda: NOW,
        secret_provider=secrets,
        mcp_client_factory=factory,
    )
    try:
        by_id = {
            status.binding.binding_id: status
            for status in await drifted.list_mcp_servers()
        }
        assert not by_id[beta_status.binding.binding_id].active_in_runtime
        refreshed = await drifted.refresh_mcp_server(beta_status.binding.binding_id)
        assert refreshed.binding.state is MCPBindingState.STALE
        assert refreshed.binding.stale_reason == "tool_schema_changed:lookup"

        beta.tool("lookup")["inputSchema"] = original_schema
        recovered = await drifted.refresh_mcp_server(beta_status.binding.binding_id)
        assert recovered.binding.state is MCPBindingState.ACTIVE
        assert recovered.reopen_required
    finally:
        await drifted.close()

    reopened = await Agent.open(
        "mcp-multi-binding",
        root=tmp_path,
        clock=lambda: NOW,
        secret_provider=secrets,
        mcp_client_factory=factory,
    )
    try:
        by_id = {
            status.binding.binding_id: status
            for status in await reopened.list_mcp_servers()
        }
        assert by_id[beta_status.binding.binding_id].active_in_runtime
    finally:
        await reopened.close()


async def test_public_outbound_and_call_time_auth_use_current_admission(tmp_path):
    alpha, beta = conformance_identities()
    secrets = MappingSecretProvider({"env:BETA_TOKEN": "fixture-beta-secret"})
    factory = StreamableHTTPMCPClientFactory(http_transport=mock_transport(alpha, beta))
    agent = await Agent.create(
        "mcp-boundary-failures",
        root=tmp_path,
        clock=lambda: NOW,
        secret_provider=secrets,
        mcp_client_factory=factory,
    )
    public_status = await agent.attach_mcp_server(
        endpoint=alpha.endpoint,
        maximum_outbound_sensitivity=ModelSensitivity.PUBLIC,
        selections=(
            MCPToolSelection(
                remote_name="lookup",
                local_alias="public_only",
                description="Read through a public-only outbound binding.",
            ),
        ),
    )
    bearer_status = await agent.attach_mcp_server(
        endpoint=beta.endpoint,
        authentication=MCPAuthentication.bearer(
            SecretReference.environment("BETA_TOKEN")
        ),
        selections=(
            MCPToolSelection(
                remote_name="lookup",
                local_alias="bearer_lookup",
                description="Read through a bearer binding.",
            ),
        ),
    )
    await agent.close()
    provider = _MCPBatchProvider(
        (
            ToolCall(
                id="sensitivity-call",
                name=public_status.binding.tools[0].local_name,
                arguments={"query": "x"},
            ),
            ToolCall(
                id="auth-call",
                name=bearer_status.binding.tools[0].local_name,
                arguments={"id": 1},
            ),
        )
    )
    reopened = await Agent.open(
        "mcp-boundary-failures",
        root=tmp_path,
        clock=lambda: NOW,
        model=provider,
        model_profile=provider.model_profile,
        secret_provider=secrets,
        mcp_client_factory=factory,
    )
    try:
        secrets.values["env:BETA_TOKEN"] = "wrong-at-call-time"
        result = await reopened.run("Exercise safe boundary failures.")
        transcript = await reopened.transcript(result.run_id)
        blocks = tuple(
            block
            for message in transcript.messages
            if message.role is MessageRole.TOOL
            for block in message.content
            if isinstance(block, ToolResultBlock)
        )
        assert not blocks[0].is_error
        assert _error_code(blocks[1]) == "mcp_authentication_failed"
        assert alpha.calls == [("lookup", {"query": "x"})]
        assert beta.calls == []
        assert "wrong-at-call-time" not in repr(blocks)
    finally:
        await reopened.close()


async def test_current_run_sensitivity_blocks_later_lower_ceiling_egress(tmp_path):
    alpha, _beta = conformance_identities()
    factory = StreamableHTTPMCPClientFactory(http_transport=mock_transport(alpha))
    agent = await Agent.create(
        "mcp-sensitivity-floor",
        root=tmp_path,
        clock=lambda: NOW,
        mcp_client_factory=factory,
    )
    high = await agent.attach_mcp_server(
        endpoint=alpha.endpoint,
        maximum_outbound_sensitivity=ModelSensitivity.INTERNAL,
        selections=(
            MCPToolSelection(
                remote_name="lookup",
                local_alias="confidential_result",
                description="Return a confidential classified result.",
                result_sensitivity=ModelSensitivity.CONFIDENTIAL,
            ),
        ),
    )
    low = await agent.attach_mcp_server(
        endpoint=alpha.endpoint,
        maximum_outbound_sensitivity=ModelSensitivity.INTERNAL,
        selections=(
            MCPToolSelection(
                remote_name="lookup",
                local_alias="internal_egress",
                description="Use an internal-only outbound binding.",
            ),
        ),
    )
    await agent.close()
    provider = _MCPSequenceProvider(
        (
            (
                ToolCall(
                    id="raise-floor",
                    name=high.binding.tools[0].local_name,
                    arguments={"query": "x"},
                ),
            ),
            (
                ToolCall(
                    id="blocked-egress",
                    name=low.binding.tools[0].local_name,
                    arguments={"query": "confidential-derived"},
                ),
            ),
        )
    )
    reopened = await Agent.open(
        "mcp-sensitivity-floor",
        root=tmp_path,
        clock=lambda: NOW,
        model=provider,
        model_profile=provider.model_profile,
        mcp_client_factory=factory,
    )
    try:
        result = await reopened.run("Exercise the monotonic run floor.")
        transcript = await reopened.transcript(result.run_id)
        blocks = tuple(
            block
            for message in transcript.messages
            if message.role is MessageRole.TOOL
            for block in message.content
            if isinstance(block, ToolResultBlock)
        )
        assert tuple(request.sensitivity for request in provider.requests) == (
            ModelSensitivity.PUBLIC,
            ModelSensitivity.CONFIDENTIAL,
            ModelSensitivity.CONFIDENTIAL,
        )
        assert blocks[0].sensitivity is ModelSensitivity.CONFIDENTIAL
        assert blocks[0].sensitivity_provenance["run_sensitivity_floor"] == "public"
        assert _error_code(blocks[1]) == "mcp_outbound_sensitivity_exceeded"
        assert alpha.calls == [("lookup", {"query": "x"})]
    finally:
        await reopened.close()


async def test_host_close_waits_for_remote_call_then_closes_mcp_client(tmp_path):
    alpha, _beta = conformance_identities()
    alpha.block_calls = asyncio.Event()
    factory = StreamableHTTPMCPClientFactory(http_transport=mock_transport(alpha))
    agent = await Agent.create(
        "mcp-close",
        root=tmp_path,
        clock=lambda: NOW,
        mcp_client_factory=factory,
    )
    status = await agent.attach_mcp_server(
        endpoint=alpha.endpoint,
        selections=(
            MCPToolSelection(
                remote_name="lookup",
                local_alias="close_lookup",
                description="Read while host close is requested.",
            ),
        ),
    )
    await agent.close()
    provider = _MCPBatchProvider(
        (
            ToolCall(
                id="close-call",
                name=status.binding.tools[0].local_name,
                arguments={"query": "x"},
            ),
        )
    )
    reopened = await Agent.open(
        "mcp-close",
        root=tmp_path,
        clock=lambda: NOW,
        model=provider,
        model_profile=provider.model_profile,
        mcp_client_factory=factory,
    )
    activated = tuple(reopened._embedded._mcp_activated_bindings.values())
    run = asyncio.create_task(reopened.run("Use the blocking MCP read."))
    while "tools/call" not in alpha.request_methods:
        await asyncio.sleep(0)
    closing = asyncio.create_task(reopened.close())
    await asyncio.sleep(0)
    assert not closing.done()
    alpha.block_calls.set()
    await run
    await closing
    assert all(getattr(item.client, "_closed", False) for item in activated)


async def test_oversized_remote_result_becomes_one_bounded_transcript_error(tmp_path):
    alpha, _beta = conformance_identities()
    factory = StreamableHTTPMCPClientFactory(http_transport=mock_transport(alpha))
    agent = await Agent.create(
        "mcp-oversized-result",
        root=tmp_path,
        clock=lambda: NOW,
        mcp_client_factory=factory,
    )
    status = await agent.attach_mcp_server(
        endpoint=alpha.endpoint,
        selections=(
            MCPToolSelection(
                remote_name="lookup",
                local_alias="bounded_lookup",
                description="Read a bounded remote value.",
            ),
        ),
    )
    await agent.close()
    alpha.results["lookup"] = {
        "content": [{"type": "text", "text": "REMOTE-SECRET" * 30_000}],
        "structuredContent": {"answer": "unused"},
    }
    provider = _MCPBatchProvider(
        (
            ToolCall(
                id="oversized-call",
                name=status.binding.tools[0].local_name,
                arguments={"query": "x"},
            ),
        )
    )
    reopened = await Agent.open(
        "mcp-oversized-result",
        root=tmp_path,
        clock=lambda: NOW,
        model=provider,
        model_profile=provider.model_profile,
        mcp_client_factory=factory,
    )
    try:
        result = await reopened.run("Exercise the result boundary.")
        transcript = await reopened.transcript(result.run_id)
        block = next(
            block
            for message in transcript.messages
            if message.role is MessageRole.TOOL
            for block in message.content
            if isinstance(block, ToolResultBlock)
        )
        assert _error_code(block) == "mcp_result_too_large"
        assert "REMOTE-SECRET" not in repr(block)
    finally:
        await reopened.close()
