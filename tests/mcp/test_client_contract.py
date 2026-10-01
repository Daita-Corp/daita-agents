"""Agent owns injected protocol clients without depending on their SDK type."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

import pytest

from daita import Agent, MCPAuthentication, MCPToolSelection
from daita._json import FrozenJsonObject
from daita.adapters.mcp import (
    MCPAuthenticationError,
    MCPAuthenticationMode,
    MCPClient,
    MCPClientFactory,
    MCPConnectionProvider,
    MCPInspectedTool,
    MCPPersonalConnectionClient,
    MCPServerInspection,
    MCPToolResult,
    canonical_mcp_schema,
)
from daita.llm.models import (
    FinishReason,
    ModelResponse,
    ToolCall,
    ToolResultBlock,
)
from daita.security import EmptySecretProvider, SecretProvider, SecretReference
from tests.support.toolbox_model import ToolboxAwareMockModelProvider

NOW = datetime(2026, 9, 30, tzinfo=UTC)
ENDPOINT = "https://injected-client.example.test/mcp"


class ClientOptions(TypedDict):
    root: Path
    hosted: bool
    clock: Callable[[], datetime]
    secret_provider: SecretProvider
    mcp_client_factory: MCPClientFactory
    mcp_connection_provider: MCPConnectionProvider


class RecordedClient:
    def __init__(self, inspection: MCPServerInspection) -> None:
        self.inspection = inspection
        self.inspections: list[datetime] = []
        self.calls: list[tuple[str, FrozenJsonObject]] = []
        self.close_calls = 0

    async def inspect(self, *, observed_at: datetime) -> MCPServerInspection:
        assert self.close_calls == 0
        self.inspections.append(observed_at)
        return replace(self.inspection, observed_at=observed_at)

    async def call_tool(
        self, remote_name: str, arguments: Mapping[str, object]
    ) -> MCPToolResult:
        assert self.close_calls == 0
        self.calls.append((remote_name, FrozenJsonObject.from_mapping(arguments)))
        return MCPToolResult(text=("injected result",))

    async def close(self) -> None:
        self.close_calls += 1


class PersonalRecordedClient(RecordedClient):
    def __init__(self, inspection: MCPServerInspection, *, reject: bool) -> None:
        super().__init__(inspection)
        self.reject = reject
        self.bindings: list[tuple[MCPConnectionProvider, str]] = []

    def bind_personal_connection(
        self, provider: MCPConnectionProvider, principal_id: str
    ) -> None:
        self.bindings.append((provider, principal_id))
        if self.reject:
            raise MCPAuthenticationError(
                "account_unavailable", "The injected connection is unavailable."
            )


class RecordedFactory:
    def __init__(self) -> None:
        self.clients: list[RecordedClient] = []
        self.arguments: list[tuple[str, MCPAuthentication, SecretProvider]] = []
        self.personal_support = True
        self.reject_binding = False

    def create(
        self,
        *,
        endpoint: str,
        authentication: MCPAuthentication,
        secrets: SecretProvider,
    ) -> MCPClient:
        schema, digest = canonical_mcp_schema(
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            }
        )
        inspection = MCPServerInspection(
            endpoint=endpoint,
            protocol_version="2026-07-28",
            server_name=None,
            server_version=None,
            observed_at=NOW,
            tools=(
                MCPInspectedTool(
                    remote_name="lookup",
                    remote_description=None,
                    input_schema=schema,
                    input_schema_digest=digest,
                    output_schema=None,
                    output_schema_digest=None,
                    supported=True,
                ),
            ),
        )
        client = (
            PersonalRecordedClient(inspection, reject=self.reject_binding)
            if authentication.mode is MCPAuthenticationMode.PERSONAL_CONNECTION
            and self.personal_support
            else RecordedClient(inspection)
        )
        self.arguments.append((endpoint, authentication, secrets))
        self.clients.append(client)
        return client


class RecordedConnections:
    def __init__(self) -> None:
        self.claims: list[tuple[str, str, str, tuple[str, ...]]] = []

    async def check_access(
        self,
        *,
        connection_id: str,
        principal_id: str,
        resource_uri: str,
        required_scopes: tuple[str, ...],
    ) -> None:
        claims = (connection_id, principal_id, resource_uri, required_scopes)
        assert claims == ("alice-connection", "alice", ENDPOINT, ("read:records",))
        self.claims.append(claims)

    async def access_token(
        self,
        *,
        connection_id: str,
        principal_id: str,
        resource_uri: str,
        required_scopes: tuple[str, ...],
    ) -> str:
        raise AssertionError("Injected protocol fixture performs no credential I/O")


@pytest.mark.parametrize("mode", tuple(MCPAuthenticationMode))
async def test_injected_factory_owns_fresh_inspection_and_persistent_binding_clients(
    tmp_path: Path, mode: MCPAuthenticationMode
) -> None:
    recording = RecordedFactory()
    factory: MCPClientFactory = recording
    connections = RecordedConnections()
    secrets = EmptySecretProvider()
    authentication = {
        MCPAuthenticationMode.NONE: MCPAuthentication.no_auth(),
        MCPAuthenticationMode.BEARER: MCPAuthentication.bearer(
            SecretReference.environment("INJECTED_TOKEN")
        ),
        MCPAuthenticationMode.PERSONAL_CONNECTION: MCPAuthentication.personal_connection(
            "alice-connection", "alice", ENDPOINT, ("read:records",)
        ),
    }[mode]
    options: ClientOptions = {
        "root": tmp_path,
        "hosted": True,
        "clock": lambda: NOW,
        "secret_provider": secrets,
        "mcp_client_factory": factory,
        "mcp_connection_provider": connections,
    }
    agent = await Agent.create("injected-mcp", **options)
    try:
        assert recording.clients == []
        status = await agent.attach_mcp_server(
            endpoint=ENDPOINT,
            authentication=authentication,
            selections=(MCPToolSelection("lookup", "lookup", "Read a fixture."),),
            caller_principal_id="alice",
        )
        assert len(recording.clients) == 1
        assert recording.clients[0].close_calls == 1
    finally:
        await agent.close()
    tool = status.binding.tools[0].local_name
    model = ToolboxAwareMockModelProvider(
        (
            *(
                ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(
                        ToolCall(
                            id=f"read-{index}",
                            name=tool,
                            arguments={"query": str(index)},
                        ),
                    ),
                )
                for index in range(2)
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="done"),
        )
    )
    agent = await Agent.open(
        "injected-mcp", model=model, model_profile=model.model_profile, **options
    )
    try:
        assert len(recording.clients) == 1
        result = await agent.run("Read twice.", caller_principal_id="alice")
        transcript = await agent.transcript(result.run_id, caller_principal_id="alice")
        successful = [
            block
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
            and block.call_id in {"read-0", "read-1"}
        ]
        assert len(successful) == 2 and all(not block.is_error for block in successful)
        assert len(recording.clients) == 2
        client = recording.clients[1]
        assert client.inspections == [NOW, NOW]
        assert [(name, args.to_dict()) for name, args in client.calls] == [
            ("lookup", {"query": "0"}),
            ("lookup", {"query": "1"}),
        ]
        assert all(
            args == (ENDPOINT, authentication, secrets) for args in recording.arguments
        )
        if mode is MCPAuthenticationMode.PERSONAL_CONNECTION:
            assert connections.claims
            for created in recording.clients:
                assert isinstance(created, MCPPersonalConnectionClient)
                assert isinstance(created, PersonalRecordedClient)
                assert created.bindings == [(connections, "alice")]
        else:
            assert connections.claims == []
            assert not isinstance(client, MCPPersonalConnectionClient)
        await agent.revoke_mcp_server(
            status.binding.binding_id, caller_principal_id="alice"
        )
        assert client.close_calls == 1
    finally:
        await agent.close()
    assert all(client.close_calls == 1 for client in recording.clients)


@pytest.mark.parametrize("failure", ["unsupported", "rejected"])
@pytest.mark.parametrize("stage", ["inspection", "execution"])
async def test_personal_client_binding_failure_closes_before_inspection_or_dispatch(
    tmp_path: Path, failure: str, stage: str
) -> None:
    factory = RecordedFactory()
    connections = RecordedConnections()
    authentication = MCPAuthentication.personal_connection(
        "alice-connection", "alice", ENDPOINT, ("read:records",)
    )
    options: ClientOptions = {
        "root": tmp_path,
        "hosted": True,
        "clock": lambda: NOW,
        "secret_provider": EmptySecretProvider(),
        "mcp_client_factory": factory,
        "mcp_connection_provider": connections,
    }
    agent = await Agent.create("rejected-client", **options)
    if stage == "execution":
        try:
            status = await agent.attach_mcp_server(
                endpoint=ENDPOINT,
                authentication=authentication,
                selections=(MCPToolSelection("lookup", "lookup", "Read a fixture."),),
                caller_principal_id="alice",
            )
        finally:
            await agent.close()
        model = ToolboxAwareMockModelProvider(
            (
                ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(
                        ToolCall(
                            id="rejected",
                            name=status.binding.tools[0].local_name,
                            arguments={"query": "x"},
                        ),
                    ),
                ),
                ModelResponse(finish_reason=FinishReason.STOP, text="done"),
            )
        )
        agent = await Agent.open(
            "rejected-client", model=model, model_profile=model.model_profile, **options
        )
    factory.personal_support = failure != "unsupported"
    factory.reject_binding = failure == "rejected"
    try:
        if stage == "inspection":
            with pytest.raises(MCPAuthenticationError) as caught:
                await agent.inspect_mcp_server(
                    endpoint=ENDPOINT,
                    authentication=authentication,
                    caller_principal_id="alice",
                )
            assert caught.value.code == "account_unavailable"
        else:
            result = await agent.run("Read.", caller_principal_id="alice")
            transcript = await agent.transcript(
                result.run_id, caller_principal_id="alice"
            )
            (block,) = (
                block
                for message in transcript.messages
                for block in message.content
                if isinstance(block, ToolResultBlock) and block.call_id == "rejected"
            )
            assert block.is_error
            error = block.output["error"]
            assert isinstance(error, Mapping)
            assert error["code"] == "account_unavailable"
        failed = factory.clients[-1]
        assert failed.close_calls == 1
        assert failed.inspections == [] and failed.calls == []
    finally:
        await agent.close()
    assert failed.close_calls == 1
