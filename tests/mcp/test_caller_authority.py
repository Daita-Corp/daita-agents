"""Personal MCP connections stay bound to one verified host caller."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256

import httpx2 as httpx
import pytest

from daita import (
    Agent,
    ApprovalDecision,
    EffectResolutionDecision,
    MCPAuthentication,
    MCPToolSelection,
)
from daita._json import FrozenJsonObject
from daita.adapters.mcp import (
    MCPAuthenticationError,
    MCPPersonalConnectionClient,
    SDKMCPClientFactory,
    mcp_execution_origin_digest,
)
from daita.capabilities import (
    AccessMode,
    ApprovalRequest,
    CapabilityInputError,
    ExecutionContractBindings,
    ExecutionScope,
    ExecutionScopeKind,
    OperationalEffect,
    ToolExecution,
)
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelResponse,
    ModelSensitivity,
    ToolCall,
    ToolResultBlock,
)
from daita.loop.models import (
    InstructionAuthority,
    RunInput,
    RunOrigin,
    RunStartEnvelope,
)
from daita.security import EmptySecretProvider
from tests.support.mcp import (
    MCPBatchProvider,
    MCPConformanceTransport,
    conformance_identities,
    mock_transport,
)
from tests.support.toolbox_model import ToolboxAwareMockModelProvider
from tests.support.workspace import workspace_for

NOW = datetime(2026, 9, 29, tzinfo=UTC)
TOKEN = "test-only-personal-token-never-persist"


class PersonalConnections:
    def __init__(self, resource_uri: str, owner_principal_id: str = "alice") -> None:
        self.resource_uri = resource_uri
        self.owner_principal_id = owner_principal_id
        self.status: str | None = None
        self.checks: list[tuple[str, str, str, tuple[str, ...]]] = []
        self.token_requests = 0

    async def check_access(
        self,
        *,
        connection_id: str,
        principal_id: str,
        resource_uri: str,
        required_scopes: tuple[str, ...],
    ) -> None:
        self.checks.append((connection_id, principal_id, resource_uri, required_scopes))
        if self.status is not None:
            raise MCPAuthenticationError(self.status, f"secret: {TOKEN}")
        if (
            connection_id != "connection-alice"
            or principal_id != self.owner_principal_id
            or resource_uri != self.resource_uri
            or required_scopes != ("read:records",)
        ):
            raise MCPAuthenticationError("needs_authorization", f"secret: {TOKEN}")

    async def access_token(
        self,
        *,
        connection_id: str,
        principal_id: str,
        resource_uri: str,
        required_scopes: tuple[str, ...],
    ) -> str:
        await self.check_access(
            connection_id=connection_id,
            principal_id=principal_id,
            resource_uri=resource_uri,
            required_scopes=required_scopes,
        )
        self.token_requests += 1
        return TOKEN


@pytest.mark.parametrize("state", ["bound", "started", "closed"])
async def test_sdk_personal_client_binds_only_once_before_first_use(state):
    identity, _ = conformance_identities()
    identity.bearer_token = TOKEN
    provider = PersonalConnections(identity.endpoint)
    client = SDKMCPClientFactory(http_transport=mock_transport(identity)).create(
        endpoint=identity.endpoint,
        authentication=MCPAuthentication.personal_connection(
            "connection-alice", "alice", identity.endpoint, ("read:records",)
        ),
        secrets=EmptySecretProvider(),
    )
    assert isinstance(client, MCPPersonalConnectionClient)
    try:
        if state == "bound":
            client.bind_personal_connection(provider, "alice")
        elif state == "started":
            with pytest.raises(MCPAuthenticationError):
                await client.inspect(observed_at=NOW)
        else:
            await client.close()
        with pytest.raises(ValueError, match="once before use"):
            client.bind_personal_connection(provider, "alice")
        assert provider.token_requests == 0
        assert identity.calls == []
    finally:
        await client.close()


async def _attached(tmp_path, *, action: bool = False):
    identity, _ = conformance_identities()
    identity.bearer_token = TOKEN
    vault = PersonalConnections(identity.endpoint)
    factory = SDKMCPClientFactory(http_transport=mock_transport(identity))
    authentication = MCPAuthentication.personal_connection(
        "connection-alice", "alice", identity.endpoint, ("read:records",)
    )
    agent = await Agent.create(
        "personal-mcp",
        root=tmp_path,
        hosted=True,
        clock=lambda: NOW,
        mcp_client_factory=factory,
        mcp_connection_provider=vault,
    )
    status = await agent.attach_mcp_server(
        endpoint=identity.endpoint,
        authentication=authentication,
        selections=(
            MCPToolSelection(
                remote_name="lookup",
                local_alias="lookup",
                description="Call a fixture tool.",
                access_mode=AccessMode.NONE if action else AccessMode.READ,
                operational_effect=(
                    OperationalEffect.EXTERNAL_ACTION
                    if action
                    else OperationalEffect.NONE
                ),
            ),
        ),
        caller_principal_id="alice",
    )
    await agent.close()
    return identity, vault, factory, status.binding


def _tool_error(transcript, call_id: str) -> str:
    blocks = (
        block
        for message in transcript.messages
        if message.role is MessageRole.TOOL
        for block in message.content
        if isinstance(block, ToolResultBlock) and block.call_id == call_id
    )
    block = next(blocks)
    error = block.output["error"]
    assert isinstance(error, Mapping)
    code = error["code"]
    assert isinstance(code, str)
    return code


async def test_local_personal_projection_checks_once_per_binding_and_rechecks_next_time(
    tmp_path,
):
    identity, _ = conformance_identities()
    identity.bearer_token = TOKEN
    vault = PersonalConnections(identity.endpoint)
    factory = SDKMCPClientFactory(http_transport=mock_transport(identity))
    agent = await Agent.create(
        "personal-projection-cost",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        clock=lambda: NOW,
        mcp_client_factory=factory,
        mcp_connection_provider=vault,
    )
    try:
        vault.owner_principal_id = agent.id
        status = await agent.attach_mcp_server(
            endpoint=identity.endpoint,
            authentication=MCPAuthentication.personal_connection(
                "connection-alice", agent.id, identity.endpoint, ("read:records",)
            ),
            selections=(
                MCPToolSelection(
                    remote_name="lookup", local_alias="lookup", description="Read."
                ),
                MCPToolSelection(
                    remote_name="not_admitted",
                    local_alias="other_read",
                    description="Read another tool.",
                ),
            ),
        )
    finally:
        await agent.close()

    agent = await Agent.open(
        "personal-projection-cost",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        clock=lambda: NOW,
        mcp_client_factory=factory,
        mcp_connection_provider=vault,
    )
    try:
        vault.checks.clear()
        vault.token_requests = 0
        runtime = agent._embedded._capability_runtime
        run = RunInput(
            id="projection-one",
            agent_id=agent.id,
            message="Read.",
            created_at=NOW,
        )
        expected = {tool.local_name for tool in status.binding.tools}
        first = await runtime.prepare_run(run)
        assert expected <= {entry.view.name for entry in first.entries}
        assert len(vault.checks) == 1

        vault.status = "connection_revoked"
        second = await runtime.prepare_run(run)
        assert expected.isdisjoint(entry.view.name for entry in second.entries)
        assert len(vault.checks) == 2
        assert vault.token_requests == 0
    finally:
        await agent.close()


async def test_hosted_personal_connection_requires_explicit_actor_even_for_agent_owner(
    tmp_path,
):
    identity, _ = conformance_identities()
    factory = SDKMCPClientFactory(http_transport=mock_transport(identity))
    vault = PersonalConnections(identity.endpoint)
    agent = await Agent.create(
        "owner-personal-mcp",
        root=tmp_path,
        hosted=True,
        clock=lambda: NOW,
        mcp_client_factory=factory,
        mcp_connection_provider=vault,
    )
    authentication = MCPAuthentication.personal_connection(
        "connection-alice", agent.id, identity.endpoint, ("read:records",)
    )
    try:
        for operation in (
            agent.inspect_mcp_server(
                endpoint=identity.endpoint, authentication=authentication
            ),
            agent.attach_mcp_server(
                endpoint=identity.endpoint,
                authentication=authentication,
                selections=(
                    MCPToolSelection(
                        remote_name="lookup", local_alias="lookup", description="Read."
                    ),
                ),
            ),
        ):
            with pytest.raises(ValueError, match="authenticated caller"):
                await operation
        vault.owner_principal_id = agent.id
        status = await agent.attach_mcp_server(
            endpoint=identity.endpoint,
            authentication=authentication,
            selections=(
                MCPToolSelection(
                    remote_name="lookup", local_alias="lookup", description="Read."
                ),
            ),
            caller_principal_id=agent.id,
        )
    finally:
        await agent.close()
    reopened = await Agent.open(
        "owner-personal-mcp",
        root=tmp_path,
        hosted=True,
        clock=lambda: NOW,
        mcp_client_factory=factory,
        mcp_connection_provider=PersonalConnections(identity.endpoint, agent.id),
    )
    try:
        assert await reopened.list_mcp_servers() == ()
        assert (
            len(await reopened.list_mcp_servers(caller_principal_id=reopened.id)) == 1
        )
        with pytest.raises(ValueError, match="does not exist"):
            await reopened.refresh_mcp_server(status.binding.binding_id)
        with pytest.raises(ValueError, match="does not exist"):
            await reopened.revoke_mcp_server(status.binding.binding_id)
        runtime = reopened._embedded._capability_runtime
        unauthenticated = RunInput(
            id="owner-implicit",
            agent_id=reopened.id,
            message="read",
            created_at=NOW,
            caller_principal_verified=False,
        )
        projected = await runtime.prepare_run(unauthenticated)
        assert status.binding.tools[0].local_name not in {
            item.view.name for item in projected.entries
        }
        authenticated = RunInput(
            id="owner-explicit",
            agent_id=reopened.id,
            message="read",
            created_at=NOW,
        )
        projected = await runtime.prepare_run(authenticated)
        assert status.binding.tools[0].local_name in {
            item.view.name for item in projected.entries
        }
    finally:
        await reopened.close()


@pytest.mark.parametrize("caller", [None, "alice"])
async def test_hosted_artifact_read_save_and_delete_follow_run_caller(tmp_path, caller):
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    model = ToolboxAwareMockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="create-artifact",
                        name="artifact_create_document",
                        arguments={
                            "format": "txt",
                            "filename": "alice.txt",
                            "content": "Alice owns this artifact.",
                        },
                    ),
                ),
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="done"),
        )
    )
    agent = await Agent.create(
        "hosted-artifact",
        root=tmp_path,
        hosted=True,
        model=model,
        model_profile=model.model_profile,
        downloads_directory=downloads,
    )
    try:
        result = await agent.run("Create a text artifact.", caller_principal_id=caller)
        artifact_id = result.artifacts[0].artifact_id
        assert (
            await agent.list_artifacts(caller_principal_id=caller) == result.artifacts
        )
        assert await agent.list_artifacts(caller_principal_id="bob") == ()
        assert (
            await agent.list_artifacts(caller_principal_id="bob", limit=1, offset=1)
            == ()
        )
        with pytest.raises(ValueError, match="unavailable to this caller"):
            await agent.read_artifact(artifact_id, caller_principal_id="bob")
        with pytest.raises(ValueError, match="unavailable to this caller"):
            await agent.save_artifact(artifact_id, caller_principal_id="bob")
        with pytest.raises(ValueError, match="unavailable to this caller"):
            await agent.delete_artifact(artifact_id, caller_principal_id="bob")
        assert (
            await agent.read_artifact(artifact_id, caller_principal_id=caller)
        ).content == b"Alice owns this artifact."
        assert await agent.delete_artifact(artifact_id, caller_principal_id=caller)
        assert await agent.list_artifacts(caller_principal_id=caller) == ()
        # Once physically deleted there is no owner metadata to disclose or retain.
        assert (
            await agent.delete_artifact(artifact_id, caller_principal_id="bob") is False
        )
    finally:
        await agent.close()
    reopened = await Agent.open("hosted-artifact", root=tmp_path, hosted=True)
    try:
        assert not await reopened.delete_artifact(
            artifact_id, caller_principal_id="bob"
        )
        assert not await reopened.delete_artifact(
            artifact_id, caller_principal_id=caller
        )
    finally:
        await reopened.close()


async def test_personal_token_cannot_be_sent_to_another_endpoint_origin(tmp_path):
    identity, _ = conformance_identities()
    vault = PersonalConnections(identity.endpoint)
    agent = await Agent.create(
        "personal-origin",
        root=tmp_path,
        hosted=True,
        mcp_connection_provider=vault,
    )
    try:
        authentication = MCPAuthentication.personal_connection(
            "connection-alice", "alice", identity.endpoint, ("read:records",)
        )
        with pytest.raises(ValueError, match="origins differ"):
            await agent.inspect_mcp_server(
                endpoint="https://other.example.test/mcp",
                authentication=authentication,
                caller_principal_id="alice",
            )
        assert vault.token_requests == 0
    finally:
        await agent.close()


async def test_echoed_personal_token_is_rejected_before_transcript_or_storage(
    tmp_path, caplog
):
    identity, vault, _factory, binding = await _attached(tmp_path)

    class EchoToken:
        def __init__(self) -> None:
            self.base = MCPConformanceTransport(identity)

        async def __call__(self, request: httpx.Request) -> httpx.Response:
            response = await self.base(request)
            if (
                request.method == "POST"
                and json.loads(request.content).get("method") == "tools/call"
            ):
                return httpx.Response(
                    response.status_code,
                    headers=response.headers,
                    content=response.content + TOKEN.encode("utf-8"),
                )
            return response

    factory = SDKMCPClientFactory(http_transport=httpx.MockTransport(EchoToken()))
    model = ToolboxAwareMockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="echoed-token",
                        name=binding.tools[0].local_name,
                        arguments={"query": "x"},
                    ),
                ),
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="done"),
        )
    )
    agent = await Agent.open(
        "personal-mcp",
        root=tmp_path,
        hosted=True,
        clock=lambda: NOW,
        model=model,
        model_profile=model.model_profile,
        mcp_client_factory=factory,
        mcp_connection_provider=vault,
    )
    try:
        result = await agent.run("Read.", caller_principal_id="alice")
        transcript = await agent.transcript(result.run_id, caller_principal_id="alice")
        assert _tool_error(transcript, "echoed-token") == "mcp_credential_echo_rejected"
        assert TOKEN not in repr(transcript)
        assert TOKEN.encode("utf-8") not in (agent.home / "state.db").read_bytes()
        assert TOKEN not in caplog.text
    finally:
        await agent.close()


async def test_personal_binding_isolated_across_projection_dispatch_management_and_restart(
    tmp_path,
):
    identity, vault, factory, binding = await _attached(tmp_path)
    name = binding.tools[0].local_name
    model = ToolboxAwareMockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(id="alice-call", name=name, arguments={"query": "x"}),
                ),
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="done"),
        )
    )
    agent = await Agent.open(
        "personal-mcp",
        root=tmp_path,
        hosted=True,
        clock=lambda: NOW,
        model=model,
        model_profile=model.model_profile,
        mcp_client_factory=factory,
        mcp_connection_provider=vault,
    )
    try:
        assert await agent.list_mcp_servers() == ()
        assert name not in {
            item.view.name
            for item in (
                await agent._embedded._capability_runtime.prepare_run(
                    RunInput(
                        id="run-host-default",
                        agent_id=agent.id,
                        message="read",
                        created_at=NOW,
                    )
                )
            ).entries
        }
        assert await agent.list_mcp_servers(caller_principal_id="bob") == ()
        assert len(await agent.list_mcp_servers(caller_principal_id="alice")) == 1
        for action in (
            agent.refresh_mcp_server(binding.binding_id, caller_principal_id="bob"),
            agent.revoke_mcp_server(binding.binding_id, caller_principal_id="bob"),
            agent.update_mcp_discovery(
                binding.binding_id,
                summary="x",
                when_to_use="x",
                caller_principal_id="bob",
            ),
        ):
            with pytest.raises(ValueError, match="does not exist"):
                await action

        runtime = agent._embedded._capability_runtime
        alice_run = RunInput(
            id="run-projection-alice",
            agent_id=agent.id,
            message="read",
            created_at=NOW,
            caller_principal_id="alice",
        )
        bob_run = RunInput(
            id="run-projection-bob",
            agent_id=agent.id,
            message="read",
            created_at=NOW,
            caller_principal_id="bob",
        )
        assert name in {
            item.view.name for item in (await runtime.prepare_run(alice_run)).entries
        }
        assert name not in {
            item.view.name for item in (await runtime.prepare_run(bob_run)).entries
        }
        domain = runtime._domains["mcp"]
        capability = agent._embedded._capabilities.resolve_execution(
            binding.tools[0].capability_id
        )[0]
        with pytest.raises(CapabilityInputError) as denied:
            await domain.prepare_call(
                bob_run,
                ToolCall(id="forged", name=name, arguments={"query": "x"}),
                capability,
                FrozenJsonObject.from_mapping({"query": "x"}),
                request_sensitivity=binding.tools[0].result_sensitivity,
            )
        assert denied.value.code == "needs_authorization"

        result = await agent.run("Read the fixture.", caller_principal_id="alice")
        transcript = await agent.transcript(result.run_id, caller_principal_id="alice")
        assert transcript.run.caller_principal_id == "alice"
        assert identity.calls == [("lookup", {"query": "x"})]
        assert vault.token_requests >= 4
        assert vault.checks and all(
            check == ("connection-alice", "alice", identity.endpoint, ("read:records",))
            for check in vault.checks
        )
        with pytest.raises(ValueError, match="unavailable"):
            await agent.transcript(result.run_id, caller_principal_id="bob")
        with pytest.raises(ValueError, match="unavailable"):
            await agent.conversation_runs(
                result.conversation_id, caller_principal_id="bob"
            )
        assert not await agent.conversation_exists(
            result.conversation_id, caller_principal_id="bob"
        )
        with sqlite3.connect(agent.home / "state.db") as connection:
            stored = "\n".join(
                str(value)
                for table in (
                    "runs",
                    "messages",
                    "mcp_server_bindings",
                    "effect_receipts",
                )
                for row in connection.execute(f"SELECT * FROM {table}")
                for value in row
            )
        assert TOKEN not in stored
        assert TOKEN not in repr(transcript)
        assert TOKEN not in repr(model.requests)
        model.replace_script(
            (ModelResponse(finish_reason=FinishReason.STOP, text="learned"),)
        )
        learned = await agent.learn("Learn this.", caller_principal_id="alice")
        assert (
            await agent.transcript(learned.run_id, caller_principal_id="alice")
        ).run.caller_principal_id == "alice"
    finally:
        await agent.close()


async def test_personal_revocation_between_projection_and_call_stops_dispatch(tmp_path):
    identity, vault, factory, binding = await _attached(tmp_path)
    name = binding.tools[0].local_name
    model = MCPBatchProvider(
        (ToolCall(id="revoked-call", name=name, arguments={"query": "x"}),),
        block_first_response=True,
    )
    agent = await Agent.open(
        "personal-mcp",
        root=tmp_path,
        hosted=True,
        clock=lambda: NOW,
        model=model,
        model_profile=model.model_profile,
        mcp_client_factory=factory,
        mcp_connection_provider=vault,
    )
    try:
        task = asyncio.create_task(agent.run("Read", caller_principal_id="alice"))
        await asyncio.wait_for(model.started.wait(), timeout=3)
        vault.status = "connection_revoked"
        model.release.set()
        result = await task
        transcript = await agent.transcript(result.run_id, caller_principal_id="alice")
        assert _tool_error(transcript, "revoked-call") == "connection_revoked"
        assert identity.calls == []
        projected = await agent._embedded._capability_runtime.prepare_run(
            RunInput(
                id="run-after-revocation",
                agent_id=agent.id,
                message="read",
                created_at=NOW,
                caller_principal_id="alice",
            )
        )
        assert name not in {item.view.name for item in projected.entries}
        assert TOKEN not in repr(transcript)
    finally:
        model.release.set()
        await agent.close()


def _machine_run(
    agent_id: str,
    capability_id: str,
    binding_ids: tuple[str, ...],
    origin_digest: str | None = None,
):
    instruction = "Read the admitted connector."
    scope = ExecutionScope(
        scope_id="scope-personal-test",
        revision=1,
        agent_id=agent_id,
        principal_id="alice",
        grant_id="grant-personal-test",
        job_id=None,
        job_revision=None,
        allowed_source_ids=(),
        allowed_resource_ids=(),
        allowed_capability_ids=(capability_id,),
        allowed_access_modes=frozenset({AccessMode.READ}),
        allowed_operational_effects=frozenset({OperationalEffect.NONE}),
        sensitivity_ceiling=ModelSensitivity.INTERNAL,
        eligible_model_routes=("mock:machine",),
        per_run_max_cost_usd=Decimal("0.01"),
        per_run_max_tokens=1000,
        distribution_plan_digest="sha256:" + "d" * 64,
        contract_bindings=ExecutionContractBindings(
            capability_contracts={capability_id: "sha256:" + "a" * 64},
            tool_origins=(
                {} if origin_digest is None else {capability_id: origin_digest}
            ),
            model_routes={"mock:machine": "sha256:" + "c" * 64},
        ),
        scope_kind=ExecutionScopeKind.SCHEDULED_ROUTINE,
        routine_id="routine-personal-test",
        routine_revision=1,
        occurrence_id="occurrence-personal-test",
        allowed_connector_binding_ids=binding_ids,
    )
    start = RunStartEnvelope(
        origin=RunOrigin.SCHEDULED_ROUTINE,
        instruction_authority=InstructionAuthority.FOREGROUND_AUTHORIZED,
        trusted_instruction_id="routine:personal-test:revision:1",
        trusted_instruction=instruction,
        instruction_digest="sha256:" + sha256(instruction.encode()).hexdigest(),
        untrusted_payload={},
        payload_digest="sha256:" + sha256(b"{}").hexdigest(),
        execution_scope=scope,
    )
    return RunInput(
        id="run-personal-machine",
        agent_id=agent_id,
        message=instruction,
        created_at=NOW,
        start=start,
    )


async def test_machine_personal_connection_requires_frozen_principal_and_binding(
    tmp_path,
):
    identity, vault, factory, binding = await _attached(tmp_path)
    agent = await Agent.open(
        "personal-mcp",
        root=tmp_path,
        hosted=True,
        clock=lambda: NOW,
        mcp_client_factory=factory,
        mcp_connection_provider=vault,
    )
    try:
        domain = agent._embedded._capability_runtime._domains["mcp"]
        tool = binding.tools[0]
        without_right = _machine_run(agent.id, tool.capability_id, ())
        without_origin = _machine_run(
            agent.id, tool.capability_id, (binding.binding_id,)
        )
        with_right = _machine_run(
            agent.id,
            tool.capability_id,
            (binding.binding_id,),
            mcp_execution_origin_digest(binding, tool),
        )
        assert await domain.project(without_right) == ()
        assert await domain.project(without_origin) == ()
        assert await domain.project(with_right) == (tool.local_name,)
        executor = agent._embedded._mcp_activated_bindings[binding.binding_id].executor
        request = ToolExecution(
            run_id=without_right.id,
            call_id="machine-call",
            capability_id=tool.capability_id,
            arguments={"query": "x"},
            caller_principal_id="alice",
            execution_scope=without_right.execution_scope,
        )
        with pytest.raises(Exception, match="frozen machine scope"):
            await executor.execute(request)
        assert identity.calls == []
    finally:
        await agent.close()


@pytest.mark.parametrize("lose_response", (False, True))
async def test_personal_action_receipt_never_contains_token_or_replays(
    tmp_path,
    lose_response: bool,
):
    identity, vault, factory, binding = await _attached(tmp_path, action=True)

    class ResponseLoss:
        def __init__(self) -> None:
            self.base = MCPConformanceTransport(identity)

        async def __call__(self, request: httpx.Request) -> httpx.Response:
            result = await self.base(request)
            if (
                lose_response
                and json.loads(request.content).get("method") == "tools/call"
            ):
                raise httpx.ReadError("response lost", request=request)
            return result

    if lose_response:
        factory = SDKMCPClientFactory(
            http_transport=httpx.MockTransport(ResponseLoss())
        )

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.APPROVE

    model = ToolboxAwareMockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="personal-action",
                        name=binding.tools[0].local_name,
                        arguments={"query": "x"},
                    ),
                ),
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="done"),
        )
    )
    agent = await Agent.open(
        "personal-mcp",
        root=tmp_path,
        hosted=True,
        clock=lambda: NOW,
        model=model,
        model_profile=model.model_profile,
        approval_handler=approve,
        mcp_client_factory=factory,
        mcp_connection_provider=vault,
    )
    try:
        result = await agent.run(
            "Call the reviewed action.", caller_principal_id="alice"
        )
        transcript = await agent.transcript(result.run_id, caller_principal_id="alice")
        assert identity.calls == [("lookup", {"query": "x"})]
        receipts = await agent.list_effects(caller_principal_id="alice")
        assert len(receipts) == 1
        assert await agent.list_effects(caller_principal_id="bob") == ()
        assert (
            await agent.inspect_effect(
                receipts[0].receipt_id, caller_principal_id="bob"
            )
            is None
        )
        assert receipts[0].outcome.value == (
            "uncertain" if lose_response else "succeeded"
        )
        if lose_response:
            with pytest.raises(ValueError, match="owned receipt"):
                await agent.resolve_effect(
                    receipts[0].receipt_id,
                    expected_digest=receipts[0].receipt_digest,
                    decision=EffectResolutionDecision.CLOSE_WITHOUT_RETRY,
                    note="Reviewed response loss.",
                    caller_principal_id="bob",
                )
            resolved = await agent.resolve_effect(
                receipts[0].receipt_id,
                expected_digest=receipts[0].receipt_digest,
                decision=EffectResolutionDecision.CLOSE_WITHOUT_RETRY,
                note="Reviewed response loss.",
                caller_principal_id="alice",
            )
            assert resolved.resolution is not None
            assert identity.calls == [("lookup", {"query": "x"})]
        assert TOKEN not in repr(transcript)
        assert TOKEN not in repr(receipts)
        assert TOKEN not in (agent.home / "state.db").read_bytes().decode(
            "utf-8", errors="ignore"
        )
    finally:
        await agent.close()
