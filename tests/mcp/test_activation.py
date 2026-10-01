"""SDK admission publishes between runs and preserves exact frozen authority."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import httpx2 as httpx
import pytest

from daita import Agent, ApprovalDecision, MCPAuthentication, MCPToolSelection
from daita._json import FrozenJsonObject
from daita.adapters.mcp import SDKMCPClientFactory
from daita.adapters.mcp_sdk import SDKMCPClient
from daita.capabilities import (
    MAX_TOOL_PRESENTATION_GUIDANCE_CHARACTERS,
    MAX_TOOL_PRESENTATION_SUMMARY_CHARACTERS,
    ApprovalRequest,
)
from daita.llm.models import (
    CostEstimate,
    FinishReason,
    ModelResponse,
    ModelUsage,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.provider_definitions import tool_schema_incompatibility
from daita.loop.models import LoopExitKind, LoopLimits
from daita.security import SecretReference
from tests.support.mcp import (
    MCPConformanceTransport,
    MemoryKeychain,
    conformance_identities,
    mock_transport,
)
from tests.support.toolbox_model import ToolboxAwareMockModelProvider
from tests.support.workspace import workspace_for

NOW = datetime(2026, 9, 30, tzinfo=UTC)


@pytest.mark.parametrize("protocol", ["2026-07-28", "2025-11-25", "2025-06-18"])
@pytest.mark.parametrize("description", ["short", "multiline_long", "blank"])
async def test_sdk_rich_contract_activates_calls_and_rechecks_in_same_agent(
    tmp_path, protocol, description
):
    identity, _ = conformance_identities()
    identity.protocol_version = protocol
    if description == "multiline_long":
        identity.tool("lookup")["description"] = (
            "\n " + "Read current documentation. " * 64 + "\n "
        )
    elif description == "blank":
        identity.tool("lookup")["description"] = " \n\t "
    identity.tool("lookup")["inputSchema"] = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "definitions": {"query": {"type": ["string", "null"]}},
        "properties": {"query": {"$ref": "#/definitions/query"}},
        "required": ["query"],
        "additionalProperties": False,
    }
    model = ToolboxAwareMockModelProvider((), provider_id="openai:fixture")
    agent = await Agent.create(
        "same-agent",
        root=tmp_path,
        hosted=True,
        model=model,
        model_profile=model.model_profile,
        limits=LoopLimits(),
        clock=lambda: NOW,
        mcp_client_factory=SDKMCPClientFactory(http_transport=mock_transport(identity)),
    )
    try:
        status = await agent.attach_mcp_server(
            endpoint=identity.endpoint, selections=(MCPToolSelection("lookup"),)
        )
        assert status.active_in_runtime
        tool = status.binding.tools[0]
        assert (
            tool.raw_input_schema is not None and "definitions" in tool.raw_input_schema
        )
        assert "definitions" not in tool.input_schema
        expected_description = (
            str(identity.tool("lookup")["description"]).strip()[:1_024].rstrip()
            or "Use the admitted MCP tool lookup."
        )
        assert tool.description == expected_description
        assert (
            tool.presentation.summary
            == expected_description[:MAX_TOOL_PRESENTATION_SUMMARY_CHARACTERS].rstrip()
        )
        assert (
            tool.presentation.when_to_use
            == expected_description[:MAX_TOOL_PRESENTATION_GUIDANCE_CHARACTERS].rstrip()
        )
        model.replace_script(
            (
                ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(ToolCall("read-1", tool.local_name, {"query": None}),),
                ),
                ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(
                        ToolCall("read-2", tool.local_name, {"query": "next"}),
                    ),
                ),
                ModelResponse(finish_reason=FinishReason.STOP, text="done"),
            )
        )
        runtime = agent._embedded._capability_runtime
        loop = agent._embedded._loop
        result = await agent.run("Read the admitted fixture twice.")
        assert result.kind is LoopExitKind.COMPLETED
        transcript = await agent.transcript(result.run_id)
        blocks = [
            block
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
            and block.call_id in {"read-1", "read-2"}
        ]
        assert len(blocks) == 2 and all(not block.is_error for block in blocks)
        assert identity.calls == [
            ("lookup", {"query": None}),
            ("lookup", {"query": "next"}),
        ]
        assert (
            identity.request_methods.count("tools/list") == 3
        )  # attach + each exact call
        client = agent._embedded._mcp_activated_bindings[
            status.binding.binding_id
        ].client
        refreshed = await agent.refresh_mcp_server(status.binding.binding_id)
        assert refreshed.active_in_runtime
        assert (
            agent._embedded._capability_runtime is runtime
            and agent._embedded._loop is loop
        )
        assert isinstance(client, SDKMCPClient)
        assert client._owner is not None and client._owner.done()
        contracts = await agent._embedded._execution_contract_reader(
            agent_id=agent.id,
            source_ids=(),
            resource_ids=(),
            capability_ids=(tool.capability_id,),
            connector_binding_ids=(status.binding.binding_id,),
            model_route_ids=(),
        )
        assert contracts.capability_contracts
        assert await agent.list_effects() == ()
    finally:
        await agent.close()
        await model.close()


async def test_attachment_waits_for_frozen_run_and_reuses_sibling_sdk_owner(tmp_path):
    alpha, beta = conformance_identities()
    beta.bearer_token = None
    entered, release = asyncio.Event(), asyncio.Event()

    blocking = False

    class GatedProvider(ToolboxAwareMockModelProvider):
        async def generate(self, request):
            if blocking:
                entered.set()
                await release.wait()
            return await super().generate(request)

    model = GatedProvider(())
    agent = await Agent.create(
        "activation-race",
        root=tmp_path,
        model=model,
        model_profile=model.model_profile,
        limits=LoopLimits(),
        workspace=workspace_for(tmp_path),
        mcp_client_factory=SDKMCPClientFactory(
            http_transport=mock_transport(alpha, beta)
        ),
    )
    try:
        first = await agent.attach_mcp_server(
            endpoint=alpha.endpoint, selections=(MCPToolSelection("lookup"),)
        )
        name = first.binding.tools[0].local_name
        model.replace_script(
            (
                ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(ToolCall("first", name, {"query": "x"}),),
                ),
                ModelResponse(finish_reason=FinishReason.STOP, text="done"),
            )
        )
        await agent.run("Read the first fixture.")
        sibling = agent._embedded._mcp_activated_bindings[first.binding.binding_id]
        client = sibling.client
        blocking = True
        model.replace_script(
            (ModelResponse(finish_reason=FinishReason.STOP, text="finished"),)
        )
        run = asyncio.create_task(agent.run("Finish this frozen run."))
        await asyncio.wait_for(entered.wait(), 5)
        attaching = asyncio.create_task(
            agent.attach_mcp_server(
                endpoint=beta.endpoint, selections=(MCPToolSelection("lookup"),)
            )
        )
        await asyncio.sleep(0)
        assert not attaching.done() and beta.request_methods == []
        release.set()
        await asyncio.wait_for(run, 5)
        second = await asyncio.wait_for(attaching, 5)
        blocking = False
        assert second.active_in_runtime
        assert (
            agent._embedded._mcp_activated_bindings[first.binding.binding_id] is sibling
        )
        assert sibling.client is client
        model.replace_script(
            (
                ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(
                        ToolCall(
                            "second", second.binding.tools[0].local_name, {"id": 3}
                        ),
                    ),
                ),
                ModelResponse(finish_reason=FinishReason.STOP, text="done"),
            )
        )
        await agent.run("Read the newly attached fixture.")
        assert beta.calls == [("lookup", {"id": 3})]
    finally:
        release.set()
        await agent.close()
        await model.close()


async def test_revocation_during_refresh_inspection_wins_cas(tmp_path):
    alpha, _ = conformance_identities()
    transport = MCPConformanceTransport(alpha)
    entered, release = asyncio.Event(), asyncio.Event()
    inspecting = False

    async def boundary(request: httpx.Request):
        if (
            inspecting
            and request.method == "POST"
            and json.loads(request.content).get("method") == "tools/list"
        ):
            entered.set()
            await release.wait()
        return await transport(request)

    agent = await Agent.create(
        "refresh-cas",
        root=tmp_path,
        hosted=True,
        mcp_client_factory=SDKMCPClientFactory(
            http_transport=httpx.MockTransport(boundary)
        ),
    )
    try:
        status = await agent.attach_mcp_server(
            endpoint=alpha.endpoint, selections=(MCPToolSelection("lookup"),)
        )
        inspecting = True
        refresh = asyncio.create_task(
            agent.refresh_mcp_server(status.binding.binding_id)
        )
        await asyncio.wait_for(entered.wait(), 5)
        revoked = await asyncio.wait_for(
            agent.revoke_mcp_server(status.binding.binding_id), 5
        )
        assert not revoked.active_in_runtime
        release.set()
        with pytest.raises(ValueError, match="revision precondition"):
            await refresh
        assert not (await agent.list_mcp_servers())[0].active_in_runtime
        assert alpha.calls == []
    finally:
        release.set()
        await agent.close()


def test_every_model_route_candidate_must_accept_projection_without_weakening_it():
    projected = FrozenJsonObject.from_mapping(
        {
            "type": "object",
            "properties": {"value": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
        }
    )
    assert (
        tool_schema_incompatibility(projected, ("openai:fixture", "anthropic:fixture"))
        is None
    )
    assert (
        tool_schema_incompatibility(projected, ("openai:fixture", "gemini:fixture"))
        == "model_schema_unsupported:gemini:fixture:anyOf"
    )
    assert tool_schema_incompatibility(projected, ("custom:fixture",)) is not None


async def test_failed_catalog_staging_preserves_previous_admission(
    tmp_path, monkeypatch
):
    alpha, beta = conformance_identities()
    beta.bearer_token = None
    agent = await Agent.create(
        "stage-failure",
        root=tmp_path,
        hosted=True,
        mcp_client_factory=SDKMCPClientFactory(
            http_transport=mock_transport(alpha, beta)
        ),
    )
    try:
        first = await agent.attach_mcp_server(
            endpoint=alpha.endpoint, selections=(MCPToolSelection("lookup"),)
        )
        registry = agent._embedded._capabilities

        async def fail_staging(_bindings, _existing):
            raise ValueError("invalid staged catalog")

        with monkeypatch.context() as patch:
            patch.setattr(agent._embedded, "_stage_mcp_catalog", fail_staging)
            with pytest.raises(ValueError, match="invalid staged catalog"):
                await agent.attach_mcp_server(
                    endpoint=beta.endpoint, selections=(MCPToolSelection("lookup"),)
                )
        statuses = await agent.list_mcp_servers()
        assert len(statuses) == 1 and statuses[0].binding == first.binding
        assert statuses[0].active_in_runtime
        assert agent._embedded._capabilities is registry
        assert beta.calls == []
    finally:
        await agent.close()


async def test_owned_mcp_credentials_cleanup_preserves_shared_secrets(tmp_path):
    alpha, beta = conformance_identities()
    token = "private-mcp-fixture-token"
    alpha.bearer_token = beta.bearer_token = token
    keychain = MemoryKeychain()
    shared = SecretReference.keychain("externally-owned-mcp-secret")
    keychain.values[shared.to_uri()] = token
    agent = await Agent.create(
        "mcp-credentials",
        root=tmp_path,
        keychain=keychain,
        workspace=workspace_for(tmp_path),
        mcp_client_factory=SDKMCPClientFactory(
            http_transport=mock_transport(alpha, beta)
        ),
    )
    try:
        unused = await agent.store_mcp_bearer(token)
        await agent.delete_mcp_bearer(unused)
        assert unused.to_uri() not in keychain.values
        owned = await agent.store_mcp_bearer(token)
        status = await agent.attach_mcp_server(
            endpoint=alpha.endpoint,
            authentication=MCPAuthentication.bearer(owned),
            selections=(MCPToolSelection("lookup"),),
        )
        await agent.attach_mcp_server(
            endpoint=beta.endpoint,
            authentication=MCPAuthentication.bearer(shared),
            selections=(MCPToolSelection("lookup"),),
        )
        with pytest.raises(ValueError, match="retained by an MCP binding"):
            await agent.delete_mcp_bearer(owned)
        await agent.revoke_mcp_server(status.binding.binding_id)
        assert (
            owned.to_uri() in keychain.values
        )  # revocation does not delete a shared credential
        assert all(
            token.encode() not in path.read_bytes()
            for path in agent.home.rglob("*")
            if path.is_file()
        )
    finally:
        await agent.close()
    await Agent.delete("mcp-credentials", root=tmp_path, keychain=keychain)
    assert owned.to_uri() not in keychain.values
    assert keychain.values == {shared.to_uri(): token}
    assert keychain.deleted == [unused.to_uri(), owned.to_uri()]


async def test_graph_admission_sees_new_binding_and_retains_frozen_origin(tmp_path):
    from decimal import Decimal

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.APPROVE

    alpha, _ = conformance_identities()
    model = ToolboxAwareMockModelProvider((), complete_pricing=True)
    agent = await Agent.create(
        "mcp-graph-activation",
        root=tmp_path,
        hosted=True,
        approval_handler=approve,
        model=model,
        model_profile=model.model_profile,
        limits=LoopLimits(max_estimated_cost_usd=Decimal("1")),
        mcp_client_factory=SDKMCPClientFactory(http_transport=mock_transport(alpha)),
    )
    try:
        # Admission is exercised separately from machine execution, which has
        # its own runtime integration suite. Keep the new graph queued here.
        await agent._embedded._job_supervisor.close()
        status = await agent.attach_mcp_server(
            endpoint=alpha.endpoint, selections=(MCPToolSelection("lookup"),)
        )
        tool = status.binding.tools[0]
        model.replace_script(
            (
                ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(
                        ToolCall(
                            "admit-graph",
                            "start_graph_job",
                            {
                                "objective": "Read the admitted fixture.",
                                "outcome_contract": {
                                    "required_result_kind": "mcp.tool.result"
                                },
                                "deadline_seconds": 300,
                                "initial_task": {
                                    "capability_id": tool.capability_id,
                                    "arguments": {"query": "x"},
                                    "expected_result_contract": {
                                        "result_kind": "mcp.tool.result"
                                    },
                                    "retained_references": {
                                        "source_ids": (),
                                        "resource_ids": (),
                                        "connector_binding_ids": (
                                            status.binding.binding_id,
                                        ),
                                    },
                                },
                            },
                        ),
                    ),
                    usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0"))),
                ),
                ModelResponse(
                    finish_reason=FinishReason.STOP,
                    text="Graph admitted.",
                    usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0"))),
                ),
            )
        )
        result = await agent.run("Admit this exact graph.")
        transcript = await agent.transcript(result.run_id)
        receipts = [
            block
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock) and block.call_id == "admit-graph"
        ]
        assert len(receipts) == 1 and not receipts[0].is_error
        data = receipts[0].output["data"]
        assert isinstance(data, FrozenJsonObject)
        job_id = data["job_id"]
        assert isinstance(job_id, str)
        graph = await agent.inspect_job(job_id)
        assert graph is not None
        retained = graph.job.specification.authority.digest
        await agent.refresh_mcp_server(status.binding.binding_id)
        refreshed_graph = await agent.inspect_job(job_id)
        assert refreshed_graph is not None
        assert refreshed_graph.job.specification.authority.digest == retained
        assert alpha.calls == []
    finally:
        await agent.close()
        await model.close()
