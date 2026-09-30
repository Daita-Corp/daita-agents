"""Opt-in SDK and real-model reads against an operator-selected MCP server."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from daita import Agent, LoopLimits, create_llm_provider
from daita.adapters.mcp import (
    MCPAuthentication,
    MCPToolSelection,
    SDKMCPClientFactory,
)
from daita.llm.profiles import reviewed_model_profile
from daita.loop.models import LoopExitKind, validate_completed_transcript
from daita.security import (
    EmptySecretProvider,
    EnvironmentSecretProvider,
    SecretProvider,
    SecretReference,
)
from tests.support.job_benchmarks import (
    RecordingProvider,
    RunCapture,
    assert_on_demand_invocation,
    results_for,
)


@pytest.mark.integration
@pytest.mark.requires_network
async def test_remote_streamable_http_interoperability_smoke(
    record_property: Callable[[str, object], None],
) -> None:
    """Exercise production inspect/call translation only when explicitly enabled."""

    if os.environ.get("DAITA_RUN_LIVE_MCP") != "1":
        pytest.skip("set DAITA_RUN_LIVE_MCP=1 to authorize live MCP network I/O")

    endpoint = os.environ["MCP_HOST"]
    remote_tool = os.environ["MCP_TOOL"]
    arguments = json.loads(os.environ["MCP_ARGUMENTS"])
    expected_text = os.environ["MCP_EXPECT_TEXT"]
    if not isinstance(arguments, dict):
        raise ValueError("MCP_ARGUMENTS must be a JSON object")
    if not expected_text.strip():
        raise ValueError("MCP_EXPECT_TEXT must be a nonempty independent result marker")

    if os.environ.get("MCP_TOKEN"):
        authentication = MCPAuthentication.bearer(
            SecretReference.environment("MCP_TOKEN")
        )
        secrets: SecretProvider = EnvironmentSecretProvider()
    else:
        authentication = MCPAuthentication.no_auth()
        secrets = EmptySecretProvider()

    client = SDKMCPClientFactory().create(
        endpoint=endpoint,
        authentication=authentication,
        secrets=secrets,
    )
    try:
        inspection = await client.inspect(observed_at=datetime.now(UTC))
        record_property("mcp_protocol", inspection.protocol_version)
        selected = next(
            (tool for tool in inspection.tools if tool.remote_name == remote_tool),
            None,
        )
        assert selected is not None
        assert selected.supported, selected.unsupported_reason
        result = await client.call_tool(remote_tool, arguments)
        assert not result.is_error
        assert result.text or result.structured is not None
        assert (
            expected_text.casefold() in str((result.text, result.structured)).casefold()
        )
    finally:
        await client.close()


@pytest.mark.acceptance
@pytest.mark.integration
@pytest.mark.requires_network
@pytest.mark.requires_llm
async def test_real_model_reads_remote_mcp_after_agent_restart(
    tmp_path: Path,
    record_property: Callable[[str, object], None],
) -> None:
    """Real admission, restart, discovery, dispatch and grounded model completion."""

    if (
        os.environ.get("DAITA_RUN_LIVE_MCP") != "1"
        or os.environ.get("DAITA_RUN_LIVE_MCP_LLM") != "1"
    ):
        pytest.skip(
            "set DAITA_RUN_LIVE_MCP=1 and DAITA_RUN_LIVE_MCP_LLM=1 to authorize "
            "one live model run and a remote read; Firecrawl may charge credits"
        )

    endpoint = os.environ["MCP_HOST"]
    remote_tool = os.environ["MCP_TOOL"]
    arguments = json.loads(os.environ["MCP_ARGUMENTS"])
    expected_text = os.environ["MCP_EXPECT_TEXT"]
    if not isinstance(arguments, dict):
        raise ValueError("MCP_ARGUMENTS must be a JSON object")
    if not expected_text.strip():
        raise ValueError("MCP_EXPECT_TEXT must be a nonempty independent result marker")
    if os.environ.get("MCP_TOKEN"):
        authentication = MCPAuthentication.bearer(
            SecretReference.environment("MCP_TOKEN")
        )
        secrets: SecretProvider = EnvironmentSecretProvider()
    else:
        authentication = MCPAuthentication.no_auth()
        secrets = EmptySecretProvider()

    model_id = os.environ.get("DAITA_LIVE_MCP_MODEL_ID", "openai:gpt-5.6-terra")
    profile = reviewed_model_profile(model_id)
    if (
        not model_id.startswith("openai:")
        or profile is None
        or not profile.supports_tools
    ):
        raise ValueError("DAITA_LIVE_MCP_MODEL_ID must be a reviewed OpenAI tool model")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for the real-model MCP test")
    amount = Decimal(os.environ.get("DAITA_LIVE_MCP_MAX_COST_USD", "0.50"))
    if not amount.is_finite() or amount <= 0:
        raise ValueError("DAITA_LIVE_MCP_MAX_COST_USD must be finite and positive")
    limits = LoopLimits(max_estimated_cost_usd=amount)

    agent = await Agent.create(
        "remote-mcp-read",
        root=tmp_path,
        hosted=True,
        secret_provider=secrets,
    )
    try:
        inspection = await agent.inspect_mcp_server(
            endpoint=endpoint, authentication=authentication
        )
        record_property("mcp_protocol", inspection.protocol_version)
        selected = next(
            (tool for tool in inspection.tools if tool.remote_name == remote_tool),
            None,
        )
        assert selected is not None, "MCP_TOOL was not advertised by this endpoint"
        assert selected.supported, selected.unsupported_reason
        status = await agent.attach_mcp_server(
            endpoint=endpoint,
            authentication=authentication,
            selections=(
                MCPToolSelection(
                    remote_name=remote_tool,
                    local_alias="remote_read",
                    description="Read the requested public web content from the admitted remote server.",
                ),
            ),
        )
        binding = status.binding
        local_name = binding.tools[0].local_name
        record_property("mcp_binding_id", binding.binding_id)
        record_property("mcp_input_schema_digest", selected.input_schema_digest)
    finally:
        await agent.close()

    # Construct the paid provider only after remote inspection/admission succeeds.
    provider = RecordingProvider(
        create_llm_provider(
            model_id,
            api_key=api_key,
            max_output_tokens=min(profile.max_output_tokens, 2_048),
        )
    )
    try:
        agent = await Agent.open(
            "remote-mcp-read",
            root=tmp_path,
            hosted=True,
            model=provider,
            model_profile=profile,
            limits=limits,
            secret_provider=secrets,
        )
        try:
            statuses = await agent.list_mcp_servers()
            assert len(statuses) == 1 and statuses[0].active_in_runtime
            result = await agent.run(
                f"Use {local_name} with exactly these arguments: "
                f"{json.dumps(arguments, sort_keys=True)}. Read it once and briefly "
                "summarize the returned content, including its identifying name or title. "
                "Ground your answer in the successful tool result."
            )
            transcript = await agent.transcript(result.run_id)
            record_property("mcp_run_id", result.run_id)
            record_property("mcp_model_requests", len(provider.requests))
            record_property("mcp_total_tokens", result.usage.total_tokens)
            record_property(
                "mcp_estimated_cost_usd", result.usage.cost_estimate.amount_usd
            )
            assert result.kind is LoopExitKind.COMPLETED, result.reason
            assert result.final_text and result.usage.total_tokens > 0
            validate_completed_transcript(transcript, result)
            capture = RunCapture(result, transcript, tuple(provider.requests))
            assert_on_demand_invocation(capture, local_name)
            calls = [
                call
                for message in transcript.messages
                for call in message.tool_calls
                if call.name == local_name
            ]
            assert len(calls) == 1 and calls[0].arguments == arguments
            outputs = results_for(transcript, local_name)
            assert len(outputs) == 1
            data = outputs[0].output["data"]
            assert isinstance(data, Mapping)
            assert expected_text.casefold() in str(data).casefold()
            provenance = data["provenance"]
            assert isinstance(provenance, Mapping)
            assert provenance["binding_id"] == binding.binding_id
            assert provenance["binding_revision"] == binding.revision
            assert provenance["remote_tool_name"] == remote_tool
            assert provenance["input_schema_digest"] == selected.input_schema_digest
            assert expected_text.casefold() in result.final_text.casefold()
            assert await agent.list_effects() == ()
        finally:
            await agent.close()
    finally:
        await provider.close()
