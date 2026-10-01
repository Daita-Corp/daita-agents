"""Natural requests and session reuse against an authorized real MCP service."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

import httpx2 as httpx
import pytest

from daita import Agent, LoopLimits, MCPAuthentication, MCPToolSelection
from daita.adapters.mcp import SDKMCPClientFactory
from daita.llm.models import ToolResultBlock
from daita.llm.profiles import reviewed_model_profile
from daita.llm.providers.openai.messages import _response_input
from daita.loop.models import LoopExitKind, validate_completed_transcript
from daita.security import (
    EmptySecretProvider,
    EnvironmentSecretProvider,
    SecretProvider,
    SecretReference,
)
from tests.support.job_benchmarks import assert_on_demand_invocation, results_for
from tests.support.mcp_read_harness import (
    AUTHORIZATION,
    MeasuredMCPTransport,
    ReadBudget,
    evaluate_reads,
    repetitions,
)

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.requires_network,
    pytest.mark.skipif(
        os.environ.get(AUTHORIZATION) != "1"
        or os.environ.get("DAITA_RUN_LIVE_MCP") != "1",
        reason=f"set {AUTHORIZATION}=1 and DAITA_RUN_LIVE_MCP=1 to authorize four bounded model runs and three remote reads per repetition",
    ),
]


@pytest.mark.parametrize("repetition", repetitions())
async def test_real_remote_natural_reads_reuse_session_restart_and_answer_follow_up(
    tmp_path: Path,
    read_budget: ReadBudget,
    read_report_path: Path,
    request: pytest.FixtureRequest,
    repetition: int,
) -> None:
    endpoint = os.environ["MCP_HOST"]
    remote_name = os.environ["MCP_TOOL"]
    prompt = os.environ["MCP_READ_PROMPT"]
    expected_text = os.environ["MCP_EXPECT_TEXT"]
    expected_arguments = json.loads(os.environ["MCP_READ_ARGUMENT_MATCH"])
    if (
        not prompt.strip()
        or not expected_text.strip()
        or not isinstance(expected_arguments, dict)
        or not expected_arguments
    ):
        raise ValueError(
            "provide a natural read prompt, independent result marker, and nonempty JSON argument subset"
        )
    if remote_name in prompt or "remote_read" in prompt:
        raise ValueError("MCP_READ_PROMPT must not specify the tool name")
    authentication = MCPAuthentication.no_auth()
    secrets: SecretProvider = EmptySecretProvider()
    if os.environ.get("MCP_TOKEN"):
        authentication = MCPAuthentication.bearer(
            SecretReference.environment("MCP_TOKEN")
        )
        secrets = EnvironmentSecretProvider()
    transport = MeasuredMCPTransport(
        httpx.AsyncHTTPTransport(trust_env=False, retries=0)
    )
    async with evaluate_reads(
        tmp_path / "home",
        transport=transport,
        budget=read_budget,
        report=read_report_path,
        case_id=request.node.nodeid,
        secrets=secrets,
    ) as scenario:
        status = await scenario.agent.attach_mcp_server(
            endpoint=endpoint,
            authentication=authentication,
            selections=(MCPToolSelection(remote_name),),
        )
        assert status.active_in_runtime
        binding = status.binding
        tool = binding.tools[0]
        scenario.setup.update(
            protocol=binding.protocol_version,
            binding_id=binding.binding_id,
            binding_revision=binding.revision,
            input_schema_digest=tool.input_schema_digest,
            service_kind="real_remote",
            workload="public_web_read",
            repetition=repetition,
        )
        model_id = os.environ.get("DAITA_LIVE_MCP_MODEL_ID", "openai:gpt-5.6-terra")
        for phase in ("cold", "warm", "restarted"):
            if phase == "restarted":
                await scenario.agent.close()
                # Opening the home reconstructs contracts without remote I/O.
                scenario.agent = await Agent.open(
                    "mcp-read-evaluation",
                    root=tmp_path / "home",
                    hosted=True,
                    model=scenario.provider,
                    model_profile=reviewed_model_profile(model_id),
                    limits=LoopLimits(max_estimated_cost_usd=read_budget.per_run),
                    secret_provider=secrets,
                    mcp_client_factory=SDKMCPClientFactory(http_transport=transport),
                )
            capture = await scenario.run(prompt, phase=phase)
            assert capture.result.kind is LoopExitKind.COMPLETED, capture.result.reason
            validate_completed_transcript(capture.transcript, capture.result)
            assert_on_demand_invocation(capture, tool.local_name)
            calls = [
                call
                for message in capture.transcript.messages
                for call in message.tool_calls
                if call.name == tool.local_name
            ]
            assert len(calls) == 1
            assert all(
                calls[0].arguments.get(key) == value
                for key, value in expected_arguments.items()
            )
            outputs = results_for(capture.transcript, tool.local_name)
            assert len(outputs) == 1
            data = outputs[0].output["data"]
            assert isinstance(data, Mapping)
            assert expected_text.casefold() in str(data).casefold()
            provenance = data["provenance"]
            assert isinstance(provenance, Mapping)
            assert provenance["binding_id"] == binding.binding_id
            assert provenance["binding_revision"] == binding.revision
            assert provenance["input_schema_digest"] == tool.input_schema_digest
            assert (
                expected_text.casefold() in (capture.result.final_text or "").casefold()
            )
            metrics = scenario.runs[-1]
            methods = metrics["mcp_methods"]
            assert isinstance(methods, Mapping)
            assert methods.get("tools/call") == 1
            # Each call must inspect current contracts, including warm owners.
            assert methods.get("tools/list", 0) >= 1
            if phase == "warm":
                assert methods.get("initialize", 0) == 0
                assert methods.get("server/discover", 0) == 0
        follow_up = await scenario.run(
            os.environ.get(
                "MCP_FOLLOWUP_PROMPT",
                "What is the identifying title of that page? Base your answer on what you just read; no new lookup is needed.",
            ),
            phase="follow_up",
            conversation_id=capture.result.conversation_id,
        )
        assert follow_up.result.kind is LoopExitKind.COMPLETED, follow_up.result.reason
        validate_completed_transcript(follow_up.transcript, follow_up.result)
        earlier_results = [
            block
            for message in follow_up.requests[0].messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
            and block.output.get("kind") == "mcp.tool.result"
        ]
        assert len(earlier_results) == 1
        assert earlier_results[0].call_id.startswith("hist_")
        assert expected_text.casefold() in str(earlier_results[0].output).casefold()
        native_input = _response_input(
            follow_up.requests[0].messages, scenario.provider.provider_id
        )
        assert any(
            item.get("type") == "function_call_output"
            and expected_text.casefold() in str(item.get("output")).casefold()
            for item in native_input
        )
        assert (
            expected_text.casefold() in (follow_up.result.final_text or "").casefold()
        )
        assert scenario.runs[-1]["mcp_methods"] == {}
        assert await scenario.agent.list_effects() == ()
