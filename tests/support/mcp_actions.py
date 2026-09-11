"""Shared helpers extracted from ``test_actions.py``."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import httpx
import pytest

from daita import (
    Agent,
    CalendarDaySelector,
    CalendarSchedule,
    EffectRequirement,
    MCPCompletionSemantics,
    MCPToolSelection,
    MisfirePolicy,
    ReportingMode,
    RequestedCapabilityGrant,
    ScheduledRoutineDraft,
)
from daita._json import FrozenJsonObject, canonical_json
from daita.adapters.mcp import StreamableHTTPMCPClientFactory
from daita.capabilities import (
    AccessMode,
    ApprovalDecision,
    AutomationEligibility,
    CapabilityInputError,
    EffectEvidenceBasis,
    EffectOutcome,
    OperationalEffect,
)
from daita.distribution.models import OutcomeState
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    ModelSensitivity,
    ModelUsage,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.pricing import CostEstimate
from daita.routines.models import RoutineState
from daita.routines.owner import RoutineError
from daita.storage.sqlite_records import EffectResolutionDecision
from tests.support.distribution import no_artifact_outcome_contract
from tests.support.mcp import MCPConformanceTransport, MCPFixtureIdentity
from tests.support.workspace import workspace_for

NOW = datetime(2026, 9, 6, 12, tzinfo=UTC)


class ActionTransport(MCPConformanceTransport):
    """Faults happen after the fake service records application, not before send."""

    def __init__(self, identity):
        super().__init__(identity)
        self.mode = "success"
        self.dispatched = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, request):
        payload = json.loads(request.content)
        result = await super().__call__(request)
        if (
            payload.get("method") != "tools/call"
            or payload["params"]["name"] != "notify"
        ):
            return result
        self.dispatched.set()
        if self.mode == "disconnect":
            raise httpx.ReadError("response lost after applying", request=request)
        if self.mode in {"timeout", "cancel"}:
            await self.release.wait()
        if self.mode == "malformed":
            return httpx.Response(
                200, content=b"{invalid", headers={"content-type": "application/json"}
            )
        if self.mode == "oversized":
            return httpx.Response(
                200,
                content=b"x" * (513 * 1024),
                headers={"content-type": "application/json"},
            )
        return result


def response(*calls, text=None):
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS if calls else FinishReason.STOP,
        tool_calls=calls,
        text=text,
        usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0"))),
    )


class ActionModel:
    provider_id = "mock:mcp-actions"
    model_profile = ModelProfile(
        id=provider_id,
        context_window_tokens=64000,
        max_output_tokens=2000,
        supports_tools=True,
        supports_parallel_tools=True,
    )

    def __init__(self):
        self.steps = []
        self.requests = []

    def supports_request_policy(self, request):
        return True

    def has_complete_pricing(self, request):
        return True

    async def generate(self, request):
        self.requests.append(request)
        assert self.steps, "script exhausted"
        return self.steps.pop(0)


class ActionFixture:
    def __init__(self, tmp_path):
        self.root = tmp_path
        self.clock = NOW
        self.model = ActionModel()
        self.approvals = []
        self.decision = ApprovalDecision.APPROVE
        self.server = MCPFixtureIdentity(
            host="actions.fixture.test",
            server_name="ordinary-service",
            server_version="1",
            protocol_version="2025-11-25",
            tools=[
                {
                    "name": "notify",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "destination": {"type": "string"},
                            "content": {"type": "string"},
                            "options": {
                                "type": "object",
                                "properties": {"recipient": {"type": "string"}},
                            },
                        },
                        "required": ["destination", "content"],
                        "additionalProperties": False,
                    },
                    "annotations": {"readOnlyHint": True, "idempotentHint": True},
                },
                {
                    "name": "research",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            ],
            results={
                "notify": {"content": [{"type": "text", "text": "success"}]},
                "research": {
                    "content": [
                        {
                            "type": "text",
                            "text": "Source https://research.test/report reports improved status; coverage is limited.",
                        }
                    ]
                },
            },
        )
        self.transport = ActionTransport(self.server)
        self.factory = StreamableHTTPMCPClientFactory(
            http_transport=httpx.MockTransport(self.transport), timeout_seconds=0.2
        )

    async def approve(self, request):
        self.approvals.append(request)
        return self.decision

    def kwargs(self) -> dict[str, Any]:
        return dict(
            root=self.root,
            workspace=workspace_for(self.root),
            model=self.model,
            model_profile=self.model.model_profile,
            clock=lambda: self.clock,
            mcp_client_factory=self.factory,
            approval_handler=self.approve,
        )

    async def start(self, *, selection=None, outbound=ModelSensitivity.INTERNAL):
        self.agent = await Agent.create("mcp-actions", **self.kwargs())
        selection = selection or MCPToolSelection(
            "notify",
            "notify",
            "Send the reviewed notification.",
            access_mode=AccessMode.NONE,
            operational_effect=OperationalEffect.EXTERNAL_ACTION,
            automation_eligibility=AutomationEligibility.AUTOMATION_DIRECT,
        )
        status = await self.agent.attach_mcp_server(
            endpoint=self.server.endpoint,
            selections=(
                selection,
                MCPToolSelection("research", "research", "Read cited research."),
            ),
            maximum_outbound_sensitivity=outbound,
        )
        self.binding = status.binding
        self.tool = next(
            item for item in self.binding.tools if item.remote_name == "notify"
        )
        self.research = next(
            item for item in self.binding.tools if item.remote_name == "research"
        )
        await self.reopen()
        return self

    async def reopen(self):
        await self.agent.close()
        self.agent = await Agent.open("mcp-actions", **self.kwargs())

    def script(self, calls=None, *, research=False, no_action=False):
        steps = [
            response(
                ToolCall(
                    id="load",
                    name="toolbox_load",
                    arguments={
                        "tool_names": (self.tool.local_name, self.research.local_name)
                    },
                )
            )
        ]
        if research:
            steps.append(
                response(
                    ToolCall(
                        id="research",
                        name=self.research.local_name,
                        arguments={"query": "Latest status with sources"},
                    )
                )
            )
        if not no_action:
            for call_id, arguments in calls or [
                ("action", {"destination": "fixed-room", "content": "Status"})
            ]:
                steps.append(
                    response(
                        ToolCall(
                            id=call_id, name=self.tool.local_name, arguments=arguments
                        )
                    )
                )
        steps.append(
            response(
                text="Research found improved status based on https://research.test/report; coverage is limited. Check the recorded action evidence."
            )
        )
        self.model.steps = steps

    async def results(self, run_id):
        transcript = await self.agent._embedded._store.load(run_id)
        return [
            block
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        ]

    async def receipts(self):
        return await self.agent._embedded._store.list_effect_receipts(self.agent.id)

    async def draft(
        self, *, minimum=1, ceiling=1, basis=EffectEvidenceBasis.SERVER_REPORTED
    ):
        self.model.steps = [
            response(text="Prepare the exact immediate and recurring assignment.")
        ]
        origin = await self.agent.run(
            "Research status and notify fixed-room now and every Monday."
        )
        self.origin = origin
        destination = (
            await self.agent.distribution_destinations(
                origin.conversation_id, sensitivity_ceiling=ModelSensitivity.INTERNAL
            )
        )[0]
        return ScheduledRoutineDraft(
            origin_run_id=origin.run_id,
            title="Weekly status",
            authorized_instruction="Research current status with citations, then invoke the admitted notification tool for fixed-room. Preserve research and report invocation evidence honestly.",
            schedule=CalendarSchedule(
                timezone="America/Chicago",
                hour=9,
                minute=0,
                day_selector=CalendarDaySelector.WEEKDAYS,
                weekdays=(1,),
            ),
            misfire_policy=MisfirePolicy.LATEST_ONLY,
            reporting_mode=ReportingMode.ALWAYS,
            precheck=None,
            allowed_source_ids=(),
            allowed_resource_ids=(),
            allowed_connector_binding_ids=(self.binding.binding_id,),
            allowed_capability_ids=(
                self.tool.capability_id,
                self.research.capability_id,
            ),
            sensitivity_ceiling=ModelSensitivity.INTERNAL,
            outcome_contract=replace(
                no_artifact_outcome_contract(sensitivity=ModelSensitivity.INTERNAL),
                effect_requirements=(
                    EffectRequirement(
                        self.tool.capability_id, minimum, frozenset({basis})
                    ),
                ),
            ),
            distribution_destination_id=destination.destination_id,
            eligible_model_routes=(self.model.provider_id,),
            per_run_max_tokens=8000,
            per_run_max_cost_usd=Decimal("1"),
            cumulative_max_tokens=40000,
            cumulative_max_cost_usd=Decimal("5"),
            cumulative_max_attempts=5,
            cumulative_max_occurrences=5,
            maximum_consecutive_failures=3,
            expires_at=NOW + timedelta(days=30),
            run_immediately=True,
            requested_capability_grants=(
                RequestedCapabilityGrant(
                    self.tool.capability_id,
                    FrozenJsonObject.from_mapping(
                        {
                            "binding_id": self.binding.binding_id,
                            "binding_revision": self.binding.revision,
                            "remote_tool_name": self.tool.remote_name,
                            "fixed_arguments": {"destination": "fixed-room"},
                            "variable_argument_names": ("content",),
                        }
                    ),
                    ceiling,
                ),
            ),
        )

    async def delivery(self, count=1):
        for _ in range(1000):
            inbox = await self.agent.inbox(conversation_id=self.origin.conversation_id)
            if len(inbox) >= count:
                return await self.agent.inspect_delivery(inbox[0].delivery_id)
            await asyncio.sleep(0.005)
        pytest.fail(f"MCP routine did not deliver: {await self.agent.list_routines()}")
