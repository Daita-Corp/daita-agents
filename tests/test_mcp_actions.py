"""Shared MCP action admission and invocation evidence with fake external I/O."""

import asyncio
import json
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import httpx
import pytest
from _distribution_support import no_artifact_outcome_contract
from _mcp_fixtures import MCPConformanceTransport, MCPFixtureIdentity
from _workspace_support import workspace_for

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
from daita.routines.owner import RoutineError
from daita.routines.models import RoutineState
from daita.storage.sqlite_records import EffectResolutionDecision

NOW = datetime(2026, 9, 6, 12, tzinfo=UTC)


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


@pytest.fixture
async def action(tmp_path):
    fixture = await ActionFixture(tmp_path).start()
    try:
        yield fixture
    finally:
        await fixture.agent.close()


async def test_foreground_plain_text_action_requires_exact_approval_and_deduplicates(
    action,
):
    args = {"destination": "fixed-room", "content": "Private content"}
    action.script(
        [
            ("first", args),
            ("first", args),
            ("new-id", dict(reversed(list(args.items())))),
        ]
    )
    result = await action.agent.run("Send this exact notification once.")
    receipts = await action.receipts()
    assert len(action.server.calls) == len(receipts) == 1
    receipt = receipts[0]
    assert receipt.receipt_kind == "mcp.tool_call"
    assert receipt.outcome is EffectOutcome.SUCCEEDED
    assert receipt.evidence_basis is EffectEvidenceBasis.SERVER_REPORTED
    assert "Private content" not in canonical_json(receipt.payload)
    assert action.approvals[0].arguments["arguments"] == FrozenJsonObject.from_mapping(
        args
    )
    assert "server-reported invocation" in action.approvals[0].reason
    results = await action.results(result.run_id)
    success = next(item for item in results if item.call_id == "first")
    assert not success.is_error and success.output["data"]["text"] == ("success",)
    assert action.tool.output_schema is None
    assert not any(
        method.startswith("tasks/") for method in action.server.request_methods
    )


async def test_discovery_exposes_exact_automation_contract_without_granting_authority(
    action,
):
    from daita.domains.mcp import MCP_GRANT_POLICY

    action.model.steps = [
        response(
            ToolCall(
                "search-contract", "toolbox_search", {"query": "notify", "limit": 20}
            )
        ),
        response(
            ToolCall(
                "unloaded-action",
                action.tool.local_name,
                {"destination": "fixed-room", "content": "Must not dispatch"},
            )
        ),
        response(
            ToolCall(
                "load-contract",
                "toolbox_load",
                {"tool_names": [action.tool.local_name, action.research.local_name]},
            )
        ),
        response(
            text="The contracts are available for a separately approved proposal."
        ),
    ]
    result = await action.agent.run(
        "Inspect the notification's exact standing-grant contract."
    )
    results = {block.call_id: block for block in await action.results(result.run_id)}
    assert results["unloaded-action"].is_error
    assert results["unloaded-action"].output["error"]["code"] == "tool_not_available"
    assert action.tool.local_name not in {
        tool.name for tool in action.model.requests[1].tools
    }
    assert not results["load-contract"].is_error
    matches = results["search-contract"].output["data"]["matches"]
    match = next(
        item for item in matches if item["tool_name"] == action.tool.local_name
    )
    assert match["capability_id"] == action.tool.capability_id
    assert match["requires_automation_grant"] is True
    assert "requires_grant" not in match
    contracts = {
        item["tool_name"]: item
        for item in results["load-contract"].output["data"]["contracts"]
    }
    contract = contracts[action.tool.local_name]
    assert contract["complete"] is False
    assert match["automation_contract"]["complete"] is True
    assert match["automation_contract"]["input_schema"] == action.tool.input_schema
    assert all(
        match["automation_contract"][key] == value
        for key, value in contract.items()
        if key != "complete"
    )
    assert (
        contract["grant_policy"]["constraints_schema"]
        == MCP_GRANT_POLICY.constraints_schema
    )
    assert contract["connector"]["id"] == action.binding.binding_id
    assert contract["connector"]["binding_revision"] == action.binding.revision
    assert contract["connector"]["remote_tool_name"] == "notify"
    assert contract["effect_evidence_basis"] == "server_reported"
    assert contracts[action.research.local_name]["requires_automation_grant"] is False
    assert contracts[action.research.local_name]["grant_policy"] is None
    assert action.server.calls == []
    assert action.approvals == []
    assert await action.receipts() == ()


async def test_search_omits_whole_authoring_contract_before_losing_candidates(action):
    def search_steps():
        return [
            response(
                ToolCall("search", "toolbox_search", {"query": "notify", "limit": 20})
            ),
            response(text="Discovery only."),
        ]

    action.model.steps = search_steps()
    full_run = await action.agent.run("Discover the notification contract.")
    full = (await action.results(full_run.run_id))[0].output["data"]
    byte_limit = len(canonical_json(full).encode("utf-8")) - 1
    runtime = action.agent._embedded._capability_runtime
    runtime._limits = replace(
        runtime._limits, max_toolbox_search_result_bytes=byte_limit
    )
    action.model.steps = search_steps()
    bounded_run = await action.agent.run("Discover the notification contract.")
    bounded_result = (await action.results(bounded_run.run_id))[0]
    assert not bounded_result.is_error
    bounded = bounded_result.output["data"]
    assert len(canonical_json(bounded).encode("utf-8")) <= byte_limit
    assert [item["tool_name"] for item in bounded["matches"]] == [
        item["tool_name"] for item in full["matches"]
    ]
    assert bounded["returned_count"] == full["returned_count"]
    assert bounded["truncated"] == full["truncated"]
    assert bool(bounded["next_cursor"]) == bool(full["next_cursor"])
    match = next(
        item
        for item in bounded["matches"]
        if item["tool_name"] == action.tool.local_name
    )
    assert match["automation_contract_omitted"] is True
    assert "automation_contract" not in match
    assert match["requires_automation_grant"] is True
    assert action.server.calls == action.approvals == []
    assert await action.receipts() == ()


@pytest.mark.parametrize(
    "fixed,variable,missing,unknown",
    [
        (
            {"room": "fixed-room"},
            ["message"],
            ["content", "destination"],
            ["message", "room"],
        ),
        (
            {"destination": "fixed-room", "require_confirmation": True},
            ["content"],
            [],
            ["require_confirmation"],
        ),
    ],
)
async def test_grant_argument_errors_identify_exact_correction(
    action, fixed, variable, missing, unknown
):
    from daita.domains.mcp import MCPCapabilityDomain

    constraints = FrozenJsonObject.from_mapping(
        {
            "binding_id": action.binding.binding_id,
            "binding_revision": action.binding.revision,
            "remote_tool_name": action.tool.remote_name,
            "fixed_arguments": fixed,
            "variable_argument_names": variable,
        }
    )
    with pytest.raises(CapabilityInputError) as failure:
        MCPCapabilityDomain._validate_constraints(
            action.binding, action.tool, constraints
        )
    assert failure.value.code == "mcp_grant_constraints_invalid"
    details = failure.value.details
    assert details["missing_argument_names"] == tuple(missing)
    assert details["unknown_argument_names"] == tuple(unknown)
    assert details["inspect_tool_name"] == action.tool.local_name
    assert action.server.calls == action.approvals == []
    assert await action.receipts() == ()


async def test_mcp_contract_survives_authoring_switch_and_exact_inspection(action):
    action.model.steps = [
        response(
            ToolCall(
                "inspect-via-load",
                "toolbox_load",
                {"tool_names": [action.tool.local_name, action.research.local_name]},
            )
        ),
        response(
            ToolCall("authoring", "toolbox_load", {"tool_names": ["routine_create"]})
        ),
        response(
            ToolCall(
                "inspect-exact",
                "toolbox_inspect",
                {"tool_name": action.tool.local_name},
            )
        ),
        response(text="Inspection only; no assignment or notification was created."),
    ]
    result = await action.agent.run(
        "Inspect the connected action and then prepare to author an assignment."
    )
    assert result.kind.value == "completed"
    request = action.model.requests[2]
    assert "routine_create" in {tool.name for tool in request.tools}
    assert action.tool.local_name not in {tool.name for tool in request.tools}
    retained = next(
        block
        for message in request.messages
        for block in message.content
        if isinstance(block, ToolResultBlock) and block.call_id == "inspect-via-load"
    )
    retained_data = json.loads(canonical_json(retained.output))["data"]
    contracts = {item["tool_name"]: item for item in retained_data["contracts"]}
    for contract in contracts.values():
        assert contract["complete"] is False
        assert "input_schema" not in contract
        assert contract["inspection_tool"] == "toolbox_inspect"
    inspected = next(
        block
        for block in await action.results(result.run_id)
        if block.call_id == "inspect-exact"
    )
    inspected_contract = inspected.output["data"]["value"]
    assert inspected_contract["input_schema"] == action.tool.input_schema
    assert (
        inspected_contract["contract_digest"]
        == contracts[action.tool.local_name]["contract_digest"]
    )
    retained_request = action.model.requests[3]
    assert any(
        isinstance(block, ToolResultBlock) and block == inspected
        for message in retained_request.messages
        for block in message.content
    )
    assert {tool.name for tool in action.model.requests[3].tools} == {
        tool.name for tool in request.tools
    }
    assert action.server.calls == action.approvals == []
    assert await action.receipts() == ()


async def test_exact_inspection_rechecks_revocation_without_remote_io(action):
    from daita.loop.models import RunInput

    run = RunInput(
        id="inspection-revocation",
        agent_id=action.agent.id,
        message="Inspect admitted contracts",
        created_at=NOW,
    )
    runtime = action.agent._embedded._capability_runtime
    catalog = await runtime.prepare_run(run)
    assert action.tool.local_name in {entry.view.name for entry in catalog.entries}
    projection = runtime.project(catalog, ())
    await action.agent.revoke_mcp_server(action.binding.binding_id)
    methods = list(action.server.request_methods)
    outcome = await runtime.execute_all(
        run,
        (
            ToolCall(
                "inspect-revoked",
                "toolbox_inspect",
                {"tool_name": action.tool.local_name},
            ),
        ),
        projection=projection,
        messages=(),
        sensitivity=ModelSensitivity.INTERNAL,
    )
    result = outcome.ordered_results[0]
    assert result.is_error
    assert result.output["error"]["code"] == "toolbox_inspect_stale"
    assert action.server.request_methods == methods
    assert action.server.calls == action.approvals == []


@pytest.mark.parametrize("extra_confirmation", [False, True])
async def test_grant_correction_details_reach_next_real_request_before_any_approval(
    action, extra_confirmation
):
    from daita.routines.capabilities import _spec_schema
    from daita.routines.owner import _routine_proposal_payload

    proposal = await action.agent.propose_routine(
        replace(await action.draft(), run_immediately=False)
    )
    properties = _spec_schema(update=False)["properties"]
    assert isinstance(properties, Mapping)
    arguments = {
        key: value
        for key, value in _routine_proposal_payload(proposal).items()
        if key in properties and value is not None
    }
    grant = proposal.capability_grants[0]
    arguments.update(
        skill_names=(),
        distribution_destination_id=proposal.distribution_plan.targets[
            0
        ].destination_id,
        requested_capability_grants=[
            {
                "capability_id": grant.capability_id,
                "constraints": grant.constraints,
                "max_calls_per_occurrence": grant.max_calls_per_occurrence,
            }
        ],
    )
    invalid = json.loads(canonical_json(arguments))
    constraints = invalid["requested_capability_grants"][0]["constraints"]
    if extra_confirmation:
        constraints["fixed_arguments"]["require_confirmation"] = True
    else:
        constraints["fixed_arguments"] = {"room": "fixed-room"}
        constraints["variable_argument_names"] = ["message"]
    before = len(action.model.requests)
    action.model.steps = [
        response(
            ToolCall(
                "action-schema",
                "toolbox_load",
                {"tool_names": [action.tool.local_name]},
            )
        ),
        response(
            ToolCall(
                "routine-schema", "toolbox_load", {"tool_names": ["routine_create"]}
            )
        ),
        response(ToolCall("invalid-grant", "routine_create", invalid)),
        response(ToolCall("correct-grant", "routine_create", arguments)),
        response(text="Assignment saved; no notification was sent."),
    ]
    approvals_before_correction = []
    original_generate = action.model.generate

    async def generate(request):
        if len(action.model.requests) == before + 3:
            approvals_before_correction.extend(action.approvals)
            assert await action.agent.list_routines() == ()
        return await original_generate(request)

    action.model.generate = generate
    result = await action.agent.run(
        "Save the reviewed assignment for next Monday.",
        conversation_id=action.origin.conversation_id,
    )
    assert result.kind.value == "completed"
    correction_request = action.model.requests[before + 3]
    error = next(
        block
        for message in correction_request.messages
        for block in message.content
        if isinstance(block, ToolResultBlock) and block.call_id == "invalid-grant"
    )
    error_payload = json.loads(canonical_json(error.output))
    assert error_payload["error"]["code"] == "mcp_grant_constraints_invalid"
    assert error_payload["error"]["details"]["unknown_argument_names"] == (
        ["require_confirmation"] if extra_confirmation else ["message", "room"]
    )
    assert approvals_before_correction == []
    assert len(action.approvals) == 1
    assert len(await action.agent.list_routines()) == 1
    assert action.server.calls == []
    assert await action.receipts() == ()


async def test_grant_diagnostic_names_are_bounded_and_omission_is_explicit(action):
    from daita.domains.mcp import MCPCapabilityDomain

    constraints = FrozenJsonObject.from_mapping(
        {
            "binding_id": action.binding.binding_id,
            "binding_revision": action.binding.revision,
            "remote_tool_name": action.tool.remote_name,
            "fixed_arguments": {f"bad_{i}_" + "é" * 200: "unused" for i in range(40)},
            "variable_argument_names": ["content"],
        }
    )
    with pytest.raises(CapabilityInputError) as failure:
        MCPCapabilityDomain._validate_constraints(
            action.binding, action.tool, constraints
        )
    details = json.loads(canonical_json(failure.value.details))
    assert details["names_truncated"] is True
    assert len(details["unknown_argument_names"]) == 32
    assert details["argument_counts"]["unknown_argument_names"] == 40
    assert all(len(name) <= 128 for name in details["unknown_argument_names"])
    assert len(canonical_json(details).encode()) < 16 * 1024


@pytest.mark.parametrize(
    "fault",
    [
        "disconnect",
        "tool_error",
        "malformed",
        "oversized",
        "timeout",
        "accepted_async",
        "schema_mismatch",
        "runtime_size",
    ],
)
async def test_possible_dispatch_failures_are_uncertain_and_never_replayed(
    action, fault
):
    if fault == "tool_error":
        action.server.results["notify"] = {
            "content": [{"type": "text", "text": "Partially applied, then failed"}],
            "isError": True,
        }
    elif fault == "accepted_async":
        action.server.results["notify"] = {
            "task": {"taskId": "operation-42", "status": "working"}
        }
    elif fault == "schema_mismatch":
        action.server.results["notify"] = {
            "content": [{"type": "image", "data": "invalid"}]
        }
    elif fault == "runtime_size":
        action.server.results["notify"] = {
            "content": [{"type": "text", "text": "x" * (256 * 1024)}]
        }
    else:
        action.transport.mode = fault
    action.script(
        [
            ("action", {"destination": "fixed-room", "content": "first"}),
            ("changed", {"destination": "another-room", "content": "changed"}),
        ]
    )
    result = await action.agent.run("Send the notification.")
    receipts = await action.receipts()
    assert len(action.server.calls) == len(receipts) == 1
    assert receipts[0].outcome is EffectOutcome.UNCERTAIN
    if fault == "accepted_async":
        assert receipts[0].payload["operation_handle"] == "operation-42"
        assert receipts[0].evidence_basis is EffectEvidenceBasis.SERVER_REPORTED
    if fault == "tool_error":
        assert receipts[0].payload["classification"] == "tool_error"
    results = await action.results(result.run_id)
    assert next(item for item in results if item.call_id == "action").is_error
    assert not any(
        method.startswith("tasks/") for method in action.server.request_methods
    )
    await action.reopen()
    action.script()
    await action.agent.run("Send a fresh notification.")
    assert len(action.server.calls) == 1


@pytest.mark.acceptance
async def test_immediate_and_recurring_mcp_only_assignment_uses_standing_grant(action):
    proposal = await action.agent.propose_routine(await action.draft())
    from daita.routines.capabilities import _spec_schema
    from daita.routines.owner import _routine_proposal_payload

    properties = _spec_schema(update=False)["properties"]
    assert isinstance(properties, Mapping)
    arguments = {
        key: value
        for key, value in _routine_proposal_payload(proposal).items()
        if key in properties and value is not None
    }
    arguments.update(
        skill_names=(),
        distribution_destination_id=proposal.distribution_plan.targets[
            0
        ].destination_id,
        requested_capability_grants=tuple(
            {
                "capability_id": grant.capability_id,
                "constraints": grant.constraints,
                "max_calls_per_occurrence": grant.max_calls_per_occurrence,
            }
            for grant in proposal.capability_grants
        ),
    )
    action.script(research=True)
    action.model.steps = [
        response(
            ToolCall(
                id="load-create",
                name="toolbox_load",
                arguments={"tool_names": ("routine_create",)},
            )
        ),
        response(
            ToolCall(id="create-assignment", name="routine_create", arguments=arguments)
        ),
        response(text="The exact assignment was approved."),
        *action.model.steps,
    ]
    creation = await action.agent.run(
        "Create the reviewed immediate and weekly assignment.",
        conversation_id=action.origin.conversation_id,
    )
    creation_results = await action.results(creation.run_id)
    created = next(
        item for item in creation_results if item.call_id == "create-assignment"
    )
    assert not created.is_error, created.output
    delivery = await action.delivery()
    assert delivery.delivery.outcome.conclusion_state is OutcomeState.SUCCEEDED
    assert [name for name, _ in action.server.calls] == ["research", "notify"]
    assert len(action.approvals) == 1
    approval = canonical_json(action.approvals[0].arguments)
    assert '"fixed_arguments":{"destination":"fixed-room"}' in approval
    assert '"variable_argument_names":["content"]' in approval
    action.script(
        [("weekly", {"destination": "fixed-room", "content": "Weekly content"})],
        research=True,
    )
    action.clock = datetime(2026, 9, 7, 14, tzinfo=UTC)
    action.agent._embedded._routine_supervisor.wake()
    weekly = await action.delivery(2)
    assert weekly.delivery.outcome.conclusion_state is OutcomeState.SUCCEEDED
    assert len(await action.receipts()) == 2
    assert len(action.approvals) == 1
    assert action.server.calls[-1] == (
        "notify",
        {"destination": "fixed-room", "content": "Weekly content"},
    )


@pytest.mark.parametrize(
    "minimum,no_action,error",
    [(1, True, False), (0, True, False), (1, False, True), (0, False, True)],
)
@pytest.mark.acceptance
async def test_required_action_and_partial_research_are_reported_honestly(
    action, minimum, no_action, error
):
    proposal = await action.agent.propose_routine(await action.draft(minimum=minimum))
    if error:
        action.server.results["notify"] = {
            "content": [{"type": "text", "text": "Partial application error"}],
            "isError": True,
        }
    action.script(research=True, no_action=no_action)
    routine = await action.agent.create_routine(proposal)
    delivery = await action.delivery()
    expected = (
        OutcomeState.SUCCEEDED if minimum == 0 and no_action else OutcomeState.FAILED
    )
    assert delivery.delivery.outcome.conclusion_state is expected
    inspection = await action.agent.inspect_routine(routine.routine_id)
    results = await action.results(inspection.recent_occurrences[0].reserved_run_id)
    research = next(item for item in results if item.call_id == "research")
    assert not research.is_error
    assert "research.test/report" in canonical_json(research.output)


@pytest.mark.acceptance
async def test_server_reported_action_cannot_promise_adapter_verified_completion(
    action,
):
    with pytest.raises((ValueError, CapabilityInputError, RoutineError)):
        await action.agent.propose_routine(
            await action.draft(basis=EffectEvidenceBasis.ADAPTER_VERIFIED)
        )
    assert not action.server.calls


@pytest.mark.parametrize(
    "invalid", ["fixed_destination", "missing_fixed", "extra_argument", "wrong_type"]
)
async def test_standing_invocations_enforce_fixed_values_and_variable_names(
    action, invalid
):
    proposal = await action.agent.propose_routine(await action.draft())
    arguments: dict[str, object] = {"destination": "fixed-room", "content": "Status"}
    if invalid == "fixed_destination":
        arguments["destination"] = "unapproved-room"
    elif invalid == "missing_fixed":
        del arguments["destination"]
    elif invalid == "extra_argument":
        arguments["options"] = {"recipient": "unapproved-room"}
    else:
        arguments["content"] = {"nested": "unapproved"}
    action.script([("invalid", arguments)])
    routine = await action.agent.create_routine(proposal)
    delivery = await action.delivery()
    assert delivery.delivery.outcome.conclusion_state is OutcomeState.FAILED
    assert not action.server.calls and not await action.receipts()


@pytest.mark.parametrize(
    "invalid",
    [
        "binding",
        "revision",
        "tool",
        "overlap",
        "duplicate",
        "unknown",
        "required",
        "nested",
        "fixed_type",
    ],
)
async def test_grant_normalization_rejects_unenforceable_or_wrong_contracts(
    action, invalid
):
    draft = await action.draft()
    grant = draft.requested_capability_grants[0]
    constraints = grant.constraints.to_dict()
    if invalid == "binding":
        constraints["binding_id"] = "another-binding"
    elif invalid == "revision":
        constraints["binding_revision"] += 1
    elif invalid == "tool":
        constraints["remote_tool_name"] = "research"
    elif invalid == "overlap":
        constraints["variable_argument_names"] = ["content", "destination"]
    elif invalid == "duplicate":
        constraints["variable_argument_names"] = ["content", "content"]
    elif invalid == "unknown":
        constraints["variable_argument_names"] = ["content", "invented"]
    elif invalid == "required":
        constraints["fixed_arguments"] = {}
    elif invalid == "nested":
        constraints["variable_argument_names"] = ["content", "options"]
    else:
        constraints["fixed_arguments"] = {"destination": 1}
    draft = replace(
        draft,
        requested_capability_grants=(
            replace(grant, constraints=FrozenJsonObject.from_mapping(constraints)),
        ),
    )
    with pytest.raises((ValueError, RuntimeError)):
        await action.agent.propose_routine(draft)
    assert not action.server.calls and not await action.agent.list_routines()


async def test_fixed_nested_values_are_exact_and_scalar_call_allowance_is_durable(
    action,
):
    draft = await action.draft(ceiling=2, minimum=2)
    grant = draft.requested_capability_grants[0]
    constraints = grant.constraints.to_dict()
    constraints["fixed_arguments"] = {
        "destination": "fixed-room",
        "options": {"recipient": "reviewed"},
    }
    draft = replace(
        draft,
        requested_capability_grants=(
            replace(grant, constraints=FrozenJsonObject.from_mapping(constraints)),
        ),
    )
    proposal = await action.agent.propose_routine(draft)
    fixed = constraints["fixed_arguments"]
    action.script(
        [
            ("first", {**fixed, "content": "one"}),
            ("same", {"content": "one", **fixed}),
            ("second", {**fixed, "content": "two"}),
            ("exhausted", {**fixed, "content": "three"}),
            (
                "nested-change",
                {
                    "destination": "fixed-room",
                    "options": {"recipient": "changed"},
                    "content": "four",
                },
            ),
        ]
    )
    routine = await action.agent.create_routine(proposal)
    await action.delivery()
    assert len(action.server.calls) == len(await action.receipts()) == 2
    inspection = await action.agent.inspect_routine(routine.routine_id)
    results = await action.results(inspection.recent_occurrences[0].reserved_run_id)
    assert next(item for item in results if item.call_id == "exhausted").is_error
    assert next(item for item in results if item.call_id == "nested-change").is_error


@pytest.mark.parametrize(
    "boundary",
    [
        "denied",
        "reservation",
        "finish",
        "cancel_before_dispatch",
        "crash_before_dispatch",
    ],
)
async def test_runtime_owned_durable_boundaries_control_mcp_dispatch(
    action, monkeypatch, boundary
):
    store = action.agent._embedded._store
    if boundary == "denied":
        action.decision = ApprovalDecision.DENY
    elif boundary in {"reservation", "finish"}:

        async def fail(*args, **kwargs):
            raise OSError("disk unavailable")

        monkeypatch.setattr(
            store,
            (
                "start_effect_receipt"
                if boundary == "reservation"
                else "finish_effect_receipt"
            ),
            fail,
        )
    else:
        original = store.start_effect_receipt

        async def reserve(*args, **kwargs):
            result = await original(*args, **kwargs)
            if boundary == "crash_before_dispatch":
                raise OSError("process stopped after durable reservation")
            current = asyncio.current_task()
            assert current is not None
            current.cancel()
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                pass
            return result

        monkeypatch.setattr(store, "start_effect_receipt", reserve)
    action.script()
    await action.agent.run("Send the exact notification once.")
    receipts = await action.receipts()
    assert len(action.server.calls) == (1 if boundary == "finish" else 0)
    if boundary in {"denied", "reservation"}:
        assert not receipts
    elif boundary == "cancel_before_dispatch":
        assert receipts[0].outcome is EffectOutcome.NOT_APPLIED
        assert receipts[0].evidence_basis is EffectEvidenceBasis.LOCAL_NOT_DISPATCHED
    else:
        assert receipts[0].outcome is EffectOutcome.STARTED
        action.script()
        await action.agent.run("Send with a new run identity.")
        assert len(action.server.calls) == (1 if boundary == "finish" else 0)
        await action.reopen()
        assert (await action.receipts())[0].outcome is EffectOutcome.UNCERTAIN
        action.script()
        await action.agent.run("Try another message after restart.")
        assert len(action.server.calls) == (1 if boundary == "finish" else 0)


async def test_cancel_after_possible_dispatch_is_uncertain(action):
    action.transport.mode = "cancel"
    action.script()
    task = asyncio.create_task(action.agent.run("Send the notification."))
    await asyncio.wait_for(action.transport.dispatched.wait(), timeout=5)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    receipts = await action.receipts()
    assert len(action.server.calls) == len(receipts) == 1
    assert receipts[0].outcome is EffectOutcome.UNCERTAIN


@pytest.mark.parametrize(
    "change",
    [
        "schema",
        "identity",
        "task_support",
        "revoked",
        "authentication",
        "outbound",
        "hints",
    ],
)
async def test_exact_call_admission_drift_and_full_request_sensitivity(action, change):
    if change == "schema":
        action.server.tool("notify")["inputSchema"]["properties"]["content"][
            "maxLength"
        ] = 20
    elif change == "identity":
        action.server.server_version = "changed"
    elif change == "task_support":
        action.server.tool("notify")["execution"] = {"taskSupport": "required"}
    elif change == "revoked":
        await action.agent.revoke_mcp_server(action.binding.binding_id)
    elif change == "authentication":
        action.server.bearer_token = "now-required"
    elif change == "outbound":
        await action.agent.set_memory(
            "A retained private research note.", sensitivity=ModelSensitivity.RESTRICTED
        )
    else:
        await action.agent.update_mcp_discovery(
            action.binding.binding_id,
            summary="Changed wording",
            when_to_use="Use for reviewed status",
            keywords=("status",),
        )
    action.script()
    await action.agent.run(
        "Send the notification with the current full request context."
    )
    assert len(action.server.calls) == (1 if change == "hints" else 0)
    assert len(await action.receipts()) == (1 if change == "hints" else 0)


@pytest.mark.parametrize("known", ["local_async", "remote_required", "remote_optional"])
async def test_known_completion_semantics_never_enable_unsupported_unattended_calls(
    tmp_path, known
):
    fixture = ActionFixture(tmp_path)
    if known != "local_async":
        fixture.server.tool("notify")["execution"] = {
            "taskSupport": "required" if known == "remote_required" else "optional"
        }
    selection = MCPToolSelection(
        "notify",
        "notify",
        "Reviewed invocation.",
        access_mode=AccessMode.NONE,
        operational_effect=OperationalEffect.EXTERNAL_ACTION,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
        completion_semantics=(
            MCPCompletionSemantics.ASYNCHRONOUS_ONLY
            if known == "local_async"
            else MCPCompletionSemantics.DIRECT_RESULT
        ),
    )
    await fixture.start(selection=selection)
    try:
        with pytest.raises((ValueError, RuntimeError, RoutineError)):
            await fixture.agent.propose_routine(await fixture.draft())
        fixture.script()
        await fixture.agent.run("Invoke only a supported direct-result action.")
        assert len(fixture.server.calls) == (1 if known == "remote_optional" else 0)
        assert not any(
            method.startswith("tasks/") for method in fixture.server.request_methods
        )
    finally:
        await fixture.agent.close()


@pytest.mark.parametrize("valid", [True, False])
async def test_optional_output_schema_is_enforced_when_present(tmp_path, valid):
    fixture = ActionFixture(tmp_path)
    fixture.server.tool("notify")["outputSchema"] = {
        "type": "object",
        "properties": {"id": {"type": "integer"}},
        "required": ["id"],
        "additionalProperties": False,
    }
    fixture.server.results["notify"]["structuredContent"] = {
        "id": 42 if valid else "invalid"
    }
    await fixture.start()
    try:
        fixture.script()
        await fixture.agent.run("Invoke the reviewed action.")
        receipts = await fixture.receipts()
        assert len(fixture.server.calls) == len(receipts) == 1
        assert receipts[0].outcome is (
            EffectOutcome.SUCCEEDED if valid else EffectOutcome.UNCERTAIN
        )
    finally:
        await fixture.agent.close()


@pytest.mark.parametrize(
    "corrupt",
    ["missing_observation", "invalid_observation", "provenance", "output_sensitivity"],
)
async def test_unusable_domain_outputs_use_runtime_uncertainty(
    action, monkeypatch, corrupt
):
    executor = action.agent._embedded._mcp_activated_bindings[
        action.binding.binding_id
    ].executor
    original = executor.execute

    async def changed(request):
        output = await original(request)
        if corrupt == "missing_observation":
            return replace(output, effect_observation=None)
        if corrupt == "invalid_observation":
            return replace(
                output,
                effect_observation=replace(
                    output.effect_observation,
                    payload=FrozenJsonObject.from_mapping({"invented": "evidence"}),
                ),
            )
        if corrupt == "output_sensitivity":
            return replace(output, sensitivity=None, sensitivity_provenance={})
        data = output.data.to_dict()
        data["provenance"]["remote_tool_name"] = "wrong-tool"
        return replace(output, data=data)

    monkeypatch.setattr(executor, "execute", changed)
    action.script()
    await action.agent.run("Invoke the exact action once.")
    assert len(action.server.calls) == 1
    assert (await action.receipts())[0].outcome is EffectOutcome.UNCERTAIN


async def test_revocation_after_reservation_proves_local_non_dispatch(
    action, monkeypatch
):
    store = action.agent._embedded._store
    reserve = store.start_effect_receipt

    async def revoke_after_reservation(*args, **kwargs):
        receipt = await reserve(*args, **kwargs)
        await store.store_mcp_binding(
            action.binding.revoke(revoked_at=action.clock),
            expected_revision=action.binding.revision,
        )
        return receipt

    monkeypatch.setattr(store, "start_effect_receipt", revoke_after_reservation)
    action.script()
    await action.agent.run("Invoke the exact current tool.")
    receipts = await action.receipts()
    assert len(receipts) == 1 and not action.server.calls
    assert receipts[0].outcome is EffectOutcome.NOT_APPLIED
    assert receipts[0].evidence_basis is EffectEvidenceBasis.LOCAL_NOT_DISPATCHED


async def test_binding_codec_and_origin_retain_authority_but_exclude_hints(action):
    from daita.adapters.mcp import mcp_execution_origin_digest
    from daita.storage.sqlite_codecs import encode_mcp_binding, decode_mcp_binding

    binding = action.binding
    encoded = encode_mcp_binding(binding)
    assert (
        decode_mcp_binding(
            encoded, agent_id=binding.agent_id, binding_id=binding.binding_id
        )
        == binding
    )
    origin = mcp_execution_origin_digest(binding, action.tool)
    changed_hints = replace(
        binding, summary="Edited wording", when_to_use="Different discovery hint"
    )
    assert (
        mcp_execution_origin_digest(
            changed_hints, replace(action.tool, description="New local description")
        )
        == origin
    )
    for tool in (
        replace(action.tool, access_mode=AccessMode.READ),
        replace(action.tool, operational_effect=OperationalEffect.NONE),
        replace(
            action.tool, automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY
        ),
        replace(action.tool, result_sensitivity=ModelSensitivity.RESTRICTED),
        replace(action.tool, maximum_outbound_sensitivity=ModelSensitivity.INTERNAL),
        replace(action.tool, task_support="optional"),
    ):
        assert mcp_execution_origin_digest(binding, tool) != origin
    payload = json.loads(encoded)

    # The strict codec accepts only the current shape, with no compatibility defaults.
    def remove_field(value):
        if isinstance(value, dict):
            if "task_support" in value:
                del value["task_support"]
                return True
            return any(remove_field(item) for item in value.values())
        if isinstance(value, list):
            return any(remove_field(item) for item in value)
        return False

    assert remove_field(payload)
    with pytest.raises(ValueError):
        decode_mcp_binding(
            json.dumps(payload),
            agent_id=binding.agent_id,
            binding_id=binding.binding_id,
        )


async def test_guided_ui_explicit_action_permissions_reach_public_admission(tmp_path):
    from textual.widgets import Input, OptionList, Select
    from daita.tui.app import DaitaApp
    from daita.tui.screens.confirm import ConfirmScreen
    from daita.tui.screens.mcp import MCPSetupScreen, MCPToolAdmissionScreen
    from daita.tui.screens.selection import SelectionScreen

    fixture = ActionFixture(tmp_path)
    opened = await Agent.create("mcp-ui-action", **fixture.kwargs())
    app = DaitaApp(
        root=tmp_path, start_bootstrap=False, workspace=workspace_for(tmp_path)
    )
    app.controller.agent = opened
    async with app.run_test(size=(104, 42)) as pilot:
        await app._show_chat()
        setup_task = asyncio.create_task(app._open_command_screen("mcp_setup", {}))
        try:
            await pilot.pause()
            assert isinstance(app.screen, MCPSetupScreen)
            app.screen.query_one("#mcp-endpoint", Input).value = fixture.server.endpoint
            assert await pilot.click("#mcp-inspect")
            await pilot.pause()
            assert await pilot.click("#mcp-select")
            await pilot.pause()
            assert isinstance(app.screen, SelectionScreen)
            app.screen.query_one("#picker-options", OptionList).highlighted = 0
            app.screen.action_toggle_selected()
            app.screen.action_confirm()
            await pilot.pause()
            assert await pilot.click("#mcp-configure")
            await pilot.pause()
            assert isinstance(app.screen, SelectionScreen)
            app.screen.query_one("#picker-options", OptionList).highlighted = 0
            app.screen.action_confirm()
            await pilot.pause()
            assert isinstance(app.screen, MCPToolAdmissionScreen)
            app.screen.query_one("#mcp-tool-description", Input).value = (
                "Send the reviewed notification."
            )
            for name, value in {
                "access": "none",
                "effect": "external_action",
                "eligibility": "automation_direct",
                "outbound": "internal",
                "result": "internal",
                "completion": "direct_result",
            }.items():
                app.screen.query_one(f"#mcp-tool-{name}", Select).value = value
            assert await pilot.click("#mcp-admission-save")
            await pilot.pause()
            assert isinstance(app.screen, MCPSetupScreen)
            assert await pilot.click("#mcp-attach")
            await pilot.pause()
            attestation = app.screen
            assert isinstance(attestation, ConfirmScreen)
            await pilot.press("y")
            for _ in range(100):
                await pilot.pause(0.05)
                if (
                    isinstance(app.screen, ConfirmScreen)
                    and app.screen is not attestation
                ):
                    break
            assert (
                isinstance(app.screen, ConfirmScreen) and app.screen is not attestation
            )
            await pilot.press("n")
            await setup_task
            statuses = await app.controller.list_mcp_servers()
            assert len(statuses) == 1
            tool = statuses[0].binding.tools[0]
            assert tool.remote_name == "notify"
            assert tool.access_mode is AccessMode.NONE
            assert tool.operational_effect is OperationalEffect.EXTERNAL_ACTION
            assert (
                tool.automation_eligibility is AutomationEligibility.AUTOMATION_DIRECT
            )
            assert tool.maximum_outbound_sensitivity is ModelSensitivity.INTERNAL
            assert not fixture.server.calls
            app.exit(0)

        finally:
            setup_task.cancel()
            await asyncio.gather(setup_task, return_exceptions=True)


@pytest.mark.parametrize(
    "decision",
    [
        EffectResolutionDecision.ALLOW_FUTURE_WORK,
        EffectResolutionDecision.CLOSE_WITHOUT_RETRY,
    ],
)
@pytest.mark.acceptance
async def test_disconnect_blocks_restart_future_slots_controls_revision_clones_until_exact_recovery(
    action, decision
):
    draft = await action.draft()
    proposal = await action.agent.propose_routine(draft)
    action.transport.mode = "disconnect"
    action.script(research=True)
    routine = await action.agent.create_routine(proposal)
    delivery = await action.delivery()
    assert delivery.delivery.outcome.conclusion_state is OutcomeState.FAILED
    receipt = (await action.receipts())[0]
    assert receipt.outcome is EffectOutcome.UNCERTAIN
    assert receipt.receipt_id in delivery.delivery.outcome.effect_receipt_ids
    await action.reopen()
    action.clock = datetime(2026, 9, 7, 14, tzinfo=UTC)
    action.agent._embedded._routine_supervisor.wake()
    inspection = await action.agent.inspect_routine(routine.routine_id)
    assert inspection.routine.state is RoutineState.PAUSED
    revision = inspection.routine.revision
    for control in (action.agent.resume_routine, action.agent.run_routine_now):
        with pytest.raises((ValueError, RuntimeError, RoutineError)):
            await control(routine.routine_id, expected_revision=revision)
    with pytest.raises((ValueError, RuntimeError, RoutineError)):
        await action.agent.update_routine(
            routine.routine_id,
            expected_revision=revision,
            draft=replace(
                draft,
                run_immediately=False,
                authorized_instruction="Send different content to fixed-room.",
            ),
        )
    with pytest.raises((ValueError, RuntimeError, RoutineError)):
        clone = await action.agent.propose_routine(replace(draft, title="Clone"))
        await action.agent.create_routine(clone)
    action.script([("changed-run", {"destination": "changed-room", "content": "new"})])
    await action.agent.run("Send a changed notification now.")
    assert [name for name, _ in action.server.calls].count("notify") == 1
    with pytest.raises((ValueError, RuntimeError, RoutineError)):
        await action.agent.resolve_effect(
            receipt.receipt_id,
            expected_digest="sha256:" + "0" * 64,
            decision=decision,
            note="Stale review.",
        )
    action.decision = ApprovalDecision.DENY
    with pytest.raises(PermissionError):
        await action.agent.resolve_effect(
            receipt.receipt_id,
            expected_digest=receipt.receipt_digest,
            decision=decision,
            note="Denied review.",
        )
    assert (await action.agent.inspect_effect(receipt.receipt_id)).unresolved
    action.decision = ApprovalDecision.APPROVE
    resolved = await action.agent.resolve_effect(
        receipt.receipt_id,
        expected_digest=receipt.receipt_digest,
        decision=decision,
        note="Reviewed external state; accept this exact recovery decision without repeating the old operation.",
    )
    assert resolved.receipt_digest == receipt.receipt_digest
    assert resolved.outcome is EffectOutcome.UNCERTAIN and not resolved.unresolved
    assert [name for name, _ in action.server.calls].count("notify") == 1
    inspection = await action.agent.inspect_routine(routine.routine_id)
    assert inspection.routine.state is (
        RoutineState.PAUSED
        if decision is EffectResolutionDecision.ALLOW_FUTURE_WORK
        else RoutineState.DISABLED
    )
    if decision is EffectResolutionDecision.ALLOW_FUTURE_WORK:
        action.transport.mode = "success"
        action.script(
            [
                (
                    "future-authorized",
                    {
                        "destination": "fixed-room",
                        "content": "A separately authorized future notification",
                    },
                )
            ]
        )
        resumed = await action.agent.resume_routine(
            routine.routine_id, expected_revision=inspection.routine.revision
        )
        await action.agent.run_routine_now(
            routine.routine_id, expected_revision=resumed.revision
        )
        await action.delivery(2)
        assert [name for name, _ in action.server.calls].count("notify") == 2
        assert len(await action.receipts()) == 2
        assert await action.agent.inspect_effect(receipt.receipt_id) == resolved


def test_action_admission_is_explicit_and_independent_of_data_access():
    read = MCPToolSelection("lookup", "lookup", "Read the selected record.")
    assert read.access_mode is AccessMode.READ
    assert read.operational_effect is OperationalEffect.NONE
    assert read.automation_eligibility is AutomationEligibility.AUTOMATION_DIRECT
    action = MCPToolSelection(
        "notify",
        "notify",
        "Send a notification.",
        access_mode=AccessMode.NONE,
        operational_effect=OperationalEffect.EXTERNAL_ACTION,
    )
    assert action.automation_eligibility is AutomationEligibility.INTERACTIVE_ONLY
    admitted = replace(
        action, automation_eligibility=AutomationEligibility.AUTOMATION_DIRECT
    )
    assert admitted.access_mode is AccessMode.NONE
    with pytest.raises(ValueError, match="unsupported"):
        replace(action, operational_effect=OperationalEffect.CHANGE_INFRASTRUCTURE)
    with pytest.raises(ValueError, match="effect"):
        replace(read, access_mode=AccessMode.WRITE)
