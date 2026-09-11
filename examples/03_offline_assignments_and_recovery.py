"""Run research, an artifact, a weekly MCP action and human recovery without a server."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx

from _shared import (
    OFFLINE_PROFILE,
    ScriptedModel,
    example_root,
    final_response,
    parser,
    tool_response,
)
from daita import (
    Agent,
    ApprovalDecision,
    CalendarDaySelector,
    CalendarSchedule,
    EffectRequirement,
    EffectResolutionDecision,
    LocalWorkspace,
    MCPToolSelection,
    MisfirePolicy,
    ReportingMode,
    RequestedCapabilityGrant,
    ScheduledRoutineDraft,
)
from daita._json import FrozenJsonObject
from daita.adapters.mcp import StreamableHTTPMCPClientFactory
from daita.artifacts.models import ArtifactAuthorship
from daita.capabilities import (
    AccessMode,
    AutomationEligibility,
    EffectEvidenceBasis,
    OperationalEffect,
)
from daita.distribution import ArtifactRequirement, OutcomeContract
from daita.llm.models import ModelSensitivity, ModelUsage
from daita.llm.pricing import CostEstimate


class OfflineModel(ScriptedModel):
    def has_complete_pricing(self, request):
        return True

    def extend(self, *responses):
        # Explicit fictional usage keeps the real budget controller exercised.
        super().extend(
            *(
                replace(
                    item,
                    usage=ModelUsage(
                        input_tokens=100,
                        output_tokens=20,
                        cost_estimate=CostEstimate.complete(Decimal("0")),
                    ),
                )
                for item in responses
            )
        )


class OfflineService:
    """HTTP MockTransport handles every request in memory; no socket is opened."""

    def __init__(self):
        self.calls = []
        self.disconnect = False

    async def __call__(self, request):
        payload = json.loads(request.content)
        method = payload["method"]
        if method == "notifications/initialized":
            return httpx.Response(202, request=request)
        if method == "initialize":
            result = {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "offline-example", "version": "1"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "research",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                            "additionalProperties": False,
                        },
                    },
                    {
                        "name": "notify",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "destination": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["destination", "content"],
                            "additionalProperties": False,
                        },
                    },
                ]
            }
        elif method == "tools/call":
            name = payload["params"]["name"]
            self.calls.append((name, payload["params"]["arguments"]))
            if name == "notify" and self.disconnect:
                raise httpx.ReadError(
                    "Offline simulation: service acted, then lost its response",
                    request=request,
                )
            text = (
                "Invocation returned normally."
                if name == "notify"
                else (
                    "Example Co reports growth at https://example.test/research. This is fictional, incomplete research."
                )
            )
            result = {"content": [{"type": "text", "text": text}]}
        else:
            raise AssertionError(f"Unexpected method: {method}")
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": payload["id"], "result": result},
            request=request,
        )


async def run() -> None:
    arguments = parser(__doc__).parse_args()
    with example_root(arguments.root, "assignments") as root:
        workspace_root = root.parent / "workspace"
        workspace_root.mkdir(exist_ok=True)
        model = OfflineModel()
        service = OfflineService()
        now = [datetime(2026, 9, 6, 12, tzinfo=UTC)]

        async def approve(request):
            # Auto-approval is restricted to this entirely simulated walkthrough.
            document = request.render_arguments_for_review()
            assert document is not None
            print(f"Offline approval: {request.tool_name}\n{document}")
            return ApprovalDecision.APPROVE

        options = dict(
            root=root,
            workspace=LocalWorkspace(workspace_root),
            model=model,
            model_profile=OFFLINE_PROFILE,
            clock=lambda: now[0],
            approval_handler=approve,
            mcp_client_factory=StreamableHTTPMCPClientFactory(
                http_transport=httpx.MockTransport(service)
            ),
        )
        agent = await Agent.create("offline-assignments", **options)
        try:
            status = await agent.attach_mcp_server(
                endpoint="https://offline.example.test/mcp",
                selections=(
                    MCPToolSelection(
                        "research", "research", "Read fictional cited research."
                    ),
                    MCPToolSelection(
                        "notify",
                        "notify",
                        "Simulate a notification.",
                        access_mode=AccessMode.NONE,
                        operational_effect=OperationalEffect.EXTERNAL_ACTION,
                        automation_eligibility=AutomationEligibility.AUTOMATION_DIRECT,
                    ),
                ),
                maximum_outbound_sensitivity=ModelSensitivity.INTERNAL,
            )
            # Admission is persisted; one controlled reopen composes its immutable tools.
            await agent.close()
            agent = await Agent.open("offline-assignments", **options)
            tools = {item.remote_name: item for item in status.binding.tools}
            model.extend(final_response("Prepare one immediate and weekly assignment."))
            origin = await agent.run(
                "Research status, create a briefing and notify demo-room now and every Monday."
            )
            assert origin.conversation_id is not None
            destination = (
                await agent.distribution_destinations(
                    origin.conversation_id,
                    sensitivity_ceiling=ModelSensitivity.INTERNAL,
                )
            )[0]

            def script():
                model.extend(
                    tool_response(
                        "load",
                        "toolbox_load",
                        {
                            "tool_names": (
                                tools["research"].local_name,
                                tools["notify"].local_name,
                                "artifact_create_document",
                            )
                        },
                    ),
                    tool_response(
                        "research",
                        tools["research"].local_name,
                        {"query": "latest example status"},
                    ),
                    tool_response(
                        "brief",
                        "artifact_create_document",
                        {
                            "format": "markdown",
                            "content": "# Example briefing\n[Source](https://example.test/research) reports growth. Fictional research; coverage is incomplete.",
                        },
                    ),
                    tool_response(
                        "notify",
                        tools["notify"].local_name,
                        {
                            "destination": "demo-room",
                            "content": "The fictional briefing is ready.",
                        },
                    ),
                    final_response(
                        "Research and briefing are ready; consult the receipt for notification evidence."
                    ),
                )

            script()
            outcome = OutcomeContract(
                require_terminal_conclusion=True,
                artifact_requirements=(
                    ArtifactRequirement(
                        required=True,
                        minimum_count=1,
                        maximum_count=1,
                        allowed_media_types=("text/markdown",),
                        allowed_authorships=(
                            ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS,
                        ),
                        allowed_producer_capability_ids=("artifact.create_document",),
                        maximum_artifact_bytes=4096,
                        maximum_total_bytes=4096,
                        maximum_sensitivity=ModelSensitivity.INTERNAL,
                    ),
                ),
                maximum_total_artifact_bytes=4096,
                maximum_effective_sensitivity=ModelSensitivity.INTERNAL,
                require_current_run_provenance=True,
                require_exact_source_bindings=False,
                effect_requirements=(
                    EffectRequirement(
                        tools["notify"].capability_id,
                        1,
                        frozenset({EffectEvidenceBasis.SERVER_REPORTED}),
                    ),
                ),
            )
            draft = ScheduledRoutineDraft(
                origin_run_id=origin.run_id,
                title="Weekly example briefing",
                authorized_instruction="Read cited example research, create a Markdown briefing, and invoke the notification tool for demo-room. Report incomplete coverage and invocation evidence only.",
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
                allowed_connector_binding_ids=(status.binding.binding_id,),
                allowed_capability_ids=(
                    tools["research"].capability_id,
                    tools["notify"].capability_id,
                    "artifact.create_document",
                ),
                sensitivity_ceiling=ModelSensitivity.INTERNAL,
                outcome_contract=outcome,
                distribution_destination_id=destination.destination_id,
                eligible_model_routes=(model.provider_id,),
                per_run_max_tokens=8000,
                per_run_max_cost_usd=Decimal("1"),
                cumulative_max_tokens=24000,
                cumulative_max_cost_usd=Decimal("3"),
                cumulative_max_attempts=3,
                cumulative_max_occurrences=3,
                maximum_consecutive_failures=2,
                expires_at=now[0] + timedelta(days=30),
                run_immediately=True,
                requested_capability_grants=(
                    RequestedCapabilityGrant(
                        tools["notify"].capability_id,
                        FrozenJsonObject.from_mapping(
                            {
                                "binding_id": status.binding.binding_id,
                                "binding_revision": status.binding.revision,
                                "remote_tool_name": "notify",
                                "fixed_arguments": {"destination": "demo-room"},
                                "variable_argument_names": ("content",),
                            }
                        ),
                        1,
                    ),
                ),
            )
            routine = await agent.create_routine(
                await agent.propose_routine(draft), confirmation_handler=approve
            )

            async def wait_for_deliveries(count):
                for _ in range(1000):
                    items = await agent.inbox(conversation_id=origin.conversation_id)
                    if len(items) >= count:
                        return items
                    await asyncio.sleep(0.01)
                raise RuntimeError("Offline assignment did not deliver")

            first = await wait_for_deliveries(1)
            assert first[0].conclusion_state.value == "succeeded"
            print(
                "Immediate assignment succeeded with one server-reported invocation and an artifact."
            )
            await agent.close()
            now[0] = datetime(2026, 9, 7, 14, tzinfo=UTC)
            assert len(service.calls) == 2
            service.disconnect = True
            script()
            agent = await Agent.open("offline-assignments", **options)
            second = await wait_for_deliveries(2)
            assert second[0].conclusion_state.value == "failed"
            receipt = (await agent.list_effects(unresolved_only=True))[0]
            inspection = await agent.inspect_routine(routine.routine_id)
            assert inspection is not None and inspection.routine.state.value == "paused"
            calls_before_recovery = len(service.calls)
            resolved = await agent.resolve_effect(
                receipt.receipt_id,
                expected_digest=receipt.receipt_digest,
                decision=EffectResolutionDecision.CLOSE_WITHOUT_RETRY,
                note="Offline investigation: preserve the uncertain observation and close this assignment without replay.",
            )
            assert (
                resolved.receipt_digest == receipt.receipt_digest
                and len(service.calls) == calls_before_recovery
            )
            print(
                "Weekly response loss retained partial research/artifact evidence and paused the assignment."
            )
            print(
                f"Receipt {receipt.receipt_id}: {resolved.outcome.value}; recovery recorded without another call."
            )
            print(
                "No live model, database or server was used. Native upsert acceptance is covered by deterministic tests."
            )
        finally:
            await agent.close()


if __name__ == "__main__":
    asyncio.run(run())
