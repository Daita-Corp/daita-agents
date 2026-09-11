"""Phase F evaluation fixtures; only the injected model may use the network."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import inspect
import subprocess
import sys
from collections import Counter
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from time import perf_counter
from typing import cast
from statistics import mean, median
from uuid import uuid4

import httpx
from _mcp_fixtures import MCPConformanceTransport, MCPFixtureIdentity
from _workspace_support import workspace_for
from live.benchmarks._support import RecordingProvider

from daita import (
    Agent,
    LoopLimits,
    MCPToolSelection,
    create_llm_provider,
    CalendarDaySelector,
    CalendarSchedule,
    ScheduledRoutineDraft,
    MisfirePolicy,
    ReportingMode,
    RequestedCapabilityGrant,
    EffectRequirement,
)
from _distribution_support import no_artifact_outcome_contract
from daita._json import FrozenJsonObject
from dataclasses import replace
from daita._json import canonical_json
from daita.adapters.mcp import StreamableHTTPMCPClientFactory
from daita.capabilities import (
    AccessMode,
    ApprovalDecision,
    ApprovalRequest,
    AutomationEligibility,
    EffectEvidenceBasis,
    EffectOutcome,
    OperationalEffect,
)
from daita.capability_runtime import CapabilityRuntime
from daita.llm.models import ModelProfile, ModelSensitivity
from daita.llm.profiles import reviewed_model_profile
from daita.llm.protocols import ManagedModelProvider
from daita.loop.models import (
    LoopExit,
    LoopExitKind,
    Transcript,
    validate_completed_transcript,
)
from daita.storage.sqlite_codecs.transcripts import encode_loop_exit, encode_message

AUTHORIZATION = "DAITA_RUN_LIVE_PHASE_F_MCP"
MODEL_IDS_ENV = "DAITA_PHASE_F_LIVE_MODEL_IDS"
REPEATS_ENV = "DAITA_PHASE_F_LIVE_REPEATS"
COST_ENV = "DAITA_PHASE_F_LIVE_MAX_COST_USD"
PROFILE_ENV = "DAITA_PHASE_F_LIVE_PROFILE"
REPORT_ENV = "DAITA_PHASE_F_LIVE_REPORT_DIR"
NOW = datetime(2026, 9, 6, 12, tzinfo=UTC)
NEXT_SLOT = datetime(2026, 9, 7, 14, tzinfo=UTC)
EXPIRES = datetime(2026, 9, 14, 12, tzinfo=UTC)
DESTINATION = "release-room"
CONTENT = "Release readiness check F_37C2"
SOURCE = "https://research.fixture.test/release-readiness"
RESEARCH_TOKEN = "F_RESEARCH_82B9"
REPORT_INSTRUCTION = (
    'Return only a JSON object with "invocation_status" ("server_reported", '
    '"uncertain", or "not_dispatched"), "downstream_verified" (boolean), '
    '"citation" (source URL or null), and "finding" (research finding including '
    "its verification marker, or null). "
    "Distinguish invocation evidence from a verified downstream business outcome."
)


def model_ids() -> tuple[str, ...]:
    values = tuple(
        value.strip()
        for value in os.environ.get(MODEL_IDS_ENV, "openai:gpt-5.6-terra").split(",")
    )
    if not 1 <= len(values) <= 3 or not all(values) or len(set(values)) != len(values):
        raise ValueError(f"{MODEL_IDS_ENV} requires one to three distinct model IDs")
    return values


def repeats() -> int:
    value = int(os.environ.get(REPEATS_ENV, "1"))
    if not 1 <= value <= 5:
        raise ValueError(f"{REPEATS_ENV} must be between 1 and 5")
    return value


def evaluation_profile() -> str:
    selected = os.environ.get(PROFILE_ENV, "strict")
    if selected not in {"strict", "user_flow"}:
        raise ValueError(f"{PROFILE_ENV} must be strict or user_flow")
    return selected


def limits() -> LoopLimits:
    user_flow = evaluation_profile() == "user_flow"
    try:
        amount = Decimal(os.environ.get(COST_ENV, "0.50" if user_flow else "0.15"))
    except InvalidOperation as error:
        raise ValueError(f"{COST_ENV} must be a finite positive decimal") from error
    if not amount.is_finite() or amount <= 0:
        raise ValueError(f"{COST_ENV} must be a finite positive decimal")
    return LoopLimits(
        max_steps=24 if user_flow else 14,
        max_total_tokens=100_000 if user_flow else 30_000,
        max_wall_time_seconds=300 if user_flow else 180,
        max_estimated_cost_usd=amount,
    )


def live_provider(model_id: str) -> tuple[ModelProfile, ManagedModelProvider]:
    # Check again at construction so importing or directly invoking a test cannot
    # accidentally spend credentials merely because they are present.
    if os.environ.get(AUTHORIZATION) != "1":
        raise ValueError(
            f"{AUTHORIZATION}=1 is required before constructing a provider"
        )
    limits()
    profile = reviewed_model_profile(model_id)
    provider = model_id.partition(":")[0]
    if (
        provider not in {"openai", "anthropic", "gemini", "grok"}
        or profile is None
        or not profile.supports_tools
    ):
        raise ValueError(
            "Phase F evaluations require a reviewed API tool-capable model"
        )
    key = os.environ.get(
        f"DAITA_PHASE_F_LIVE_{provider.upper()}_API_KEY"
    ) or os.environ.get("DAITA_PHASE_F_LIVE_LLM_API_KEY")
    if not key or not key.strip():
        raise ValueError("Set the Phase F benchmark API key for the selected provider")
    return profile, create_llm_provider(
        model_id, api_key=key, max_output_tokens=min(profile.max_output_tokens, 2_048)
    )


class EvaluationTransport(MCPConformanceTransport):
    def __init__(self, identity: MCPFixtureIdentity, fault: str) -> None:
        super().__init__(identity)
        self.fault = fault
        self.timings: list[dict[str, object]] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        started = perf_counter()
        payload = json.loads(request.content)
        try:
            result = await super().__call__(request)
            if (
                payload.get("method") == "tools/call"
                and payload["params"]["name"] == "notify_release"
                and self.fault == "disconnect"
            ):
                raise httpx.ReadError(
                    "fixture lost response after dispatch", request=request
                )
            return result
        finally:
            self.timings.append(
                {"method": payload.get("method"), "seconds": perf_counter() - started}
            )


def identity(tool_count: int, fault: str) -> MCPFixtureIdentity:
    if not 2 <= tool_count <= 128:
        raise ValueError("Phase F fixture admits between 2 and 128 tools")
    if fault not in {"success", "tool_error", "disconnect"}:
        raise ValueError("unknown Phase F fixture fault")
    tools: list[dict[str, object]] = [
        {
            "name": "notify_release",
            "description": "Post a release-readiness notification to a destination.",
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
        {
            "name": "research_release",
            "description": "Read the current release-readiness research and citation.",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    ]
    tools.extend(
        {
            "name": f"release_archive_{index:03}",
            "description": f"Read archived release-readiness notification {index}; historical only, cannot post notifications.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        }
        for index in range(tool_count - 2)
    )
    results: dict[str, dict[str, object]] = {
        str(tool["name"]): {
            "content": [{"type": "text", "text": "Archived release, not current."}]
        }
        for tool in tools
    }
    results["research_release"] = {
        "content": [
            {
                "type": "text",
                "text": f"Current release readiness: 7 of 9 checks passed; marker {RESEARCH_TOKEN}. Source: {SOURCE}. This report does not verify notification delivery.",
            }
        ]
    }
    results["notify_release"] = {
        "content": [
            {
                "type": "text",
                "text": (
                    "Partial application error."
                    if fault == "tool_error"
                    else "Invocation completed."
                ),
            }
        ],
        "isError": fault == "tool_error",
    }
    return MCPFixtureIdentity(
        "phase-f.fixture.test",
        "release-notifications",
        "1",
        "2025-11-25",
        tools,
        results,
    )


class Evaluation:
    def __init__(
        self,
        root: Path,
        provider: ManagedModelProvider,
        profile: ModelProfile,
        *,
        tool_count: int = 2,
        fault: str = "success",
        routine: bool = False,
        run_immediately: bool = True,
        action_argument_names: tuple[str, str] = ("destination", "content"),
    ) -> None:
        self.root = root
        expected_runtime_root = os.environ.get("DAITA_EXPECTED_RUNTIME_ROOT")
        if expected_runtime_root and not Path(
            inspect.getfile(CapabilityRuntime)
        ).resolve().is_relative_to(Path(expected_runtime_root).resolve()):
            raise ValueError(
                "qualification imported a different production source root"
            )
        self.provider = RecordingProvider(provider)
        self.profile = profile
        self.evaluation_profile = evaluation_profile()
        if (
            len(action_argument_names) != 2
            or len(set(action_argument_names)) != 2
            or any(
                not name.isidentifier() or len(name) > 64
                for name in action_argument_names
            )
        ):
            raise ValueError("fixture action needs two distinct bounded argument names")
        if self.evaluation_profile != "user_flow" and (
            not run_immediately or action_argument_names != ("destination", "content")
        ):
            raise ValueError("contract comparisons require explicit user_flow profile")
        self.run_immediately = run_immediately
        self.action_argument_names = action_argument_names
        self.limits = limits()
        self.answer_reviews: list[dict[str, object]] = []
        assert self.limits.max_estimated_cost_usd is not None
        self.cost_limit = self.limits.max_estimated_cost_usd
        self.conversation_id: str | None = None
        self.clock = NOW
        self.routine = routine
        self.setup_mode = "model_authored"
        self.server = identity(tool_count, fault)
        if action_argument_names != ("destination", "content"):
            schema = cast(
                dict[str, object], self.server.tool("notify_release")["inputSchema"]
            )
            schema["properties"] = {
                name: {"type": "string"} for name in action_argument_names
            }
            schema["required"] = list(action_argument_names)
        self.transport = EvaluationTransport(self.server, fault)
        self.approvals: list[dict[str, object]] = []
        self.captures: list[tuple[LoopExit, Transcript, float | None]] = []
        self.started = perf_counter()

    async def start(self) -> None:
        factory = StreamableHTTPMCPClientFactory(
            http_transport=httpx.MockTransport(self.transport)
        )
        self.agent = await Agent.create(
            "phase-f-evaluation",
            root=self.root,
            workspace=workspace_for(self.root),
            model=self.provider,
            model_profile=self.profile,
            limits=self.limits,
            mcp_client_factory=factory,
            approval_handler=self.approve,
            clock=lambda: self.clock,
        )
        selections = tuple(
            MCPToolSelection(
                str(tool["name"]),
                str(tool["name"]),
                str(tool["description"]),
                access_mode=(
                    AccessMode.NONE
                    if tool["name"] == "notify_release"
                    else AccessMode.READ
                ),
                operational_effect=(
                    OperationalEffect.EXTERNAL_ACTION
                    if tool["name"] == "notify_release"
                    else OperationalEffect.NONE
                ),
                automation_eligibility=AutomationEligibility.AUTOMATION_DIRECT,
                maximum_outbound_sensitivity=ModelSensitivity.INTERNAL,
            )
            for tool in self.server.tools
        )
        status = await self.agent.attach_mcp_server(
            endpoint=self.server.endpoint,
            selections=selections,
            maximum_outbound_sensitivity=ModelSensitivity.INTERNAL,
        )
        self.binding = status.binding
        self.action = next(
            tool for tool in self.binding.tools if tool.remote_name == "notify_release"
        )
        self.research = next(
            tool
            for tool in self.binding.tools
            if tool.remote_name == "research_release"
        )
        # MCP declarations are reconstructed at the existing composition boundary.
        await self.agent.close()
        self.agent = await Agent.open(
            "phase-f-evaluation",
            root=self.root,
            workspace=workspace_for(self.root),
            model=self.provider,
            model_profile=self.profile,
            limits=self.limits,
            mcp_client_factory=factory,
            approval_handler=self.approve,
            clock=lambda: self.clock,
        )
        self.server.request_methods.clear()
        self.transport.timings.clear()

    def grant_is_exact(self, grant: Mapping[str, object]) -> bool:
        return (
            grant.get("capability_id") == self.action.capability_id
            and grant.get("max_calls_per_occurrence") == 1
            and canonical_json(grant.get("constraints"))
            == canonical_json(
                {
                    "binding_id": self.binding.binding_id,
                    "binding_revision": self.binding.revision,
                    "remote_tool_name": self.action.remote_name,
                    "fixed_arguments": {self.action_argument_names[0]: DESTINATION},
                    "variable_argument_names": [self.action_argument_names[1]],
                }
            )
        )

    async def approve(self, request: ApprovalRequest) -> ApprovalDecision:
        accepted = False
        arguments = request.arguments
        already_approved = any(item["approved"] for item in self.approvals)
        if (
            not already_approved
            and request.capability_id == self.action.capability_id
            and not self.routine
        ):
            values = arguments.get("arguments")
            accepted = (
                isinstance(values, Mapping)
                and values.get("destination") == DESTINATION
                and set(values) == {"destination", "content"}
            )
        if (
            not already_approved
            and request.tool_name == "routine_create"
            and self.routine
        ):
            proposal = arguments.get("proposal")
            if not isinstance(proposal, Mapping):
                return ApprovalDecision.DENY
            arguments = proposal
            grants = arguments.get("capability_grants")
            schedule = arguments.get("schedule")
            capabilities = arguments.get("allowed_capability_ids")
            accepted = (
                isinstance(grants, tuple)
                and len(grants) == 1
                and isinstance(grants[0], Mapping)
                and self.grant_is_exact(grants[0])
                and isinstance(schedule, Mapping)
                and schedule.get("kind") == "calendar"
                and schedule.get("timezone") == "America/Chicago"
                and schedule.get("hour") == 9
                and schedule.get("minute") == 0
                and schedule.get("weekdays") == (1,)
                and arguments.get("run_immediately") is self.run_immediately
                and arguments.get("allowed_connector_binding_ids")
                == (self.binding.binding_id,)
                and isinstance(capabilities, tuple)
                and set(capabilities)
                == {self.action.capability_id, self.research.capability_id}
                and arguments.get("cumulative_max_attempts") == 2
                and arguments.get("cumulative_max_occurrences") == 2
                and arguments.get("per_run_max_tokens") == self.limits.max_total_tokens
                and arguments.get("cumulative_max_tokens")
                == 2 * self.limits.max_total_tokens
                and Decimal(str(arguments.get("per_run_max_cost_usd")))
                == self.limits.max_estimated_cost_usd
                and Decimal(str(arguments.get("cumulative_max_cost_usd")))
                == 2 * self.cost_limit
                and arguments.get("expires_at") == EXPIRES.isoformat()
            )
        self.approvals.append(
            {
                "tool": request.tool_name,
                "arguments": request.arguments.to_dict(),
                "approved": accepted,
            }
        )
        return ApprovalDecision.APPROVE if accepted else ApprovalDecision.DENY

    async def run(self, prompt: str) -> tuple[LoopExit, Transcript]:
        started = perf_counter()
        result = await self.agent.run(prompt, conversation_id=self.conversation_id)
        self.conversation_id = result.conversation_id
        transcript = await self.agent.transcript(result.run_id)
        self.captures.append((result, transcript, perf_counter() - started))
        return result, transcript

    async def scheduled_result(self, count: int) -> tuple[LoopExit, Transcript]:
        async with asyncio.timeout(self.limits.max_wall_time_seconds + 30):
            while True:
                assert self.conversation_id is not None
                items = await self.agent.inbox(conversation_id=self.conversation_id)
                if len(items) >= count:
                    routines = await self.agent.list_routines()
                    assert len(routines) == 1
                    inspection = await self.agent.inspect_routine(
                        routines[0].routine_id
                    )
                    assert inspection is not None
                    captured = {result.run_id for result, _, _ in self.captures}
                    for occurrence in inspection.recent_occurrences:
                        if (
                            occurrence.terminal_run_id
                            and occurrence.terminal_run_id not in captured
                        ):
                            result = await self.agent._embedded._store.result(
                                occurrence.terminal_run_id
                            )
                            assert result is not None
                            transcript = await self.agent.transcript(result.run_id)
                            self.captures.append((result, transcript, None))
                            return result, transcript
                await asyncio.sleep(0.05)

    def action_prompt(self, phrasing: int) -> str:
        prompts = (
            f'Post a release-readiness notification to {DESTINATION} with exactly this content: "{CONTENT}".',
            f'Let {DESTINATION} know this release-readiness message: "{CONTENT}". Preserve that text exactly.',
        )
        if self.evaluation_profile == "user_flow":
            return (
                prompts[phrasing] + " Do not perform research. Tell me what happened."
            )
        return (
            prompts[phrasing]
            + " Do not perform research. Set citation and finding to null. "
            + REPORT_INSTRUCTION
        )

    def research_prompt(self) -> str:
        prompt = (
            f"Research current release readiness and post the finding, its verification marker, "
            f"and its citation to {DESTINATION}. Preserve the research in your answer even if "
            "the notification cannot be confirmed. "
        )
        return prompt + (
            "Tell me what happened and what remains unconfirmed."
            if self.evaluation_profile == "user_flow"
            else REPORT_INSTRUCTION
        )

    def check_answer(self, result: LoopExit, *, status: str, research: bool) -> None:
        assert result.kind is LoopExitKind.COMPLETED, result
        if self.evaluation_profile == "strict":
            assert_report(result, status=status, research=research)
        else:
            assert result.final_text and result.final_text.strip()
            self.answer_reviews.append(
                {
                    "run_id": result.run_id,
                    "expected_status": status,
                    "research": research,
                    "review": "pending_evidence_review",
                }
            )

    def routine_prompt(self) -> str:
        if self.evaluation_profile == "user_flow":
            timing = (
                "now and every Monday at 09:00 America/Chicago. "
                if self.run_immediately
                else "every Monday at 09:00 America/Chicago, starting next Monday; do not run it today. "
            )
            return (
                "Set up a saved assignment using our connected release research and notifications: "
                f"research current release readiness and post the finding, verification marker, and citation to {DESTINATION} "
                + timing
                + "Allow one notification each time, "
                "always to that room; only the message content may change. Use the current model. "
                "Keep the information internal and put each run's report in this conversation's inbox. "
                "Use no database tables, files, saved procedures, or artifacts. "
                "Always report what happened, including unconfirmed results. Require confirmation "
                "from the notification service, without claiming downstream delivery was verified. "
                "If a scheduled time is missed, run only the latest missed occurrence. "
                "Stop after two occurrences or two attempts, or one consecutive failure, "
                f"and expire at {EXPIRES.isoformat()}. "
                f"Use exactly {self.limits.max_total_tokens} tokens and ${self.cost_limit} per run, "
                f"{2 * self.limits.max_total_tokens} tokens and ${2 * self.cost_limit} total. "
                "Tell me when the assignment is saved. Do not send an additional notification while setting it up."
            )
        contract = {
            "binding_id": self.binding.binding_id,
            "binding_revision": self.binding.revision,
            "notification_capability_id": self.action.capability_id,
            "notification_remote_tool_name": self.action.remote_name,
            "research_capability_id": self.research.capability_id,
            "model_route": self.provider.provider_id,
        }
        return (
            "Create one routine that researches current release readiness and posts the finding, "
            f"verification marker, and citation to {DESTINATION}, immediately and every Monday at 09:00 America/Chicago. "
            "Grant exactly one notification per occurrence, fixing the destination and allowing only "
            "the scalar content argument to vary. Require server-reported notification evidence. "
            "Use only the admitted research and notification capabilities; no sources, resources, "
            "skills, or artifacts. Discover this conversation's inbox destination. Always report, "
            "use latest_only misfire handling, and stop after two occurrences/two attempts or "
            f"{EXPIRES.isoformat()}, with one consecutive failure allowed. "
            f"Authorize {self.limits.max_total_tokens} tokens and ${self.limits.max_estimated_cost_usd} "
            f"per run, {2 * self.limits.max_total_tokens} tokens and ${2 * self.cost_limit} "
            "cumulatively. Keep sensitivity internal, require a terminal conclusion and current-run "
            "provenance with exact bindings. Embed this reporting instruction in the saved routine: "
            f"{REPORT_INSTRUCTION} Do not send a separate foreground notification. "
            "These current admission references are provided by the fixture owner; construct "
            f"the routine and grant yourself: {canonical_json(contract)}"
        )

    async def report(self, status: str, error: str | None) -> dict[str, object]:
        receipts = await self.agent.list_effects() if hasattr(self, "agent") else ()
        routine_states = []
        if hasattr(self, "agent") and self.routine:
            from daita.routines.capabilities import routine_inspection_projection

            for routine in await self.agent.list_routines():
                inspection = await self.agent.inspect_routine(routine.routine_id)
                assert inspection is not None
                routine_states.append(
                    json.loads(
                        canonical_json(routine_inspection_projection(inspection))
                    )
                )
        runs = [
            {
                "result": json.loads(encode_loop_exit(result)),
                "messages": [
                    json.loads(encode_message(message))
                    for message in transcript.messages
                ],
                "foreground_elapsed_seconds": elapsed,
            }
            for result, transcript, elapsed in self.captures
        ]
        calls = [
            call
            for _, transcript, _ in self.captures
            for message in transcript.messages
            for call in message.tool_calls
        ]
        from daita.llm.models import ToolResultBlock

        errors = [
            block
            for _, transcript, _ in self.captures
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock) and block.is_error
        ]
        usages = self.provider.usages
        estimates = [usage.cost_estimate for usage in usages]
        usage_complete = len(self.provider.timings) == len(
            self.provider.requests
        ) and all(item["usage_complete"] for item in self.provider.timings)
        complete = (
            usage_complete
            and bool(estimates)
            and all(item.status.value == "complete" for item in estimates)
        )
        return {
            "status": status,
            "error": error,
            "model_id": self.profile.id,
            "tool_count": len(self.server.tools),
            "fault": self.transport.fault,
            "setup_mode": self.setup_mode,
            "evaluation_profile": self.evaluation_profile,
            "action_argument_names": list(self.action_argument_names),
            "run_immediately": self.run_immediately,
            "production_runtime_file": inspect.getfile(CapabilityRuntime),
            "production_runtime_sha256": hashlib.sha256(
                Path(inspect.getfile(CapabilityRuntime)).read_bytes()
            ).hexdigest(),
            "answer_reviews": self.answer_reviews,
            "routine_states": routine_states,
            "metrics": {
                "model_requests": len(self.provider.requests),
                "model_responses": len(self.provider.responses),
                "input_tokens": sum(item.input_tokens for item in usages),
                "output_tokens": sum(item.output_tokens for item in usages),
                "reasoning_tokens": sum(item.reasoning_tokens for item in usages),
                "cache_read_tokens": sum(item.cache_read_tokens for item in usages),
                "cache_write_tokens": sum(item.cache_write_tokens for item in usages),
                "estimated_cost_usd": (
                    str(
                        sum(
                            (item.amount_usd or Decimal(0) for item in estimates),
                            Decimal(0),
                        )
                    )
                    if complete
                    else None
                ),
                "known_estimated_cost_usd": (
                    str(
                        sum(
                            (
                                item.amount_usd
                                for item in estimates
                                if item.amount_usd is not None
                            ),
                            Decimal(0),
                        )
                    )
                    if any(item.amount_usd is not None for item in estimates)
                    else None
                ),
                "cost_complete": complete,
                "usage_complete": usage_complete,
                "steps": sum(item[0].steps for item in self.captures),
                "tool_errors": len(errors),
                "tool_calls_by_name": dict(Counter(call.name for call in calls)),
                "action_attempts": (
                    sum(call.name == self.action.local_name for call in calls)
                    if hasattr(self, "action")
                    else 0
                ),
                "action_dispatches": sum(
                    name == "notify_release" for name, _ in self.server.calls
                ),
                "mcp_methods": dict(Counter(self.server.request_methods)),
                "elapsed_seconds": perf_counter() - self.started,
                "model_seconds": (
                    sum(cast(float, item["seconds"]) for item in self.provider.timings)
                    if len(self.provider.timings) == len(self.provider.requests)
                    else None
                ),
                "model_timing_complete": len(self.provider.timings)
                == len(self.provider.requests),
                "mcp_seconds": sum(
                    cast(float, item["seconds"]) for item in self.transport.timings
                ),
            },
            "runs": runs,
            "approvals": self.approvals,
            "server_calls": self.server.calls,
            "receipts": [
                {
                    "outcome": item.outcome.value if item.outcome else None,
                    "evidence_basis": (
                        item.evidence_basis.value if item.evidence_basis else None
                    ),
                    "payload": (
                        item.payload.to_dict() if item.payload is not None else None
                    ),
                }
                for item in receipts
            ],
            "model_requests": [
                {
                    "tool_names": [tool.name for tool in request.tools],
                    "tool_definitions": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": json.loads(
                                canonical_json(tool.input_schema)
                            ),
                        }
                        for tool in request.tools
                    ],
                    "tool_definitions_sha256": hashlib.sha256(
                        canonical_json(
                            [
                                {
                                    "name": tool.name,
                                    "description": tool.description,
                                    "input_schema": tool.input_schema,
                                }
                                for tool in request.tools
                            ]
                        ).encode()
                    ).hexdigest(),
                    "sensitivity": request.sensitivity.value,
                    "remaining_tokens": request.max_total_tokens,
                    "remaining_estimated_cost_usd": (
                        str(request.max_estimated_cost_usd)
                        if request.max_estimated_cost_usd is not None
                        else None
                    ),
                    "messages": [
                        json.loads(encode_message(message))
                        for message in request.messages
                    ],
                }
                for request in self.provider.requests
            ],
            "model_timings": self.provider.timings,
            "model_responses": [
                {
                    "text": response.text,
                    "finish_reason": response.finish_reason.value,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "name": call.name,
                            "arguments": json.loads(canonical_json(call.arguments)),
                        }
                        for call in response.tool_calls
                    ],
                }
                for response in self.provider.responses
            ],
            "mcp_timings": self.transport.timings,
        }


def owner_routine_draft(
    scenario: Evaluation, origin_run_id: str, destination_id: str
) -> ScheduledRoutineDraft:
    """Exact typed fixture authorization, independent of model proposal authoring."""
    amount = scenario.cost_limit
    return ScheduledRoutineDraft(
        origin_run_id=origin_run_id,
        title="Release readiness",
        authorized_instruction="Research current release readiness and post the cited finding and marker to release-room. Report invocation evidence honestly.",
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
        allowed_connector_binding_ids=(scenario.binding.binding_id,),
        allowed_capability_ids=(
            scenario.action.capability_id,
            scenario.research.capability_id,
        ),
        sensitivity_ceiling=scenario.binding.maximum_outbound_sensitivity,
        outcome_contract=replace(
            no_artifact_outcome_contract(),
            effect_requirements=(
                EffectRequirement(
                    scenario.action.capability_id,
                    1,
                    frozenset({EffectEvidenceBasis.SERVER_REPORTED}),
                ),
            ),
        ),
        distribution_destination_id=destination_id,
        eligible_model_routes=(scenario.provider.provider_id,),
        per_run_max_tokens=scenario.limits.max_total_tokens,
        per_run_max_cost_usd=amount,
        cumulative_max_tokens=2 * scenario.limits.max_total_tokens,
        cumulative_max_cost_usd=2 * amount,
        cumulative_max_attempts=2,
        cumulative_max_occurrences=2,
        maximum_consecutive_failures=1,
        expires_at=EXPIRES,
        run_immediately=scenario.run_immediately,
        requested_capability_grants=(
            RequestedCapabilityGrant(
                scenario.action.capability_id,
                FrozenJsonObject.from_mapping(
                    {
                        "binding_id": scenario.binding.binding_id,
                        "binding_revision": scenario.binding.revision,
                        "remote_tool_name": scenario.action.remote_name,
                        "fixed_arguments": {
                            scenario.action_argument_names[0]: DESTINATION
                        },
                        "variable_argument_names": [scenario.action_argument_names[1]],
                    }
                ),
                1,
            ),
        ),
    )


@asynccontextmanager
async def evaluate(
    root: Path,
    provider: ManagedModelProvider,
    profile: ModelProfile,
    report_path: Path,
    *,
    case_id: str = "offline-harness",
    **options,
) -> AsyncIterator[Evaluation]:
    evaluation = Evaluation(root, provider, profile, **options)
    status, error = "passed", None
    try:
        await evaluation.start()
        yield evaluation
    except BaseException as failure:
        status, error = "failed", f"{type(failure).__name__}: {failure}"
        raise
    finally:
        # Stop scheduled work before reading evidence and closing the borrowed provider.
        try:
            if hasattr(evaluation, "agent"):
                await evaluation.agent._embedded._routine_supervisor.close()
                captured = {result.run_id for result, _, _ in evaluation.captures}
                for routine in await evaluation.agent.list_routines():
                    inspection = await evaluation.agent.inspect_routine(
                        routine.routine_id
                    )
                    assert inspection is not None
                    for occurrence in inspection.recent_occurrences:
                        run_id = occurrence.reserved_run_id
                        if run_id and run_id not in captured:
                            result = await evaluation.agent._embedded._store.result(
                                run_id
                            )
                            if result is not None:
                                evaluation.captures.append(
                                    (
                                        result,
                                        await evaluation.agent.transcript(run_id),
                                        None,
                                    )
                                )
            report = await evaluation.report(status, error)
            report.update(
                {
                    "case_id": case_id,
                    "recorded_at": datetime.now(UTC).isoformat(),
                    "revision": subprocess.run(
                        ["git", "rev-parse", "HEAD"],
                        capture_output=True,
                        text=True,
                        check=False,
                    ).stdout.strip()
                    or None,
                    "working_tree_dirty": bool(
                        subprocess.run(
                            ["git", "status", "--porcelain"],
                            capture_output=True,
                            text=True,
                            check=False,
                        ).stdout.strip()
                    ),
                    "harness_sha256": hashlib.sha256(
                        b"".join(
                            (Path(__file__).parent / name).read_bytes()
                            for name in (
                                "_phase_f_live_support.py",
                                "_mcp_fixtures.py",
                                "live/benchmarks/_support.py",
                                "live/test_phase_f_mcp_live.py",
                                "live/test_phase_f_scheduled_live.py",
                            )
                        )
                    ).hexdigest(),
                    "working_tree_diff_sha256": hashlib.sha256(
                        subprocess.run(
                            ["git", "diff", "HEAD"], capture_output=True, check=False
                        ).stdout
                    ).hexdigest(),
                    "limits": {
                        "max_steps": evaluation.limits.max_steps,
                        "max_tokens": evaluation.limits.max_total_tokens,
                        "max_seconds": evaluation.limits.max_wall_time_seconds,
                        "max_estimated_cost_usd": str(
                            evaluation.limits.max_estimated_cost_usd
                        ),
                    },
                }
            )
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
        finally:
            try:
                if hasattr(evaluation, "agent"):
                    await evaluation.agent.close()
            finally:
                await evaluation.provider.close()


def report_path(node_id: str) -> Path:
    root = Path(os.environ.get(REPORT_ENV, "test-results/phase-f"))
    if evaluation_profile() == "user_flow":
        root = root / "user_flow"
    key = hashlib.sha256(node_id.encode()).hexdigest()[:16]
    return root / f"phase-f-{key}-{uuid4().hex[:8]}.json"


def summarize_reports(directory: Path) -> list[dict[str, object]]:
    """Include failures in each denominator; never present unknown usage as zero."""
    groups: dict[tuple, list[dict]] = {}
    for path in sorted(directory.glob("phase-f-*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        family = report["case_id"].partition("[")[0].split("::")[-1]
        key = (
            report["model_id"],
            report.get("evaluation_profile", "strict"),
            family,
            report["tool_count"],
            report["fault"],
            report["revision"],
            report["harness_sha256"],
            report["working_tree_diff_sha256"],
            tuple(report.get("action_argument_names", ("destination", "content"))),
            report.get("run_immediately", True),
            report.get("production_runtime_sha256"),
        )
        groups.setdefault(key, []).append(report)
    output: list[dict[str, object]] = []
    for (
        model,
        selected_profile,
        family,
        tools,
        fault,
        revision,
        harness,
        diff,
        argument_names,
        run_immediately,
        runtime_sha256,
    ), reports in groups.items():
        metrics = [report["metrics"] for report in reports]
        passed = sum(report["status"] == "passed" for report in reports)
        costs = [
            Decimal(item["known_estimated_cost_usd"])
            for item in metrics
            if item["known_estimated_cost_usd"] is not None
        ]
        output.append(
            {
                "model_id": model,
                "evaluation_profile": selected_profile,
                "scenario": family,
                "tool_count": tools,
                "fault": fault,
                "revision": revision,
                "harness_sha256": harness,
                "working_tree_diff_sha256": diff,
                "action_argument_names": argument_names,
                "run_immediately": run_immediately,
                "production_runtime_sha256": runtime_sha256,
                "cases": len(reports),
                "passed": passed,
                "failed": len(reports) - passed,
                "success_rate": passed / len(reports),
                "mean_model_requests": mean(item["model_requests"] for item in metrics),
                "mean_observed_tokens": mean(
                    item["input_tokens"] + item["output_tokens"] for item in metrics
                ),
                "median_elapsed_seconds": median(
                    item["elapsed_seconds"] for item in metrics
                ),
                "mean_model_seconds": (
                    mean(
                        item["model_seconds"]
                        for item in metrics
                        if item["model_seconds"] is not None
                    )
                    if any(item["model_seconds"] is not None for item in metrics)
                    else None
                ),
                "cases_with_incomplete_model_timing": sum(
                    not item["model_timing_complete"] for item in metrics
                ),
                "mean_mcp_inspections": mean(
                    item["mcp_methods"].get("tools/list", 0) for item in metrics
                ),
                "action_attempts": sum(item["action_attempts"] for item in metrics),
                "action_dispatches": sum(item["action_dispatches"] for item in metrics),
                "cases_with_incomplete_usage": sum(
                    not item["usage_complete"] for item in metrics
                ),
                "cases_with_incomplete_cost": sum(
                    not item["cost_complete"] for item in metrics
                ),
                "known_estimated_cost_usd": (
                    str(sum(costs, Decimal(0))) if costs else None
                ),
            }
        )
    return output


def assert_completed(result: LoopExit, transcript: Transcript) -> None:
    assert result.kind is LoopExitKind.COMPLETED, result
    validate_completed_transcript(transcript, result)


def assert_report(result: LoopExit, *, status: str, research: bool) -> None:
    assert result.final_text is not None
    text = result.final_text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    report = json.loads(text)
    assert report["invocation_status"] == status, report
    assert report["downstream_verified"] is False, report
    if research:
        assert report["citation"] == SOURCE, report
        assert RESEARCH_TOKEN in report["finding"], report
        assert "7" in report["finding"] and "9" in report["finding"], report


async def assert_action(
    evaluation: Evaluation, *, count: int = 1, uncertain: bool = False
) -> None:
    calls = [
        arguments
        for name, arguments in evaluation.server.calls
        if name == "notify_release"
    ]
    assert len(calls) == count, calls
    assert all(arguments["destination"] == DESTINATION for arguments in calls)
    receipts = await evaluation.agent.list_effects()
    assert len(receipts) == count
    assert all(
        receipt.outcome
        is (EffectOutcome.UNCERTAIN if uncertain else EffectOutcome.SUCCEEDED)
        for receipt in receipts
    )
    if not uncertain:
        assert all(
            receipt.evidence_basis is EffectEvidenceBasis.SERVER_REPORTED
            for receipt in receipts
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python tests/_phase_f_live_support.py REPORT_DIRECTORY"
        )
    summary = summarize_reports(Path(sys.argv[1]))
    if not summary:
        raise SystemExit("No Phase F evaluation reports found")
    print(json.dumps(summary, indent=2))
