"""Bounded real-model evaluation over the existing PostgreSQL canary fixture.

No model responses, tool arguments, previews or routine proposals are scripted.
The existing recording provider observes physical attempts inside the owned router.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from time import perf_counter

from daita import Agent, AgentConfig, ApprovalDecision, LoopLimits
from daita._json import canonical_json
from daita.llm import factory
from daita.llm.models import ModelRequest, ModelSensitivity, ToolResultBlock
from daita.llm.profiles import reviewed_model_profile
from daita.llm.routing import ModelRoute, ModelRouteCandidate
from daita.loop.models import LoopExitKind, validate_completed_transcript
from daita.routines.capabilities import ROUTINE_CREATE_CAPABILITY_ID
from daita.security import (
    CompositeSecretProvider,
    EnvironmentSecretProvider,
    SecretReference,
)
from daita.storage.sqlite_codecs.transcripts import encode_loop_exit, encode_message
from tests.support.job_benchmarks import RecordingProvider
from tests.support.paths import REPO_ROOT
from tests.support.postgresql_live import _Secrets
from tests.support.workspace import workspace_for

AUTHORIZATION = "DAITA_RUN_POSTGRES_LIVE_LLM"
KEY_ENV = "DAITA_POSTGRES_LIVE_LLM_API_KEY"
MODEL_ENV = "DAITA_POSTGRES_LIVE_MODEL_IDS"
REPEAT_ENV = "DAITA_POSTGRES_LIVE_REPEATS"
REPORT_ENV = "DAITA_POSTGRES_LIVE_REPORT_DIR"
PROFILE_ENV = "DAITA_POSTGRES_LIVE_PROFILE"
NOW = datetime(2026, 9, 13, 12, tzinfo=UTC)
NEXT_SLOT = datetime(2026, 9, 14, 14, tzinfo=UTC)
EXPIRES = datetime(2026, 9, 20, 12, tzinfo=UTC)
FIXTURE_SENSITIVITY = ModelSensitivity.INTERNAL
LIMITS = LoopLimits(
    max_steps=14,
    max_total_tokens=30_000,
    max_wall_time_seconds=180,
    max_estimated_cost_usd=Decimal("0.15"),
)
USER_FLOW_LIMITS = replace(LoopLimits(), max_estimated_cost_usd=Decimal("0.50"))
REPORT_INSTRUCTION = (
    'Return only JSON with "status" ("succeeded", "not_applied", or "uncertain"), '
    '"inserted_count", "updated_count", "unchanged_count" (integer or null), '
    'and "explanation" (a short explanation grounded in the returned evidence). '
    "Use null for counts that the effect evidence does not establish. "
    "If a write is denied, fails or is uncertain, stop and report it without "
    "retrying, changing the intended values, or performing a replacement write."
)


def model_ids():
    values = tuple(
        v.strip() for v in os.environ.get(MODEL_ENV, "openai:gpt-5.6-terra").split(",")
    )
    if not 1 <= len(values) <= 3 or not all(values) or len(set(values)) != len(values):
        raise ValueError(
            f"{MODEL_ENV} requires one to three distinct reviewed model IDs"
        )
    return values


def repeats():
    count = int(os.environ.get(REPEAT_ENV, "1"))
    if not 1 <= count <= 5:
        raise ValueError(f"{REPEAT_ENV} must be between 1 and 5")
    return count


def evaluation_profile():
    profile = os.environ.get(PROFILE_ENV, "strict")
    if profile not in {"strict", "user_flow"}:
        raise ValueError(f"{PROFILE_ENV} must be strict or user_flow")
    return profile


def live_config(model_id, *, limits=LIMITS):
    # Recheck at construction: a key or direct fixture invocation grants nothing.
    if os.environ.get(AUTHORIZATION) != "1":
        raise ValueError(f"{AUTHORIZATION}=1 requires explicit paid-run authorization")
    if limits.max_estimated_cost_usd is None or limits.max_estimated_cost_usd <= 0:
        raise ValueError(
            "Live diagnostics require an explicit positive finite cost cap"
        )
    profile = reviewed_model_profile(model_id)
    provider = model_id.partition(":")[0]
    if (
        provider not in {"openai", "anthropic", "gemini"}
        or profile is None
        or not profile.supports_tools
    ):
        raise ValueError(
            "A reviewed API model with tools and input counting is required"
        )
    key_name = f"DAITA_POSTGRES_LIVE_{provider.upper()}_API_KEY"
    if not os.environ.get(key_name, "").strip():
        key_name = KEY_ENV
    if not os.environ.get(key_name, "").strip():
        raise ValueError(
            f"Set {key_name} in the environment; never put credentials in evidence"
        )
    return AgentConfig(
        model_route=ModelRoute(
            candidates=(
                ModelRouteCandidate(
                    model_id,
                    replace(
                        profile, max_output_tokens=min(2048, profile.max_output_tokens)
                    ),
                    secret_reference=SecretReference("env", key_name),
                    allowed_sensitivities=frozenset(ModelSensitivity),
                ),
            )
        ),
        limits=limits,
    )


def report_path(case_id):
    folder = Path(os.environ.get(REPORT_ENV, "test-results/postgresql-live-llm"))
    return folder / (hashlib.sha256(case_id.encode()).hexdigest()[:20] + ".json")


def calls_for(transcript, tool_name):
    return [
        call
        for message in transcript.messages
        for call in message.tool_calls
        if call.name == tool_name
    ]


def assert_report(result, status, counts=None):
    assert result.final_text, (
        getattr(result, "reason", "missing_final_answer"),
        "A terminal run without an answer cannot satisfy reporting acceptance.",
    )
    report = json.loads(result.final_text or "")
    assert report["status"] == status, report
    assert isinstance(report["explanation"], str) and report["explanation"].strip()
    actual = tuple(
        report[key] for key in ("inserted_count", "updated_count", "unchanged_count")
    )
    assert actual == (counts if counts is not None else (None, None, None)), report
    assert all(value is None or type(value) is int for value in actual)


class Evaluation:
    def __init__(
        self,
        database,
        config,
        recordings,
        max_runs,
        *,
        setup_mode="model_authored",
        profile="strict",
    ):
        assert 1 <= max_runs <= 3
        assert setup_mode in {
            "model_authored",
            "owner_admitted_routine",
            "injected_commit_loss_fixture",
        }
        self.db, self.config, self.recordings = database, config, recordings
        self.max_runs = max_runs
        self.setup_mode = setup_mode
        assert profile in {"strict", "user_flow"}
        self.profile = profile
        self.answer_reviews = []
        self.setup_request_count = (
            len(database.model.requests)
            if setup_mode == "injected_commit_loss_fixture"
            else 0
        )
        self.stages = []
        self.clock = NOW
        self.captures = []
        self.approvals = []
        self.expected_write = None
        self.deny_write = False
        self.approval_hook: Callable[[], Awaitable[object]] | None = None
        self.routine_contract: Mapping[str, object] | None = None
        self.routine_grant: object | None = None
        self.recovery_digest = None
        self.conversation_id = None

    @property
    def agent(self):
        return self.db.agent

    @property
    def routine_budgets(self):
        limits = self.config.limits
        return {
            "per_run_max_tokens": limits.max_total_tokens,
            "per_run_max_cost_usd": str(limits.max_estimated_cost_usd),
            "cumulative_max_tokens": 2 * limits.max_total_tokens,
            "cumulative_max_cost_usd": str(2 * limits.max_estimated_cost_usd),
            "cumulative_max_attempts": 2,
            "cumulative_max_occurrences": 2,
        }

    async def reopen(self):
        await self.agent.close()
        self.db.agent = await Agent.open(
            self.agent.name,
            root=self.db.root,
            config=self.config,
            workspace=workspace_for(self.db.root),
            secret_provider=CompositeSecretProvider(
                (EnvironmentSecretProvider(), _Secrets())
            ),
            approval_handler=self.approve,
            clock=lambda: self.clock,
        )

    def expect_upsert(self, rows):
        self.expected_write = {
            "source_id": self.db.source_id,
            "resource_id": self.db.resource_id,
            "key_columns": ["domain"],
            "insert_columns": ["domain", "name", "evidence_url"],
            "update_columns": ["name", "evidence_url"],
            "rows": rows,
        }

    def expect_update(self, *, intended_row=None, unique_keys=(("id",), ("domain",))):
        # Owner-authored fixture identity, never inferred from model claims.
        self.intended_update_row = dict(
            intended_row or {"id": 1, "domain": "existing.test", "name": "Before"}
        )
        self.intended_update_keys = tuple(tuple(key) for key in unique_keys)
        self.expected_write = {
            "source_id": self.db.source_id,
            "resource_id": self.db.resource_id,
            "where": [{"column": "id", "operator": "eq", "value": 1}],
            "assignments": [{"column": "name", "value": "Updated"}],
            "expected_affected_rows": 1,
        }

    def write_is_exact(self, request):
        if self.expected_write is None:
            return False
        operation = "upsert" if "rows" in self.expected_write else "update"
        if (
            request.tool_name != f"data_{operation}_rows"
            or request.capability_id != f"data.{operation}_rows"
        ):
            return False
        review = request.arguments.to_dict()
        if set(review) != {"arguments", "target", "preview"}:
            return False
        args = review["arguments"]
        if not isinstance(args, dict):
            return False
        if not isinstance(review["target"], dict) or (
            review["target"].get("source_id") != self.db.source_id
            or review["target"].get("resource_id") != self.db.resource_id
        ):
            return False
        fingerprint = args.pop("preview_fingerprint", None)
        if (
            not isinstance(fingerprint, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", fingerprint) is None
        ):
            return False
        # Model-derived lineage is validated by the production domain, never by
        # approval. Its presence cannot change the approved target or row values.
        args.pop("evidence_call_ids", None)
        expected = dict(self.expected_write)
        domain_selector = [
            {"column": "domain", "operator": "eq", "value": "existing.test"}
        ]
        if operation == "update" and self.profile == "user_flow":
            if set(review["target"]) - {
                "source_id",
                "resource_id",
                "name",
                "aliases",
                "source_name",
                "revision",
            }:
                return False
            where = args.get("where")
            if not isinstance(where, list) or not 1 <= len(where) <= 16:
                return False
            cells = []
            for predicate in where:
                if (
                    not isinstance(predicate, dict)
                    or set(predicate) != {"column", "operator", "value"}
                    or predicate["operator"] != "eq"
                ):
                    return False
                cells.append(
                    {"column": predicate["column"], "value": predicate["value"]}
                )
            if not self._intended_update_cells(cells, require_key=True):
                return False
            preview = review["preview"]
            if not isinstance(preview, dict) or set(preview) != {
                "matched_rows",
                "samples",
                "warnings",
            }:
                return False
            if (
                type(preview["matched_rows"]) is not int
                or preview["matched_rows"] != expected["expected_affected_rows"]
            ):
                return False
            samples = preview["samples"]
            if (
                not isinstance(samples, list)
                or len(samples) != 1
                or not isinstance(preview["warnings"], list)
            ):
                return False
            sample = samples[0]
            if not isinstance(sample, dict) or set(sample) != {
                "primary_key",
                "before",
                "after",
            }:
                return False
            if not self._intended_update_cells(
                sample["primary_key"], require_key=True
            ) or not self._intended_update_cells(sample["before"]):
                return False
            if canonical_json(sample["after"]) != canonical_json(
                expected["assignments"]
            ):
                return False
            expected["where"] = where
        elif operation == "update" and canonical_json(
            args.get("where")
        ) == canonical_json(domain_selector):
            # This fixture's unique domain identifies the same approved row as
            # id=1. No other predicate, count or assignment becomes acceptable.
            expected["where"] = domain_selector
        for key in ("rows", "key_columns", "insert_columns", "update_columns"):
            if key in args and key in expected:
                args[key] = sorted(args[key], key=canonical_json)
                expected[key] = sorted(expected[key], key=canonical_json)
        return canonical_json(args) == canonical_json(expected)

    def _intended_update_cells(self, cells, *, require_key=False):
        """Bounded fixture equality, not a SQL evaluator or execution authority."""
        if not isinstance(cells, list) or not 1 <= len(cells) <= 16:
            return False
        columns = set()
        for cell in cells:
            if not isinstance(cell, dict) or set(cell) != {"column", "value"}:
                return False
            column, value = cell["column"], cell["value"]
            if (
                not isinstance(column, str)
                or column in columns
                or column not in self.intended_update_row
            ):
                return False
            intended = self.intended_update_row[column]
            if (
                value is None
                or type(value) not in (str, int, float)
                or type(value) is not type(intended)
                or value != intended
            ):
                return False
            columns.add(column)
        return not require_key or any(
            set(key) <= columns for key in self.intended_update_keys
        )

    def routine_is_exact(self, request):
        if (
            self.routine_contract is None
            or request.tool_name != "routine_create"
            or request.capability_id != ROUTINE_CREATE_CAPABILITY_ID
        ):
            return False
        proposal = request.arguments.get("proposal")
        if not isinstance(proposal, Mapping):
            return False
        grants = proposal.get("capability_grants")
        if (
            not isinstance(grants, tuple)
            or len(grants) != 1
            or not isinstance(grants[0], Mapping)
        ):
            return False
        grant = grants[0]
        if (
            grant.get("capability_id") != "data.upsert_rows"
            or grant.get("max_calls_per_occurrence") != 1
            or canonical_json(grant.get("constraints"))
            != canonical_json(self.routine_grant)
        ):
            return False
        for key, value in self.routine_contract.items():
            actual = proposal.get(key)
            if key in {"per_run_max_cost_usd", "cumulative_max_cost_usd"}:
                if (
                    not isinstance(actual, str)
                    or not isinstance(value, str)
                    or Decimal(actual) != Decimal(value)
                ):
                    return False
                continue
            if key == "allowed_capability_ids" and isinstance(actual, tuple):
                if not isinstance(value, (list, tuple)):
                    return False
                if self.profile == "user_flow":
                    # A user allowing this table does not prescribe the tool
                    # sequence. Permit ordinary reads of that exact frozen
                    # source/resource alongside the same single native grant.
                    required = set(value)
                    allowed = required | {
                        "catalog.search",
                        "catalog.inspect",
                        "catalog.schema",
                        "catalog.traverse",
                        "data.query",
                    }
                    if not required <= set(actual) <= allowed:
                        return False
                    continue
                actual = sorted(actual)
            if canonical_json(actual) != canonical_json(value):
                return False
        return True

    async def approve(self, request):
        # At most one approved write/routine per foreground run; repeats get a
        # fresh run and authenticated preview. Unexpected effects fail closed.
        already = any(
            item["approved"] and item["run_id"] == request.run_id
            for item in self.approvals
        )
        accepted = not already and (
            (not self.deny_write and self.write_is_exact(request))
            or self.routine_is_exact(request)
            or (
                request.tool_name == "resolve_effect"
                and request.capability_id == "control.resolve_effect"
                and self.recovery_digest is not None
                and request.arguments.get("receipt_digest") == self.recovery_digest
                and request.arguments.get("decision") == "close_without_retry"
                and request.arguments.get("observation_unchanged") is True
            )
        )
        self.approvals.append(
            {
                "run_id": request.run_id,
                "call_id": request.call_id,
                "tool": request.tool_name,
                "arguments": request.arguments.to_dict(),
                "approved": accepted,
            }
        )
        if accepted and self.approval_hook is not None:
            hook, self.approval_hook = self.approval_hook, None
            await hook()
        return ApprovalDecision.APPROVE if accepted else ApprovalDecision.DENY

    async def run(self, prompt, *, user_prompt=None):
        self.assert_completed_runs()
        assert len(self.captures) < self.max_runs, "No unbudgeted evaluation run"
        if self.profile == "user_flow":
            assert isinstance(user_prompt, str) and user_prompt.strip(), (
                "User-flow cases require an explicit ordinary user request; "
                "never fall back to the structured test prompt."
            )
            prompt = user_prompt
        started = perf_counter()
        result = await self.agent.run(prompt, conversation_id=self.conversation_id)
        self.conversation_id = result.conversation_id
        return await self.capture(result, perf_counter() - started)

    async def capture(self, result, elapsed=None):
        transcript = await self.agent.transcript(result.run_id)
        self.captures.append((result, transcript, elapsed))
        return result, transcript

    def assert_report(self, result, status, counts=None):
        self.assert_completed_runs()
        if self.profile == "strict":
            assert_report(result, status, counts)
        else:
            # Exact state/effect assertions remain in the case. Prose semantics
            # require a separate evidence review, not a permissive regex or a
            # hidden model judge. A pytest pass is not an answer-accuracy pass.
            assert result.kind is LoopExitKind.COMPLETED and result.final_text
            self.answer_reviews.append(
                {
                    "run_id": result.run_id,
                    "expected_status": status,
                    "expected_counts": counts,
                    "final_text": result.final_text,
                    "status": "pending_evidence_review",
                }
            )

    def assert_completed_runs(self):
        for result, transcript, _ in self.captures:
            assert result.kind is LoopExitKind.COMPLETED, (
                result.reason,
                result.final_text,
            )
            validate_completed_transcript(transcript, result)
            assert 0 < result.usage.total_tokens <= self.config.limits.max_total_tokens
            assert 0 < result.steps <= self.config.limits.max_steps
            assert result.usage.cost_estimate.status.value == "complete"
            assert (
                result.usage.cost_estimate.amount_usd
                <= self.config.limits.max_estimated_cost_usd
            )

    async def scheduled_result(self, count):
        self.assert_completed_runs()
        async with asyncio.timeout(self.config.limits.max_wall_time_seconds + 30):
            while True:
                routines = await self.agent.list_routines()
                assert len(routines) == 1
                inspection = await self.agent.inspect_routine(routines[0].routine_id)
                assert inspection is not None
                captured = {result.run_id for result, _, _ in self.captures}
                terminal = [
                    item
                    for item in inspection.recent_occurrences
                    if item.terminal_run_id
                ]
                if len(terminal) >= count:
                    for item in terminal:
                        if item.terminal_run_id not in captured:
                            result = await self.agent._embedded._store.result(
                                item.terminal_run_id
                            )
                            assert result is not None
                            return await self.capture(result)
                assert inspection.routine.state.value not in {
                    "needs_attention",
                    "disabled",
                    "paused",
                }, inspection
                await asyncio.sleep(0.05)

    def assert_accounting(self):
        assert (
            len(self.db.model.requests) == self.setup_request_count
        ), "The setup-only scripted model must never run during evaluation"
        assert 0 < len(self.captures) <= self.max_runs
        usages = [usage for recording in self.recordings for usage in recording.usages]
        requests = [
            request for recording in self.recordings for request in recording.requests
        ]
        timings = [
            timing for recording in self.recordings for timing in recording.timings
        ]
        assert len(usages) == len(requests) == len(timings) and usages
        assert all(timing["usage_complete"] for timing in timings)
        for field in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
        ):
            assert sum(
                getattr(result.usage, field) for result, _, _ in self.captures
            ) == sum(getattr(usage, field) for usage in usages), field
        assert all(usage.cost_estimate.status.value == "complete" for usage in usages)
        assert sum(
            result.usage.cost_estimate.amount_usd for result, _, _ in self.captures
        ) == sum(usage.cost_estimate.amount_usd for usage in usages)
        groups: list[list[ModelRequest]] = [[]]
        for request in requests:
            assert request.deadline is not None
            assert 0 < request.max_total_tokens <= self.config.limits.max_total_tokens
            assert (
                0
                < request.max_estimated_cost_usd
                <= self.config.limits.max_estimated_cost_usd
            )
            assert request.attempt_deadline is not None
            assert request.attempt_deadline <= request.deadline
            assert request.call_policy == self.config.model_call_policy
            # Logical model requests can have a stricter, earlier ceiling than
            # their enclosing run. A fresh run resets its cumulative allowance;
            # a fresh attempt does not.
            if (
                groups[-1]
                and request.max_total_tokens > groups[-1][-1].max_total_tokens
            ):
                groups.append([])
            groups[-1].append(request)
        assert len(groups) <= len(self.captures)
        for group in groups:
            assert all(
                later.max_total_tokens is not None
                and earlier.max_total_tokens is not None
                and later.max_estimated_cost_usd is not None
                and earlier.max_estimated_cost_usd is not None
                and later.max_total_tokens <= earlier.max_total_tokens
                and later.max_estimated_cost_usd <= earlier.max_estimated_cost_usd
                for earlier, later in zip(group, group[1:])
            )

    async def report(self, case_id, status, failure):
        from daita.routines.capabilities import routine_inspection_projection

        receipts = await self.agent.list_effects()
        routines = [
            json.loads(
                canonical_json(
                    routine_inspection_projection(
                        await self.agent.inspect_routine(item.routine_id)
                    )
                )
            )
            for item in await self.agent.list_routines()
        ]
        return {
            "case_id": case_id,
            "status": status,
            "failure": failure,
            "evaluation_profile": self.profile,
            "answer_accuracy": {
                "status": (
                    "requires_separate_review"
                    if self.profile == "user_flow"
                    else "structured_assertions"
                ),
                "reviews": self.answer_reviews,
            },
            "recorded_at": datetime.now(UTC).isoformat(),
            "dependencies": {
                "python": platform.python_version(),
                **{
                    name: importlib.metadata.version(name)
                    for name in (
                        "openai",
                        "anthropic",
                        "google-genai",
                        "httpx",
                        "httpcore",
                    )
                },
            },
            "composition": "AgentConfig.model_route; real API counting/generation; real asyncpg/PostgreSQL",
            "model_id": self.config.model_route.candidates[0].provider_id,
            "setup": {
                "mode": self.setup_mode,
                "scripted_requests": self.setup_request_count,
            },
            "verified_stages": self.stages,
            "revision": subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip(),
            "source_sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (
                    Path(__file__),
                    REPO_ROOT / "tests/live/data/test_model_write_acceptance.py",
                    REPO_ROOT / "tests/live/data/test_write_release.py",
                    REPO_ROOT / "tests/support/postgresql_write_release.py",
                    *sorted((REPO_ROOT / "src/daita").rglob("*.py")),
                )
            },
            "limits": {
                "steps": self.config.limits.max_steps,
                "tokens": self.config.limits.max_total_tokens,
                "seconds": self.config.limits.max_wall_time_seconds,
                "estimated_cost_usd": str(self.config.limits.max_estimated_cost_usd),
                "profile": "strict" if self.config.limits == LIMITS else "diagnostic",
                "output_tokens": self.config.model_route.candidates[
                    0
                ].profile.max_output_tokens,
                "maximum_runs": self.max_runs,
            },
            "runs": [
                {
                    "result": json.loads(encode_loop_exit(result)),
                    "messages": [
                        json.loads(encode_message(message))
                        for message in transcript.messages
                    ],
                    "elapsed_seconds": elapsed,
                }
                for result, transcript, elapsed in self.captures
            ],
            "approvals": self.approvals,
            "routines": routines,
            "receipts": [
                {
                    "observation": json.loads(canonical_json(receipt.material())),
                    "receipt_digest": receipt.receipt_digest,
                    "resolution": (
                        asdict(receipt.resolution) if receipt.resolution else None
                    ),
                }
                for receipt in receipts
            ],
            "database_rows": await self.db.rows(),
            "driver": {
                "fault": self.db.probe.mode,
                "mutations": self.db.probe.mutations,
                "commit_attempts": self.db.probe.commit_attempts,
                "server_commits": self.db.probe.server_commits,
                "connections_closed": all(
                    connection.is_closed() for connection in self.db.probe.connections
                ),
            },
            "physical_attempts": [
                timing for recording in self.recordings for timing in recording.timings
            ],
            "requests": [
                {
                    "deadline": request.deadline,
                    "remaining_tokens": request.max_total_tokens,
                    "remaining_estimated_cost_usd": str(request.max_estimated_cost_usd),
                    "sensitivity": request.sensitivity.value,
                    "messages": [
                        json.loads(encode_message(message))
                        for message in request.messages
                    ],
                    "tools": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": json.loads(
                                canonical_json(tool.input_schema)
                            ),
                        }
                        for tool in request.tools
                    ],
                }
                for recording in self.recordings
                for request in recording.requests
            ],
            "responses": [
                {
                    "finish_reason": response.finish_reason.value,
                    "text": response.text,
                    "usage": {
                        field: getattr(response.usage, field)
                        for field in (
                            "input_tokens",
                            "output_tokens",
                            "reasoning_tokens",
                            "cache_read_tokens",
                            "cache_write_tokens",
                        )
                    },
                    "estimated_cost_status": response.usage.cost_estimate.status.value,
                    "estimated_cost_usd": str(response.usage.cost_estimate.amount_usd),
                    "provider_id": response.provider_id,
                    "provider_response_id": response.provider_response_id,
                    "calls": [
                        {
                            "name": call.name,
                            "id": call.id,
                            "arguments": json.loads(canonical_json(call.arguments)),
                        }
                        for call in response.tool_calls
                    ],
                }
                for recording in self.recordings
                for response in recording.responses
            ],
        }


@asynccontextmanager
async def evaluate(
    database,
    monkeypatch,
    model_id,
    evidence_path,
    case_id,
    *,
    max_runs=1,
    limits=LIMITS,
    setup_mode="model_authored",
    profile="strict",
    origin=None,
):
    config = live_config(model_id, limits=limits)
    recordings = []
    original = factory.create_llm_provider

    def record(*args, **kwargs):
        recording = RecordingProvider(original(*args, **kwargs))
        recordings.append(recording)
        return recording

    monkeypatch.setattr(factory, "create_llm_provider", record)
    scenario = Evaluation(
        database, config, recordings, max_runs, setup_mode=setup_mode, profile=profile
    )
    if origin is not None:
        assert origin.conversation_id, "An origin run must retain its conversation"
        scenario.conversation_id = origin.conversation_id
    status, failure = "passed", None
    validation_error = None
    try:
        await scenario.reopen()
        yield scenario
    except BaseException as error:
        status, failure = "failed", f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            await scenario.agent._embedded._routine_supervisor.close()
            # Retain failed scheduled runs too, even when no inbox result arrived.
            captured = {result.run_id for result, _, _ in scenario.captures}
            for routine in await scenario.agent.list_routines():
                inspection = await scenario.agent.inspect_routine(routine.routine_id)
                for occurrence in inspection.recent_occurrences:
                    if (
                        occurrence.reserved_run_id
                        and occurrence.reserved_run_id not in captured
                    ):
                        result = await scenario.agent._embedded._store.result(
                            occurrence.reserved_run_id
                        )
                        if result is not None:
                            scenario.captures.append(
                                (
                                    result,
                                    await scenario.agent.transcript(result.run_id),
                                    None,
                                )
                            )
            accounting = {"status": "passed", "failure": None}
            try:
                scenario.assert_accounting()
            except Exception as error:
                accounting = {
                    "status": "failed",
                    "failure": f"{type(error).__name__}: {error}",
                }
                if failure is None:
                    validation_error = error
            if failure is None:
                try:
                    if validation_error is not None:
                        raise validation_error
                    scenario.assert_completed_runs()
                except Exception as error:
                    validation_error = error
                    status, failure = "failed", f"{type(error).__name__}: {error}"
            report = await scenario.report(case_id, status, failure)
            report["accounting"] = accounting
        finally:
            await scenario.agent.close()  # drains and closes the owned router/delegates
        report["agent_closed"] = True
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
        if validation_error is not None:
            raise validation_error


def assert_preview_binding(transcript, operation="upsert"):
    calls = calls_for(transcript, f"data_{operation}_rows")
    assert len(calls) == 1, (
        f"Expected exactly one data_{operation}_rows call; observed {len(calls)}. "
        "A missing write or duplicate request fails effectiveness."
    )
    call = calls[0]
    preview_ids = {
        item.id for item in calls_for(transcript, f"data_preview_{operation}_rows")
    }
    previews = [
        block
        for message in transcript.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
        and block.call_id in preview_ids
        and not block.is_error
    ]
    assert any(
        isinstance(block.output["data"], Mapping)
        and block.output["data"]["preview_fingerprint"]
        == call.arguments["preview_fingerprint"]
        for block in previews
    )
    return call


def assert_exact_preview(scenario, transcript, operation="upsert"):
    call = assert_preview_binding(transcript, operation)
    matching = [item for item in scenario.approvals if item["call_id"] == call.id]
    assert len(matching) == 1 and matching[0]["approved"]
    assert canonical_json(matching[0]["arguments"]["arguments"]) == canonical_json(
        call.arguments
    )
    previews = [
        block.output["data"]
        for message in transcript.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
        and block.capability_id == f"data.preview_{operation}_rows"
        and not block.is_error
        and isinstance(block.output["data"], Mapping)
        and block.output["data"]["preview_fingerprint"]
        == call.arguments["preview_fingerprint"]
    ]
    assert previews
    review = matching[0]["arguments"]["preview"]
    fields = (
        ("matched_rows", "samples", "warnings")
        if operation == "update"
        else (
            "input_count",
            "inserted_count",
            "updated_count",
            "unchanged_count",
            "identity_sequence_gaps_possible",
            "classifications",
        )
    )
    expected = {key: previews[-1][key] for key in fields}
    if operation == "upsert":
        expected["classifications"] = expected["classifications"][:5]
    assert canonical_json(review) == canonical_json(expected)
