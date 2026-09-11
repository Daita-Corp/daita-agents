"""Offline checks of the live evaluation's evidence and execution plumbing."""

import asyncio
import json
from collections.abc import Mapping
from dataclasses import replace
from decimal import Decimal
from hashlib import sha256

import pytest
from _phase_f_live_support import (
    AUTHORIZATION,
    COST_ENV,
    DESTINATION,
    EXPIRES,
    NEXT_SLOT,
    REPEATS_ENV,
    RESEARCH_TOKEN,
    SOURCE,
    assert_action,
    assert_completed,
    assert_report,
    evaluate,
    limits,
    live_provider,
    model_ids,
    owner_routine_draft,
    repeats,
    summarize_reports,
)
from live.benchmarks._support import RecordingProvider
from test_mcp_actions import ActionModel, response

from daita._json import canonical_json
from daita.llm._lifecycle import closing_stream
from daita.llm.errors import ModelProviderError, ProviderErrorCode
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ModelStreamCompleted,
    ModelTextDelta,
    ModelUsage,
    TextBlock,
    ToolCall,
)
from daita.llm.pricing import CostEstimate
from daita.routines.capabilities import _spec_schema
from daita.routines.owner import _routine_proposal_payload


class HarnessModel(ActionModel):
    closed = False

    async def close(self, *, deadline: float | None = None) -> None:
        self.closed = True


class StreamingHarnessModel(HarnessModel):
    model_profile = replace(HarnessModel.model_profile, supports_streaming=True)

    async def stream(self, request):
        yield ModelStreamCompleted(await self.generate(request))


@pytest.mark.parametrize(
    "names", [("destination", "content"), ("channel_ref", "body_text")]
)
async def test_contract_comparison_harness_freezes_schema_and_prevents_dispatch(
    tmp_path, monkeypatch, names
):
    monkeypatch.setenv("DAITA_PHASE_F_LIVE_PROFILE", "user_flow")
    model = HarnessModel()
    path = tmp_path / "comparison.json"
    async with evaluate(
        tmp_path / "home",
        model,
        model.model_profile,
        path,
        routine=True,
        run_immediately=False,
        action_argument_names=names,
    ) as scenario:
        model.steps = [response(text="Prepare the assignment.")]
        origin, _ = await scenario.run("Prepare the assignment.")
        destination = (
            await scenario.agent.distribution_destinations(
                origin.conversation_id,
                sensitivity_ceiling=scenario.binding.maximum_outbound_sensitivity,
            )
        )[0]
        draft = owner_routine_draft(scenario, origin.run_id, destination.destination_id)
        proposal = await scenario.agent.propose_routine(draft)
        properties = _spec_schema(update=False)["properties"]
        assert isinstance(properties, Mapping)
        arguments = {
            key: value
            for key, value in _routine_proposal_payload(proposal).items()
            if key in properties and value is not None
        }
        arguments.update(
            skill_names=(),
            distribution_destination_id=destination.destination_id,
            requested_capability_grants=[
                {
                    "capability_id": grant.capability_id,
                    "constraints": grant.constraints,
                    "max_calls_per_occurrence": grant.max_calls_per_occurrence,
                }
                for grant in proposal.capability_grants
            ],
        )
        model.steps = [
            response(
                ToolCall("load", "toolbox_load", {"tool_names": ["routine_create"]})
            ),
            response(ToolCall("create", "routine_create", arguments)),
            response(
                text="Assignment saved for next Monday; no notification was sent."
            ),
        ]
        result, transcript = await scenario.run(scenario.routine_prompt())
        assert_completed(result, transcript)
        assert len(await scenario.agent.list_routines()) == 1
        assert (
            len(scenario.approvals) == 1 and scenario.approvals[0]["approved"] is True
        )
        assert scenario.server.calls == []
        assert await scenario.agent.list_effects() == ()
        schema_properties = scenario.action.input_schema["properties"]
        assert isinstance(schema_properties, Mapping)
        assert set(schema_properties) == set(names)
        prompt = scenario.routine_prompt()
        assert "do not run it today" in prompt
        assert "channel_ref" not in prompt and "body_text" not in prompt
        assert "binding_id" not in prompt and "fixed_arguments" not in prompt
    report = json.loads(path.read_text())
    assert report["status"] == "passed"
    assert report["action_argument_names"] == list(names)
    assert report["run_immediately"] is False
    assert report["metrics"]["action_dispatches"] == 0


async def test_routine_setup_context_follows_working_set_and_actual_usage(tmp_path):
    model = HarnessModel()
    path = tmp_path / "setup.json"
    async with evaluate(
        tmp_path / "home", model, model.model_profile, path, routine=True
    ) as scenario:
        model.steps = [
            replace(
                response(
                    ToolCall(
                        "load-routine",
                        "toolbox_load",
                        {"tool_names": ["routine_create"]},
                    )
                ),
                usage=ModelUsage(
                    input_tokens=3644,
                    output_tokens=80,
                    cost_estimate=CostEstimate.complete(Decimal(0)),
                ),
            ),
            replace(
                response(ToolCall("destination", "distribution_destination_list", {})),
                usage=ModelUsage(
                    input_tokens=4533,
                    output_tokens=47,
                    cost_estimate=CostEstimate.complete(Decimal(0)),
                ),
            ),
            replace(
                response(
                    ToolCall(
                        "load-action",
                        "toolbox_load",
                        {"tool_names": [scenario.action.local_name]},
                    )
                ),
                usage=ModelUsage(
                    input_tokens=6915,
                    output_tokens=16,
                    cost_estimate=CostEstimate.complete(Decimal(0)),
                ),
            ),
            replace(
                response(text="Setup inspection only; no routine has been created."),
                usage=ModelUsage(
                    input_tokens=7061,
                    output_tokens=121,
                    cost_estimate=CostEstimate.complete(Decimal(0)),
                ),
            ),
        ]
        model.steps = [
            replace(item, request_input_tokens=item.usage.input_tokens)
            for item in model.steps
        ]
        result, transcript = await scenario.run(
            "Prepare a weekly test notification with the latest status and tell me when it can run."
        )
        assert result.usage.total_tokens == 22417
        requests = scenario.provider.requests
        assert [r.max_total_tokens for r in requests] == [30000, 26276, 21696, 14765]
        texts = [
            "".join(b.text for b in r.messages[0].content if isinstance(b, TextBlock))
            for r in requests
        ]
        assert "Workspace text edits are artifact-backed" not in texts[0]
        assert "Available user-authorized procedural skill index" not in texts[0]
        assert len(texts[0].encode()) < 6000
        assert '"kind":"toolbox"' not in texts[0]
        assert "For scheduled work" not in texts[0]
        assert "For scheduled work" in texts[1]
        assert "For scheduled work" not in texts[3]
        assert "exact argument schema" in texts[1]
        assert "Foreground actions request approval when invoked" in texts[3]
        assert "Framework inbox destinations" in texts[3]
        assert "Framework inbox destinations" not in texts[0]
        assert model.provider_id not in texts[0]
        assert model.provider_id in texts[1]
        assert '"maximum_per_run_tokens":30000' in texts[1]
        assert '"maximum_per_run_cost_usd":"0.15"' in texts[1]
        assert "Routine authoring choices" not in texts[3]
        assert model.provider_id not in transcript.run.message
        for request in requests:
            assert (
                request.messages[1:] == transcript.messages[: len(request.messages) - 1]
            )
            assert request.sensitivity is scenario.binding.maximum_outbound_sensitivity
        assert "Latest measured request input: 6915 tokens" in texts[3]
        assert "recent input growth: 2382 tokens" in texts[3]
        assert "24976 tokens" in texts[3]
        assert "Budget pressure" in texts[3]
        assert "not reserved capacity or a guaranteed count" in texts[3]
        assert len(await scenario.agent.list_routines()) == 0
        assert scenario.server.calls == []
    report = json.loads(path.read_text())
    for request, saved in zip(requests, report["model_requests"], strict=True):
        expected = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in request.tools
        ]
        assert canonical_json(saved["tool_definitions"]) == canonical_json(expected)
        assert (
            saved["tool_definitions_sha256"]
            == sha256(canonical_json(expected).encode()).hexdigest()
        )


def action_script(scenario, *, uncertain=False, retry=False):
    args = {
        "destination": DESTINATION,
        "content": f"7 of 9 checks passed {RESEARCH_TOKEN} {SOURCE}",
    }
    steps = [
        response(
            ToolCall(
                "load",
                "toolbox_load",
                {
                    "tool_names": [
                        scenario.action.local_name,
                        scenario.research.local_name,
                    ]
                },
            )
        ),
        response(
            ToolCall(
                "research", scenario.research.local_name, {"query": "release readiness"}
            )
        ),
        response(ToolCall("action", scenario.action.local_name, args)),
    ]
    if retry:
        steps.append(
            response(
                ToolCall(
                    "retry",
                    scenario.action.local_name,
                    {**args, "content": args["content"] + " again"},
                )
            )
        )
    steps.append(
        response(
            text=json.dumps(
                {
                    "invocation_status": (
                        "uncertain" if uncertain else "server_reported"
                    ),
                    "downstream_verified": False,
                    "citation": SOURCE,
                    "finding": f"7 of 9 checks passed {RESEARCH_TOKEN}",
                }
            )
        )
    )
    if scenario.evaluation_profile == "user_flow":
        steps[-1] = response(
            text=(
                f"Research found 7 of 9 checks passed ({RESEARCH_TOKEN}), source {SOURCE}. "
                + (
                    "The notification is unconfirmed; I did not retry it."
                    if uncertain
                    else "The service reports the notification was invoked; downstream delivery is unverified."
                )
            )
        )
    return [
        replace(
            item,
            usage=ModelUsage(
                input_tokens=100,
                output_tokens=20,
                cost_estimate=CostEstimate.complete(Decimal("0.001")),
            ),
        )
        for item in steps
    ]


@pytest.mark.parametrize("profile_name", ["strict", "user_flow"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("fault", ["success", "tool_error", "disconnect"])
async def test_phase_f_harness_measures_real_runtime_and_preserves_failure_evidence(
    tmp_path, fault, streaming, profile_name, monkeypatch
):
    monkeypatch.setenv("DAITA_PHASE_F_LIVE_PROFILE", profile_name)
    model = StreamingHarnessModel() if streaming else HarnessModel()
    path = tmp_path / "phase-f-report.json"
    with pytest.raises(AssertionError, match="intentional evaluation failure"):
        async with evaluate(
            tmp_path / "home",
            model,
            model.model_profile,
            path,
            fault=fault,
            tool_count=128,
        ) as scenario:
            model.steps = action_script(
                scenario, uncertain=fault != "success", retry=fault != "success"
            )
            result, transcript = await scenario.run(
                "Research readiness and notify the release room."
            )
            assert_completed(result, transcript)
            await assert_action(scenario, uncertain=fault != "success")
            scenario.check_answer(
                result,
                status="server_reported" if fault == "success" else "uncertain",
                research=True,
            )
            raise AssertionError("intentional evaluation failure")
    report = json.loads(path.read_text())
    assert report["status"] == "failed"
    metrics = report["metrics"]
    requests = 4 if fault == "success" else 5
    assert metrics["model_requests"] == requests
    assert metrics["input_tokens"] == 100 * requests
    assert metrics["output_tokens"] == 20 * requests
    assert Decimal(metrics["estimated_cost_usd"]) == Decimal("0.001") * requests
    assert metrics["action_attempts"] == (1 if fault == "success" else 2)
    assert metrics["action_dispatches"] == 1
    assert metrics["mcp_methods"]["tools/call"] == 2
    assert metrics["mcp_methods"]["tools/list"] >= 3
    assert len(report["model_timings"]) == requests
    assert metrics["model_seconds"] > 0
    assert metrics["model_timing_complete"]
    assert all(item["outcome"] == "completed" for item in report["model_timings"])
    assert all(
        (item["first_event_seconds"] is not None) == streaming
        for item in report["model_timings"]
    )
    ceiling = 100_000 if profile_name == "user_flow" else 30_000
    assert report["model_requests"][0]["remaining_tokens"] == ceiling
    assert report["model_requests"][1]["remaining_tokens"] == ceiling - 120
    assert report["evaluation_profile"] == profile_name
    assert bool(report["answer_reviews"]) is (profile_name == "user_flow")
    assert report["runs"][0]["messages"]
    assert len(report["receipts"]) == 1
    assert report["approvals"][0]["approved"] is True
    assert model.closed
    report["status"] = "passed"
    (tmp_path / "phase-f-second.json").write_text(json.dumps(report))
    summary = summarize_reports(tmp_path)
    assert len(summary) == 1
    assert summary[0]["cases"] == 2
    assert summary[0]["failed"] == 1
    assert summary[0]["success_rate"] == 0.5
    assert summary[0]["action_dispatches"] == 2


@pytest.mark.parametrize("profile_name", ["strict", "user_flow"])
@pytest.mark.parametrize("fail_confirmation", [False, True])
async def test_phase_f_routine_harness_reaches_both_occurrences_with_one_approval(
    tmp_path,
    fail_confirmation,
    profile_name,
    monkeypatch,
):
    monkeypatch.setenv("DAITA_PHASE_F_LIVE_PROFILE", profile_name)
    model = HarnessModel()
    original_generate = model.generate

    async def generate(request):
        if (
            fail_confirmation
            and model.steps
            and model.steps[0].text == "Created the approved routine."
        ):
            model.steps.pop(0)
            raise ModelProviderError(
                ProviderErrorCode.TOKEN_BUDGET_INSUFFICIENT,
                "Fixture: next request cannot fit after committed creation.",
                usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal(0))),
            )
        return await original_generate(request)

    model.generate = generate  # type: ignore[method-assign]
    path = tmp_path / "routine.json"
    async with evaluate(
        tmp_path / "home", model, model.model_profile, path, routine=True
    ) as scenario:
        model.steps = [
            response(text="The owner requests one exact immediate and weekly routine.")
        ]
        origin, _ = await scenario.run(
            "Prepare the scheduled release-readiness report."
        )
        destination = (
            await scenario.agent.distribution_destinations(
                origin.conversation_id,
                sensitivity_ceiling=scenario.binding.maximum_outbound_sensitivity,
            )
        )[0]
        assert scenario.limits.max_total_tokens == (
            100_000 if profile_name == "user_flow" else 30_000
        )
        draft = owner_routine_draft(scenario, origin.run_id, destination.destination_id)
        proposal = await scenario.agent.propose_routine(draft)
        properties = _spec_schema(update=False)["properties"]
        assert isinstance(properties, Mapping)
        arguments = {
            key: value
            for key, value in _routine_proposal_payload(proposal).items()
            if key in properties and value is not None
        }
        arguments.update(
            skill_names=(),
            distribution_destination_id=destination.destination_id,
            requested_capability_grants=[
                {
                    "capability_id": grant.capability_id,
                    "constraints": grant.constraints,
                    "max_calls_per_occurrence": grant.max_calls_per_occurrence,
                }
                for grant in proposal.capability_grants
            ],
        )
        model.steps = [
            response(
                ToolCall(
                    "load-routine", "toolbox_load", {"tool_names": ["routine_create"]}
                )
            ),
            response(ToolCall("create", "routine_create", arguments)),
            response(text="Created the approved routine."),
            *action_script(scenario),
        ]
        result, transcript = await scenario.run(scenario.routine_prompt())
        from daita.loop.models import ConversationRun, LoopExitKind
        from daita.tui.projection import project_conversation

        receipt = next(
            outcome
            for call, outcome in transcript.tool_pairs
            if call.name == "routine_create"
        )
        assert receipt is not None and not receipt.is_error
        assert len(canonical_json(receipt.output).encode()) < 1600
        data = receipt.output["data"]
        assert isinstance(data, Mapping)
        details = data["routine"]
        assert isinstance(details, Mapping)
        assert isinstance(details["routine_id"], str)
        assert details["reserved_occurrences"] == 1
        assert (
            "contract_bindings" not in details
            and "authorized_instruction" not in details
        )
        inspection = await scenario.agent.inspect_routine(details["routine_id"])
        assert inspection is not None and inspection.routine.capability_grants
        assert inspection.routine.authorized_instruction == draft.authorized_instruction
        assert inspection.routine.contract_bindings.capability_contracts
        if fail_confirmation:
            assert result.kind is LoopExitKind.FAILED
            assert (
                result.reason == "token_budget_insufficient"
                and result.final_text is None
            )
            projected = project_conversation((ConversationRun(0, transcript, result),))
            notice = projected[-1].text
            assert "not rolled back" in notice
            assert details["routine_id"] in notice and "create committed" in notice
        else:
            assert_completed(result, transcript)
        assert len(await scenario.agent.list_routines()) == 1, result
        immediate, transcript = await scenario.scheduled_result(1)
        assert_completed(immediate, transcript)
        scenario.check_answer(immediate, status="server_reported", research=True)
        model.steps = action_script(scenario)
        scenario.clock = NEXT_SLOT
        scenario.agent._embedded._routine_supervisor.wake()
        weekly, transcript = await scenario.scheduled_result(2)
        assert_completed(weekly, transcript)
        scenario.check_answer(weekly, status="server_reported", research=True)
        await assert_action(scenario, count=2)
        assert len(scenario.approvals) == 1
    report = json.loads(path.read_text())
    assert report["status"] == "passed"
    assert len(report["runs"]) == 4  # one offline seed, creation, and two occurrences
    assert report["metrics"]["action_dispatches"] == 2
    assert model.closed


def test_phase_f_live_provider_requires_explicit_authorization(monkeypatch):
    monkeypatch.delenv(AUTHORIZATION, raising=False)
    monkeypatch.setenv("DAITA_PHASE_F_LIVE_LLM_API_KEY", "fixture-must-never-be-used")
    with pytest.raises(ValueError, match="required before constructing"):
        live_provider("openai:gpt-5.6-terra")


@pytest.mark.parametrize("streaming", [False, True])
async def test_phase_f_provider_failure_retains_unknown_usage_and_closes_provider(
    tmp_path,
    streaming,
):
    class FailedModel(StreamingHarnessModel):
        model_profile = replace(
            StreamingHarnessModel.model_profile, supports_streaming=streaming
        )

        async def generate(self, request):
            raise RuntimeError("fixture provider failed without usage")

    model = FailedModel()
    path = tmp_path / "phase-f-provider-error.json"
    with pytest.raises(RuntimeError, match="fixture provider failed"):
        async with evaluate(
            tmp_path / "home", model, model.model_profile, path
        ) as scenario:
            await scenario.run("Notify the release room.")
    report = json.loads(path.read_text())
    assert report["status"] == "failed"
    assert report["metrics"]["model_requests"] == 1
    assert report["metrics"]["model_responses"] == 0
    assert report["metrics"]["estimated_cost_usd"] is None
    assert report["metrics"]["usage_complete"] is False
    assert report["model_timings"][0]["returned_response"] is False
    assert model.closed
    summary = summarize_reports(tmp_path)[0]
    assert summary["cases_with_incomplete_usage"] == 1
    assert summary["cases_with_incomplete_cost"] == 1
    assert summary["known_estimated_cost_usd"] is None


async def test_stream_cancellation_retains_timing_and_closes_only_request():
    started = asyncio.Event()
    released = asyncio.Event()

    class WaitingModel(StreamingHarnessModel):
        async def stream(self, request):
            try:
                yield ModelTextDelta("partial")
                started.set()
                await asyncio.Event().wait()
            finally:
                released.set()

    model = WaitingModel()
    recorder = RecordingProvider(model)

    async def consume():
        async for _ in recorder.stream(
            ModelRequest(
                messages=(
                    CanonicalMessage(
                        role=MessageRole.USER, content=(TextBlock("fixture"),)
                    ),
                )
            )
        ):
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert released.is_set()
    assert not model.closed
    assert len(recorder.timings) == 1
    timing = recorder.timings[0]
    assert timing["outcome"] == "cancelled"
    assert timing["returned_response"] is False
    assert timing["usage_complete"] is False
    assert timing["first_event_seconds"] is not None
    assert timing["input_tokens"] is None
    await recorder.close()
    assert model.closed


@pytest.mark.parametrize("stage", ["completed", "partial", "cleanup_failure"])
async def test_recorder_distinguishes_terminal_close_from_interruption(stage):
    released = False
    completed = response(text="Recorded terminal evidence.")

    class ClosingModel(StreamingHarnessModel):
        async def stream(self, request):
            nonlocal released
            try:
                if stage == "partial":
                    yield ModelTextDelta("partial")
                else:
                    yield ModelStreamCompleted(completed)
            finally:
                released = True
                if stage == "cleanup_failure":
                    raise RuntimeError("fixture cleanup failed")

    model = ClosingModel()
    recorder = RecordingProvider(model)
    stream = recorder.stream(
        ModelRequest(
            messages=(
                CanonicalMessage(
                    role=MessageRole.USER, content=(TextBlock("fixture"),)
                ),
            )
        )
    )
    # Use the same terminal-close boundary as the production router.
    async with closing_stream(stream) as events:
        await anext(events)
        if stage == "cleanup_failure":
            with pytest.raises(RuntimeError, match="fixture cleanup failed"):
                await anext(events)
    assert released and not model.closed
    assert len(recorder.timings) == 1
    timing = recorder.timings[0]
    assert (
        timing["outcome"]
        == {
            "completed": "completed",
            "partial": "cancelled",
            "cleanup_failure": "failed",
        }[stage]
    )
    assert (
        timing["failure_type"]
        == {
            "completed": None,
            "partial": "GeneratorExit",
            "cleanup_failure": "RuntimeError",
        }[stage]
    )
    assert recorder.responses == ([] if stage == "partial" else [completed])
    assert recorder.usages == ([] if stage == "partial" else [completed.usage])
    assert timing["usage_complete"] is (stage != "partial")
    await recorder.close()
    assert model.closed


async def test_recorder_does_not_treat_missing_response_usage_as_measured_zero():
    model = HarnessModel()
    model.steps = [ModelResponse(finish_reason=FinishReason.STOP, text="fixture")]
    recorder = RecordingProvider(model)
    await recorder.generate(
        ModelRequest(
            messages=(
                CanonicalMessage(
                    role=MessageRole.USER,
                    content=(TextBlock("fixture"),),
                ),
            )
        )
    )
    assert recorder.timings[0]["returned_response"] is True
    assert recorder.timings[0]["usage_complete"] is False
    assert recorder.timings[0]["input_tokens"] is None
    assert recorder.timings[0]["known_estimated_cost_usd"] is None
    await recorder.close()


async def test_stream_failure_summary_keeps_known_cost_from_incomplete_attempt(
    tmp_path,
):
    class PartialModel(StreamingHarnessModel):
        async def generate(self, request):
            raise ModelProviderError(
                ProviderErrorCode.TIMEOUT,
                usage=ModelUsage(
                    input_tokens=123,
                    cost_estimate=CostEstimate.partial(
                        Decimal("0.002"),
                        code="unpriced_attempt",
                    ),
                ),
            )

    model = PartialModel()
    path = tmp_path / "phase-f-partial.json"
    async with evaluate(
        tmp_path / "home", model, model.model_profile, path
    ) as scenario:
        result, _ = await scenario.run("Notify the release room.")
        assert result.reason == "timeout"
    report = json.loads(path.read_text())
    assert report["metrics"]["input_tokens"] == 123
    assert report["metrics"]["estimated_cost_usd"] is None
    assert report["metrics"]["known_estimated_cost_usd"] == "0.002"
    assert report["model_timings"][0]["outcome"] == "failed"
    summary = summarize_reports(tmp_path)[0]
    assert summary["cases_with_incomplete_usage"] == 1
    assert summary["cases_with_incomplete_cost"] == 1
    assert summary["known_estimated_cost_usd"] == "0.002"


@pytest.mark.parametrize("value", ["0", "-1", "NaN", "Infinity", "bad"])
def test_phase_f_live_budget_rejects_unbounded_values(monkeypatch, value):
    monkeypatch.setenv(COST_ENV, value)
    with pytest.raises(ValueError):
        limits()


def test_phase_f_matrix_and_repetitions_are_bounded(monkeypatch):
    monkeypatch.setenv(REPEATS_ENV, "6")
    with pytest.raises(ValueError):
        repeats()
    monkeypatch.setenv("DAITA_PHASE_F_LIVE_MODEL_IDS", "one,one")
    with pytest.raises(ValueError):
        model_ids()


async def test_independent_scheduled_live_fixture_needs_no_model_creation(tmp_path):
    from live.test_phase_f_scheduled_live import admit_owner_routine

    model = HarnessModel()
    path = tmp_path / "independent-scheduled.json"
    async with evaluate(
        tmp_path / "home", model, model.model_profile, path, routine=True
    ) as scenario:
        model.steps = action_script(scenario)
        await admit_owner_routine(scenario)
        immediate, transcript = await scenario.scheduled_result(1)
        assert_completed(immediate, transcript)
        assert_report(immediate, status="server_reported", research=True)
        model.steps = action_script(scenario)
        scenario.clock = NEXT_SLOT
        scenario.agent._embedded._routine_supervisor.wake()
        weekly, transcript = await scenario.scheduled_result(2)
        assert_completed(weekly, transcript)
        await assert_action(scenario, count=2)
        assert scenario.approvals == []
        assert len(scenario.captures) == 2
    report = json.loads(path.read_text())
    assert report["setup_mode"] == "owner_admitted_fixture"
    assert len(report["routine_states"]) == 1
    assert len(report["runs"]) == 2
    assert report["metrics"]["tool_calls_by_name"].get("routine_create", 0) == 0


@pytest.mark.parametrize("selected", [None, "strict", "user_flow", "unknown"])
def test_phase_f_explicit_profile_preserves_strict_defaults(monkeypatch, selected):
    monkeypatch.delenv(COST_ENV, raising=False)
    if selected is None:
        monkeypatch.delenv("DAITA_PHASE_F_LIVE_PROFILE", raising=False)
    else:
        monkeypatch.setenv("DAITA_PHASE_F_LIVE_PROFILE", selected)
    if selected == "unknown":
        with pytest.raises(ValueError, match="DAITA_PHASE_F_LIVE_PROFILE"):
            limits()
        return
    value = limits()
    assert (
        value.max_steps,
        value.max_total_tokens,
        value.max_wall_time_seconds,
        value.max_estimated_cost_usd,
    ) == (
        (24, 100_000, 300, Decimal("0.50"))
        if selected == "user_flow"
        else (14, 30_000, 180, Decimal("0.15"))
    )


async def test_phase_f_user_flow_prompts_and_review_remain_separate(
    tmp_path, monkeypatch
):
    from _phase_f_live_support import REPORT_INSTRUCTION, report_path

    monkeypatch.setenv("DAITA_PHASE_F_LIVE_PROFILE", "user_flow")
    model = HarnessModel()
    path = tmp_path / "natural.json"
    async with evaluate(
        tmp_path / "home", model, model.model_profile, path, routine=True
    ) as scenario:
        prompts = [
            scenario.action_prompt(0),
            scenario.action_prompt(1),
            scenario.research_prompt(),
            scenario.routine_prompt(),
        ]
        for prompt in prompts:
            for internal in (
                REPORT_INSTRUCTION,
                scenario.binding.binding_id,
                scenario.action.capability_id,
                scenario.action.local_name,
                model.provider_id,
                "toolbox",
                "JSON",
                "fixed_arguments",
            ):
                assert internal not in prompt
            assert DESTINATION in prompt
        assert "100000" in prompts[-1] and "200000" in prompts[-1]
        assert "Monday" in prompts[-1] and "09:00 America/Chicago" in prompts[-1]
        assert "user_flow" in report_path("case").parts
        model.steps = [response(text="The assignment could not be saved.")]
        result, transcript = await scenario.run(prompts[-1])
        assert_completed(result, transcript)
        scenario.check_answer(result, status="not_dispatched", research=False)
        # Changing the environment cannot change the already selected configuration.
        monkeypatch.setenv("DAITA_PHASE_F_LIVE_PROFILE", "strict")
        assert scenario.routine_prompt() == prompts[-1]
        with pytest.raises((ValueError, AssertionError)):
            assert_report(result, status="not_dispatched", research=False)
    report = json.loads(path.read_text())
    assert report["evaluation_profile"] == "user_flow"
    assert report["answer_reviews"] == [
        {
            "run_id": result.run_id,
            "expected_status": "not_dispatched",
            "research": False,
            "review": "pending_evidence_review",
        }
    ]
    assert report["setup_mode"] == "model_authored"
    assert not report["receipts"] and model.closed


@pytest.mark.parametrize("stop", ["admission", "returned_usage"])
async def test_committed_mcp_effect_survives_budget_stop_without_replay(tmp_path, stop):
    from daita.loop.models import ConversationRun, LoopExitKind
    from daita.tui.projection import project_conversation

    model = HarnessModel()
    path = tmp_path / "stopped-effect.json"
    async with evaluate(
        tmp_path / "home", model, model.model_profile, path
    ) as scenario:
        model.steps = action_script(scenario)
        if stop == "admission":
            original_generate = model.generate

            async def fail_final(request):
                if len(model.steps) == 1:
                    model.steps.pop()
                    raise ModelProviderError(
                        ProviderErrorCode.TOKEN_BUDGET_INSUFFICIENT,
                        usage=ModelUsage(
                            cost_estimate=CostEstimate.complete(Decimal(0))
                        ),
                    )
                return await original_generate(request)

            model.generate = fail_final  # type: ignore[method-assign]
        else:
            model.steps[-1] = replace(
                model.steps[-1],
                usage=ModelUsage(
                    input_tokens=scenario.limits.max_total_tokens,
                    output_tokens=10,
                    cost_estimate=CostEstimate.complete(Decimal("0.001")),
                ),
            )
        result, transcript = await scenario.run(
            "Research release readiness and notify release-room once."
        )
        assert result.kind is LoopExitKind.FAILED
        assert (result.final_text is None) is (stop == "admission")
        assert len(scenario.provider.requests) == 4 and not model.steps
        assert result.usage.total_tokens == (
            360 if stop == "admission" else scenario.limits.max_total_tokens + 370
        )
        await assert_action(scenario)
        assert len(scenario.approvals) == 1
        receipt = (await scenario.agent.list_effects())[0]
        view = project_conversation((ConversationRun(0, transcript, result),))
        text = " ".join(item.text for item in view)
        assert receipt.receipt_id in text and "succeeded" in text
        assert "downstream outcome unverified" in text
        assert (
            len([call for call in scenario.server.calls if call[0] == "notify_release"])
            == 1
        )
    report = json.loads(path.read_text())
    from daita.storage.sqlite_codecs.transcripts import decode_loop_exit

    assert (
        decode_loop_exit(json.dumps(report["runs"][0]["result"])).kind
        is LoopExitKind.FAILED
    )
    assert len(report["receipts"]) == 1
