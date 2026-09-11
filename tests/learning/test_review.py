"""Component-owned tests split from ``test_learning_candidates_phase4.py``."""

from __future__ import annotations

from tests.learning._candidate_support import (
    LEARNING_REVIEW_MAX_PROPOSALS,
    LEARNING_REVIEW_MAX_TOTAL_TOKENS,
    LEARNING_REVIEW_MAX_WALL_TIME_SECONDS,
    Agent,
    CandidateReviewMeasurement,
    CandidateReviewReport,
    Decimal,
    FinishReason,
    LearningCandidateError,
    LearningCandidateStatus,
    LearningReviewStatus,
    Mapping,
    MockModelProvider,
    ModelResponse,
    ModelUsage,
    OneShotCandidateReviewer,
    TextBlock,
    ToolCall,
    ToolResultBlock,
    _BlockingReviewer,
    _ids,
    _response,
    _review_response,
    asyncio,
    json,
    pytest,
    sqlite3,
    workspace_for,
)


@pytest.mark.parametrize("fails", [False, True])
async def test_reviewer_preserves_original_deadline_and_one_call(
    tmp_path, monkeypatch, fails
):
    from dataclasses import replace

    from daita.config import AgentConfig
    from daita.llm.errors import (
        ModelProviderError,
        ProviderErrorCode,
        before_generation,
    )
    from daita.llm.models import ModelCallPolicy, ModelSensitivity
    from daita.llm.providers.mock import MockModelProvider as DirectMock
    from daita.llm.routing import ModelProviderRegistration, ModelRouter, RetryPolicy

    policy = ModelCallPolicy(read_timeout_seconds=45)
    response = _response('{"candidates": []}')
    failure = before_generation(
        ModelProviderError(ProviderErrorCode.PROVIDER_UNAVAILABLE),
        code="test_transient_setup",
    )
    direct = DirectMock([failure if fails else response, response])
    reviewer = ModelRouter(
        (
            ModelProviderRegistration(
                provider=direct,
                profile=replace(direct.model_profile, supports_structured_output=True),
                allowed_sensitivities=frozenset(ModelSensitivity),
            ),
        ),
        retry_policy=RetryPolicy(max_attempts_per_candidate=1, max_total_attempts=1),
    )
    foreground = MockModelProvider([_response("Remembered for this conversation.")])
    agent = await Agent.create(
        "review-deadline",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=foreground,
        model_profile=foreground.model_profile,
        config=AgentConfig(model_call_policy=policy),
        reviewer_model=reviewer,
        id_factory=_ids(),
    )
    original = OneShotCandidateReviewer._review_once
    observed = {}

    async def delayed_review(self, started, progress, **kwargs):
        observed["started"] = started
        await asyncio.sleep(0.02)
        return await original(self, started, progress, **kwargs)

    monkeypatch.setattr(OneShotCandidateReviewer, "_review_once", delayed_review)
    try:
        await agent.run("Remember that booked revenue excludes completed refunds.")
        result = await agent.review_learning_candidates()
        assert result.status is (
            LearningReviewStatus.PROVIDER_FAILED
            if fails
            else LearningReviewStatus.COMPLETED
        )
        assert result.model_calls == 1
        assert len(direct.requests) == 1
        request = direct.requests[0]
        assert request.deadline is not None and request.attempt_deadline is not None
        assert request.call_policy == policy
        assert (
            request.deadline
            == observed["started"] + LEARNING_REVIEW_MAX_WALL_TIME_SECONDS
        )
        assert request.attempt_deadline <= request.deadline
    finally:
        await agent.close()
        await reviewer.close()


async def test_explicit_review_creates_only_inactive_idempotent_candidate(tmp_path):
    foreground = MockModelProvider(
        [
            _response("Understood."),
            _response("Ordinary answer."),
        ]
    )
    reviewer = MockModelProvider(
        [_review_response("run-1", text="Booked revenue excludes completed refunds.")]
    )
    agent = await Agent.create(
        "phase4-review",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        first = await agent.run(
            "Remember that booked revenue excludes completed refunds."
        )
        assert first.run_id == "run-1"
        assert await agent.read_memory() == ""

        review = await agent.review_learning_candidates()
        assert review.status is LearningReviewStatus.COMPLETED
        assert review.model_calls == 1
        assert len(review.candidates) == 1
        candidate = review.candidates[0]
        assert candidate.status is LearningCandidateStatus.AWAITING_REVIEW
        assert candidate.candidate.supporting_run_ids == ("run-1",)
        assert await agent.read_memory() == ""

        repeated = await agent.review_learning_candidates()
        assert repeated.status is LearningReviewStatus.ALREADY_REVIEWED
        assert repeated.model_calls == 0
        assert len(reviewer.requests) == 1

        ordinary = await agent.run("What is booked revenue?")
        assert ordinary.final_text == "Ordinary answer."
        still_inactive = await agent.read_learning_candidate(candidate.candidate.id)
        assert still_inactive is not None
        assert still_inactive.status is LearningCandidateStatus.AWAITING_REVIEW
        ordinary_prompt = "\n".join(
            block.text
            for message in foreground.requests[1].messages
            for block in message.content
            if isinstance(block, TextBlock)
        )
        assert "<untrusted-learning-candidate>" not in ordinary_prompt
        assert "Booked revenue excludes completed refunds." not in ordinary_prompt
    finally:
        await agent.close()


async def test_review_skips_unreadable_history_and_reviews_new_compatible_runs(
    tmp_path,
):
    ids = _ids()
    foreground = MockModelProvider(
        [
            _response("Historical answer."),
            _response("Current answer."),
        ]
    )
    reviewer = MockModelProvider(
        [_review_response("run-2", text="Booked revenue excludes completed refunds.")]
    )
    agent = await Agent.create(
        "phase4-unreadable-history",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=ids,
        workspace=workspace_for(tmp_path),
    )
    await agent.run("Historical request.")
    await agent.close()

    database = tmp_path / "agents" / "phase4-unreadable-history" / "state.db"
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT id, result FROM runs ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        run_id, result_data = row
        document = json.loads(result_data)
        usage = document["fields"]["usage"]
        assert usage["__record__"] == "ModelUsage"
        usage["fields"]["estimated_cost_usd"] = {"__decimal__": "0.01"}
        connection.execute(
            "UPDATE runs SET result = ? WHERE id = ?",
            (json.dumps(document), run_id),
        )

    agent = await Agent.open(
        "phase4-unreadable-history",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=ids,
        workspace=workspace_for(tmp_path),
    )
    try:
        unavailable = await agent.review_learning_candidates()
        assert unavailable.status is LearningReviewStatus.HISTORY_UNAVAILABLE
        assert unavailable.skipped_run_count == 1
        assert unavailable.model_calls == 0
        assert reviewer.requests == ()

        await agent.run("Remember that booked revenue excludes completed refunds.")
        reviewed = await agent.review_learning_candidates()

        assert reviewed.status is LearningReviewStatus.COMPLETED
        assert reviewed.reviewed_run_ids == ("run-2",)
        assert reviewed.skipped_run_count == 1
        assert reviewed.model_calls == 1
        assert len(reviewed.candidates) == 1
        assert len(reviewer.requests) == 1
    finally:
        await agent.close()


async def test_local_review_preparation_failure_is_not_a_provider_failure(
    tmp_path,
    monkeypatch,
):
    foreground = MockModelProvider([])
    reviewer = MockModelProvider([])
    agent = await Agent.create(
        "phase4-local-review-failure",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        workspace=workspace_for(tmp_path),
    )

    async def fail_artifact_state(self):
        raise RuntimeError("local state unavailable")

    monkeypatch.setattr(
        OneShotCandidateReviewer,
        "_artifact_state",
        fail_artifact_state,
    )
    try:
        result = await agent.review_learning_candidates()

        assert result.status is LearningReviewStatus.LOCAL_FAILED
        assert result.model_calls == 0
        assert reviewer.requests == ()
    finally:
        await agent.close()


async def test_cancelled_review_writes_no_candidate_or_review_stamp(tmp_path):
    foreground = MockModelProvider([_response("Understood.")])
    reviewer = _BlockingReviewer()
    agent = await Agent.create(
        "phase4-cancelled-review",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        reviewer_profile=reviewer.model_profile,
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.run("Remember that our fiscal year begins in February.")
        task = asyncio.create_task(agent.review_learning_candidates())
        await reviewer.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await agent.list_learning_candidates() == ()
        with sqlite3.connect(agent.home / "state.db") as connection:
            rows = connection.execute(
                "SELECT data FROM metadata " "WHERE key LIKE 'learning_review_stamps:%'"
            ).fetchall()
        assert rows == []
    finally:
        await agent.close()


async def test_malformed_provider_and_cost_preconditions_have_no_candidate_effect(
    tmp_path,
):
    foreground = MockModelProvider([_response("Understood.")])
    reviewer = MockModelProvider(
        [_response("not-json")],
        complete_pricing=True,
    )
    agent = await Agent.create(
        "phase4-fail-closed",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.run("Remember that our fiscal year begins in February.")
        required = await agent.review_learning_candidates()
        assert required.status is LearningReviewStatus.COST_LIMIT_REQUIRED
        assert len(reviewer.requests) == 0
        assert await agent.list_learning_candidates() == ()
    finally:
        await agent.close()


async def test_malformed_and_token_exhausted_reviews_fail_closed_without_retry(
    tmp_path,
):
    for name, response, expected in (
        (
            "malformed",
            _response("not-json"),
            LearningReviewStatus.MALFORMED_RESPONSE,
        ),
        (
            "tokens",
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text=json.dumps({"candidates": []}),
                usage=ModelUsage(input_tokens=LEARNING_REVIEW_MAX_TOTAL_TOKENS + 1),
            ),
            LearningReviewStatus.TOKEN_LIMIT_EXCEEDED,
        ),
    ):
        foreground = MockModelProvider([_response("Understood.")])
        reviewer = MockModelProvider([response])
        agent = await Agent.create(
            f"phase4-{name}",
            root=tmp_path,
            model=foreground,
            model_profile=foreground.model_profile,
            reviewer_model=reviewer,
            id_factory=_ids(),
            workspace=workspace_for(tmp_path),
        )
        try:
            await agent.run("Remember that our fiscal year begins in February.")
            result = await agent.review_learning_candidates()
            assert result.status is expected
            assert result.model_calls == 1
            assert len(reviewer.requests) == 1
            assert await agent.list_learning_candidates() == ()
        finally:
            await agent.close()


async def test_duplicate_model_proposals_collapse_before_persistence(tmp_path):
    content = "Our fiscal year begins in February."
    proposal = {
        "target": "memory",
        "source_ids": [],
        "supporting_run_ids": ["run-1"],
        "content": {"text": content},
    }
    foreground = MockModelProvider([_response("Understood.")])
    reviewer = MockModelProvider(
        [_response(json.dumps({"candidates": [proposal, proposal]}))]
    )
    agent = await Agent.create(
        "phase4-duplicate",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.run("Remember that our fiscal year begins in February.")
        result = await agent.review_learning_candidates()
        assert result.status is LearningReviewStatus.COMPLETED
        assert len(result.candidates) == 1
        assert result.duplicate_proposals_suppressed == 1
        assert len(await agent.list_learning_candidates()) == 1
    finally:
        await agent.close()


async def test_reviewer_never_accepts_more_than_four_proposals(tmp_path):
    proposals = [
        {
            "target": "memory",
            "source_ids": [],
            "supporting_run_ids": ["run-1"],
            "content": {"text": f"Durable convention {index}."},
        }
        for index in range(LEARNING_REVIEW_MAX_PROPOSALS + 1)
    ]
    foreground = MockModelProvider([_response("Understood.")])
    reviewer = MockModelProvider([_response(json.dumps({"candidates": proposals}))])
    agent = await Agent.create(
        "phase4-max-proposals",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.run("Remember these durable business conventions.")
        result = await agent.review_learning_candidates()
        assert result.status is LearningReviewStatus.MALFORMED_RESPONSE
        assert await agent.list_learning_candidates() == ()
    finally:
        await agent.close()


async def test_reviewer_is_disabled_by_default_and_foreground_never_invokes_it(
    tmp_path,
):
    foreground = MockModelProvider([_response("Done.")])
    agent = await Agent.create(
        "phase4-disabled",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        result = await agent.run("Remember this.")
        assert result.final_text == "Done."
        review = await agent.review_learning_candidates()
        assert review.status is LearningReviewStatus.DISABLED
        assert len(foreground.requests) == 1
    finally:
        await agent.close()


def test_candidate_review_evaluation_is_caller_owned_content_free_and_bounded():
    report = CandidateReviewReport(
        (
            CandidateReviewMeasurement(
                proposed_candidates=4,
                accepted_candidates=2,
                rejected_candidates=2,
                false_positive_candidates=1,
                duplicate_candidates_suppressed=3,
                background_model_calls=1,
                background_total_tokens=900,
                background_duration_ms=125,
                background_estimated_cost_usd=Decimal("0.02"),
                background_cost_complete=True,
            ),
        )
    )
    data = report.to_mapping()
    assert data["candidate_precision"] == "0.750000"
    assert data["acceptance_rate"] == "0.500000"
    assert data["rejection_rate"] == "0.500000"
    assert data["hard_safety_passed"] is True
    rendered = report.render_markdown()
    assert "Background model calls" in rendered
    assert "Booked revenue" not in rendered


async def test_reviewer_request_excludes_secret_shaped_input_material(tmp_path):
    assistant_secret = "ASSISTANT_SECRET_123456"
    tool_argument_secret = "TOOL_ARGUMENT_SECRET_123456"
    prefixed_assistant_secret = "PREFIXED_ASSISTANT_SECRET_123456"
    generic_token_secret = "GENERIC_TOKEN_SECRET_123456"
    natural_language_secret = "NATURAL_LANGUAGE_SECRET_123456"
    foreground = MockModelProvider(
        [
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="catalog-secret-probe",
                        name="catalog_search",
                        arguments={
                            "query": f"api_key={tool_argument_secret}",
                        },
                    ),
                ),
            ),
            _response(
                f"Understood. password={assistant_secret} "
                f"MY_ACCESS_TOKEN={prefixed_assistant_secret} "
                f"SESSION_TOKEN={generic_token_secret}. "
                f"API key is {natural_language_secret}"
            ),
        ]
    )
    reviewer = MockModelProvider([_response(json.dumps({"candidates": []}))])
    agent = await Agent.create(
        "phase4-reviewer-privacy",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        memory_secret = "MEMORY_SECRET_123456"
        profile_secret = "PROFILE_SECRET_123456"
        transcript_secret = "TRANSCRIPT_SECRET_123456"
        prefixed_memory_secret = "PREFIXED_MEMORY_SECRET_123456"
        prefixed_profile_secret = "PREFIXED_PROFILE_SECRET_123456"
        prefixed_transcript_secret = "PREFIXED_TRANSCRIPT_SECRET_123456"
        connection_secret = (
            "postgresql://reviewer_user:REVIEWER_PASSWORD@" "private.internal/reviewer"
        )
        secret_reference = "env:DAITA_PRIVATE_REVIEW_TOKEN"
        keychain_reference = "keychain:daita-private-review"
        await agent.set_memory(
            f"api_key: {memory_secret}\ncredential: {secret_reference}\n"
            f"DATABASE_PASSWORD={prefixed_memory_secret}\n"
            f"APP_CONNECTION_STRING={connection_secret}\n"
            "budget_token=24000"
        )
        await agent.set_user_profile(
            f"access_token={profile_secret}\nsecret={keychain_reference}\n"
            f"OPENAI_API_KEY={prefixed_profile_secret}"
        )
        await agent.run(
            "Remember that the fiscal year starts in February. "
            f"Bearer {transcript_secret} "
            f"AWS_SECRET_ACCESS_KEY={prefixed_transcript_secret}"
        )

        review = await agent.review_learning_candidates()

        assert review.status is LearningReviewStatus.COMPLETED
        assert len(reviewer.requests) == 1
        rendered_request = "\n".join(
            block.text
            for message in reviewer.requests[0].messages
            for block in message.content
            if isinstance(block, TextBlock)
        )
        assert memory_secret not in rendered_request
        assert profile_secret not in rendered_request
        assert transcript_secret not in rendered_request
        assert assistant_secret not in rendered_request
        assert tool_argument_secret not in rendered_request
        assert prefixed_assistant_secret not in rendered_request
        assert generic_token_secret not in rendered_request
        assert natural_language_secret not in rendered_request
        assert prefixed_memory_secret not in rendered_request
        assert prefixed_profile_secret not in rendered_request
        assert prefixed_transcript_secret not in rendered_request
        assert connection_secret not in rendered_request
        assert "private.internal/reviewer" not in rendered_request
        assert secret_reference not in rendered_request
        assert keychain_reference not in rendered_request
        assert "[redacted-secret]]" not in rendered_request
        assert "budget_token=24000" in rendered_request
    finally:
        await agent.close()


async def test_reviewer_redacts_secret_values_inside_bounded_tool_results(tmp_path):
    database = tmp_path / "reviewer-secrets.db"
    tool_result_secret = "TOOL_RESULT_SECRET_123456"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE credentials(api_key TEXT)")
        connection.execute(
            "INSERT INTO credentials(api_key) VALUES (?)",
            (tool_result_secret,),
        )
    foreground = MockModelProvider(())
    reviewer = MockModelProvider([_response(json.dumps({"candidates": []}))])
    agent = await Agent.create(
        "phase4-reviewer-result-privacy",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        source = await agent.attach_sqlite(database, name="credentials")
        (resource,) = await agent.list_catalog_resources(source_id=source.id)
        foreground.replace_script(
            (
                ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(
                        ToolCall(
                            id="secret-row",
                            name="data_query",
                            arguments={
                                "source_id": source.id,
                                "resource_ids": (resource.id,),
                                "sql": "SELECT api_key FROM credentials",
                            },
                        ),
                    ),
                ),
                _response("The requested record was inspected."),
            )
        )
        run = await agent.run(
            "Inspect the credential record without retaining its value.",
            source_scope_ids=(source.id,),
        )
        transcript = await agent.transcript(run.run_id)
        tool_result = transcript.messages[2].content[0]
        assert isinstance(tool_result, ToolResultBlock)
        data = tool_result.output["data"]
        assert isinstance(data, Mapping)
        rows = data["rows"]
        assert isinstance(rows, tuple)
        assert isinstance(rows[0], Mapping)
        assert rows[0]["api_key"] == tool_result_secret

        review = await agent.review_learning_candidates()

        assert review.status is LearningReviewStatus.COMPLETED
        rendered_request = "\n".join(
            block.text
            for message in reviewer.requests[0].messages
            for block in message.content
            if isinstance(block, TextBlock)
        )
        assert tool_result_secret not in rendered_request
    finally:
        await agent.close()


async def test_review_measurements_count_a_model_call_before_persistence_failure(
    tmp_path,
    monkeypatch,
):
    usage = ModelUsage(input_tokens=17, output_tokens=3)
    foreground = MockModelProvider([_response("Understood.")])
    reviewer = MockModelProvider(
        [
            _response(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "target": "memory",
                                "source_ids": [],
                                "supporting_run_ids": ["run-1"],
                                "content": {
                                    "text": (
                                        "Booked revenue excludes completed " "refunds."
                                    )
                                },
                            }
                        ]
                    }
                ),
                usage=usage,
            )
        ]
    )
    agent = await Agent.create(
        "phase4-review-measurement",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.run("Remember that booked revenue excludes completed refunds.")

        async def fail_after_review(*args, **kwargs):
            raise LearningCandidateError("candidate capacity exhausted")

        monkeypatch.setattr(
            agent._embedded._candidate_reviewer._store,
            "save_learning_candidate_review",
            fail_after_review,
        )

        result = await agent.review_learning_candidates()

        assert result.status is LearningReviewStatus.CAPACITY_EXHAUSTED
        assert result.model_calls == 1
        assert result.reviewed_run_ids == ("run-1",)
        assert result.usage == usage
        assert len(reviewer.requests) == 1
    finally:
        await agent.close()
