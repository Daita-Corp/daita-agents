"""Component-owned tests split from ``test_review.py``."""

from __future__ import annotations

from tests.learning._candidate_support import (
    LEARNING_CANDIDATE_MAX_RECORDS,
    Agent,
    DocumentCandidateContent,
    LearningCandidateError,
    LearningCandidateStatus,
    LearningCandidateTarget,
    MockModelProvider,
    SQLiteStateStore,
    _ids,
    _response,
    _review_response,
    _stored_candidate,
    learning_candidate_content_from_mapping,
    learning_candidate_content_to_mapping,
    pytest,
    sqlite3,
    workspace_for,
)


@pytest.mark.parametrize("target", ["memory", "user"])
def test_document_candidate_content_is_immutable(target: str):
    content = DocumentCandidateContent("Durable definition.")
    with pytest.raises(AttributeError):
        content.text = target  # type: ignore[misc]


async def test_candidate_storage_round_trips_exactly_and_enforces_64_and_owner(
    tmp_path,
):
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    for index in range(LEARNING_CANDIDATE_MAX_RECORDS):
        candidate, stamp = _stored_candidate(index)
        inserted = await store.save_learning_candidate_review(
            "agent-one",
            stamps=(stamp,),
            candidates=(candidate,),
        )
        assert inserted == (candidate,)
    values = await store.list_learning_candidates("agent-one")
    assert len(values) == LEARNING_CANDIDATE_MAX_RECORDS
    assert await store.load_learning_candidate("agent-one", "candidate-0") == values[0]

    extra, extra_stamp = _stored_candidate(LEARNING_CANDIDATE_MAX_RECORDS)
    with pytest.raises(LearningCandidateError, match="capacity"):
        await store.save_learning_candidate_review(
            "agent-one",
            stamps=(extra_stamp,),
            candidates=(extra,),
        )

    foreign, foreign_stamp = _stored_candidate(0, agent_id="agent-two")
    with pytest.raises(LearningCandidateError, match="another agent"):
        await store.save_learning_candidate_review(
            "agent-one",
            stamps=(foreign_stamp,),
            candidates=(foreign,),
        )


async def test_candidate_obsolescence_is_derived_without_mutating_candidate_row(
    tmp_path,
):
    foreground = MockModelProvider([_response("Understood.")])
    reviewer = MockModelProvider(
        [_review_response("run-1", text="Our fiscal year begins in February.")]
    )
    agent = await Agent.create(
        "phase4-obsolete",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.run("Remember that our fiscal year begins in February.")
        candidate = (await agent.review_learning_candidates()).candidates[0]
        database = agent.home / "state.db"
        with sqlite3.connect(database) as connection:
            stored_before = connection.execute(
                "SELECT data FROM learning_candidates WHERE id = ?",
                (candidate.candidate.id,),
            ).fetchone()[0]

        await agent.set_memory("An unrelated active artifact changed.")
        obsolete = await agent.read_learning_candidate(candidate.candidate.id)
        assert obsolete is not None
        assert obsolete.status is LearningCandidateStatus.OBSOLETE
        assert "referenced artifacts changed" in obsolete.obsolete_reasons
        with pytest.raises(ValueError, match="not awaiting review"):
            await agent.accept_learning_candidate(candidate.candidate.id)

        with sqlite3.connect(database) as connection:
            stored_after = connection.execute(
                "SELECT data FROM learning_candidates WHERE id = ?",
                (candidate.candidate.id,),
            ).fetchone()[0]
        assert stored_after == stored_before
    finally:
        await agent.close()


async def test_candidate_rows_and_review_metadata_never_copy_transcript_material(
    tmp_path,
):
    marker = "RAW_PROMPT_TOOL_ARGUMENT_ROW_CREDENTIAL_MARKER"
    foreground = MockModelProvider([_response("Understood.")])
    reviewer = MockModelProvider(
        [_review_response("run-1", text="Our fiscal year begins in February.")]
    )
    agent = await Agent.create(
        "phase4-sanitized",
        root=tmp_path,
        model=foreground,
        model_profile=foreground.model_profile,
        reviewer_model=reviewer,
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.run(
            "Remember that our fiscal year begins in February. "
            f"Do not persist this transcript marker: {marker}"
        )
        review = await agent.review_learning_candidates()
        assert len(review.candidates) == 1
        with sqlite3.connect(agent.home / "state.db") as connection:
            candidate_data = "\n".join(
                row[0]
                for row in connection.execute("SELECT data FROM learning_candidates")
            )
            stamp_data = "\n".join(
                row[0]
                for row in connection.execute(
                    "SELECT data FROM metadata WHERE key LIKE 'learning_review_stamps:%'"
                )
            )
        assert marker not in candidate_data
        assert marker not in stamp_data
        assert "tool_arguments" not in candidate_data
        assert '"rows"' not in candidate_data
        assert "credential" not in stamp_data.lower()
    finally:
        await agent.close()


def test_candidate_edit_projection_round_trips_frozen_mapping_values():
    original = DocumentCandidateContent("Use a February fiscal-year start.")
    projected = learning_candidate_content_to_mapping(original)
    assert (
        learning_candidate_content_from_mapping(
            LearningCandidateTarget.MEMORY,
            projected,
        )
        == original
    )
