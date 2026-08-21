from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from daita import (
    Agent,
    ResourceRevisionBinding,
    SemanticAnnotation,
    SemanticAnnotationState,
    SemanticEvidence,
    SemanticEvidenceKind,
    SemanticFieldReference,
    SemanticKind,
    SemanticSubject,
)
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import LoopLimits, ToolProjectionMode
from daita.semantics import (
    SEMANTIC_MAINTENANCE_MAX_NOTICES,
    SEMANTIC_MAX_ANNOTATIONS,
    SEMANTIC_RECALL_MAX_ANNOTATIONS,
    SEMANTIC_RECALL_MAX_UTF8_BYTES,
    SemanticResourceFact,
    SemanticValidationError,
    inspect_semantic_annotations,
    render_semantic_recall,
    semantic_duplicate_identity,
)

EAGER_LIMITS = LoopLimits(tool_projection_mode=ToolProjectionMode.EAGER)
from daita.skills import SKILL_MAX_COUNT
from daita.storage.sqlite import SQLiteStateStore

NOW = datetime(2026, 7, 28, 12, tzinfo=UTC)


def _annotation(
    annotation_id: str,
    *,
    agent_id: str = "agent-1",
    source_id: str = "source-1",
    resource_ids: tuple[str, ...] = ("resource-1",),
    field_names: tuple[str, ...] = ("metric",),
    revision: str = "revision-1",
    statement: str | None = None,
    kind: SemanticKind = SemanticKind.METRIC_DEFINITION,
    supersedes_id: str | None = None,
    created_at: datetime = NOW,
) -> SemanticAnnotation:
    return SemanticAnnotation(
        id=annotation_id,
        agent_id=agent_id,
        subject=SemanticSubject(
            source_ids=(source_id,),
            resource_ids=resource_ids,
            fields=tuple(
                SemanticFieldReference(resource_id, field_name)
                for resource_id, field_name in zip(
                    resource_ids,
                    field_names,
                    strict=True,
                )
            ),
        ),
        kind=kind,
        statement=statement or f"Definition for {annotation_id}.",
        evidence=(
            SemanticEvidence(
                SemanticEvidenceKind.USER_ASSERTION,
                "run-1",
                message_position=0,
            ),
        ),
        catalog_revisions=tuple(
            ResourceRevisionBinding(resource_id, revision)
            for resource_id in resource_ids
        ),
        supersedes_id=supersedes_id,
        created_at=created_at,
        confirmed_at=created_at,
    )


def _facts(
    *resource_ids: str,
    source_id: str = "source-1",
    revision: str = "revision-1",
    fields: tuple[str, ...] = ("metric",),
) -> tuple[SemanticResourceFact, ...]:
    return tuple(
        SemanticResourceFact(resource_id, source_id, revision, fields)
        for resource_id in resource_ids
    )


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=32_000,
        max_output_tokens=2_000,
        supports_tools=True,
        supports_parallel_tools=True,
    )


def _request_text(provider: MockModelProvider, index: int) -> str:
    return "\n".join(
        block.text
        for message in provider.requests[index].messages
        for block in message.content
        if isinstance(block, TextBlock)
    )


def test_duplicate_identity_recall_representative_and_inspection_are_deterministic():
    first = _annotation(
        "a-definition",
        statement="Paid revenue uses paid orders.",
    )
    duplicate = _annotation(
        "b-definition",
        statement="  PAID   revenue uses paid orders.  ",
        created_at=NOW + timedelta(seconds=1),
    )
    before = (duplicate, first)

    views = inspect_semantic_annotations(before, _facts("resource-1"))
    by_id = {view.annotation.id: view for view in views}

    assert semantic_duplicate_identity(first) == semantic_duplicate_identity(duplicate)
    assert by_id["a-definition"].state is SemanticAnnotationState.ACTIVE
    assert by_id["a-definition"].duplicate_ids == ("b-definition",)
    assert by_id["a-definition"].duplicate_of_id is None
    assert by_id["b-definition"].state is SemanticAnnotationState.DUPLICATE
    assert by_id["b-definition"].duplicate_ids == ("a-definition",)
    assert by_id["b-definition"].duplicate_of_id == "a-definition"
    assert before == (duplicate, first)

    recall = render_semantic_recall(
        tuple(reversed(views)),
        selected_resource_ids=("resource-1",),
        query="paid revenue",
    )
    assert recall.count("<semantic-annotation ") == 1
    assert 'id="a-definition"' in recall
    assert 'id="b-definition"' not in recall
    assert '<semantic-maintenance reason="exact_duplicate"' in recall
    assert "PAID   revenue" not in recall
    assert len(recall.encode("utf-8")) <= SEMANTIC_RECALL_MAX_UTF8_BYTES

    explicit_successor = replace(
        duplicate,
        id="successor",
        statement=first.statement,
        supersedes_id=first.id,
    )
    superseded_views = {
        view.annotation.id: view
        for view in inspect_semantic_annotations(
            (first, explicit_successor),
            _facts("resource-1"),
        )
    }
    assert superseded_views["a-definition"].state is (
        SemanticAnnotationState.SUPERSEDED
    )
    assert superseded_views["successor"].state is SemanticAnnotationState.ACTIVE
    assert superseded_views["successor"].duplicate_ids == ()


def test_conflict_supersession_and_all_stale_reasons_are_read_time_only():
    old = _annotation("old", statement="Use gross value.")
    superseder = _annotation(
        "new",
        statement="Use net value.",
        supersedes_id="old",
        created_at=NOW + timedelta(seconds=1),
    )
    conflict = _annotation(
        "other",
        statement="Use settled value.",
        created_at=NOW + timedelta(seconds=2),
    )
    stale = _annotation(
        "stale",
        resource_ids=("missing-resource", "changed-resource"),
        field_names=("gone", "metric"),
        revision="stored-revision",
        statement="STALE_CONTENT_SENTINEL",
    )
    annotations = (stale, conflict, superseder, old)
    views = inspect_semantic_annotations(
        annotations,
        (
            SemanticResourceFact(
                "changed-resource",
                "source-1",
                "current-revision",
                ("other_field",),
            ),
            *_facts("resource-1"),
        ),
    )
    by_id = {view.annotation.id: view for view in views}

    assert by_id["old"].state is SemanticAnnotationState.SUPERSEDED
    assert by_id["old"].superseded_by_id == "new"
    assert by_id["new"].state is SemanticAnnotationState.CONFLICTING
    assert by_id["other"].state is SemanticAnnotationState.CONFLICTING
    assert by_id["stale"].state is SemanticAnnotationState.STALE
    assert by_id["stale"].stale_reasons == (
        "missing_field:changed-resource.metric",
        "missing_resource:missing-resource",
        "revision_mismatch:changed-resource",
    )
    assert annotations == (stale, conflict, superseder, old)

    recall = render_semantic_recall(
        views,
        selected_resource_ids=("resource-1", "changed-resource"),
        query="changed resource metric",
    )
    assert "Use gross value." not in recall
    assert "Use net value." not in recall
    assert "Use settled value." not in recall
    assert "STALE_CONTENT_SENTINEL" not in recall
    assert '<semantic-maintenance reason="conflict"' in recall
    assert '<semantic-maintenance reason="stale"' in recall

    missing_resource_notice = render_semantic_recall(
        views,
        selected_resource_ids=(),
        query="Review missing-resource",
    )
    assert "missing_resource:missing-resource" in missing_resource_notice
    assert "STALE_CONTENT_SENTINEL" not in missing_resource_notice


@pytest.mark.parametrize("stored_count", (0, 32, 128, 256))
def test_annotation_saturation_and_prompt_bounds_remain_constant(stored_count: int):
    annotations = tuple(
        _annotation(
            f"definition-{index:03d}",
            resource_ids=(f"resource-{index:03d}",),
            field_names=(f"field_{index:03d}",),
        )
        for index in range(stored_count)
    )
    facts = tuple(
        SemanticResourceFact(
            f"resource-{index:03d}",
            "source-1",
            "revision-1",
            (f"field_{index:03d}",),
        )
        for index in range(stored_count)
    )
    views = inspect_semantic_annotations(annotations, facts)
    recall = render_semantic_recall(
        views,
        selected_resource_ids=tuple(
            f"resource-{index:03d}" for index in range(stored_count)
        ),
        query="all definitions",
    )

    assert len(views) == stored_count
    assert recall.count("<semantic-annotation ") <= (SEMANTIC_RECALL_MAX_ANNOTATIONS)
    assert len(recall.encode("utf-8")) <= SEMANTIC_RECALL_MAX_UTF8_BYTES

    if stored_count == SEMANTIC_MAX_ANNOTATIONS:
        with pytest.raises(SemanticValidationError, match="exceeds 256"):
            inspect_semantic_annotations(
                (*annotations, _annotation("overflow")),
                facts,
            )


@pytest.mark.parametrize("stored_count", (0, 32, 128, 256))
async def test_sqlite_stored_annotation_saturation_is_exact(
    tmp_path, stored_count: int
):
    store = await SQLiteStateStore.open(tmp_path / f"state-{stored_count}.db")
    try:
        for index in range(stored_count):
            await store.save_semantic_annotation(
                "agent-1",
                _annotation(f"stored-{index:03d}"),
            )
        assert len(await store.list_semantic_annotations("agent-1")) == stored_count
        if stored_count == SEMANTIC_MAX_ANNOTATIONS:
            with pytest.raises(SemanticValidationError, match="limited to 256"):
                await store.save_semantic_annotation(
                    "agent-1",
                    _annotation("stored-overflow"),
                )
    finally:
        await store.close()


@pytest.mark.parametrize("relevant_count", (1, 8, 24))
def test_relevant_recall_saturation_uses_every_available_slot(relevant_count: int):
    annotations = tuple(
        _annotation(
            f"relevant-{index:02d}",
            resource_ids=(f"resource-{index:02d}",),
            field_names=(f"field_{index:02d}",),
            statement="x",
        )
        for index in range(relevant_count)
    )
    facts = tuple(
        SemanticResourceFact(
            f"resource-{index:02d}",
            "source-1",
            "revision-1",
            (f"field_{index:02d}",),
        )
        for index in range(relevant_count)
    )

    recall = render_semantic_recall(
        inspect_semantic_annotations(annotations, facts),
        selected_resource_ids=tuple(
            f"resource-{index:02d}" for index in range(relevant_count)
        ),
        query="relevant",
    )

    assert recall.count("<semantic-annotation ") == relevant_count


def test_many_maintenance_records_broad_scopes_and_source_isolation_stay_bounded():
    duplicate_group = tuple(
        _annotation(
            f"duplicate-{index:02d}",
            resource_ids=("source-a-resource",),
            field_names=("metric",),
            source_id="source-a",
            statement="One exact definition.",
        )
        for index in range(30)
    )
    broad = _annotation(
        "broad",
        source_id="source-a",
        resource_ids=("source-a-resource", "source-a-other"),
        field_names=("metric", "metric"),
        statement="Both resources are required.",
    )
    foreign = _annotation(
        "foreign",
        source_id="source-b",
        resource_ids=("source-b-resource",),
        field_names=("metric",),
        statement="FOREIGN_SOURCE_SENTINEL",
    )
    stale = tuple(
        _annotation(
            f"stale-{index:02d}",
            resource_ids=(f"stale-resource-{index:02d}",),
            field_names=("metric",),
            revision="old",
            statement=f"STALE_{index:02d}",
        )
        for index in range(40)
    )
    facts = (
        SemanticResourceFact(
            "source-a-resource",
            "source-a",
            "revision-1",
            ("metric",),
        ),
        SemanticResourceFact(
            "source-a-other",
            "source-a",
            "revision-1",
            ("metric",),
        ),
        SemanticResourceFact(
            "source-b-resource",
            "source-b",
            "revision-1",
            ("metric",),
        ),
        *tuple(
            SemanticResourceFact(
                f"stale-resource-{index:02d}",
                "source-1",
                "new",
                ("metric",),
            )
            for index in range(40)
        ),
    )
    views = inspect_semantic_annotations(
        (*duplicate_group, broad, foreign, *stale),
        facts,
    )

    partial = render_semantic_recall(
        views,
        selected_resource_ids=("source-a-resource",),
        query="metric",
    )
    assert partial.count("<semantic-annotation ") == 1
    assert "Both resources are required." not in partial
    assert "FOREIGN_SOURCE_SENTINEL" not in partial
    assert partial.count('reason="exact_duplicate"') == 1

    complete = render_semantic_recall(
        views,
        selected_resource_ids=("source-a-resource", "source-a-other"),
        query="metric",
    )
    assert "Both resources are required." in complete
    assert "FOREIGN_SOURCE_SENTINEL" not in complete

    all_stale = render_semantic_recall(
        views,
        selected_resource_ids=tuple(
            f"stale-resource-{index:02d}" for index in range(40)
        ),
        query="stale metrics",
    )
    assert all_stale.count("<semantic-maintenance ") == (
        SEMANTIC_MAINTENANCE_MAX_NOTICES
    )
    assert not any(f"STALE_{index:02d}" in all_stale for index in range(40))
    assert len(all_stale.encode("utf-8")) <= SEMANTIC_RECALL_MAX_UTF8_BYTES


async def test_related_foreground_run_can_inspect_stale_record_without_using_it(
    tmp_path,
):
    database = tmp_path / "foreground-revalidation.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE invoices(id INTEGER PRIMARY KEY, booked_at TEXT)"
        )
    seed_provider = MockModelProvider(
        (ModelResponse(finish_reason=FinishReason.STOP, text="seeded"),)
    )
    agent = await Agent.create(
        "foreground-revalidation",
        root=tmp_path,
        model=seed_provider,
        model_profile=_profile(seed_provider),
        clock=lambda: NOW,
    )
    try:
        source = await agent.attach_sqlite(database)
        resource = (await agent.list_catalog_resources())[0]
        run = await agent.run("Teach invoice meaning.")
        transcript = await agent.transcript(run.run_id)
        annotation = SemanticAnnotation(
            id="invoice-time",
            agent_id=agent.id,
            subject=SemanticSubject(
                source_ids=(source.id,),
                resource_ids=(resource.id,),
                fields=(SemanticFieldReference(resource.id, "booked_at"),),
            ),
            kind=SemanticKind.TIME_SEMANTICS,
            statement="STALE_STATEMENT_SENTINEL",
            evidence=(
                SemanticEvidence(
                    SemanticEvidenceKind.USER_ASSERTION,
                    run.run_id,
                    message_position=0,
                ),
            ),
            catalog_revisions=(
                ResourceRevisionBinding(resource.id, resource.current_revision),
            ),
            created_at=transcript.run.created_at,
            confirmed_at=transcript.run.created_at,
        )
        await agent.save_semantic_annotation(annotation)
        with sqlite3.connect(database) as connection:
            connection.execute("ALTER TABLE invoices ADD COLUMN paid_at TEXT")
        await agent.refresh_source(source.id)
    finally:
        await agent.close()

    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="inspect-stale",
                        name="semantic_view",
                        arguments={"id": "invoice-time"},
                    ),
                ),
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="needs review"),
        )
    )
    reopened = await Agent.open(
        "foreground-revalidation",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
        clock=lambda: NOW,
    )
    try:
        result = await reopened.run("What does invoices.booked_at mean?")
        initial_prompt = _request_text(provider, 0)
        assert "STALE_STATEMENT_SENTINEL" not in initial_prompt
        assert '<semantic-maintenance reason="stale"' in initial_prompt
        assert {"semantic_view", "semantic_save"} <= {
            tool.name for tool in provider.requests[0].tools
        }
        transcript = await reopened.transcript(result.run_id)
        viewed = next(
            block
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock) and block.call_id == "inspect-stale"
        )
        assert not viewed.is_error
        data = viewed.output["data"]
        assert isinstance(data, Mapping)
        maintenance = data["maintenance"]
        assert isinstance(maintenance, Mapping)
        assert maintenance["state"] == "stale"
        assert maintenance["usable_as_current_meaning"] is False
        assert maintenance["requires_revalidation"] is True
        assert "STALE_STATEMENT_SENTINEL" in repr(viewed)
        assert "review material only" in _request_text(provider, 1)
    finally:
        await reopened.close()


async def test_maximum_skill_index_stays_shallow_and_multiple_bodies_load_progressively(
    tmp_path,
):
    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="first-skill",
                        name="skill_view",
                        arguments={"name": "skill-00"},
                    ),
                    ToolCall(
                        id="last-skill",
                        name="skill_view",
                        arguments={"name": f"skill-{SKILL_MAX_COUNT - 1:02d}"},
                    ),
                ),
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="used both"),
        )
    )
    agent = await Agent.create(
        "maximum-skill-index",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
    )
    try:
        for index in range(SKILL_MAX_COUNT):
            await agent.save_skill(
                f"skill-{index:02d}",
                f"Use procedure {index:02d}.",
                f"FULL_SKILL_BODY_{index:02d}",
            )
        await agent.run("Use the first and last relevant procedures.")

        initial = _request_text(provider, 0)
        assert initial.count("- skill-") == SKILL_MAX_COUNT
        assert "FULL_SKILL_BODY" not in initial
        continued = repr(provider.requests[1].messages)
        assert "FULL_SKILL_BODY_00" in continued
        assert f"FULL_SKILL_BODY_{SKILL_MAX_COUNT - 1:02d}" in continued
        assert "FULL_SKILL_BODY_01" not in continued
    finally:
        await agent.close()


async def test_identically_named_resources_remain_isolated_by_selected_source(
    tmp_path,
):
    first_database = tmp_path / "first-source.db"
    second_database = tmp_path / "second-source.db"
    for database in (first_database, second_database):
        with sqlite3.connect(database) as connection:
            connection.execute(
                "CREATE TABLE invoices(id INTEGER PRIMARY KEY, amount REAL)"
            )
    provider = MockModelProvider(
        (
            ModelResponse(finish_reason=FinishReason.STOP, text="seed"),
            ModelResponse(finish_reason=FinishReason.STOP, text="answer"),
        )
    )
    agent = await Agent.create(
        "same-resource-names",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        clock=lambda: NOW,
    )
    try:
        first_source = await agent.attach_sqlite(first_database, name="first")
        second_source = await agent.attach_sqlite(second_database, name="second")
        resources = await agent.list_catalog_resources()
        first_resource = next(
            item for item in resources if item.source_id == first_source.id
        )
        second_resource = next(
            item for item in resources if item.source_id == second_source.id
        )
        assert first_resource.name == second_resource.name == "invoices"
        seed = await agent.run(
            "Teach the first invoice meaning.",
            source_id=first_source.id,
        )
        seed_transcript = await agent.transcript(seed.run_id)
        await agent.save_semantic_annotation(
            SemanticAnnotation(
                id="first-only",
                agent_id=agent.id,
                subject=SemanticSubject(
                    source_ids=(first_source.id,),
                    resource_ids=(first_resource.id,),
                    fields=(SemanticFieldReference(first_resource.id, "amount"),),
                ),
                kind=SemanticKind.METRIC_DEFINITION,
                statement="FIRST_SOURCE_ONLY_SENTINEL",
                evidence=(
                    SemanticEvidence(
                        SemanticEvidenceKind.USER_ASSERTION,
                        seed.run_id,
                        message_position=0,
                    ),
                ),
                catalog_revisions=(
                    ResourceRevisionBinding(
                        first_resource.id,
                        first_resource.current_revision,
                    ),
                ),
                created_at=seed_transcript.run.created_at,
                confirmed_at=seed_transcript.run.created_at,
            )
        )

        await agent.run(
            "What does invoices.amount mean?",
            source_id=second_source.id,
        )
        prompt = _request_text(provider, 1)
        assert "FIRST_SOURCE_ONLY_SENTINEL" not in prompt
        assert first_resource.id not in prompt
        assert second_resource.id in prompt
    finally:
        await agent.close()
