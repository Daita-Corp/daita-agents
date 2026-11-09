"""The work board places a real agent's jobs, routines and results in columns."""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from daita import (
    Agent,
    DeliveryView,
    IntervalSchedule,
    OutcomeState,
    ResourceRevisionBinding,
    RoutineIntent,
    SemanticAnnotation,
    SemanticEvidence,
    SemanticEvidenceKind,
    SemanticFieldReference,
    SemanticKind,
    SemanticSubject,
    SQLiteSource,
    WorkBoard,
    WorkColumn,
    WorkKind,
)
from daita.jobs import JobState
from daita.llm.errors import ModelProviderError, ProviderErrorCode
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    ModelUsage,
    ToolCall,
)
from daita.llm.pricing import CostEstimate
from daita.llm.protocols import ModelProvider
from daita.llm.providers.mock import MockModelProvider
from daita.storage.sqlite import SQLiteStateStore
from tests.support.jobs import stored_job
from tests.support.toolbox_model import ToolboxAwareMockModelProvider
from tests.support.workspace import workspace_for

pytestmark = pytest.mark.acceptance

NOW = datetime(2026, 9, 16, 12, 10, tzinfo=UTC)
NAME = "work-board"
USAGE = ModelUsage(
    input_tokens=100,
    output_tokens=10,
    cost_estimate=CostEstimate.complete(Decimal("0")),
)


def _stop(text: str) -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text, usage=USAGE)


def _profile(provider: ModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=128_000,
        max_output_tokens=2_000,
        supports_tools=True,
        supports_parallel_tools=False,
    )


async def _seed_waiting_job(state_path: Path, agent_id: str) -> None:
    """Store a job that is waiting for the user's answer."""

    store = await SQLiteStateStore.open(state_path)
    try:
        await store.admit_job(
            replace(
                stored_job("job-blocked", agent_id, at=NOW - timedelta(minutes=5)),
                state=JobState.WAITING_FOR_YOU,
                question="Which region should the report cover?",
            )
        )
    finally:
        await store.close()


async def _run_routine(
    agent: Agent,
    *,
    conversation_id: str,
    title: str,
    source_id: str,
) -> DeliveryView:
    created = await agent.create_routine(
        RoutineIntent(
            title=title,
            prompt="Report the current value from the exact resource.",
            schedule=IntervalSchedule(3_600, NOW + timedelta(hours=1)),
            sources=(source_id,),
        ),
        conversation_id=conversation_id,
    )
    seen = {
        item.delivery_id
        for item in await agent.list_deliveries(conversation_id=conversation_id)
    }
    await agent.run_routine_now(created.routine_id, expected_revision=created.revision)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 15.0
    while loop.time() < deadline:
        deliveries = await agent.list_deliveries(conversation_id=conversation_id)
        # The fixed clock gives every delivery the same time, so match by identity.
        new = [item for item in deliveries if item.delivery_id not in seen]
        if new:
            (delivery,) = new
            return delivery
        await asyncio.sleep(0.01)
    raise AssertionError(await agent.inspect_routine(created.routine_id))


def _columns(
    board: WorkBoard,
) -> dict[WorkColumn, set[tuple[WorkKind, str, str | None]]]:
    return {
        column: {(card.kind, card.id, card.delivery_id) for card in cards}
        for column, cards in board.columns.items()
    }


async def test_work_board_shows_real_jobs_routines_and_results_until_dismissed(
    tmp_path: Path,
) -> None:
    database = tmp_path / "values.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE current_value (value INTEGER NOT NULL)")
        connection.execute("INSERT INTO current_value VALUES (7)")
    provider = MockModelProvider(
        (
            _stop("Foreground authorization complete."),
            _stop("The current value is 7."),
            # The second routine's run fails, so its result needs the user.
            ModelProviderError(ProviderErrorCode.INVALID_REQUEST),
        ),
        provider_id="mock:work-board",
        complete_pricing=True,
    )
    options = {
        "root": tmp_path,
        "workspace": workspace_for(tmp_path),
        "model": provider,
        "model_profile": _profile(provider),
        "clock": lambda: NOW,
    }
    created = await Agent.create(NAME, **options)  # type: ignore[arg-type]
    agent_id = created.id
    await created.close()
    state_path = tmp_path / "agents" / NAME / "state.db"
    assert state_path.is_file()
    await _seed_waiting_job(state_path, agent_id)

    agent = await Agent.open(NAME, **options)  # type: ignore[arg-type]
    try:
        source = await agent.attach(SQLiteSource(database))
        origin = await agent.run("Authorize the scheduled value reports.")
        assert origin.conversation_id is not None
        succeeded = await _run_routine(
            agent,
            title="Current value report",
            conversation_id=origin.conversation_id,
            source_id=source.id,
        )
        failed = await _run_routine(
            agent,
            title="Current value export",
            conversation_id=origin.conversation_id,
            source_id=source.id,
        )
        provider.assert_consumed()
        assert succeeded.conclusion_state is OutcomeState.SUCCEEDED
        assert failed.conclusion_state is OutcomeState.FAILED
        routines = {item.title: item for item in await agent.list_routines()}

        board = await agent.work_board()
        assert board.generated_at == NOW
        assert _columns(board) == {
            WorkColumn.SCHEDULED: {
                (WorkKind.ROUTINE, routines["Current value report"].routine_id, None),
                (WorkKind.ROUTINE, routines["Current value export"].routine_id, None),
            },
            WorkColumn.RUNNING: set(),
            WorkColumn.NEEDS_YOU: {
                (WorkKind.JOB, "job-blocked", None),
                (WorkKind.ROUTINE_RUN, failed.subject_id, failed.delivery_id),
            },
            WorkColumn.DONE: {
                (WorkKind.ROUTINE_RUN, succeeded.subject_id, succeeded.delivery_id),
            },
        }
        titles = {
            card.id: card.title for cards in board.columns.values() for card in cards
        }
        assert titles[succeeded.subject_id] == "Current value report"
        assert titles[failed.subject_id] == "Current value export"
        assert all(
            card.due_at == NOW + timedelta(hours=1)
            for card in board.columns[WorkColumn.SCHEDULED]
            if card.kind is WorkKind.ROUTINE
        )

        assert await agent.acknowledge_delivery(succeeded.delivery_id) is not None
        dismissed = await agent.work_board()
        assert dismissed.columns[WorkColumn.DONE] == ()
        assert (
            dismissed.columns[WorkColumn.NEEDS_YOU]
            == board.columns[WorkColumn.NEEDS_YOU]
        )
    finally:
        await agent.close()


async def test_work_board_rejects_hosted_callers_and_out_of_range_windows(
    tmp_path: Path,
) -> None:
    provider = MockModelProvider((), provider_id="mock:work-board-bounds")
    local = await Agent.create(
        "work-board-bounds",
        root=tmp_path / "local",
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=_profile(provider),
        clock=lambda: NOW,
    )
    try:
        assert (await local.work_board(done_window=timedelta(days=7))).generated_at == (
            NOW
        )
        for window in (
            timedelta(0),
            timedelta(days=7, microseconds=1),
            24,
        ):
            with pytest.raises(ValueError, match="done window"):
                await local.work_board(done_window=window)  # type: ignore[arg-type]
    finally:
        await local.close()

    hosted = await Agent.create(
        "work-board-hosted",
        root=tmp_path / "hosted",
        hosted=True,
        model=provider,
        model_profile=_profile(provider),
    )
    try:
        with pytest.raises(ValueError, match="hosted work board is not available yet"):
            await hosted.work_board()
    finally:
        await hosted.close()


def _dump(path: Path) -> list[str]:
    with sqlite3.connect(path) as connection:
        return list(connection.iterdump())


async def test_stale_notes_become_ranked_review_cards_and_the_board_writes_nothing(
    tmp_path: Path,
) -> None:
    database = tmp_path / "shop.sqlite"
    with sqlite3.connect(database) as connection:
        for table in ("orders", "items", "gone", "fine"):
            connection.execute(f"CREATE TABLE {table} (id INTEGER, qty INTEGER)")
    provider = ToolboxAwareMockModelProvider((), provider_id="mock:work-board-review")
    clock = [NOW - timedelta(days=40)]
    agent = await Agent.create(
        "work-board-review",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=_profile(provider),
        clock=lambda: clock[0],
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        tables = {
            item.name: item
            for item in await agent.list_catalog_resources(source_id=source.id)
        }

        def query(name: str, call_id: str) -> ModelResponse:
            call = ToolCall(
                id=call_id,
                name="data_query",
                arguments={
                    "source_id": source.id,
                    "resource_ids": (tables[name].id,),
                    "sql": f"SELECT qty FROM {name}",
                },
            )
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS, tool_calls=(call,)
            )

        provider.replace_script(
            (
                _stop("hello"),
                query("orders", "q1"),
                _stop("done"),
                query("orders", "q2"),
                _stop("done"),
                query("items", "q3"),
                _stop("done"),
            )
        )
        first = await agent.run("hello")
        for days, name in ((3, "orders"), (2, "orders"), (20, "items")):
            clock[0] = NOW - timedelta(days=days)
            await agent.run(f"count {name}")
        provider.assert_consumed()

        def note(note_id: str, name: str) -> SemanticAnnotation:
            table = tables[name]
            return SemanticAnnotation(
                id=note_id,
                agent_id=agent.id,
                subject=SemanticSubject(
                    source_ids=(source.id,),
                    resource_ids=(table.id,),
                    fields=(SemanticFieldReference(table.id, "qty"),),
                ),
                kind=SemanticKind.METRIC_DEFINITION,
                statement=f"Meaning {note_id}.",
                evidence=(
                    SemanticEvidence(
                        SemanticEvidenceKind.USER_ASSERTION,
                        first.run_id,
                        message_position=0,
                    ),
                ),
                catalog_revisions=(
                    ResourceRevisionBinding(table.id, table.current_revision),
                ),
                created_at=NOW - timedelta(days=30),
                confirmed_at=NOW - timedelta(days=30),
            )

        for note_id, name in (
            ("n1", "orders"),
            ("n2", "orders"),
            ("n3", "items"),
            ("n4", "gone"),
            ("n5", "fine"),
        ):
            await agent.save_semantic_annotation(note(note_id, name))
        assert await agent.work_board() is not None

        changed = NOW - timedelta(hours=1)
        clock[0] = changed
        with sqlite3.connect(database) as connection:
            connection.execute("ALTER TABLE orders DROP COLUMN qty")
            connection.execute("ALTER TABLE items DROP COLUMN qty")
            connection.execute("DROP TABLE gone")
        await agent.refresh_source(source.id)
        clock[0] = NOW

        state = tmp_path / "agents" / "work-board-review" / "state.db"
        before = _dump(state)
        board = await agent.work_board()
        await agent.work_board()
        assert _dump(state) == before

        cards = board.columns[WorkColumn.NEEDS_YOU]
        assert [card.kind for card in cards] == [WorkKind.CATALOG_REVIEW] * 3
        assert [(c.reads_30d, c.reads_7d) for c in cards] == [(2, 2), (1, 0), (0, 0)]
        assert all(card.updated_at == changed for card in cards)
        orders, items, gone = cards
        assert "orders" in orders.title and orders.title.startswith("2 ")
        assert "items" in items.title and items.title.startswith("1 ")
        assert "gone" in gone.title

        # A damaged note row costs the review cards, never the rest of the board.
        with sqlite3.connect(state) as connection:
            connection.execute("UPDATE semantic_annotations SET data = 'not json'")
        damaged = await agent.work_board()
        assert damaged.columns[WorkColumn.NEEDS_YOU] == ()
    finally:
        await agent.close()


async def test_review_cards_skip_replaced_notes_and_accept_non_utc_note_times(
    tmp_path: Path,
) -> None:
    database = tmp_path / "shop.sqlite"
    with sqlite3.connect(database) as connection:
        for table in ("orders", "items", "fine"):
            connection.execute(f"CREATE TABLE {table} (id INTEGER, qty INTEGER)")
    provider = ToolboxAwareMockModelProvider(
        (), provider_id="mock:work-board-review-edges"
    )
    agent = await Agent.create(
        "work-board-review-edges",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=_profile(provider),
        clock=lambda: NOW,
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        tables = {
            item.name: item
            for item in await agent.list_catalog_resources(source_id=source.id)
        }
        provider.replace_script((_stop("hello"),))
        first = await agent.run("hello")
        local = NOW - timedelta(days=3)
        zone = local.astimezone(timezone(timedelta(hours=5)))

        def note(
            note_id: str, name: str, confirmed_at: datetime, replaces: str | None = None
        ) -> SemanticAnnotation:
            table = tables[name]
            return SemanticAnnotation(
                id=note_id,
                agent_id=agent.id,
                subject=SemanticSubject(
                    source_ids=(source.id,),
                    resource_ids=(table.id,),
                    fields=(SemanticFieldReference(table.id, "qty"),),
                ),
                kind=SemanticKind.METRIC_DEFINITION,
                statement=f"Meaning {note_id}.",
                evidence=(
                    SemanticEvidence(
                        SemanticEvidenceKind.USER_ASSERTION,
                        first.run_id,
                        message_position=0,
                    ),
                ),
                catalog_revisions=(
                    ResourceRevisionBinding(table.id, table.current_revision),
                ),
                supersedes_id=replaces,
                created_at=NOW - timedelta(days=30),
                confirmed_at=confirmed_at,
            )

        await agent.save_semantic_annotation(note("n1", "orders", zone))
        await agent.save_semantic_annotation(note("n2", "items", zone))
        old = await agent.read_semantic_annotation("n2")
        assert old is not None
        await agent.save_semantic_annotation(
            note("n3", "items", zone, replaces="n2"), expected_sha256=old.sha256
        )
        with sqlite3.connect(database) as connection:
            connection.execute("ALTER TABLE orders DROP COLUMN qty")
            connection.execute("ALTER TABLE items DROP COLUMN qty")
        await agent.refresh_source(source.id)
        state = tmp_path / "agents" / "work-board-review-edges" / "state.db"
        with sqlite3.connect(state) as connection:
            connection.execute("DELETE FROM catalog_changes")

        cards = (await agent.work_board()).columns[WorkColumn.NEEDS_YOU]
        by_table = {
            name: card for card in cards for name in tables if name in card.title
        }
        assert set(by_table) == {"orders", "items"}
        assert by_table["orders"].updated_at == local
        # n2 is replaced by n3, so the items card counts one note, not two.
        assert by_table["items"].title.startswith("1 ")
    finally:
        await agent.close()
