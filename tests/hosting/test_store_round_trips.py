"""How much state-store work a run does before its first model call (#150).

Counts go through the store's ``SQLDatabase`` boundary, the same one custom
adapters implement, so they hold for every adapter. Ceilings are generous on
purpose: they catch a return to per-source or per-call reads, not small drift.
"""

from __future__ import annotations

import asyncio
import contextvars
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from daita import Agent, IntervalSchedule, RoutineIntent, SQLiteSource
from daita.llm.models import FinishReason, ModelProfile, ModelResponse, ModelUsage
from daita.llm.pricing import CostEstimate
from daita.llm.providers.mock import MockModelProvider
from daita.storage.sql import SQLStateStore
from tests.support.workspace import workspace_for

_COUNTING: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "store_round_trip_counting", default=False
)


@dataclass
class _Counts:
    armed: bool = False
    connections: int = 0
    write_transactions: int = 0
    first_call: list[tuple[int, int]] = field(default_factory=list)


class _CountingConnection:
    def __init__(self, connection, counts: _Counts) -> None:
        self._connection = connection
        self._counts = counts

    def begin(self, *, write: bool = False) -> None:
        if write and self._counts.armed:
            self._counts.write_transactions += 1
        self._connection.begin(write=write)

    def __enter__(self):
        self._connection.__enter__()
        return self

    def __exit__(self, *exc):
        return self._connection.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._connection, name)


class _CountingDatabase:
    """Count connections opened by the run under test, not background work."""

    def __init__(self, database, counts: _Counts, *, all_tasks: bool) -> None:
        self._database = database
        self._counts = counts
        self._all_tasks = all_tasks

    def connect(self, *, read_only: bool = False):
        connection = self._database.connect(read_only=read_only)
        if self._counts.armed and (self._all_tasks or _COUNTING.get()):
            self._counts.connections += 1
            return _CountingConnection(connection, self._counts)
        return connection

    def normalize_error(self, error: BaseException) -> BaseException:
        return self._database.normalize_error(error)


class _FirstCallModel(MockModelProvider):
    def __init__(self, counts: _Counts, replies: int) -> None:
        usage = ModelUsage(cost_estimate=CostEstimate.complete(Decimal(0)))
        super().__init__(
            tuple(
                ModelResponse(
                    finish_reason=FinishReason.STOP, text="Done.", usage=usage
                )
                for _ in range(replies)
            ),
            provider_id="mock:store-round-trips",
            complete_pricing=True,
        )
        self._counts = counts

    async def generate(self, request):
        if self._counts.armed:
            self._counts.armed = False
            self._counts.first_call.append(
                (self._counts.connections, self._counts.write_transactions)
            )
        return await super().generate(request)


def _database(path: Path) -> Path:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, total INTEGER)")
    connection.execute("INSERT INTO orders VALUES (1, 7)")
    connection.commit()
    connection.close()
    return path


async def _open(tmp_path: Path, counts: _Counts, *, all_tasks: bool) -> Agent:
    model = _FirstCallModel(counts, replies=8)
    agent = await Agent.create(
        "store-round-trips",
        root=tmp_path / "home",
        workspace=workspace_for(tmp_path / "home"),
        model=model,
        model_profile=ModelProfile(
            id=model.provider_id,
            context_window_tokens=64_000,
            max_output_tokens=1_000,
            supports_tools=True,
        ),
    )
    store = _sql_store(agent)
    store._database = _CountingDatabase(store._database, counts, all_tasks=all_tasks)
    return agent


def _sql_store(agent: Agent) -> SQLStateStore:
    store = agent._embedded._store
    assert isinstance(store, SQLStateStore)
    return store


async def _user_run_count(agent: Agent, counts: _Counts) -> tuple[int, int]:
    # Only this task and what it starts carry the flag; background tasks don't.
    token = _COUNTING.set(True)
    try:
        counts.connections = counts.write_transactions = 0
        counts.armed = True
        result = await agent.run("What is the total?")
    finally:
        _COUNTING.reset(token)
    assert result.reason == "completed"
    return counts.first_call[-1]


async def test_user_run_store_reads_do_not_grow_with_sources(tmp_path):
    counts = _Counts()
    agent = await _open(tmp_path, counts, all_tasks=False)
    try:
        await agent.attach(SQLiteSource(_database(tmp_path / "s0.sqlite"), name="S0"))
        await agent.save_skill("report", "How to report.", "Report the total.")
        one_source = await _user_run_count(agent, counts)
        for index in range(1, 5):
            await agent.attach(
                SQLiteSource(_database(tmp_path / f"s{index}.sqlite"), name=f"S{index}")
            )
        five_sources = await _user_run_count(agent, counts)
    finally:
        await agent.close()
    # Measured 17 connections in 1.3.4 development (57 and 133 before #150).
    assert one_source[0] <= 25, one_source
    assert five_sources[0] <= one_source[0] + 1, (one_source, five_sources)


async def test_source_snapshot_is_one_read_and_ends_with_its_scope(tmp_path):
    counts = _Counts()
    agent = await _open(tmp_path, counts, all_tasks=False)
    try:
        source = await agent.attach(
            SQLiteSource(_database(tmp_path / "s0.sqlite"), name="S0")
        )
        store = _sql_store(agent)
        detached = asyncio.Event()

        async def read_after_detach():
            await detached.wait()
            return await store.list_sources(agent.id)

        async with store.source_snapshot(agent.id):
            assert (await store.list_sources(agent.id))[0].active
            # A detach outside the scope is not seen inside it...
            await asyncio.create_task(
                agent.detach(source.id), context=contextvars.Context()
            )
            assert (await store.list_sources(agent.id))[0].active
            loaded = await store.load_source(agent.id, source.id)
            assert loaded is not None and loaded.active
            lingering = asyncio.create_task(read_after_detach())
        # ...and every read after it, even from a task it started, is live.
        detached.set()
        assert not (await lingering)[0].active
        assert not (await store.list_sources(agent.id))[0].active
    finally:
        await agent.close()


async def test_routine_run_store_reads_stay_bounded(tmp_path):
    counts = _Counts()
    agent = await _open(tmp_path, counts, all_tasks=True)
    try:
        await agent.attach(SQLiteSource(_database(tmp_path / "s0.sqlite"), name="S0"))
        origin = await agent.run("Set up a report.")
        created = await agent.create_routine(
            RoutineIntent(
                title="Total report",
                prompt="Report the total.",
                schedule=IntervalSchedule(
                    3_600, datetime.now(UTC) + timedelta(hours=1)
                ),
            ),
            conversation_id=origin.conversation_id,
        )
        counts.connections = counts.write_transactions = 0
        counts.armed = True
        await agent.run_routine_now(
            created.routine_id, expected_revision=created.revision
        )
        for _ in range(2_000):
            if counts.first_call:
                break
            await asyncio.sleep(0.005)
        assert counts.first_call, "the routine run never reached the model"
        routine = counts.first_call[-1]
    finally:
        await agent.close()
    # Measured 30-32 including routine bookkeeping and background reads (63 before #150).
    assert routine[0] <= 45, routine
