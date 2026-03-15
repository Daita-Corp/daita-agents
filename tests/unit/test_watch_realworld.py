"""
Real-world integration tests for the @agent.watch() system.

Three scenarios that model actual production use cases without requiring
any external services:

1. SQLite order queue — polls a real SQL table, fires when pending orders
   exceed a threshold, resolves when the backlog clears.

2. asyncio.Queue depth monitor — models a task queue (Celery, RQ, Redis list)
   where a producer adds work; watch fires when the backlog grows too large.

3. In-memory metrics counter — models polling a stats endpoint or Prometheus
   gauge; watch fires when error rate spikes, resolves when it recovers.

Timing strategy: all tests use timedelta(seconds=0) so PollingWatchSource
calls asyncio.sleep(0) between polls. _drain_tasks() also calls
asyncio.sleep(0), so one drain call = one watch poll cycle. State is set up
before watches start wherever possible so the first poll already sees the data.
"""

import asyncio
from datetime import timedelta

import pytest

from daita.agents.agent import Agent
from daita.core.watch import PollingWatchSource, WatchEvent
from daita.llm.mock import MockLLMProvider
from daita.plugins.sqlite import SQLitePlugin

ZERO = timedelta(seconds=0)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _agent() -> Agent:
    return Agent(name="watcher", llm_provider=MockLLMProvider(delay=0))


def _polling(condition, plugin=None) -> PollingWatchSource:
    """Build a zero-interval PollingWatchSource for test use."""
    return PollingWatchSource(plugin=plugin, condition=condition, interval=ZERO)


async def _drain(agent: Agent, cycles: int = 30) -> None:
    """Yield control `cycles` times; each sleep(0) lets the watch run one poll."""
    for _ in range(cycles):
        await asyncio.sleep(0)


async def _stop(agent: Agent) -> None:
    for task in agent._tasks:
        task.cancel()
    await asyncio.gather(*agent._tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# Scenario 1: SQLite order queue
#
# A fulfilment system inserts orders into a SQLite table. The watch polls
# SELECT COUNT(*) FROM orders WHERE status='pending' and alerts when the
# backlog exceeds 5 orders. When orders are processed the watch resolves.
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE orders (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    status TEXT NOT NULL DEFAULT 'pending'
);
"""


@pytest.fixture
async def order_db():
    async with SQLitePlugin(path=":memory:") as db:
        await db.execute_script(SCHEMA)
        yield db


class TestSQLiteOrderQueue:
    async def test_fires_when_pending_orders_exceed_threshold(self, order_db):
        # Insert 6 orders before starting the watch so the first poll triggers
        for _ in range(6):
            await order_db.execute("INSERT INTO orders (status) VALUES ('pending')")

        agent = _agent()
        alerts = []

        @agent.watch(
            source=order_db,
            condition="SELECT COUNT(*) FROM orders WHERE status='pending'",
            threshold=lambda count: count > 5,
            interval=ZERO,
        )
        async def on_backlog(event: WatchEvent):
            alerts.append(event.value)

        await agent._start_watches()
        await _drain(agent)
        await _stop(agent)

        assert len(alerts) > 0
        assert all(v > 5 for v in alerts)

    async def test_does_not_fire_below_threshold(self, order_db):
        # Only 3 orders — below threshold of 5
        for _ in range(3):
            await order_db.execute("INSERT INTO orders (status) VALUES ('pending')")

        agent = _agent()
        alerts = []

        @agent.watch(
            source=order_db,
            condition="SELECT COUNT(*) FROM orders WHERE status='pending'",
            threshold=lambda count: count > 5,
            interval=ZERO,
        )
        async def on_backlog(event: WatchEvent):
            alerts.append(event.value)

        await agent._start_watches()
        await _drain(agent)
        await _stop(agent)

        assert alerts == []

    async def test_resolves_when_backlog_clears(self, order_db):
        # Start with 6 pending — above threshold
        for _ in range(6):
            await order_db.execute("INSERT INTO orders (status) VALUES ('pending')")

        agent = _agent()
        trigger_values = []
        resolve_values = []

        @agent.watch(
            source=order_db,
            condition="SELECT COUNT(*) FROM orders WHERE status='pending'",
            threshold=lambda count: count > 3,
            interval=ZERO,
            on_resolve=True,
        )
        async def on_change(event: WatchEvent):
            if event.resolved:
                resolve_values.append(event.value)
            else:
                trigger_values.append(event.value)

        await agent._start_watches()
        await _drain(agent, 10)  # trigger fires

        # Process all orders
        await order_db.execute("UPDATE orders SET status='done'")

        await _drain(agent, 20)  # resolve fires
        await _stop(agent)

        assert len(trigger_values) > 0, "Expected trigger events"
        assert len(resolve_values) > 0, "Expected resolve events"
        assert all(v <= 3 for v in resolve_values)

    async def test_previous_value_reflects_last_poll(self, order_db):
        """previous_value lets handlers compute delta (e.g. orders/second)."""
        # Start with 2 orders, then add 4 more mid-run
        for _ in range(2):
            await order_db.execute("INSERT INTO orders (status) VALUES ('pending')")

        agent = _agent()
        captured = []

        @agent.watch(
            source=order_db,
            condition="SELECT COUNT(*) FROM orders WHERE status='pending'",
            threshold=lambda count: count > 0,
            interval=ZERO,
        )
        async def on_any(event: WatchEvent):
            captured.append(event)

        await agent._start_watches()
        await _drain(agent, 5)

        # Add more orders — later events should have a non-None previous_value
        for _ in range(4):
            await order_db.execute("INSERT INTO orders (status) VALUES ('pending')")

        await _drain(agent, 20)
        await _stop(agent)

        assert len(captured) >= 1
        # First event starts from None or 0; subsequent ones track previous
        assert captured[0].previous_value in (None, 0, 2)
        if len(captured) > 1:
            assert captured[1].previous_value is not None


# ---------------------------------------------------------------------------
# Scenario 2: asyncio.Queue depth monitor
#
# Models a task queue (Celery, RQ, Redis list). The watch polls qsize()
# and fires when the backlog grows beyond a limit.
# ---------------------------------------------------------------------------


class TestQueueDepthMonitor:
    async def test_fires_when_queue_exceeds_depth(self):
        queue: asyncio.Queue = asyncio.Queue()

        # Fill queue before starting the watch so the first poll triggers
        for i in range(15):
            await queue.put(f"job_{i}")

        agent = _agent()
        alerts = []

        @agent.watch(
            source=object(),  # not used — condition is callable
            condition=queue.qsize,
            threshold=lambda depth: depth > 10,
            interval=ZERO,
        )
        async def on_lag(event: WatchEvent):
            alerts.append(event.value)

        agent._watches[0].source = _polling(queue.qsize)

        await agent._start_watches()
        await _drain(agent)
        await _stop(agent)

        assert len(alerts) > 0
        assert all(v > 10 for v in alerts)

    async def test_no_alert_when_queue_stays_shallow(self):
        queue: asyncio.Queue = asyncio.Queue()

        # Only 3 items — well below threshold of 10
        for i in range(3):
            await queue.put(f"job_{i}")

        agent = _agent()
        alerts = []

        @agent.watch(
            source=object(),
            condition=queue.qsize,
            threshold=lambda depth: depth > 10,
            interval=ZERO,
        )
        async def on_lag(event: WatchEvent):
            alerts.append(event.value)

        agent._watches[0].source = _polling(queue.qsize)

        await agent._start_watches()
        await _drain(agent)
        await _stop(agent)

        assert alerts == []

    async def test_resolves_when_queue_drains(self):
        queue: asyncio.Queue = asyncio.Queue()

        # Start with overloaded queue
        for i in range(12):
            await queue.put(f"job_{i}")

        agent = _agent()
        triggers = []
        resolves = []

        @agent.watch(
            source=object(),
            condition=queue.qsize,
            threshold=lambda depth: depth > 5,
            interval=ZERO,
            on_resolve=True,
        )
        async def on_queue(event: WatchEvent):
            if event.resolved:
                resolves.append(event.value)
            else:
                triggers.append(event.value)

        agent._watches[0].source = _polling(queue.qsize)

        await agent._start_watches()
        await _drain(agent, 10)  # trigger fires

        # Drain the queue to below threshold
        while queue.qsize() > 0:
            queue.get_nowait()
            queue.task_done()

        await _drain(agent, 20)  # resolve fires
        await _stop(agent)

        assert len(triggers) > 0, "Expected trigger events"
        assert len(resolves) > 0, "Expected resolve events"
        assert all(v <= 5 for v in resolves)

    async def test_event_contains_actual_depth(self):
        """Handler receives the real queue depth, not just True/False."""
        queue: asyncio.Queue = asyncio.Queue()
        for i in range(8):
            await queue.put(f"job_{i}")

        agent = _agent()
        received = []

        @agent.watch(
            source=object(),
            condition=queue.qsize,
            threshold=lambda depth: depth > 0,
            interval=ZERO,
        )
        async def capture(event: WatchEvent):
            received.append(event.value)

        agent._watches[0].source = _polling(queue.qsize)

        await agent._start_watches()
        await _drain(agent, 5)
        await _stop(agent)

        assert len(received) > 0
        assert all(isinstance(v, int) for v in received)
        assert received[0] == 8


# ---------------------------------------------------------------------------
# Scenario 3: In-memory error rate monitor
#
# Models polling a stats endpoint or Prometheus gauge. An ErrorRateCounter
# tracks total requests and errors. The watch fires when error rate exceeds
# 10%, and resolves when it drops back below.
# ---------------------------------------------------------------------------


class ErrorRateCounter:
    """Simple in-process error rate tracker (simulates a /metrics endpoint)."""

    def __init__(self):
        self.total = 0
        self.errors = 0

    def record(self, success: bool) -> None:
        self.total += 1
        if not success:
            self.errors += 1

    def error_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.errors / self.total

    def reset(self) -> None:
        self.total = 0
        self.errors = 0


class TestErrorRateMonitor:
    async def test_fires_when_error_rate_spikes(self):
        counter = ErrorRateCounter()

        # 50% error rate — well above 10% threshold
        for _ in range(5):
            counter.record(success=False)
            counter.record(success=True)

        agent = _agent()
        alerts = []

        @agent.watch(
            source=object(),
            condition=counter.error_rate,
            threshold=lambda rate: rate > 0.10,
            interval=ZERO,
        )
        async def on_spike(event: WatchEvent):
            alerts.append(event.value)

        agent._watches[0].source = _polling(counter.error_rate)

        await agent._start_watches()
        await _drain(agent)
        await _stop(agent)

        assert len(alerts) > 0
        assert all(rate > 0.10 for rate in alerts)

    async def test_does_not_fire_when_rate_healthy(self):
        counter = ErrorRateCounter()

        # 2% error rate — below threshold
        counter.record(success=False)
        for _ in range(49):
            counter.record(success=True)

        agent = _agent()
        alerts = []

        @agent.watch(
            source=object(),
            condition=counter.error_rate,
            threshold=lambda rate: rate > 0.10,
            interval=ZERO,
        )
        async def on_spike(event: WatchEvent):
            alerts.append(event.value)

        agent._watches[0].source = _polling(counter.error_rate)

        await agent._start_watches()
        await _drain(agent)
        await _stop(agent)

        assert alerts == []

    async def test_resolves_when_error_rate_recovers(self):
        counter = ErrorRateCounter()

        # Start spiking: 80% error rate
        for _ in range(4):
            counter.record(success=False)
        counter.record(success=True)

        agent = _agent()
        triggers = []
        resolves = []

        @agent.watch(
            source=object(),
            condition=counter.error_rate,
            threshold=lambda rate: rate > 0.10,
            interval=ZERO,
            on_resolve=True,
        )
        async def on_rate(event: WatchEvent):
            if event.resolved:
                resolves.append(event.value)
            else:
                triggers.append(event.value)

        agent._watches[0].source = _polling(counter.error_rate)

        await agent._start_watches()
        await _drain(agent, 10)  # trigger fires

        # Recovery: reset and record only successes
        counter.reset()
        for _ in range(10):
            counter.record(success=True)

        await _drain(agent, 20)  # resolve fires
        await _stop(agent)

        assert len(triggers) > 0, "Expected trigger events during spike"
        assert len(resolves) > 0, "Expected resolve events after recovery"
        assert all(rate <= 0.10 for rate in resolves)

    async def test_event_carries_float_rate_not_bool(self):
        """Handler receives the actual float rate, not just True."""
        counter = ErrorRateCounter()
        for _ in range(3):
            counter.record(success=False)
        counter.record(success=True)

        agent = _agent()
        received = []

        @agent.watch(
            source=object(),
            condition=counter.error_rate,
            threshold=lambda rate: rate > 0.10,
            interval=ZERO,
        )
        async def capture(event: WatchEvent):
            received.append(event.value)

        agent._watches[0].source = _polling(counter.error_rate)

        await agent._start_watches()
        await _drain(agent, 5)
        await _stop(agent)

        assert len(received) > 0
        assert all(isinstance(v, float) for v in received)
        assert all(0.0 < v <= 1.0 for v in received)
