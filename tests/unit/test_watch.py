"""
Unit tests for the @agent.watch() polling foundation (Phase 1).

Covers:
- _parse_interval correctness and validation
- WatchEvent and WatchConfig dataclass defaults
- Trigger fires when threshold returns True
- Trigger does NOT fire when threshold returns False
- on_resolve fires when condition clears after being triggered
- Handler exception is caught — watch loop continues
- on_error callback invoked on handler failure
- Invalid config combos raise ValueError at decoration time
- Watches auto-start on first run() without explicit start()
- stop() cancels all watch tasks cleanly
"""

import asyncio
from datetime import timedelta, datetime, timezone

import pytest

from daita.agents.agent import Agent
from daita.core.watch import (
    WatchEvent,
    WatchConfig,
    WatchState,
    PollingWatchSource,
    _parse_interval,
)
from daita.core.streaming import EventType
from daita.llm.mock import MockLLMProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent() -> Agent:
    return Agent(name="watcher", llm_provider=MockLLMProvider(delay=0))


class MockSource:
    """Async generator source that yields a fixed sequence of values then stops."""

    def __init__(self, values: list):
        self._values = values
        self.connected = False
        self.disconnected = False

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.disconnected = True

    async def events(self):
        for v in self._values:
            yield v


class InfiniteSource:
    """Yields the same value forever at zero interval (for cancellation tests)."""

    def __init__(self, value):
        self._value = value
        self.connected = False

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        pass

    async def events(self):
        while True:
            yield self._value
            await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# _parse_interval
# ---------------------------------------------------------------------------


class TestParseInterval:
    def test_seconds(self):
        assert _parse_interval("30s") == timedelta(seconds=30)

    def test_minutes(self):
        assert _parse_interval("5m") == timedelta(minutes=5)

    def test_hours(self):
        assert _parse_interval("1h") == timedelta(hours=1)

    def test_days(self):
        assert _parse_interval("2d") == timedelta(days=2)

    def test_float_value(self):
        assert _parse_interval("0.5m") == timedelta(seconds=30)

    def test_whitespace_stripped(self):
        assert _parse_interval(" 10s ") == timedelta(seconds=10)

    def test_invalid_unit(self):
        with pytest.raises(ValueError):
            _parse_interval("10x")

    def test_invalid_no_unit(self):
        with pytest.raises(ValueError):
            _parse_interval("10")

    def test_empty_string(self):
        with pytest.raises(ValueError):
            _parse_interval("")


# ---------------------------------------------------------------------------
# WatchEvent defaults
# ---------------------------------------------------------------------------


class TestWatchEvent:
    def test_defaults(self):
        ev = WatchEvent(
            value=42,
            triggered_at=datetime.now(timezone.utc),
            source_type="polling",
        )
        assert ev.resolved is False
        assert ev.previous_value is None

    def test_resolved_flag(self):
        ev = WatchEvent(
            value=0,
            triggered_at=datetime.now(timezone.utc),
            source_type="polling",
            resolved=True,
        )
        assert ev.resolved is True


# ---------------------------------------------------------------------------
# watch() decorator validation
# ---------------------------------------------------------------------------


class TestWatchDecoratorValidation:
    def test_on_resolve_without_interval_raises(self):
        agent = _make_agent()
        with pytest.raises(ValueError, match="on_resolve"):

            @agent.watch(source=object(), condition="SELECT 1", on_resolve=True)
            async def handler(event):
                pass

    def test_interval_without_condition_raises(self):
        agent = _make_agent()
        with pytest.raises(ValueError, match="condition="):

            @agent.watch(source=object(), interval="10s")
            async def handler(event):
                pass

    def test_invalid_interval_string_raises(self):
        agent = _make_agent()
        with pytest.raises(ValueError):

            @agent.watch(source=object(), condition="SELECT 1", interval="bad")
            async def handler(event):
                pass

    def test_valid_decoration_registers_watch(self):
        agent = _make_agent()

        @agent.watch(source=object(), condition="SELECT 1", interval="10s")
        async def handler(event):
            pass

        assert len(agent._watches) == 1
        assert agent._watches[0].name == "handler"

    def test_custom_name_used(self):
        agent = _make_agent()

        @agent.watch(source=object(), condition="SELECT 1", interval="10s", name="my_watch")
        async def handler(event):
            pass

        assert agent._watches[0].name == "my_watch"


# ---------------------------------------------------------------------------
# Trigger logic
# ---------------------------------------------------------------------------


class TestWatchTrigger:
    async def test_trigger_fires_when_threshold_true(self):
        agent = _make_agent()
        source = MockSource([150, 50])
        fired = []

        @agent.watch(
            source=source,
            condition=lambda: 150,
            threshold=lambda v: v > 100,
            interval=timedelta(seconds=0),
        )
        async def on_spike(event: WatchEvent):
            fired.append(event.value)

        # Override the source on the registered watch config
        agent._watches[0].source = source

        await agent._start_watches()
        # Allow the loop to run both values
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        # Cancel the watch task so the test can end
        for task in agent._tasks:
            task.cancel()
        await asyncio.gather(*agent._tasks, return_exceptions=True)

        assert 150 in fired

    async def test_trigger_does_not_fire_when_threshold_false(self):
        agent = _make_agent()
        source = MockSource([50])
        fired = []

        @agent.watch(
            source=source,
            condition=lambda: 50,
            threshold=lambda v: v > 100,
            interval=timedelta(seconds=0),
        )
        async def on_spike(event: WatchEvent):
            fired.append(event.value)

        agent._watches[0].source = source

        await agent._start_watches()
        await asyncio.sleep(0)

        for task in agent._tasks:
            task.cancel()
        await asyncio.gather(*agent._tasks, return_exceptions=True)

        assert fired == []

    async def test_no_threshold_always_triggers(self):
        agent = _make_agent()
        source = MockSource(["anything"])
        fired = []

        @agent.watch(
            source=source,
            condition=lambda: "anything",
            interval=timedelta(seconds=0),
        )
        async def on_any(event: WatchEvent):
            fired.append(event.value)

        agent._watches[0].source = source

        await agent._start_watches()
        await asyncio.sleep(0)

        for task in agent._tasks:
            task.cancel()
        await asyncio.gather(*agent._tasks, return_exceptions=True)

        assert len(fired) == 1


# ---------------------------------------------------------------------------
# on_resolve
# ---------------------------------------------------------------------------


class TestWatchOnResolve:
    async def test_on_resolve_fires_when_condition_clears(self):
        agent = _make_agent()
        # First value triggers, second clears
        source = MockSource([150, 50])
        events_seen = []

        @agent.watch(
            source=source,
            condition=lambda: 0,
            threshold=lambda v: v > 100,
            interval=timedelta(seconds=1),  # won't actually wait in mock
            on_resolve=True,
        )
        async def on_spike(event: WatchEvent):
            events_seen.append(event)

        agent._watches[0].source = source

        await agent._start_watches()
        # Give the loop time to process both values from MockSource
        for _ in range(10):
            await asyncio.sleep(0)

        for task in agent._tasks:
            task.cancel()
        await asyncio.gather(*agent._tasks, return_exceptions=True)

        assert any(not e.resolved for e in events_seen), "Should have a trigger event"
        assert any(e.resolved for e in events_seen), "Should have a resolve event"


# ---------------------------------------------------------------------------
# Error isolation
# ---------------------------------------------------------------------------


class TestWatchErrorIsolation:
    async def test_handler_exception_does_not_stop_loop(self):
        agent = _make_agent()
        source = MockSource([1, 2, 3])
        seen = []

        @agent.watch(
            source=source,
            condition=lambda: 0,
            interval=timedelta(seconds=0),
        )
        async def bad_handler(event: WatchEvent):
            seen.append(event.value)
            if event.value == 1:
                raise RuntimeError("handler crash")

        agent._watches[0].source = source

        await agent._start_watches()
        for _ in range(20):
            await asyncio.sleep(0)

        for task in agent._tasks:
            task.cancel()
        await asyncio.gather(*agent._tasks, return_exceptions=True)

        # All three values should have been processed despite the crash on value=1
        assert 2 in seen
        assert 3 in seen

    async def test_on_error_callback_invoked(self):
        agent = _make_agent()
        source = MockSource([99])
        errors_caught = []

        async def my_on_error(exc: Exception):
            errors_caught.append(exc)

        @agent.watch(
            source=source,
            condition=lambda: 0,
            interval=timedelta(seconds=0),
            on_error=my_on_error,
        )
        async def crashing_handler(event: WatchEvent):
            raise ValueError("boom")

        agent._watches[0].source = source

        await agent._start_watches()
        for _ in range(10):
            await asyncio.sleep(0)

        for task in agent._tasks:
            task.cancel()
        await asyncio.gather(*agent._tasks, return_exceptions=True)

        assert len(errors_caught) == 1
        assert isinstance(errors_caught[0], ValueError)


# ---------------------------------------------------------------------------
# Auto-start and lifecycle
# ---------------------------------------------------------------------------


class TestWatchLifecycle:
    async def test_watches_auto_start_on_first_start_watches(self):
        agent = _make_agent()
        source = InfiniteSource(1)

        @agent.watch(
            source=source,
            condition=lambda: 1,
            interval=timedelta(seconds=0),
        )
        async def handler(event):
            pass

        agent._watches[0].source = source

        assert not agent._watches_started
        await agent._start_watches()
        assert agent._watches_started
        assert len(agent._tasks) == 1

        for task in agent._tasks:
            task.cancel()
        await asyncio.gather(*agent._tasks, return_exceptions=True)

    async def test_start_watches_is_idempotent(self):
        agent = _make_agent()
        source = InfiniteSource(1)

        @agent.watch(
            source=source,
            condition=lambda: 1,
            interval=timedelta(seconds=0),
        )
        async def handler(event):
            pass

        agent._watches[0].source = source

        await agent._start_watches()
        await agent._start_watches()  # second call should be a no-op

        assert len(agent._tasks) == 1  # not doubled

        for task in agent._tasks:
            task.cancel()
        await asyncio.gather(*agent._tasks, return_exceptions=True)

    async def test_stop_cancels_watch_tasks(self):
        agent = _make_agent()
        await agent.start()

        source = InfiniteSource(1)

        @agent.watch(
            source=source,
            condition=lambda: 1,
            interval=timedelta(seconds=0),
        )
        async def handler(event):
            pass

        agent._watches[0].source = source
        await agent._start_watches()

        assert any(not t.done() for t in agent._tasks)

        await agent.stop()

        # After stop(), all tasks should be done
        assert all(t.done() for t in agent._tasks)

    async def test_watch_state_status_running(self):
        agent = _make_agent()
        source = InfiniteSource(42)

        @agent.watch(
            source=source,
            condition=lambda: 42,
            interval=timedelta(seconds=0),
        )
        async def handler(event):
            pass

        agent._watches[0].source = source
        await agent._start_watches()
        await asyncio.sleep(0)

        state = agent._watch_states.get("handler")
        assert state is not None
        assert state.status == "running"

        for task in agent._tasks:
            task.cancel()
        await asyncio.gather(*agent._tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# PollingWatchSource unit tests
# ---------------------------------------------------------------------------


class TestPollingWatchSource:
    async def test_callable_condition(self):
        results = []
        plugin = object()

        async def my_condition():
            return 99

        source = PollingWatchSource(plugin=plugin, condition=my_condition, interval=timedelta(seconds=1))
        # Directly call _evaluate_condition
        val = await source._evaluate_condition()
        assert val == 99

    async def test_disconnect_is_noop(self):
        source = PollingWatchSource(plugin=object(), condition="SELECT 1", interval=timedelta(seconds=1))
        # Should not raise
        await source.disconnect()

    async def test_connect_calls_plugin_connect(self):
        connected = []

        class FakePlugin:
            async def connect(self):
                connected.append(True)

        source = PollingWatchSource(
            plugin=FakePlugin(), condition="SELECT 1", interval=timedelta(seconds=1)
        )
        await source.connect()
        assert connected == [True]
