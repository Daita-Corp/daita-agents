"""
Watch system — polling and streaming data source monitoring for agents.

Provides the core types and source implementations for @agent.watch() decorator.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncGenerator, Callable, Awaitable, Literal, Optional, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interval parsing
# ---------------------------------------------------------------------------

_INTERVAL_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([smhd])$")

_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def _parse_interval(s: str) -> timedelta:
    """Parse a human-readable interval string into a timedelta.

    Accepted formats: "30s", "5m", "1h", "2d" (integer or float + unit).

    Raises ValueError for any other input.
    """
    m = _INTERVAL_RE.match(s.strip())
    if not m:
        raise ValueError(
            f"Invalid interval {s!r}. Expected format like '30s', '5m', '1h', '2d'."
        )
    value, unit = float(m.group(1)), m.group(2)
    return timedelta(seconds=value * _UNIT_SECONDS[unit])


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass
class WatchEvent:
    """A single event fired by a watch when its condition activates.

    Fields:
        value: The current value returned by the watch source.
        triggered_at: UTC timestamp of the trigger.
        source_type: Whether the event came from a polling or streaming source.
        resolved: True when the condition has cleared (on_resolve=True only).
        previous_value: The value from the previous poll cycle, if available.
    """

    value: Any
    triggered_at: datetime
    source_type: Literal["polling", "streaming"]
    resolved: bool = False
    previous_value: Any = None


@dataclass
class WatchConfig:
    """Configuration for a single registered watch.

    Created by the @agent.watch() decorator; stored in agent._watches.
    """

    handler: Callable[["WatchEvent"], Awaitable[None]]
    source: Any  # BasePlugin or WatchSource implementation
    name: str  # auto-derived from handler.__name__

    # polling
    condition: Any = None  # SQL string or async callable
    threshold: Optional[Callable[[Any], bool]] = None
    interval: Optional[timedelta] = None
    on_resolve: bool = False

    # streaming (Phase 2 — stored but not used yet)
    topic: Optional[str] = None
    filter: Optional[Callable[[Any], bool]] = None

    # shared
    relay_channel: Optional[str] = None
    on_error: Optional[Callable[[Exception], Awaitable[None]]] = None


@dataclass
class WatchState:
    """Runtime state for a running watch loop."""

    config: "WatchConfig"
    task: asyncio.Task
    status: Literal["running", "stopped", "error"]
    triggered: bool = False  # True while condition is currently active
    last_error: Optional[Exception] = None
    _previous_value: Any = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# WatchSource protocol
# ---------------------------------------------------------------------------


class WatchSource(Protocol):
    """Protocol for watch data sources.

    Implementations: PollingWatchSource (Phase 1), StreamingWatchSource (Phase 2).
    """

    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...

    async def events(self) -> AsyncGenerator[Any, None]: ...


# ---------------------------------------------------------------------------
# PollingWatchSource
# ---------------------------------------------------------------------------


class PollingWatchSource:
    """Wraps a plugin + condition + interval into a polling WatchSource.

    Borrows the plugin reference — never calls plugin.disconnect().
    The plugin may be shared with agent tool calls.
    """

    def __init__(self, plugin: Any, condition: Any, interval: timedelta) -> None:
        self._plugin = plugin
        self._condition = condition
        self.interval = interval

    async def connect(self) -> None:
        """Ensure the underlying plugin is connected (idempotent)."""
        if hasattr(self._plugin, "connect"):
            await self._plugin.connect()

    async def disconnect(self) -> None:
        """Intentionally a no-op — we don't own the plugin connection."""

    async def events(self) -> AsyncGenerator[Any, None]:
        """Yield one value per poll cycle, forever."""
        while True:
            value = await self._evaluate_condition()
            yield value
            await asyncio.sleep(self.interval.total_seconds())

    async def _evaluate_condition(self) -> Any:
        if callable(self._condition) and not isinstance(self._condition, str):
            result = self._condition()
            return await result if asyncio.iscoroutine(result) else result

        # SQL string — use plugin.query()
        result = await self._plugin.query(self._condition)

        # Unwrap common scalar patterns (e.g. SELECT COUNT(*))
        if isinstance(result, dict) and "rows" in result:
            rows = result["rows"]
            if len(rows) == 1 and len(rows[0]) == 1:
                return list(rows[0].values())[0]
            return rows

        return result
