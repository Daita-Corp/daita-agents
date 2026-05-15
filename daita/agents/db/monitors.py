"""
Structured monitor definitions and local watch registration for ``from_db()``.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional

if TYPE_CHECKING:
    from ..agent import Agent
    from ...core.watch import WatchEvent


VALID_MONITOR_TYPES = {"freshness", "row_count"}
VALID_SEVERITIES = {"info", "warning", "critical"}


@dataclass(frozen=True)
class MonitorDefinition:
    """Portable, JSON-serializable DB monitor definition."""

    name: str
    type: str
    severity: str
    entity: Dict[str, Any]
    sql: str
    threshold: Dict[str, Any]
    interval: str = "1h"
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "severity": self.severity,
            "entity": dict(self.entity),
            "sql": self.sql,
            "threshold": dict(self.threshold),
            "interval": self.interval,
            "evidence": dict(self.evidence),
        }


def normalize_monitor_definition(raw: Any) -> MonitorDefinition:
    """Validate and normalize a monitor definition."""
    if isinstance(raw, MonitorDefinition):
        monitor = raw
    elif isinstance(raw, dict):
        monitor = MonitorDefinition(
            name=str(raw.get("name") or "").strip(),
            type=str(raw.get("type") or "").strip(),
            severity=str(raw.get("severity") or "warning").strip(),
            entity=dict(raw.get("entity") or {}),
            sql=str(raw.get("sql") or "").strip(),
            threshold=dict(raw.get("threshold") or {}),
            interval=str(raw.get("interval") or "1h").strip(),
            evidence=dict(raw.get("evidence") or {}),
        )
    else:
        raise TypeError(
            "Monitor definitions must be dictionaries or MonitorDefinition instances"
        )

    if not monitor.name:
        raise ValueError("Monitor definition requires a name")
    if monitor.type not in VALID_MONITOR_TYPES:
        raise ValueError(
            f"Unsupported monitor type {monitor.type!r}; expected one of "
            f"{sorted(VALID_MONITOR_TYPES)}"
        )
    if monitor.severity not in VALID_SEVERITIES:
        raise ValueError(
            f"Unsupported monitor severity {monitor.severity!r}; expected one of "
            f"{sorted(VALID_SEVERITIES)}"
        )
    if not monitor.sql:
        raise ValueError("Monitor definition requires sql")
    if not monitor.threshold:
        raise ValueError("Monitor definition requires threshold")
    if "table" not in monitor.entity:
        raise ValueError("Monitor definition entity requires a table")

    return monitor


def normalize_monitor_definitions(raw: Iterable[Any]) -> List[MonitorDefinition]:
    """Normalize a collection of monitor definitions."""
    return [normalize_monitor_definition(item) for item in raw]


def register_local_monitors(
    agent: "Agent",
    monitors: Iterable[Any],
    *,
    handler: Optional[Callable[["WatchEvent", Dict[str, Any]], Any]] = None,
    name_prefix: str = "db",
) -> List[Dict[str, Any]]:
    """Register DB monitor definitions as local polling watches.

    Returns the normalized monitor dictionaries with an added ``watch_name`` field.
    """
    plugin = getattr(agent, "_db_plugin", None)
    if plugin is None:
        raise ValueError("Local DB monitors require an agent created by from_db()")

    registered: List[Dict[str, Any]] = []
    used_names = {
        getattr(watch, "name", "") for watch in getattr(agent, "_watches", [])
    }
    for monitor in normalize_monitor_definitions(monitors):
        threshold = _build_threshold(monitor)
        watch_name = _unique_watch_name(
            _watch_name(name_prefix, monitor.name), used_names
        )
        used_names.add(watch_name)

        async def _default_handler(
            event: "WatchEvent", *, _monitor: MonitorDefinition = monitor
        ) -> None:
            await _record_monitor_event(agent, event, _monitor)

        async def _custom_handler(
            event: "WatchEvent", *, _monitor: MonitorDefinition = monitor
        ) -> None:
            await _record_monitor_event(agent, event, _monitor)
            result = handler(event, _monitor.to_dict())  # type: ignore[misc]
            if hasattr(result, "__await__"):
                await result

        agent.watch(
            source=plugin,
            condition=monitor.sql,
            threshold=threshold,
            interval=monitor.interval,
            on_resolve=True,
            cooldown=True,
            name=watch_name,
        )(_custom_handler if handler is not None else _default_handler)

        item = monitor.to_dict()
        item["watch_name"] = watch_name
        registered.append(item)

    return registered


def _build_threshold(monitor: MonitorDefinition) -> Callable[[Any], bool]:
    if monitor.type == "freshness":
        max_age_hours = float(monitor.threshold.get("max_age_hours", 24))
        return lambda value: _is_stale(value, max_age_hours)

    if "min_rows" in monitor.threshold:
        min_rows = int(monitor.threshold["min_rows"])
        return lambda value: _to_number(value) < min_rows

    if "change_pct" in monitor.threshold:
        change_pct = float(monitor.threshold["change_pct"])
        baseline: Dict[str, Optional[float]] = {"value": None}

        def threshold(value: Any) -> bool:
            current = _to_number(value)
            if baseline["value"] is None:
                baseline["value"] = current
                return False
            if baseline["value"] == 0:
                return current != 0
            return (
                abs(current - baseline["value"]) / abs(baseline["value"]) * 100
                > change_pct
            )

        return threshold

    raise ValueError(
        f"Monitor {monitor.name!r} has unsupported threshold {monitor.threshold!r}"
    )


def _is_stale(value: Any, max_age_hours: float) -> bool:
    timestamp = _to_datetime(value)
    if timestamp is None:
        return True
    age_hours = (datetime.now(timezone.utc) - timestamp).total_seconds() / 3600
    return age_hours > max_age_hours


def _to_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, str):
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None

    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _to_number(value: Any) -> float:
    if value is None:
        return 0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0


async def _record_monitor_event(
    agent: "Agent", event: "WatchEvent", monitor: MonitorDefinition
) -> None:
    from .findings import record_monitor_finding

    finding = record_monitor_finding(agent, event, monitor)
    entries = getattr(agent, "_db_monitor_events", None)
    if entries is None:
        entries = []
        agent._db_monitor_events = entries
    entries.append(
        {
            "monitor": monitor.to_dict(),
            "value": event.value,
            "previous_value": event.previous_value,
            "resolved": event.resolved,
            "triggered_at": event.triggered_at.isoformat(),
            "finding_id": finding["id"],
        }
    )


def _watch_name(prefix: str, monitor_name: str) -> str:
    slug = "".join(
        ch.lower() if ch.isalnum() else "_" for ch in monitor_name.strip()
    ).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return f"{prefix}:{slug or 'monitor'}"


def _unique_watch_name(base: str, used_names: set[str]) -> str:
    if base not in used_names:
        return base
    index = 2
    while f"{base}_{index}" in used_names:
        index += 1
    return f"{base}_{index}"
