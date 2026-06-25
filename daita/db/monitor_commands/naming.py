"""Semantic monitor display naming and id generation."""

from __future__ import annotations

import re
from typing import Any, Mapping

from .intent import MonitorCreateIntent
from .prompt_parsing import _monitor_id_from_phrase

MAX_MONITOR_DISPLAY_NAME_LENGTH = 64
MAX_MONITOR_ID_LENGTH = 48


def monitor_display_name(intent: MonitorCreateIntent) -> str:
    """Render a bounded display name from normalized monitor semantics."""
    explicit = _clean_display_text(intent.display.explicit_name)
    if explicit:
        return _cap_display_name(explicit)
    suggested = _clean_display_text(intent.display.suggested_name)
    if suggested:
        return _cap_display_name(suggested)
    target = _humanize_identifier(intent.target.name)
    condition = _condition_label(intent.condition.kind)
    if target and condition:
        return _cap_display_name(f"{target} {condition}")
    if target:
        return _cap_display_name(target)
    return "DB Monitor"


def monitor_id(intent: MonitorCreateIntent, *, explicit_id: str | None = None) -> str:
    """Generate a compact stable id from normalized monitor semantics."""
    if explicit_id:
        return _cap_monitor_id(_monitor_id_from_phrase(explicit_id))
    explicit = _clean_display_text(intent.display.explicit_name)
    if explicit:
        return _cap_monitor_id(_monitor_id_from_phrase(explicit))
    name = monitor_display_name(intent)
    if name != "DB Monitor":
        return _cap_monitor_id(_monitor_id_from_phrase(name))
    parts = [
        intent.target.name or "",
        intent.condition.kind or "",
        _schedule_slug_part(intent),
    ]
    return _cap_monitor_id(_monitor_id_from_phrase(" ".join(parts)))


def monitor_display_name_from_proposal(proposal: Mapping[str, Any]) -> str:
    name = _clean_display_text(proposal.get("name"))
    if name:
        return _cap_display_name(name)
    target = _humanize_identifier(proposal.get("target_name"))
    trigger = proposal.get("trigger")
    condition = None
    if isinstance(trigger, Mapping):
        condition = _condition_label(str(trigger.get("type") or ""))
    if target and condition:
        return _cap_display_name(f"{target} {condition}")
    if target:
        return _cap_display_name(target)
    return "DB Monitor"


def _condition_label(kind: str | None) -> str:
    labels = {
        "new_rows": "New Rows",
        "freshness": "Freshness",
        "threshold": "Threshold",
        "anomaly": "Anomaly",
        "report": "Report",
        "rows_present": "",
    }
    return labels.get(str(kind or ""), _humanize_identifier(kind))


def _schedule_slug_part(intent: MonitorCreateIntent) -> str:
    schedule = intent.schedule
    if schedule is None:
        return ""
    if schedule.kind == "interval" and schedule.interval_seconds:
        return f"every {schedule.interval_seconds} seconds"
    if schedule.expression:
        return schedule.expression
    return schedule.kind


def _humanize_identifier(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"[_\-]+", " ", text)
    words = [word for word in re.split(r"\s+", text) if word]
    return " ".join(word[:1].upper() + word[1:] for word in words)


def _clean_display_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return " ".join(text.split())


def _cap_display_name(value: str) -> str:
    text = _clean_display_text(value)
    if len(text) <= MAX_MONITOR_DISPLAY_NAME_LENGTH:
        return text or "DB Monitor"
    clipped = text[:MAX_MONITOR_DISPLAY_NAME_LENGTH].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0].rstrip()
    return clipped or text[:MAX_MONITOR_DISPLAY_NAME_LENGTH].rstrip() or "DB Monitor"


def _cap_monitor_id(value: str) -> str:
    text = value.strip("_") or "db_monitor"
    if len(text) <= MAX_MONITOR_ID_LENGTH:
        return text
    clipped = text[:MAX_MONITOR_ID_LENGTH].rstrip("_")
    if "_" in clipped:
        clipped = clipped.rsplit("_", 1)[0].rstrip("_")
    return clipped or text[:MAX_MONITOR_ID_LENGTH].rstrip("_") or "db_monitor"
