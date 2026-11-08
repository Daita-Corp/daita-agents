"""Plain-words catalog status for the sources screen; times format only here."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from .sanitization import safe_display

_MAX_TODAY_PARTS = 3


def _plural(count: int, noun: str) -> str:
    return f"{count:,} {noun}" if count == 1 else f"{count:,} {noun}s"


def when_text(moment: datetime, now: datetime) -> str:
    """Relative when recent, else clock time today or a day word, in local time."""

    local, today = moment.astimezone(), now.astimezone()
    minutes = int((today - local).total_seconds() // 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes} min ago"
    days = (today.date() - local.date()).days
    if days <= 0:
        return f"{local.hour}:{local.minute:02d}"
    if days == 1:
        return "yesterday"
    if days < 7:
        return local.strftime("%A")
    return f"{local.strftime('%b')} {local.day}"


def day_part(moment: datetime, now: datetime) -> str:
    """Where in the day a catalog is from: "this morning", "yesterday", "Tuesday"."""

    local = moment.astimezone()
    if (now.astimezone().date() - local.date()).days != 0:
        return when_text(moment, now)
    return "this " + (
        "morning" if local.hour < 12 else "afternoon" if local.hour < 18 else "evening"
    )


def status_line(status: Any, now: datetime, noun: str = "object") -> str:
    """The one status line under a source's connection block."""

    state = status.state.value
    if state == "not_checked":
        return "Not checked yet"
    if state == "updating":
        return (
            f"Updating · {status.described:,} of {_plural(status.resources, noun)}"
            " described"
        )
    stamps = [t for t in (status.changed_at, status.full_check_at) if t is not None]
    refreshed = max(stamps, default=None)
    if state == "paused":
        parts = ["Keep current is off"]
        if refreshed is not None:
            parts.append("last refreshed " + when_text(refreshed, now))
    elif state == "unreachable":
        parts = ["Can't reach source"]
        if status.checked_at is not None:
            parts.append("last checked " + when_text(status.checked_at, now))
        if refreshed is not None:
            parts.append("catalog from " + day_part(refreshed, now))
    else:
        parts = ["Up to date"]
        if status.checked_at is not None:
            parts.append("checked " + when_text(status.checked_at, now))
        parts.append(_plural(status.resources, noun))
    return " · ".join(parts)


def _column_change(before: Any, after: Any) -> str:
    old, new = dict(before.columns), dict(after.columns)
    name = safe_display(after.name, fallback="table", maximum=80)
    for column in new:
        if column not in old:
            return f"{name}.{safe_display(column, fallback='column', maximum=80)} added"
    for column in old:
        if column not in new:
            return (
                f"{name}.{safe_display(column, fallback='column', maximum=80)} removed"
            )
    for column, kind in new.items():
        if old[column] != kind:
            return (
                f"{name}.{safe_display(column, fallback='column', maximum=80)}"
                " changed type"
            )
    return f"{name} changed"


def today_line(changes: Sequence[Any], now: datetime, noun: str = "object") -> str:
    """ "Today: 3 tables added in sales · orders.status changed type", or ""."""

    today = now.astimezone().date()
    counts: dict[tuple[str, str], int] = {}
    changed: list[str] = []
    for change in changes:
        if change.changed_at.astimezone().date() != today:
            continue
        if change.kind.value == "changed":
            if change.before is not None and change.after is not None:
                changed.append(_column_change(change.before, change.after))
            continue
        scope = change.native_identity.rpartition(".")[0]
        key = (change.kind.value, safe_display(scope, fallback="", maximum=80))
        counts[key] = counts.get(key, 0) + 1
    parts = [
        f"{_plural(count, noun)} {kind}" + (f" in {scope}" if scope else "")
        for (kind, scope), count in counts.items()
    ] + changed
    if not parts:
        return ""
    shown = " · ".join(parts[:_MAX_TODAY_PARTS])
    if len(parts) > _MAX_TODAY_PARTS:
        shown += f" and {len(parts) - _MAX_TODAY_PARTS} more"
    return "Today: " + shown
