"""Small, code-owned interpretations of user-local scheduling time."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def valid_iana_timezone(value: str) -> ZoneInfo:
    if not isinstance(value, str) or "/" not in value or len(value) > 128:
        raise ValueError("an exact IANA timezone is required")
    try:
        return ZoneInfo(value)
    except ZoneInfoNotFoundError as error:
        raise ValueError("the IANA timezone is unavailable") from error


def system_user_timezone() -> str | None:
    """Best-effort local TUI timezone; never invent an IANA zone from an offset."""

    candidates = [os.environ.get("TZ", "")]
    for path in (Path("/etc/localtime"), Path("/var/db/timezone/localtime")):
        try:
            resolved = str(path.resolve(strict=True))
        except (OSError, RuntimeError):
            continue
        marker = "/zoneinfo/"
        if marker in resolved:
            candidates.append(resolved.rsplit(marker, 1)[1])
    for candidate in candidates:
        if candidate == "UTC":
            candidate = "Etc/UTC"
        try:
            valid_iana_timezone(candidate)
        except ValueError:
            continue
        return candidate
    return None


def next_weekday_utc(
    *,
    now: datetime,
    timezone: str,
    weekday: int,
    hour: int,
    minute: int,
) -> datetime:
    """Resolve the next named local weekday; reject DST gaps and overlaps."""

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("a timezone-aware current instant is required")
    zone = valid_iana_timezone(timezone)
    if not 1 <= weekday <= 7 or not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError("weekday or local time is outside its bound")
    local_now = now.astimezone(zone)
    days = (weekday - local_now.isoweekday()) % 7
    candidate_date = local_now.date() + timedelta(days=days)
    candidate = datetime.combine(candidate_date, datetime.min.time()).replace(
        hour=hour, minute=minute
    )
    if days == 0 and candidate <= local_now.replace(tzinfo=None):
        candidate += timedelta(days=7)
    choices = set()
    for fold in (0, 1):
        aware = candidate.replace(tzinfo=zone, fold=fold)
        utc = aware.astimezone(UTC)
        if utc.astimezone(zone).replace(tzinfo=None) == candidate:
            choices.add(utc)
    if len(choices) != 1:
        raise ValueError("the requested local time is nonexistent or ambiguous")
    return choices.pop()


__all__ = ["next_weekday_utc", "system_user_timezone", "valid_iana_timezone"]
