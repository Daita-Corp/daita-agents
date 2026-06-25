"""Prompt parsing helpers for DB monitor commands."""

from __future__ import annotations

import re
from typing import Any

_MONITOR_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,}$")
_CRON_RE = re.compile(r"(?P<cron>(?:\S+\s+){4}\S+(?:\s+[A-Za-z_/-]+)?)")
_EVERY_MINUTES_RE = re.compile(r"\bevery\s+(\d{1,3})\s+minutes?\b")
_EVERY_HOURS_RE = re.compile(r"\bevery\s+(\d{1,2})\s+hours?\b")
_TIME_RE = re.compile(r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b")
_HOSTED_IN_APP_NOTIFICATION_RE = re.compile(
    r"\b(?:"
    r"notify\s+me\s+in\s+(?:the\s+)?app|"
    r"notify\s+me\s+via\s+(?:the\s+)?app|"
    r"notify\s+me\s+inside\s+(?:the\s+)?app|"
    r"notified\s+in\s+(?:the\s+)?app|"
    r"notified\s+via\s+(?:the\s+)?app|"
    r"alert\s+me\s+in\s+(?:the\s+)?app|"
    r"alert\s+me\s+via\s+(?:the\s+)?app|"
    r"send\s+me\s+(?:an?\s+)?app\s+notification|"
    r"in[-\s]?app\s+notification|"
    r"app\s+notification"
    r")\b",
    re.IGNORECASE,
)
_LOCAL_NOTIFICATION_RE = re.compile(
    r"\b(?:"
    r"notify\s+me\s+(?:locally|in\s+(?:the\s+)?console|via\s+(?:the\s+)?console)|"
    r"alert\s+me\s+(?:locally|in\s+(?:the\s+)?console|via\s+(?:the\s+)?console)|"
    r"runtime\s+console|"
    r"local\s+notification"
    r")\b",
    re.IGNORECASE,
)
_WEEKDAY_INDEX = {
    "monday": 1,
    "tuesday": 2,
    "wednesday": 3,
    "thursday": 4,
    "friday": 5,
    "saturday": 6,
    "sunday": 0,
}
_STOP_WORDS = {
    "the",
    "a",
    "an",
    "active",
    "paused",
    "db",
    "database",
    "monitor",
    "monitors",
    "make",
    "update",
    "change",
    "set",
    "require",
    "pause",
    "resume",
    "restart",
    "unpause",
    "delete",
    "remove",
    "inspect",
    "describe",
    "status",
    "of",
    "why",
    "did",
    "please",
}


def _looks_like_monitor_update(lowered: str) -> bool:
    if "monitor" not in lowered:
        return False
    return bool(
        re.search(r"\b(make|update|change|set|require|rename|adjust)\b", lowered)
    )


def _approval_action_from_prompt(lowered: str) -> str | None:
    if re.search(r"\b(approve|authorize)\b", lowered):
        return "approve"
    if re.search(r"\b(reject|deny|decline)\b", lowered):
        return "reject"
    if re.search(r"\b(cancel|withdraw)\b", lowered):
        return "cancel"
    return None


def _has_recurring_or_scheduled_time(lowered: str) -> bool:
    return bool(
        re.search(
            r"\b(every|each|daily|weekly|weekday|weekdays|monday|tuesday|"
            r"wednesday|thursday|friday|saturday|sunday|cron|at\s+\d)",
            lowered,
        )
    )


def _requires_monitor_create_approval(lowered: str) -> bool:
    return bool(re.search(r"\b(create|add|set\s+up|setup)\b.*\bmonitors?\b", lowered))


def _extract_monitor_id(prompt: str) -> str | None:
    lowered = prompt.lower()
    quoted = re.search(r"['\"]([^'\"]+)['\"]", prompt)
    if quoted:
        return _monitor_id_from_phrase(quoted.group(1))
    command_explicit = re.search(
        r"\b(?:inspect|describe|pause|resume|restart|unpause|delete|remove)"
        r"\s+(?:the\s+)?monitor\s+([a-z][a-z0-9_]{1,})\b",
        lowered,
    )
    if command_explicit and _MONITOR_ID_RE.match(command_explicit.group(1)):
        return command_explicit.group(1)
    by_monitor = re.search(
        r"\bby\s+(?:the\s+)?([a-z][a-z0-9_ -]{1,60}?)\s+monitor\b",
        lowered,
    )
    if by_monitor:
        return _monitor_id_from_phrase(by_monitor.group(1))
    possessive = re.search(r"\b([a-z][a-z0-9_ -]{1,60}?)\s+monitor\b", lowered)
    if possessive:
        return _monitor_id_from_phrase(possessive.group(1))
    explicit = re.search(r"\bmonitor\s+([a-z][a-z0-9_]{1,})\b", lowered)
    if explicit and _MONITOR_ID_RE.match(explicit.group(1)):
        return explicit.group(1)
    trailing = re.search(
        r"\b(?:pause|resume|restart|unpause|delete|remove|inspect|describe|status of)"
        r"\s+(?:the\s+)?([a-z][a-z0-9_ -]{1,60})",
        lowered,
    )
    if trailing:
        return _monitor_id_from_phrase(trailing.group(1))
    return None


def _state_patch(kind: str, prompt: str) -> dict[str, Any]:
    if kind != "pause":
        return {}
    paused_until = _paused_until(prompt)
    return {"paused_until": paused_until} if paused_until else {}


def _paused_until(prompt: str) -> str | None:
    match = re.search(r"\buntil\s+(.+)$", prompt, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip().rstrip(".")


def _update_patch(prompt: str) -> dict[str, Any]:
    patch: dict[str, Any] = {
        "metadata": {"last_prompt_update": prompt},
    }
    lowered = prompt.lower()
    if "less noisy" in lowered:
        patch["policy"] = {"noise": "reduced"}
    consecutive = re.search(
        r"\brequire\s+(\d+|two|three|four|five)\s+bad checks?\b", lowered
    )
    if consecutive:
        count = _small_number(consecutive.group(1))
        patch["trigger"] = {
            "type": "condition",
            "expression": f"requires {count} consecutive bad checks",
            "consecutive_matches": count,
        }
    return patch


def _schedule_from_prompt(prompt: str) -> dict[str, Any] | None:
    lowered = prompt.lower()
    cron = _CRON_RE.search(prompt)
    if cron and any(char in cron.group("cron") for char in ("*", "/")):
        return {"expression": cron.group("cron").strip()}
    minutes = _EVERY_MINUTES_RE.search(lowered)
    if minutes:
        return {"expression": f"*/{int(minutes.group(1))} * * * *"}
    hours = _EVERY_HOURS_RE.search(lowered)
    if hours:
        return {"expression": f"0 */{int(hours.group(1))} * * *"}
    if "hourly" in lowered:
        return {"expression": "0 * * * *"}
    if "monday through friday" in lowered or "weekdays" in lowered:
        hour, minute = _time_from_prompt(lowered)
        expression = f"{minute} {hour} * * 1-5"
        if "cst" in lowered or "central" in lowered or "chicago" in lowered:
            expression += " America/Chicago"
        return {"expression": expression}
    weekday = _weekday_from_prompt(lowered)
    if weekday is not None:
        hour, minute = _time_from_prompt(lowered)
        return {"expression": f"{minute} {hour} * * {weekday}"}
    if "daily" in lowered or "every day" in lowered:
        hour, minute = _time_from_prompt(lowered)
        return {"expression": f"{minute} {hour} * * *"}
    return None


def _time_from_prompt(lowered: str) -> tuple[int, int]:
    match = _TIME_RE.search(lowered)
    if not match:
        return (9, 0)
    hour = int(match.group(1))
    minute = int(match.group(2) or "0")
    if match.group(3) == "pm" and hour != 12:
        hour += 12
    if match.group(3) == "am" and hour == 12:
        hour = 0
    return (hour, minute)


def _weekday_from_prompt(lowered: str) -> int | None:
    for weekday, index in _WEEKDAY_INDEX.items():
        if weekday in lowered:
            return index
    return None


def _trigger_from_prompt(
    prompt: str,
    *,
    schedule: dict[str, Any] | None,
) -> dict[str, Any]:
    match = re.search(r"(?i)\bif\s+(.+?)(?:\bthen\b|$)", prompt)
    if not match:
        match = re.search(r"(?i)\bwhen\s+(.+?)(?:\bthen\b|$)", prompt)
    if match:
        return {
            "type": "condition",
            "expression": match.group(1).strip(" ,;."),
        }
    if schedule is not None:
        return {"type": "schedule", "expression": "always on schedule"}
    return {"type": "manual", "expression": "manual tick"}


def _actions_from_prompt(prompt: str) -> tuple[str, ...]:
    match = re.search(r"(?i)\bthen\b\s+(.+)$", prompt)
    action_text = match.group(1) if match else ""
    if not action_text:
        notify = re.search(r"(?i)\bnotify\s+([^.;]+)", prompt)
        if notify:
            action_text = f"notify {notify.group(1)}"
    if not action_text:
        alert = re.search(r"(?i)\balert\s+([^.;]+)", prompt)
        if alert:
            action_text = f"alert {alert.group(1)}"
    if not action_text and "report" in prompt.lower():
        action_text = "generate the requested report"
    if not action_text:
        return ()
    parts = re.split(r"(?i),\s*|\s+and\s+", action_text.strip(" ."))
    return tuple(part.strip(" .") for part in parts if part.strip(" ."))


def _hosted_in_app_notification_requested(prompt: str) -> bool:
    return bool(_HOSTED_IN_APP_NOTIFICATION_RE.search(prompt))


def _action_steps_from_prompt(prompt: str) -> tuple[dict[str, Any], ...]:
    steps: list[dict[str, Any]] = []
    for action in _actions_from_prompt(prompt):
        lowered = action.lower()
        if (
            lowered.startswith("notify ")
            or lowered.startswith("alert ")
            or _hosted_in_app_notification_requested(action)
        ):
            if lowered.startswith("notify "):
                target = action[len("notify ") :].strip()
            elif lowered.startswith("alert "):
                target = action[len("alert ") :].strip()
            else:
                target = "requesting_user"
            step: dict[str, Any] = {
                "kind": "notify",
                "target_hint": target,
                "instruction": action,
            }
            if _hosted_in_app_notification_requested(action):
                step["delivery_hint"] = "in_app"
            elif target.startswith("#") or "slack" in lowered or "channel" in lowered:
                step["delivery_hint"] = "slack"
            elif "email" in lowered or "@" in target:
                step["delivery_hint"] = "email"
            elif "webhook" in lowered:
                step["delivery_hint"] = "webhook"
            elif _LOCAL_NOTIFICATION_RE.search(action):
                step["delivery_hint"] = "local"
            if action:
                step["matched_text"] = action
            steps.append(step)
        elif _hosted_in_app_notification_requested(action):
            steps.append(
                {
                    "kind": "notify",
                    "target_hint": "me in app",
                    "delivery_hint": "in_app",
                    "instruction": action,
                    "matched_text": action,
                }
            )
        elif "report" in lowered:
            steps.append({"kind": "report_generate", "instruction": action})
        elif "approval" in lowered or "approve" in lowered:
            steps.append(
                {
                    "kind": "approval_prepare",
                    "instruction": action,
                    "required_capability": "approval_or_write_action",
                }
            )
        else:
            steps.append({"kind": "instruction", "instruction": action})
    if _hosted_in_app_notification_requested(prompt) and not any(
        step.get("delivery_hint") == "in_app" for step in steps
    ):
        steps.append(
            {
                "kind": "notify",
                "target_hint": "me in app",
                "delivery_hint": "in_app",
                "instruction": "Notify me in app",
                "matched_text": "in-app notification",
            }
        )
    return tuple(steps)


def _budgets_from_prompt(prompt: str) -> dict[str, Any]:
    lowered = prompt.lower()
    rows = re.search(r"\bmax(?:imum)?\s+(\d+)\s+rows?\b", lowered)
    if rows:
        return {"max_rows_per_tick": int(rows.group(1))}
    return {}


def _policy_from_prompt(prompt: str) -> dict[str, Any]:
    lowered = prompt.lower()
    policy: dict[str, Any] = {}
    if "approval" in lowered or "approve" in lowered:
        policy["requires_approval"] = True
    if "read-only" in lowered or "readonly" in lowered:
        policy["access"] = "read"
    return policy


def _target_resource_from_prompt(prompt: str) -> str:
    lowered = prompt.lower()
    for pattern in (
        r"\b(?:for|on|against)\s+(?:the\s+)?([a-z][a-z0-9_]*)\s+table\b",
        r"\b(?:monitor|watch)\s+(?:the\s+)?([a-z][a-z0-9_]*)\s+table\b",
        r"\btable\s+([a-z][a-z0-9_]*)\b",
        r"\bfrom\s+(?:the\s+)?([a-z][a-z0-9_]*)\b",
    ):
        match = re.search(pattern, lowered)
        if match:
            return match.group(1)
    return ""


def _monitor_id_from_phrase(phrase: str) -> str:
    words = [
        word
        for word in re.split(r"[^a-z0-9]+", phrase.lower())
        if word and word not in _STOP_WORDS
    ]
    return "_".join(words) or "db_monitor"


def _small_number(value: str) -> int:
    numbers = {"two": 2, "three": 3, "four": 4, "five": 5}
    return numbers.get(value, int(value) if value.isdigit() else 1)
