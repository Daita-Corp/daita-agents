"""Monitor create intent extraction for prompt-created DB monitors."""

from __future__ import annotations

import re
from typing import Any, Protocol

from daita.runtime import current_host_runtime_context

from ..models import DbRequest
from ..query_metadata import matching_tables
from .intent import (
    MonitorActionIntent,
    MonitorBudgetIntent,
    MonitorConditionIntent,
    MonitorCreateIntent,
    MonitorDeliveryRequest,
    MonitorDisplayIntent,
    MonitorPolicyIntent,
    MonitorScheduleIntent,
    MonitorTargetIntent,
)
from .prompt_parsing import (
    _action_steps_from_prompt,
    _actions_from_prompt,
    _budgets_from_prompt,
    _policy_from_prompt,
    _schedule_from_prompt,
    _target_resource_from_prompt,
    _trigger_from_prompt,
)
from .types import DbMonitorCommand

_MONITOR_DELIVERY_TEMPLATE = "New rows were observed for the monitored table."
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_SLACK_CHANNEL_RE = re.compile(r"#[A-Za-z0-9_-]+")
_URL_RE = re.compile(r"https?://[^\s,;.]+")
_EVERY_MINUTES_RE = re.compile(r"\bevery\s+(\d{1,3})\s+(?:minutes?|mins?)\b")
_EVERY_HOURS_RE = re.compile(r"\bevery\s+(\d{1,2})\s+(?:hours?|hrs?)\b")
_QUOTED_NAME_RE = re.compile(
    r"\b(?:named|called|name(?:d)?\s+as)\s+['\"]([^'\"]{1,96})['\"]",
    re.IGNORECASE,
)
_HOSTED_IN_APP_DELIVERY_RE = re.compile(
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


class MonitorIntentExtractionStrategy(Protocol):
    def extract(
        self,
        command: DbMonitorCommand,
        request: DbRequest,
        *,
        schema: dict[str, Any] | None = None,
        host_defaults: dict[str, Any] | None = None,
    ) -> MonitorCreateIntent: ...


class DeterministicMonitorIntentExtractor:
    """Deterministically normalize monitor create prompts into typed intents."""

    def extract(
        self,
        command: DbMonitorCommand,
        request: DbRequest,
        *,
        schema: dict[str, Any] | None = None,
        host_defaults: dict[str, Any] | None = None,
    ) -> MonitorCreateIntent:
        if command.kind != "create":
            raise ValueError("monitor intent extraction only supports create commands")
        prompt = command.prompt or request.prompt
        normalized_prompt = " ".join(prompt.strip().split())
        target = _target_intent(
            normalized_prompt,
            source_scope=request.source_scope,
            schema=schema,
        )
        schedule = _schedule_intent(normalized_prompt)
        condition = _condition_intent(normalized_prompt, schedule=schedule)
        actions = _actions_from_prompt(normalized_prompt)
        action_steps = _action_steps_from_prompt(normalized_prompt)
        delivery = _delivery_request(
            normalized_prompt,
            actions=actions,
            action_steps=action_steps,
            delivery_default=(host_defaults or {}).get("delivery_default"),
        )
        explicit_name = _explicit_monitor_name(normalized_prompt)
        confidence = min(
            command.confidence or 0.75,
            target.confidence,
            0.94 if schedule is not None else 0.88,
        )
        diagnostics = {
            "extractor": "deterministic",
            "route": dict(command.diagnostics),
            "target_evidence": list(target.evidence),
            "delivery_explicit": None if delivery is None else delivery.explicit,
        }
        return MonitorCreateIntent(
            target=target,
            condition=condition,
            schedule=schedule
            or MonitorScheduleIntent(
                kind="interval",
                interval_seconds=300,
            ),
            delivery=delivery,
            action=MonitorActionIntent(actions=actions, steps=action_steps),
            display=MonitorDisplayIntent(
                explicit_name=explicit_name,
                description=normalized_prompt,
            ),
            policy=MonitorPolicyIntent(_policy_from_prompt(normalized_prompt)),
            budget=MonitorBudgetIntent(_budgets_from_prompt(normalized_prompt)),
            confidence=confidence,
            diagnostics=diagnostics,
        )


def _target_intent(
    prompt: str,
    *,
    source_scope: tuple[str, ...],
    schema: dict[str, Any] | None,
) -> MonitorTargetIntent:
    raw_target = _target_candidate_from_prompt(prompt) or _target_resource_from_prompt(
        prompt
    )
    evidence = [f"prompt:{raw_target}"] if raw_target else []
    target_name = _resolve_schema_target(raw_target, schema) or raw_target or None
    if not target_name and source_scope:
        target_name = source_scope[0]
        evidence.append("request.source_scope")
    confidence = 0.9 if target_name else 0.25
    if target_name and target_name != raw_target:
        evidence.append(f"schema:{target_name}")
        confidence = 0.96
    scope = tuple(source_scope or ((target_name,) if target_name else ()))
    return MonitorTargetIntent(
        target_type="table",
        name=target_name,
        source_scope=scope,
        confidence=confidence,
        evidence=tuple(evidence),
    )


def _target_candidate_from_prompt(prompt: str) -> str | None:
    lowered = prompt.lower()
    quoted_name = _explicit_monitor_name(prompt)
    if quoted_name:
        quoted_slug = re.escape(quoted_name.lower())
        lowered = re.sub(rf"['\"]{quoted_slug}['\"]", "", lowered)
    for pattern in (
        r"\bwatch\s+(?:the\s+)?([a-z][a-z0-9_ ]{1,80}?)(?:\s+table)?\s+for\s+(?:new|stale|fresh|rows?|records?)\b",
        r"\bfor\s+(?:the\s+)?([a-z][a-z0-9_ ]{1,80}?)(?:\s+table)?\b(?=(?:[\s.,;]+(?:i\s+want|when|if|every|notify|alert|poll|$)|$))",
        r"\bon\s+(?:the\s+)?([a-z][a-z0-9_ ]{1,80}?)(?:\s+table)?\b(?=(?:[\s.,;]+(?:i\s+want|when|if|every|notify|alert|poll|$)|$))",
        r"\b(?:monitor|watch)\s+(?:the\s+)?([a-z][a-z0-9_ ]{1,80}?)(?:\s+table)?\b(?=(?:[\s.,;]+(?:i\s+want|when|if|every|notify|alert|poll|$)|$))",
        r"\btable\s+([a-z][a-z0-9_]*)\b",
    ):
        match = re.search(pattern, lowered)
        if match:
            candidate = _slug_phrase(match.group(1))
            if candidate and candidate.split("_", 1)[0] not in {
                "and",
                "or",
                "when",
                "if",
                "the",
                "every",
                "notify",
                "alert",
                "poll",
            }:
                return candidate
    return None


def _slug_phrase(value: str) -> str:
    words = [word for word in re.split(r"[^a-z0-9]+", value.lower()) if word]
    return "_".join(words)


def _resolve_schema_target(
    raw_target: str | None,
    schema: dict[str, Any] | None,
) -> str | None:
    if not raw_target or not schema:
        return None
    matches = matching_tables(schema, raw_target)
    if len(matches) == 1:
        name = matches[0].get("name")
        return str(name) if name else None
    return None


def _condition_intent(
    prompt: str,
    *,
    schedule: MonitorScheduleIntent | None,
) -> MonitorConditionIntent:
    lowered = prompt.lower()
    if re.search(r"\b(new|inserted|added|created|appear|shows?\s+up)\b", lowered):
        return MonitorConditionIntent(
            kind="new_rows",
            path="rows",
            operator="count_gt",
            value=0,
        )
    threshold = re.search(
        r"\b(?:exceeds?|greater\s+than|above|over)\s+(\d+(?:\.\d+)?)\b",
        lowered,
    )
    if threshold:
        value: int | float
        raw = threshold.group(1)
        value = float(raw) if "." in raw else int(raw)
        return MonitorConditionIntent(
            kind="threshold",
            expression=_trigger_from_prompt(
                prompt,
                schedule=None if schedule is None else schedule.to_schedule_dict(),
            )["expression"],
            operator="gt",
            value=value,
        )
    if re.search(r"\b(stale|freshness|out\s+of\s+date)\b", lowered):
        return MonitorConditionIntent(kind="freshness", operator="truthy", value=True)
    return MonitorConditionIntent(
        kind="rows_present",
        path="rows",
        operator="count_gt",
        value=0,
        expression=_trigger_from_prompt(
            prompt,
            schedule=None if schedule is None else schedule.to_schedule_dict(),
        ).get("expression"),
    )


def _schedule_intent(prompt: str) -> MonitorScheduleIntent | None:
    lowered = prompt.lower()
    minutes = _EVERY_MINUTES_RE.search(lowered)
    if minutes:
        return MonitorScheduleIntent(
            kind="interval",
            interval_seconds=int(minutes.group(1)) * 60,
        )
    hours = _EVERY_HOURS_RE.search(lowered)
    if hours:
        return MonitorScheduleIntent(
            kind="interval",
            interval_seconds=int(hours.group(1)) * 3600,
        )
    schedule = _schedule_from_prompt(prompt)
    if not schedule:
        return None
    expression = str(schedule.get("expression") or "").strip() or None
    timezone = None
    if expression and len(expression.split()) > 5:
        parts = expression.split()
        expression = " ".join(parts[:5])
        timezone = " ".join(parts[5:])
    return MonitorScheduleIntent(
        kind="cron" if expression else "interval",
        expression=expression,
        timezone=timezone,
        interval_seconds=schedule.get("interval_seconds"),
    )


def _delivery_request(
    prompt: str,
    *,
    actions: tuple[str, ...],
    action_steps: tuple[dict[str, Any], ...],
    delivery_default: str | None,
) -> MonitorDeliveryRequest | None:
    lowered = prompt.lower()
    hints = {
        str(step.get("delivery_hint") or "")
        for step in action_steps
        if step.get("kind") == "notify"
    }
    hints = {hint for hint in hints if hint}
    wants_notification = bool(
        hints
        or re.search(
            r"\b(notify|notified|alert|tell me|let me know|email me|"
            r"send me an email)\b",
            lowered,
        )
    )
    if not wants_notification:
        return None
    delivery_kind, target = _default_delivery(delivery_default)
    explicit = bool(hints)
    if "in_app" in hints or _HOSTED_IN_APP_DELIVERY_RE.search(prompt):
        delivery_kind = "in_app"
        target = {"type": "requesting_user"}
        explicit = True
    elif "local" in hints:
        delivery_kind = "local"
        target = {"type": "runtime_console"}
        explicit = True
    elif "email" in hints or "email" in lowered or _EMAIL_RE.search(prompt):
        delivery_kind = "email"
        target = {"type": "email"}
        email = _EMAIL_RE.search(prompt)
        if email:
            target["recipient"] = email.group(0)
        explicit = True
    elif "slack" in hints or "slack" in lowered or _SLACK_CHANNEL_RE.search(prompt):
        delivery_kind = "slack"
        target = {"type": "slack"}
        channel = _SLACK_CHANNEL_RE.search(prompt)
        if channel:
            target["channel"] = channel.group(0)
        explicit = True
    elif "webhook" in hints or "webhook" in lowered or "callback" in lowered:
        delivery_kind = "webhook"
        target = {"type": "webhook"}
        url = _URL_RE.search(prompt)
        if url:
            target["url"] = url.group(0)
        explicit = True
    return MonitorDeliveryRequest(
        delivery_kind=delivery_kind,
        target=target,
        explicit=explicit,
        template=_MONITOR_DELIVERY_TEMPLATE,
        include_observed_rows=True,
    )


def _default_delivery(delivery_default: str | None) -> tuple[str, dict[str, Any]]:
    host_context = current_host_runtime_context()
    default_delivery = delivery_default or (
        host_context.delivery_defaults[0]
        if host_context and host_context.delivery_defaults
        else "local"
    )
    if default_delivery == "in_app":
        return "in_app", {"type": "requesting_user"}
    return "local", {"type": "runtime_console"}


def _explicit_monitor_name(prompt: str) -> str | None:
    match = _QUOTED_NAME_RE.search(prompt)
    if not match:
        return None
    name = " ".join(match.group(1).strip().split())
    return name or None
