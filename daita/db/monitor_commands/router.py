"""Prompt-level routing for DB monitor management commands."""

from __future__ import annotations

import re

from .prompt_parsing import (
    _approval_action_from_prompt,
    _create_name_phrase,
    _extract_monitor_id,
    _has_recurring_or_scheduled_time,
    _looks_like_monitor_update,
    _monitor_id_from_phrase,
    _requires_monitor_create_approval,
    _state_patch,
    _update_patch,
)
from .types import DbMonitorCommand


class DbCommandRouter:
    """Conservatively route prompt-level monitor management commands."""

    def route(self, prompt: str) -> DbMonitorCommand | None:
        text = " ".join(prompt.strip().split())
        if not text:
            return None
        lowered = text.lower()

        command = self._route_non_create(text, lowered)
        if command is not None:
            return command
        if self._looks_like_monitor_create(lowered):
            monitor_id = _monitor_id_from_phrase(_create_name_phrase(text))
            return DbMonitorCommand(
                kind="create",
                monitor_id=monitor_id,
                prompt=text,
                confidence=0.88,
                diagnostics={
                    "route": "monitor.create",
                    "approval_required": _requires_monitor_create_approval(lowered),
                },
            )
        return None

    def _route_non_create(
        self,
        text: str,
        lowered: str,
    ) -> DbMonitorCommand | None:
        approval_action = _approval_action_from_prompt(lowered)
        if approval_action is not None and "monitor" in lowered:
            return DbMonitorCommand(
                kind="approve_action",
                monitor_id=_extract_monitor_id(text),
                patch={"approval_action": approval_action},
                prompt=text,
                confidence=0.76,
                diagnostics={"route": "monitor.approve_action"},
            )
        if re.search(r"\bwhy\b.*\bmonitor\b.*\b(trigger|ran|run)\b", lowered):
            return DbMonitorCommand(
                kind="explain_run",
                monitor_id=_extract_monitor_id(text),
                prompt=text,
                confidence=0.8,
                diagnostics={"route": "monitor.explain_run"},
            )
        if re.search(r"\b(list|show)\b.*\bmonitors?\b", lowered):
            status = None
            if "active monitor" in lowered:
                status = "active"
            elif "paused monitor" in lowered:
                status = "paused"
            return DbMonitorCommand(
                kind="list",
                patch={"status": status} if status else {},
                prompt=text,
                confidence=0.92,
                diagnostics={"route": "monitor.list"},
            )
        for kind, verbs in (
            ("pause", ("pause", "stop")),
            ("resume", ("resume", "restart", "unpause")),
            ("delete", ("delete", "remove")),
        ):
            if any(re.search(rf"\b{verb}\b", lowered) for verb in verbs):
                if "monitor" not in lowered:
                    continue
                return DbMonitorCommand(
                    kind=kind,  # type: ignore[arg-type]
                    monitor_id=_extract_monitor_id(text),
                    patch=_state_patch(kind, text),
                    prompt=text,
                    confidence=0.9,
                    diagnostics={"route": f"monitor.{kind}"},
                )
        if re.search(r"\b(inspect|describe|status|show)\b.*\bmonitor\b", lowered):
            return DbMonitorCommand(
                kind="inspect",
                monitor_id=_extract_monitor_id(text),
                prompt=text,
                confidence=0.86,
                diagnostics={"route": "monitor.inspect"},
            )
        if _looks_like_monitor_update(lowered):
            return DbMonitorCommand(
                kind="update",
                monitor_id=_extract_monitor_id(text),
                patch=_update_patch(text),
                prompt=text,
                confidence=0.82,
                diagnostics={"route": "monitor.update"},
            )
        return None

    def _looks_like_monitor_create(self, lowered: str) -> bool:
        if lowered.startswith(("monitor ", "watch ")):
            return bool(
                re.search(
                    r"\b(every|hourly|daily|weekly|if|when|alert|notify|schedule)\b",
                    lowered,
                )
            )
        if re.search(r"\b(create|add|set\s+up|setup)\b.*\bmonitors?\b", lowered):
            return bool(
                re.search(
                    r"\b(every|hourly|daily|weekly|if|when|alert|notify|notified|watch|table|new|added|created|inserted|schedule)\b",
                    lowered,
                )
            )
        if " report " in f" {lowered} " and _has_recurring_or_scheduled_time(lowered):
            return True
        return False
