"""Prompt routing and planning for DB monitor management commands."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Literal

from daita.plugins import ExtensionRegistry
from daita.runtime import OperationStatus

from .models import DbOperationResult, DbRequest
from .monitors import DbMonitor

DbMonitorCommandKind = Literal[
    "create",
    "list",
    "inspect",
    "update",
    "pause",
    "resume",
    "delete",
    "explain_run",
    "approve_action",
]

_MONITOR_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,}$")
_CRON_RE = re.compile(r"(?P<cron>(?:\S+\s+){4}\S+(?:\s+[A-Za-z_/-]+)?)")
_EVERY_MINUTES_RE = re.compile(r"\bevery\s+(\d{1,3})\s+minutes?\b")
_EVERY_HOURS_RE = re.compile(r"\bevery\s+(\d{1,2})\s+hours?\b")
_TIME_RE = re.compile(r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b")
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


@dataclass(frozen=True)
class DbMonitorCommand:
    """Typed prompt-level monitor command.

    The command is intentionally a control-plane route. It carries only the
    monitor CRUD target and planner diagnostics; it does not contain SQL,
    runtime tasks, evidence plans, or governance decisions.
    """

    kind: DbMonitorCommandKind
    monitor_id: str | None = None
    patch: dict[str, Any] = field(default_factory=dict)
    prompt: str = ""
    confidence: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "patch", dict(self.patch))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


@dataclass(frozen=True)
class DbMonitorValidation:
    """Machine-readable validation result for a planned monitor."""

    accepted: bool
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    missing_capabilities: tuple[str, ...] = ()
    unsupported_actions: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "errors", tuple(self.errors))
        object.__setattr__(
            self, "required_capabilities", tuple(self.required_capabilities)
        )
        object.__setattr__(
            self, "missing_capabilities", tuple(self.missing_capabilities)
        )
        object.__setattr__(self, "unsupported_actions", tuple(self.unsupported_actions))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "required_capabilities": list(self.required_capabilities),
            "missing_capabilities": list(self.missing_capabilities),
            "unsupported_actions": list(self.unsupported_actions),
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DbMonitorValidation":
        values = dict(data)
        return cls(
            accepted=bool(values.get("accepted")),
            warnings=tuple(values.get("warnings") or ()),
            errors=tuple(values.get("errors") or ()),
            required_capabilities=tuple(values.get("required_capabilities") or ()),
            missing_capabilities=tuple(values.get("missing_capabilities") or ()),
            unsupported_actions=tuple(values.get("unsupported_actions") or ()),
            diagnostics=dict(values.get("diagnostics") or {}),
        )


@dataclass(frozen=True)
class DbMonitorResolution:
    """Store-aware monitor reference resolution result."""

    monitor: DbMonitor | None
    monitor_ref: str | None
    matches: tuple[DbMonitor, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "matches", tuple(self.matches))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "errors", tuple(self.errors))

    @property
    def accepted(self) -> bool:
        return self.monitor is not None and not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "monitor_id": None if self.monitor is None else self.monitor.id,
            "monitor_ref": self.monitor_ref,
            "matches": [monitor.id for monitor in self.matches],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


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
                diagnostics={"route": "monitor.create"},
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
        if " report " in f" {lowered} " and _has_recurring_or_scheduled_time(lowered):
            return True
        return False


class DbMonitorPlanner:
    """Parse a monitor-management prompt into a narrow `DbMonitor` spec."""

    def __init__(
        self,
        *,
        registry: ExtensionRegistry | None = None,
        limits: dict[str, Any] | None = None,
    ) -> None:
        self.registry = registry
        self.limits = dict(limits or {})

    def create_monitor(
        self,
        command: DbMonitorCommand,
        *,
        source_scope: tuple[str, ...] = (),
        owner: dict[str, Any] | None = None,
    ) -> DbMonitor:
        if command.kind != "create":
            raise ValueError("DbMonitorPlanner can only create monitor specs")
        prompt = command.prompt
        name = _title_from_phrase(_create_name_phrase(prompt))
        schedule = _schedule_from_prompt(prompt)
        trigger = _trigger_from_prompt(prompt, schedule=schedule)
        actions = _actions_from_prompt(prompt)
        watch = _watch_from_prompt(prompt)
        budgets = _budgets_from_prompt(prompt)
        policy = _policy_from_prompt(prompt)
        action_steps = _action_steps_from_prompt(prompt)
        validation = self.validate(
            action_steps=action_steps,
            actions=actions,
            source_scope=source_scope,
            policy=policy,
            budgets=budgets,
        )
        return DbMonitor(
            id=command.monitor_id or _monitor_id_from_phrase(name),
            name=name,
            description=watch,
            status="active",
            source_scope=source_scope,
            schedule=schedule,
            trigger=trigger,
            observation_plan={"watch": [watch] if watch else []},
            action_plan={"steps": [dict(step) for step in action_steps]},
            policy=policy,
            budgets=budgets,
            owner=dict(owner or {}),
            metadata={
                "created_from_prompt": True,
                "prompt": prompt,
                "command": command.diagnostics,
                "validation": validation.to_dict(),
            },
        )

    def validate(
        self,
        *,
        action_steps: tuple[dict[str, Any], ...],
        actions: tuple[str, ...],
        source_scope: tuple[str, ...],
        policy: dict[str, Any],
        budgets: dict[str, Any],
    ) -> DbMonitorValidation:
        capability_ids = (
            {capability.id for capability in self.registry.capabilities}
            if self.registry is not None
            else set()
        )
        required: list[str] = []
        unsupported: list[str] = []
        for step in action_steps:
            capability_id = step.get("required_capability")
            if isinstance(capability_id, str) and capability_id:
                required.append(capability_id)
                if capability_id not in capability_ids:
                    unsupported.append(str(step.get("instruction") or step["kind"]))
        if policy.get("requires_approval") and not _has_any(
            capability_ids, ("approval", "write", "sql")
        ):
            required.append("approval_or_write_action")
            unsupported.append("approval_or_write_action")
        max_rows = budgets.get("max_rows_per_tick")
        limit_max_rows = self.limits.get("max_rows")
        budget_warnings = []
        if isinstance(max_rows, int) and isinstance(limit_max_rows, int):
            if max_rows > limit_max_rows:
                budget_warnings.append("max_rows_per_tick_exceeds_runtime_limit")
        policy_warnings = []
        if policy.get("access") not in {None, "read"}:
            policy_warnings.append("monitor_policy_access_requires_runtime_review")
        missing = tuple(
            capability_id
            for capability_id in dict.fromkeys(required)
            if capability_id not in capability_ids
        )
        errors = tuple(
            f"missing_capability:{capability_id}" for capability_id in missing
        )
        warnings = tuple((*budget_warnings, *policy_warnings))
        return DbMonitorValidation(
            accepted=not errors,
            warnings=warnings,
            errors=errors,
            required_capabilities=tuple(dict.fromkeys(required)),
            missing_capabilities=missing,
            unsupported_actions=tuple(unsupported),
            diagnostics={
                "capability_ids": sorted(capability_ids),
                "source_scope": list(source_scope),
                "connector_guardrails": {
                    "sql_planning": "deferred_to_db_runtime",
                    "execution": "deferred_to_runtime_capabilities",
                },
                "runtime_limits": dict(self.limits),
            },
        )


class DbMonitorResolver:
    """Resolve prompt monitor references against persisted monitor records."""

    def resolve(
        self,
        command: DbMonitorCommand,
        monitors: tuple[DbMonitor, ...],
    ) -> DbMonitorResolution:
        monitor_ref = command.monitor_id
        if not monitor_ref:
            return DbMonitorResolution(
                monitor=None,
                monitor_ref=monitor_ref,
                errors=("monitor_reference_required",),
            )
        normalized = _monitor_id_from_phrase(monitor_ref)
        exact = [monitor for monitor in monitors if monitor.id == monitor_ref]
        if exact:
            return DbMonitorResolution(exact[0], monitor_ref, tuple(exact))
        normalized_id_matches = [
            monitor for monitor in monitors if monitor.id == normalized
        ]
        if normalized_id_matches:
            return DbMonitorResolution(
                normalized_id_matches[0], monitor_ref, tuple(normalized_id_matches)
            )
        name_matches = [
            monitor
            for monitor in monitors
            if _monitor_id_from_phrase(monitor.name) == normalized
        ]
        if len(name_matches) == 1:
            return DbMonitorResolution(
                name_matches[0], monitor_ref, tuple(name_matches)
            )
        substring_matches = [
            monitor
            for monitor in monitors
            if normalized
            and (
                normalized in monitor.id
                or normalized in _monitor_id_from_phrase(monitor.name)
            )
        ]
        if len(substring_matches) == 1:
            return DbMonitorResolution(
                substring_matches[0],
                monitor_ref,
                tuple(substring_matches),
                warnings=("monitor_reference_resolved_by_partial_match",),
            )
        if len((*name_matches, *substring_matches)) > 1:
            matches = tuple(
                {
                    monitor.id: monitor
                    for monitor in (*name_matches, *substring_matches)
                }.values()
            )
            return DbMonitorResolution(
                monitor=None,
                monitor_ref=monitor_ref,
                matches=matches,
                errors=("monitor_reference_ambiguous",),
            )
        return DbMonitorResolution(
            monitor=None,
            monitor_ref=monitor_ref,
            errors=("monitor_not_found",),
        )


class DbMonitorCommandService:
    """Execute prompt-managed monitor commands through the DB runtime control plane."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.router = DbCommandRouter()
        self.resolver = DbMonitorResolver()

    async def run(self, request: DbRequest) -> DbOperationResult | None:
        command = self.router.route(request.prompt)
        if command is None:
            return None
        try:
            return await self._run_command(command, request)
        except ValueError as exc:
            return await self.runtime.record_monitor_command_result(
                request=request,
                kind=command.kind,
                command=_command_payload(command),
                status=OperationStatus.FAILED,
                answer=str(exc),
                evidence_kind="monitor.command.error",
                payload={
                    "command": _command_payload(command),
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
                warnings=("db_monitor_command_failed",),
            )

    async def _run_command(
        self,
        command: DbMonitorCommand,
        request: DbRequest,
    ) -> DbOperationResult:
        if command.kind == "create":
            return await self._create(command, request)
        if command.kind == "list":
            return await self._list(command, request)
        if command.kind in {"inspect", "explain_run"}:
            return await self._inspect(command, request)
        if command.kind == "update":
            return await self._update(command, request)
        if command.kind == "pause":
            return await self._pause(command, request)
        if command.kind == "resume":
            return await self._resume(command, request)
        if command.kind == "delete":
            return await self._delete(command, request)
        if command.kind == "approve_action":
            return await self._approve_action(command, request)
        raise AssertionError(f"unsupported monitor command kind: {command.kind}")

    async def _create(
        self,
        command: DbMonitorCommand,
        request: DbRequest,
    ) -> DbOperationResult:
        planner = DbMonitorPlanner(
            registry=self.runtime.registry,
            limits=self.runtime.config.limits.to_dict(),
        )
        monitor = planner.create_monitor(
            command,
            source_scope=request.source_scope,
            owner=_owner_from_request(request),
        )
        validation = DbMonitorValidation.from_dict(monitor.metadata["validation"])
        if not validation.accepted:
            return await self.runtime.record_monitor_command_result(
                request=request,
                kind=command.kind,
                command=_command_payload(command),
                status=OperationStatus.BLOCKED,
                answer=_monitor_validation_answer(validation),
                evidence_kind="monitor.command.validation",
                payload={
                    "command": _command_payload(command),
                    "planned_monitor": monitor.to_dict(),
                    "validation": validation.to_dict(),
                },
                warnings=("db_monitor_validation_failed", *validation.warnings),
                diagnostics={"validation": validation.to_dict()},
            )
        committed = await self.runtime.create_monitor(monitor)
        inspection = await self.runtime.inspect_monitor(committed.id)
        operation_id = _inspection_operation_id(inspection)
        evidence = await _operation_evidence(self.runtime, operation_id)
        return await self.runtime.record_monitor_command_result(
            request=request,
            kind=command.kind,
            command=_command_payload(command),
            status=OperationStatus.SUCCEEDED,
            answer=_create_monitor_answer(committed),
            operation_id=operation_id,
            evidence=tuple(evidence),
            warnings=validation.warnings,
            diagnostics={
                "monitor": committed.to_dict(),
                "inspection": None if inspection is None else inspection.to_dict(),
                "validation": validation.to_dict(),
            },
            persist_operation=False,
        )

    async def _list(
        self,
        command: DbMonitorCommand,
        request: DbRequest,
    ) -> DbOperationResult:
        monitors = await self.runtime.list_monitors(status=command.patch.get("status"))
        return await self.runtime.record_monitor_command_result(
            request=request,
            kind=command.kind,
            command=_command_payload(command),
            status=OperationStatus.SUCCEEDED,
            answer=_list_monitors_answer(monitors),
            evidence_kind="monitor.listing",
            payload={
                "command": _command_payload(command),
                "status": command.patch.get("status"),
                "monitors": [monitor.to_dict() for monitor in monitors],
            },
            diagnostics={"monitors": [monitor.to_dict() for monitor in monitors]},
        )

    async def _inspect(
        self,
        command: DbMonitorCommand,
        request: DbRequest,
    ) -> DbOperationResult:
        resolution = await self._resolve(command)
        if not resolution.accepted:
            return await self._resolution_failure(command, request, resolution)
        assert resolution.monitor is not None
        inspection = await self.runtime.inspect_monitor(resolution.monitor.id)
        if inspection is None:
            return await self._resolution_failure(
                command,
                request,
                DbMonitorResolution(
                    None,
                    resolution.monitor_ref,
                    errors=("monitor_not_found",),
                ),
            )
        evidence_kind = (
            "monitor.run_summary"
            if command.kind == "explain_run"
            else "monitor.inspection"
        )
        return await self.runtime.record_monitor_command_result(
            request=request,
            kind=command.kind,
            command=_command_payload(command),
            status=OperationStatus.SUCCEEDED,
            answer=_inspect_monitor_answer(inspection, command=command),
            evidence_kind=evidence_kind,
            payload={
                "command": _command_payload(command),
                "resolution": resolution.to_dict(),
                "inspection": inspection.to_dict(),
            },
            warnings=resolution.warnings,
            diagnostics={
                "resolution": resolution.to_dict(),
                "inspection": inspection.to_dict(),
            },
        )

    async def _update(
        self,
        command: DbMonitorCommand,
        request: DbRequest,
    ) -> DbOperationResult:
        resolution = await self._resolve(command)
        if not resolution.accepted:
            return await self._resolution_failure(command, request, resolution)
        assert resolution.monitor is not None
        updated = await self.runtime.update_monitor(
            resolution.monitor.id, command.patch
        )
        inspection = await self.runtime.inspect_monitor(updated.id)
        operation_id = _inspection_operation_id(inspection)
        evidence = await _operation_evidence(self.runtime, operation_id)
        return await self.runtime.record_monitor_command_result(
            request=request,
            kind=command.kind,
            command=_command_payload(command, monitor_id=updated.id),
            status=OperationStatus.SUCCEEDED,
            answer=f"Updated monitor {updated.name} ({updated.id}).",
            operation_id=operation_id,
            evidence=tuple(evidence),
            warnings=resolution.warnings,
            diagnostics={
                "resolution": resolution.to_dict(),
                "monitor": updated.to_dict(),
                "inspection": None if inspection is None else inspection.to_dict(),
            },
            persist_operation=False,
        )

    async def _pause(
        self,
        command: DbMonitorCommand,
        request: DbRequest,
    ) -> DbOperationResult:
        resolution = await self._resolve(command)
        if not resolution.accepted:
            return await self._resolution_failure(command, request, resolution)
        assert resolution.monitor is not None
        paused = await self.runtime.pause_monitor(
            resolution.monitor.id,
            paused_until=command.patch.get("paused_until"),
        )
        inspection = await self.runtime.inspect_monitor(paused.id)
        operation_id = _inspection_operation_id(inspection)
        evidence = await _operation_evidence(self.runtime, operation_id)
        return await self.runtime.record_monitor_command_result(
            request=request,
            kind=command.kind,
            command=_command_payload(command, monitor_id=paused.id),
            status=OperationStatus.SUCCEEDED,
            answer=f"Paused monitor {paused.name} ({paused.id}).",
            operation_id=operation_id,
            evidence=tuple(evidence),
            warnings=resolution.warnings,
            diagnostics={
                "resolution": resolution.to_dict(),
                "monitor": paused.to_dict(),
                "inspection": None if inspection is None else inspection.to_dict(),
            },
            persist_operation=False,
        )

    async def _resume(
        self,
        command: DbMonitorCommand,
        request: DbRequest,
    ) -> DbOperationResult:
        resolution = await self._resolve(command)
        if not resolution.accepted:
            return await self._resolution_failure(command, request, resolution)
        assert resolution.monitor is not None
        resumed = await self.runtime.resume_monitor(resolution.monitor.id)
        inspection = await self.runtime.inspect_monitor(resumed.id)
        operation_id = _inspection_operation_id(inspection)
        evidence = await _operation_evidence(self.runtime, operation_id)
        return await self.runtime.record_monitor_command_result(
            request=request,
            kind=command.kind,
            command=_command_payload(command, monitor_id=resumed.id),
            status=OperationStatus.SUCCEEDED,
            answer=f"Resumed monitor {resumed.name} ({resumed.id}).",
            operation_id=operation_id,
            evidence=tuple(evidence),
            warnings=resolution.warnings,
            diagnostics={
                "resolution": resolution.to_dict(),
                "monitor": resumed.to_dict(),
                "inspection": None if inspection is None else inspection.to_dict(),
            },
            persist_operation=False,
        )

    async def _delete(
        self,
        command: DbMonitorCommand,
        request: DbRequest,
    ) -> DbOperationResult:
        resolution = await self._resolve(command)
        if not resolution.accepted:
            return await self._resolution_failure(command, request, resolution)
        assert resolution.monitor is not None
        deleted = await self.runtime.delete_monitor(resolution.monitor.id)
        operation_id = await _latest_monitor_operation_id(
            self.runtime,
            "delete",
            deleted.id,
        )
        evidence = await _operation_evidence(self.runtime, operation_id)
        return await self.runtime.record_monitor_command_result(
            request=request,
            kind=command.kind,
            command=_command_payload(command, monitor_id=deleted.id),
            status=OperationStatus.SUCCEEDED,
            answer=f"Deleted monitor {deleted.name} ({deleted.id}).",
            operation_id=operation_id,
            evidence=tuple(evidence),
            warnings=resolution.warnings,
            diagnostics={
                "resolution": resolution.to_dict(),
                "monitor": deleted.to_dict(),
            },
            persist_operation=False,
        )

    async def _approve_action(
        self,
        command: DbMonitorCommand,
        request: DbRequest,
    ) -> DbOperationResult:
        resolution = await self._resolve(command)
        if not resolution.accepted:
            return await self._resolution_failure(command, request, resolution)
        assert resolution.monitor is not None
        action = str(command.patch.get("approval_action") or "approve")
        approvals = await self.runtime.list_monitor_approvals(
            monitor_id=resolution.monitor.id,
            pending_only=True,
        )
        if not approvals:
            return await self.runtime.record_monitor_command_result(
                request=request,
                kind=command.kind,
                command=_command_payload(command, monitor_id=resolution.monitor.id),
                status=OperationStatus.FAILED,
                answer=f"Monitor {resolution.monitor.id} has no pending approvals.",
                evidence_kind="monitor.command.approval",
                payload={
                    "command": _command_payload(
                        command, monitor_id=resolution.monitor.id
                    ),
                    "resolution": resolution.to_dict(),
                    "approval_action": action,
                    "approvals": [],
                    "status": "not_found",
                },
                warnings=("db_monitor_approval_not_found",),
                diagnostics={"resolution": resolution.to_dict()},
            )
        if len(approvals) > 1:
            return await self.runtime.record_monitor_command_result(
                request=request,
                kind=command.kind,
                command=_command_payload(command, monitor_id=resolution.monitor.id),
                status=OperationStatus.FAILED,
                answer=(
                    f"Monitor {resolution.monitor.id} has multiple pending approvals; "
                    "use the approval id to choose one."
                ),
                evidence_kind="monitor.command.approval",
                payload={
                    "command": _command_payload(
                        command, monitor_id=resolution.monitor.id
                    ),
                    "resolution": resolution.to_dict(),
                    "approval_action": action,
                    "approvals": [dict(item) for item in approvals],
                    "status": "ambiguous",
                },
                warnings=("db_monitor_approval_ambiguous",),
                diagnostics={
                    "resolution": resolution.to_dict(),
                    "approval_ids": [item["approval_id"] for item in approvals],
                },
            )

        approval_context = dict(approvals[0])
        approval_id = str(approval_context["approval_id"])
        if action == "reject":
            approval = await self.runtime.reject_monitor_approval(approval_id)
        elif action == "cancel":
            approval = await self.runtime.cancel_monitor_approval(approval_id)
        else:
            approval = await self.runtime.approve_monitor_approval(approval_id)

        operation_id = str(approval_context.get("operation_id") or "")
        resumed = None
        if operation_id:
            resumed = await self.runtime.resume_operation(operation_id)
        status = (
            resumed.operation.status
            if resumed is not None
            else OperationStatus.SUCCEEDED
        )
        answer = _approval_action_answer(
            action,
            resolution.monitor.id,
            approval_id,
            operation_status=status.value,
        )
        return await self.runtime.record_monitor_command_result(
            request=request,
            kind=command.kind,
            command=_command_payload(command, monitor_id=resolution.monitor.id),
            status=status,
            answer=answer,
            evidence_kind="monitor.command.approval",
            payload={
                "command": _command_payload(command, monitor_id=resolution.monitor.id),
                "resolution": resolution.to_dict(),
                "approval_action": action,
                "approval_id": approval_id,
                "approval_status": approval.status.value,
                "operation_id": operation_id or None,
                "operation_status": status.value,
                "approval_context": approval_context.get("context"),
            },
            warnings=resolution.warnings,
            diagnostics={
                "resolution": resolution.to_dict(),
                "approval_id": approval_id,
                "approval_status": approval.status.value,
                "operation_id": operation_id or None,
                "operation_status": status.value,
            },
        )

    async def _resolve(self, command: DbMonitorCommand) -> DbMonitorResolution:
        monitors = await self.runtime.list_monitors()
        return self.resolver.resolve(command, monitors)

    async def _resolution_failure(
        self,
        command: DbMonitorCommand,
        request: DbRequest,
        resolution: DbMonitorResolution,
    ) -> DbOperationResult:
        reason = resolution.errors[0] if resolution.errors else "monitor_not_found"
        answer = _resolution_failure_answer(reason, resolution)
        return await self.runtime.record_monitor_command_result(
            request=request,
            kind=command.kind,
            command=_command_payload(command),
            status=OperationStatus.FAILED,
            answer=answer,
            evidence_kind="monitor.command.resolution",
            payload={
                "command": _command_payload(command),
                "resolution": resolution.to_dict(),
            },
            warnings=(f"db_{reason}",),
            diagnostics={"resolution": resolution.to_dict()},
        )


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


def _create_name_phrase(prompt: str) -> str:
    text = prompt.strip().rstrip(".")
    match = re.match(r"(?i)^(?:please\s+)?(?:monitor|watch)\s+(.+)$", text)
    if match:
        return _before_boundary(match.group(1))
    report = re.search(
        r"(?i)(?:give me|generate|send|prepare)\s+(?:a\s+)?(.+?)\s+report\b", text
    )
    if report:
        return f"{report.group(1)} report"
    return _before_boundary(text)


def _before_boundary(text: str) -> str:
    return re.split(
        r"(?i)\b(?:every|if|when|then|and alert|notify|at\s+\d|until)\b",
        text,
        maxsplit=1,
    )[0].strip(" ,;:") or text.strip(" ,;:")


def _watch_from_prompt(prompt: str) -> str:
    return _before_boundary(
        re.sub(r"(?i)^(?:please\s+)?(?:monitor|watch)\s+", "", prompt.strip())
    )


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
    if not action_text and "report" in prompt.lower():
        action_text = "generate the requested report"
    if not action_text:
        return ()
    parts = re.split(r"(?i),\s*|\s+and\s+", action_text.strip(" ."))
    return tuple(part.strip(" .") for part in parts if part.strip(" ."))


def _action_steps_from_prompt(prompt: str) -> tuple[dict[str, Any], ...]:
    steps: list[dict[str, Any]] = []
    for action in _actions_from_prompt(prompt):
        lowered = action.lower()
        if lowered.startswith("notify "):
            target = action[len("notify ") :].strip()
            step: dict[str, Any] = {
                "kind": "notify",
                "target": target,
                "instruction": action,
            }
            if target.startswith("#") or "slack" in lowered or "channel" in lowered:
                step["required_capability"] = "slack.message.send"
            elif "email" in lowered or "@" in target:
                step["required_capability"] = "email.message.send"
            else:
                step["required_capability"] = "notification.send"
            steps.append(step)
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


def _title_from_phrase(phrase: str) -> str:
    words = [word for word in re.split(r"\s+", phrase.strip()) if word]
    if not words:
        return "DB Monitor"
    return " ".join(word[:1].upper() + word[1:] for word in words)


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


def _has_any(capability_ids: set[str], tokens: tuple[str, ...]) -> bool:
    return any(
        token in capability_id for capability_id in capability_ids for token in tokens
    )


def _owner_from_request(request: DbRequest) -> dict[str, Any]:
    owner: dict[str, Any] = {}
    if request.user_id is not None:
        owner["user_id"] = request.user_id
    if request.session_id is not None:
        owner["session_id"] = request.session_id
    return owner


def _command_payload(
    command: DbMonitorCommand,
    *,
    monitor_id: str | None = None,
) -> dict[str, Any]:
    return {
        "kind": command.kind,
        "monitor_id": monitor_id or command.monitor_id,
        "patch": command.patch,
        "prompt": command.prompt,
        "confidence": command.confidence,
        "diagnostics": command.diagnostics,
    }


def _inspection_operation_id(inspection: Any) -> str | None:
    if inspection is None or inspection.state is None:
        return None
    return inspection.state.last_operation_id


async def _operation_evidence(
    runtime: Any, operation_id: str | None
) -> tuple[Any, ...]:
    if not operation_id:
        return ()
    return tuple(await runtime.store.list_evidence(operation_id))


async def _latest_monitor_operation_id(
    runtime: Any,
    action: str,
    monitor_id: str,
) -> str | None:
    operations = await runtime.store.list_operations()
    for operation in reversed(operations):
        if (
            operation.operation_type == f"monitor.{action}"
            and operation.metadata.get("monitor_id") == monitor_id
        ):
            return operation.id
    return None


def _create_monitor_answer(monitor: DbMonitor) -> str:
    schedule = ""
    if monitor.schedule:
        schedule = f" on {monitor.schedule.get('expression', 'its schedule')}"
    return f"Created monitor {monitor.name} ({monitor.id}){schedule}."


def _monitor_validation_answer(validation: DbMonitorValidation) -> str:
    missing = ", ".join(validation.missing_capabilities)
    if missing:
        return f"Monitor was not created because required capabilities are missing: {missing}."
    return "Monitor was not created because its definition did not pass validation."


def _list_monitors_answer(monitors: tuple[DbMonitor, ...]) -> str:
    if not monitors:
        return "No monitors are currently defined."
    lines = ["Monitors:"]
    for monitor in monitors:
        lines.append(f"- {monitor.id}: {monitor.name} [{monitor.status}]")
    return "\n".join(lines)


def _inspect_monitor_answer(
    inspection: Any,
    *,
    command: DbMonitorCommand,
) -> str:
    monitor = inspection.monitor
    if command.kind == "explain_run":
        if not inspection.runs:
            return f"Monitor {monitor.name} ({monitor.id}) has no recorded runs yet."
        last_run = inspection.runs[-1]
        return (
            f"Monitor {monitor.name} ({monitor.id}) last run "
            f"{last_run.status}; operation {last_run.operation_id}."
        )
    schedule = ""
    if monitor.schedule:
        schedule = f", schedule {monitor.schedule.get('expression')}"
    return f"Monitor {monitor.name} ({monitor.id}) is {monitor.status}{schedule}."


def _approval_action_answer(
    action: str,
    monitor_id: str,
    approval_id: str,
    *,
    operation_status: str,
) -> str:
    verb = {
        "approve": "Approved",
        "reject": "Rejected",
        "cancel": "Cancelled",
    }.get(action, "Updated")
    return (
        f"{verb} monitor approval {approval_id} for {monitor_id}; "
        f"operation is {operation_status}."
    )


def _resolution_failure_answer(
    reason: str,
    resolution: DbMonitorResolution,
) -> str:
    if reason == "monitor_reference_required":
        return "Please specify which monitor to manage."
    if reason == "monitor_reference_ambiguous":
        matches = ", ".join(monitor.id for monitor in resolution.matches)
        return f"Monitor reference is ambiguous; matching monitors: {matches}."
    return f"Monitor {resolution.monitor_ref!r} was not found."
