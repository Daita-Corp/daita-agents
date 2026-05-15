"""
Finding contract for DB agents.

Findings are durable, JSON-safe records of notable DB observations. Local
``from_db()`` agents keep them in memory; hosted runtimes can persist the same
contract later.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from .monitors import MonitorDefinition

if TYPE_CHECKING:
    from ..agent import Agent
    from ...core.watch import WatchEvent


VALID_FINDING_SEVERITIES = {"info", "warning", "critical"}
VALID_FINDING_STATUSES = {"open", "resolved"}


@dataclass(frozen=True)
class Finding:
    """Portable, JSON-safe finding record."""

    id: str
    title: str
    severity: str
    status: str
    kind: str
    source: Dict[str, Any]
    entity: Dict[str, Any]
    observed: Dict[str, Any]
    evidence: Dict[str, Any] = field(default_factory=dict)
    detected_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    resolved_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "severity": self.severity,
            "status": self.status,
            "kind": self.kind,
            "source": _json_safe(self.source),
            "entity": _json_safe(self.entity),
            "observed": _json_safe(self.observed),
            "evidence": _json_safe(self.evidence),
            "detected_at": self.detected_at,
            "resolved_at": self.resolved_at,
        }


class DBFindings:
    """Developer-facing findings collection for ``agent.db.findings``."""

    def __init__(self, agent: "Agent"):
        self._agent = agent

    @property
    def all(self) -> List[Dict[str, Any]]:
        return list(getattr(self._agent, "_db_findings", []))

    @property
    def open(self) -> List[Dict[str, Any]]:
        return [finding for finding in self.all if finding.get("status") == "open"]

    @property
    def resolved(self) -> List[Dict[str, Any]]:
        return [finding for finding in self.all if finding.get("status") == "resolved"]

    def last(self) -> Optional[Dict[str, Any]]:
        findings = self.all
        return findings[-1] if findings else None

    def add(self, finding: Any) -> Dict[str, Any]:
        normalized = normalize_finding(finding).to_dict()
        _findings_store(self._agent).append(normalized)
        return normalized

    def export_json(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(self.all, indent=indent, default=str)


def normalize_finding(raw: Any) -> Finding:
    """Validate and normalize a finding into the portable DB finding contract."""
    if isinstance(raw, Finding):
        finding = raw
    elif isinstance(raw, dict):
        finding = Finding(
            id=str(raw.get("id") or _finding_id()),
            title=str(raw.get("title") or "").strip(),
            severity=str(raw.get("severity") or "warning").strip(),
            status=str(raw.get("status") or "open").strip(),
            kind=str(raw.get("kind") or "db_observation").strip(),
            source=dict(raw.get("source") or {}),
            entity=dict(raw.get("entity") or {}),
            observed=dict(raw.get("observed") or {}),
            evidence=dict(raw.get("evidence") or {}),
            detected_at=str(raw.get("detected_at") or _now_iso()),
            resolved_at=raw.get("resolved_at"),
        )
    else:
        raise TypeError("Findings must be dictionaries or Finding instances")

    if not finding.id:
        raise ValueError("Finding requires an id")
    if not finding.title:
        raise ValueError("Finding requires a title")
    if finding.severity not in VALID_FINDING_SEVERITIES:
        raise ValueError(
            f"Unsupported finding severity {finding.severity!r}; expected one of "
            f"{sorted(VALID_FINDING_SEVERITIES)}"
        )
    if finding.status not in VALID_FINDING_STATUSES:
        raise ValueError(
            f"Unsupported finding status {finding.status!r}; expected one of "
            f"{sorted(VALID_FINDING_STATUSES)}"
        )
    if not finding.kind:
        raise ValueError("Finding requires a kind")

    return Finding(
        id=finding.id,
        title=finding.title,
        severity=finding.severity,
        status=finding.status,
        kind=finding.kind,
        source=_json_safe(finding.source),
        entity=_json_safe(finding.entity),
        observed=_json_safe(finding.observed),
        evidence=_json_safe(finding.evidence),
        detected_at=finding.detected_at,
        resolved_at=finding.resolved_at,
    )


def normalize_findings(raw: Iterable[Any]) -> List[Finding]:
    """Normalize a collection of findings."""
    return [normalize_finding(item) for item in raw]


def finding_from_monitor_event(
    monitor: MonitorDefinition,
    event: "WatchEvent",
    *,
    finding_id: Optional[str] = None,
) -> Finding:
    """Create a finding from a monitor threshold or resolve event."""
    status = "resolved" if event.resolved else "open"
    title = f"{monitor.name} resolved" if event.resolved else monitor.name
    detected_at = event.triggered_at.isoformat()
    return Finding(
        id=finding_id or _finding_id(),
        title=title,
        severity=monitor.severity,
        status=status,
        kind=f"db_monitor.{monitor.type}",
        source={
            "type": "monitor",
            "name": monitor.name,
            "monitor_type": monitor.type,
        },
        entity=monitor.entity,
        observed={
            "value": event.value,
            "previous_value": event.previous_value,
            "resolved": event.resolved,
        },
        evidence={
            "sql": monitor.sql,
            "threshold": monitor.threshold,
            "monitor": monitor.to_dict(),
        },
        detected_at=detected_at,
        resolved_at=detected_at if event.resolved else None,
    )


def record_monitor_finding(
    agent: "Agent", event: "WatchEvent", monitor: MonitorDefinition
) -> Dict[str, Any]:
    """Record or resolve a local finding for a monitor event."""
    active = getattr(agent, "_db_active_findings", None)
    if active is None:
        active = {}
        agent._db_active_findings = active

    active_key = monitor.name
    if event.resolved:
        finding_id = active.pop(active_key, None)
        if finding_id:
            resolved = finding_from_monitor_event(
                monitor, event, finding_id=finding_id
            ).to_dict()
            for index, existing in enumerate(_findings_store(agent)):
                if existing.get("id") == finding_id:
                    existing.update(
                        {
                            "status": "resolved",
                            "resolved_at": resolved["resolved_at"],
                            "observed": resolved["observed"],
                            "evidence": resolved["evidence"],
                        }
                    )
                    return existing
        finding = finding_from_monitor_event(monitor, event).to_dict()
    else:
        finding = finding_from_monitor_event(monitor, event).to_dict()
        active[active_key] = finding["id"]

    _findings_store(agent).append(finding)
    return finding


def _findings_store(agent: "Agent") -> List[Dict[str, Any]]:
    findings = getattr(agent, "_db_findings", None)
    if findings is None:
        findings = []
        agent._db_findings = findings
    return findings


def _finding_id() -> str:
    return f"fnd_{uuid.uuid4().hex[:16]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
