from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest
import daita

from daita._json import FrozenJsonObject
from daita.events.models import CommittedEvent, EventCursor, RuntimeEvent
from daita.events.projection import EventAudience, project_committed_event
from daita.hosting.local_server import _event_projection
from daita.telemetry import CommittedEventObserver, TelemetryExporter

NOW = datetime(2026, 7, 19, 18, 0, tzinfo=timezone.utc)


def _committed_event() -> CommittedEvent:
    return CommittedEvent(
        cursor=EventCursor(agent_id="agent-1", sequence=17),
        event=RuntimeEvent(
            id="event-1",
            type="model_response.recorded",
            agent_id="agent-1",
            operation_id="operation-1",
            session_id="session-1",
            turn_id="turn-1",
            model_call_id="model-call-1",
            call_id="call-1",
            task_id="task-1",
            evidence_id="evidence-1",
            approval_id="approval-1",
            monitor_id="monitor-1",
            capability_id="example.read",
            executor_id="example.read.executor",
            created_at=NOW,
            payload={
                "status": "completed",
                "total_tokens": 23,
                "estimated_cost_usd": "0.0042",
                "nested": {"count": 2, "password": "audit-secret"},
                "authorization": "Bearer public-secret",
                "headers": {"X-Api-Key": "header-secret"},
                "connection_string": "postgresql://user:password@host/db",
                "query_params": {"token": "query-secret"},
                "arguments": {"customer": "Ada"},
                "rows": [{"ssn": "123-45-6789"}],
                "file_content": "private document",
                "prompt": "private prompt",
            },
        ),
    )


def test_public_and_audit_event_views_are_separate_bounded_projections() -> None:
    committed = _committed_event()

    public = project_committed_event(committed, audience=EventAudience.PUBLIC)
    audit = project_committed_event(committed, audience=EventAudience.AUDIT)

    assert isinstance(public, FrozenJsonObject)
    assert public["sequence"] == 17
    assert public["agent_id"] == "agent-1"
    assert public["operation_id"] == "operation-1"
    assert public["created_at"] == NOW.isoformat()
    assert public["payload"] == FrozenJsonObject.from_mapping({})

    audit_payload = audit["payload"]
    assert isinstance(audit_payload, FrozenJsonObject)
    assert audit_payload["status"] == "completed"
    nested = audit_payload["nested"]
    assert isinstance(nested, FrozenJsonObject)
    assert nested["count"] == 2
    assert nested["password"] == "[redacted]"
    for field_name in (
        "authorization",
        "headers",
        "connection_string",
        "query_params",
        "arguments",
        "rows",
        "file_content",
        "prompt",
    ):
        assert audit_payload[field_name] == "[redacted]"

    rendered = str(public.to_dict()) + str(audit.to_dict())
    for secret in (
        "audit-secret",
        "public-secret",
        "header-secret",
        "password@host",
        "query-secret",
        "123-45-6789",
        "private document",
        "private prompt",
    ):
        assert secret not in rendered


def test_telemetry_projection_contains_only_allowlisted_metrics_and_codes() -> None:
    projection = project_committed_event(
        _committed_event(),
        audience=EventAudience.TELEMETRY,
    )
    payload = projection["payload"]

    assert isinstance(payload, FrozenJsonObject)
    assert payload.to_dict() == {
        "estimated_cost_usd": "0.0042",
        "status": "completed",
        "total_tokens": 23,
    }
    assert "secret" not in str(projection.to_dict()).lower()


def test_local_protocol_reuses_the_public_projection() -> None:
    committed = _committed_event()

    assert (
        _event_projection(committed)
        == project_committed_event(
            committed,
            audience=EventAudience.PUBLIC,
        ).to_dict()
    )


class _RecordingExporter:
    exporter_id = "recording"

    def __init__(self) -> None:
        self.events: list[FrozenJsonObject] = []

    async def export(self, event: FrozenJsonObject) -> None:
        self.events.append(event)


class _FailingExporter:
    exporter_id = "failing"

    def __init__(self, secret: str) -> None:
        self._secret = secret

    async def export(self, event: FrozenJsonObject) -> None:
        del event
        raise RuntimeError(self._secret)


async def test_committed_event_observer_is_optional_and_failure_isolated() -> None:
    recording = _RecordingExporter()
    observer = CommittedEventObserver(
        exporters=(
            _FailingExporter("exporter-secret"),
            recording,
        )
    )

    failures = await observer.observe(_committed_event())

    assert observer.exporter_ids == ("failing", "recording")
    assert len(failures) == 1
    assert failures[0].exporter_id == "failing"
    assert failures[0].event_id == "event-1"
    assert failures[0].code == "telemetry.export_failed"
    assert "exporter-secret" not in repr(failures[0])
    assert len(recording.events) == 1
    assert recording.events[0] == project_committed_event(
        _committed_event(),
        audience=EventAudience.TELEMETRY,
    )

    empty = CommittedEventObserver()
    assert await empty.observe(_committed_event()) == ()


async def test_observer_rejects_uncommitted_events_and_exporter_collisions() -> None:
    raw = _committed_event().event
    with pytest.raises(TypeError, match="CommittedEvent"):
        await CommittedEventObserver().observe(cast(CommittedEvent, raw))

    exporter = _RecordingExporter()
    with pytest.raises(ValueError, match="already registered"):
        CommittedEventObserver(exporters=(exporter, exporter))

    assert isinstance(exporter, TelemetryExporter)


def test_commit_owners_do_not_depend_on_telemetry() -> None:
    source_root = Path(__file__).resolve().parents[3] / "src" / "daita"
    commit_owners = (
        source_root / "operations" / "runtime.py",
        source_root / "operations" / "store.py",
        source_root / "storage" / "sqlite.py",
    )

    for path in commit_owners:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert all(
            "telemetry"
            not in (
                node.module
                if isinstance(node, ast.ImportFrom) and node.module is not None
                else " ".join(alias.name for alias in cast(ast.Import, node).names)
            )
            for node in imports
        ), f"commit owner imports telemetry: {path}"


def test_root_surface_replaces_global_tracing_with_explicit_observers() -> None:
    assert daita.CommittedEventObserver is CommittedEventObserver
    assert daita.TelemetryExporter is TelemetryExporter
    assert daita.EventAudience is EventAudience
    assert daita.project_committed_event is project_committed_event
    for removed in ("configure_tracing", "get_trace_manager", "set_trace_context"):
        assert not hasattr(daita, removed)
