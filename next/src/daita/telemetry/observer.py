"""Failure-isolated observers over redacted committed-event projections."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import re
from typing import Protocol, runtime_checkable

from .._json import FrozenJsonObject
from ..events.models import CommittedEvent
from ..events.projection import EventAudience, project_committed_event

_EXPORTER_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")


@runtime_checkable
class TelemetryExporter(Protocol):
    """Optional sink that receives only a redacted immutable event projection."""

    @property
    def exporter_id(self) -> str: ...

    async def export(self, event: FrozenJsonObject) -> None: ...


@dataclass(frozen=True, slots=True)
class TelemetryExportFailure:
    """Safe exporter failure fact that never includes exporter exception text."""

    exporter_id: str
    event_id: str
    code: str = "telemetry.export_failed"

    def __post_init__(self) -> None:
        _validated_exporter_id(self.exporter_id)
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError("telemetry failure event_id must be non-empty")
        if self.code != "telemetry.export_failed":
            raise ValueError("telemetry failure code is not recognized")


class CommittedEventObserver:
    """Project and fan out committed events without affecting their commit."""

    def __init__(self, exporters: Iterable[TelemetryExporter] = ()) -> None:
        values = tuple(exporters)
        seen: set[str] = set()
        for exporter in values:
            exporter_id = _validated_exporter_id(getattr(exporter, "exporter_id", None))
            if not callable(getattr(exporter, "export", None)):
                raise TypeError("telemetry exporter must provide export(event)")
            if exporter_id in seen:
                raise ValueError(
                    f"telemetry exporter already registered: {exporter_id}"
                )
            seen.add(exporter_id)
        self._exporters = values
        self._exporter_ids = tuple(
            getattr(exporter, "exporter_id") for exporter in values
        )

    @property
    def exporter_ids(self) -> tuple[str, ...]:
        return self._exporter_ids

    async def observe(
        self,
        committed: CommittedEvent,
    ) -> tuple[TelemetryExportFailure, ...]:
        if not isinstance(committed, CommittedEvent):
            raise TypeError("telemetry observer requires a CommittedEvent")
        projection = project_committed_event(
            committed,
            audience=EventAudience.TELEMETRY,
        )
        failures: list[TelemetryExportFailure] = []
        for exporter in self._exporters:
            try:
                await exporter.export(projection)
            except Exception:
                failures.append(
                    TelemetryExportFailure(
                        exporter_id=exporter.exporter_id,
                        event_id=committed.event.id,
                    )
                )
        return tuple(failures)


def _validated_exporter_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value.encode("utf-8")) > 128
        or _EXPORTER_ID_RE.fullmatch(value) is None
    ):
        raise ValueError("telemetry exporter_id must be a bounded lowercase identifier")
    return value


__all__ = [
    "CommittedEventObserver",
    "TelemetryExporter",
    "TelemetryExportFailure",
]
