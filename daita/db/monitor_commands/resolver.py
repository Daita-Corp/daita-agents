"""Monitor reference resolution for DB monitor commands."""

from __future__ import annotations

from ..monitors import DbMonitor
from .prompt_parsing import _monitor_id_from_phrase
from .types import DbMonitorCommand, DbMonitorResolution


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
