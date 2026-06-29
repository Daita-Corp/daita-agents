"""Durable DB monitor control-plane records and stores."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Literal, Protocol

from daita.runtime import (
    Evidence,
    Operation,
    RuntimeEvent,
    RuntimeStore,
)

DbMonitorStatus = Literal["active", "paused", "disabled", "error"]
DbMonitorRunStatus = Literal["skipped", "triggered", "blocked", "failed", "succeeded"]
DbMonitorMutationAction = Literal[
    "create", "update", "pause", "resume", "delete", "tick", "run"
]


@dataclass(frozen=True)
class DbMonitor:
    """Standing autonomous DB instruction owned by the from_db control plane."""

    id: str
    name: str
    description: str = ""
    status: DbMonitorStatus = "active"
    source_scope: tuple[str, ...] = ()
    schedule: dict[str, Any] | None = None
    stream: dict[str, Any] | None = None
    trigger: dict[str, Any] = field(default_factory=dict)
    observation_plan: dict[str, Any] = field(default_factory=dict)
    action_plan: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    owner: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("monitor id is required")
        if not self.name:
            raise ValueError("monitor name is required")
        if self.status not in {"active", "paused", "disabled", "error"}:
            raise ValueError(f"unsupported monitor status: {self.status!r}")
        now = _now_iso()
        object.__setattr__(self, "source_scope", tuple(self.source_scope))
        object.__setattr__(
            self, "schedule", None if self.schedule is None else dict(self.schedule)
        )
        object.__setattr__(
            self, "stream", None if self.stream is None else dict(self.stream)
        )
        object.__setattr__(self, "trigger", dict(self.trigger))
        object.__setattr__(self, "observation_plan", dict(self.observation_plan))
        object.__setattr__(self, "action_plan", dict(self.action_plan))
        object.__setattr__(self, "policy", dict(self.policy))
        object.__setattr__(self, "budgets", dict(self.budgets))
        object.__setattr__(self, "owner", dict(self.owner))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "created_at", self.created_at or now)
        object.__setattr__(self, "updated_at", self.updated_at or now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "source_scope": list(self.source_scope),
            "schedule": self.schedule,
            "stream": self.stream,
            "trigger": self.trigger,
            "observation_plan": self.observation_plan,
            "action_plan": self.action_plan,
            "policy": self.policy,
            "budgets": self.budgets,
            "owner": self.owner,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DbMonitor":
        values = dict(data)
        values["source_scope"] = tuple(values.get("source_scope") or ())
        return cls(**values)


@dataclass(frozen=True)
class DbMonitorState:
    """Durable cursor and lifecycle state for a standing DB monitor."""

    monitor_id: str
    cursor: dict[str, Any] = field(default_factory=dict)
    last_tick_at: str | None = None
    last_triggered_at: str | None = None
    last_operation_id: str | None = None
    last_tick_operation_id: str | None = None
    last_triggered_operation_id: str | None = None
    last_management_operation_id: str | None = None
    last_value_summary: dict[str, Any] | None = None
    consecutive_matches: int = 0
    consecutive_failures: int = 0
    cooldown_until: str | None = None
    paused_until: str | None = None
    error: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.monitor_id:
            raise ValueError("monitor_id is required")
        if self.consecutive_matches < 0:
            raise ValueError("consecutive_matches cannot be negative")
        if self.consecutive_failures < 0:
            raise ValueError("consecutive_failures cannot be negative")
        object.__setattr__(self, "cursor", dict(self.cursor))
        object.__setattr__(
            self,
            "last_value_summary",
            None if self.last_value_summary is None else dict(self.last_value_summary),
        )
        object.__setattr__(
            self, "error", None if self.error is None else dict(self.error)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "monitor_id": self.monitor_id,
            "cursor": self.cursor,
            "last_tick_at": self.last_tick_at,
            "last_triggered_at": self.last_triggered_at,
            "last_operation_id": self.last_operation_id,
            "last_tick_operation_id": self.last_tick_operation_id,
            "last_triggered_operation_id": self.last_triggered_operation_id,
            "last_management_operation_id": self.last_management_operation_id,
            "last_value_summary": self.last_value_summary,
            "consecutive_matches": self.consecutive_matches,
            "consecutive_failures": self.consecutive_failures,
            "cooldown_until": self.cooldown_until,
            "paused_until": self.paused_until,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DbMonitorState":
        return cls(**dict(data))


@dataclass(frozen=True)
class DbMonitorRun:
    """One durable DB monitor tick/run summary."""

    id: str
    monitor_id: str
    operation_id: str
    tick_started_at: str
    tick_finished_at: str | None = None
    triggered: bool = False
    trigger_decision_evidence_id: str | None = None
    status: DbMonitorRunStatus = "skipped"
    summary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("run id is required")
        if not self.monitor_id:
            raise ValueError("monitor_id is required")
        if not self.operation_id:
            raise ValueError("operation_id is required")
        if not self.tick_started_at:
            raise ValueError("tick_started_at is required")
        if self.status not in {
            "skipped",
            "triggered",
            "blocked",
            "failed",
            "succeeded",
        }:
            raise ValueError(f"unsupported monitor run status: {self.status!r}")
        object.__setattr__(self, "summary", dict(self.summary))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "monitor_id": self.monitor_id,
            "operation_id": self.operation_id,
            "tick_started_at": self.tick_started_at,
            "tick_finished_at": self.tick_finished_at,
            "triggered": self.triggered,
            "trigger_decision_evidence_id": self.trigger_decision_evidence_id,
            "status": self.status,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DbMonitorRun":
        return cls(**dict(data))


@dataclass(frozen=True)
class DbMonitorInspection:
    """Stable inspection shape for prompt synthesis and typed callers."""

    monitor: DbMonitor
    state: DbMonitorState | None = None
    runs: tuple[DbMonitorRun, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "runs", tuple(self.runs))

    def to_dict(self) -> dict[str, Any]:
        return {
            "monitor": self.monitor.to_dict(),
            "state": None if self.state is None else self.state.to_dict(),
            "runs": [run.to_dict() for run in self.runs],
        }


@dataclass(frozen=True)
class DbMonitorMutation:
    """One atomic DB monitor control-plane mutation plus runtime audit artifacts."""

    action: DbMonitorMutationAction
    operation: Operation
    evidence: tuple[Evidence, ...] = ()
    events: tuple[RuntimeEvent, ...] = ()
    monitor_before: DbMonitor | None = None
    monitor_after: DbMonitor | None = None
    state_before: DbMonitorState | None = None
    state_after: DbMonitorState | None = None
    run_after: DbMonitorRun | None = None
    lease_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "events", tuple(self.events))
        if self.action == "create" and self.monitor_after is None:
            raise ValueError("create mutations require monitor_after")
        if self.action in {"update", "pause", "resume"}:
            if self.monitor_before is None or self.monitor_after is None:
                raise ValueError(
                    f"{self.action} mutations require before and after monitors"
                )
            if self.monitor_before.id != self.monitor_after.id:
                raise ValueError("monitor id cannot change during mutation")
        if self.action == "delete" and self.monitor_before is None:
            raise ValueError("delete mutations require monitor_before")
        if self.action == "run":
            if self.monitor_before is None:
                raise ValueError("run mutations require monitor_before")
            if self.run_after is None:
                raise ValueError("run mutations require run_after")
            if self.run_after.monitor_id != self.monitor_before.id:
                raise ValueError("run monitor id must match monitor_before")
            if (
                self.state_after is not None
                and self.state_after.monitor_id != self.monitor_before.id
            ):
                raise ValueError("run state_after monitor id must match monitor_before")
            if (
                self.state_before is not None
                and self.state_before.monitor_id != self.monitor_before.id
            ):
                raise ValueError(
                    "run state_before monitor id must match monitor_before"
                )


class DbMonitorStore(Protocol):
    """Persistence boundary for DB monitor control-plane records."""

    async def save_monitor(self, monitor: DbMonitor) -> None:
        """Persist a monitor definition."""

    async def load_monitor(self, monitor_id: str) -> DbMonitor | None:
        """Load a monitor definition by id."""

    async def list_monitors(
        self, *, status: str | None = None
    ) -> tuple[DbMonitor, ...]:
        """List persisted monitors, optionally filtered by status."""

    async def delete_monitor(self, monitor_id: str) -> None:
        """Delete a monitor and its DB monitor control-plane records."""

    async def save_monitor_state(self, state: DbMonitorState) -> None:
        """Persist durable monitor state."""

    async def load_monitor_state(self, monitor_id: str) -> DbMonitorState | None:
        """Load durable monitor state."""

    async def save_monitor_run(self, run: DbMonitorRun) -> None:
        """Persist a monitor run summary."""

    async def list_monitor_runs(self, monitor_id: str) -> tuple[DbMonitorRun, ...]:
        """List run summaries for one monitor."""

    async def claim_monitor_tick_lease(
        self,
        monitor_id: str,
        *,
        lease_id: str,
        now: str,
        expires_at: str,
    ) -> bool:
        """Claim a durable tick lease if no unexpired lease is held."""

    async def release_monitor_tick_lease(
        self,
        monitor_id: str,
        *,
        lease_id: str,
    ) -> None:
        """Release a tick lease after the scheduler decision finishes."""

    async def commit_monitor_mutation(self, mutation: DbMonitorMutation) -> None:
        """Atomically persist monitor records and runtime audit artifacts."""


class InMemoryDbMonitorStore:
    """Transient monitor store used with the default in-memory runtime store."""

    def __init__(self, runtime_store: RuntimeStore | None = None) -> None:
        self.runtime_store = runtime_store
        self._monitors: dict[str, DbMonitor] = {}
        self._monitor_order: list[str] = []
        self._states: dict[str, DbMonitorState] = {}
        self._runs: dict[str, list[DbMonitorRun]] = {}
        self._leases: dict[str, dict[str, str]] = {}
        self._commit_lock = asyncio.Lock()

    async def save_monitor(self, monitor: DbMonitor) -> None:
        if monitor.id not in self._monitors:
            self._monitor_order.append(monitor.id)
        self._monitors[monitor.id] = monitor

    async def load_monitor(self, monitor_id: str) -> DbMonitor | None:
        return self._monitors.get(monitor_id)

    async def list_monitors(
        self, *, status: str | None = None
    ) -> tuple[DbMonitor, ...]:
        monitors = tuple(
            self._monitors[monitor_id]
            for monitor_id in self._monitor_order
            if monitor_id in self._monitors
        )
        if status is None:
            return monitors
        return tuple(monitor for monitor in monitors if monitor.status == status)

    async def delete_monitor(self, monitor_id: str) -> None:
        self._monitors.pop(monitor_id, None)
        self._states.pop(monitor_id, None)
        self._runs.pop(monitor_id, None)
        self._leases.pop(monitor_id, None)

    async def save_monitor_state(self, state: DbMonitorState) -> None:
        self._states[state.monitor_id] = state

    async def load_monitor_state(self, monitor_id: str) -> DbMonitorState | None:
        return self._states.get(monitor_id)

    async def save_monitor_run(self, run: DbMonitorRun) -> None:
        self._runs.setdefault(run.monitor_id, []).append(run)

    async def list_monitor_runs(self, monitor_id: str) -> tuple[DbMonitorRun, ...]:
        return tuple(self._runs.get(monitor_id, ()))

    async def claim_monitor_tick_lease(
        self,
        monitor_id: str,
        *,
        lease_id: str,
        now: str,
        expires_at: str,
    ) -> bool:
        async with self._commit_lock:
            lease = self._leases.get(monitor_id)
            if lease is not None and lease["expires_at"] > now:
                return False
            self._leases[monitor_id] = {
                "lease_id": lease_id,
                "claimed_at": now,
                "expires_at": expires_at,
            }
            return True

    async def release_monitor_tick_lease(
        self,
        monitor_id: str,
        *,
        lease_id: str,
    ) -> None:
        async with self._commit_lock:
            lease = self._leases.get(monitor_id)
            if lease is not None and lease["lease_id"] == lease_id:
                self._leases.pop(monitor_id, None)

    async def commit_monitor_mutation(self, mutation: DbMonitorMutation) -> None:
        if self.runtime_store is None:
            raise RuntimeError("runtime_store is required to commit monitor mutations")
        async with self._commit_lock:
            self._validate_expected_monitor(mutation)
            self._validate_expected_state(mutation)
            self._validate_expected_lease(mutation)
            monitor_ids_before = list(self._monitor_order)
            monitors_before = dict(self._monitors)
            states_before = dict(self._states)
            runs_before = {key: list(value) for key, value in self._runs.items()}
            leases_before = {key: dict(value) for key, value in self._leases.items()}
            try:
                self._apply_monitor_mutation(mutation)
                await self.runtime_store.save_operation(mutation.operation)
                for evidence in mutation.evidence:
                    await self.runtime_store.save_evidence(evidence)
                for event in mutation.events:
                    await self.runtime_store.append_event(event)
            except Exception:
                self._monitor_order = monitor_ids_before
                self._monitors = monitors_before
                self._states = states_before
                self._runs = runs_before
                self._leases = leases_before
                raise

    def _validate_expected_monitor(self, mutation: DbMonitorMutation) -> None:
        expected = mutation.monitor_before
        current_id = _mutation_monitor_id(mutation)
        current = self._monitors.get(current_id) if current_id is not None else None
        if mutation.action == "create":
            if current is not None:
                raise ValueError(f"monitor {current_id!r} already exists")
            return
        if expected is None:
            return
        if current is None:
            raise ValueError(f"monitor {expected.id!r} does not exist")
        if current != expected:
            raise ValueError(f"monitor {expected.id!r} changed during mutation")

    def _validate_expected_state(self, mutation: DbMonitorMutation) -> None:
        expected = mutation.state_before
        if expected is None:
            return
        current = self._states.get(expected.monitor_id)
        if current != expected:
            raise ValueError(
                f"monitor state {expected.monitor_id!r} changed during mutation"
            )

    def _validate_expected_lease(self, mutation: DbMonitorMutation) -> None:
        if mutation.lease_id is None:
            return
        monitor_id = _mutation_monitor_id(mutation)
        lease = self._leases.get(monitor_id) if monitor_id is not None else None
        if lease is None or lease["lease_id"] != mutation.lease_id:
            raise ValueError(f"monitor {monitor_id!r} tick lease is no longer held")

    def _apply_monitor_mutation(self, mutation: DbMonitorMutation) -> None:
        if mutation.monitor_after is not None:
            if mutation.monitor_after.id not in self._monitors:
                self._monitor_order.append(mutation.monitor_after.id)
            self._monitors[mutation.monitor_after.id] = mutation.monitor_after
        elif mutation.action == "delete" and mutation.monitor_before is not None:
            monitor_id = mutation.monitor_before.id
            self._monitors.pop(monitor_id, None)
            self._states.pop(monitor_id, None)
            self._runs.pop(monitor_id, None)
            self._leases.pop(monitor_id, None)
            return
        if mutation.state_after is not None:
            self._states[mutation.state_after.monitor_id] = mutation.state_after
        if mutation.run_after is not None:
            runs = self._runs.setdefault(mutation.run_after.monitor_id, [])
            for index, existing in enumerate(runs):
                if existing.id == mutation.run_after.id:
                    runs[index] = mutation.run_after
                    break
            else:
                runs.append(mutation.run_after)


class SQLiteDbMonitorStore:
    """SQLite-backed monitor store that can share the runtime SQLite database."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    async def save_monitor(self, monitor: DbMonitor) -> None:
        with self._transaction() as conn:
            conn.execute(
                """
                insert into monitors (id, status, updated_at, data)
                values (?, ?, ?, ?)
                on conflict(id) do update set
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    data=excluded.data
                """,
                (
                    monitor.id,
                    monitor.status,
                    monitor.updated_at,
                    _json(monitor.to_dict()),
                ),
            )

    async def load_monitor(self, monitor_id: str) -> DbMonitor | None:
        with self._connect() as conn:
            row = conn.execute(
                "select data from monitors where id = ?", (monitor_id,)
            ).fetchone()
        return DbMonitor.from_dict(_loads(row["data"])) if row else None

    async def list_monitors(
        self, *, status: str | None = None
    ) -> tuple[DbMonitor, ...]:
        sql = "select data from monitors"
        params: tuple[Any, ...] = ()
        if status is not None:
            sql += " where status = ?"
            params = (status,)
        sql += " order by rowid"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return tuple(DbMonitor.from_dict(_loads(row["data"])) for row in rows)

    async def delete_monitor(self, monitor_id: str) -> None:
        with self._transaction() as conn:
            conn.execute("delete from monitors where id = ?", (monitor_id,))
            conn.execute(
                "delete from monitor_states where monitor_id = ?", (monitor_id,)
            )
            conn.execute("delete from monitor_runs where monitor_id = ?", (monitor_id,))
            conn.execute(
                "delete from monitor_tick_leases where monitor_id = ?", (monitor_id,)
            )

    async def save_monitor_state(self, state: DbMonitorState) -> None:
        with self._transaction() as conn:
            conn.execute(
                """
                insert into monitor_states (monitor_id, data)
                values (?, ?)
                on conflict(monitor_id) do update set data=excluded.data
                """,
                (state.monitor_id, _json(state.to_dict())),
            )

    async def load_monitor_state(self, monitor_id: str) -> DbMonitorState | None:
        with self._connect() as conn:
            row = conn.execute(
                "select data from monitor_states where monitor_id = ?", (monitor_id,)
            ).fetchone()
        return DbMonitorState.from_dict(_loads(row["data"])) if row else None

    async def save_monitor_run(self, run: DbMonitorRun) -> None:
        with self._transaction() as conn:
            conn.execute(
                """
                insert into monitor_runs (id, monitor_id, operation_id, status, data)
                values (?, ?, ?, ?, ?)
                on conflict(id) do update set
                    monitor_id=excluded.monitor_id,
                    operation_id=excluded.operation_id,
                    status=excluded.status,
                    data=excluded.data
                """,
                (
                    run.id,
                    run.monitor_id,
                    run.operation_id,
                    run.status,
                    _json(run.to_dict()),
                ),
            )

    async def list_monitor_runs(self, monitor_id: str) -> tuple[DbMonitorRun, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                select data from monitor_runs
                where monitor_id = ?
                order by rowid
                """,
                (monitor_id,),
            ).fetchall()
        return tuple(DbMonitorRun.from_dict(_loads(row["data"])) for row in rows)

    async def claim_monitor_tick_lease(
        self,
        monitor_id: str,
        *,
        lease_id: str,
        now: str,
        expires_at: str,
    ) -> bool:
        with self._transaction() as conn:
            row = conn.execute(
                """
                select lease_id, expires_at
                from monitor_tick_leases
                where monitor_id = ?
                """,
                (monitor_id,),
            ).fetchone()
            if row is not None and row["expires_at"] > now:
                return False
            conn.execute(
                """
                insert into monitor_tick_leases
                    (monitor_id, lease_id, claimed_at, expires_at)
                values (?, ?, ?, ?)
                on conflict(monitor_id) do update set
                    lease_id=excluded.lease_id,
                    claimed_at=excluded.claimed_at,
                    expires_at=excluded.expires_at
                """,
                (monitor_id, lease_id, now, expires_at),
            )
            return True

    async def release_monitor_tick_lease(
        self,
        monitor_id: str,
        *,
        lease_id: str,
    ) -> None:
        with self._transaction() as conn:
            conn.execute(
                """
                delete from monitor_tick_leases
                where monitor_id = ? and lease_id = ?
                """,
                (monitor_id, lease_id),
            )

    async def commit_monitor_mutation(self, mutation: DbMonitorMutation) -> None:
        with self._transaction() as conn:
            self._validate_expected_monitor(conn, mutation)
            self._validate_expected_state(conn, mutation)
            self._validate_expected_lease(conn, mutation)
            self._apply_monitor_mutation(conn, mutation)
            _upsert_operation(conn, mutation.operation)
            for evidence in mutation.evidence:
                _insert_evidence(conn, evidence)
            for event in mutation.events:
                _insert_event(conn, event)

    def _initialize(self) -> None:
        with self._transaction() as conn:
            conn.executescript(
                """
                create table if not exists monitors (
                    id text primary key,
                    status text not null,
                    updated_at text not null,
                    data text not null
                );
                create table if not exists monitor_states (
                    monitor_id text primary key,
                    data text not null
                );
                create table if not exists monitor_runs (
                    id text primary key,
                    monitor_id text not null,
                    operation_id text not null,
                    status text not null,
                    data text not null
                );
                create table if not exists monitor_tick_leases (
                    monitor_id text primary key,
                    lease_id text not null,
                    claimed_at text not null,
                    expires_at text not null
                );
                """
            )

    def _validate_expected_monitor(
        self,
        conn: sqlite3.Connection,
        mutation: DbMonitorMutation,
    ) -> None:
        monitor_id = _mutation_monitor_id(mutation)
        row = (
            conn.execute(
                "select data from monitors where id = ?", (monitor_id,)
            ).fetchone()
            if monitor_id is not None
            else None
        )
        if mutation.action == "create":
            if row is not None:
                raise ValueError(f"monitor {monitor_id!r} already exists")
            return
        expected = mutation.monitor_before
        if expected is None:
            return
        if row is None:
            raise ValueError(f"monitor {expected.id!r} does not exist")
        if _loads(row["data"]) != expected.to_dict():
            raise ValueError(f"monitor {expected.id!r} changed during mutation")

    def _validate_expected_state(
        self,
        conn: sqlite3.Connection,
        mutation: DbMonitorMutation,
    ) -> None:
        expected = mutation.state_before
        if expected is None:
            return
        row = conn.execute(
            "select data from monitor_states where monitor_id = ?",
            (expected.monitor_id,),
        ).fetchone()
        current = DbMonitorState.from_dict(_loads(row["data"])) if row else None
        if current != expected:
            raise ValueError(
                f"monitor state {expected.monitor_id!r} changed during mutation"
            )

    def _validate_expected_lease(
        self,
        conn: sqlite3.Connection,
        mutation: DbMonitorMutation,
    ) -> None:
        if mutation.lease_id is None:
            return
        monitor_id = _mutation_monitor_id(mutation)
        row = (
            conn.execute(
                """
                select lease_id
                from monitor_tick_leases
                where monitor_id = ?
                """,
                (monitor_id,),
            ).fetchone()
            if monitor_id is not None
            else None
        )
        if row is None or row["lease_id"] != mutation.lease_id:
            raise ValueError(f"monitor {monitor_id!r} tick lease is no longer held")

    def _apply_monitor_mutation(
        self,
        conn: sqlite3.Connection,
        mutation: DbMonitorMutation,
    ) -> None:
        if mutation.monitor_after is not None:
            conn.execute(
                """
                insert into monitors (id, status, updated_at, data)
                values (?, ?, ?, ?)
                on conflict(id) do update set
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    data=excluded.data
                """,
                (
                    mutation.monitor_after.id,
                    mutation.monitor_after.status,
                    mutation.monitor_after.updated_at,
                    _json(mutation.monitor_after.to_dict()),
                ),
            )
        elif mutation.action == "delete" and mutation.monitor_before is not None:
            monitor_id = mutation.monitor_before.id
            conn.execute("delete from monitors where id = ?", (monitor_id,))
            conn.execute(
                "delete from monitor_states where monitor_id = ?", (monitor_id,)
            )
            conn.execute("delete from monitor_runs where monitor_id = ?", (monitor_id,))
            conn.execute(
                "delete from monitor_tick_leases where monitor_id = ?", (monitor_id,)
            )
            return
        if mutation.state_after is not None:
            conn.execute(
                """
                insert into monitor_states (monitor_id, data)
                values (?, ?)
                on conflict(monitor_id) do update set data=excluded.data
                """,
                (
                    mutation.state_after.monitor_id,
                    _json(mutation.state_after.to_dict()),
                ),
            )
        if mutation.run_after is not None:
            conn.execute(
                """
                insert into monitor_runs (id, monitor_id, operation_id, status, data)
                values (?, ?, ?, ?, ?)
                on conflict(id) do update set
                    monitor_id=excluded.monitor_id,
                    operation_id=excluded.operation_id,
                    status=excluded.status,
                    data=excluded.data
                """,
                (
                    mutation.run_after.id,
                    mutation.run_after.monitor_id,
                    mutation.run_after.operation_id,
                    mutation.run_after.status,
                    _json(mutation.run_after.to_dict()),
                ),
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _transaction(self) -> sqlite3.Connection:
        return _Transaction(self)


def monitor_with_updates(monitor: DbMonitor, patch: dict[str, Any]) -> DbMonitor:
    """Return a validated monitor with a shallow patch applied."""
    if "id" in patch and patch["id"] != monitor.id:
        raise ValueError("monitor id cannot be changed")
    values = monitor.to_dict()
    values.update(patch)
    values["id"] = monitor.id
    values["created_at"] = monitor.created_at
    values["updated_at"] = _now_iso()
    return DbMonitor.from_dict(values)


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True)


def _loads(data: str) -> dict[str, Any]:
    return json.loads(data)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mutation_monitor_id(mutation: DbMonitorMutation) -> str | None:
    if mutation.monitor_after is not None:
        return mutation.monitor_after.id
    if mutation.monitor_before is not None:
        return mutation.monitor_before.id
    if mutation.state_after is not None:
        return mutation.state_after.monitor_id
    if mutation.run_after is not None:
        return mutation.run_after.monitor_id
    return None


def _upsert_operation(conn: sqlite3.Connection, operation: Operation) -> None:
    conn.execute(
        """
        insert into operations (id, data)
        values (?, ?)
        on conflict(id) do update set data=excluded.data
        """,
        (operation.id, _json(operation.to_dict())),
    )


def _insert_evidence(conn: sqlite3.Connection, evidence: Evidence) -> None:
    conn.execute(
        """
        insert into evidence (evidence_id, operation_id, task_id, data)
        values (?, ?, ?, ?)
        """,
        (
            evidence.id,
            evidence.operation_id,
            evidence.task_id,
            _json(evidence.to_dict()),
        ),
    )


def _insert_event(conn: sqlite3.Connection, event: RuntimeEvent) -> None:
    conn.execute(
        "insert into events (operation_id, task_id, data) values (?, ?, ?)",
        (event.operation_id, event.task_id, _json(event.to_dict())),
    )


class _Transaction:
    def __init__(self, store: SQLiteDbMonitorStore) -> None:
        self._store = store
        self._conn: sqlite3.Connection | None = None

    def __enter__(self) -> sqlite3.Connection:
        self._store._lock.acquire()
        self._conn = self._store._connect()
        self._conn.execute("begin immediate")
        return self._conn

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self._conn is not None
        try:
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()
        finally:
            self._conn.close()
            self._store._lock.release()
