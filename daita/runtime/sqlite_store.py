"""SQLite-backed runtime operation store."""

from __future__ import annotations

import asyncio
from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any

from .primitives import (
    ApprovalRequest,
    Evidence,
    GovernanceAuditRecord,
    Operation,
    OperationStatus,
    PolicyDecision,
    RuntimeEvent,
    RuntimeStore,
    Task,
    TaskStatus,
)
from .store import OperationSnapshot


class SQLiteRuntimeStore(RuntimeStore):
    """Durable local runtime store backed by SQLite."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    async def save_operation(self, operation: Operation) -> None:
        return await asyncio.to_thread(self._save_operation_sync, operation)

    def _save_operation_sync(self, operation: Operation) -> None:
        with self._transaction() as conn:
            conn.execute(
                """
                insert into operations (id, data)
                values (?, ?)
                on conflict(id) do update set data=excluded.data
                """,
                (operation.id, _json(operation.to_dict())),
            )

    async def load_operation(self, operation_id: str) -> Operation | None:
        return await asyncio.to_thread(self._load_operation_sync, operation_id)

    def _load_operation_sync(self, operation_id: str) -> Operation | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "select data from operations where id = ?", (operation_id,)
            ).fetchone()
        return Operation.from_dict(_loads(row["data"])) if row else None

    async def list_operations(self) -> list[Operation]:
        return await asyncio.to_thread(self._list_operations_sync)

    def _list_operations_sync(self) -> list[Operation]:
        with closing(self._connect()) as conn:
            rows = conn.execute("select data from operations order by rowid").fetchall()
        return [Operation.from_dict(_loads(row["data"])) for row in rows]

    async def save_task(self, task: Task) -> None:
        return await asyncio.to_thread(self._save_task_sync, task)

    def _save_task_sync(self, task: Task) -> None:
        with self._transaction() as conn:
            conn.execute(
                """
                insert into tasks (id, operation_id, status, data)
                values (?, ?, ?, ?)
                on conflict(id) do update set
                    operation_id=excluded.operation_id,
                    status=excluded.status,
                    data=excluded.data
                """,
                (
                    task.id,
                    task.operation_id,
                    task.status.value,
                    _json(task.to_dict()),
                ),
            )

    async def load_task(self, task_id: str) -> Task | None:
        return await asyncio.to_thread(self._load_task_sync, task_id)

    def _load_task_sync(self, task_id: str) -> Task | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "select data from tasks where id = ?", (task_id,)
            ).fetchone()
        return Task.from_dict(_loads(row["data"])) if row else None

    async def list_tasks(self, operation_id: str | None = None) -> list[Task]:
        return await asyncio.to_thread(self._list_tasks_sync, operation_id)

    def _list_tasks_sync(self, operation_id: str | None = None) -> list[Task]:
        sql = "select data from tasks"
        params: tuple[Any, ...] = ()
        if operation_id is not None:
            sql += " where operation_id = ?"
            params = (operation_id,)
        sql += " order by rowid"
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [Task.from_dict(_loads(row["data"])) for row in rows]

    async def claim_task(
        self,
        task_id: str,
        *,
        lease_id: str | None = None,
        lease_owner: str,
        lease_expires_at: float | None = None,
        worker_id: str | None = None,
        worker_owner: str | None = None,
    ) -> Task | None:
        return await asyncio.to_thread(
            self._claim_task_sync,
            task_id,
            lease_id=lease_id,
            lease_owner=lease_owner,
            lease_expires_at=lease_expires_at,
            worker_id=worker_id,
            worker_owner=worker_owner,
        )

    def _claim_task_sync(
        self,
        task_id: str,
        *,
        lease_id: str | None = None,
        lease_owner: str,
        lease_expires_at: float | None = None,
        worker_id: str | None = None,
        worker_owner: str | None = None,
    ) -> Task | None:
        with self._transaction() as conn:
            row = conn.execute(
                "select data from tasks where id = ?", (task_id,)
            ).fetchone()
            if row is None:
                return None
            task = Task.from_dict(_loads(row["data"]))
            if task.status not in {TaskStatus.PENDING, TaskStatus.BLOCKED}:
                return None
            lease_metadata = {
                "attempt_count": int(task.metadata.get("attempt_count") or 0) + 1,
                "lease_id": lease_id,
                "lease_owner": lease_owner,
                "lease_expires_at": lease_expires_at,
                "claimed_at": time.time(),
            }
            if worker_id is not None:
                lease_metadata["worker_id"] = worker_id
            if worker_owner is not None:
                lease_metadata["worker_owner"] = worker_owner
            claimed = replace(
                task,
                status=TaskStatus.RUNNING,
                metadata={
                    **task.metadata,
                    **lease_metadata,
                },
            )
            cursor = conn.execute(
                """
                update tasks
                set status = ?, data = ?
                where id = ? and status in (?, ?)
                """,
                (
                    claimed.status.value,
                    _json(claimed.to_dict()),
                    task_id,
                    TaskStatus.PENDING.value,
                    TaskStatus.BLOCKED.value,
                ),
            )
            if cursor.rowcount < 1:
                return None
            return claimed

    async def heartbeat_task(
        self,
        task_id: str,
        *,
        lease_id: str,
        lease_expires_at: float,
    ) -> Task | None:
        return await asyncio.to_thread(
            self._heartbeat_task_sync,
            task_id,
            lease_id=lease_id,
            lease_expires_at=lease_expires_at,
        )

    def _heartbeat_task_sync(
        self,
        task_id: str,
        *,
        lease_id: str,
        lease_expires_at: float,
    ) -> Task | None:
        with self._transaction() as conn:
            row = conn.execute(
                "select data from tasks where id = ?", (task_id,)
            ).fetchone()
            if row is None:
                return None
            task = Task.from_dict(_loads(row["data"]))
            if task.status is not TaskStatus.RUNNING:
                return None
            if task.metadata.get("lease_id") != lease_id:
                return None
            updated = replace(
                task,
                metadata={
                    **task.metadata,
                    "lease_expires_at": lease_expires_at,
                    "heartbeat_at": time.time(),
                },
            )
            cursor = conn.execute(
                "update tasks set status = ?, data = ? where id = ?",
                (updated.status.value, _json(updated.to_dict()), task_id),
            )
            if cursor.rowcount < 1:
                return None
            return updated

    async def commit_task_blocked(
        self,
        *,
        operation: Operation | None,
        task: Task,
        events: tuple[RuntimeEvent, ...],
        lease_id: str | None = None,
    ) -> bool:
        return await asyncio.to_thread(
            self._commit_task_blocked_sync,
            operation=operation,
            task=task,
            events=events,
            lease_id=lease_id,
        )

    def _commit_task_blocked_sync(
        self,
        *,
        operation: Operation | None,
        task: Task,
        events: tuple[RuntimeEvent, ...],
        lease_id: str | None = None,
    ) -> bool:
        with self._transaction() as conn:
            if not self._lease_matches(conn, task.id, lease_id):
                return False
            if operation is not None:
                self._upsert_operation(conn, operation)
            self._upsert_task(conn, task)
            for event in events:
                self._insert_event(conn, event)
            return True

    async def commit_task_started(self, task: Task, event: RuntimeEvent) -> None:
        return await asyncio.to_thread(self._commit_task_started_sync, task, event)

    def _commit_task_started_sync(self, task: Task, event: RuntimeEvent) -> None:
        with self._transaction() as conn:
            self._upsert_task(conn, task)
            self._insert_event(conn, event)

    async def commit_task_succeeded(
        self,
        task: Task,
        evidence: tuple[Evidence, ...],
        event: RuntimeEvent,
        *,
        lease_id: str | None = None,
    ) -> bool:
        return await asyncio.to_thread(
            self._commit_task_succeeded_sync,
            task,
            evidence,
            event,
            lease_id=lease_id,
        )

    def _commit_task_succeeded_sync(
        self,
        task: Task,
        evidence: tuple[Evidence, ...],
        event: RuntimeEvent,
        *,
        lease_id: str | None = None,
    ) -> bool:
        with self._transaction() as conn:
            if not self._lease_matches(conn, task.id, lease_id):
                return False
            self._upsert_task(conn, task)
            for item in evidence:
                self._insert_evidence(conn, item)
            self._insert_event(conn, event)
            return True

    async def commit_task_failed(
        self,
        task: Task,
        event: RuntimeEvent,
        *,
        lease_id: str | None = None,
    ) -> bool:
        return await asyncio.to_thread(
            self._commit_task_failed_sync,
            task,
            event,
            lease_id=lease_id,
        )

    def _commit_task_failed_sync(
        self,
        task: Task,
        event: RuntimeEvent,
        *,
        lease_id: str | None = None,
    ) -> bool:
        with self._transaction() as conn:
            if not self._lease_matches(conn, task.id, lease_id):
                return False
            self._upsert_task(conn, task)
            self._insert_event(conn, event)
            return True

    async def commit_approval_update(
        self,
        request: ApprovalRequest,
        event: RuntimeEvent,
    ) -> None:
        return await asyncio.to_thread(
            self._commit_approval_update_sync, request, event
        )

    def _commit_approval_update_sync(
        self,
        request: ApprovalRequest,
        event: RuntimeEvent,
    ) -> None:
        with self._transaction() as conn:
            self._upsert_approval_request(conn, request)
            self._insert_event(conn, event)

    async def save_evidence(self, evidence: Evidence) -> None:
        return await asyncio.to_thread(self._save_evidence_sync, evidence)

    def _save_evidence_sync(self, evidence: Evidence) -> None:
        with self._transaction() as conn:
            self._insert_evidence(conn, evidence)

    async def append_event(self, event: RuntimeEvent) -> None:
        return await asyncio.to_thread(self._append_event_sync, event)

    def _append_event_sync(self, event: RuntimeEvent) -> None:
        with self._transaction() as conn:
            self._insert_event(conn, event)

    async def save_policy_decision(self, decision: PolicyDecision) -> None:
        return await asyncio.to_thread(self._save_policy_decision_sync, decision)

    def _save_policy_decision_sync(self, decision: PolicyDecision) -> None:
        with self._transaction() as conn:
            conn.execute(
                """
                insert into policy_decisions (operation_id, data)
                values (?, ?)
                """,
                (decision.operation_id, _json(decision.to_dict())),
            )

    async def save_governance_audit_record(self, record: GovernanceAuditRecord) -> None:
        return await asyncio.to_thread(self._save_governance_audit_record_sync, record)

    def _save_governance_audit_record_sync(self, record: GovernanceAuditRecord) -> None:
        with self._transaction() as conn:
            self._insert_governance_audit_record(conn, record)

    async def commit_governance_evaluation(
        self,
        *,
        decisions: tuple[PolicyDecision, ...],
        audit_record: GovernanceAuditRecord,
        approval_requests: tuple[ApprovalRequest, ...],
        events: tuple[RuntimeEvent, ...],
    ) -> None:
        return await asyncio.to_thread(
            self._commit_governance_evaluation_sync,
            decisions=decisions,
            audit_record=audit_record,
            approval_requests=approval_requests,
            events=events,
        )

    def _commit_governance_evaluation_sync(
        self,
        *,
        decisions: tuple[PolicyDecision, ...],
        audit_record: GovernanceAuditRecord,
        approval_requests: tuple[ApprovalRequest, ...],
        events: tuple[RuntimeEvent, ...],
    ) -> None:
        with self._transaction() as conn:
            for decision in decisions:
                conn.execute(
                    """
                    insert into policy_decisions (operation_id, data)
                    values (?, ?)
                    """,
                    (decision.operation_id, _json(decision.to_dict())),
                )
            self._insert_governance_audit_record(conn, audit_record)
            for request in approval_requests:
                self._upsert_approval_request(conn, request)
            for event in events:
                self._insert_event(conn, event)

    async def commit_governance_blocked(
        self,
        *,
        operation: Operation,
        task: Task | None,
        decisions: tuple[PolicyDecision, ...],
        audit_record: GovernanceAuditRecord,
        approval_requests: tuple[ApprovalRequest, ...],
        events: tuple[RuntimeEvent, ...],
    ) -> None:
        return await asyncio.to_thread(
            self._commit_governance_blocked_sync,
            operation=operation,
            task=task,
            decisions=decisions,
            audit_record=audit_record,
            approval_requests=approval_requests,
            events=events,
        )

    def _commit_governance_blocked_sync(
        self,
        *,
        operation: Operation,
        task: Task | None,
        decisions: tuple[PolicyDecision, ...],
        audit_record: GovernanceAuditRecord,
        approval_requests: tuple[ApprovalRequest, ...],
        events: tuple[RuntimeEvent, ...],
    ) -> None:
        with self._transaction() as conn:
            self._upsert_operation(conn, operation)
            if task is not None:
                self._upsert_task(conn, task)
            for decision in decisions:
                conn.execute(
                    """
                    insert into policy_decisions (operation_id, data)
                    values (?, ?)
                    """,
                    (decision.operation_id, _json(decision.to_dict())),
                )
            self._insert_governance_audit_record(conn, audit_record)
            for request in approval_requests:
                self._upsert_approval_request(conn, request)
            for event in events:
                self._insert_event(conn, event)

    async def save_approval_request(self, request: ApprovalRequest) -> None:
        return await asyncio.to_thread(self._save_approval_request_sync, request)

    def _save_approval_request_sync(self, request: ApprovalRequest) -> None:
        with self._transaction() as conn:
            self._upsert_approval_request(conn, request)

    async def list_evidence(self, operation_id: str) -> list[Evidence]:
        return await asyncio.to_thread(self._list_evidence_sync, operation_id)

    def _list_evidence_sync(self, operation_id: str) -> list[Evidence]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "select data from evidence where operation_id = ? order by rowid",
                (operation_id,),
            ).fetchall()
        return [Evidence.from_dict(_loads(row["data"])) for row in rows]

    async def list_events(self, operation_id: str | None = None) -> list[RuntimeEvent]:
        return await asyncio.to_thread(self._list_events_sync, operation_id)

    def _list_events_sync(self, operation_id: str | None = None) -> list[RuntimeEvent]:
        sql = "select data from events"
        params: tuple[Any, ...] = ()
        if operation_id is not None:
            sql += " where operation_id = ?"
            params = (operation_id,)
        sql += " order by rowid"
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [RuntimeEvent.from_dict(_loads(row["data"])) for row in rows]

    async def list_policy_decisions(
        self, operation_id: str | None = None
    ) -> list[PolicyDecision]:
        return await asyncio.to_thread(self._list_policy_decisions_sync, operation_id)

    def _list_policy_decisions_sync(
        self, operation_id: str | None = None
    ) -> list[PolicyDecision]:
        sql = "select data from policy_decisions"
        params: tuple[Any, ...] = ()
        if operation_id is not None:
            sql += " where operation_id = ?"
            params = (operation_id,)
        sql += " order by rowid"
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [PolicyDecision.from_dict(_loads(row["data"])) for row in rows]

    async def list_governance_audit_records(
        self, operation_id: str | None = None
    ) -> list[GovernanceAuditRecord]:
        return await asyncio.to_thread(
            self._list_governance_audit_records_sync, operation_id
        )

    def _list_governance_audit_records_sync(
        self, operation_id: str | None = None
    ) -> list[GovernanceAuditRecord]:
        sql = "select data from governance_audit_records"
        params: tuple[Any, ...] = ()
        if operation_id is not None:
            sql += " where operation_id = ?"
            params = (operation_id,)
        sql += " order by rowid"
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [GovernanceAuditRecord.from_dict(_loads(row["data"])) for row in rows]

    async def list_approval_requests(
        self, operation_id: str | None = None
    ) -> list[ApprovalRequest]:
        return await asyncio.to_thread(self._list_approval_requests_sync, operation_id)

    def _list_approval_requests_sync(
        self, operation_id: str | None = None
    ) -> list[ApprovalRequest]:
        sql = "select data from approval_requests"
        params: tuple[Any, ...] = ()
        if operation_id is not None:
            sql += " where operation_id = ?"
            params = (operation_id,)
        sql += " order by rowid"
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [ApprovalRequest.from_dict(_loads(row["data"])) for row in rows]

    async def inspect_operation(self, operation_id: str) -> OperationSnapshot | None:
        return await asyncio.to_thread(self._inspect_operation_sync, operation_id)

    def _inspect_operation_sync(self, operation_id: str) -> OperationSnapshot | None:
        operation = self._load_operation_sync(operation_id)
        if operation is None:
            return None
        tasks = tuple(self._list_tasks_sync(operation_id))
        completed_task_ids = tuple(
            task.id
            for task in tasks
            if task.status
            in {
                TaskStatus.SUCCEEDED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.SKIPPED,
            }
        )
        resumable_task_ids = tuple(
            task.id
            for task in tasks
            if task.status in {TaskStatus.PENDING, TaskStatus.BLOCKED}
        )
        return OperationSnapshot(
            operation=operation,
            tasks=tasks,
            evidence=tuple(self._list_evidence_sync(operation_id)),
            events=tuple(self._list_events_sync(operation_id)),
            policy_decisions=tuple(self._list_policy_decisions_sync(operation_id)),
            governance_audit_records=tuple(
                self._list_governance_audit_records_sync(operation_id)
            ),
            approval_requests=tuple(self._list_approval_requests_sync(operation_id)),
            resumable_task_ids=resumable_task_ids,
            completed_task_ids=completed_task_ids,
            metadata={
                "operation_status": operation.status.value,
                "resumable": operation.status
                in {OperationStatus.BLOCKED, OperationStatus.RUNNING},
            },
        )

    def _initialize(self) -> None:
        with self._transaction() as conn:
            conn.executescript("""
                create table if not exists operations (
                    id text primary key,
                    data text not null
                );
                create table if not exists tasks (
                    id text primary key,
                    operation_id text not null,
                    status text not null,
                    data text not null
                );
                create table if not exists evidence (
                    evidence_id text,
                    operation_id text,
                    task_id text,
                    data text not null
                );
                create table if not exists events (
                    operation_id text not null,
                    task_id text,
                    data text not null
                );
                create table if not exists policy_decisions (
                    operation_id text,
                    data text not null
                );
                create table if not exists governance_audit_records (
                    audit_id text primary key,
                    operation_id text not null,
                    data text not null
                );
                create table if not exists approval_requests (
                    approval_id text primary key,
                    operation_id text not null,
                    status text not null,
                    data text not null
                );
                """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _transaction(self) -> _Transaction:
        return _Transaction(self)

    @staticmethod
    def _upsert_operation(conn: sqlite3.Connection, operation: Operation) -> None:
        conn.execute(
            """
            insert into operations (id, data)
            values (?, ?)
            on conflict(id) do update set data=excluded.data
            """,
            (operation.id, _json(operation.to_dict())),
        )

    @staticmethod
    def _upsert_task(conn: sqlite3.Connection, task: Task) -> None:
        conn.execute(
            """
            insert into tasks (id, operation_id, status, data)
            values (?, ?, ?, ?)
            on conflict(id) do update set
                operation_id=excluded.operation_id,
                status=excluded.status,
                data=excluded.data
            """,
            (
                task.id,
                task.operation_id,
                task.status.value,
                _json(task.to_dict()),
            ),
        )

    @staticmethod
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

    @staticmethod
    def _insert_event(conn: sqlite3.Connection, event: RuntimeEvent) -> None:
        conn.execute(
            "insert into events (operation_id, task_id, data) values (?, ?, ?)",
            (event.operation_id, event.task_id, _json(event.to_dict())),
        )

    @staticmethod
    def _insert_governance_audit_record(
        conn: sqlite3.Connection,
        record: GovernanceAuditRecord,
    ) -> None:
        conn.execute(
            """
            insert into governance_audit_records (audit_id, operation_id, data)
            values (?, ?, ?)
            """,
            (record.audit_id, record.operation_id, _json(record.to_dict())),
        )

    @staticmethod
    def _upsert_approval_request(
        conn: sqlite3.Connection,
        request: ApprovalRequest,
    ) -> None:
        conn.execute(
            """
            insert into approval_requests (approval_id, operation_id, status, data)
            values (?, ?, ?, ?)
            on conflict(approval_id) do update set
                operation_id=excluded.operation_id,
                status=excluded.status,
                data=excluded.data
            """,
            (
                request.approval_id,
                request.operation_id,
                request.status_value,
                _json(request.to_dict()),
            ),
        )

    @staticmethod
    def _lease_matches(
        conn: sqlite3.Connection,
        task_id: str,
        lease_id: str | None,
    ) -> bool:
        if lease_id is None:
            return True
        row = conn.execute("select data from tasks where id = ?", (task_id,)).fetchone()
        if row is None:
            return False
        current = Task.from_dict(_loads(row["data"]))
        return (
            current.status is TaskStatus.RUNNING
            and current.metadata.get("lease_id") == lease_id
        )


class _Transaction:
    def __init__(self, store: SQLiteRuntimeStore) -> None:
        self.store = store
        self.conn: sqlite3.Connection | None = None

    def __enter__(self) -> sqlite3.Connection:
        self.store._lock.acquire()
        self.conn = self.store._connect()
        self.conn.execute("begin immediate")
        return self.conn

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        assert self.conn is not None
        try:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
        finally:
            self.conn.close()
            self.store._lock.release()


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True)


def _loads(data: str) -> dict[str, Any]:
    loaded = json.loads(data)
    if not isinstance(loaded, dict):
        raise TypeError("runtime store records must decode to objects")
    return loaded
