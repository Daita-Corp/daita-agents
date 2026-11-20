"""Effect-receipt persistence for the SQL state store.

Every function runs inside a caller-owned transaction or read connection;
``SQLStateStore`` owns the transaction boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime

from ..capabilities import EffectEvidenceBasis, EffectObservation, ExecutionScopeKind
from ..jobs.models import JobState
from ..routines.models import RoutineOccurrenceDisposition, RoutineState
from . import sql_jobs as _job_store
from .sql_connection import SQLConnection, required_row
from .sql_routines import (
    _load_routine_occurrence_row,
    _load_routine_row,
    _replace_routine_row,
    _require_effects_unblocked,
)
from .sqlite_codecs import (
    decode_identity,
    decode_receipt,
    decode_run_input,
    encode_receipt,
)
from .sqlite_records import (
    EffectOutcome,
    EffectReceipt,
    EffectReceiptConflictError,
    EffectResolution,
    EffectResolutionDecision,
)


def recover_started(
    connection: SQLConnection, agent_id: str, completed_at: datetime
) -> None:
    rows = tuple(
        connection.execute(
            "SELECT agent_id, id, data, job_id FROM effect_receipts WHERE agent_id = ?",
            (agent_id,),
        )
    )
    started = tuple(
        (agent_id, receipt_id, receipt, job_id)
        for (agent_id, receipt_id, data, job_id) in rows
        if (receipt := decode_receipt(data)).outcome is EffectOutcome.STARTED
    )
    for agent_id, receipt_id, receipt, job_id in started:
        recovered = receipt.finish(
            EffectObservation(EffectOutcome.UNCERTAIN, EffectEvidenceBasis.UNKNOWN),
            finished_at=completed_at,
        )
        result = connection.execute(
            """UPDATE effect_receipts SET data = ?
               WHERE agent_id = ? AND id = ? AND data = ?""",
            (
                encode_receipt(recovered),
                agent_id,
                receipt_id,
                encode_receipt(receipt),
            ),
        )
        if result.rowcount != 1:
            raise RuntimeError("effect receipt changed during startup recovery")
        _pause_routine(connection, recovered, completed_at)
        if job_id is not None:
            _job_store.wait_for_uncertain_write(
                connection, agent_id, str(job_id), completed_at
            )


def job_receipts(
    connection: SQLConnection, agent_id: str, job_id: str
) -> tuple[EffectReceipt, ...]:
    rows = connection.execute(
        "SELECT data FROM effect_receipts "
        "WHERE agent_id = ? AND job_id = ? "
        f"ORDER BY {connection.insertion_order} LIMIT 257",
        (agent_id, job_id),
    ).fetchall()
    return tuple(decode_receipt(row[0]) for row in rows[:256])


def load_receipt(
    connection: SQLConnection, agent_id: str, receipt_id: str
) -> EffectReceipt | None:
    row = connection.execute(
        """SELECT data FROM effect_receipts
           WHERE agent_id = ? AND id = ?""",
        (agent_id, receipt_id),
    ).fetchone()
    return None if row is None else decode_receipt(row[0])


def load_receipt_for_call(
    connection: SQLConnection, agent_id: str, run_id: str, call_id: str
) -> EffectReceipt | None:
    row = connection.execute(
        """SELECT data FROM effect_receipts
           WHERE agent_id = ? AND run_id = ? AND call_id = ?""",
        (agent_id, run_id, call_id),
    ).fetchone()
    return None if row is None else decode_receipt(row[0])


def load_receipt_for_operation(
    connection: SQLConnection, agent_id: str, operation_key: str
) -> EffectReceipt | None:
    row = connection.execute(
        "SELECT data FROM effect_receipts WHERE agent_id = ? AND operation_key = ?",
        (agent_id, operation_key),
    ).fetchone()
    return None if row is None else decode_receipt(row[0])


def list_receipts(
    connection: SQLConnection,
    agent_id: str,
    *,
    run_id: str | None,
    routine_id: str | None,
    unresolved_only: bool,
    limit: int,
    offset: int,
    caller_principal_id: str | None,
) -> tuple[EffectReceipt, ...]:
    if caller_principal_id is None:
        rows = connection.execute(
            "SELECT data FROM effect_receipts WHERE agent_id = ? AND (CAST(? AS TEXT) IS NULL OR run_id = ?) AND (CAST(? AS TEXT) IS NULL OR routine_id = ?) AND (? = 0 OR unresolved = 1) ORDER BY id LIMIT ? OFFSET ?",
            (
                agent_id,
                run_id,
                run_id,
                routine_id,
                routine_id,
                int(unresolved_only),
                limit,
                offset,
            ),
        )
    else:
        rows = connection.execute(
            "SELECT effect_receipts.data FROM effect_receipts "
            "JOIN runs ON runs.id = effect_receipts.run_id "
            "AND runs.agent_id = effect_receipts.agent_id "
            "WHERE effect_receipts.agent_id = ? "
            f"AND {connection.caller_expression('runs.input')} = ? "
            "AND (CAST(? AS TEXT) IS NULL OR effect_receipts.run_id = ?) "
            "AND (CAST(? AS TEXT) IS NULL OR effect_receipts.routine_id = ?) "
            "AND (? = 0 OR effect_receipts.unresolved = 1) "
            "ORDER BY effect_receipts.id LIMIT ? OFFSET ?",
            (
                agent_id,
                caller_principal_id,
                run_id,
                run_id,
                routine_id,
                routine_id,
                int(unresolved_only),
                limit,
                offset,
            ),
        )
    return tuple(decode_receipt(row[0]) for row in rows)


def start_receipt(
    connection: SQLConnection, receipt: EffectReceipt, max_receipts_per_run: int
) -> EffectReceipt:
    identity = connection.execute(
        "SELECT data FROM metadata WHERE key = 'identity'"
    ).fetchone()
    run_row = connection.execute(
        "SELECT input, result FROM runs WHERE agent_id = ? AND id = ?",
        (receipt.agent_id, receipt.run_id),
    ).fetchone()
    if (
        identity is None
        or decode_identity(identity[0]).id != receipt.agent_id
        or run_row is None
        or run_row[1] is not None
    ):
        raise EffectReceiptConflictError(
            "receipt requires its current owned nonterminal run"
        )
    run = decode_run_input(run_row[0])
    _require_effects_unblocked(
        connection,
        receipt.agent_id,
        run_id=receipt.run_id,
        routine_id=receipt.routine_id,
    )
    existing = connection.execute(
        "SELECT id FROM effect_receipts WHERE agent_id = ? AND (id = ? OR (run_id = ? AND call_id = ?) OR operation_key = ?)",
        (
            receipt.agent_id,
            receipt.receipt_id,
            receipt.run_id,
            receipt.call_id,
            receipt.operation_key,
        ),
    ).fetchone()
    if existing is not None:
        raise EffectReceiptConflictError(
            "execution or operation identity already has a receipt"
        )
    reserved_count = required_row(
        connection.execute(
            "SELECT COUNT(*) FROM effect_receipts WHERE agent_id = ? AND run_id = ?",
            (receipt.agent_id, receipt.run_id),
        )
    )[0]
    if reserved_count >= max_receipts_per_run:
        raise EffectReceiptConflictError(
            "the run receipt reservation bound is exhausted"
        )
    scope = run.start.execution_scope if run.start is not None else None
    job_id = (
        scope.job_id
        if scope is not None and scope.scope_kind is ExecutionScopeKind.JOB
        else None
    )
    if receipt.capability_grant_digest is not None:
        raise EffectReceiptConflictError(
            "effects are approved at the time; a receipt cannot carry a grant"
        )
    if job_id is not None:
        job = _job_store.load_job(connection, receipt.agent_id, job_id)
        if (
            receipt.routine_id is not None
            or job is None
            or job.state is not JobState.RUNNING
        ):
            raise EffectReceiptConflictError("job receipt requires its running job")
    elif scope is not None:
        routine_id = receipt.routine_id
        if routine_id is None:
            raise EffectReceiptConflictError(
                "machine receipt lacks routine or job authorization"
            )
        if (
            scope.routine_id != receipt.routine_id
            or scope.routine_revision != receipt.routine_revision
            or scope.occurrence_id != receipt.occurrence_id
        ):
            raise EffectReceiptConflictError(
                "receipt does not match its scope authorization"
            )
        occurrence = _load_routine_occurrence_row(
            connection, receipt.agent_id, receipt.occurrence_id or ""
        )
        routine = _load_routine_row(connection, receipt.agent_id, routine_id)
        if (
            occurrence is None
            or routine is None
            or (
                occurrence[0].reserved_run_id != receipt.run_id
                or occurrence[0].routine_revision != receipt.routine_revision
                or occurrence[0].disposition is not RoutineOccurrenceDisposition.RUNNING
                or occurrence[0].claim_token is None
                # The lease fences unbound claims. Once bound, the
                # immutable run/scope and nonterminal run row fence
                # dispatch; model reasoning may outlast the claim lease.
                or routine[0].revision != receipt.routine_revision
                or routine[0].active_occurrence_id != receipt.occurrence_id
                or routine[0].state is not RoutineState.ACTIVE
            )
        ):
            raise EffectReceiptConflictError(
                "receipt occurrence claim is no longer current"
            )
    connection.execute(
        "INSERT INTO effect_receipts(agent_id, id, run_id, call_id, operation_key, routine_id, occurrence_id, grant_digest, unresolved, job_id, data) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
        (
            receipt.agent_id,
            receipt.receipt_id,
            receipt.run_id,
            receipt.call_id,
            receipt.operation_key,
            receipt.routine_id,
            receipt.occurrence_id,
            receipt.capability_grant_digest,
            job_id,
            encode_receipt(receipt),
        ),
    )
    return receipt


def finish_receipt(
    connection: SQLConnection,
    receipt: EffectReceipt,
    clock: Callable[[], datetime],
) -> EffectReceipt:
    row = connection.execute(
        "SELECT data, job_id FROM effect_receipts WHERE agent_id = ? AND id = ?",
        (receipt.agent_id, receipt.receipt_id),
    ).fetchone()
    if row is None:
        raise EffectReceiptConflictError("started receipt does not exist")
    current = decode_receipt(row[0])
    if current.outcome is not EffectOutcome.STARTED:
        if replace(current, resolution=None) == receipt:
            return current
        raise EffectReceiptConflictError("terminal receipt is immutable")
    if current != receipt.as_started():
        raise EffectReceiptConflictError(
            "terminal receipt does not match its started identity"
        )
    connection.execute(
        "UPDATE effect_receipts SET data = ?, unresolved = ? WHERE agent_id = ? AND id = ?",
        (
            encode_receipt(receipt),
            int(receipt.unresolved),
            receipt.agent_id,
            receipt.receipt_id,
        ),
    )
    if receipt.unresolved:
        _pause_routine(connection, receipt, receipt.finished_at or clock())
        if row[1] is not None:
            _job_store.wait_for_uncertain_write(
                connection,
                receipt.agent_id,
                str(row[1]),
                receipt.finished_at or clock(),
            )
    return receipt


def resolve_receipt(
    connection: SQLConnection, agent_id: str, resolution: EffectResolution
) -> EffectReceipt:
    row = connection.execute(
        "SELECT data, job_id FROM effect_receipts WHERE agent_id = ? AND id = ?",
        (agent_id, resolution.receipt_id),
    ).fetchone()
    if row is None:
        raise EffectReceiptConflictError("the exact owned receipt is unavailable")
    receipt = decode_receipt(row[0])
    if receipt.receipt_digest != resolution.receipt_digest:
        raise EffectReceiptConflictError(
            "the receipt observation changed before recovery"
        )
    if receipt.resolution is not None:
        if receipt.resolution == resolution:
            return receipt
        raise EffectReceiptConflictError("an existing human resolution is immutable")
    if receipt.outcome is not EffectOutcome.UNCERTAIN:
        raise EffectReceiptConflictError("only terminal uncertainty requires recovery")
    resolved = replace(receipt, resolution=resolution)
    connection.execute(
        "UPDATE effect_receipts SET data = ?, unresolved = 0 WHERE agent_id = ? AND id = ?",
        (encode_receipt(resolved), agent_id, receipt.receipt_id),
    )
    if receipt.routine_id is not None:
        loaded = _load_routine_row(connection, agent_id, receipt.routine_id)
        if loaded is None:
            raise EffectReceiptConflictError("the producing routine is unavailable")
        routine, data = loaded
        state = (
            RoutineState.DISABLED
            if resolution.decision is EffectResolutionDecision.CLOSE_WITHOUT_RETRY
            else RoutineState.PAUSED
        )
        _replace_routine_row(
            connection,
            data,
            replace(routine, state=state, updated_at=resolution.resolved_at),
        )
    if row[1] is not None:
        _job_store.release_uncertain_write(
            connection, agent_id, str(row[1]), resolution.resolved_at
        )
    return resolved


def _pause_routine(
    connection: SQLConnection, receipt: EffectReceipt, changed_at: datetime
) -> None:
    if receipt.routine_id is None:
        return
    loaded = _load_routine_row(connection, receipt.agent_id, receipt.routine_id)
    if loaded is None:
        raise EffectReceiptConflictError("the producing routine is unavailable")
    routine, encoded = loaded
    _replace_routine_row(
        connection,
        encoded,
        replace(
            routine,
            state=RoutineState.PAUSED,
            updated_at=max(changed_at, routine.updated_at),
        ),
    )
