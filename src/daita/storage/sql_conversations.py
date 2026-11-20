"""Run-transcript and conversation persistence for the SQL state store.

Every function runs inside a caller-owned transaction or read connection;
``SQLStateStore`` owns the transaction boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from ..artifacts.models import ArtifactRef, artifact_ref_from_mapping
from ..jobs.models import LIVE_JOB_STATES, MAX_JOB_LIST_PAGE_SIZE
from ..llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelSensitivity,
    ToolResultBlock,
)
from ..loop.models import (
    ConversationRun,
    LoopExit,
    LoopExitKind,
    RunInput,
    Transcript,
    validate_completed_transcript,
)
from ..loop.transcripts import ConversationPredecessor
from ..routines.models import RoutineOccurrenceDisposition
from . import sql_jobs as _job_store
from .sql_connection import SQLConnection, StateIntegrityError
from .sqlite_codecs import (
    decode_loop_exit,
    decode_message,
    decode_routine_occurrence,
    decode_run_input,
    decode_scheduled_routine,
    encode_loop_exit,
    encode_message,
    encode_run_input,
)
from .sqlite_codecs.analysis import decode_analysis_evidence


def start_run(
    connection: SQLConnection,
    run: RunInput,
    predecessor: ConversationPredecessor | None,
    *,
    predecessor_given: bool,
) -> Transcript:
    try:
        row = connection.execute(
            """SELECT id, turn_index, input, result
                   FROM runs
                   WHERE agent_id = ? AND conversation_id = ?
                   ORDER BY turn_index DESC
                   LIMIT 1""",
            (run.agent_id, run.conversation_id),
        ).fetchone()
        if not predecessor_given:
            turn_index = 0 if row is None else int(row[1]) + 1
        elif predecessor is None:
            if row is not None:
                raise ValueError("conversation predecessor changed before run start")
            turn_index = 0
        else:
            if not isinstance(predecessor, ConversationPredecessor):
                raise TypeError("conversation predecessor is invalid")
            if (
                row is None
                or row[0] != predecessor.run_id
                or int(row[1]) != predecessor.turn_index
                or row[3] is None
            ):
                raise ValueError("conversation predecessor changed before run start")
            current_predecessor = ConversationPredecessor.from_run(
                ConversationRun(
                    turn_index=int(row[1]),
                    transcript=Transcript(run=decode_run_input(row[2])),
                    result=decode_loop_exit(row[3]),
                )
            )
            if current_predecessor != predecessor:
                raise ValueError("conversation predecessor changed before run start")
            turn_index = predecessor.turn_index + 1
        connection.execute(
            """INSERT INTO runs(
                       id, agent_id, conversation_id, turn_index, input
                   ) VALUES (?, ?, ?, ?, ?)""",
            (
                run.id,
                run.agent_id,
                run.conversation_id,
                turn_index,
                encode_run_input(run),
            ),
        )
    except StateIntegrityError as error:
        raise ValueError(f"run already exists: {run.id}") from error
    return Transcript(run=run)


def next_position(connection: SQLConnection, run_id: str) -> int:
    row = connection.execute(
        "SELECT COALESCE(MAX(position), -1) + 1 FROM messages WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    assert row is not None
    return int(row[0])


def append_message(
    connection: SQLConnection,
    run_id: str,
    position: int,
    message: CanonicalMessage,
) -> None:
    run_row = connection.execute(
        "SELECT result FROM runs WHERE id = ?", (run_id,)
    ).fetchone()
    if run_row is None:
        raise KeyError(f"unknown run: {run_id}")
    if run_row[0] is not None:
        raise ValueError(f"run is already terminal: {run_id}")
    if type(position) is not int or next_position(connection, run_id) != position:
        raise ValueError("transcript append position is out of order")
    connection.execute(
        "INSERT INTO messages(run_id, position, data) VALUES (?, ?, ?)",
        (run_id, position, encode_message(message)),
    )


def finish_run(connection: SQLConnection, result: LoopExit) -> None:
    cursor = connection.execute(
        """UPDATE runs SET result = ?
           WHERE id = ? AND conversation_id = ? AND result IS NULL""",
        (
            encode_loop_exit(result),
            result.run_id,
            result.conversation_id,
        ),
    )
    if cursor.rowcount != 1:
        raise KeyError(f"unknown run: {result.run_id}")


def complete_run(
    connection: SQLConnection,
    result: LoopExit,
    final_message: CanonicalMessage,
) -> None:
    run_row = connection.execute(
        """SELECT input, result FROM runs
           WHERE id = ? AND conversation_id = ?""",
        (result.run_id, result.conversation_id),
    ).fetchone()
    if run_row is None:
        raise KeyError(f"unknown run: {result.run_id}")
    if run_row[1] is not None:
        raise ValueError(f"run is already terminal: {result.run_id}")
    rows = connection.execute(
        """SELECT data FROM messages
           WHERE run_id = ? ORDER BY position""",
        (result.run_id,),
    ).fetchall()
    transcript = Transcript(
        run=decode_run_input(run_row[0]),
        messages=(
            *(decode_message(row[0]) for row in rows),
            final_message,
        ),
    )
    validate_completed_transcript(transcript, result)
    connection.execute(
        """INSERT INTO messages(run_id, position, data)
           VALUES (?, ?, ?)""",
        (result.run_id, len(rows), encode_message(final_message)),
    )
    cursor = connection.execute(
        """UPDATE runs SET result = ?
           WHERE id = ? AND conversation_id = ? AND result IS NULL""",
        (
            encode_loop_exit(result),
            result.run_id,
            result.conversation_id,
        ),
    )
    if cursor.rowcount != 1:
        raise RuntimeError("run changed during atomic completion")


def recover_unfinished_runs(
    connection: SQLConnection,
    agent_id: str,
    created_at: datetime,
) -> tuple[LoopExit, ...]:
    rows = connection.execute(
        """SELECT id, conversation_id, input
           FROM runs
           WHERE agent_id = ? AND result IS NULL
           ORDER BY conversation_id, turn_index""",
        (agent_id,),
    ).fetchall()
    recovered: list[LoopExit] = []
    for run_id, conversation_id, run_data in rows:
        run = decode_run_input(run_data)
        messages = connection.execute(
            "SELECT data FROM messages WHERE run_id = ? ORDER BY position",
            (run_id,),
        ).fetchall()
        decoded = tuple(decode_message(row[0]) for row in messages)
        steps = sum(message.role is MessageRole.ASSISTANT for message in decoded)
        refs: dict[str, ArtifactRef] = {}
        sensitivity = max(
            run.history_sensitivity,
            # Machine content is bounded by its approved ceiling.
            # An interrupted foreground request has no persisted
            # request classification, so unknown content stays private.
            (
                run.execution_scope.sensitivity_ceiling
                if run.execution_scope is not None
                else ModelSensitivity.RESTRICTED
            ),
            key=lambda item: item.routing_rank,
        )
        for message in decoded:
            if message.role is not MessageRole.TOOL:
                continue
            for block in message.content:
                if not isinstance(block, ToolResultBlock):
                    continue
                if block.sensitivity is not None:
                    sensitivity = max(
                        sensitivity,
                        block.sensitivity,
                        key=lambda item: item.routing_rank,
                    )
                value = block.output.get("artifact")
                if block.is_error or not isinstance(value, Mapping):
                    continue
                ref = artifact_ref_from_mapping(value)
                if (
                    ref.run_id != run_id
                    or ref.conversation_id != conversation_id
                    or ref.call_id != block.call_id
                    or (ref.artifact_id in refs and refs[ref.artifact_id] != ref)
                ):
                    raise RuntimeError(
                        "recovered artifact identity does not match its run"
                    )
                refs[ref.artifact_id] = ref
        result = LoopExit(
            run_id=run_id,
            conversation_id=conversation_id,
            kind=LoopExitKind.INTERRUPTED,
            reason="previous_process_terminated",
            steps=steps,
            created_at=created_at,
            sensitivity=sensitivity,
            artifacts=tuple(refs.values()),
        )
        cursor = connection.execute(
            "UPDATE runs SET result = ? WHERE id = ? AND result IS NULL",
            (encode_loop_exit(result), run_id),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("unfinished run changed during recovery")
        recovered.append(result)
    return tuple(recovered)


def load_transcript(connection: SQLConnection, run_id: str) -> Transcript:
    run_row = connection.execute(
        "SELECT input FROM runs WHERE id = ?", (run_id,)
    ).fetchone()
    if run_row is None:
        raise KeyError(f"unknown run: {run_id}")
    rows = connection.execute(
        "SELECT data FROM messages WHERE run_id = ? ORDER BY position",
        (run_id,),
    ).fetchall()
    return Transcript(
        run=decode_run_input(run_row[0]),
        messages=tuple(decode_message(row[0]) for row in rows),
    )


def load_result(connection: SQLConnection, run_id: str) -> LoopExit | None:
    row = connection.execute(
        "SELECT result FROM runs WHERE id = ?", (run_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"unknown run: {run_id}")
    return None if row[0] is None else decode_loop_exit(row[0])


def conversation_records(
    connection: SQLConnection,
    agent_id: str,
    conversation_id: str,
    *,
    after_turn_index: int = -1,
    limit: int | None = None,
) -> tuple[ConversationRun, ...]:
    # Both statements use the caller's read snapshot. Joining the selected turns
    # fetches all messages in one query instead of one query per turn.
    selection = """SELECT id, turn_index, input, result FROM runs
        WHERE agent_id = ? AND conversation_id = ? AND turn_index > ?
        ORDER BY turn_index LIMIT ?"""
    parameters = (
        agent_id,
        conversation_id,
        after_turn_index,
        limit if limit is not None else 9223372036854775807,
    )
    rows = connection.execute(selection, parameters).fetchall()
    messages: dict[str, list[CanonicalMessage]] = {row[0]: [] for row in rows}
    if rows:
        for run_id, data in connection.execute(
            "WITH selected AS (" + selection + ") "
            "SELECT m.run_id, m.data FROM messages AS m JOIN selected AS s ON s.id = m.run_id "
            "ORDER BY s.turn_index, m.position",
            parameters,
        ):
            messages[run_id].append(decode_message(data))
    return tuple(
        ConversationRun(
            turn_index=int(turn_index),
            transcript=Transcript(
                run=decode_run_input(input_data), messages=tuple(messages[run_id])
            ),
            result=None if result_data is None else decode_loop_exit(result_data),
        )
        for run_id, turn_index, input_data, result_data in rows
    )


def conversation_access(
    connection: SQLConnection,
    agent_id: str,
    conversation_id: str,
    caller_principal_id: str | None,
) -> tuple[bool, bool]:
    row = connection.execute(
        f"""SELECT
            EXISTS(SELECT 1 FROM runs WHERE agent_id = ? AND conversation_id = ?),
            EXISTS(SELECT 1 FROM runs WHERE agent_id = ? AND conversation_id = ?
                AND {connection.caller_expression('input')} {connection.distinct_operator} ?)
        """,
        (
            agent_id,
            conversation_id,
            agent_id,
            conversation_id,
            caller_principal_id,
        ),
    ).fetchone()
    assert row is not None
    exists = bool(row[0])
    return exists, exists and (caller_principal_id is None or not row[1])


def conversation_exists(
    connection: SQLConnection, agent_id: str, conversation_id: str
) -> bool:
    row = connection.execute(
        """SELECT 1 FROM runs
           WHERE agent_id = ? AND conversation_id = ?
           LIMIT 1""",
        (agent_id, conversation_id),
    ).fetchone()
    return row is not None


def clear_conversations(connection: SQLConnection, agent_id: str) -> int:
    # A live job's runs and running sub-runs are its working state.
    live_job_conversations = {
        conversation_id
        for job in _job_store.list_jobs(
            connection,
            agent_id,
            states=LIVE_JOB_STATES,
            limit=MAX_JOB_LIST_PAGE_SIZE,
        )
        for conversation_id in (
            job.job_id,
            *(item.id for item in job.running_delegations),
        )
    }
    protected_run_ids = {
        str(row[0])
        for conversation_id in live_job_conversations
        for row in connection.execute(
            "SELECT id FROM runs WHERE agent_id = ? AND conversation_id = ?",
            (agent_id, conversation_id),
        )
    }
    protected_run_ids.update(
        record.run_id
        for row in connection.execute(
            "SELECT data FROM analysis_evidence WHERE agent_id = ? AND kind = 'generation'",
            (agent_id,),
        )
        if (record := decode_analysis_evidence(row[0])).facts.get("proof_disposed")
        is not True
    )
    occurrence_rows = connection.execute(
        "SELECT occurrence_id, data FROM routine_occurrences " "WHERE agent_id = ?",
        (agent_id,),
    ).fetchall()
    protected_run_ids.update(
        occurrence.reserved_run_id
        for occurrence_id, data in occurrence_rows
        for occurrence in (
            decode_routine_occurrence(
                data,
                agent_id=agent_id,
                occurrence_id=occurrence_id,
            ),
        )
        if occurrence.reserved_run_id is not None
        and occurrence.disposition
        in {
            RoutineOccurrenceDisposition.CLAIMED,
            RoutineOccurrenceDisposition.RUNNING,
            (RoutineOccurrenceDisposition.RUN_TERMINAL_PENDING_FINALIZATION),
            RoutineOccurrenceDisposition.RETRYABLE,
        }
    )
    active_routine_conversations = {
        routine.conversation_id
        for routine_id, data in connection.execute(
            "SELECT routine_id, data FROM scheduled_routines " "WHERE agent_id = ?",
            (agent_id,),
        )
        for routine in (
            decode_scheduled_routine(
                data,
                agent_id=agent_id,
                routine_id=routine_id,
            ),
        )
        if routine.active_occurrence_id is not None
    }
    for conversation_id in active_routine_conversations:
        anchor = connection.execute(
            "SELECT id FROM runs "
            "WHERE agent_id = ? AND conversation_id = ? "
            "ORDER BY turn_index, id LIMIT 1",
            (agent_id, conversation_id),
        ).fetchone()
        if anchor is not None:
            protected_run_ids.add(str(anchor[0]))
    run_ids = tuple(
        str(row[0])
        for row in connection.execute(
            "SELECT id FROM runs WHERE agent_id = ?",
            (agent_id,),
        )
        if row[0] not in protected_run_ids
    )
    connection.executemany(
        "DELETE FROM messages WHERE run_id = ?",
        ((run_id,) for run_id in run_ids),
    )
    connection.executemany(
        "DELETE FROM runs WHERE id = ? AND agent_id = ?",
        ((run_id, agent_id) for run_id in run_ids),
    )
    return len(run_ids)


def completed_conversation_tail(
    connection: SQLConnection,
    agent_id: str,
    conversation_id: str,
    limit: int,
) -> tuple[bool, tuple[ConversationRun, ...], bool]:
    rows = connection.execute(
        """SELECT id, turn_index, input, result
           FROM runs
           WHERE agent_id = ? AND conversation_id = ?
           ORDER BY turn_index DESC""",
        (agent_id, conversation_id),
    )
    exists = False
    older_completed_exists = False
    records: list[ConversationRun] = []
    for run_id, turn_index, input_data, result_data in rows:
        exists = True
        if result_data is None:
            continue
        result = decode_loop_exit(result_data)
        if result.kind is not LoopExitKind.COMPLETED:
            continue
        if len(records) >= limit:
            older_completed_exists = True
            break
        message_rows = connection.execute(
            """SELECT data FROM messages
               WHERE run_id = ? ORDER BY position""",
            (run_id,),
        ).fetchall()
        records.append(
            ConversationRun(
                turn_index=int(turn_index),
                transcript=Transcript(
                    run=decode_run_input(input_data),
                    messages=tuple(
                        decode_message(message[0]) for message in message_rows
                    ),
                ),
                result=result,
            )
        )
    records.reverse()
    return exists, tuple(records), older_completed_exists


def latest_terminal_conversation_run(
    connection: SQLConnection, agent_id: str, conversation_id: str
) -> ConversationRun | None:
    row = connection.execute(
        """SELECT id, turn_index, input, result
           FROM runs
           WHERE agent_id = ? AND conversation_id = ?
           ORDER BY turn_index DESC LIMIT 1""",
        (agent_id, conversation_id),
    ).fetchone()
    if row is None:
        return None
    run_id, turn_index, input_data, result_data = row
    if result_data is None:
        raise ValueError("latest conversation run is not terminal")
    messages = tuple(
        decode_message(message_row[0])
        for message_row in connection.execute(
            "SELECT data FROM messages WHERE run_id = ? ORDER BY position",
            (run_id,),
        )
    )
    return ConversationRun(
        turn_index=int(turn_index),
        transcript=Transcript(
            run=decode_run_input(input_data),
            messages=messages,
        ),
        result=decode_loop_exit(result_data),
    )
