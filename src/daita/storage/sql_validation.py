"""Open-time integrity validation of every persisted state record.

Runs inside one caller-owned read connection when a store opens.
"""

from __future__ import annotations

from ..adapters.mcp import MCP_MAX_BINDINGS_PER_AGENT
from ..adapters.models import SourceRegistration
from ..artifacts.models import MAX_ARTIFACTS_PER_AGENT
from ..distribution.models import (
    MAX_DELIVERIES_PER_AGENT,
    DeliverySubjectKind,
)
from ..identity import AgentIdentity
from ..jobs.models import Job
from ..llm.models import CanonicalMessage
from ..loop.models import (
    LoopExit,
    LoopExitKind,
    RunInput,
    Transcript,
    validate_completed_transcript,
)
from ..routines.models import (
    MAX_ACTIVE_ROUTINES_PER_AGENT,
    MAX_SCHEDULED_ROUTINES_PER_AGENT,
    RoutineOccurrence,
    RoutineState,
    ScheduledRoutine,
)
from ..routines.schedule import validate_schedule
from . import (
    sql_artifacts as _artifact_store,
    sql_mcp as _mcp_store,
)
from .sql_connection import SQLConnection
from .sql_routines import _datetime_us
from .sqlite_codecs import (
    decode_catalog_link,
    decode_delivery,
    decode_identity,
    decode_loop_exit,
    decode_mcp_binding,
    decode_message,
    decode_receipt,
    decode_relational_write_scope,
    decode_routine_occurrence,
    decode_run_input,
    decode_scheduled_routine,
    decode_semantic_annotation,
    decode_source,
    decode_source_read_scope,
)
from .sqlite_codecs.analysis import decode_analysis_evidence
from .sqlite_codecs.jobs import decode_job
from .sqlite_records import SourceReadScope


def validate_current_records(connection: SQLConnection) -> AgentIdentity | None:
    _mcp_store.validate_stored_bounds(connection)
    identity: AgentIdentity | None = None
    for key, data in connection.execute("SELECT key, data FROM metadata"):
        if key == "identity":
            if identity is not None:
                raise ValueError("state contains duplicate agent identity")
            identity = decode_identity(data)
        else:
            raise ValueError("state metadata key is unsupported")

    records = tuple(
        _artifact_store.record_from_row(row)
        for row in connection.execute(
            "SELECT artifact_id, agent_id, run_id, conversation_id, caller_principal_id, state, byte_size, created_at_us, data FROM artifacts"
        )
    )
    if len(records) > MAX_ARTIFACTS_PER_AGENT:
        raise ValueError("stored artifact registry exceeds its fixed bound")
    if records and (
        identity is None or any(record.agent_id != identity.id for record in records)
    ):
        raise ValueError("stored artifact belongs to another agent")

    analytical_counts: dict[str, int] = {}
    analytical_bytes: dict[str, int] = {}
    for (
        agent_id,
        run_id,
        evidence_id,
        kind,
        data,
        reserved_bytes,
        run_owner,
    ) in connection.execute(
        "SELECT a.agent_id, a.run_id, a.evidence_id, a.kind, a.data, a.reserved_bytes, r.agent_id "
        "FROM analysis_evidence a LEFT JOIN runs r ON r.id = a.run_id"
    ):
        evidence = decode_analysis_evidence(data)
        if (
            agent_id != run_owner
            or identity is None
            or agent_id != identity.id
            or (evidence.run_id, evidence.evidence_id, evidence.kind)
            != (run_id, evidence_id, kind)
            or type(reserved_bytes) is not int
            or reserved_bytes < len(data.encode())
        ):
            raise ValueError(
                "Stored analytical evidence identity or byte reservation is invalid"
            )
        analytical_counts[run_id] = analytical_counts.get(run_id, 0) + 1
        analytical_bytes[run_id] = analytical_bytes.get(run_id, 0) + reserved_bytes
        if (
            analytical_counts[run_id] > 256
            or analytical_bytes[run_id] > 4 * 1024 * 1024
        ):
            raise ValueError("Stored analytical evidence exceeds its run bounds")

    mcp_binding_counts: dict[str, int] = {}
    for agent_id, binding_id, data in connection.execute(
        "SELECT agent_id, binding_id, data FROM mcp_server_bindings"
    ):
        mcp_binding_counts[agent_id] = mcp_binding_counts.get(agent_id, 0) + 1
        if mcp_binding_counts[agent_id] > MCP_MAX_BINDINGS_PER_AGENT:
            raise ValueError("stored MCP binding count exceeds its fixed bound")
        binding = decode_mcp_binding(
            data,
            agent_id=agent_id,
            binding_id=binding_id,
        )
        if binding.agent_id != agent_id or binding.binding_id != binding_id:
            raise ValueError("stored MCP binding ownership is invalid")
        if identity is not None and binding.agent_id != identity.id:
            raise ValueError("stored MCP binding belongs to another agent")

    sources: dict[tuple[str, str], SourceRegistration] = {}
    for agent_id, source_id, data in connection.execute(
        "SELECT agent_id, id, data FROM sources"
    ):
        registration = decode_source(data)
        if registration.agent_id != agent_id or registration.id != source_id:
            raise ValueError("stored source ownership is invalid")
        sources[(agent_id, source_id)] = registration
    if identity is not None and any(agent_id != identity.id for agent_id, _ in sources):
        raise ValueError("stored source belongs to another agent")
    read_scopes: dict[tuple[str, str], SourceReadScope] = {}
    for agent_id, source_id, data in connection.execute(
        "SELECT agent_id, source_id, data FROM source_read_scopes"
    ):
        key = (agent_id, source_id)
        if key in read_scopes:
            raise ValueError("stored source read scope identity is duplicated")
        read_scopes[key] = decode_source_read_scope(
            data,
            agent_id=agent_id,
            source_id=source_id,
        )
    for key, registration in sources.items():
        scope = read_scopes.pop(key, None)
        if registration.active and scope is None:
            raise ValueError("active source is missing its read scope")
        if not registration.active and scope is not None:
            raise ValueError("detached source retains a read scope")
    if read_scopes:
        raise ValueError("stored source read scope is foreign")

    for agent_id, source_id, resource_id, fingerprint, data in connection.execute(
        """SELECT agent_id, source_id, resource_id,
                  authorization_fingerprint, data
           FROM relational_write_scopes"""
    ):
        scope_registration = sources.get((agent_id, source_id))
        if scope_registration is None or not scope_registration.active:
            raise ValueError("stored PostgreSQL update scope is foreign")
        decode_relational_write_scope(
            data,
            agent_id=agent_id,
            source_id=source_id,
            resource_id=resource_id,
            authorization_fingerprint=fingerprint,
        )

    # Catalog rows are not read here: a damaged row is refreshed
    # from its source, so it must not stop the home opening.

    run_ids: set[str] = set()
    run_inputs: dict[str, RunInput] = {}
    run_results: dict[str, LoopExit] = {}
    run_messages: dict[str, list[CanonicalMessage]] = {}
    message_positions: dict[str, list[int]] = {}
    for (
        run_id,
        agent_id,
        conversation_id,
        turn_index,
        input_data,
        result,
    ) in connection.execute(
        """SELECT id, agent_id, conversation_id, turn_index, input, result
           FROM runs"""
    ):
        run_input = decode_run_input(input_data)
        if (
            run_input.id != run_id
            or run_input.agent_id != agent_id
            or run_input.conversation_id != conversation_id
            or not isinstance(turn_index, int)
            or isinstance(turn_index, bool)
            or turn_index < 0
        ):
            raise ValueError("stored run ownership is invalid")
        if identity is not None and run_input.agent_id != identity.id:
            raise ValueError("stored run belongs to another agent")
        if result is not None:
            exit_record = decode_loop_exit(result)
            if (
                exit_record.run_id != run_id
                or exit_record.conversation_id != conversation_id
            ):
                raise ValueError("stored run result ownership is invalid")
            run_results[run_id] = exit_record
        run_ids.add(run_id)
        run_inputs[run_id] = run_input
    for run_id, position, data in connection.execute(
        "SELECT run_id, position, data FROM messages ORDER BY run_id, position"
    ):
        if run_id not in run_ids:
            raise ValueError("stored message belongs to an unknown run")
        message = decode_message(data)
        message_positions.setdefault(run_id, []).append(position)
        run_messages.setdefault(run_id, []).append(message)
    if any(
        positions != list(range(len(positions)))
        for positions in message_positions.values()
    ):
        raise ValueError("stored transcript message positions are not contiguous")
    for run_id, result in run_results.items():
        if result.kind is LoopExitKind.COMPLETED:
            validate_completed_transcript(
                Transcript(
                    run=run_inputs[run_id],
                    messages=tuple(run_messages.get(run_id, ())),
                ),
                result,
            )

    for (
        agent_id,
        receipt_id,
        run_id,
        call_id,
        operation_key,
        routine_id,
        occurrence_id,
        grant_digest,
        unresolved,
        data,
    ) in connection.execute(
        """SELECT agent_id, id, run_id, call_id, operation_key, routine_id, occurrence_id, grant_digest, unresolved, data
           FROM effect_receipts"""
    ):
        receipt = decode_receipt(data)
        if (
            receipt.agent_id != agent_id
            or receipt.receipt_id != receipt_id
            or receipt.run_id != run_id
            or receipt.call_id != call_id
            or receipt.operation_key != operation_key
            or receipt.routine_id != routine_id
            or receipt.occurrence_id != occurrence_id
            or receipt.capability_grant_digest != grant_digest
            or int(receipt.unresolved) != unresolved
        ):
            raise ValueError(
                "stored effect receipt identity or indexed state is invalid"
            )
        if identity is not None and receipt.agent_id != identity.id:
            raise ValueError("stored effect receipt belongs to another agent")

    for agent_id, annotation_id, data in connection.execute(
        "SELECT agent_id, id, data FROM semantic_annotations"
    ):
        annotation = decode_semantic_annotation(data)
        if annotation.agent_id != agent_id or annotation.id != annotation_id:
            raise ValueError("stored semantic annotation ownership is invalid")
        if identity is not None and annotation.agent_id != identity.id:
            raise ValueError("stored semantic annotation belongs to another agent")
    for agent_id, link_id, data in connection.execute(
        "SELECT agent_id, id, data FROM catalog_links"
    ):
        link = decode_catalog_link(data)
        if link.agent_id != agent_id or link.id != link_id:
            raise ValueError("stored catalog link ownership is invalid")
        if identity is not None and link.agent_id != identity.id:
            raise ValueError("stored catalog link belongs to another agent")

    jobs: dict[tuple[str, str], Job] = {}
    for agent_id, job_id, conversation_id, state, revision, data in connection.execute(
        "SELECT agent_id, job_id, conversation_id, state, revision, data FROM jobs"
    ):
        job = decode_job(data)
        if (
            job.agent_id != agent_id
            or job.job_id != job_id
            or job.conversation_id != conversation_id
            or job.state.value != state
            or job.revision != revision
        ):
            raise ValueError("stored job projection is invalid")
        if identity is not None and job.agent_id != identity.id:
            raise ValueError("stored job belongs to another agent")
        jobs[(agent_id, job_id)] = job

    routines: dict[tuple[str, str], ScheduledRoutine] = {}
    routine_counts: dict[str, int] = {}
    active_routine_counts: dict[str, int] = {}
    for (
        agent_id,
        routine_id,
        conversation_id,
        state,
        next_due_at_us,
        data,
    ) in connection.execute(
        """SELECT agent_id, routine_id, conversation_id, state,
                  next_due_at_us, data
           FROM scheduled_routines"""
    ):
        routine = decode_scheduled_routine(
            data,
            agent_id=agent_id,
            routine_id=routine_id,
        )
        if (
            routine.conversation_id != conversation_id
            or routine.state.value != state
            or _datetime_us(routine.next_due_at) != next_due_at_us
        ):
            raise ValueError("stored scheduled routine projection is invalid")
        validate_schedule(routine.schedule)
        if identity is not None and routine.agent_id != identity.id:
            raise ValueError("stored scheduled routine belongs to another agent")
        key = (agent_id, routine_id)
        routines[key] = routine
        routine_counts[agent_id] = routine_counts.get(agent_id, 0) + 1
        if routine.state is RoutineState.ACTIVE:
            active_routine_counts[agent_id] = active_routine_counts.get(agent_id, 0) + 1
    if any(
        value > MAX_SCHEDULED_ROUTINES_PER_AGENT for value in routine_counts.values()
    ):
        raise ValueError("stored routine count exceeds its fixed bound")
    if any(
        value > MAX_ACTIVE_ROUTINES_PER_AGENT
        for value in active_routine_counts.values()
    ):
        raise ValueError("stored active routine count exceeds its fixed bound")

    occurrences: dict[tuple[str, str], RoutineOccurrence] = {}
    for (
        agent_id,
        occurrence_id_value,
        routine_id,
        routine_revision,
        slot_key,
        state,
        lease_expires_at_us,
        reserved_run_id,
        data,
    ) in connection.execute(
        """SELECT agent_id, occurrence_id, routine_id, routine_revision,
                  slot_key, state, lease_expires_at_us, reserved_run_id, data
           FROM routine_occurrences"""
    ):
        occurrence = decode_routine_occurrence(
            data,
            agent_id=agent_id,
            occurrence_id=occurrence_id_value,
        )
        owning_routine = routines.get((agent_id, routine_id))
        if (
            owning_routine is None
            or occurrence.routine_id != routine_id
            or occurrence.routine_revision != routine_revision
            or occurrence.routine_revision > owning_routine.revision
            or occurrence.slot_key != slot_key
            or occurrence.disposition.value != state
            or _datetime_us(occurrence.lease_expires_at) != lease_expires_at_us
            or occurrence.reserved_run_id != reserved_run_id
        ):
            raise ValueError("stored routine occurrence projection is invalid")
        occurrences[(agent_id, occurrence_id_value)] = occurrence
    for routine in routines.values():
        if routine.active_occurrence_id is None:
            continue
        active = occurrences.get((routine.agent_id, routine.active_occurrence_id))
        if active is None or active.routine_id != routine.routine_id:
            raise ValueError("stored routine active occurrence is invalid")

    delivery_counts: dict[str, int] = {}
    for (
        agent_id,
        delivery_id,
        conversation_id,
        subject_kind,
        subject_id,
        logical_key,
        target_kind,
        target_fingerprint_value,
        state,
        created_at_us,
        data,
    ) in connection.execute(
        """SELECT agent_id, delivery_id, conversation_id, subject_kind,
                  subject_id, logical_key, target_kind, target_fingerprint,
                  state, created_at_us, data
           FROM deliveries"""
    ):
        delivery = decode_delivery(data, agent_id=agent_id, delivery_id=delivery_id)
        if (
            delivery.conversation_id != conversation_id
            or delivery.subject_kind.value != subject_kind
            or delivery.subject_id != subject_id
            or delivery.logical_key != logical_key
            or target_kind != "conversation_inbox"
            or delivery.target.target_fingerprint != target_fingerprint_value
            or delivery.visibility_state.value != state
            or _datetime_us(delivery.created_at) != created_at_us
        ):
            raise ValueError("stored delivery projection is invalid")
        if identity is not None and delivery.agent_id != identity.id:
            raise ValueError("stored delivery belongs to another agent")
        if delivery.subject_kind is DeliverySubjectKind.ROUTINE_OCCURRENCE:
            producer = occurrences.get((agent_id, subject_id))
            producer_references_delivery = (
                producer is not None and delivery_id in producer.delivery_ids
            )
        elif delivery.subject_kind is DeliverySubjectKind.JOB:
            producing_job = jobs.get((agent_id, subject_id))
            producer_references_delivery = (
                producing_job is not None
                and producing_job.conversation_id == delivery.conversation_id
            )
        else:
            raise ValueError("stored delivery retained a legacy producer")
        if not producer_references_delivery:
            raise ValueError("stored delivery producer reference is invalid")
        delivery_counts[agent_id] = delivery_counts.get(agent_id, 0) + 1
        if delivery_counts[agent_id] > MAX_DELIVERIES_PER_AGENT:
            raise ValueError("stored delivery count exceeds its fixed bound")
    return identity
