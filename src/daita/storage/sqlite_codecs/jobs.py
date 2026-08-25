"""Encode and decode the current codec-v1 durable JobRun aggregate."""

from __future__ import annotations

from ...artifacts.models import artifact_ref_from_mapping, artifact_ref_to_mapping
from ...jobs.models import (
    ConnectedExecutorBinding,
    ExternalIntent,
    ExternalIntentDisposition,
    ExternalIntentKind,
    ExternalObservation,
    ExternalObservedStatus,
    JobAttempt,
    JobAttemptStatus,
    JobCompletionBinding,
    JobCompletionOwnerKind,
    JobDesiredState,
    JobExecutionMode,
    JobResourceBinding,
    JobResult,
    JobRun,
    JobSpecification,
    JobStatus,
)
from ...llm.models import ModelSensitivity
from .common import (
    datetime_decode,
    datetime_encode,
    dump_payload,
    integer,
    load_payload,
    number,
    optional_datetime_decode,
    optional_datetime_encode,
    optional_text,
    plain_decode,
    plain_encode,
    record,
    record_fields,
    sequence,
    text,
)

_JOB_RUN_VERSION = 1


def encode_job_run(value: JobRun) -> str:
    if not isinstance(value, JobRun):
        raise TypeError("job codec requires JobRun")
    return dump_payload(
        record(
            "JobRun",
            {
                "version": _JOB_RUN_VERSION,
                "conversation_id": value.conversation_id,
                "origin_run_id": value.origin_run_id,
                "origin_call_id": value.origin_call_id,
                "specification": _encode_specification(value.specification),
                "specification_digest": value.specification_digest,
                "status": value.status.value,
                "desired_state": value.desired_state.value,
                "created_at": datetime_encode(value.created_at),
                "updated_at": datetime_encode(value.updated_at),
                "revision": value.revision,
                "fencing_epoch": value.fencing_epoch,
                "attempts": [_encode_attempt(item) for item in value.attempts],
                "cancel_requested_at": optional_datetime_encode(
                    value.cancel_requested_at
                ),
                "terminal_at": optional_datetime_encode(value.terminal_at),
                "terminal_observed_at": optional_datetime_encode(
                    value.terminal_observed_at
                ),
                "completion_binding": (
                    None
                    if value.completion_binding is None
                    else _encode_completion_binding(value.completion_binding)
                ),
                "result": (
                    None if value.result is None else _encode_result(value.result)
                ),
                "failure_code": value.failure_code,
            },
        )
    )


def decode_job_run(value: str, *, agent_id: str, job_id: str) -> JobRun:
    fields = record_fields(
        load_payload(value),
        "JobRun",
        (
            "version",
            "conversation_id",
            "origin_run_id",
            "origin_call_id",
            "specification",
            "specification_digest",
            "status",
            "desired_state",
            "created_at",
            "updated_at",
            "revision",
            "fencing_epoch",
            "attempts",
            "cancel_requested_at",
            "terminal_at",
            "terminal_observed_at",
            "completion_binding",
            "result",
            "failure_code",
        ),
    )
    if integer(fields["version"], "job version") != _JOB_RUN_VERSION:
        raise ValueError("stored job version is unsupported")
    try:
        status = JobStatus(text(fields["status"], "job status"))
        desired = JobDesiredState(text(fields["desired_state"], "job desired state"))
    except ValueError:
        raise ValueError("stored job enum is invalid") from None
    result_value = fields["result"]
    return JobRun(
        job_id=job_id,
        agent_id=agent_id,
        conversation_id=text(fields["conversation_id"], "job conversation_id"),
        origin_run_id=text(fields["origin_run_id"], "job origin_run_id"),
        origin_call_id=text(fields["origin_call_id"], "job origin_call_id"),
        specification=_decode_specification(fields["specification"]),
        specification_digest=text(
            fields["specification_digest"], "job specification digest"
        ),
        status=status,
        desired_state=desired,
        created_at=datetime_decode(fields["created_at"]),
        updated_at=datetime_decode(fields["updated_at"]),
        revision=integer(fields["revision"], "job revision"),
        fencing_epoch=integer(fields["fencing_epoch"], "job fencing epoch"),
        attempts=tuple(
            _decode_attempt(item)
            for item in sequence(fields["attempts"], "job attempts")
        ),
        cancel_requested_at=optional_datetime_decode(fields["cancel_requested_at"]),
        terminal_at=optional_datetime_decode(fields["terminal_at"]),
        terminal_observed_at=optional_datetime_decode(fields["terminal_observed_at"]),
        completion_binding=(
            None
            if fields["completion_binding"] is None
            else _decode_completion_binding(fields["completion_binding"])
        ),
        result=None if result_value is None else _decode_result(result_value),
        failure_code=optional_text(fields["failure_code"], "job failure code"),
    )


def _encode_completion_binding(value: JobCompletionBinding):
    return record(
        "JobCompletionBinding",
        {
            "owner_kind": value.owner_kind.value,
            "owner_id": value.owner_id,
            "terminal_event_id": value.terminal_event_id,
            "bound_at": datetime_encode(value.bound_at),
        },
    )


def _decode_completion_binding(value) -> JobCompletionBinding:
    fields = record_fields(
        value,
        "JobCompletionBinding",
        ("owner_kind", "owner_id", "terminal_event_id", "bound_at"),
    )
    try:
        owner_kind = JobCompletionOwnerKind(
            text(fields["owner_kind"], "job completion owner kind")
        )
    except ValueError:
        raise ValueError("stored job completion owner kind is invalid") from None
    return JobCompletionBinding(
        owner_kind=owner_kind,
        owner_id=text(fields["owner_id"], "job completion owner id"),
        terminal_event_id=text(
            fields["terminal_event_id"],
            "job completion event id",
        ),
        bound_at=datetime_decode(fields["bound_at"]),
    )


def _encode_specification(value: JobSpecification):
    return record(
        "JobSpecification",
        {
            "job_kind": value.job_kind,
            "arguments": plain_encode(value.arguments),
            "resource_bindings": [
                _encode_resource_binding(item) for item in value.resource_bindings
            ],
            "execution_capability_id": value.execution_capability_id,
            "execution_contract_digest": value.execution_contract_digest,
            "execution_mode": value.execution_mode.value,
            "sensitivity": value.sensitivity.value,
            "deadline_at": datetime_encode(value.deadline_at),
            "max_wall_time_seconds": value.max_wall_time_seconds,
            "external_executor": (
                None
                if value.external_executor is None
                else _encode_external_binding(value.external_executor)
            ),
        },
    )


def _decode_specification(value) -> JobSpecification:
    fields = record_fields(
        value,
        "JobSpecification",
        (
            "job_kind",
            "arguments",
            "resource_bindings",
            "execution_capability_id",
            "execution_contract_digest",
            "execution_mode",
            "sensitivity",
            "deadline_at",
            "max_wall_time_seconds",
            "external_executor",
        ),
    )
    arguments = plain_decode(fields["arguments"])
    if not isinstance(arguments, dict):
        raise ValueError("stored job arguments are invalid")
    try:
        mode = JobExecutionMode(text(fields["execution_mode"], "job mode"))
        sensitivity = ModelSensitivity(text(fields["sensitivity"], "job sensitivity"))
    except ValueError:
        raise ValueError("stored job specification enum is invalid") from None
    external_value = fields["external_executor"]
    return JobSpecification(
        job_kind=text(fields["job_kind"], "job kind"),
        arguments=arguments,
        resource_bindings=tuple(
            _decode_resource_binding(item)
            for item in sequence(fields["resource_bindings"], "job resource bindings")
        ),
        execution_capability_id=text(
            fields["execution_capability_id"], "job execution capability id"
        ),
        execution_contract_digest=text(
            fields["execution_contract_digest"],
            "job execution contract digest",
        ),
        execution_mode=mode,
        sensitivity=sensitivity,
        deadline_at=datetime_decode(fields["deadline_at"]),
        max_wall_time_seconds=number(fields["max_wall_time_seconds"], "job wall time"),
        external_executor=(
            None if external_value is None else _decode_external_binding(external_value)
        ),
    )


def _encode_resource_binding(value: JobResourceBinding):
    return record(
        "JobResourceBinding",
        {
            "source_id": value.source_id,
            "source_revision": value.source_revision,
            "resource_id": value.resource_id,
            "resource_revision": value.resource_revision,
            "adapter_id": value.adapter_id,
            "sensitivity": value.sensitivity.value,
        },
    )


def _decode_resource_binding(value) -> JobResourceBinding:
    fields = record_fields(
        value,
        "JobResourceBinding",
        (
            "source_id",
            "source_revision",
            "resource_id",
            "resource_revision",
            "adapter_id",
            "sensitivity",
        ),
    )
    try:
        sensitivity = ModelSensitivity(
            text(fields["sensitivity"], "job resource sensitivity")
        )
    except ValueError:
        raise ValueError("stored job resource sensitivity is invalid") from None
    return JobResourceBinding(
        source_id=text(fields["source_id"], "job source id"),
        source_revision=text(fields["source_revision"], "job source revision"),
        resource_id=text(fields["resource_id"], "job resource id"),
        resource_revision=text(fields["resource_revision"], "job resource revision"),
        adapter_id=text(fields["adapter_id"], "job adapter id"),
        sensitivity=sensitivity,
    )


def _encode_external_binding(value: ConnectedExecutorBinding):
    return record(
        "ConnectedExecutorBinding",
        {
            "profile_id": value.profile_id,
            "binding_id": value.binding_id,
            "execution_identity": value.execution_identity,
            "contract_digest": value.contract_digest,
            "revision": value.revision,
            "maximum_sensitivity": value.maximum_sensitivity.value,
        },
    )


def _decode_external_binding(value) -> ConnectedExecutorBinding:
    fields = record_fields(
        value,
        "ConnectedExecutorBinding",
        (
            "profile_id",
            "binding_id",
            "execution_identity",
            "contract_digest",
            "revision",
            "maximum_sensitivity",
        ),
    )
    try:
        maximum = ModelSensitivity(
            text(fields["maximum_sensitivity"], "external maximum sensitivity")
        )
    except ValueError:
        raise ValueError("stored external maximum sensitivity is invalid") from None
    return ConnectedExecutorBinding(
        profile_id=text(fields["profile_id"], "external profile id"),
        binding_id=text(fields["binding_id"], "external binding id"),
        execution_identity=text(
            fields["execution_identity"], "external execution identity"
        ),
        contract_digest=text(fields["contract_digest"], "external contract digest"),
        revision=integer(fields["revision"], "external revision"),
        maximum_sensitivity=maximum,
    )


def _encode_attempt(value: JobAttempt):
    return record(
        "JobAttempt",
        {
            "number": value.number,
            "fencing_epoch": value.fencing_epoch,
            "claim_token": value.claim_token,
            "execution_run_id": value.execution_run_id,
            "reserved_artifact_id": value.reserved_artifact_id,
            "status": value.status.value,
            "claimed_at": datetime_encode(value.claimed_at),
            "lease_expires_at": datetime_encode(value.lease_expires_at),
            "renewals": value.renewals,
            "completed_at": optional_datetime_encode(value.completed_at),
            "error_code": value.error_code,
            "external_intents": [
                _encode_external_intent(item) for item in value.external_intents
            ],
            "external_observations": [
                _encode_external_observation(item)
                for item in value.external_observations
            ],
        },
    )


def _decode_attempt(value) -> JobAttempt:
    fields = record_fields(
        value,
        "JobAttempt",
        (
            "number",
            "fencing_epoch",
            "claim_token",
            "execution_run_id",
            "reserved_artifact_id",
            "status",
            "claimed_at",
            "lease_expires_at",
            "renewals",
            "completed_at",
            "error_code",
            "external_intents",
            "external_observations",
        ),
    )
    try:
        status = JobAttemptStatus(text(fields["status"], "job attempt status"))
    except ValueError:
        raise ValueError("stored job attempt status is invalid") from None
    return JobAttempt(
        number=integer(fields["number"], "job attempt number"),
        fencing_epoch=integer(fields["fencing_epoch"], "job attempt fencing epoch"),
        claim_token=text(fields["claim_token"], "job attempt claim token"),
        execution_run_id=text(
            fields["execution_run_id"], "job attempt execution run id"
        ),
        reserved_artifact_id=text(
            fields["reserved_artifact_id"], "job attempt reserved artifact id"
        ),
        status=status,
        claimed_at=datetime_decode(fields["claimed_at"]),
        lease_expires_at=datetime_decode(fields["lease_expires_at"]),
        renewals=integer(fields["renewals"], "job attempt renewals"),
        completed_at=optional_datetime_decode(fields["completed_at"]),
        error_code=optional_text(fields["error_code"], "job attempt error code"),
        external_intents=tuple(
            _decode_external_intent(item)
            for item in sequence(fields["external_intents"], "external intents")
        ),
        external_observations=tuple(
            _decode_external_observation(item)
            for item in sequence(
                fields["external_observations"], "external observations"
            )
        ),
    )


def _encode_external_intent(value: ExternalIntent):
    return record(
        "ExternalIntent",
        {
            "kind": value.kind.value,
            "idempotency_key": value.idempotency_key,
            "requested_at": datetime_encode(value.requested_at),
            "disposition": value.disposition.value,
            "completed_at": optional_datetime_encode(value.completed_at),
            "external_job_id": value.external_job_id,
            "reason_code": value.reason_code,
        },
    )


def _decode_external_intent(value) -> ExternalIntent:
    fields = record_fields(
        value,
        "ExternalIntent",
        (
            "kind",
            "idempotency_key",
            "requested_at",
            "disposition",
            "completed_at",
            "external_job_id",
            "reason_code",
        ),
    )
    try:
        kind = ExternalIntentKind(text(fields["kind"], "external intent kind"))
        disposition = ExternalIntentDisposition(
            text(fields["disposition"], "external intent disposition")
        )
    except ValueError:
        raise ValueError("stored external intent enum is invalid") from None
    return ExternalIntent(
        kind=kind,
        idempotency_key=text(
            fields["idempotency_key"], "external intent idempotency key"
        ),
        requested_at=datetime_decode(fields["requested_at"]),
        disposition=disposition,
        completed_at=optional_datetime_decode(fields["completed_at"]),
        external_job_id=optional_text(
            fields["external_job_id"], "external intent job id"
        ),
        reason_code=optional_text(fields["reason_code"], "external intent reason"),
    )


def _encode_external_observation(value: ExternalObservation):
    return record(
        "ExternalObservation",
        {
            "sequence": value.sequence,
            "observed_at": datetime_encode(value.observed_at),
            "status": value.status.value,
            "observation_digest": value.observation_digest,
            "external_job_id": value.external_job_id,
        },
    )


def _decode_external_observation(value) -> ExternalObservation:
    fields = record_fields(
        value,
        "ExternalObservation",
        (
            "sequence",
            "observed_at",
            "status",
            "observation_digest",
            "external_job_id",
        ),
    )
    try:
        status = ExternalObservedStatus(
            text(fields["status"], "external observation status")
        )
    except ValueError:
        raise ValueError("stored external observation status is invalid") from None
    return ExternalObservation(
        sequence=integer(fields["sequence"], "external observation sequence"),
        observed_at=datetime_decode(fields["observed_at"]),
        status=status,
        observation_digest=text(
            fields["observation_digest"], "external observation digest"
        ),
        external_job_id=text(fields["external_job_id"], "external observation job id"),
    )


def _encode_result(value: JobResult):
    return record(
        "JobResult",
        {
            "result_id": value.result_id,
            "summary": plain_encode(value.summary),
            "sensitivity": value.sensitivity.value,
            "provenance": plain_encode(value.provenance),
            "artifact_refs": [
                plain_encode(artifact_ref_to_mapping(item))
                for item in value.artifact_refs
            ],
            "completed_at": datetime_encode(value.completed_at),
        },
    )


def _decode_result(value) -> JobResult:
    fields = record_fields(
        value,
        "JobResult",
        (
            "result_id",
            "summary",
            "sensitivity",
            "provenance",
            "artifact_refs",
            "completed_at",
        ),
    )
    summary = plain_decode(fields["summary"])
    provenance = plain_decode(fields["provenance"])
    if not isinstance(summary, dict) or not isinstance(provenance, dict):
        raise ValueError("stored job result payload is invalid")
    try:
        sensitivity = ModelSensitivity(
            text(fields["sensitivity"], "job result sensitivity")
        )
    except ValueError:
        raise ValueError("stored job result sensitivity is invalid") from None
    refs = []
    for item in sequence(fields["artifact_refs"], "job artifact refs"):
        decoded = plain_decode(item)
        if not isinstance(decoded, dict):
            raise ValueError("stored job artifact ref is invalid")
        refs.append(artifact_ref_from_mapping(decoded))
    return JobResult(
        result_id=text(fields["result_id"], "job result id"),
        summary=summary,
        sensitivity=sensitivity,
        provenance=provenance,
        artifact_refs=tuple(refs),
        completed_at=datetime_decode(fields["completed_at"]),
    )


__all__ = ["decode_job_run", "encode_job_run"]
