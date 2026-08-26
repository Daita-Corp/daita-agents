from __future__ import annotations

from _workspace_support import workspace_for

from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest

from daita import Agent
from daita._json import canonical_json
from daita.adapters.job_profiles import (
    ExternalStartRequest,
    ExternalStatusReceipt,
)
from daita.artifacts.models import (
    ArtifactAuthorship,
    ArtifactDraft,
    ArtifactError,
    ArtifactProvenance,
    ArtifactRef,
)
from daita.catalog.models import Sensitivity
from daita.domains.data.profile_jobs import DATA_PROFILE_EXECUTION_CAPABILITY_ID
from daita.jobs.models import (
    MAX_ACTIVE_JOBS_PER_AGENT,
    MAX_EXTERNAL_REQUEST_BYTES,
    MAX_EXTERNAL_RESPONSE_BYTES,
    MAX_JOB_ARTIFACTS,
    MAX_JOB_ATTEMPTS,
    MAX_JOB_DEADLINE_SECONDS,
    MAX_JOB_EXTERNAL_OBSERVATIONS,
    MAX_JOB_INLINE_RESULT_BYTES,
    MAX_JOB_LIST_PAGE_SIZE,
    MAX_JOB_RENEWALS,
    MAX_JOB_RESOURCE_BINDINGS,
    MAX_JOB_RESULT_DEPTH,
    MAX_JOB_SPECIFICATION_BYTES,
    MAX_JOB_SPECIFICATION_DEPTH,
    MAX_JOB_SUMMARY_BYTES,
    MAX_JOB_WALL_TIME_SECONDS,
    MAX_JOBS_PER_AGENT,
    MAX_QUEUED_JOBS_PER_AGENT,
    MAX_RUNNING_JOBS_PER_AGENT,
    ExternalObservation,
    ExternalObservedStatus,
    JobAttempt,
    JobAttemptStatus,
    JobDesiredState,
    JobExecutionMode,
    JobResourceBinding,
    JobResult,
    JobRun,
    JobSpecification,
    JobStatus,
)
from daita.jobs.owner import JobError, JobOwner
from daita.llm.models import ModelSensitivity
from daita.loop.models import RunInput
from daita.storage.sqlite import SQLiteStateStore

NOW = datetime(2026, 8, 21, 12, tzinfo=UTC)


def _binding(index: int = 0) -> JobResourceBinding:
    return JobResourceBinding(
        source_id=f"source-{index}",
        source_revision=f"source-revision-{index}",
        resource_id=f"resource-{index}",
        resource_revision="sha256:" + sha256(f"resource-{index}".encode()).hexdigest(),
        adapter_id="sqlite",
        sensitivity=ModelSensitivity.INTERNAL,
    )


def _specification(
    *,
    arguments: Mapping[str, object] | None = None,
    bindings: tuple[JobResourceBinding, ...] | None = None,
    deadline_at: datetime | None = None,
    wall_time: float = 60,
) -> JobSpecification:
    return JobSpecification(
        job_kind="data_profile",
        arguments=arguments or {"resource_ids": ("resource-0",), "sample_rows": 10},
        resource_bindings=bindings or (_binding(),),
        execution_capability_id="jobs.data_profile.execute",
        execution_contract_digest="sha256:" + sha256(b"capability-v1").hexdigest(),
        execution_mode=JobExecutionMode.DAITA,
        sensitivity=ModelSensitivity.INTERNAL,
        deadline_at=deadline_at or NOW + timedelta(minutes=10),
        max_wall_time_seconds=wall_time,
    )


def _job(
    job_id: str,
    *,
    created_at: datetime = NOW,
    agent_id: str = "agent-1",
    specification: JobSpecification | None = None,
) -> JobRun:
    spec = specification or _specification()
    return JobRun(
        job_id=job_id,
        agent_id=agent_id,
        conversation_id="conversation-1",
        origin_run_id="run-origin",
        origin_call_id="call-origin",
        specification=spec,
        specification_digest=spec.digest,
        status=JobStatus.QUEUED,
        desired_state=JobDesiredState.RUN,
        created_at=created_at,
        updated_at=created_at,
    )


def _attempt(
    number: int,
    *,
    renewals: int = 0,
    status: JobAttemptStatus = JobAttemptStatus.FENCED,
    observations: tuple[ExternalObservation, ...] = (),
) -> JobAttempt:
    claimed_at = NOW + timedelta(seconds=number)
    return JobAttempt(
        number=number,
        fencing_epoch=number,
        claim_token=f"claim-{number}",
        execution_run_id=f"run-{number:032x}",
        reserved_artifact_id=f"artifact-{number:032x}",
        status=status,
        claimed_at=claimed_at,
        lease_expires_at=claimed_at + timedelta(seconds=60),
        renewals=renewals,
        completed_at=(
            None
            if status is JobAttemptStatus.CLAIMED
            else claimed_at + timedelta(seconds=1)
        ),
        external_observations=observations,
    )


def _payload_with_bytes(maximum: int) -> dict[str, str]:
    overhead = len(canonical_json({"payload": ""}).encode("utf-8"))
    return {"payload": "x" * (maximum - overhead)}


def _nested(depth: int) -> dict[str, object]:
    value: object = 0
    for _ in range(depth):
        value = {"value": value}
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _artifact_ref(index: int) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=f"artifact-{index:032x}",
        run_id="run-00000000000000000000000000000001",
        conversation_id="conversation-1",
        call_id="call-1",
        capability_id="jobs.data_profile.execute",
        filename=f"profile-{index}.json",
        media_type="application/json",
        byte_size=1,
        sha256="sha256:" + sha256(str(index).encode()).hexdigest(),
        sensitivity=Sensitivity.INTERNAL,
        provenance=ArtifactProvenance(
            authorship=ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS,
            evidence_call_ids=("call-1",),
        ),
        created_at=NOW,
    )


def _result(
    summary: Mapping[str, object],
    provenance: Mapping[str, object],
    artifact_refs: tuple[ArtifactRef, ...] = (),
) -> JobResult:
    return JobResult(
        result_id="result-1",
        summary=summary,
        sensitivity=ModelSensitivity.INTERNAL,
        provenance=provenance,
        artifact_refs=artifact_refs,
        completed_at=NOW,
    )


async def test_retained_job_and_list_page_bounds_are_inclusive(tmp_path: Path) -> None:
    store = await SQLiteStateStore.open(tmp_path / "retention.sqlite")
    try:
        for index in range(MAX_JOBS_PER_AGENT):
            job = _job(
                f"job-{index:03d}", created_at=NOW + timedelta(microseconds=index)
            )
            await store.admit_job(job)
            await store.request_job_cancel(
                job.agent_id,
                job.job_id,
                requested_at=NOW + timedelta(seconds=1),
            )
        assert len(await store.list_jobs("agent-1", limit=MAX_JOB_LIST_PAGE_SIZE)) == 50
        with pytest.raises(ValueError, match="job list limit"):
            await store.list_jobs("agent-1", limit=MAX_JOB_LIST_PAGE_SIZE + 1)
        with pytest.raises(ValueError, match="job_retention_limit_exceeded"):
            await store.admit_job(_job("job-over-retention"))
    finally:
        await store.close()


async def test_active_job_bound_is_reachable_and_inclusive(tmp_path: Path) -> None:
    store = await SQLiteStateStore.open(tmp_path / "active.sqlite")
    try:
        for index in range(MAX_RUNNING_JOBS_PER_AGENT):
            spec = _specification(
                arguments={"resource_ids": (f"resource-{index}",), "sample_rows": 10},
                bindings=(_binding(index),),
            )
            await store.admit_job(_job(f"running-{index}", specification=spec))
            claimed = await store.claim_next_job(
                "agent-1",
                claim_token=f"claim-{index}",
                execution_run_id=f"run-{index + 1:032x}",
                reserved_artifact_id=f"artifact-{index + 1:032x}",
                claimed_at=NOW + timedelta(seconds=1),
                lease_seconds=60,
            )
            assert claimed is not None
        for index in range(MAX_QUEUED_JOBS_PER_AGENT):
            await store.admit_job(
                _job(
                    f"queued-{index}",
                    created_at=NOW + timedelta(microseconds=index + 1),
                )
            )
        assert MAX_RUNNING_JOBS_PER_AGENT + MAX_QUEUED_JOBS_PER_AGENT == (
            MAX_ACTIVE_JOBS_PER_AGENT
        )
        with pytest.raises(ValueError, match="job_active_limit_exceeded"):
            await store.admit_job(_job("job-over-active"))
    finally:
        await store.close()


def test_specification_byte_depth_and_resource_bounds_are_inclusive() -> None:
    exact_bytes = _payload_with_bytes(MAX_JOB_SPECIFICATION_BYTES)
    assert (
        len(canonical_json(exact_bytes).encode("utf-8")) == MAX_JOB_SPECIFICATION_BYTES
    )
    _specification(arguments=exact_bytes)
    with pytest.raises(ValueError, match="job specification exceeds its byte bound"):
        _specification(arguments={"payload": exact_bytes["payload"] + "x"})

    _specification(arguments=_nested(MAX_JOB_SPECIFICATION_DEPTH))
    with pytest.raises(ValueError, match="job specification exceeds its depth bound"):
        _specification(arguments=_nested(MAX_JOB_SPECIFICATION_DEPTH + 1))

    bindings = tuple(
        sorted(
            (_binding(index) for index in range(MAX_JOB_RESOURCE_BINDINGS)),
            key=lambda item: (item.source_id, item.resource_id),
        )
    )
    _specification(bindings=bindings)
    with pytest.raises(ValueError, match="job resource bindings exceed their bound"):
        _specification(
            bindings=tuple(
                sorted(
                    (*bindings, _binding(MAX_JOB_RESOURCE_BINDINGS)),
                    key=lambda item: (item.source_id, item.resource_id),
                )
            )
        )


def test_attempt_renewal_and_observation_bounds_are_inclusive() -> None:
    attempts = tuple(_attempt(index) for index in range(1, MAX_JOB_ATTEMPTS + 1))
    job = _job("job-attempt-bound")
    replace(job, fencing_epoch=MAX_JOB_ATTEMPTS, attempts=attempts)
    with pytest.raises(ValueError, match="job attempt number is outside its bound"):
        _attempt(MAX_JOB_ATTEMPTS + 1)

    _attempt(1, renewals=MAX_JOB_RENEWALS)
    with pytest.raises(
        ValueError, match="job attempt renewals are outside their bound"
    ):
        _attempt(1, renewals=MAX_JOB_RENEWALS + 1)

    observations = tuple(
        ExternalObservation(
            sequence=index,
            observed_at=NOW + timedelta(seconds=index),
            status=ExternalObservedStatus.RUNNING,
            observation_digest="sha256:" + sha256(f"obs-{index}".encode()).hexdigest(),
            external_job_id="external-job-1",
        )
        for index in range(1, MAX_JOB_EXTERNAL_OBSERVATIONS + 1)
    )
    _attempt(1, status=JobAttemptStatus.CLAIMED, observations=observations)
    with pytest.raises(ValueError, match="observation sequence is outside its bound"):
        ExternalObservation(
            sequence=MAX_JOB_EXTERNAL_OBSERVATIONS + 1,
            observed_at=NOW,
            status=ExternalObservedStatus.RUNNING,
            observation_digest="sha256:" + sha256(b"over").hexdigest(),
            external_job_id="external-job-1",
        )


async def test_deadline_horizon_and_wall_time_bounds_are_inclusive(
    tmp_path: Path,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "deadline.sqlite")
    counter = 0

    def create_id(prefix: str) -> str:
        nonlocal counter
        counter += 1
        return f"{prefix}-{counter}"

    owner = JobOwner(
        agent_id="agent-1",
        store=store,
        clock=lambda: NOW,
        id_factory=create_id,
    )
    run = RunInput(
        id="run-origin",
        agent_id="agent-1",
        message="Start a bounded job.",
        created_at=NOW,
        conversation_id="conversation-1",
    )
    try:
        await owner.admit(
            run=run,
            call_id="call-at-limit",
            specification=_specification(
                deadline_at=NOW + timedelta(seconds=MAX_JOB_DEADLINE_SECONDS),
                wall_time=MAX_JOB_WALL_TIME_SECONDS,
            ),
        )
        with pytest.raises(JobError) as raised:
            await owner.admit(
                run=run,
                call_id="call-over-limit",
                specification=_specification(
                    deadline_at=NOW
                    + timedelta(seconds=MAX_JOB_DEADLINE_SECONDS, microseconds=1),
                ),
            )
        assert raised.value.code == "job_deadline_limit_exceeded"
        with pytest.raises(ValueError, match="max_wall_time_seconds"):
            _specification(wall_time=MAX_JOB_WALL_TIME_SECONDS + 0.001)
    finally:
        await store.close()


def test_result_summary_provenance_depth_and_artifact_count_bounds_are_inclusive() -> (
    None
):
    exact_summary = _payload_with_bytes(MAX_JOB_INLINE_RESULT_BYTES)
    exact_provenance = _payload_with_bytes(MAX_JOB_SUMMARY_BYTES)
    _result(exact_summary, {"authority": "test"})
    with pytest.raises(ValueError, match="job result summary exceeds its byte bound"):
        _result(
            {"payload": exact_summary["payload"] + "x"},
            {"authority": "test"},
        )
    _result({"ok": True}, exact_provenance)
    with pytest.raises(
        ValueError, match="job result provenance exceeds its byte bound"
    ):
        _result({"ok": True}, {"payload": exact_provenance["payload"] + "x"})
    _result(_nested(MAX_JOB_RESULT_DEPTH), {})
    with pytest.raises(ValueError, match="job result summary exceeds its depth bound"):
        _result(_nested(MAX_JOB_RESULT_DEPTH + 1), {})

    one_ref = (_artifact_ref(1),)
    assert len(one_ref) == MAX_JOB_ARTIFACTS
    _result({"ok": True}, {}, one_ref)
    with pytest.raises(ValueError, match="job artifact references exceed their bound"):
        _result({"ok": True}, {}, (*one_ref, _artifact_ref(2)))


def test_external_request_response_byte_and_depth_bounds_are_inclusive() -> None:
    exact_request = _payload_with_bytes(MAX_EXTERNAL_REQUEST_BYTES)
    ExternalStartRequest(
        job_id="job-1",
        specification_digest="sha256:" + sha256(b"spec").hexdigest(),
        idempotency_key="start-key",
        arguments=exact_request,
    )
    with pytest.raises(
        ValueError, match="external start request exceeds its byte bound"
    ):
        ExternalStartRequest(
            job_id="job-1",
            specification_digest="sha256:" + sha256(b"spec").hexdigest(),
            idempotency_key="start-key",
            arguments={"payload": exact_request["payload"] + "x"},
        )
    ExternalStartRequest(
        job_id="job-1",
        specification_digest="sha256:" + sha256(b"spec").hexdigest(),
        idempotency_key="start-key",
        arguments=_nested(MAX_JOB_RESULT_DEPTH),
    )
    with pytest.raises(
        ValueError, match="external start request exceeds its depth bound"
    ):
        ExternalStartRequest(
            job_id="job-1",
            specification_digest="sha256:" + sha256(b"spec").hexdigest(),
            idempotency_key="start-key",
            arguments=_nested(MAX_JOB_RESULT_DEPTH + 1),
        )

    exact_response = _payload_with_bytes(MAX_EXTERNAL_RESPONSE_BYTES)
    ExternalStatusReceipt(
        status=ExternalObservedStatus.RUNNING,
        external_job_id="external-job-1",
        observed_at=NOW,
        observation=exact_response,
    )
    with pytest.raises(
        ValueError, match="external status response exceeds its byte bound"
    ):
        ExternalStatusReceipt(
            status=ExternalObservedStatus.RUNNING,
            external_job_id="external-job-1",
            observed_at=NOW,
            observation={"payload": exact_response["payload"] + "x"},
        )
    ExternalStatusReceipt(
        status=ExternalObservedStatus.RUNNING,
        external_job_id="external-job-1",
        observed_at=NOW,
        observation=_nested(MAX_JOB_RESULT_DEPTH),
    )
    with pytest.raises(
        ValueError, match="external status response exceeds its depth bound"
    ):
        ExternalStatusReceipt(
            status=ExternalObservedStatus.RUNNING,
            external_job_id="external-job-1",
            observed_at=NOW,
            observation=_nested(MAX_JOB_RESULT_DEPTH + 1),
        )


async def test_profile_artifact_byte_bound_is_inclusive(tmp_path: Path) -> None:
    agent = await Agent.create(
        "stage-b-artifact-pressure", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    capability, _ = agent._embedded._capabilities.resolve_execution(
        DATA_PROFILE_EXECUTION_CAPABILITY_ID
    )
    policy = capability.artifact_policy
    assert policy is not None
    assert policy.max_artifact_count == MAX_JOB_ARTIFACTS
    assert policy.max_bytes_per_artifact == 1 * 1024 * 1024
    store = agent._embedded._artifact_store
    provenance = ArtifactProvenance(
        authorship=ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS,
        evidence_call_ids=("call-1",),
    )
    try:
        ref = await store.commit(
            ArtifactDraft(
                content=b"x" * policy.max_bytes_per_artifact,
                suggested_filename="profile.json",
                media_type="application/json",
                sensitivity=Sensitivity.INTERNAL,
                provenance=provenance,
            ),
            policy,
            run_id="run-00000000000000000000000000000001",
            conversation_id="conversation-1",
            call_id="call-1",
            capability_id="jobs.data_profile.execute",
        )
        assert ref.byte_size == policy.max_bytes_per_artifact
        with pytest.raises(ArtifactError) as raised:
            await store.commit(
                ArtifactDraft(
                    content=b"x" * (policy.max_bytes_per_artifact + 1),
                    suggested_filename="profile-over.json",
                    media_type="application/json",
                    sensitivity=Sensitivity.INTERNAL,
                    provenance=provenance,
                ),
                policy,
                run_id="run-00000000000000000000000000000002",
                conversation_id="conversation-1",
                call_id="call-2",
                capability_id="jobs.data_profile.execute",
            )
        assert raised.value.code == "artifact_quota_exceeded"
    finally:
        await agent.close()
