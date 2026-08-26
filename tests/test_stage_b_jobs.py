from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest
from _capability_runtime_support import StaticTestDomain, static_registry

from daita import (
    Agent,
    AgentEvent,
    AgentEventKind,
    JobExecutionMode,
    JobStatus,
    LocalDirectorySource,
    SQLiteSource,
)
from daita.adapters.job_profiles import (
    ConnectedJobProfileState,
    ExternalCancelReceipt,
    ExternalResultPayload,
    ExternalStartReceipt,
    ExternalStatusReceipt,
)
from daita.capabilities import (
    AccessMode,
    Capability,
    CapabilityDeclarations,
    CapabilityInputError,
    CapabilityRegistry,
    ToolLoadMode,
    ToolExecution,
    ToolOutput,
    ToolOutputValidationError,
    capability_contract_digest,
)
from daita.capability_runtime import (
    CapabilityRuntime,
    InternalCapabilityRequest,
)
from daita.domains.data.profile_jobs import DATA_PROFILE_EXECUTION_CAPABILITY_ID
from daita.jobs.capabilities import (
    JOB_CANCEL_TOOL_NAME,
    JOB_DOMAIN_OWNER_ID,
    JOB_INSPECT_TOOL_NAME,
    JOB_LIST_TOOL_NAME,
    JOB_READ_RESULTS_TOOL_NAME,
    JobCancelExecutor,
    JobCapabilityDomain,
    JobInspectExecutor,
    JobListExecutor,
    JobReadResultsExecutor,
    job_capability_declarations,
)
from daita.jobs.models import (
    MAX_ACTIVE_JOBS_PER_AGENT,
    MAX_JOB_ATTEMPTS,
    MAX_JOB_DEADLINE_SECONDS,
    MAX_JOB_EXTERNAL_OBSERVATIONS,
    MAX_JOB_LIST_PAGE_SIZE,
    MAX_JOB_SPECIFICATION_BYTES,
    MAX_JOB_WALL_TIME_SECONDS,
    MAX_JOBS_PER_AGENT,
    MAX_QUEUED_JOBS_PER_AGENT,
    MAX_RUNNING_JOBS_GLOBAL,
    MAX_RUNNING_JOBS_PER_AGENT,
    MAX_RUNNING_JOBS_PER_SOURCE,
    ConnectedExecutorBinding,
    ExternalIntentDisposition,
    ExternalIntentKind,
    ExternalObservedStatus,
    JobAttemptStatus,
    JobDesiredState,
    JobResourceBinding,
    JobRun,
    JobSpecification,
)
from daita.jobs.owner import JobOwner
from daita.jobs.supervisor import _ProcessJobCapacity
from daita.llm.models import (
    FinishReason,
    ModelResponse,
    ModelSensitivity,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.loop.models import LoopLimits, RunInput
from daita.storage.sqlite import SQLiteStateStore
from daita.storage.sqlite_records import SourceReadMode, SourceReadScope
from _toolbox_model_support import (
    ToolboxAwareMockModelProvider as MockModelProvider,
)

EAGER_LIMITS = LoopLimits()


class _OversizedInternalExecutor:
    executor_id = "test.internal.result.executor"

    async def execute(self, request: ToolExecution) -> ToolOutput:
        del request
        return ToolOutput(kind="test.internal.result", data={"payload": "x" * 1_024})


def _call(resource_id: str) -> ModelResponse:
    return _call_resources((resource_id,))


def _call_resources(resource_ids: tuple[str, ...]) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCall(
                id="start-profile",
                name="start_data_profile",
                arguments={"resource_ids": list(resource_ids), "sample_rows": 20},
            ),
        ),
    )


def _stop() -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text="Job accepted.")


def _profile(provider: MockModelProvider):
    return provider.model_profile


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE customers(id INTEGER PRIMARY KEY, region TEXT, spend REAL);
            INSERT INTO customers(region, spend) VALUES
                ('north', 12.5), ('south', NULL), ('north', 21.0);
            """)


async def _seed_agent(tmp_path: Path, name: str) -> tuple[str, str]:
    database = tmp_path / f"{name}.sqlite"
    _database(database)
    agent = await Agent.create(name, root=tmp_path)
    try:
        source = await agent.attach(SQLiteSource(database))
        resources = await agent.list_catalog_resources(source_id=source.id)
        assert len(resources) == 1
        await agent._embedded._store.replace_source_permission_scopes(
            SourceReadScope(
                agent_id=agent.id,
                source_id=source.id,
                mode=SourceReadMode.ALL,
            ),
            (),
        )
        return source.id, resources[0].id
    finally:
        await agent.close()


def _job_id(provider: MockModelProvider) -> str:
    return _job_id_at(provider, 1)


def _job_id_at(provider: MockModelProvider, request_index: int) -> str:
    results = tuple(
        block
        for message in provider.logical_requests[request_index].messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
        and block.output.get("kind") != "toolbox_load_receipt"
    )
    assert len(results) == 1
    assert results[0].is_error is False
    data = results[0].output["data"]
    assert isinstance(data, Mapping)
    job_id = data["job_id"]
    assert isinstance(job_id, str)
    return job_id


async def _terminal(agent: Agent, job_id: str):
    inspection = None
    for _ in range(300):
        inspection = await agent.inspect_job(job_id)
        assert inspection is not None
        if inspection.summary.status in {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.NEEDS_ATTENTION,
        }:
            return inspection
        await asyncio.sleep(0.01)
    driver = agent._embedded._job_supervisor._driver
    driver_error = None if driver is None or not driver.done() else driver.exception()
    raise AssertionError(
        f"job did not reach a terminal state: {inspection!r}; driver={driver_error!r}"
    )


async def _running(agent: Agent, job_id: str):
    for _ in range(200):
        inspection = await agent.inspect_job(job_id)
        assert inspection is not None
        if inspection.summary.status is JobStatus.RUNNING:
            return inspection
        await asyncio.sleep(0.005)
    raise AssertionError("job did not enter running state")


def _job_run_input(run_id: str, *, agent_id: str = "agent-1") -> RunInput:
    return RunInput(
        id=run_id,
        agent_id=agent_id,
        message="Inspect durable jobs.",
        created_at=datetime.now(UTC),
        conversation_id=f"conversation-{run_id}",
    )


def _job_runtime(
    store: SQLiteStateStore,
    *,
    agent_id: str = "agent-1",
) -> tuple[CapabilityRuntime, JobOwner]:
    owner = JobOwner(
        agent_id=agent_id,
        store=store,
        clock=lambda: datetime.now(UTC),
        id_factory=lambda prefix: f"{prefix}-projection",
    )
    declarations = job_capability_declarations(owner)
    bundle = CapabilityDeclarations(
        domain_owner_id=JOB_DOMAIN_OWNER_ID,
        capabilities=declarations.capabilities,
        executor_ids=tuple(item.executor_id for item in declarations.capabilities),
        tool_views=declarations.tool_views,
    )
    domain = JobCapabilityDomain(bundle, owner)
    return (
        CapabilityRuntime(
            CapabilityRegistry(
                declarations=(bundle,),
                executors=declarations.executors,
            ),
            (domain,),
        ),
        owner,
    )


def _job_catalog_modes(catalog) -> dict[str, ToolLoadMode]:
    return {
        entry.view.name: entry.load_mode
        for entry in catalog.entries
        if entry.domain_owner_id == JOB_DOMAIN_OWNER_ID
    }


class _OfflineProfile:
    profile_id = "offline-profile"

    def __init__(
        self,
        *,
        lose_start_response: bool = False,
        wait_for_cancel: bool = False,
        active: bool = True,
        oversized_result: bool = False,
        drift_for_worker: bool = False,
        hold_status: bool = False,
        supported_job_kinds: frozenset[str] = frozenset({"data_profile"}),
        maximum_sensitivity: ModelSensitivity = ModelSensitivity.RESTRICTED,
    ) -> None:
        self.binding = ConnectedExecutorBinding(
            profile_id=self.profile_id,
            binding_id="offline-binding",
            execution_identity="offline-executor-v1",
            contract_digest="sha256:" + sha256(b"offline-profile-v1").hexdigest(),
            revision=1,
            maximum_sensitivity=maximum_sensitivity,
        )
        self.lose_start_response = lose_start_response
        self.wait_for_cancel = wait_for_cancel
        self.active = active
        self.oversized_result = oversized_result
        self.drift_for_worker = drift_for_worker
        self.status_entered = asyncio.Event() if hold_status else None
        self.status_release = asyncio.Event() if hold_status else None
        self.supported_job_kinds = supported_job_kinds
        self.started = False
        self.cancelled = False
        self.start_calls = 0
        self.cancel_calls = 0
        self.status_calls = 0

    def current_state(self) -> ConnectedJobProfileState:
        binding = self.binding
        task = asyncio.current_task()
        if (
            self.drift_for_worker
            and task is not None
            and task.get_name().startswith("daita-job:")
        ):
            binding = replace(self.binding, revision=self.binding.revision + 1)
        return ConnectedJobProfileState(
            binding=binding,
            supported_job_kinds=self.supported_job_kinds,
            active=self.active,
        )

    async def start(self, request) -> ExternalStartReceipt:
        self.start_calls += 1
        self.started = True
        if self.lose_start_response:
            raise ConnectionError("offline lost response")
        return ExternalStartReceipt(
            disposition=ExternalIntentDisposition.ACCEPTED,
            observed_at=datetime.now(UTC),
            external_job_id="external-job-1",
        )

    async def status(self, request) -> ExternalStatusReceipt:
        self.status_calls += 1
        if self.status_entered is not None and self.status_release is not None:
            self.status_entered.set()
            await self.status_release.wait()
        if self.wait_for_cancel:
            await asyncio.sleep(0.01)
        status = (
            ExternalObservedStatus.CANCELLED
            if self.cancelled
            else (
                ExternalObservedStatus.RUNNING
                if self.wait_for_cancel
                else ExternalObservedStatus.SUCCEEDED
            )
        )
        return ExternalStatusReceipt(
            status=status,
            external_job_id="external-job-1",
            observed_at=datetime.now(UTC),
            observation={"state": status.value},
        )

    async def cancel(self, request) -> ExternalCancelReceipt:
        self.cancel_calls += 1
        self.cancelled = True
        raise ConnectionError("offline lost cancel response")

    async def read_result(self, request) -> ExternalResultPayload:
        return ExternalResultPayload(
            summary=(
                {"payload": "x" * (70 * 1024)}
                if self.oversized_result
                else {"profiled_resources": 1, "external": True}
            ),
            sensitivity=ModelSensitivity.INTERNAL,
            provenance={"fixture": "offline-connected-executor"},
            observed_at=datetime.now(UTC),
        )


async def test_job_projection_is_bounded_direct_and_frozen_per_run(
    tmp_path: Path,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "job-projection.sqlite")
    runtime, owner = _job_runtime(store)
    now = datetime.now(UTC)
    try:
        empty = await runtime.prepare_run(_job_run_input("run-empty"))
        assert _job_catalog_modes(empty) == {JOB_LIST_TOOL_NAME: ToolLoadMode.PINNED}

        await store.admit_job(
            _stored_job(
                "job-projection",
                agent_id=owner.agent_id,
                now=now,
            )
        )
        active = await runtime.prepare_run(_job_run_input("run-active"))
        assert _job_catalog_modes(active) == {
            JOB_CANCEL_TOOL_NAME: ToolLoadMode.ON_DEMAND,
            JOB_INSPECT_TOOL_NAME: ToolLoadMode.PINNED,
            JOB_LIST_TOOL_NAME: ToolLoadMode.PINNED,
            JOB_READ_RESULTS_TOOL_NAME: ToolLoadMode.PINNED,
        }
        foreign = await runtime.prepare_run(
            _job_run_input("run-foreign", agent_id="agent-2")
        )
        assert _job_catalog_modes(foreign) == {}
        assert _job_catalog_modes(empty) == {JOB_LIST_TOOL_NAME: ToolLoadMode.PINNED}

        cancelled = await owner.cancel("job-projection")
        assert cancelled is not None
        assert cancelled.summary.status is JobStatus.CANCELLED
        terminal = await runtime.prepare_run(_job_run_input("run-terminal"))
        assert _job_catalog_modes(terminal) == {
            JOB_INSPECT_TOOL_NAME: ToolLoadMode.PINNED,
            JOB_LIST_TOOL_NAME: ToolLoadMode.PINNED,
            JOB_READ_RESULTS_TOOL_NAME: ToolLoadMode.PINNED,
        }
    finally:
        await store.close()


async def test_job_lifecycle_executors_are_agent_scoped_and_origin_is_provenance(
    tmp_path: Path,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "job-agent-scope.sqlite")
    _, owner = _job_runtime(store)
    foreign_owner = _job_runtime(store, agent_id="agent-2")[1]
    job = _stored_job(
        "job-agent-scope",
        agent_id=owner.agent_id,
        now=datetime.now(UTC),
    )
    await store.admit_job(job)
    request = {
        "run_id": "run-new-conversation",
        "conversation_id": "conversation-new",
    }
    try:
        listed = await JobListExecutor(owner).execute(
            ToolExecution(
                **request,
                call_id="list",
                capability_id="jobs.list",
                arguments={"statuses": (JobStatus.QUEUED.value,)},
            )
        )
        summaries = listed.data["jobs"]
        assert isinstance(summaries, tuple)
        assert summaries[0]["job_id"] == job.job_id
        assert summaries[0]["origin_conversation_id"] == job.conversation_id
        assert dict(listed.sensitivity_provenance) == {
            "authority": "job_owner_agent_scope",
            "agent_id": owner.agent_id,
        }

        inspected = await JobInspectExecutor(owner).execute(
            ToolExecution(
                **request,
                call_id="inspect",
                capability_id="jobs.inspect",
                arguments={"job_id": job.job_id},
            )
        )
        assert inspected.data["origin_conversation_id"] == job.conversation_id
        assert "conversation_id" not in inspected.data
        assert inspected.sensitivity_provenance["authority"] == (
            "job_owner_agent_scope"
        )

        cancelled = await JobCancelExecutor(owner).execute(
            ToolExecution(
                **request,
                call_id="cancel",
                capability_id="jobs.cancel",
                arguments={"job_id": job.job_id},
            )
        )
        assert cancelled.data["status"] == JobStatus.CANCELLED.value
        assert cancelled.sensitivity_provenance["agent_id"] == owner.agent_id

        assert await foreign_owner.inspect(job.job_id) is None
        assert await foreign_owner.list() == ()
        with pytest.raises(
            CapabilityInputError,
            match="not owned by this agent",
        ):
            await JobInspectExecutor(foreign_owner).execute(
                ToolExecution(
                    **request,
                    call_id="foreign-inspect",
                    capability_id="jobs.inspect",
                    arguments={"job_id": job.job_id},
                )
            )
    finally:
        await store.close()


async def test_job_context_is_agent_scoped_and_result_first(tmp_path: Path) -> None:
    provider = MockModelProvider((_stop(),), provider_id="mock:job-context")
    agent = await Agent.create(
        "job-context",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
    )
    try:
        await agent._embedded._store.admit_job(
            _stored_job(
                "job-context",
                agent_id=agent.id,
                now=datetime.now(UTC),
            )
        )
        await agent.run("What jobs are running?")
        system = provider.requests[0].messages[0].content[0]
        assert isinstance(system, TextBlock)
        assert "owned by this agent across conversations" in system.text
        assert "origin_conversation_id is provenance" in system.text
        assert "known job ID, call job_read_results first" in system.text
        assert "Use job_inspect only" in system.text
        search = next(
            item for item in provider.requests[0].tools if item.name == "toolbox_search"
        )
        properties = search.input_schema["properties"]
        assert isinstance(properties, Mapping)
        data_access = properties["data_access"]
        assert isinstance(data_access, Mapping)
        assert "External/source-data filter" in data_access["description"]
        assert "Omit when uncertain" in data_access["description"]
    finally:
        await agent.close()


async def test_daita_data_profile_runs_after_originating_interaction_and_reopens(
    tmp_path: Path,
) -> None:
    source_id, resource_id = await _seed_agent(tmp_path, "stage-b-daita")
    unused_profile = _OfflineProfile()
    events: list[AgentEvent] = []
    provider = MockModelProvider((_call(resource_id), _stop()))
    agent = await Agent.open(
        "stage-b-daita",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
        connected_job_profiles=(unused_profile,),
        observer=events.append,
    )
    try:
        exit = await agent.run("Profile the customer data.", source_id=source_id)
        assert exit.final_text == "Job accepted."
        job_id = _job_id(provider)
        inspection = await _terminal(agent, job_id)
        assert inspection.summary.status is JobStatus.SUCCEEDED
        assert inspection.summary.execution_mode is JobExecutionMode.DAITA
        assert unused_profile.start_calls == 0
        result = await agent.read_job_result(job_id)
        assert result is not None
        assert result.summary["profiled_resources"] == 1
        assert len(result.artifact_refs) == 1
        payload = await agent.read_artifact(result.artifact_refs[0].artifact_id)
        document = json.loads(payload.content)
        assert document["job_id"] == job_id
        assert document["resources"][0]["sampled_rows"] == 3
        internal_events = tuple(
            event
            for event in events
            if event.data.get("capability_id") == DATA_PROFILE_EXECUTION_CAPABILITY_ID
        )
        assert tuple(event.kind for event in internal_events) == (
            AgentEventKind.TOOL_STARTED,
            AgentEventKind.TOOL_COMPLETED,
        )
        assert internal_events[-1].data["success"] is True
    finally:
        await agent.close()

    reopened = await Agent.open("stage-b-daita", root=tmp_path)
    try:
        persisted = await reopened.inspect_job(job_id)
        assert persisted is not None
        assert persisted.summary.status is JobStatus.SUCCEEDED
        assert (await reopened.read_job_result(job_id)) is not None
    finally:
        await reopened.close()


async def test_data_profile_reads_csv_and_json_through_the_existing_file_domain(
    tmp_path: Path,
) -> None:
    files = tmp_path / "profile-files"
    files.mkdir()
    (files / "customers.csv").write_text(
        "id,region\n1,north\n2,south\n",
        encoding="utf-8",
    )
    (files / "orders.json").write_text(
        '[{"id": 10, "amount": 5.5}, {"id": 11, "amount": null}]',
        encoding="utf-8",
    )
    seed = await Agent.create("stage-b-local-files", root=tmp_path)
    try:
        source = await seed.attach(LocalDirectorySource(files))
        resources = await seed.list_catalog_resources(source_id=source.id)
        profiled_resources = tuple(
            item for item in resources if item.name in {"customers.csv", "orders.json"}
        )
        assert len(profiled_resources) == 2
        await seed._embedded._store.replace_source_permission_scopes(
            SourceReadScope(
                agent_id=seed.id,
                source_id=source.id,
                mode=SourceReadMode.ALL,
            ),
            (),
        )
    finally:
        await seed.close()

    provider = MockModelProvider(
        (_call_resources(tuple(item.id for item in profiled_resources)), _stop())
    )
    agent = await Agent.open(
        "stage-b-local-files",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
    )
    try:
        await agent.run("Profile both local files.", source_id=source.id)
        job_id = _job_id(provider)
        inspection = await _terminal(agent, job_id)
        assert inspection.summary.status is JobStatus.SUCCEEDED
        result = await agent.read_job_result(job_id)
        assert result is not None
        assert result.summary["profiled_resources"] == 2
        payload = await agent.read_artifact(result.artifact_refs[0].artifact_id)
        document = json.loads(payload.content)
        assert {item["name"] for item in document["resources"]} == {
            "customers.csv",
            "orders.json",
        }
    finally:
        await agent.close()


async def test_explicit_external_selection_reconciles_a_lost_start_response(
    tmp_path: Path,
) -> None:
    source_id, resource_id = await _seed_agent(tmp_path, "stage-b-external")
    profile = _OfflineProfile(lose_start_response=True)
    provider = MockModelProvider((_call(resource_id), _stop()))
    agent = await Agent.open(
        "stage-b-external",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
        connected_job_profiles=(profile,),
    )
    try:
        await agent.run(
            "Use the selected connected executor.",
            source_id=source_id,
            job_executor_profile_id=profile.profile_id,
        )
        job_id = _job_id(provider)
        inspection = await _terminal(agent, job_id)
        assert inspection.summary.status is JobStatus.SUCCEEDED
        assert inspection.summary.execution_mode is JobExecutionMode.CONNECTED_EXECUTOR
        assert profile.start_calls == 1
        assert profile.status_calls >= 1
        start = inspection.attempts[0].external_intents[0]
        assert start.kind is ExternalIntentKind.START
        assert start.disposition is ExternalIntentDisposition.OUTCOME_UNKNOWN
        assert inspection.attempts[0].external_observations[-1].status is (
            ExternalObservedStatus.SUCCEEDED
        )
        result = await agent.read_job_result(job_id)
        assert result is not None
        assert result.summary["external"] is True
        assert not result.artifact_refs
    finally:
        await agent.close()


async def test_external_cancel_intent_is_durable_before_lost_response(
    tmp_path: Path,
) -> None:
    source_id, resource_id = await _seed_agent(tmp_path, "stage-b-cancel")
    profile = _OfflineProfile(wait_for_cancel=True)
    provider = MockModelProvider((_call(resource_id), _stop()))
    agent = await Agent.open(
        "stage-b-cancel",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
        connected_job_profiles=(profile,),
    )
    try:
        await agent.run(
            "Start then cancel the selected connected job.",
            source_id=source_id,
            job_executor_profile_id=profile.profile_id,
        )
        job_id = _job_id(provider)
        await _running(agent, job_id)
        cancelled = await agent.cancel_job(job_id)
        assert cancelled is not None
        inspection = await _terminal(agent, job_id)
        assert inspection.summary.status is JobStatus.CANCELLED
        assert profile.cancel_calls == 1
        intents = {item.kind: item for item in inspection.attempts[0].external_intents}
        assert intents[ExternalIntentKind.CANCEL].disposition is (
            ExternalIntentDisposition.OUTCOME_UNKNOWN
        )
        assert await agent.read_job_result(job_id) is None
    finally:
        await agent.close()


@pytest.mark.parametrize(
    ("case", "profile", "selection", "expected_code"),
    (
        ("unconnected", None, "missing-profile", "external_executor_not_connected"),
        (
            "revoked",
            _OfflineProfile(active=False),
            _OfflineProfile.profile_id,
            "external_executor_not_admitted",
        ),
        (
            "unsupported",
            _OfflineProfile(supported_job_kinds=frozenset({"other_job"})),
            _OfflineProfile.profile_id,
            "external_executor_not_admitted",
        ),
        (
            "unauthorized",
            _OfflineProfile(maximum_sensitivity=ModelSensitivity.PUBLIC),
            _OfflineProfile.profile_id,
            "external_executor_sensitivity_blocked",
        ),
    ),
)
async def test_invalid_external_selection_fails_before_admission_or_io(
    tmp_path: Path,
    case: str,
    profile: _OfflineProfile | None,
    selection: str,
    expected_code: str,
) -> None:
    source_id, resource_id = await _seed_agent(tmp_path, f"stage-b-{case}")
    provider = MockModelProvider((_call(resource_id), _stop()))
    agent = await Agent.open(
        f"stage-b-{case}",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
        connected_job_profiles=(() if profile is None else (profile,)),
    )
    try:
        await agent.run(
            "Try only the explicitly selected connected executor.",
            source_id=source_id,
            job_executor_profile_id=selection,
        )
        results = tuple(
            block
            for message in provider.logical_requests[1].messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
            and block.output.get("kind") != "toolbox_load_receipt"
        )
        assert len(results) == 1 and results[0].is_error is True
        error = results[0].output["error"]
        assert isinstance(error, Mapping)
        assert error["code"] == expected_code
        if profile is not None:
            assert profile.start_calls == 0
        assert await agent.list_jobs() == ()
    finally:
        await agent.close()


async def test_external_result_bound_fails_closed_without_daita_fallback(
    tmp_path: Path,
) -> None:
    source_id, resource_id = await _seed_agent(tmp_path, "stage-b-result-bound")
    profile = _OfflineProfile(oversized_result=True)
    provider = MockModelProvider((_call(resource_id), _stop()))
    agent = await Agent.open(
        "stage-b-result-bound",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
        connected_job_profiles=(profile,),
    )
    try:
        await agent.run(
            "Use the selected executor and enforce the result bound.",
            source_id=source_id,
            job_executor_profile_id=profile.profile_id,
        )
        job_id = _job_id(provider)
        inspection = await _terminal(agent, job_id)
        assert inspection.summary.status is JobStatus.NEEDS_ATTENTION
        assert inspection.summary.execution_mode is JobExecutionMode.CONNECTED_EXECUTOR
        assert inspection.failure_code == "job_contract_revalidation_failed"
        assert profile.start_calls == 1
        assert await agent.read_job_result(job_id) is None
    finally:
        await agent.close()


async def test_external_binding_drift_is_rejected_before_external_io(
    tmp_path: Path,
) -> None:
    source_id, resource_id = await _seed_agent(tmp_path, "stage-b-drift")
    profile = _OfflineProfile(drift_for_worker=True)
    provider = MockModelProvider((_call(resource_id), _stop()))
    agent = await Agent.open(
        "stage-b-drift",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
        connected_job_profiles=(profile,),
    )
    try:
        await agent.run(
            "Use the exact selected executor only.",
            source_id=source_id,
            job_executor_profile_id=profile.profile_id,
        )
        inspection = await _terminal(agent, _job_id(provider))
        assert inspection.summary.status is JobStatus.NEEDS_ATTENTION
        assert inspection.failure_code == "external_executor_drifted"
        assert profile.start_calls == 0
        intent = inspection.attempts[0].external_intents[0]
        assert intent.kind is ExternalIntentKind.START
        assert intent.disposition is ExternalIntentDisposition.REJECTED
        assert intent.reason_code == "external_executor_drifted"
    finally:
        await agent.close()


async def test_external_cancel_intent_precedes_fresh_profile_revalidation(
    tmp_path: Path,
) -> None:
    source_id, resource_id = await _seed_agent(tmp_path, "stage-b-cancel-drift")
    profile = _OfflineProfile(wait_for_cancel=True, hold_status=True)
    provider = MockModelProvider((_call(resource_id), _stop()))
    agent = await Agent.open(
        "stage-b-cancel-drift",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
        connected_job_profiles=(profile,),
    )
    try:
        await agent.run(
            "Start the selected executor and then cancel it.",
            source_id=source_id,
            job_executor_profile_id=profile.profile_id,
        )
        job_id = _job_id(provider)
        assert profile.status_entered is not None
        assert profile.status_release is not None
        await asyncio.wait_for(profile.status_entered.wait(), timeout=2)
        profile.active = False
        cancelled = await agent.cancel_job(job_id)
        assert cancelled is not None
        profile.status_release.set()
        inspection = await _terminal(agent, job_id)
        assert inspection.summary.status is JobStatus.NEEDS_ATTENTION
        assert inspection.failure_code == "external_executor_drifted"
        assert profile.cancel_calls == 0
        intents = {item.kind: item for item in inspection.attempts[0].external_intents}
        cancel_intent = intents[ExternalIntentKind.CANCEL]
        assert cancel_intent.disposition is ExternalIntentDisposition.REJECTED
        assert cancel_intent.reason_code == "external_executor_drifted"
    finally:
        await agent.close()


async def test_reopen_past_deadline_reconciles_authoritative_external_success(
    tmp_path: Path,
) -> None:
    source_id, resource_id = await _seed_agent(tmp_path, "stage-b-expired-external")
    current_time = [datetime.now(UTC)]
    clock = lambda: current_time[0]
    profile = _OfflineProfile(hold_status=True)
    provider = MockModelProvider((_call(resource_id), _stop()))
    agent = await Agent.open(
        "stage-b-expired-external",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
        clock=clock,
        connected_job_profiles=(profile,),
    )
    await agent.run(
        "Start external work that must be reconciled after host loss.",
        source_id=source_id,
        job_executor_profile_id=profile.profile_id,
    )
    job_id = _job_id(provider)
    assert profile.status_entered is not None
    assert profile.status_release is not None
    await asyncio.wait_for(profile.status_entered.wait(), timeout=2)
    await agent.close()

    current_time[0] += timedelta(seconds=301)
    profile.status_release.set()
    reopened = await Agent.open(
        "stage-b-expired-external",
        root=tmp_path,
        clock=clock,
        connected_job_profiles=(profile,),
    )
    try:
        inspection = await _terminal(reopened, job_id)
        assert inspection.summary.status is JobStatus.SUCCEEDED
        result = await reopened.read_job_result(job_id)
        assert result is not None
        assert result.summary["external"] is True
        assert profile.status_calls >= 2
    finally:
        await reopened.close()


async def test_public_and_model_job_result_reads_are_side_effect_free(
    tmp_path: Path,
) -> None:
    source_id, resource_id = await _seed_agent(tmp_path, "stage-b-result-read")
    provider = MockModelProvider((_call(resource_id), _stop()))
    agent = await Agent.open(
        "stage-b-result-read",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
    )
    try:
        await agent.run("Profile the source.", source_id=source_id)
        job_id = _job_id(provider)
        await _terminal(agent, job_id)
        before = await agent._embedded._store.load_job(agent.id, job_id)
        assert before is not None and before.terminal_observed_at is None

        assert await agent.read_job_result(job_id) is not None
        after_public = await agent._embedded._store.load_job(agent.id, job_id)
        assert after_public == before

        output = await JobReadResultsExecutor(agent._embedded._job_owner).execute(
            ToolExecution(
                run_id="run-result-read",
                call_id="call-result-read",
                capability_id="jobs.read_results",
                arguments={"job_id": job_id},
                conversation_id="conversation-independent-result-read",
            )
        )
        assert output.data["job_id"] == job_id
        assert output.sensitivity_provenance["authority"] == ("job_owner_agent_scope")
        assert output.sensitivity_provenance["agent_id"] == agent.id
        after_model = await agent._embedded._store.load_job(agent.id, job_id)
        assert after_model == before
    finally:
        await agent.close()


async def test_same_source_jobs_run_concurrently_and_cancelling_one_isolates_its_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_id, resource_id = await _seed_agent(tmp_path, "stage-b-siblings")
    provider = MockModelProvider(
        (_call(resource_id), _stop(), _call(resource_id), _stop())
    )
    agent = await Agent.open(
        "stage-b-siblings",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
    )
    _, executor = agent._embedded._capabilities.resolve_execution(
        DATA_PROFILE_EXECUTION_CAPABILITY_ID
    )
    original_execute = executor.execute
    releases: dict[str, asyncio.Event] = {}
    started: set[str] = set()
    both_started = asyncio.Event()

    async def controlled_execute(request):
        job_id = request.arguments["job_id"]
        assert isinstance(job_id, str)
        release = releases.setdefault(job_id, asyncio.Event())
        started.add(job_id)
        if len(started) == 2:
            both_started.set()
        await release.wait()
        return await original_execute(request)

    monkeypatch.setattr(executor, "execute", controlled_execute)
    try:
        await agent.run("Start the first profile.", source_id=source_id)
        first_id = _job_id_at(provider, 1)
        await agent.run("Start an independent sibling profile.", source_id=source_id)
        second_id = _job_id_at(provider, 3)
        await asyncio.wait_for(both_started.wait(), timeout=2)
        assert started == {first_id, second_id}

        cancelled = await agent.cancel_job(first_id)
        assert cancelled is not None
        first = await _terminal(agent, first_id)
        assert first.summary.status is JobStatus.CANCELLED
        assert await agent.read_job_result(first_id) is None

        sibling = await agent.inspect_job(second_id)
        assert sibling is not None
        assert sibling.summary.status is JobStatus.RUNNING
        releases[second_id].set()
        sibling = await _terminal(agent, second_id)
        assert sibling.summary.status is JobStatus.SUCCEEDED
        assert await agent.read_job_result(second_id) is not None
        assert len(await agent.list_jobs()) == 2
    finally:
        for release in releases.values():
            release.set()
        await agent.close()


async def test_host_reopen_fences_and_safely_restarts_an_interrupted_daita_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_id, resource_id = await _seed_agent(tmp_path, "stage-b-host-reopen")
    provider = MockModelProvider((_call(resource_id), _stop()))
    agent = await Agent.open(
        "stage-b-host-reopen",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
    )
    _, executor = agent._embedded._capabilities.resolve_execution(
        DATA_PROFILE_EXECUTION_CAPABILITY_ID
    )
    entered = asyncio.Event()
    never = asyncio.Event()

    async def interrupted_execute(request):
        del request
        entered.set()
        await never.wait()
        raise AssertionError("interrupted execution unexpectedly resumed")

    monkeypatch.setattr(executor, "execute", interrupted_execute)
    closed = False
    try:
        await agent.run("Start a profile before host close.", source_id=source_id)
        job_id = _job_id(provider)
        await asyncio.wait_for(entered.wait(), timeout=2)
        running = await agent.inspect_job(job_id)
        assert running is not None and running.summary.status is JobStatus.RUNNING
        await agent.close()
        closed = True
    finally:
        if not closed:
            await agent.close()

    reopened = await Agent.open("stage-b-host-reopen", root=tmp_path)
    try:
        terminal = await _terminal(reopened, job_id)
        assert terminal.summary.status is JobStatus.SUCCEEDED
        assert tuple(item.status for item in terminal.attempts) == (
            JobAttemptStatus.FENCED,
            JobAttemptStatus.SUCCEEDED,
        )
        assert terminal.attempts[1].fencing_epoch > terminal.attempts[0].fencing_epoch
        result = await reopened.read_job_result(job_id)
        assert result is not None and len(result.artifact_refs) == 1
        assert len(await reopened._embedded._artifact_store.list_refs()) == 1
    finally:
        await reopened.close()


async def test_stale_internal_capability_digest_fails_before_executor_or_source_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_id, _ = await _seed_agent(tmp_path, "stage-b-stale-capability")
    agent = await Agent.open("stage-b-stale-capability", root=tmp_path)
    _, executor = agent._embedded._capabilities.resolve_execution(
        DATA_PROFILE_EXECUTION_CAPABILITY_ID
    )
    original_execute = executor.execute
    executor_calls = 0

    async def counted_execute(request):
        nonlocal executor_calls
        executor_calls += 1
        return await original_execute(request)

    monkeypatch.setattr(executor, "execute", counted_execute)
    job = _stored_job(
        "stale-capability-job",
        agent_id=agent.id,
        source_id=source_id,
        now=datetime.now(UTC),
        contract_digest="sha256:" + sha256(b"stale-capability").hexdigest(),
    )
    try:
        await agent._embedded._store.admit_job(job)
        agent._embedded._job_supervisor.wake()
        terminal = await _terminal(agent, job.job_id)
        assert terminal.summary.status is JobStatus.NEEDS_ATTENTION
        assert terminal.failure_code == "job_contract_revalidation_failed"
        assert executor_calls == 0
        assert await agent.read_job_result(job.job_id) is None
    finally:
        await agent.close()


async def test_revoked_read_scope_is_rechecked_before_internal_executor_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_id, resource_id = await _seed_agent(tmp_path, "stage-b-scope-recheck")
    provider = MockModelProvider((_call(resource_id), _stop()))
    held_capacity = 0
    for _ in range(MAX_RUNNING_JOBS_GLOBAL):
        assert _ProcessJobCapacity.acquire() is True
        held_capacity += 1
    agent = await Agent.open(
        "stage-b-scope-recheck",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
    )
    _, executor = agent._embedded._capabilities.resolve_execution(
        DATA_PROFILE_EXECUTION_CAPABILITY_ID
    )
    original_execute = executor.execute
    executor_calls = 0

    async def counted_execute(request):
        nonlocal executor_calls
        executor_calls += 1
        return await original_execute(request)

    monkeypatch.setattr(executor, "execute", counted_execute)
    try:
        await agent.run("Queue a profile before scope revocation.", source_id=source_id)
        job_id = _job_id(provider)
        queued = await agent.inspect_job(job_id)
        assert queued is not None and queued.summary.status is JobStatus.QUEUED
        await agent._embedded._store.replace_source_permission_scopes(
            SourceReadScope(
                agent_id=agent.id,
                source_id=source_id,
                mode=SourceReadMode.NONE,
            ),
            (),
        )
        for _ in range(held_capacity):
            _ProcessJobCapacity.release()
        held_capacity = 0
        agent._embedded._job_supervisor.wake()
        terminal = await _terminal(agent, job_id)
        assert terminal.summary.status is JobStatus.NEEDS_ATTENTION
        assert terminal.failure_code == "resource_read_not_allowed"
        assert executor_calls == 0
        assert await agent.read_job_result(job_id) is None
    finally:
        for _ in range(held_capacity):
            _ProcessJobCapacity.release()
        await agent.close()


async def test_external_start_revalidates_current_scope_after_persisting_intent(
    tmp_path: Path,
) -> None:
    source_id, resource_id = await _seed_agent(tmp_path, "stage-b-external-scope")
    profile = _OfflineProfile()
    provider = MockModelProvider((_call(resource_id), _stop()))
    held_capacity = 0
    for _ in range(MAX_RUNNING_JOBS_GLOBAL):
        assert _ProcessJobCapacity.acquire() is True
        held_capacity += 1
    agent = await Agent.open(
        "stage-b-external-scope",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
        connected_job_profiles=(profile,),
    )
    try:
        await agent.run(
            "Queue selected external work before scope revocation.",
            source_id=source_id,
            job_executor_profile_id=profile.profile_id,
        )
        job_id = _job_id(provider)
        await agent._embedded._store.replace_source_permission_scopes(
            SourceReadScope(
                agent_id=agent.id,
                source_id=source_id,
                mode=SourceReadMode.NONE,
            ),
            (),
        )
        for _ in range(held_capacity):
            _ProcessJobCapacity.release()
        held_capacity = 0
        agent._embedded._job_supervisor.wake()
        terminal = await _terminal(agent, job_id)
        assert terminal.summary.status is JobStatus.NEEDS_ATTENTION
        assert terminal.failure_code == "resource_read_not_allowed"
        assert profile.start_calls == 0
        intent = terminal.attempts[0].external_intents[0]
        assert intent.kind is ExternalIntentKind.START
        assert intent.disposition is ExternalIntentDisposition.REJECTED
        assert intent.reason_code == "resource_read_not_allowed"
    finally:
        for _ in range(held_capacity):
            _ProcessJobCapacity.release()
        await agent.close()


async def test_published_artifact_wins_a_cancellation_completion_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_id, resource_id = await _seed_agent(tmp_path, "stage-b-cancel-race")
    provider = MockModelProvider((_call(resource_id), _stop()))
    agent = await Agent.open(
        "stage-b-cancel-race",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        limits=EAGER_LIMITS,
    )
    store = agent._embedded._store
    original_finalize = store.finalize_job_attempt
    finalize_entered = asyncio.Event()
    intercepted = False

    async def gated_finalize(*args, **kwargs):
        nonlocal intercepted
        if (
            kwargs.get("attempt_status") is JobAttemptStatus.SUCCEEDED
            and not intercepted
        ):
            intercepted = True
            finalize_entered.set()
            await asyncio.Event().wait()
        return await original_finalize(*args, **kwargs)

    monkeypatch.setattr(store, "finalize_job_attempt", gated_finalize)
    try:
        await agent.run(
            "Start a profile and retain any known result.", source_id=source_id
        )
        job_id = _job_id(provider)
        await asyncio.wait_for(finalize_entered.wait(), timeout=2)
        requested = await agent.cancel_job(job_id)
        assert requested is not None
        terminal = await _terminal(agent, job_id)
        assert terminal.summary.status is JobStatus.SUCCEEDED
        result = await agent.read_job_result(job_id)
        assert result is not None and len(result.artifact_refs) == 1
        assert result.summary["recovered_artifact_id"] == (
            result.artifact_refs[0].artifact_id
        )
        assert len(await agent._embedded._artifact_store.list_refs()) == 1
    finally:
        await agent.close()


async def test_internal_runtime_reuses_result_bounds_and_failure_observation() -> None:
    capability = Capability(
        id="test.internal.result",
        description="Return one deliberately oversized internal result.",
        input_schema={"type": "object", "properties": {}},
        output_kind="test.internal.result",
        output_schema={
            "type": "object",
            "properties": {"payload": {"type": "string"}},
            "required": ["payload"],
            "additionalProperties": False,
        },
        executor_id=_OversizedInternalExecutor.executor_id,
        access_mode=AccessMode.READ,
    )
    domain = StaticTestDomain((capability,), ())
    registry = static_registry(domain, (_OversizedInternalExecutor(),))
    events: list[AgentEvent] = []
    runtime = CapabilityRuntime(
        registry,
        (domain,),
        observer=events.append,
        limits=LoopLimits(max_tool_result_bytes=128),
    )
    run = RunInput(
        id="run-" + "a" * 32,
        agent_id="agent-1",
        message="Execute a bounded internal capability.",
        created_at=datetime.now(UTC),
        conversation_id="conversation-1",
    )
    with pytest.raises(ToolOutputValidationError):
        await runtime.execute_internal(
            InternalCapabilityRequest(
                run=run,
                call_id="internal-call",
                capability_id=capability.id,
                contract_digest=capability_contract_digest(
                    capability,
                    domain_owner_id=domain.domain_owner_id,
                ),
                arguments={},
                sensitivity=ModelSensitivity.INTERNAL,
            )
        )
    assert tuple(event.kind for event in events) == (
        AgentEventKind.TOOL_STARTED,
        AgentEventKind.TOOL_COMPLETED,
    )
    assert events[-1].data["success"] is False
    assert events[-1].data["error_code"] == "tool_result_too_large"


def test_job_pressure_constants_and_process_global_boundary_are_exact() -> None:
    assert (
        MAX_JOBS_PER_AGENT,
        MAX_ACTIVE_JOBS_PER_AGENT,
        MAX_QUEUED_JOBS_PER_AGENT,
        MAX_RUNNING_JOBS_PER_AGENT,
        MAX_RUNNING_JOBS_GLOBAL,
        MAX_RUNNING_JOBS_PER_SOURCE,
        MAX_JOB_SPECIFICATION_BYTES,
        MAX_JOB_ATTEMPTS,
        MAX_JOB_EXTERNAL_OBSERVATIONS,
        MAX_JOB_WALL_TIME_SECONDS,
        MAX_JOB_DEADLINE_SECONDS,
        MAX_JOB_LIST_PAGE_SIZE,
    ) == (256, 52, 48, 4, 8, 2, 64 * 1024, 3, 32, 300.0, 3_600.0, 50)

    acquired = tuple(_ProcessJobCapacity.acquire() for _ in range(9))
    assert acquired == (True,) * 8 + (False,)
    for _ in range(8):
        _ProcessJobCapacity.release()


async def test_queue_limit_is_inclusive_and_overflow_fails_before_claim(
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    store = await SQLiteStateStore.open(tmp_path / "job-pressure.sqlite")
    try:
        for index in range(MAX_QUEUED_JOBS_PER_AGENT):
            await store.admit_job(
                _stored_job(
                    f"job-{index:03d}",
                    now=now + timedelta(microseconds=index),
                )
            )
        assert len(await store.list_jobs("agent-1", limit=50)) == 48
        with pytest.raises(ValueError, match="job_queue_limit_exceeded"):
            await store.admit_job(
                _stored_job("job-overflow", now=now + timedelta(seconds=1))
            )
    finally:
        await store.close()


async def test_store_enforces_source_and_agent_concurrency_with_sibling_isolation(
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    store = await SQLiteStateStore.open(tmp_path / "job-concurrency.sqlite")

    async def claim(agent_id: str, index: int):
        suffix = f"{index:032x}"
        return await store.claim_next_job(
            agent_id,
            claim_token=f"claim-{agent_id}-{index}",
            execution_run_id=f"run-{suffix}",
            reserved_artifact_id=f"artifact-{suffix}",
            claimed_at=now + timedelta(seconds=1),
            lease_seconds=60,
        )

    try:
        for index in range(3):
            await store.admit_job(
                _stored_job(
                    f"same-source-{index}",
                    now=now + timedelta(microseconds=index),
                )
            )
        await store.admit_job(
            _stored_job(
                "independent-source",
                now=now + timedelta(microseconds=3),
                source_id="source-2",
                resource_id="resource-2",
            )
        )

        first = await claim("agent-1", 1)
        second = await claim("agent-1", 2)
        independent = await claim("agent-1", 3)
        assert first is not None and second is not None and independent is not None
        assert first.source_ids == second.source_ids == ("source-1",)
        assert independent.job_id == "independent-source"
        assert independent.source_ids == ("source-2",)

        cancelled = await store.request_job_cancel(
            "agent-1",
            first.job_id,
            requested_at=now + timedelta(seconds=2),
        )
        assert cancelled is not None
        assert cancelled.status is JobStatus.CANCEL_REQUESTED
        first_attempt = first.current_attempt
        assert first_attempt is not None
        settled = await store.finalize_job_attempt(
            "agent-1",
            first.job_id,
            claim_token=first_attempt.claim_token,
            fencing_epoch=first_attempt.fencing_epoch,
            attempt_status=JobAttemptStatus.CANCELLED,
            completed_at=now + timedelta(seconds=3),
        )
        assert settled is not None and settled.status is JobStatus.CANCELLED
        current_second = await store.load_job("agent-1", second.job_id)
        current_independent = await store.load_job("agent-1", independent.job_id)
        assert current_second is not None
        assert current_independent is not None
        assert current_second.status is JobStatus.RUNNING
        assert current_independent.status is JobStatus.RUNNING
        released = await claim("agent-1", 4)
        assert released is not None and released.job_id == "same-source-2"

        for index in range(MAX_RUNNING_JOBS_PER_AGENT + 1):
            await store.admit_job(
                _stored_job(
                    f"agent-bound-{index}",
                    agent_id="agent-2",
                    now=now + timedelta(microseconds=index),
                    source_id=f"source-{index + 10}",
                    resource_id=f"resource-{index + 10}",
                )
            )
        claimed = []
        for index in range(MAX_RUNNING_JOBS_PER_AGENT):
            claimed.append(await claim("agent-2", index + 20))
        assert all(item is not None for item in claimed)
        assert await claim("agent-2", 99) is None
    finally:
        await store.close()


def _stored_job(
    job_id: str,
    *,
    now: datetime,
    agent_id: str = "agent-1",
    source_id: str = "source-1",
    resource_id: str = "resource-1",
    contract_digest: str | None = None,
) -> JobRun:
    resource_revision = "sha256:" + sha256(b"resource-v1").hexdigest()
    resolved_contract_digest = contract_digest or (
        "sha256:" + sha256(b"capability-v1").hexdigest()
    )
    specification = JobSpecification(
        job_kind="data_profile",
        arguments={"resource_ids": (resource_id,), "sample_rows": 10},
        resource_bindings=(
            JobResourceBinding(
                source_id=source_id,
                source_revision="source-revision-1",
                resource_id=resource_id,
                resource_revision=resource_revision,
                adapter_id="sqlite",
                sensitivity=ModelSensitivity.INTERNAL,
            ),
        ),
        execution_capability_id="jobs.data_profile.execute",
        execution_contract_digest=resolved_contract_digest,
        execution_mode=JobExecutionMode.DAITA,
        sensitivity=ModelSensitivity.INTERNAL,
        deadline_at=now + timedelta(minutes=10),
        max_wall_time_seconds=60,
    )
    return JobRun(
        job_id=job_id,
        agent_id=agent_id,
        conversation_id="conversation-1",
        origin_run_id="run-origin",
        origin_call_id="call-origin",
        specification=specification,
        specification_digest=specification.digest,
        status=JobStatus.QUEUED,
        desired_state=JobDesiredState.RUN,
        created_at=now,
        updated_at=now,
    )


async def test_store_claims_once_fences_stale_finalization_and_safely_restarts(
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    path = tmp_path / "job-state.sqlite"
    store = await SQLiteStateStore.open(path)
    first = _stored_job("job-one", now=now)
    second = _stored_job("job-two", now=now + timedelta(microseconds=1))
    third = _stored_job("job-three", now=now + timedelta(microseconds=2))
    await store.admit_job(first)
    await store.admit_job(second)
    await store.admit_job(third)

    claims = await asyncio.gather(
        store.claim_next_job(
            "agent-1",
            claim_token="claim-a",
            execution_run_id="run-" + "a" * 32,
            reserved_artifact_id="artifact-" + "a" * 32,
            claimed_at=now + timedelta(seconds=1),
            lease_seconds=60,
        ),
        store.claim_next_job(
            "agent-1",
            claim_token="claim-b",
            execution_run_id="run-" + "b" * 32,
            reserved_artifact_id="artifact-" + "b" * 32,
            claimed_at=now + timedelta(seconds=1),
            lease_seconds=60,
        ),
    )
    claimed = tuple(item for item in claims if item is not None)
    assert len(claimed) == 2
    assert {item.job_id for item in claimed} == {"job-one", "job-two"}
    assert len({item.attempts[-1].claim_token for item in claimed}) == 2
    assert (
        await store.claim_next_job(
            "agent-1",
            claim_token="claim-c",
            execution_run_id="run-" + "c" * 32,
            reserved_artifact_id="artifact-" + "c" * 32,
            claimed_at=now + timedelta(seconds=1),
            lease_seconds=60,
        )
        is None
    )

    stale = claimed[0]
    attempt = stale.current_attempt
    assert attempt is not None
    recovered = await store.recover_stale_job(
        "agent-1",
        stale.job_id,
        recovered_at=now + timedelta(seconds=2),
        restart_safe=True,
    )
    assert recovered is not None
    assert recovered.status is JobStatus.QUEUED
    assert recovered.attempts[-1].status is JobAttemptStatus.FENCED
    assert (
        await store.finalize_job_attempt(
            "agent-1",
            stale.job_id,
            claim_token=attempt.claim_token,
            fencing_epoch=attempt.fencing_epoch,
            attempt_status=JobAttemptStatus.FAILED,
            completed_at=now + timedelta(seconds=3),
            failure_code="stale_worker",
        )
        is None
    )
    await store.close()

    reopened = await SQLiteStateStore.open(path)
    try:
        persisted = await reopened.load_job("agent-1", stale.job_id)
        assert persisted is not None
        assert persisted.status is JobStatus.QUEUED
        assert persisted.fencing_epoch > attempt.fencing_epoch
    finally:
        await reopened.close()
