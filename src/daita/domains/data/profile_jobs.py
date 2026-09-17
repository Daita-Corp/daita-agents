"""Admit and execute durable data-profile jobs through the data capability domain."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from ..._json import FrozenJsonObject, canonical_json, freeze_json
from ...artifacts.models import (
    ArtifactAuthorship,
    ArtifactDraft,
    ArtifactProvenance,
    ArtifactResourceBinding,
    artifact_ref_from_mapping,
)
from ...artifacts.store import AgentHomeArtifactStore
from ...capabilities import (
    AccessMode,
    ArtifactPolicy,
    AutomationEligibility,
    AutomationScopeProposal,
    Capability,
    CapabilityDeclarations,
    CapabilityInputError,
    Executor,
    OperationalEffect,
    ToolboxId,
    ToolExecution,
    ToolLoadMode,
    ToolOutput,
    ToolPresentation,
    ToolTextTrust,
    ToolView,
    capability_contract_digest,
)
from ...capability_runtime import CapabilityFailure, SideEffectPlan
from ...catalog.models import Sensitivity
from ...distribution.owner import DistributionOwner
from ...jobs.graph.models import (
    BudgetAmount,
    BudgetLimit,
    EdgeKind,
    GraphAdmission,
    GraphAuthority,
    GraphDesiredState,
    GraphJob,
    GraphJobSpecification,
    GraphLimits,
    GraphState,
    GraphTask,
    GraphTaskSpecification,
    JobGraph,
    TaskDependency,
    TaskExecutionKind,
    TaskRole,
    TaskState,
    topology_digest,
)
from ...jobs.models import (
    MAX_JOB_DEADLINE_SECONDS,
    MAX_JOB_RESOURCE_BINDINGS,
    MAX_JOB_WALL_TIME_SECONDS,
    JobExecutionMode,
    JobResourceBinding,
    JobRun,
    JobSpecification,
)
from ...jobs.owner import JobError, JobOwner
from ...llm.models import ModelSensitivity, ToolCall
from ...loop.models import RunInput
from ...loop.session import RunSession
from ...scope import resolve_effective_source_scope
from ...storage.sqlite_records import SourcePermissionStateError
from ..learning import LearningCandidateGuard
from .capabilities import SqlReadBackend, SqlReadResult
from .sql import ResourceSchema

DATA_PROFILE_DOMAIN_OWNER_ID = "data_profile_jobs"
START_DATA_PROFILE_CAPABILITY_ID = "jobs.data_profile.start"
START_DATA_PROFILE_EXECUTOR_ID = "jobs.data_profile.start.executor"
START_DATA_PROFILE_OUTPUT_KIND = "job.start_receipt"
START_DATA_PROFILE_TOOL_NAME = "start_data_profile"
DATA_PROFILE_EXECUTION_CAPABILITY_ID = "jobs.data_profile.execute"
DATA_PROFILE_EXECUTION_EXECUTOR_ID = "jobs.data_profile.execute.executor"
DATA_PROFILE_EXECUTION_OUTPUT_KIND = "job.data_profile.result"
DATA_PROFILE_FINALIZE_CAPABILITY_ID = "jobs.data_profile.finalize"
DATA_PROFILE_FINALIZE_EXECUTOR_ID = "jobs.data_profile.finalize.executor"
DATA_PROFILE_FINALIZE_OUTPUT_KIND = "job.data_profile.final_result"

_MAX_PROFILE_SAMPLE_ROWS = 100
_MAX_PROFILE_READ_BYTES = 256 * 1024
_MAX_PROFILE_ARTIFACT_BYTES = 1 * 1024 * 1024


class DataProfileCatalog(Protocol):
    async def source_routing_facts(
        self,
        agent_id: str,
        source_ids: tuple[str, ...] = (),
    ) -> tuple[Mapping[str, object], ...]: ...

    async def source_adapter_id(self, agent_id: str, source_id: str) -> str | None: ...

    async def resource_schemas(
        self,
        agent_id: str,
        source_id: str,
    ) -> tuple[ResourceSchema, ...]: ...

    async def resource_identity(
        self,
        agent_id: str,
        resource_id: str,
    ) -> tuple[str, str, str] | None: ...

    async def readable_resource_ids(
        self,
        agent_id: str,
        source_ids: tuple[str, ...] = (),
    ) -> frozenset[str]: ...


@dataclass(frozen=True, slots=True)
class DataProfileDeclarations:
    capabilities: tuple[Capability, ...]
    executors: tuple[Executor, ...]
    tool_views: tuple[ToolView, ...]


class DataProfileAdmission:
    """Resolve exact current resource facts for one frozen profile specification."""

    def __init__(
        self,
        *,
        agent_id: str,
        catalog: DataProfileCatalog,
        owner: JobOwner,
        execution_capability: Capability,
    ) -> None:
        self._agent_id = agent_id
        self._catalog = catalog
        self._owner = owner
        self._execution_capability = execution_capability
        self._execution_contract_digest = capability_contract_digest(
            execution_capability,
            domain_owner_id=DATA_PROFILE_DOMAIN_OWNER_ID,
        )

    async def build_specification(
        self,
        arguments: Mapping[str, object],
    ) -> JobSpecification:
        resource_ids = arguments.get("resource_ids")
        sample_rows = arguments.get("sample_rows", _MAX_PROFILE_SAMPLE_ROWS)
        deadline_raw = arguments.get("_deadline_at")
        wall_time = arguments.get("_max_wall_time_seconds")
        selected_profile = arguments.get("_connected_profile_id")
        if not isinstance(resource_ids, tuple):
            raise CapabilityInputError(
                "data_profile_invalid",
                "The data profile resource selection is malformed.",
            )
        if not isinstance(sample_rows, int) or isinstance(sample_rows, bool):
            raise CapabilityInputError(
                "data_profile_invalid",
                "The data profile sample bound is malformed.",
            )
        if not isinstance(deadline_raw, str):
            raise CapabilityInputError(
                "data_profile_invalid",
                "The data profile deadline is missing.",
            )
        if not isinstance(wall_time, (int, float)) or isinstance(wall_time, bool):
            raise CapabilityInputError(
                "data_profile_invalid",
                "The data profile wall-time bound is malformed.",
            )
        try:
            deadline = datetime.fromisoformat(deadline_raw)
        except ValueError:
            raise CapabilityInputError(
                "data_profile_invalid",
                "The data profile deadline is malformed.",
            ) from None
        bindings = await self._current_bindings(resource_ids)
        sensitivity = _maximum_sensitivity(tuple(item.sensitivity for item in bindings))
        external = None
        mode = JobExecutionMode.DAITA
        if selected_profile is not None:
            if not isinstance(selected_profile, str):
                raise CapabilityInputError(
                    "external_executor_not_admitted",
                    "The connected executor selection is malformed.",
                )
            external = self._owner.resolve_external_selection(
                selected_profile,
                job_kind="data_profile",
                sensitivity=sensitivity,
            )
            mode = JobExecutionMode.CONNECTED_EXECUTOR
        return JobSpecification(
            job_kind="data_profile",
            arguments={
                "resource_ids": resource_ids,
                "sample_rows": sample_rows,
            },
            resource_bindings=bindings,
            execution_capability_id=self._execution_capability.id,
            execution_contract_digest=self._execution_contract_digest,
            execution_mode=mode,
            sensitivity=sensitivity,
            deadline_at=deadline.astimezone(UTC),
            max_wall_time_seconds=float(wall_time),
            external_executor=external,
        )

    async def build_static_graph_admission(
        self,
        *,
        run: RunInput,
        call_id: str,
        arguments: Mapping[str, object],
        finalizer_capability: Capability,
        distribution: DistributionOwner,
        clock: Callable[[], datetime],
        id_factory: Callable[[str], str],
    ) -> GraphAdmission:
        """Build the explicit integration-only static profile graph."""

        if run.agent_id != self._agent_id or run.conversation_id is None:
            raise CapabilityInputError(
                "graph_profile_origin_invalid",
                "A static profile graph requires the exact owned conversation.",
            )
        prepared = dict(arguments)
        deadline_seconds = prepared.get("deadline_seconds", 300)
        if not isinstance(deadline_seconds, int) or isinstance(deadline_seconds, bool):
            raise CapabilityInputError(
                "data_profile_invalid", "The data-profile deadline is malformed."
            )
        now = clock()
        deadline = now + timedelta(seconds=deadline_seconds)
        if deadline_seconds < 60:
            raise CapabilityInputError(
                "graph_deadline_too_short",
                "A durable graph deadline must be at least sixty seconds.",
            )
        prepared["_deadline_at"] = deadline.isoformat()
        prepared["_max_wall_time_seconds"] = min(
            float(deadline_seconds), MAX_JOB_WALL_TIME_SECONDS
        )
        profile = await self.build_specification(prepared)
        profile_sample_rows = profile.arguments.get("sample_rows")
        if not isinstance(profile_sample_rows, int):
            raise CapabilityInputError(
                "data_profile_invalid", "The frozen sample bound is malformed."
            )
        if profile.execution_mode is not JobExecutionMode.DAITA:
            raise CapabilityInputError(
                "graph_external_executor_forbidden",
                "The static graph slice admits only Daita-owned internal work.",
            )
        finalizer_digest = capability_contract_digest(
            finalizer_capability,
            domain_owner_id=DATA_PROFILE_DOMAIN_OWNER_ID,
        )
        target = distribution.resolve_conversation_inbox(
            run.conversation_id,
            sensitivity_ceiling=profile.sensitivity,
        )
        plan = distribution.resolve_plan(
            run.conversation_id,
            destination_id=target.destination_id,
            sensitivity_ceiling=profile.sensitivity,
        )
        job_id = id_factory("job")
        root_authority = GraphAuthority(
            source_ids=tuple(
                sorted({item.source_id for item in profile.resource_bindings})
            ),
            resource_ids=tuple(item.resource_id for item in profile.resource_bindings),
            capability_ids=(
                self._execution_capability.id,
                finalizer_capability.id,
            ),
            access_modes=("read",),
            operational_effects=("none",),
            sensitivity=profile.sensitivity,
            contract_bindings={
                self._execution_capability.id: self._execution_contract_digest,
                finalizer_capability.id: finalizer_digest,
                **{
                    item.resource_id: _binding_payload(item)
                    for item in profile.resource_bindings
                },
            },
        )
        root_specification = GraphJobSpecification(
            principal_id=self._agent_id,
            objective="Create one verified profile over the exact selected resources.",
            outcome_contract={
                "kind": "data_profile",
                "resource_ids": tuple(
                    item.resource_id for item in profile.resource_bindings
                ),
                "artifact_media_type": "application/json",
                "producer_capability_id": finalizer_capability.id,
                "profile_specification_digest": profile.digest,
            },
            authority=root_authority,
            distribution_plan_digest=plan.plan_digest,
            budgets=(
                BudgetLimit(
                    "work_units",
                    (len(profile.resource_bindings) + 1) * 3,
                    control_reserved=3,
                ),
            ),
            deadline_at=deadline,
            limits=GraphLimits(
                max_tasks=len(profile.resource_bindings) + 1,
                max_edges=len(profile.resource_bindings),
                max_parallelism=min(4, len(profile.resource_bindings)),
            ),
            retry_policy={"max_attempts": 3, "backoff_seconds": (1, 5)},
            cancellation_policy={"preserve_evidence": True},
            finalizer_task_template={
                "kind": "data_profile_internal_finalizer",
                "capability_id": finalizer_capability.id,
                "contract_digest": finalizer_digest,
            },
        )
        work_tasks: list[GraphTask] = []
        for binding in profile.resource_bindings:
            task_id = id_factory("task")
            authority = GraphAuthority(
                source_ids=(binding.source_id,),
                resource_ids=(binding.resource_id,),
                capability_ids=(self._execution_capability.id,),
                access_modes=("read",),
                operational_effects=("none",),
                sensitivity=binding.sensitivity,
                contract_bindings={
                    self._execution_capability.id: self._execution_contract_digest,
                    binding.resource_id: _binding_payload(binding),
                },
            )
            execution_arguments = {
                "job_id": job_id,
                "specification_digest": profile.digest,
                "resource_ids": (binding.resource_id,),
                "sample_rows": profile_sample_rows,
                "resource_bindings": (_binding_payload(binding),),
            }
            task_specification = GraphTaskSpecification(
                title=f"Profile {binding.resource_id}",
                description="Profile one exact frozen resource through the internal capability.",
                expected_result_contract={
                    "kind": "data_profile_work",
                    "output_kind": DATA_PROFILE_EXECUTION_OUTPUT_KIND,
                    "capability_id": self._execution_capability.id,
                    "contract_digest": self._execution_contract_digest,
                    "arguments": execution_arguments,
                    "source_permit_key": f"source:{binding.source_id}",
                    "attempt_budgets": {"work_units": 1},
                },
                authority=authority,
                budgets=(BudgetAmount("work_units", 3),),
                max_steps=1,
                max_wall_time_seconds=min(
                    300, max(1, int(profile.max_wall_time_seconds))
                ),
                created_by="job_owner",
            )
            work_tasks.append(
                GraphTask(
                    agent_id=self._agent_id,
                    job_id=job_id,
                    task_id=task_id,
                    state=TaskState.READY,
                    role=TaskRole.INTERNAL,
                    execution_kind=TaskExecutionKind.INTERNAL_CAPABILITY,
                    priority=100,
                    not_before=None,
                    current_attempt_id=None,
                    task_revision=1,
                    specification=task_specification,
                    task_spec_digest=task_specification.digest,
                    task_scope_digest=authority.digest,
                    attempt_count=0,
                    failure_streak=0,
                    fencing_epoch=0,
                    created_at=now,
                    updated_at=now,
                )
            )
        finalizer_id = id_factory("task")
        finalizer_authority = GraphAuthority(
            capability_ids=(finalizer_capability.id,),
            access_modes=("read",),
            operational_effects=("none",),
            sensitivity=profile.sensitivity,
            contract_bindings={finalizer_capability.id: finalizer_digest},
        )
        finalizer_specification = GraphTaskSpecification(
            title="Finalize data profile",
            description="Authenticate required task results and publish the verified profile.",
            expected_result_contract={
                "kind": "data_profile_finalizer",
                "output_kind": DATA_PROFILE_FINALIZE_OUTPUT_KIND,
                "capability_id": finalizer_capability.id,
                "contract_digest": finalizer_digest,
                "resource_ids": tuple(
                    item.resource_id for item in profile.resource_bindings
                ),
                "sample_rows": profile_sample_rows,
                "attempt_budgets": {"work_units": 1},
            },
            authority=finalizer_authority,
            budgets=(BudgetAmount("work_units", 3),),
            max_steps=1,
            max_wall_time_seconds=min(300, max(1, int(profile.max_wall_time_seconds))),
            created_by="job_owner",
        )
        finalizer = GraphTask(
            agent_id=self._agent_id,
            job_id=job_id,
            task_id=finalizer_id,
            state=TaskState.PENDING,
            role=TaskRole.FINALIZER,
            execution_kind=TaskExecutionKind.INTERNAL_CAPABILITY,
            priority=1_000,
            not_before=None,
            current_attempt_id=None,
            task_revision=1,
            specification=finalizer_specification,
            task_spec_digest=finalizer_specification.digest,
            task_scope_digest=finalizer_authority.digest,
            attempt_count=0,
            failure_streak=0,
            fencing_epoch=0,
            created_at=now,
            updated_at=now,
        )
        tasks = (*work_tasks, finalizer)
        dependencies = tuple(
            TaskDependency(
                agent_id=self._agent_id,
                job_id=job_id,
                upstream_task_id=task.task_id,
                downstream_task_id=finalizer_id,
                edge_kind=EdgeKind.REQUIRES_ACCEPTED_SUCCESS,
                created_at=now,
                creator_key="job_owner",
            )
            for task in work_tasks
        )
        job = GraphJob(
            agent_id=self._agent_id,
            job_id=job_id,
            conversation_id=run.conversation_id,
            origin_run_id=run.id,
            origin_call_id=call_id,
            state=GraphState.QUEUED,
            desired_state=GraphDesiredState.RUN,
            created_at=now,
            updated_at=now,
            deadline_at=deadline,
            specification=root_specification,
            specification_digest=root_specification.digest,
            finalizer_task_id=finalizer_id,
        )
        graph = JobGraph(
            agent_id=self._agent_id,
            job_id=job_id,
            revision=0,
            task_count=len(tasks),
            edge_count=len(dependencies),
            mutation_count=0,
            active_attempt_count=0,
            next_ready_at=now,
            finalization_attempt_id=None,
            finalization_started_revision=None,
            created_at=now,
            updated_at=now,
            topology_digest=topology_digest(tasks, dependencies),
        )
        return GraphAdmission(
            job=job,
            graph=graph,
            tasks=tasks,
            dependencies=dependencies,
        )

    async def validate_internal(
        self,
        arguments: Mapping[str, object],
    ) -> None:
        resource_ids = arguments.get("resource_ids")
        raw_bindings = arguments.get("resource_bindings")
        if not isinstance(resource_ids, tuple) or not isinstance(raw_bindings, tuple):
            raise CapabilityInputError(
                "job_specification_invalid",
                "The frozen job resource specification is malformed.",
            )
        expected = await self._current_bindings(resource_ids)
        material = tuple(_binding_from_mapping(item) for item in raw_bindings)
        if material != expected:
            raise CapabilityInputError(
                "job_resource_scope_stale",
                "The current resource identity or admission no longer matches the frozen job.",
            )

    async def revalidate_external(self, job: JobRun) -> None:
        """Recheck the concrete data-profile authority immediately before I/O."""

        if job.specification.execution_mode is not JobExecutionMode.CONNECTED_EXECUTOR:
            raise CapabilityInputError(
                "job_contract_revalidation_failed",
                "The external data-profile execution mode is no longer current.",
            )
        await self.revalidate_job(job)

    async def revalidate_job(self, job: JobRun) -> None:
        """Recheck the exact data-profile contract and current read scope."""

        specification = job.specification
        if (
            specification.job_kind != "data_profile"
            or specification.execution_capability_id != self._execution_capability.id
            or specification.execution_contract_digest
            != self._execution_contract_digest
        ):
            raise CapabilityInputError(
                "job_contract_revalidation_failed",
                "The external data-profile execution contract is no longer current.",
            )
        await self.validate_internal(
            {
                "resource_ids": specification.arguments.get("resource_ids"),
                "resource_bindings": tuple(
                    {
                        "source_id": item.source_id,
                        "source_revision": item.source_revision,
                        "resource_id": item.resource_id,
                        "resource_revision": item.resource_revision,
                        "adapter_id": item.adapter_id,
                        "sensitivity": item.sensitivity.value,
                    }
                    for item in specification.resource_bindings
                ),
            }
        )
        current_sensitivity = _maximum_sensitivity(
            tuple(item.sensitivity for item in specification.resource_bindings)
        )
        if current_sensitivity is not specification.sensitivity:
            raise CapabilityInputError(
                "job_sensitivity_stale",
                "The external data-profile sensitivity no longer matches its scope.",
            )

    async def _current_bindings(
        self,
        resource_ids: tuple[object, ...],
    ) -> tuple[JobResourceBinding, ...]:
        if (
            not 1 <= len(resource_ids) <= MAX_JOB_RESOURCE_BINDINGS
            or len(set(resource_ids)) != len(resource_ids)
            or any(not isinstance(item, str) or not item for item in resource_ids)
        ):
            raise CapabilityInputError(
                "data_profile_resource_limit",
                "A data profile requires distinct bounded current resource IDs.",
            )
        identities: list[tuple[str, str, str, str]] = []
        for resource_id in resource_ids:
            assert isinstance(resource_id, str)
            identity = await self._catalog.resource_identity(
                self._agent_id,
                resource_id,
            )
            if identity is None:
                raise CapabilityInputError(
                    "data_profile_resource_unavailable",
                    "One requested data-profile resource is not current.",
                )
            identities.append((resource_id, *identity))
        source_ids = tuple(sorted({item[1] for item in identities}))
        try:
            readable = await self._catalog.readable_resource_ids(
                self._agent_id,
                source_ids,
            )
        except SourcePermissionStateError as error:
            raise CapabilityInputError(
                "source_permission_state_invalid",
                "Stored source permission state is missing or invalid.",
            ) from error
        if any(item[0] not in readable for item in identities):
            raise CapabilityInputError(
                "resource_read_not_allowed",
                "One requested data-profile resource is not available for reading.",
            )
        schemas: dict[str, ResourceSchema] = {}
        adapters: dict[str, str] = {}
        for source_id in source_ids:
            adapter = await self._catalog.source_adapter_id(
                self._agent_id,
                source_id,
            )
            if adapter not in {"sqlite", "postgresql"}:
                raise CapabilityInputError(
                    "data_profile_adapter_unsupported",
                    "The first data-profile job supports SQLite and PostgreSQL resources.",
                )
            adapters[source_id] = adapter
            for current_schema in await self._catalog.resource_schemas(
                self._agent_id,
                source_id,
            ):
                schemas[current_schema.resource_id] = current_schema
        bindings = []
        for resource_id, source_id, resource_kind, resource_revision in identities:
            schema = schemas.get(resource_id)
            if (
                schema is None
                or schema.source_revision is None
                or schema.resource_kind != resource_kind
                or schema.revision != resource_revision
            ):
                raise CapabilityInputError(
                    "data_profile_resource_stale",
                    "One requested data-profile resource changed during admission.",
                )
            bindings.append(
                JobResourceBinding(
                    source_id=source_id,
                    source_revision=schema.source_revision,
                    resource_id=resource_id,
                    resource_revision=resource_revision,
                    adapter_id=adapters[source_id],
                    sensitivity=_model_sensitivity(schema.sensitivity_class),
                )
            )
        return tuple(
            sorted(bindings, key=lambda item: (item.source_id, item.resource_id))
        )


class StartDataProfileExecutor:
    executor_id = START_DATA_PROFILE_EXECUTOR_ID

    def __init__(
        self,
        *,
        owner: JobOwner,
        admission: DataProfileAdmission,
        clock,
    ) -> None:
        self._owner = owner
        self._admission = admission
        self._clock = clock

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        specification = await self._admission.build_specification(request.arguments)
        return FrozenJsonObject.from_mapping(
            {
                "specification_digest": specification.digest,
                "execution_mode": specification.execution_mode.value,
                "execution_capability_id": specification.execution_capability_id,
                "execution_contract_digest": specification.execution_contract_digest,
                "resource_ids": tuple(
                    item.resource_id for item in specification.resource_bindings
                ),
            }
        )

    async def execute(self, request: ToolExecution) -> ToolOutput:
        specification = await self._admission.build_specification(request.arguments)
        run = RunInput(
            id=request.run_id,
            agent_id=self._owner.agent_id,
            message="Start the exact admitted data profile job.",
            created_at=self._clock(),
            conversation_id=request.conversation_id,
            source_scope_ids=tuple(
                sorted({item.source_id for item in specification.resource_bindings})
            ),
            resolved_source_scope=request.source_scope,
        )
        job = await self._owner.admit(
            run=run,
            call_id=request.call_id,
            specification=specification,
        )
        return ToolOutput(
            kind=START_DATA_PROFILE_OUTPUT_KIND,
            data={
                "job_id": job.job_id,
                "job_kind": job.specification.job_kind,
                "status": job.status.value,
                "execution_mode": job.specification.execution_mode.value,
                "specification_digest": job.specification_digest,
                "execution_capability_id": (job.specification.execution_capability_id),
                "execution_contract_digest": (
                    job.specification.execution_contract_digest
                ),
            },
            sensitivity=job.specification.sensitivity,
            sensitivity_provenance={
                "authority": "frozen_job_resource_scope",
                "job_id": job.job_id,
                "resource_ids": tuple(
                    item.resource_id for item in job.specification.resource_bindings
                ),
            },
        )


class StartStaticDataProfileGraphExecutor:
    """Integration-only starter for the draft static graph vertical slice."""

    executor_id = START_DATA_PROFILE_EXECUTOR_ID

    def __init__(
        self,
        *,
        owner: JobOwner,
        admission: DataProfileAdmission,
        finalizer_capability: Capability,
        distribution: DistributionOwner,
        clock: Callable[[], datetime],
        id_factory: Callable[[str], str],
    ) -> None:
        self._owner = owner
        self._admission = admission
        self._finalizer_capability = finalizer_capability
        self._distribution = distribution
        self._clock = clock
        self._id_factory = id_factory

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        prepared = dict(request.arguments)
        deadline_seconds = prepared.get("deadline_seconds", 300)
        if not isinstance(deadline_seconds, int) or isinstance(deadline_seconds, bool):
            raise CapabilityInputError(
                "data_profile_invalid", "The data-profile deadline is malformed."
            )
        prepared["_deadline_at"] = (
            self._clock() + timedelta(seconds=deadline_seconds)
        ).isoformat()
        prepared["_max_wall_time_seconds"] = min(
            float(deadline_seconds), MAX_JOB_WALL_TIME_SECONDS
        )
        specification = await self._admission.build_specification(prepared)
        return FrozenJsonObject.from_mapping(
            {
                "profile_specification_digest": specification.digest,
                "resource_ids": tuple(
                    item.resource_id for item in specification.resource_bindings
                ),
                "graph_kind": "static_data_profile",
            }
        )

    async def execute(self, request: ToolExecution) -> ToolOutput:
        if request.conversation_id is None:
            raise CapabilityInputError(
                "graph_profile_origin_invalid",
                "A static profile graph requires the exact owned conversation.",
            )
        admission = await self._admission.build_static_graph_admission(
            run=RunInput(
                id=request.run_id,
                agent_id=self._owner.agent_id,
                message="Start the exact admitted static data-profile graph.",
                created_at=self._clock(),
                conversation_id=request.conversation_id,
                source_scope_ids=(
                    ()
                    if request.source_scope is None
                    else tuple(sorted(request.source_scope.source_ids))
                ),
                resolved_source_scope=request.source_scope,
            ),
            call_id=request.call_id,
            arguments=request.arguments,
            finalizer_capability=self._finalizer_capability,
            distribution=self._distribution,
            clock=self._clock,
            id_factory=self._id_factory,
        )
        job = await self._owner.admit_static_graph(admission)
        work_capability_id = DATA_PROFILE_EXECUTION_CAPABILITY_ID
        work_contract = job.specification.authority.contract_bindings.get(
            work_capability_id
        )
        assert isinstance(work_contract, str)
        return ToolOutput(
            kind=START_DATA_PROFILE_OUTPUT_KIND,
            data={
                "job_id": job.job_id,
                "job_kind": "data_profile",
                "status": job.state.value,
                "execution_mode": JobExecutionMode.DAITA.value,
                "specification_digest": job.specification_digest,
                "execution_capability_id": work_capability_id,
                "execution_contract_digest": work_contract,
            },
            sensitivity=job.specification.authority.sensitivity,
            sensitivity_provenance={
                "authority": "frozen_static_graph_resource_scope",
                "job_id": job.job_id,
                "resource_ids": job.specification.authority.resource_ids,
            },
        )


class DataProfileExecutor:
    executor_id = DATA_PROFILE_EXECUTION_EXECUTOR_ID

    def __init__(
        self,
        *,
        agent_id: str,
        catalog: DataProfileCatalog,
        sqlite_backend: SqlReadBackend,
        postgresql_backend: SqlReadBackend,
    ) -> None:
        self._agent_id = agent_id
        self._catalog = catalog
        self._sqlite = sqlite_backend
        self._postgresql = postgresql_backend

    async def execute(self, request: ToolExecution) -> ToolOutput:
        job_id = request.arguments["job_id"]
        resource_ids = request.arguments["resource_ids"]
        sample_rows = request.arguments["sample_rows"]
        raw_bindings = request.arguments["resource_bindings"]
        assert isinstance(job_id, str)
        assert isinstance(resource_ids, tuple)
        assert isinstance(sample_rows, int)
        assert isinstance(raw_bindings, tuple)
        bindings = tuple(_binding_from_mapping(item) for item in raw_bindings)
        schemas: dict[str, ResourceSchema] = {}
        for source_id in sorted({item.source_id for item in bindings}):
            for current_schema in await self._catalog.resource_schemas(
                self._agent_id,
                source_id,
            ):
                schemas[current_schema.resource_id] = current_schema
        profiles: list[dict[str, object]] = []
        artifact_bindings: list[ArtifactResourceBinding] = []
        sampled_rows = 0
        truncated_resources = 0
        for binding in bindings:
            schema = schemas.get(binding.resource_id)
            if schema is None:
                raise CapabilityInputError(
                    "job_resource_scope_stale",
                    "A frozen data-profile resource is no longer current.",
                )
            result = await self._read(binding, schema, sample_rows)
            rows = tuple(item.to_dict() for item in result.projection.rows)
            sampled_rows += len(rows)
            if result.projection.truncated:
                truncated_resources += 1
            profiles.append(
                {
                    "source_id": binding.source_id,
                    "source_revision": binding.source_revision,
                    "resource_id": binding.resource_id,
                    "resource_revision": binding.resource_revision,
                    "name": schema.name,
                    "columns": schema.columns,
                    "declared_types": tuple(
                        {"column": name, "type": declared}
                        for name, declared in schema.column_declared_types
                    ),
                    "sampled_rows": len(rows),
                    "source_rows_seen": result.projection.total_rows,
                    "truncated": result.projection.truncated,
                    "truncation_reasons": result.projection.truncation_reasons,
                    "column_profiles": _column_profiles(schema.columns, rows),
                }
            )
            artifact_bindings.append(
                ArtifactResourceBinding(
                    source_id=binding.source_id,
                    source_revision=binding.source_revision,
                    resource_id=binding.resource_id,
                    resource_revision=binding.resource_revision,
                )
            )
        document = {
            "kind": "data_profile",
            "job_id": job_id,
            "resource_count": len(bindings),
            "sample_row_limit": sample_rows,
            "resources": profiles,
            "trust_classification": "untrusted_external_data",
        }
        content = canonical_json(document).encode("utf-8")
        sensitivity = _maximum_sensitivity(tuple(item.sensitivity for item in bindings))
        artifact_sensitivity = Sensitivity(sensitivity.value)
        return ToolOutput(
            kind=DATA_PROFILE_EXECUTION_OUTPUT_KIND,
            data={
                "job_id": job_id,
                "profiled_resources": len(bindings),
                "sampled_rows": sampled_rows,
                "truncated_resources": truncated_resources,
            },
            artifact=ArtifactDraft(
                content=content,
                suggested_filename=f"data-profile-{job_id}.json",
                media_type="application/json",
                sensitivity=artifact_sensitivity,
                provenance=ArtifactProvenance(
                    authorship=ArtifactAuthorship.EXACT_SOURCE_DATA,
                    resource_bindings=tuple(
                        sorted(
                            artifact_bindings,
                            key=lambda item: (item.source_id, item.resource_id),
                        )
                    ),
                ),
            ),
            sensitivity=sensitivity,
            sensitivity_provenance={
                "authority": "current_validated_job_resource_scope",
                "job_id": job_id,
                "resource_ids": resource_ids,
            },
        )

    async def _read(
        self,
        binding: JobResourceBinding,
        schema: ResourceSchema,
        sample_rows: int,
    ) -> SqlReadResult:
        backend = self._sqlite if binding.adapter_id == "sqlite" else self._postgresql
        sql = f"SELECT * FROM {_quoted_relation(schema.name)} LIMIT {sample_rows}"
        result = await backend.execute_read(
            agent_id=self._agent_id,
            source_id=binding.source_id,
            sql=sql,
            parameters=(),
            max_rows=sample_rows,
            max_bytes=_MAX_PROFILE_READ_BYTES,
        )
        if result.resource_ids != (binding.resource_id,):
            raise CapabilityInputError(
                "job_resource_scope_stale",
                "The data-profile query resolved outside its exact frozen resource.",
            )
        return result


class DataProfileFinalizerExecutor:
    """Authenticate accepted work artifacts and build one final profile artifact."""

    executor_id = DATA_PROFILE_FINALIZE_EXECUTOR_ID

    def __init__(self, artifacts: AgentHomeArtifactStore) -> None:
        self._artifacts = artifacts

    async def execute(self, request: ToolExecution) -> ToolOutput:
        job_id = request.arguments["job_id"]
        expected_resource_ids = request.arguments["resource_ids"]
        work_results = request.arguments["work_results"]
        sample_rows = request.arguments["sample_rows"]
        assert isinstance(job_id, str)
        assert isinstance(expected_resource_ids, tuple)
        assert isinstance(work_results, tuple)
        assert isinstance(sample_rows, int)
        resources: list[Mapping[str, object]] = []
        resource_bindings: dict[tuple[str, str], ArtifactResourceBinding] = {}
        sensitivities: list[ModelSensitivity] = []
        sampled_rows = 0
        truncated_resources = 0
        observed_resource_ids: set[str] = set()
        for work_result in work_results:
            if not isinstance(work_result, Mapping):
                raise CapabilityInputError(
                    "graph_profile_result_invalid",
                    "A required profile result manifest is malformed.",
                )
            raw_ref = work_result.get("artifact_ref")
            if not isinstance(raw_ref, Mapping):
                raise CapabilityInputError(
                    "graph_profile_artifact_invalid",
                    "A required profile artifact reference is malformed.",
                )
            ref = artifact_ref_from_mapping(raw_ref)
            if ref.capability_id != DATA_PROFILE_EXECUTION_CAPABILITY_ID:
                raise CapabilityInputError(
                    "graph_profile_artifact_invalid",
                    "A required artifact was produced by another capability.",
                )
            payload = await self._artifacts.read_ref(ref)
            try:
                document = json.loads(payload.content.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CapabilityInputError(
                    "graph_profile_artifact_invalid",
                    "A required profile artifact is not valid canonical JSON.",
                ) from error
            if (
                not isinstance(document, dict)
                or document.get("kind") != "data_profile"
                or document.get("job_id") != job_id
                or document.get("resource_count") != 1
                or not isinstance(document.get("resources"), list)
                or len(document["resources"]) != 1
            ):
                raise CapabilityInputError(
                    "graph_profile_artifact_invalid",
                    "A required profile artifact differs from its task contract.",
                )
            resource = document["resources"][0]
            if not isinstance(resource, dict):
                raise CapabilityInputError(
                    "graph_profile_artifact_invalid",
                    "A required profile resource payload is malformed.",
                )
            resource_id = resource.get("resource_id")
            if not isinstance(resource_id, str) or resource_id in observed_resource_ids:
                raise CapabilityInputError(
                    "graph_profile_artifact_invalid",
                    "A required profile artifact has invalid resource identity.",
                )
            observed_resource_ids.add(resource_id)
            resources.append(resource)
            sampled = resource.get("sampled_rows")
            if not isinstance(sampled, int) or isinstance(sampled, bool):
                raise CapabilityInputError(
                    "graph_profile_artifact_invalid",
                    "A required profile artifact has invalid row evidence.",
                )
            sampled_rows += sampled
            if resource.get("truncated") is True:
                truncated_resources += 1
            sensitivities.append(ModelSensitivity(ref.sensitivity.value))
            for binding in ref.provenance.resource_bindings:
                resource_bindings[(binding.source_id, binding.resource_id)] = binding
        if tuple(sorted(observed_resource_ids)) != tuple(
            sorted(str(item) for item in expected_resource_ids)
        ):
            raise CapabilityInputError(
                "graph_profile_result_incomplete",
                "The accepted task results do not cover the exact graph resources.",
            )
        ordered_resources = sorted(
            resources,
            key=lambda item: (str(item.get("source_id")), str(item.get("resource_id"))),
        )
        document = {
            "kind": "data_profile",
            "job_id": job_id,
            "resource_count": len(ordered_resources),
            "sample_row_limit": sample_rows,
            "resources": ordered_resources,
            "trust_classification": "untrusted_external_data",
        }
        content = canonical_json(document).encode("utf-8")
        sensitivity = _maximum_sensitivity(tuple(sensitivities))
        return ToolOutput(
            kind=DATA_PROFILE_FINALIZE_OUTPUT_KIND,
            data={
                "job_id": job_id,
                "profiled_resources": len(ordered_resources),
                "sampled_rows": sampled_rows,
                "truncated_resources": truncated_resources,
            },
            artifact=ArtifactDraft(
                content=content,
                suggested_filename=f"data-profile-{job_id}.json",
                media_type="application/json",
                sensitivity=Sensitivity(sensitivity.value),
                provenance=ArtifactProvenance(
                    authorship=ArtifactAuthorship.EXACT_SOURCE_DATA,
                    resource_bindings=tuple(
                        resource_bindings[key] for key in sorted(resource_bindings)
                    ),
                ),
            ),
            sensitivity=sensitivity,
            sensitivity_provenance={
                "authority": "authenticated_graph_task_results",
                "job_id": job_id,
                "resource_ids": tuple(sorted(observed_resource_ids)),
            },
        )


class DataProfileCapabilityDomain:
    domain_owner_id = DATA_PROFILE_DOMAIN_OWNER_ID

    def __init__(
        self,
        declarations: CapabilityDeclarations,
        *,
        catalog: DataProfileCatalog,
        admission: DataProfileAdmission,
        learning: LearningCandidateGuard,
    ) -> None:
        if declarations.domain_owner_id != self.domain_owner_id:
            raise ValueError("data-profile declarations have the wrong owner")
        capability_ids = {item.id for item in declarations.capabilities}
        if capability_ids not in (
            {
                START_DATA_PROFILE_CAPABILITY_ID,
                DATA_PROFILE_EXECUTION_CAPABILITY_ID,
            },
            {
                START_DATA_PROFILE_CAPABILITY_ID,
                DATA_PROFILE_EXECUTION_CAPABILITY_ID,
                DATA_PROFILE_FINALIZE_CAPABILITY_ID,
            },
        ):
            raise ValueError("data-profile domain requires its exact capabilities")
        self._declarations = declarations
        self._catalog = catalog
        self._admission = admission
        self._learning = learning
        self._views = tuple(declarations.tool_views)
        self._capabilities = {item.id: item for item in declarations.capabilities}

    @property
    def declarations(self) -> CapabilityDeclarations:
        return self._declarations

    async def project(
        self,
        run: RunInput,
        session: RunSession | None = None,
    ) -> tuple[str, ...]:
        files_only = session is not None and session.options.files_only
        if files_only:
            return ()
        scope = await resolve_effective_source_scope(
            run, self._catalog, files_only=files_only
        )
        facts = (
            ()
            if not scope.resource_ids
            else await self._catalog.source_routing_facts(
                run.agent_id, tuple(sorted(scope.source_ids))
            )
        )
        if not facts:
            return ()
        return tuple(
            view.name
            for view in self._views
            if self._learning.allows(session, view.name, effectful=True)
        )

    project_session = project

    def normalize_arguments(
        self,
        capability: Capability,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        return arguments

    async def prepare_call(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        arguments: FrozenJsonObject,
        *,
        request_sensitivity: ModelSensitivity,
        session: RunSession | None = None,
    ) -> FrozenJsonObject:
        del request_sensitivity
        if capability.id == START_DATA_PROFILE_CAPABILITY_ID:
            self._learning.validate_effect(session, call)
            deadline_seconds = arguments.get("deadline_seconds", 300)
            if not isinstance(deadline_seconds, int):
                raise CapabilityInputError(
                    "data_profile_invalid",
                    "The data-profile deadline is malformed.",
                )
            prepared = dict(arguments)
            prepared["_deadline_at"] = (
                run.created_at + timedelta(seconds=deadline_seconds)
            ).isoformat()
            prepared["_max_wall_time_seconds"] = min(
                float(deadline_seconds),
                MAX_JOB_WALL_TIME_SECONDS,
            )
            selected = (
                None
                if session is None
                else session.options.selected_executor_profile_id
            )
            if selected is not None:
                prepared["_connected_profile_id"] = selected
            specification = await self._admission.build_specification(prepared)
            scope = await resolve_effective_source_scope(
                run,
                self._catalog,
                files_only=session is not None and session.options.files_only,
            )
            if any(
                item.source_id not in scope.source_ids
                or item.resource_id not in scope.resource_ids
                for item in specification.resource_bindings
            ):
                raise CapabilityInputError(
                    "source_scope_violation",
                    "This run can only profile resources in its effective scope.",
                )
            return FrozenJsonObject.from_mapping(prepared)
        if capability.id == DATA_PROFILE_FINALIZE_CAPABILITY_ID:
            return arguments
        await self._admission.validate_internal(arguments)
        return arguments

    prepare_session_call = prepare_call

    async def prepare_automation_grant(
        self,
        capability: Capability,
        constraints: FrozenJsonObject,
        max_calls_per_occurrence: int,
        proposal: AutomationScopeProposal,
    ) -> FrozenJsonObject:
        raise CapabilityInputError(
            "automation_grant_unsupported",
            "This domain does not admit unattended external effects.",
        )

    async def side_effect_plan(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        execution: ToolExecution,
        fingerprint: FrozenJsonObject,
    ) -> SideEffectPlan:
        if capability.id != START_DATA_PROFILE_CAPABILITY_ID:
            raise ValueError(
                "internal data-profile execution has no operational effect"
            )
        return SideEffectPlan(
            approval_required=False,
            recheck_after_approval=True,
        )

    async def finalize_output(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        arguments: FrozenJsonObject,
        output: ToolOutput,
        *,
        request_sensitivity: ModelSensitivity,
        session: RunSession | None = None,
    ) -> ToolOutput:
        del arguments, request_sensitivity
        if capability.id == START_DATA_PROFILE_CAPABILITY_ID:
            self._learning.mark_effect_succeeded(session, call.id)
        return output

    finalize_session_output = finalize_output

    def normalize_error(
        self,
        call: ToolCall,
        error: BaseException,
    ) -> CapabilityFailure | None:
        del call
        if isinstance(error, JobError):
            return CapabilityFailure(error.code, str(error))
        return None


def data_profile_declarations(
    *,
    agent_id: str,
    catalog: DataProfileCatalog,
    owner: JobOwner,
    sqlite_backend: SqlReadBackend,
    postgresql_backend: SqlReadBackend,
    clock,
) -> tuple[DataProfileDeclarations, DataProfileAdmission]:
    internal = Capability(
        id=DATA_PROFILE_EXECUTION_CAPABILITY_ID,
        description="Execute one exact frozen read-only data profile internally.",
        input_schema=_internal_input_schema(),
        output_kind=DATA_PROFILE_EXECUTION_OUTPUT_KIND,
        output_schema=_internal_output_schema(),
        executor_id=DATA_PROFILE_EXECUTION_EXECUTOR_ID,
        access_mode=AccessMode.READ,
        operational_effect=OperationalEffect.NONE,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
        artifact_policy=ArtifactPolicy(
            allowed_media_types=frozenset({"application/json"}),
            allowed_authorships=frozenset({ArtifactAuthorship.EXACT_SOURCE_DATA}),
            allowed_extensions=(("application/json", (".json",)),),
            artifact_required=True,
            max_artifact_count=1,
            max_bytes_per_artifact=_MAX_PROFILE_ARTIFACT_BYTES,
            max_total_bytes_per_call=_MAX_PROFILE_ARTIFACT_BYTES,
        ),
    )
    admission = DataProfileAdmission(
        agent_id=agent_id,
        catalog=catalog,
        owner=owner,
        execution_capability=internal,
    )
    start = Capability(
        id=START_DATA_PROFILE_CAPABILITY_ID,
        description=(
            "Start one bounded durable read-only data profile over exact current resources. "
            "The job is inspectable and cancellable after this run completes."
        ),
        input_schema=_start_input_schema(),
        output_kind=START_DATA_PROFILE_OUTPUT_KIND,
        output_schema=_start_output_schema(),
        executor_id=START_DATA_PROFILE_EXECUTOR_ID,
        access_mode=AccessMode.READ,
        operational_effect=OperationalEffect.START_JOB,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
    )
    executors: tuple[Executor, ...] = (
        StartDataProfileExecutor(owner=owner, admission=admission, clock=clock),
        DataProfileExecutor(
            agent_id=agent_id,
            catalog=catalog,
            sqlite_backend=sqlite_backend,
            postgresql_backend=postgresql_backend,
        ),
    )
    return (
        DataProfileDeclarations(
            capabilities=(start, internal),
            executors=executors,
            tool_views=(
                ToolView(
                    name=START_DATA_PROFILE_TOOL_NAME,
                    capability_id=start.id,
                    description=start.description,
                    presentation=ToolPresentation(
                        toolbox_id=ToolboxId.JOBS,
                        load_mode=ToolLoadMode.ON_DEMAND,
                        text_trust=ToolTextTrust.CODE,
                        summary="Start a bounded durable profile over exact data resources.",
                        when_to_use="Use when profiling should continue after this reasoning run.",
                        keywords=("data", "profile", "job", "durable"),
                    ),
                ),
            ),
        ),
        admission,
    )


def data_profile_graph_integration_declarations(
    *,
    agent_id: str,
    catalog: DataProfileCatalog,
    owner: JobOwner,
    sqlite_backend: SqlReadBackend,
    postgresql_backend: SqlReadBackend,
    artifacts: AgentHomeArtifactStore,
    distribution: DistributionOwner,
    clock,
    id_factory: Callable[[str], str],
) -> tuple[DataProfileDeclarations, DataProfileAdmission]:
    """Extend the profile domain only for explicit draft-graph integration tests."""

    base, admission = data_profile_declarations(
        agent_id=agent_id,
        catalog=catalog,
        owner=owner,
        sqlite_backend=sqlite_backend,
        postgresql_backend=postgresql_backend,
        clock=clock,
    )
    finalizer = Capability(
        id=DATA_PROFILE_FINALIZE_CAPABILITY_ID,
        description="Finalize authenticated static data-profile task results.",
        input_schema=_finalizer_input_schema(),
        output_kind=DATA_PROFILE_FINALIZE_OUTPUT_KIND,
        output_schema=_finalizer_output_schema(),
        executor_id=DATA_PROFILE_FINALIZE_EXECUTOR_ID,
        access_mode=AccessMode.READ,
        operational_effect=OperationalEffect.NONE,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
        artifact_policy=ArtifactPolicy(
            allowed_media_types=frozenset({"application/json"}),
            allowed_authorships=frozenset({ArtifactAuthorship.EXACT_SOURCE_DATA}),
            allowed_extensions=(("application/json", (".json",)),),
            artifact_required=True,
            max_artifact_count=1,
            max_bytes_per_artifact=_MAX_PROFILE_ARTIFACT_BYTES,
            max_total_bytes_per_call=_MAX_PROFILE_ARTIFACT_BYTES,
        ),
    )
    return (
        DataProfileDeclarations(
            capabilities=(*base.capabilities, finalizer),
            executors=(
                StartStaticDataProfileGraphExecutor(
                    owner=owner,
                    admission=admission,
                    finalizer_capability=finalizer,
                    distribution=distribution,
                    clock=clock,
                    id_factory=id_factory,
                ),
                base.executors[1],
                DataProfileFinalizerExecutor(artifacts),
            ),
            tool_views=base.tool_views,
        ),
        admission,
    )


def _start_input_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "resource_ids": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 512},
                "minItems": 1,
                "maxItems": MAX_JOB_RESOURCE_BINDINGS,
                "uniqueItems": True,
            },
            "sample_rows": {
                "type": "integer",
                "minimum": 1,
                "maximum": _MAX_PROFILE_SAMPLE_ROWS,
                "default": _MAX_PROFILE_SAMPLE_ROWS,
            },
            "deadline_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": int(MAX_JOB_DEADLINE_SECONDS),
                "default": 300,
            },
        },
        "required": ["resource_ids"],
        "additionalProperties": False,
    }


def _binding_schema() -> dict[str, object]:
    properties = {
        "source_id": {"type": "string"},
        "source_revision": {"type": "string"},
        "resource_id": {"type": "string"},
        "resource_revision": {"type": "string"},
        "adapter_id": {"type": "string"},
        "sensitivity": {"type": "string"},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _internal_input_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "job_id": {"type": "string"},
            "specification_digest": {"type": "string"},
            "resource_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": MAX_JOB_RESOURCE_BINDINGS,
                "uniqueItems": True,
            },
            "sample_rows": {
                "type": "integer",
                "minimum": 1,
                "maximum": _MAX_PROFILE_SAMPLE_ROWS,
            },
            "resource_bindings": {
                "type": "array",
                "items": _binding_schema(),
                "minItems": 1,
                "maxItems": MAX_JOB_RESOURCE_BINDINGS,
            },
        },
        "required": [
            "job_id",
            "specification_digest",
            "resource_ids",
            "sample_rows",
            "resource_bindings",
        ],
        "additionalProperties": False,
    }


def _start_output_schema() -> dict[str, object]:
    properties = {
        "job_id": {"type": "string"},
        "job_kind": {"type": "string"},
        "status": {"type": "string"},
        "execution_mode": {"type": "string"},
        "specification_digest": {"type": "string"},
        "execution_capability_id": {"type": "string"},
        "execution_contract_digest": {"type": "string"},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _internal_output_schema() -> dict[str, object]:
    properties = {
        "job_id": {"type": "string"},
        "profiled_resources": {"type": "integer"},
        "sampled_rows": {"type": "integer"},
        "truncated_resources": {"type": "integer"},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _finalizer_input_schema() -> dict[str, object]:
    work_result_properties = {
        "task_id": {"type": "string", "minLength": 1},
        "result_id": {"type": "string", "minLength": 1},
        "result_digest": {"type": "string", "minLength": 1},
        "artifact_ref": {"type": "object"},
    }
    return {
        "type": "object",
        "properties": {
            "job_id": {"type": "string", "minLength": 1},
            "resource_ids": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "maxItems": MAX_JOB_RESOURCE_BINDINGS,
                "uniqueItems": True,
            },
            "sample_rows": {
                "type": "integer",
                "minimum": 1,
                "maximum": _MAX_PROFILE_SAMPLE_ROWS,
            },
            "work_results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": work_result_properties,
                    "required": list(work_result_properties),
                    "additionalProperties": False,
                },
                "minItems": 1,
                "maxItems": MAX_JOB_RESOURCE_BINDINGS,
            },
        },
        "required": ["job_id", "resource_ids", "sample_rows", "work_results"],
        "additionalProperties": False,
    }


def _finalizer_output_schema() -> dict[str, object]:
    properties = {
        "job_id": {"type": "string"},
        "profiled_resources": {"type": "integer"},
        "sampled_rows": {"type": "integer"},
        "truncated_resources": {"type": "integer"},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _binding_from_mapping(value: object) -> JobResourceBinding:
    if not isinstance(value, Mapping):
        raise CapabilityInputError(
            "job_specification_invalid",
            "A frozen job resource binding is malformed.",
        )
    try:
        return JobResourceBinding(
            source_id=str(value["source_id"]),
            source_revision=str(value["source_revision"]),
            resource_id=str(value["resource_id"]),
            resource_revision=str(value["resource_revision"]),
            adapter_id=str(value["adapter_id"]),
            sensitivity=ModelSensitivity(str(value["sensitivity"])),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CapabilityInputError(
            "job_specification_invalid",
            "A frozen job resource binding is malformed.",
        ) from error


def job_resource_bindings_payload(
    values: tuple[JobResourceBinding, ...],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "source_id": item.source_id,
            "source_revision": item.source_revision,
            "resource_id": item.resource_id,
            "resource_revision": item.resource_revision,
            "adapter_id": item.adapter_id,
            "sensitivity": item.sensitivity.value,
        }
        for item in values
    )


def _binding_payload(value: JobResourceBinding) -> dict[str, object]:
    return {
        "source_id": value.source_id,
        "source_revision": value.source_revision,
        "resource_id": value.resource_id,
        "resource_revision": value.resource_revision,
        "adapter_id": value.adapter_id,
        "sensitivity": value.sensitivity.value,
    }


def _model_sensitivity(value: str) -> ModelSensitivity:
    try:
        return ModelSensitivity(value)
    except ValueError:
        return ModelSensitivity.RESTRICTED


def _maximum_sensitivity(
    values: tuple[ModelSensitivity, ...],
) -> ModelSensitivity:
    order = {
        ModelSensitivity.PUBLIC: 0,
        ModelSensitivity.INTERNAL: 1,
        ModelSensitivity.CONFIDENTIAL: 2,
        ModelSensitivity.RESTRICTED: 3,
    }
    return max(values or (ModelSensitivity.RESTRICTED,), key=order.__getitem__)


def _quoted_relation(value: str) -> str:
    return ".".join('"' + part.replace('"', '""') + '"' for part in value.split("."))


def _column_profiles(
    columns: tuple[str, ...],
    rows: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    profiles = []
    for column in columns:
        values = tuple(row.get(column) for row in rows)
        distinct = {canonical_json(freeze_json(value)) for value in values}
        profiles.append(
            {
                "column": column,
                "sampled_values": len(values),
                "null_values": sum(value is None for value in values),
                "distinct_values": len(distinct),
                "observed_types": tuple(
                    sorted({_json_type(value) for value in values})
                ),
            }
        )
    return tuple(profiles)


def _json_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    return "array"


__all__ = [
    "DATA_PROFILE_DOMAIN_OWNER_ID",
    "DATA_PROFILE_EXECUTION_CAPABILITY_ID",
    "DATA_PROFILE_EXECUTION_EXECUTOR_ID",
    "DATA_PROFILE_FINALIZE_CAPABILITY_ID",
    "DATA_PROFILE_FINALIZE_EXECUTOR_ID",
    "DATA_PROFILE_FINALIZE_OUTPUT_KIND",
    "DataProfileAdmission",
    "DataProfileCapabilityDomain",
    "DataProfileDeclarations",
    "StartStaticDataProfileGraphExecutor",
    "START_DATA_PROFILE_CAPABILITY_ID",
    "START_DATA_PROFILE_EXECUTOR_ID",
    "START_DATA_PROFILE_TOOL_NAME",
    "data_profile_declarations",
    "data_profile_graph_integration_declarations",
    "job_resource_bindings_payload",
]
