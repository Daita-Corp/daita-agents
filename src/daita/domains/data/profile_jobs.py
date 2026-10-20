"""Admit and execute durable data-profile jobs through the data capability domain."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from ..._json import FrozenJsonObject, canonical_json, freeze_json
from ...adapters.registry import SourceOperation, SourceTypeRegistry
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
    ToolExecution,
    ToolOutput,
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
    canonical_digest,
    topology_digest,
)
from ...jobs.owner import JobError, JobOwner
from ...llm.models import ModelSensitivity, ToolCall
from ...loop.models import RunInput
from ...loop.session import RunSession
from ...storage.sqlite_records import SourcePermissionStateError
from .capabilities import SqlReadBackend, SqlReadResult
from .sql import ResourceSchema, qualified_relation, read_validator

DATA_PROFILE_DOMAIN_OWNER_ID = "data_profile_jobs"
DATA_PROFILE_EXECUTION_CAPABILITY_ID = "jobs.data_profile.execute"
DATA_PROFILE_EXECUTION_EXECUTOR_ID = "jobs.data_profile.execute.executor"
DATA_PROFILE_EXECUTION_OUTPUT_KIND = "job.data_profile.result"
DATA_PROFILE_FINALIZE_CAPABILITY_ID = "jobs.data_profile.finalize"
DATA_PROFILE_FINALIZE_EXECUTOR_ID = "jobs.data_profile.finalize.executor"
DATA_PROFILE_FINALIZE_OUTPUT_KIND = "job.data_profile.final_result"

_MAX_PROFILE_SAMPLE_ROWS = 100
_MAX_PROFILE_READ_BYTES = 256 * 1024
_MAX_PROFILE_ARTIFACT_BYTES = 1 * 1024 * 1024
MAX_JOB_RESOURCE_BINDINGS = 16
MAX_JOB_WALL_TIME_SECONDS = 300.0
MAX_JOB_DEADLINE_SECONDS = 3_600.0


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


@dataclass(frozen=True, slots=True)
class ProfileResourceBinding:
    source_id: str
    source_revision: str
    resource_id: str
    resource_revision: str
    adapter_id: str
    sensitivity: ModelSensitivity

    def __post_init__(self) -> None:
        for value, label in (
            (self.source_id, "source_id"),
            (self.source_revision, "source_revision"),
            (self.resource_id, "resource_id"),
            (self.resource_revision, "resource_revision"),
            (self.adapter_id, "adapter_id"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"profile resource {label} must be non-empty text")
        if not isinstance(self.sensitivity, ModelSensitivity):
            raise TypeError("profile resource sensitivity is invalid")


@dataclass(frozen=True, slots=True)
class DataProfileSpecification:
    arguments: FrozenJsonObject
    resource_bindings: tuple[ProfileResourceBinding, ...]
    execution_capability_id: str
    execution_contract_digest: str
    sensitivity: ModelSensitivity
    deadline_at: datetime
    max_wall_time_seconds: float

    @property
    def digest(self) -> str:
        return canonical_digest(
            {
                "kind": "data_profile",
                "arguments": self.arguments,
                "resource_bindings": tuple(
                    _binding_payload(binding) for binding in self.resource_bindings
                ),
                "execution_capability_id": self.execution_capability_id,
                "execution_contract_digest": self.execution_contract_digest,
                "sensitivity": self.sensitivity.value,
                "deadline_at": self.deadline_at.isoformat(),
                "max_wall_time_seconds": self.max_wall_time_seconds,
            }
        )


class DataProfileAdmission:
    """Resolve exact current resource facts for one frozen profile specification.

    It also builds the code-authored static profile graph run by the internal
    executor and finalizer; no model tool starts one.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        catalog: DataProfileCatalog,
        owner: JobOwner,
        execution_capability: Capability,
        finalizer_capability: Capability,
        distribution: DistributionOwner,
        source_types: SourceTypeRegistry,
        clock: Callable[[], datetime],
        id_factory: Callable[[str], str],
    ) -> None:
        self._agent_id = agent_id
        self._catalog = catalog
        self._source_types = source_types
        self._owner = owner
        self._execution_capability = execution_capability
        self._finalizer_capability = finalizer_capability
        self._distribution = distribution
        self._clock = clock
        self._id_factory = id_factory
        self._execution_contract_digest = capability_contract_digest(
            execution_capability,
            domain_owner_id=DATA_PROFILE_DOMAIN_OWNER_ID,
        )

    async def build_specification(
        self,
        arguments: Mapping[str, object],
    ) -> DataProfileSpecification:
        resource_ids = arguments.get("resource_ids")
        sample_rows = arguments.get("sample_rows", _MAX_PROFILE_SAMPLE_ROWS)
        deadline_raw = arguments.get("_deadline_at")
        wall_time = arguments.get("_max_wall_time_seconds")
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
        return DataProfileSpecification(
            arguments=FrozenJsonObject.from_mapping(
                {"resource_ids": resource_ids, "sample_rows": sample_rows}
            ),
            resource_bindings=bindings,
            execution_capability_id=self._execution_capability.id,
            execution_contract_digest=self._execution_contract_digest,
            sensitivity=sensitivity,
            deadline_at=deadline.astimezone(UTC),
            max_wall_time_seconds=float(wall_time),
        )

    async def build_static_graph_admission(
        self,
        *,
        run: RunInput,
        call_id: str,
        arguments: Mapping[str, object],
    ) -> GraphAdmission:
        """Build the current code-authored static profile graph."""

        finalizer_capability = self._finalizer_capability
        distribution = self._distribution
        clock = self._clock
        id_factory = self._id_factory

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

    async def _current_bindings(
        self,
        resource_ids: tuple[object, ...],
    ) -> tuple[ProfileResourceBinding, ...]:
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
        schemas_by_source: dict[str, tuple[ResourceSchema, ...]] = {}
        adapters: dict[str, str] = {}
        for source_id in source_ids:
            adapter = await self._catalog.source_adapter_id(
                self._agent_id,
                source_id,
            )
            if adapter is None or not self._source_types.supports(
                adapter, SourceOperation.PROFILE
            ):
                raise CapabilityInputError(
                    "data_profile_adapter_unsupported",
                    "The first data-profile job supports SQLite and PostgreSQL resources.",
                )
            adapters[source_id] = adapter
            source_schemas = await self._catalog.resource_schemas(
                self._agent_id,
                source_id,
            )
            schemas_by_source[source_id] = source_schemas
            for current_schema in source_schemas:
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
            dialect = self._source_types.require(adapters[source_id]).sql_dialect
            validator = read_validator(dialect)
            if validator is None:
                raise CapabilityInputError(
                    "data_profile_adapter_unsupported",
                    "The first data-profile job supports SQLite and PostgreSQL resources.",
                )
            validation = validator(
                _profile_sql(schema, dialect, _MAX_PROFILE_SAMPLE_ROWS),
                source_id=source_id,
                resources=schemas_by_source[source_id],
                parameters=(),
                allowed_resource_ids=readable,
            )
            if not validation.valid or validation.resource_ids != (resource_id,):
                raise CapabilityInputError(
                    "data_profile_query_invalid",
                    "A selected resource cannot be profiled under its current catalog contract.",
                )
            bindings.append(
                ProfileResourceBinding(
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


class DataProfileExecutor:
    executor_id = DATA_PROFILE_EXECUTION_EXECUTOR_ID

    def __init__(
        self,
        *,
        agent_id: str,
        catalog: DataProfileCatalog,
        backends: Mapping[str, SqlReadBackend],
        source_types: SourceTypeRegistry,
    ) -> None:
        self._agent_id = agent_id
        self._catalog = catalog
        self._backends = dict(backends)
        self._source_types = source_types

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
        binding: ProfileResourceBinding,
        schema: ResourceSchema,
        sample_rows: int,
    ) -> SqlReadResult:
        backend = self._backends.get(binding.adapter_id)
        if backend is None:
            raise CapabilityInputError(
                "data_profile_adapter_unsupported",
                "Data profiles support only SQLite and PostgreSQL resources.",
            )
        sql = _profile_sql(
            schema,
            self._source_types.require(binding.adapter_id).sql_dialect,
            sample_rows,
        )
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
    """Validate internal profile graph tasks; this domain projects no tools.

    Profile graphs keep their internal executor and finalizer so migrated
    revision-2 profile jobs still run. Models profile data with data_query
    aggregates instead of a durable job.
    """

    domain_owner_id = DATA_PROFILE_DOMAIN_OWNER_ID

    def __init__(
        self,
        declarations: CapabilityDeclarations,
        *,
        admission: DataProfileAdmission,
    ) -> None:
        if declarations.domain_owner_id != self.domain_owner_id:
            raise ValueError("data-profile declarations have the wrong owner")
        capability_ids = {item.id for item in declarations.capabilities}
        if capability_ids not in (
            {DATA_PROFILE_EXECUTION_CAPABILITY_ID},
            {
                DATA_PROFILE_EXECUTION_CAPABILITY_ID,
                DATA_PROFILE_FINALIZE_CAPABILITY_ID,
            },
        ):
            raise ValueError("data-profile domain requires its exact capabilities")
        if declarations.tool_views:
            raise ValueError("data-profile capabilities are internal only")
        self._declarations = declarations
        self._admission = admission

    @property
    def declarations(self) -> CapabilityDeclarations:
        return self._declarations

    async def project(
        self,
        run: RunInput,
        session: RunSession | None = None,
    ) -> tuple[str, ...]:
        del run, session
        return ()

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
        del run, call, request_sensitivity, session
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
        raise ValueError("internal data-profile execution has no operational effect")

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
        del run, call, capability, arguments, request_sensitivity, session
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
    backends: Mapping[str, SqlReadBackend],
    source_types: SourceTypeRegistry,
    artifacts: AgentHomeArtifactStore,
    distribution: DistributionOwner,
    clock: Callable[[], datetime],
    id_factory: Callable[[str], str],
) -> tuple[DataProfileDeclarations, DataProfileAdmission]:
    """Compose the internal static data-profile graph executors."""

    internal = Capability(
        id=DATA_PROFILE_EXECUTION_CAPABILITY_ID,
        description="Execute one exact frozen read-only data-profile graph task.",
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
    admission = DataProfileAdmission(
        agent_id=agent_id,
        catalog=catalog,
        owner=owner,
        execution_capability=internal,
        finalizer_capability=finalizer,
        distribution=distribution,
        source_types=source_types,
        clock=clock,
        id_factory=id_factory,
    )
    return (
        DataProfileDeclarations(
            capabilities=(internal, finalizer),
            executors=(
                DataProfileExecutor(
                    agent_id=agent_id,
                    catalog=catalog,
                    backends=backends,
                    source_types=source_types,
                ),
                DataProfileFinalizerExecutor(artifacts),
            ),
            tool_views=(),
        ),
        admission,
    )


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


def _binding_from_mapping(value: object) -> ProfileResourceBinding:
    if not isinstance(value, Mapping):
        raise CapabilityInputError(
            "job_specification_invalid",
            "A frozen job resource binding is malformed.",
        )
    try:
        return ProfileResourceBinding(
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
    values: tuple[ProfileResourceBinding, ...],
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


def _binding_payload(value: ProfileResourceBinding) -> dict[str, object]:
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


def _profile_sql(schema: ResourceSchema, dialect: str | None, sample_rows: int) -> str:
    relation = qualified_relation(schema, dialect)
    if relation is None:
        raise CapabilityInputError(
            "data_profile_resource_stale",
            "A PostgreSQL profile resource lacks its exact qualified catalog identity.",
        )
    return f"SELECT * FROM {relation} LIMIT {sample_rows}"


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
    "DataProfileSpecification",
    "ProfileResourceBinding",
    "data_profile_declarations",
    "job_resource_bindings_payload",
]
