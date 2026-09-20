"""Explicit integration-only composition for the Phase 3 static graph slice."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from daita.artifacts.store import AgentHomeArtifactStore
from daita.capabilities import (
    Capability,
    CapabilityDeclarations,
    CapabilityRegistry,
    ToolExecution,
)
from daita.capability_runtime import CapabilityRuntime
from daita.distribution.owner import DistributionOwner
from daita.domains.data.capabilities import SqlReadResult
from daita.domains.data.profile_jobs import (
    DATA_PROFILE_DOMAIN_OWNER_ID,
    DATA_PROFILE_FINALIZE_CAPABILITY_ID,
    DataProfileAdmission,
    DataProfileCapabilityDomain,
    StartDataProfileGraphExecutor,
    data_profile_declarations,
)
from daita.domains.data.results import project_result_rows
from daita.domains.data.sql import ResourceSchema
from daita.domains.learning import LearningCandidateGuard
from daita.hosting.execution_governor import RunAdmissionCoordinator
from daita.jobs.graph.models import GraphAdmission, GraphInspection, GraphState
from daita.jobs.owner import JobOwner
from daita.jobs.supervisor import JobSupervisor
from daita.loop.models import RunInput
from daita.storage.sqlite import SQLiteStateStore

AGENT_ID = "agent-static-graph"
CONVERSATION_ID = "conversation-static-graph"


def _digest(value: str) -> str:
    return "sha256:" + sha256(value.encode("utf-8")).hexdigest()


class DeterministicIds:
    def __init__(self, namespace: str) -> None:
        self._namespace = namespace
        self._counts: Counter[str] = Counter()

    def __call__(self, prefix: str) -> str:
        self._counts[prefix] += 1
        suffix = sha256(
            f"{self._namespace}:{prefix}:{self._counts[prefix]}".encode("utf-8")
        ).hexdigest()[:32]
        return f"{prefix}-{suffix}"


class StaticProfileCatalog:
    def __init__(self, resources: tuple[tuple[str, str], ...]) -> None:
        self.schemas = tuple(
            ResourceSchema(
                resource_id=resource_id,
                source_id=source_id,
                name=f"table_{index}",
                columns=("id", "value"),
                revision=_digest(f"resource:{resource_id}"),
                source_revision=_digest(f"source:{source_id}"),
                resource_kind="table",
                sensitivity_class="internal",
                column_declared_types=(("id", "INTEGER"), ("value", "TEXT")),
            )
            for index, (source_id, resource_id) in enumerate(resources)
        )

    async def source_routing_facts(
        self, agent_id: str, source_ids: tuple[str, ...] = ()
    ) -> tuple[Mapping[str, object], ...]:
        assert agent_id == AGENT_ID
        return tuple(
            {"source_id": source_id, "adapter_id": "sqlite"}
            for source_id in sorted({item.source_id for item in self.schemas})
            if not source_ids or source_id in source_ids
        )

    async def source_adapter_id(self, agent_id: str, source_id: str) -> str | None:
        assert agent_id == AGENT_ID
        return (
            "sqlite"
            if any(item.source_id == source_id for item in self.schemas)
            else None
        )

    async def resource_schemas(
        self, agent_id: str, source_id: str
    ) -> tuple[ResourceSchema, ...]:
        assert agent_id == AGENT_ID
        return tuple(item for item in self.schemas if item.source_id == source_id)

    async def resource_identity(
        self, agent_id: str, resource_id: str
    ) -> tuple[str, str, str] | None:
        assert agent_id == AGENT_ID
        schema = next(
            (item for item in self.schemas if item.resource_id == resource_id), None
        )
        if schema is None or schema.revision is None:
            return None
        return schema.source_id, "table", schema.revision

    async def readable_resource_ids(
        self, agent_id: str, source_ids: tuple[str, ...] = ()
    ) -> frozenset[str]:
        assert agent_id == AGENT_ID
        return frozenset(
            item.resource_id
            for item in self.schemas
            if not source_ids or item.source_id in source_ids
        )


class ObservedReadBackend:
    def __init__(self, catalog: StaticProfileCatalog, *, delay: float = 0.0) -> None:
        self._catalog = catalog
        self.delay = delay
        self.calls: Counter[str] = Counter()
        self.completed: Counter[str] = Counter()
        self.active_by_source: Counter[str] = Counter()
        self.maximum_by_source: Counter[str] = Counter()
        self.maximum_global = 0
        self._active_global = 0
        self.block_after: int | None = None
        self.blocked = asyncio.Event()
        self.release = asyncio.Event()
        self.assert_store_idle: Callable[[], None] | None = None

    async def execute_read(
        self,
        *,
        agent_id: str,
        source_id: str,
        sql: str,
        parameters: tuple[object, ...],
        max_rows: int,
        max_bytes: int,
    ) -> SqlReadResult:
        del parameters
        assert agent_id == AGENT_ID
        if self.assert_store_idle is not None:
            self.assert_store_idle()
        schema = next(
            item
            for item in self._catalog.schemas
            if item.source_id == source_id and f'"{item.name}"' in sql
        )
        self.calls[schema.resource_id] += 1
        self.active_by_source[source_id] += 1
        self.maximum_by_source[source_id] = max(
            self.maximum_by_source[source_id], self.active_by_source[source_id]
        )
        self._active_global += 1
        self.maximum_global = max(self.maximum_global, self._active_global)
        try:
            if (
                self.block_after is not None
                and sum(self.completed.values()) >= self.block_after
            ):
                self.blocked.set()
                await self.release.wait()
            if self.delay:
                await asyncio.sleep(self.delay)
            rows = (
                {"id": 1, "value": schema.resource_id},
                {"id": 2, "value": None},
            )
            projection = project_result_rows(
                rows,
                max_rows=max_rows,
                max_bytes=max_bytes,
            )
            self.completed[schema.resource_id] += 1
            assert schema.revision is not None
            assert schema.source_revision is not None
            return SqlReadResult(
                source_id=source_id,
                canonical_sql=sql,
                sql_fingerprint=_digest(sql),
                resource_ids=(schema.resource_id,),
                resource_revisions=((schema.resource_id, schema.revision),),
                source_revision=schema.source_revision,
                columns=schema.columns,
                projection=projection,
            )
        finally:
            self._active_global -= 1
            self.active_by_source[source_id] -= 1


@dataclass(slots=True)
class StaticGraphIntegration:
    """All draft dependencies, deliberately absent from production composition."""

    store: SQLiteStateStore
    artifacts: AgentHomeArtifactStore
    coordinator: RunAdmissionCoordinator
    distribution: DistributionOwner
    owner: JobOwner
    runtime: CapabilityRuntime
    supervisor: JobSupervisor
    admission: DataProfileAdmission
    finalizer_capability: Capability
    starter: StartDataProfileGraphExecutor
    catalog: StaticProfileCatalog
    backend: ObservedReadBackend
    clock: Callable[[], datetime]
    ids: DeterministicIds

    @classmethod
    async def open(
        cls,
        root: Path,
        resources: tuple[tuple[str, str], ...],
        *,
        initialize: bool = True,
        namespace: str = "initial",
        graph_parallelism: int = 1,
        execution_capacity: int = 1,
        source_capacities: Mapping[str, int] | None = None,
        backend: ObservedReadBackend | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> StaticGraphIntegration:
        resolved_clock = clock or (lambda: datetime.now(UTC))
        ids = DeterministicIds(namespace)
        if not initialize and not (root / "state.sqlite").is_file():
            raise FileNotFoundError(root / "state.sqlite")
        store = await SQLiteStateStore.open(
            root / "state.sqlite",
            clock=resolved_clock,
        )
        artifacts = await AgentHomeArtifactStore.open(
            agent_id=AGENT_ID,
            agent_home=root,
            references=store,
            clock=resolved_clock,
            id_factory=ids,
        )
        owner = JobOwner(
            agent_id=AGENT_ID,
            store=store,
            clock=resolved_clock,
            id_factory=ids,
        )
        distribution = DistributionOwner(agent_id=AGENT_ID, store=store)
        catalog = StaticProfileCatalog(resources)
        resolved_backend = backend or ObservedReadBackend(catalog)
        declarations, admission = data_profile_declarations(
            agent_id=AGENT_ID,
            catalog=catalog,
            owner=owner,
            sqlite_backend=resolved_backend,
            postgresql_backend=resolved_backend,
            artifacts=artifacts,
            distribution=distribution,
            clock=resolved_clock,
            id_factory=ids,
        )
        bundle = CapabilityDeclarations(
            domain_owner_id=DATA_PROFILE_DOMAIN_OWNER_ID,
            capabilities=declarations.capabilities,
            executor_ids=tuple(item.executor_id for item in declarations.capabilities),
            tool_views=declarations.tool_views,
        )
        domain = DataProfileCapabilityDomain(
            bundle,
            catalog=catalog,
            admission=admission,
            learning=LearningCandidateGuard(),
        )
        runtime = CapabilityRuntime(
            CapabilityRegistry(
                declarations=(bundle,),
                executors=declarations.executors,
            ),
            (domain,),
            artifacts=artifacts,
            clock=resolved_clock,
        )
        coordinator = RunAdmissionCoordinator(
            execution_capacity=execution_capacity,
            source_resource_capacities=source_capacities,
            sqlite_pressure_capacity=min(4, execution_capacity),
        )
        supervisor = JobSupervisor(
            agent_id=AGENT_ID,
            store=store,
            owner=owner,
            runtime=runtime,
            artifacts=artifacts,
            clock=resolved_clock,
            id_factory=ids,
            admission_coordinator=coordinator,
            distribution=distribution,
            graph_parallelism=graph_parallelism,
            poll_seconds=0.005,
        )
        owner.bind_wake(supervisor.wake)
        finalizer = next(
            item
            for item in declarations.capabilities
            if item.id == DATA_PROFILE_FINALIZE_CAPABILITY_ID
        )
        starter = next(
            item
            for item in declarations.executors
            if isinstance(item, StartDataProfileGraphExecutor)
        )
        return cls(
            store=store,
            artifacts=artifacts,
            coordinator=coordinator,
            distribution=distribution,
            owner=owner,
            runtime=runtime,
            supervisor=supervisor,
            admission=admission,
            finalizer_capability=finalizer,
            starter=starter,
            catalog=catalog,
            backend=resolved_backend,
            clock=resolved_clock,
            ids=ids,
        )

    async def build(
        self,
        resource_ids: tuple[str, ...],
        *,
        sample_rows: int = 20,
    ) -> GraphAdmission:
        return await self.admission.build_static_graph_admission(
            run=RunInput(
                id="run-" + "a" * 32,
                agent_id=AGENT_ID,
                message="Build the exact static graph.",
                created_at=self.clock(),
                conversation_id=CONVERSATION_ID,
                source_scope_ids=tuple(
                    sorted(
                        {
                            item.source_id
                            for item in self.catalog.schemas
                            if item.resource_id in resource_ids
                        }
                    )
                ),
            ),
            call_id="static-profile-call",
            arguments={
                "resource_ids": resource_ids,
                "sample_rows": sample_rows,
                "deadline_seconds": 300,
            },
            finalizer_capability=self.finalizer_capability,
            distribution=self.distribution,
            clock=self.clock,
            id_factory=self.ids,
        )

    async def admit_and_start(self, admission: GraphAdmission) -> None:
        await self.owner.admit(admission)
        await self.supervisor.start()

    async def start_profile(
        self,
        resource_ids: tuple[str, ...],
        *,
        sample_rows: int = 20,
    ) -> str:
        output = await self.starter.execute(
            ToolExecution(
                run_id="run-" + "b" * 32,
                call_id="start-static-profile",
                capability_id="jobs.data_profile.start",
                conversation_id=CONVERSATION_ID,
                arguments={
                    "resource_ids": resource_ids,
                    "sample_rows": sample_rows,
                    "deadline_seconds": 300,
                },
            )
        )
        job_id = output.data["job_id"]
        assert isinstance(job_id, str)
        await self.supervisor.start()
        return job_id

    async def wait_terminal(
        self, job_id: str, *, timeout: float = 5.0
    ) -> GraphInspection:
        deadline = asyncio.get_running_loop().time() + timeout
        inspection = None
        while asyncio.get_running_loop().time() < deadline:
            inspection = await self.owner.inspect_graph(job_id)
            assert inspection is not None
            if inspection.job.state in {GraphState.SUCCEEDED, GraphState.FAILED}:
                return inspection
            await asyncio.sleep(0.005)
        driver = self.supervisor._driver
        error = None if driver is None or not driver.done() else driver.exception()
        raise AssertionError(
            f"graph did not terminate: {inspection!r}; driver={error!r}; "
            f"calls={self.backend.calls!r}; completed={self.backend.completed!r}"
        )

    async def close(self) -> None:
        await self.supervisor.close()
        await self.artifacts.close()
        await self.store.close()


__all__ = [
    "AGENT_ID",
    "CONVERSATION_ID",
    "ObservedReadBackend",
    "StaticGraphIntegration",
    "StaticProfileCatalog",
]
