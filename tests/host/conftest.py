"""Realistic values for every host surface type, captured from a real agent."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from daita import (
    Agent,
    FolderSource,
    IntervalSchedule,
    LocalWorkspace,
    MCPAuthentication,
    MCPAuthenticationMode,
    MCPBindingStatus,
    MCPServerInspection,
    MCPToolSelection,
    MisfirePolicy,
    PostgreSQLSource,
    ReportingMode,
    RoutineState,
    ScheduledRoutine,
    ScheduledRoutineDraft,
    ScheduledRoutineInspection,
    ScheduledRoutineSummary,
    SQLiteSource,
)
from daita._json import FrozenJsonObject
from daita.adapters.mcp import SDKMCPClientFactory
from daita.adapters.models import SourceRegistration
from daita.adapters.postgresql import PostgreSQLProbeResult, PostgreSQLProbeSchema
from daita.adapters.postgresql_write import RelationalUpdateReadiness
from daita.adapters.protocols import ResourceSource
from daita.artifacts.models import (
    ArtifactDeliveryReceipt,
    ArtifactDestination,
    ArtifactPayload,
    ArtifactRef,
)
from daita.capabilities import EffectEvidenceBasis, EffectObservation, EffectOutcome
from daita.catalog.models import CatalogResource, CatalogSummary
from daita.distribution.models import (
    DeliveryInspection,
    DeliveryView,
    DistributionDestination,
)
from daita.hosting.home_upgrade import AgentHomeStatus
from daita.hosting.sources import SourceEditResult
from daita.jobs.graph.models import (
    BudgetAmount,
    ControlKind,
    ControlState,
    GraphInspection,
    GraphJob,
    GraphMutation,
    GraphState,
    GraphTask,
    MutationDecision,
    TaskAttempt,
    TaskCheckpoint,
    TaskControl,
    TaskDependency,
    TaskResult,
    TaskState,
    canonical_digest,
)
from daita.jobs.owner import GraphBlockerProjection
from daita.jobs.projections import GraphBoardProjection, GraphTimelinePage
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    ModelSensitivity,
    ModelUsage,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.pricing import CostEstimate
from daita.llm.routing import ModelRoute, ModelRouteCandidate, RetryPolicy
from daita.loop.models import ConversationRun, LoopExit, Transcript
from daita.security import SecretReference
from daita.semantics import (
    ResourceRevisionBinding,
    SemanticAnnotation,
    SemanticAnnotationState,
    SemanticAnnotationView,
    SemanticEvidence,
    SemanticEvidenceKind,
    SemanticFieldReference,
    SemanticKind,
    SemanticSubject,
)
from daita.skills.store import Skill, SkillSummary
from daita.storage.sqlite import SQLiteStateStore
from daita.storage.sqlite_records import (
    EffectReceipt,
    EffectResolutionDecision,
    SourcePermissionsInspection,
    SourcePermissionsPreview,
    SourceReadMode,
    effect_receipt_id,
)
from daita.work_board import WorkBoard
from tests.support.distribution import no_artifact_outcome_contract
from tests.support.graph import GRAPH_NOW, graph_admission, task_result
from tests.support.mcp import MemoryKeychain, conformance_identities, mock_transport
from tests.support.toolbox_model import ToolboxAwareMockModelProvider
from tests.support.workspace import workspace_for

# Inside the seeded job's one-hour deadline, so the supervisor never expires it.
NOW = GRAPH_NOW + timedelta(minutes=10)
NAME = "surface"
USAGE = ModelUsage(
    input_tokens=100,
    output_tokens=10,
    cost_estimate=CostEstimate.complete(Decimal("0")),
)


@pytest.fixture(scope="module")
def surface_values(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[object, tuple[object, ...]]:
    """Map each surface annotation to realistic values of that type."""

    return asyncio.run(_capture(tmp_path_factory.mktemp("surface")))


def _call(call_id: str, name: str, arguments: dict[str, object]) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
        usage=USAGE,
    )


def _stop(text: str) -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text, usage=USAGE)


async def _capture(tmp_path: Path) -> dict[object, tuple[object, ...]]:
    database = tmp_path / "orders.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, region TEXT, amount REAL)"
        )
        connection.execute("INSERT INTO orders VALUES (1, 'EMEA', 110.5)")
    copy = tmp_path / "orders-copy.sqlite"
    copy.write_bytes(database.read_bytes())
    exports = tmp_path / "exports"
    exports.mkdir()
    provider = ToolboxAwareMockModelProvider(
        (), provider_id="mock:surface", complete_pricing=True
    )
    alpha, _beta = conformance_identities()
    workspace = workspace_for(tmp_path)
    options: dict[str, object] = {
        "root": tmp_path,
        "workspace": workspace,
        "model": provider,
        "model_profile": ModelProfile(
            id=provider.provider_id,
            context_window_tokens=128_000,
            max_output_tokens=2_000,
            supports_tools=True,
        ),
        "clock": lambda: NOW,
        "keychain": MemoryKeychain(),
        "mcp_client_factory": SDKMCPClientFactory(http_transport=mock_transport(alpha)),
    }
    created = await Agent.create(NAME, **options)  # type: ignore[arg-type]
    agent_id = created.id
    await created.close()
    await _seed_blocked_job(tmp_path / "agents" / NAME / "state.db", agent_id)
    home_status = await Agent.inspect_home(NAME, root=tmp_path)

    agent = await Agent.open(NAME, **options)  # type: ignore[arg-type]
    try:
        source = await agent.attach_sqlite(database, name="Orders")
        resources = await agent.list_catalog_resources(source_id=source.id)
        (resource,) = resources
        provider.replace_script(
            (
                _call("describe", "catalog_describe", {"source_id": source.id}),
                _call(
                    "note",
                    "artifact_create",
                    {
                        "from": "document",
                        "format": "txt",
                        "filename": "orders.txt",
                        "content": "EMEA has the largest order.\n",
                    },
                ),
                _stop("EMEA has the largest order."),
                _stop("The routine reports one order."),
            )
        )
        origin = await agent.run("Which region has the largest order?")
        assert origin.conversation_id is not None
        (artifact,) = artifacts = await agent.list_artifacts()

        (destination,) = destinations = await agent.distribution_destinations(
            origin.conversation_id, sensitivity_ceiling=ModelSensitivity.INTERNAL
        )
        draft = ScheduledRoutineDraft(
            origin_run_id=origin.run_id,
            title="Order count report",
            authorized_instruction="Report the number of orders.",
            schedule=IntervalSchedule(3_600, NOW + timedelta(hours=1)),
            misfire_policy=MisfirePolicy.LATEST_ONLY,
            reporting_mode=ReportingMode.ALWAYS,
            precheck=None,
            allowed_source_ids=(source.id,),
            allowed_connector_binding_ids=(),
            allowed_resource_ids=(resource.id,),
            allowed_capability_ids=("catalog.describe",),
            sensitivity_ceiling=ModelSensitivity.INTERNAL,
            outcome_contract=no_artifact_outcome_contract(),
            distribution_destination_id=destination.destination_id,
            eligible_model_routes=(provider.provider_id,),
            per_run_max_tokens=2_000,
            per_run_max_cost_usd=None,
            cumulative_max_tokens=20_000,
            cumulative_max_cost_usd=None,
            cumulative_max_attempts=10,
            cumulative_max_occurrences=10,
            maximum_consecutive_failures=3,
            expires_at=NOW + timedelta(days=30),
        )
        routine = await agent.create_routine(await agent.propose_routine(draft))
        await agent.run_routine_now(
            routine.routine_id, expected_revision=routine.revision
        )
        delivery_id = await _wait_for_delivery(agent, origin.conversation_id)
        delivery = await agent.inspect_delivery(delivery_id)
        board = await agent.work_board()
        acknowledged = await agent.acknowledge_delivery(delivery_id)
        inspection = await agent.inspect_routine(routine.routine_id)
        assert inspection is not None
        paused = await agent.pause_routine(
            routine.routine_id, expected_revision=inspection.routine.revision
        )

        job = await agent.inspect_job("job-blocked")
        jobs = await agent.list_jobs()
        job_board = await agent.job_board("job-blocked")
        timeline = await agent.job_timeline("job-blocked")
        blockers = await agent.graph_blockers("job-blocked")
        answered = await agent.answer_task_input(
            "job-blocked",
            "worker",
            "control-1",
            principal_id=agent_id,
            answer={"region": "EMEA"},
            idempotency_key="answer-1",
        )
        task = await agent.inspect_job_task("job-blocked", "worker")
        tasks = await agent.list_job_tasks("job-blocked")
        dependencies = await agent.list_job_dependencies("job-blocked")
        attempts = await agent.list_task_attempts("job-blocked", "worker")
        cancelled = await agent.cancel_job("job-blocked")

        await agent.save_skill("order-report", "Count orders.", "Count the rows.")
        annotation = _annotation(agent_id, source.id, resource, origin.run_id)
        await agent.save_semantic_annotation(annotation)
        selections = (
            MCPToolSelection(
                remote_name="lookup",
                local_alias="lookup",
                description="Read an admitted finding.",
            ),
        )
        mcp_inspection = await agent.inspect_mcp_server(endpoint=alpha.endpoint)
        mcp_status = await agent.attach_mcp_server(
            endpoint=alpha.endpoint, selections=selections
        )
        password = await agent.store_postgresql_password("surface-password")
        permissions = await agent.inspect_source_permissions(source.id)
        preview = await agent.preview_source_permissions(
            source_id=source.id,
            read_mode=SourceReadMode.SELECTED,
            read_resource_ids=(resource.id,),
            relational_write_scopes={},
        )

        receipt = _receipt(agent_id, origin.run_id, NOW)
        route = ModelRoute(
            candidates=(
                ModelRouteCandidate(
                    provider_id="openai:gpt-5",
                    profile=ModelProfile(
                        id="openai:gpt-5",
                        context_window_tokens=400_000,
                        max_output_tokens=128_000,
                        supports_tools=True,
                        supports_parallel_tools=True,
                    ),
                    secret_reference=SecretReference.parse(
                        "keychain:daita/model/openai"
                    ),
                ),
            ),
            retry_policy=RetryPolicy(max_attempts_per_candidate=1),
        )
        result = task_result(agent_id=agent_id, job_id="job-blocked")
        values: dict[object, tuple[object, ...]] = {
            # Scalars and arguments as a client builds them.
            type(None): (None,),
            bool: (True, False),
            int: (5432,),
            str: ("Which region has the largest order?",),
            str | None: (origin.conversation_id, None),
            int | None: (128_000, None),
            Path: (agent.home,),
            Path | None: (exports, None),
            str | Path: (str(database), database),
            timedelta: (timedelta(days=1, microseconds=7),),
            tuple[str, ...]: ((source.id, resource.id), ()),
            frozenset[GraphState]: (
                frozenset({GraphState.ACTIVE, GraphState.SUCCEEDED}),
            ),
            frozenset[TaskState]: (frozenset({TaskState.READY}),),
            frozenset[RoutineState]: (frozenset({RoutineState.ACTIVE}), frozenset()),
            ModelSensitivity: (ModelSensitivity.RESTRICTED,),
            EffectResolutionDecision: (EffectResolutionDecision.CLOSE_WITHOUT_RETRY,),
            SourceReadMode | str: (SourceReadMode.SELECTED, "all"),
            SemanticKind | None: (SemanticKind.GLOSSARY, None),
            SemanticAnnotationState | None: (SemanticAnnotationState.ACTIVE,),
            Mapping[str, object]: ({"region": "EMEA", "limit": 5},),
            Mapping[str, Mapping[str, object]]: (
                {resource.id: {"operations": ("update",), "key_columns": ("id",)}},
                {},
            ),
            dict[str, object] | None: (_analysis_summary(), None),
            LocalWorkspace | None: (workspace,),
            SecretReference: (password,),
            MCPAuthentication
            | None: (
                MCPAuthentication(MCPAuthenticationMode.BEARER, password),
                None,
            ),
            tuple[MCPToolSelection, ...]: (selections,),
            ResourceSource: (
                SQLiteSource(database, name="Orders"),
                FolderSource(tmp_path, subpaths=("exports",), name="Exports"),
                PostgreSQLSource(
                    host="db.internal.test",
                    database="analytics",
                    username="daita_reader",
                    credential=password,
                    schemas=("public", "sales"),
                ),
            ),
            ScheduledRoutineDraft: (draft,),
            # Results captured from the running agent.
            LoopExit: (origin,),
            Transcript: (await agent.transcript(origin.run_id),),
            tuple[ConversationRun, ...]: (
                await agent.conversation_runs(origin.conversation_id),
            ),
            tuple[tuple[ToolCall, ToolResultBlock | None], ...]: (
                await agent.tool_evidence(origin.run_id),
            ),
            SourceRegistration: (source,),
            tuple[SourceRegistration, ...]: (await agent.list_sources(),),
            tuple[CatalogResource, ...]: (resources,),
            CatalogSummary: (await agent.catalog_summary(),),
            FrozenJsonObject: (await agent.inspect_catalog_resource(resource.id),),
            SourcePermissionsInspection: (permissions,),
            SourcePermissionsPreview: (preview,),
            tuple[ArtifactRef, ...]: (artifacts,),
            ArtifactPayload: (await agent.read_artifact(artifact.artifact_id),),
            ArtifactDeliveryReceipt: (
                await agent.save_artifact(artifact.artifact_id, exports),
            ),
            ArtifactDestination: (await agent.export_destination(),),
            tuple[DistributionDestination, ...]: (destinations,),
            DeliveryInspection | None: (delivery,),
            DeliveryView | None: (acknowledged,),
            WorkBoard: (board,),
            ScheduledRoutine: (routine, paused),
            tuple[ScheduledRoutineSummary, ...]: (await agent.list_routines(),),
            ScheduledRoutineInspection | None: (inspection,),
            tuple[GraphJob, ...]: (jobs,),
            GraphJob | None: (cancelled,),
            GraphInspection | None: (job,),
            GraphTask | None: (task,),
            tuple[GraphTask, ...]: (tasks,),
            tuple[TaskDependency, ...]: (dependencies,),
            tuple[TaskAttempt, ...]: (attempts,),
            GraphBoardProjection | None: (job_board,),
            GraphTimelinePage | None: (timeline,),
            GraphBlockerProjection | None: (blockers,),
            TaskControl | None: (answered,),
            tuple[SkillSummary, ...]: (await agent.list_skills(),),
            Skill | None: (await agent.read_skill("order-report"),),
            SemanticAnnotationView
            | None: (await agent.read_semantic_annotation(annotation.id),),
            tuple[SemanticAnnotationView, ...]: (
                await agent.list_semantic_annotations(),
            ),
            MCPServerInspection: (mcp_inspection,),
            MCPBindingStatus: (mcp_status,),
            tuple[MCPBindingStatus, ...]: (await agent.list_mcp_servers(),),
            ModelProfile | None: (agent.model_profile,),
            AgentHomeStatus: (home_status,),
            # Records that need PostgreSQL, a model login or a finished effect.
            ModelRoute: (route,),
            ModelRoute | None: (route, agent.model_route),
            TaskResult: (result,),
            TaskResult | None: (result,),
            TaskControl: (answered,),
            tuple[TaskCheckpoint, ...]: ((_checkpoint(agent_id),),),
            GraphMutation: (_mutation(agent_id),),
            EffectReceipt: (receipt,),
            EffectReceipt | None: (receipt,),
            tuple[EffectReceipt, ...]: ((receipt,),),
            PostgreSQLProbeResult: (
                PostgreSQLProbeResult(
                    schemas=(
                        PostgreSQLProbeSchema("public", has_base_tables=True),
                        PostgreSQLProbeSchema("staging", has_base_tables=False),
                    ),
                ),
            ),
            RelationalUpdateReadiness: (_readiness(source.id, resource.id),),
        }

        async def accept(_preview: object) -> bool:
            return True

        # Last: the edit re-keys the catalog that the reads above describe.
        values[SourceEditResult | None] = (
            await agent.edit_sqlite_source(
                source.id, copy, confirmation_handler=accept, name="Orders copy"
            ),
        )
        return values
    finally:
        await agent.close()


async def _seed_blocked_job(state_path: Path, agent_id: str) -> None:
    """Admit one graph job whose worker waits on the user's answer."""

    store = await SQLiteStateStore.open(state_path)
    try:
        await store.admit_graph(
            graph_admission(job_id="job-blocked", agent_id=agent_id)
        )
        attempt = await store.claim_graph_task(
            agent_id,
            "job-blocked",
            "worker",
            attempt_id="attempt-1",
            claim_token="claim-1",
            run_id="run-1",
            executor_id="executor-1",
            claimed_at=GRAPH_NOW,
            lease_seconds=30,
            absolute_deadline_at=GRAPH_NOW + timedelta(minutes=2),
            budget_reservations=(BudgetAmount("work_units", 2),),
        )
        assert attempt is not None
        assert await store.start_graph_attempt(
            agent_id,
            "job-blocked",
            "worker",
            "attempt-1",
            claim_token="claim-1",
            fencing_epoch=attempt.fencing_epoch,
            started_at=GRAPH_NOW + timedelta(seconds=1),
        )
        payload = {"question": "Which region should the report cover?"}
        await store.open_graph_control(
            TaskControl(
                agent_id=agent_id,
                job_id="job-blocked",
                task_id="worker",
                control_id="control-1",
                kind=ControlKind.NEEDS_INPUT,
                state=ControlState.OPEN,
                requesting_attempt_id="attempt-1",
                payload=payload,
                created_at=GRAPH_NOW + timedelta(seconds=2),
                payload_digest=canonical_digest(payload),
            ),
            claim_token="claim-1",
            fencing_epoch=attempt.fencing_epoch,
        )
    finally:
        await store.close()


async def _wait_for_delivery(agent: Agent, conversation_id: str) -> str:
    deadline = asyncio.get_running_loop().time() + 15.0
    while asyncio.get_running_loop().time() < deadline:
        deliveries = await agent.list_deliveries(conversation_id=conversation_id)
        if deliveries:
            return deliveries[0].delivery_id
        await asyncio.sleep(0.01)
    raise AssertionError("the routine did not deliver within 15 seconds")


def _annotation(
    agent_id: str, source_id: str, resource: CatalogResource, run_id: str
) -> SemanticAnnotation:
    return SemanticAnnotation(
        id="orders-amount",
        agent_id=agent_id,
        subject=SemanticSubject(
            source_ids=(source_id,),
            resource_ids=(resource.id,),
            fields=(SemanticFieldReference(resource.id, "amount"),),
        ),
        kind=SemanticKind.GLOSSARY,
        statement="amount is the order total in euros.",
        evidence=(
            SemanticEvidence(
                SemanticEvidenceKind.USER_ASSERTION, run_id, message_position=0
            ),
        ),
        catalog_revisions=(
            ResourceRevisionBinding(resource.id, resource.current_revision),
        ),
        created_at=NOW,
        confirmed_at=NOW,
        confirmed_by="local-user",
    )


def _receipt(agent_id: str, run_id: str, started_at: datetime) -> EffectReceipt:
    operation_key = canonical_digest({"operation": "orders.update"})
    return EffectReceipt(
        receipt_id=effect_receipt_id(
            agent_id=agent_id,
            run_id=run_id,
            call_id="update-orders",
            operation_key=operation_key,
        ),
        receipt_kind="data.relational_update",
        agent_id=agent_id,
        run_id=run_id,
        call_id="update-orders",
        capability_id="data.update_rows",
        domain_owner_id="data",
        capability_contract_digest=operation_key,
        operation_key=operation_key,
        argument_fingerprint=operation_key,
        sensitivity=ModelSensitivity.INTERNAL,
        started_at=started_at,
    ).finish(
        EffectObservation(
            EffectOutcome.SUCCEEDED,
            EffectEvidenceBasis.ADAPTER_VERIFIED,
            FrozenJsonObject.from_mapping({"affected_rows": 1}),
        ),
        finished_at=started_at + timedelta(seconds=1),
    )


def _checkpoint(agent_id: str) -> TaskCheckpoint:
    payload = {"rows_scanned": 120, "phase": "aggregate"}
    return TaskCheckpoint(
        agent_id=agent_id,
        job_id="job-blocked",
        task_id="worker",
        attempt_id="attempt-1",
        checkpoint_id="checkpoint-1",
        fencing_epoch=1,
        ordinal=1,
        milestone="Scanned the orders table.",
        payload=payload,
        created_at=NOW,
        payload_digest=canonical_digest(payload),
    )


def _mutation(agent_id: str) -> GraphMutation:
    return GraphMutation(
        agent_id=agent_id,
        job_id="job-blocked",
        mutation_id="mutation-1",
        actor_kind="principal",
        actor_key=agent_id,
        idempotency_key="replace-worker-1",
        payload_digest=canonical_digest({"replace": "worker"}),
        expected_revision=1,
        committed_revision=2,
        decision=MutationDecision.COMMITTED,
        created_at=NOW,
        resulting_task_ids=("worker-2",),
        resulting_edges=(("worker-2", "finalizer"),),
    )


def _readiness(source_id: str, resource_id: str) -> RelationalUpdateReadiness:
    return RelationalUpdateReadiness(
        source_id=source_id,
        resource_id=resource_id,
        assignment_columns=("amount",),
        daita_scope_ready=False,
        ready_for_preview=False,
        proves_execution=False,
        role_attributes=FrozenJsonObject.from_mapping(
            {
                "superuser": False,
                "bypass_rls": False,
                "create_database": False,
                "create_role": False,
                "replication": False,
            }
        ),
        privileges=FrozenJsonObject.from_mapping(
            {
                "database_connect": True,
                "schema_usage": True,
                "table_select": True,
                "requested_columns_update": None,
            }
        ),
        relation=FrozenJsonObject.from_mapping(
            {
                "catalog_admitted": True,
                "base_table": True,
                "partition": False,
                "inheritance": False,
                "row_level_security": False,
                "force_row_level_security": False,
                "user_triggers": None,
                "rewrite_rules": False,
            }
        ),
        rejection_codes=("relational_write_scope_missing",),
        remediation_categories=("daita_permission",),
    )


def _analysis_summary() -> dict[str, object]:
    """The shape Agent.analysis_usage returns for one analysed run."""

    return {
        "worker_generations": 1,
        "user_cpu_seconds": 0.42,
        "system_cpu_seconds": 0.05,
        "wall_seconds": 1.3,
        "outer_tool_calls": {"analysis_execute": 1},
        "scratch_samples_incomplete": False,
        "captured_input_bytes": None,
    }
