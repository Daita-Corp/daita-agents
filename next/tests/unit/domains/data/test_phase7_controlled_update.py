from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from daita._json import FrozenJsonObject
from daita.adapters.models import SourceRegistration
from daita.capabilities import (
    CapabilityRegistry,
    ExecutionRequest,
    ExecutorKnownNoEffectError,
)
from daita.catalog import (
    CatalogFacet,
    CatalogResource,
    CatalogResourceRevision,
    CatalogService,
    CatalogSync,
    CatalogSyncStatus,
    ResourceKind,
    Sensitivity,
    TabularColumn,
    TabularFacet,
    TabularIndex,
    catalog_resource_id,
)
from daita.domains.data import (
    SQLITE_UPDATE_CAPABILITY_ID,
    SQLITE_UPDATE_EVIDENCE_KIND,
    SQLITE_UPDATE_IMPACT_CAPABILITY_ID,
    SQLITE_UPDATE_IMPACT_EVIDENCE_KIND,
    SQLITE_UPDATE_IMPACT_TOOL_NAME,
    SQLITE_UPDATE_TOOL_NAME,
    CatalogDataView,
    DataDomainController,
    ResourceSchema,
    SQLiteUpdateImpactResult,
    SQLiteUpdateKnownNoEffectError,
    SQLiteUpdateResult,
    sqlite_declared_type_affinity,
    sqlite_identifier_key,
    sqlite_update_declarations,
    validate_sqlite_update_recipe,
)
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
)
from daita.loop.models import LoopPhase
from daita.operations.governance import ApprovalStatus
from daita.operations.models import (
    ActionProposal,
    ActionRejection,
    AgentTrigger,
    TriggerKind,
)
from daita.operations.runtime import OperationRuntime

NOW = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc)
REVISION = "sha256:" + "a" * 64


def _resource(
    *,
    resource_kind: str = "table",
    writable: bool = True,
    unique_key_columns: tuple[str, ...] = ("id",),
) -> ResourceSchema:
    return ResourceSchema(
        resource_id="resource-orders",
        source_id="source-orders",
        name="orders",
        columns=("id", "status", "email"),
        aliases=("main.orders",),
        revision=REVISION,
        source_revision="schema_version:7",
        resource_kind=resource_kind,
        sensitivity_class="internal",
        writable=writable,
        unique_key_columns=unique_key_columns,
        column_declared_types=(
            ("id", "TEXT"),
            ("status", "TEXT"),
            ("email", "TEXT"),
        ),
    )


def _recipe():
    result = validate_sqlite_update_recipe(
        source_id="source-orders",
        resource_id="resource-orders",
        key_column="ID",
        key_value="order-7",
        target_column="STATUS",
        expected_value="pending",
        new_value="complete",
        resources=(_resource(),),
        source_write_access=True,
    )
    assert result.valid is True
    assert result.recipe is not None
    return result.recipe


def test_update_recipe_is_catalog_grounded_bounded_and_deterministic() -> None:
    first = _recipe()
    second = _recipe()

    assert first == second
    assert first.source_id == "source-orders"
    assert first.resource_id == "resource-orders"
    assert first.resource_revision == REVISION
    assert first.source_revision == "schema_version:7"
    assert first.table_name == "orders"
    assert first.key_column == "id"
    assert first.target_column == "status"
    assert first.recipe_fingerprint.startswith("sha256:")

    denied_access = validate_sqlite_update_recipe(
        source_id="source-orders",
        resource_id="resource-orders",
        key_column="id",
        key_value="order-7",
        target_column="status",
        expected_value="pending",
        new_value="complete",
        resources=(_resource(),),
        source_write_access=False,
    )
    view = validate_sqlite_update_recipe(
        source_id="source-orders",
        resource_id="resource-orders",
        key_column="id",
        key_value="order-7",
        target_column="status",
        expected_value="pending",
        new_value="complete",
        resources=(_resource(resource_kind="view", writable=False),),
        source_write_access=True,
    )
    non_unique = validate_sqlite_update_recipe(
        source_id="source-orders",
        resource_id="resource-orders",
        key_column="email",
        key_value="ada@example.test",
        target_column="status",
        expected_value="pending",
        new_value="complete",
        resources=(_resource(),),
        source_write_access=True,
    )
    no_op = validate_sqlite_update_recipe(
        source_id="source-orders",
        resource_id="resource-orders",
        key_column="id",
        key_value="order-7",
        target_column="status",
        expected_value="complete",
        new_value="complete",
        resources=(_resource(),),
        source_write_access=True,
    )

    assert denied_access.issue_codes[0] == "source_write_access_required"
    assert "resource_not_writable_table" in view.issue_codes
    assert "key_not_unique" in non_unique.issue_codes
    assert "no_op_update" in no_op.issue_codes


@pytest.mark.parametrize(
    ("catalog_column", "requested_column"),
    (("ß", "ss"), ("ss", "ß")),
)
def test_update_recipe_does_not_unicode_casefold_distinct_sqlite_columns(
    catalog_column: str,
    requested_column: str,
) -> None:
    resource = ResourceSchema(
        resource_id="resource-orders",
        source_id="source-orders",
        name="orders",
        columns=("id", catalog_column),
        revision=REVISION,
        source_revision="schema_version:7",
        resource_kind="table",
        writable=True,
        unique_key_columns=("id",),
        column_declared_types=(("id", "TEXT"), (catalog_column, "TEXT")),
    )

    result = validate_sqlite_update_recipe(
        source_id="source-orders",
        resource_id="resource-orders",
        key_column="ID",
        key_value="order-7",
        target_column=requested_column,
        expected_value="pending",
        new_value="complete",
        resources=(resource,),
        source_write_access=True,
    )

    assert sqlite_identifier_key("ID") == "id"
    assert sqlite_identifier_key(catalog_column) != sqlite_identifier_key(
        requested_column
    )
    assert result.valid is False
    assert result.issue_codes == ("unknown_target_column",)


def test_update_rejects_integer_target_lexical_noop_and_affinity_is_exact() -> None:
    integer_target = replace(
        _resource(),
        column_declared_types=(
            ("id", "TEXT"),
            ("status", "INTEGER"),
            ("email", "TEXT"),
        ),
    )

    result = validate_sqlite_update_recipe(
        source_id="source-orders",
        resource_id="resource-orders",
        key_column="id",
        key_value="order-7",
        target_column="status",
        expected_value="01",
        new_value="1",
        resources=(integer_target,),
        source_write_access=True,
    )

    assert result.valid is False
    assert result.issue_codes == ("target_column_affinity_not_supported",)
    assert result.issues[0].details["target_affinity"] == "integer"
    assert sqlite_declared_type_affinity("VARCHAR(64)") == "text"
    assert sqlite_declared_type_affinity("FLOATING POINT") == "integer"
    assert sqlite_declared_type_affinity("") == "blob"
    assert sqlite_declared_type_affinity("DECIMAL(10, 2)") == "numeric"


class CatalogStoreStub:
    def __init__(
        self,
        registration: SourceRegistration,
        resource: CatalogResource,
        sync: CatalogSync,
        facet: CatalogFacet,
    ) -> None:
        self.registration = registration
        self.resource = resource
        self.sync = sync
        self.facet = facet

    async def record_sync(self, sync):
        raise AssertionError("not used")

    async def commit_snapshot(self, snapshot):
        raise AssertionError("not used")

    async def load_sync(self, agent_id, sync_id):
        if agent_id == "agent-atlas" and sync_id == self.sync.id:
            return self.sync
        return None

    async def load_resource(self, agent_id, resource_id):
        if agent_id == "agent-atlas" and resource_id == self.resource.id:
            return self.resource
        return None

    async def load_revision(self, agent_id, resource_id, revision):
        raise AssertionError("not used")

    async def list_resources(self, agent_id, source_id=None):
        if agent_id == "agent-atlas" and source_id in {None, self.registration.id}:
            return (self.resource,)
        return ()

    async def load_facets(self, agent_id, resource_id, revision=None):
        if agent_id == "agent-atlas" and resource_id == self.resource.id:
            return (self.facet,)
        return ()

    async def load_incident_relationships(
        self, agent_id, resource_id, *, relationship_kinds=(), limit=50
    ):
        return ()

    async def load_relationships(self, agent_id, relationship_ids):
        return ()

    async def search(self, request):
        raise AssertionError("not used")

    async def traverse(self, request):
        raise AssertionError("not used")

    async def load_source(self, agent_id, source_id):
        if agent_id == "agent-atlas" and source_id == self.registration.id:
            return self.registration
        return None

    async def register_source(self, registration):
        self.registration = registration
        return registration

    async def list_sources(self, agent_id):
        if agent_id == "agent-atlas":
            return (self.registration,)
        return ()

    async def detach_source(self, agent_id, source_id, detached_at):
        if agent_id != "agent-atlas" or source_id != self.registration.id:
            raise KeyError(source_id)
        self.registration = self.registration.detach(detached_at)
        return self.registration


async def test_catalog_projects_only_full_single_column_keys_and_write_access() -> None:
    registration = SourceRegistration.build(
        agent_id="agent-atlas",
        adapter_id="sqlite",
        native_identity="/tmp/orders.sqlite3",
        display_name="orders",
        configuration={"path": "/tmp/orders.sqlite3", "write_access": True},
        attached_at=NOW,
    )
    resource_id = catalog_resource_id(
        registration.id,
        ResourceKind.TABLE,
        "main.orders",
    )
    facet = CatalogFacet.from_tabular(
        resource_id=resource_id,
        sync_id="sync-7",
        observed_at=NOW,
        facet=TabularFacet(
            columns=(
                TabularColumn("id", "TEXT", 0, False, primary_key_ordinal=1),
                TabularColumn("email", "TEXT", 1, False),
                TabularColumn("tenant", "TEXT", 2, False),
                TabularColumn("nickname", "TEXT", 3, True),
            ),
            indexes=(
                TabularIndex("email_unique", "btree", ("email",), True),
                TabularIndex(
                    "tenant_email_unique",
                    "btree",
                    ("tenant", "email"),
                    True,
                ),
                TabularIndex(
                    "nickname_partial",
                    "btree",
                    ("nickname",),
                    True,
                    predicate="nickname IS NOT NULL",
                ),
            ),
        ),
    )
    revision = CatalogResourceRevision.build(
        resource_id=resource_id,
        sync_id="sync-7",
        observed_at=NOW,
        facet_revisions=(facet.revision,),
        source_revision="schema_version:7",
    )
    resource = CatalogResource.build(
        agent_id="agent-atlas",
        source_id=registration.id,
        native_identity="main.orders",
        external_uri=f"sqlite://{registration.id}/main/orders",
        kind=ResourceKind.TABLE,
        name="orders",
        sensitivity=Sensitivity.CONFIDENTIAL,
        revision=revision,
        first_observed_at=NOW,
        last_observed_at=NOW,
    )
    sync = CatalogSync(
        id="sync-7",
        agent_id="agent-atlas",
        source_id=registration.id,
        adapter_id="sqlite",
        status=CatalogSyncStatus.SUCCEEDED,
        started_at=NOW,
        completed_at=NOW,
        source_revision="schema_version:7",
        resource_count=1,
    )
    store = CatalogStoreStub(registration, resource, sync, facet)
    view = CatalogDataView(store, CatalogService(store, store), store)

    schemas = await view.resource_schemas("agent-atlas", registration.id)

    assert len(schemas) == 1
    assert schemas[0].resource_kind == "table"
    assert schemas[0].writable is True
    assert schemas[0].sensitivity_class == "confidential"
    assert schemas[0].unique_key_columns == ("id", "email")
    assert schemas[0].column_declared_types == (
        ("id", "TEXT"),
        ("email", "TEXT"),
        ("tenant", "TEXT"),
        ("nickname", "TEXT"),
    )
    assert (
        await view.is_writable_sqlite_source(
            "agent-atlas",
            registration.id,
        )
        is True
    )
    assert await view.source_adapter_id("agent-atlas", registration.id) == "sqlite"
    assert await view.resource_identity("agent-atlas", resource.id) == (
        registration.id,
        "table",
        resource.current_revision,
    )
    assert await view.resource_identity("another-agent", resource.id) is None
    store.registration = replace(
        registration,
        configuration={"path": "/tmp/orders.sqlite3", "write_access": False},
    )
    assert (
        await view.is_writable_sqlite_source(
            "agent-atlas",
            registration.id,
        )
        is False
    )
    store.registration = store.registration.detach(NOW)
    assert await view.resource_identity("agent-atlas", resource.id) is None


class CatalogReader:
    async def source_routing_facts(self, agent_id, configuration_flags):
        assert agent_id == "agent-atlas"
        return (
            FrozenJsonObject.from_mapping(
                {
                    "adapter_id": "sqlite",
                    "configuration_flags": {
                        flag: flag == "write_access" for flag in configuration_flags
                    },
                    "source_id": "source-orders",
                }
            ),
        )

    async def source_adapter_id(self, agent_id, source_id):
        return "sqlite"

    async def resource_identity(self, agent_id, resource_id):
        if agent_id == "agent-atlas" and resource_id == "resource-orders":
            return ("source-orders", "table", REVISION)
        return None

    async def resource_schemas(self, agent_id, source_id):
        if agent_id == "agent-atlas" and source_id == "source-orders":
            return (_resource(),)
        return ()

    async def is_writable_sqlite_source(self, agent_id, source_id):
        return agent_id == "agent-atlas" and source_id == "source-orders"

    async def is_current_tabular_file(self, agent_id, source_id, resource_id):
        return False


class UpdateBackend:
    def __init__(self) -> None:
        self.recipe = _recipe()
        self.impact_calls: list[dict[str, object]] = []
        self.update_calls: list[dict[str, object]] = []

    async def execute_update_impact(self, **arguments):
        self.impact_calls.append(arguments)
        return SQLiteUpdateImpactResult(
            source_id=self.recipe.source_id,
            resource_id=self.recipe.resource_id,
            resource_revision=self.recipe.resource_revision,
            source_revision=self.recipe.source_revision,
            key_column=self.recipe.key_column,
            target_column=self.recipe.target_column,
            recipe_fingerprint=self.recipe.recipe_fingerprint,
            matched_rows=1,
            eligible_rows=1,
            maximum_rows=arguments["maximum_rows"],
        )

    async def execute_update(self, **arguments):
        self.update_calls.append(arguments)
        return SQLiteUpdateResult(
            source_id=self.recipe.source_id,
            resource_id=self.recipe.resource_id,
            resource_revision=self.recipe.resource_revision,
            source_revision=self.recipe.source_revision,
            key_column=self.recipe.key_column,
            target_column=self.recipe.target_column,
            recipe_fingerprint=self.recipe.recipe_fingerprint,
            impact_evidence_id=arguments["impact_evidence_id"],
            affected_rows=1,
            maximum_rows=arguments["maximum_rows"],
        )


class UnicodeScopeBackend(UpdateBackend):
    async def execute_update_impact(self, **arguments):
        self.impact_calls.append(arguments)
        return SQLiteUpdateImpactResult(
            source_id=arguments["source_id"],
            resource_id=arguments["resource_id"],
            resource_revision=REVISION,
            source_revision="schema_version:7",
            key_column=arguments["key_column"],
            target_column="ss",
            recipe_fingerprint=REVISION,
            matched_rows=1,
            eligible_rows=1,
            maximum_rows=arguments["maximum_rows"],
        )

    async def execute_update(self, **arguments):
        self.update_calls.append(arguments)
        return SQLiteUpdateResult(
            source_id=arguments["source_id"],
            resource_id=arguments["resource_id"],
            resource_revision=REVISION,
            source_revision="schema_version:7",
            key_column=arguments["key_column"],
            target_column="ss",
            recipe_fingerprint=REVISION,
            impact_evidence_id=arguments["impact_evidence_id"],
            affected_rows=1,
            maximum_rows=arguments["maximum_rows"],
        )


class KnownNoEffectBackend(UpdateBackend):
    async def execute_update(self, **arguments):
        self.update_calls.append(arguments)
        raise SQLiteUpdateKnownNoEffectError(
            "update_precondition_changed",
            "The approved SQLite update no longer matches one row.",
        )


class AmbiguousFailureBackend(UpdateBackend):
    async def execute_update(self, **arguments):
        self.update_calls.append(arguments)
        raise RuntimeError("ambiguous backend failure")


def _ids():
    counters: dict[str, int] = {}

    def factory(prefix: str) -> str:
        counters[prefix] = counters.get(prefix, 0) + 1
        return f"{prefix}-{counters[prefix]}"

    return factory


async def _commit_tool_call(runtime, registry, operation_id, call):
    turn = await runtime.begin_turn(operation_id)
    request = ModelRequest(
        operation_id=operation_id,
        turn_id=turn.id,
        messages=(
            CanonicalMessage(
                agent_id="agent-atlas",
                operation_id=operation_id,
                turn_id=turn.id,
                role=MessageRole.USER,
                content=(TextBlock("Update order 7 after checking impact."),),
            ),
        ),
        tools=registry.tool_definitions(),
    )
    model_call = await runtime.begin_model_call(
        operation_id,
        turn.id,
        "mock:phase7",
        request,
    )
    await runtime.record_model_response(
        operation_id,
        model_call.id,
        ModelResponse(finish_reason=FinishReason.TOOL_CALLS, tool_calls=(call,)),
        next_phase=LoopPhase.VALIDATING_ACTION,
    )
    return await runtime.inspect(operation_id)


def _arguments(**extra):
    return {
        "source_id": "source-orders",
        "resource_id": "resource-orders",
        "key_column": "id",
        "key_value": "order-7",
        "target_column": "status",
        "expected_value": "pending",
        "new_value": "complete",
        **extra,
    }


async def test_controller_binds_exact_impact_and_write_result_grounds_answer() -> None:
    backend = UpdateBackend()
    declarations = sqlite_update_declarations("agent-atlas", backend)
    registry = CapabilityRegistry(
        capabilities=declarations.capabilities,
        executors=declarations.executors,
        tool_views=declarations.tool_views,
    )
    controller = DataDomainController(registry, CatalogReader(), clock=lambda: NOW)
    runtime = OperationRuntime(
        clock=lambda: NOW,
        id_factory=_ids(),
        capabilities=registry,
    )
    started = await runtime.begin(
        AgentTrigger(
            id="trigger-1",
            agent_id="agent-atlas",
            kind=TriggerKind.USER,
            source_id="user:test",
            payload={"message": "Update order 7 status to complete."},
            created_at=NOW,
        )
    )
    preview_call = ToolCall(
        id="call-preview",
        name=SQLITE_UPDATE_IMPACT_TOOL_NAME,
        arguments=_arguments(),
    )
    preview_snapshot = await _commit_tool_call(
        runtime,
        registry,
        started.operation.id,
        preview_call,
    )
    preview = await controller.validate_action(preview_call, preview_snapshot)
    assert isinstance(preview, ActionProposal)
    assert preview.capability_id == SQLITE_UPDATE_IMPACT_CAPABILITY_ID
    assert preview.validation_facts.schema_version == 1
    impact = await runtime.submit(preview)
    assert impact is not None
    assert impact.kind == SQLITE_UPDATE_IMPACT_EVIDENCE_KIND
    assert set(impact.payload).isdisjoint({"key_value", "expected_value", "new_value"})

    write_call = ToolCall(
        id="call-write",
        name=SQLITE_UPDATE_TOOL_NAME,
        arguments=_arguments(impact_evidence_id=impact.id),
    )
    write_snapshot = await _commit_tool_call(
        runtime,
        registry,
        started.operation.id,
        write_call,
    )
    assert isinstance(impact.payload, FrozenJsonObject)
    tampered_payload = impact.payload.to_dict()
    tampered_payload["eligible_rows"] = 0
    tampered = replace(impact, payload=tampered_payload)
    tampered_snapshot = replace(
        write_snapshot,
        evidence=tuple(
            tampered if evidence.id == impact.id else evidence
            for evidence in write_snapshot.evidence
        ),
    )
    rejected = await controller.validate_action(write_call, tampered_snapshot)
    assert isinstance(rejected, ActionRejection)
    assert rejected.code == "data.sqlite_update.impact_evidence_invalid"

    proposal = await controller.validate_action(write_call, write_snapshot)
    assert isinstance(proposal, ActionProposal)
    assert proposal.capability_id == SQLITE_UPDATE_CAPABILITY_ID
    assert proposal.validation_facts.evidence_ids == (impact.id,)
    assert proposal.validation_facts.impact["evidence_content_hash"] == (
        impact.content_hash
    )
    assert proposal.validation_facts.impact["rollback_available"] is False
    assert await runtime.submit(proposal) is None

    waiting = await runtime.inspect(started.operation.id)
    approval = waiting.approvals[0]
    await runtime.decide_approval(
        approval.id,
        status=ApprovalStatus.APPROVED,
        decided_by="user:test",
        reason="Reviewed the single-row impact.",
    )
    assert await runtime.resume_approval(started.operation.id) is True
    accepted = await runtime.resume_task(started.operation.id, approval.task_id)

    assert accepted is not None
    assert accepted.kind == SQLITE_UPDATE_EVIDENCE_KIND
    assert accepted.payload["affected_rows"] == 1
    assert len(backend.update_calls) == 1
    replay = await runtime.resume_task(started.operation.id, approval.task_id)
    assert replay == accepted
    assert len(backend.update_calls) == 1
    final_snapshot = await runtime.inspect(started.operation.id)
    readiness = await controller.evaluate_final_answer(
        f"Updated one order. [evidence:{accepted.id}]",
        final_snapshot,
    )
    assert readiness.allowed is True


async def test_denied_update_requires_no_application_disclosure_and_impact_citation() -> (
    None
):
    backend = UpdateBackend()
    declarations = sqlite_update_declarations("agent-atlas", backend)
    registry = CapabilityRegistry(
        capabilities=declarations.capabilities,
        executors=declarations.executors,
        tool_views=declarations.tool_views,
    )
    controller = DataDomainController(registry, CatalogReader(), clock=lambda: NOW)
    runtime = OperationRuntime(
        clock=lambda: NOW,
        id_factory=_ids(),
        capabilities=registry,
    )
    started = await runtime.begin(
        AgentTrigger(
            id="trigger-denial",
            agent_id="agent-atlas",
            kind=TriggerKind.USER,
            source_id="user:test",
            payload={"message": "Update order 7 status to complete."},
            created_at=NOW,
        )
    )
    preview_call = ToolCall(
        id="call-denial-preview",
        name=SQLITE_UPDATE_IMPACT_TOOL_NAME,
        arguments=_arguments(),
    )
    preview_snapshot = await _commit_tool_call(
        runtime,
        registry,
        started.operation.id,
        preview_call,
    )
    preview = await controller.validate_action(preview_call, preview_snapshot)
    assert isinstance(preview, ActionProposal)
    impact = await runtime.submit(preview)
    assert impact is not None

    write_call = ToolCall(
        id="call-denial-write",
        name=SQLITE_UPDATE_TOOL_NAME,
        arguments=_arguments(impact_evidence_id=impact.id),
    )
    write_snapshot = await _commit_tool_call(
        runtime,
        registry,
        started.operation.id,
        write_call,
    )
    proposal = await controller.validate_action(write_call, write_snapshot)
    assert isinstance(proposal, ActionProposal)
    assert await runtime.submit(proposal) is None
    waiting = await runtime.inspect(started.operation.id)
    approval = waiting.approvals[0]
    await runtime.decide_approval(
        approval.id,
        status=ApprovalStatus.DENIED,
        decided_by="user:test",
        reason="Do not change this record.",
    )
    assert await runtime.resume_approval(started.operation.id) is True
    denied = await runtime.inspect(started.operation.id)

    missing_citation = await controller.evaluate_final_answer(
        "The update was denied and no change was applied.",
        denied,
    )
    missing_disclosure = await controller.evaluate_final_answer(
        f"Impact was one row. [evidence:{impact.id}]",
        denied,
    )
    ready = await controller.evaluate_final_answer(
        f"The update was denied; no change was applied. [evidence:{impact.id}]",
        denied,
    )

    assert missing_citation.allowed is False
    assert missing_citation.code == "data.response_contract_incomplete"
    assert missing_citation.missing_facts == (
        "a citation to the accepted impact evidence for the denied update",
    )
    assert missing_disclosure.allowed is False
    assert missing_disclosure.code == "data.response_contract_incomplete"
    assert missing_disclosure.missing_facts == (
        "an explicit statement that the update was not applied",
    )
    assert ready.allowed is True
    assert ready.code == "data.response_contract_satisfied"
    assert backend.update_calls == []


def test_update_declarations_publish_risk_effect_and_evidence_contracts() -> None:
    backend = UpdateBackend()
    declarations = sqlite_update_declarations("agent-atlas", backend)
    capabilities = {item.id: item for item in declarations.capabilities}

    assert tuple(view.name for view in declarations.tool_views) == (
        SQLITE_UPDATE_IMPACT_TOOL_NAME,
        SQLITE_UPDATE_TOOL_NAME,
    )
    assert capabilities[SQLITE_UPDATE_IMPACT_CAPABILITY_ID].side_effecting is False
    write = capabilities[SQLITE_UPDATE_CAPABILITY_ID]
    assert write.side_effecting is True
    assert write.idempotent is False
    assert write.replay_safe is False
    assert write.required_evidence_kinds == (SQLITE_UPDATE_IMPACT_EVIDENCE_KIND,)

    request = ExecutionRequest(
        operation_id="operation-test",
        task_id="task-test",
        turn_id="turn-test",
        capability_id=SQLITE_UPDATE_IMPACT_CAPABILITY_ID,
        executor_id=declarations.executors[0].executor_id,
        attempt=1,
        fencing_token=1,
        arguments=_arguments(),
    )
    assert request.arguments["source_id"] == "source-orders"


async def test_update_executors_reject_unicode_casefolded_backend_scope() -> None:
    backend = UnicodeScopeBackend()
    declarations = sqlite_update_declarations("agent-atlas", backend)
    arguments = _arguments(target_column="ß")
    impact_request = ExecutionRequest(
        operation_id="operation-test",
        task_id="task-impact",
        turn_id="turn-test",
        capability_id=SQLITE_UPDATE_IMPACT_CAPABILITY_ID,
        executor_id=declarations.executors[0].executor_id,
        attempt=1,
        fencing_token=1,
        arguments=arguments,
    )
    update_request = ExecutionRequest(
        operation_id="operation-test",
        task_id="task-update",
        turn_id="turn-test",
        capability_id=SQLITE_UPDATE_CAPABILITY_ID,
        executor_id=declarations.executors[1].executor_id,
        attempt=1,
        fencing_token=1,
        arguments={**arguments, "impact_evidence_id": "evidence-impact"},
    )

    with pytest.raises(ValueError, match="different scope"):
        await declarations.executors[0].execute(impact_request)
    with pytest.raises(ValueError, match="different scope"):
        await declarations.executors[1].execute(update_request)


async def test_update_executor_translates_only_known_no_effect_failures() -> None:
    known_backend = KnownNoEffectBackend()
    known_declarations = sqlite_update_declarations("agent-atlas", known_backend)
    request = ExecutionRequest(
        operation_id="operation-test",
        task_id="task-update",
        turn_id="turn-test",
        capability_id=SQLITE_UPDATE_CAPABILITY_ID,
        executor_id=known_declarations.executors[1].executor_id,
        attempt=1,
        fencing_token=1,
        arguments={**_arguments(), "impact_evidence_id": "evidence-impact"},
    )

    with pytest.raises(ExecutorKnownNoEffectError) as known_failure:
        await known_declarations.executors[1].execute(request)
    assert known_failure.value.code == "update_precondition_changed"
    assert isinstance(known_failure.value.__cause__, SQLiteUpdateKnownNoEffectError)

    ambiguous_backend = AmbiguousFailureBackend()
    ambiguous_declarations = sqlite_update_declarations(
        "agent-atlas",
        ambiguous_backend,
    )
    ambiguous_request = replace(
        request,
        executor_id=ambiguous_declarations.executors[1].executor_id,
    )
    with pytest.raises(RuntimeError, match="ambiguous backend failure") as ambiguous:
        await ambiguous_declarations.executors[1].execute(ambiguous_request)
    assert not isinstance(ambiguous.value, ExecutorKnownNoEffectError)
