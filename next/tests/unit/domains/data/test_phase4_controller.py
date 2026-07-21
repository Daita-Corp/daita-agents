from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone

from daita._json import FrozenJsonObject
from daita.capabilities import (
    AccessMode,
    Capability,
    CapabilityRegistry,
    EvidenceCandidate,
    ExecutionRequest,
    RiskLevel,
    ToolApplicability,
    ToolView,
)
from daita.domains.data import (
    LOCAL_FILE_READ_CAPABILITY_ID,
    LOCAL_FILE_READ_EVIDENCE_KIND,
    SQLITE_QUERY_CAPABILITY_ID,
    SQLITE_QUERY_EVIDENCE_KIND,
    TABULAR_COMPARE_CAPABILITY_ID,
    TABULAR_COMPARE_EVIDENCE_KIND,
    DataDomainController,
    ResourceSchema,
    TabularEvidenceDataset,
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
from daita.operations.models import (
    ActionProposal,
    ActionRejection,
    ActionValidationFacts,
    AgentTrigger,
    Evidence,
    TriggerKind,
)
from daita.operations.runtime import OperationRuntime

NOW = datetime(2026, 7, 19, 1, 0, tzinfo=timezone.utc)
FILE_REVISION = "sha256:" + ("a" * 64)
DATABASE_REVISION = "sha256:" + ("b" * 64)


class ComparisonDatasetReader:
    def __init__(self, datasets: tuple[TabularEvidenceDataset, ...]) -> None:
        self._datasets = {dataset.evidence_id: dataset for dataset in datasets}

    async def load_dataset(
        self,
        *,
        operation_id: str,
        evidence_id: str,
    ) -> TabularEvidenceDataset:
        dataset = self._datasets[evidence_id]
        assert dataset.operation_id == operation_id
        return dataset


def _comparison_dataset(
    *,
    operation_id: str,
    evidence_id: str,
    evidence_kind: str,
    source_id: str,
    resource_id: str,
    revision: str,
    rows: tuple[Mapping[str, object], ...] = (
        {"id": 1, "name": "Ada", "status": "active"},
    ),
) -> TabularEvidenceDataset:
    return TabularEvidenceDataset(
        operation_id=operation_id,
        evidence_id=evidence_id,
        evidence_kind=evidence_kind,
        source_id=source_id,
        source_revision=f"revision:{source_id}",
        resource_revisions=((resource_id, revision),),
        columns=("id", "name", "status"),
        rows=rows,
        complete=True,
        truncation_reasons=(),
        row_limit=100,
        byte_limit=65_536,
    )


class CatalogReader:
    def __init__(self) -> None:
        self.adapter_overrides: dict[str, str | None] = {}
        self.identity_overrides: dict[str, tuple[str, str, str] | None] = {}
        self.routing_sources: tuple[tuple[str, str, Mapping[str, object]], ...] = (
            ("source-files", "local-directory", {}),
            ("source-database", "sqlite", {"write_access": False}),
        )

    async def source_routing_facts(
        self,
        agent_id: str,
        configuration_flags: tuple[str, ...],
    ) -> tuple[FrozenJsonObject, ...]:
        assert agent_id == "agent-atlas"
        return tuple(
            FrozenJsonObject.from_mapping(
                {
                    "adapter_id": adapter_id,
                    "configuration_flags": {
                        flag: configuration.get(flag) is True
                        for flag in configuration_flags
                    },
                    "source_id": source_id,
                }
            )
            for source_id, adapter_id, configuration in self.routing_sources
        )

    async def source_adapter_id(self, agent_id: str, source_id: str) -> str | None:
        if source_id in self.adapter_overrides:
            return self.adapter_overrides[source_id]
        return "local-directory" if source_id == "source-files" else "sqlite"

    async def resource_identity(
        self,
        agent_id: str,
        resource_id: str,
    ) -> tuple[str, str, str] | None:
        assert agent_id == "agent-atlas"
        if resource_id in self.identity_overrides:
            return self.identity_overrides[resource_id]
        identities = {
            "resource-export": ("source-files", "file", FILE_REVISION),
            "resource-customers": (
                "source-database",
                "table",
                DATABASE_REVISION,
            ),
        }
        return identities.get(resource_id)

    async def resource_schemas(
        self,
        agent_id: str,
        source_id: str,
    ) -> tuple[ResourceSchema, ...]:
        assert agent_id == "agent-atlas"
        if source_id == "source-files":
            return (
                ResourceSchema(
                    resource_id="resource-export",
                    source_id=source_id,
                    name="customers.csv",
                    aliases=("customers.csv",),
                    columns=("id", "name", "status"),
                    revision=FILE_REVISION,
                    source_revision="directory-revision-1",
                    sensitivity_class="internal",
                ),
            )
        if source_id == "source-database":
            return (
                ResourceSchema(
                    resource_id="resource-customers",
                    source_id=source_id,
                    name="customers",
                    aliases=("main.customers",),
                    columns=("id", "name", "status"),
                    revision=DATABASE_REVISION,
                    source_revision="database-revision-1",
                    sensitivity_class="confidential",
                ),
            )
        return ()

    async def is_current_tabular_file(
        self,
        agent_id: str,
        source_id: str,
        resource_id: str,
    ) -> bool:
        assert agent_id == "agent-atlas"
        return source_id == "source-files" and resource_id == "resource-export"

    async def is_writable_sqlite_source(
        self,
        agent_id: str,
        source_id: str,
    ) -> bool:
        return False


class CandidateExecutor:
    def __init__(
        self,
        executor_id: str,
        evidence_kind: str,
        payload: Mapping[str, object] | None = None,
    ) -> None:
        self.executor_id = executor_id
        self.evidence_kind = evidence_kind
        self.payload = payload
        self.requests: list[ExecutionRequest] = []

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        self.requests.append(request)
        payload = self.payload
        if payload is None:
            payload = {
                "left": {
                    "evidence_id": request.arguments["left_evidence_id"],
                    "source_id": "source-files",
                },
                "right": {
                    "evidence_id": request.arguments["right_evidence_id"],
                    "source_id": "source-database",
                },
                "truncated": bool(request.arguments.get("test_truncated", False)),
            }
        return EvidenceCandidate(
            kind=self.evidence_kind,
            schema_version=1,
            payload=payload,
        )


def _capability(
    capability_id: str,
    executor_id: str,
    evidence_kind: str,
    input_properties: Mapping[str, str],
    output_properties: Mapping[str, str],
) -> Capability:
    return Capability(
        id=capability_id,
        owner="test-data",
        description=f"Test contract for {capability_id}.",
        input_schema={
            "type": "object",
            "properties": {
                name: {"type": kind} for name, kind in input_properties.items()
            },
            "required": list(input_properties),
            "additionalProperties": False,
        },
        output_evidence_kind=evidence_kind,
        output_schema_version=1,
        output_schema={
            "type": "object",
            "properties": {
                name: {"type": kind} for name, kind in output_properties.items()
            },
            "required": list(output_properties),
            "additionalProperties": False,
        },
        executor_id=executor_id,
        access_mode=AccessMode.READ,
        risk=RiskLevel.LOW,
        side_effecting=False,
        idempotent=True,
        replay_safe=True,
    )


def _registry(
    *,
    file_source: str,
    query_source: str,
) -> tuple[CapabilityRegistry, tuple[CandidateExecutor, ...]]:
    file_executor = CandidateExecutor(
        "test.file.executor",
        LOCAL_FILE_READ_EVIDENCE_KIND,
        {
            "source_id": file_source,
            "resource_id": "resource-export",
            "truncated": False,
        },
    )
    query_executor = CandidateExecutor(
        "test.query.executor",
        SQLITE_QUERY_EVIDENCE_KIND,
        {
            "source_id": query_source,
            "resource_id": "resource-customers",
            "truncated": False,
        },
    )
    comparison_executor = CandidateExecutor(
        "test.compare.executor",
        TABULAR_COMPARE_EVIDENCE_KIND,
    )
    capabilities = (
        _capability(
            LOCAL_FILE_READ_CAPABILITY_ID,
            file_executor.executor_id,
            LOCAL_FILE_READ_EVIDENCE_KIND,
            {"source_id": "string", "resource_id": "string"},
            {
                "source_id": "string",
                "resource_id": "string",
                "truncated": "boolean",
            },
        ),
        _capability(
            SQLITE_QUERY_CAPABILITY_ID,
            query_executor.executor_id,
            SQLITE_QUERY_EVIDENCE_KIND,
            {"source_id": "string"},
            {
                "source_id": "string",
                "resource_id": "string",
                "truncated": "boolean",
            },
        ),
        _capability(
            TABULAR_COMPARE_CAPABILITY_ID,
            comparison_executor.executor_id,
            TABULAR_COMPARE_EVIDENCE_KIND,
            {
                "left_evidence_id": "string",
                "right_evidence_id": "string",
                "key_columns": "array",
                "compare_columns": "array",
                "key_normalization": "string",
                "test_truncated": "boolean",
            },
            {"left": "object", "right": "object", "truncated": "boolean"},
        ),
    )
    executors = (file_executor, query_executor, comparison_executor)
    return (
        CapabilityRegistry(
            capabilities=capabilities,
            executors=executors,
            tool_views=tuple(
                ToolView(
                    name=name,
                    capability_id=capability.id,
                    description=capability.description,
                    applicability=ToolApplicability(
                        source_adapter_ids=(
                            ("local-directory",)
                            if capability.id == LOCAL_FILE_READ_CAPABILITY_ID
                            else (
                                ("sqlite",)
                                if capability.id == SQLITE_QUERY_CAPABILITY_ID
                                else ()
                            )
                        ),
                        minimum_active_sources=(
                            2 if capability.id == TABULAR_COMPARE_CAPABILITY_ID else 1
                        ),
                    ),
                )
                for name, capability in zip(
                    ("data_read_file", "data_query_sqlite", "data_compare_tabular"),
                    capabilities,
                    strict=True,
                )
            ),
        ),
        executors,
    )


def _ids(label: str):
    counters: dict[str, int] = {}

    def factory(prefix: str) -> str:
        counters[prefix] = counters.get(prefix, 0) + 1
        return f"{prefix}-{label}-{counters[prefix]}"

    return factory


async def _new_operation_snapshot(
    registry: CapabilityRegistry,
    *,
    label: str,
    kind: TriggerKind = TriggerKind.USER,
):
    runtime = OperationRuntime(
        clock=lambda: NOW,
        id_factory=_ids(label),
        capabilities=registry,
    )
    payload: dict[str, object] = {"message": "Inspect available data."}
    if kind is TriggerKind.MONITOR:
        payload["monitor_scope"] = {
            "resource_ids": [],
            "source_ids": ["source-database"],
        }
    return await runtime.begin(
        AgentTrigger(
            id=f"trigger-{label}",
            agent_id="agent-atlas",
            kind=kind,
            source_id="user:test" if kind is TriggerKind.USER else "monitor:test",
            payload=payload,
            created_at=NOW,
        )
    )


async def _snapshot_with_reads(
    *,
    label: str,
    file_source: str = "source-files",
    query_source: str = "source-database",
    with_comparison: bool = False,
    comparison_truncated: bool = False,
    message: str = "Compare the newest export for discrepancies.",
    catalog: CatalogReader | None = None,
    comparison_left_rows: tuple[Mapping[str, object], ...] = (
        {"id": 1, "name": "Ada", "status": "active"},
    ),
    comparison_right_rows: tuple[Mapping[str, object], ...] = (
        {"id": 1, "name": "Ada", "status": "active"},
    ),
) -> tuple[DataDomainController, OperationRuntime, tuple[Evidence, ...]]:
    registry, _ = _registry(file_source=file_source, query_source=query_source)
    operation_id = f"operation-{label}-1"
    comparison_datasets = ComparisonDatasetReader(
        (
            _comparison_dataset(
                operation_id=operation_id,
                evidence_id=f"evidence-{label}-1",
                evidence_kind=LOCAL_FILE_READ_EVIDENCE_KIND,
                source_id=file_source,
                resource_id="resource-export",
                revision=FILE_REVISION,
                rows=comparison_left_rows,
            ),
            _comparison_dataset(
                operation_id=operation_id,
                evidence_id=f"evidence-{label}-2",
                evidence_kind=SQLITE_QUERY_EVIDENCE_KIND,
                source_id=query_source,
                resource_id="resource-customers",
                revision=DATABASE_REVISION,
                rows=comparison_right_rows,
            ),
        )
    )
    controller = DataDomainController(
        registry,
        CatalogReader() if catalog is None else catalog,
        comparison_datasets=comparison_datasets,
        clock=lambda: NOW,
    )
    runtime = OperationRuntime(
        clock=lambda: NOW,
        id_factory=_ids(label),
        capabilities=registry,
    )
    snapshot = await runtime.begin(
        AgentTrigger(
            id=f"trigger-{label}",
            agent_id="agent-atlas",
            kind=TriggerKind.USER,
            source_id="user:test",
            payload={"message": message},
            created_at=NOW,
        )
    )
    turn = await runtime.begin_turn(snapshot.operation.id)
    calls = [
        ToolCall(
            id=f"call-{label}-file",
            name="data_read_file",
            arguments={
                "source_id": file_source,
                "resource_id": "resource-export",
            },
        ),
        ToolCall(
            id=f"call-{label}-query",
            name="data_query_sqlite",
            arguments={"source_id": query_source},
        ),
    ]
    if with_comparison:
        calls.append(
            ToolCall(
                id=f"call-{label}-compare",
                name="data_compare_tabular",
                arguments={
                    "left_evidence_id": f"evidence-{label}-1",
                    "right_evidence_id": f"evidence-{label}-2",
                    "key_columns": ["id"],
                    "compare_columns": ["name", "status"],
                    "key_normalization": "strict",
                    "test_truncated": comparison_truncated,
                },
            )
        )
    request = ModelRequest(
        operation_id=snapshot.operation.id,
        turn_id=turn.id,
        messages=(
            CanonicalMessage(
                agent_id="agent-atlas",
                operation_id=snapshot.operation.id,
                turn_id=turn.id,
                role=MessageRole.USER,
                content=(TextBlock("Compare the newest export."),),
            ),
        ),
        tools=registry.tool_definitions(),
    )
    model_call = await runtime.begin_model_call(
        snapshot.operation.id,
        turn.id,
        "mock:phase4-controller",
        request,
    )
    await runtime.record_model_response(
        snapshot.operation.id,
        model_call.id,
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=tuple(calls),
        ),
        next_phase=LoopPhase.VALIDATING_ACTION,
    )
    evidence: list[Evidence] = []
    for call in calls:
        _, capability = registry.resolve_tool(call.name)
        current = await runtime.inspect(snapshot.operation.id)
        if capability.id == TABULAR_COMPARE_CAPABILITY_ID:
            validated = await controller.validate_action(call, current)
            assert isinstance(validated, ActionProposal)
            proposal = validated
        else:
            source_id = call.arguments["source_id"]
            assert isinstance(source_id, str)
            is_file = capability.id == LOCAL_FILE_READ_CAPABILITY_ID
            resource_id = "resource-export" if is_file else "resource-customers"
            resource_revision = FILE_REVISION if is_file else DATABASE_REVISION
            source_revision = (
                "directory-revision-1" if is_file else "database-revision-1"
            )
            proposal = ActionProposal(
                operation_id=snapshot.operation.id,
                turn_id=turn.id,
                call_id=call.id,
                capability_id=capability.id,
                arguments=call.arguments,
                proposed_at=NOW,
                validation_facts=ActionValidationFacts(
                    schema_version=1,
                    sensitivity_class="internal" if is_file else "confidential",
                    source_id=source_id,
                    resource_ids=(resource_id,),
                    resource_revisions=((resource_id, resource_revision),),
                    source_revision=source_revision,
                    freshness_state="current",
                ),
            )
        accepted = await runtime.submit(proposal)
        assert accepted is not None
        evidence.append(accepted)
        await runtime.append_observation(await controller.project_observation(accepted))
    return controller, runtime, tuple(evidence)


def _comparison_call(
    label: str,
    left_id: str,
    right_id: str,
    *,
    key_normalization: str = "strict",
) -> ToolCall:
    return ToolCall(
        id=f"call-{label}-comparison-proposal",
        name="data_compare_tabular",
        arguments={
            "left_evidence_id": left_id,
            "right_evidence_id": right_id,
            "key_columns": ["id"],
            "compare_columns": ["name", "status"],
            "key_normalization": key_normalization,
            "test_truncated": False,
        },
    )


async def test_tool_projection_uses_arbitrary_declared_adapter_without_controller_code() -> (
    None
):
    base, _ = _registry(
        file_source="source-files",
        query_source="source-database",
    )
    executor = CandidateExecutor("test.warehouse.executor", "test.warehouse.result")
    capability = _capability(
        "test.warehouse.lookup",
        executor.executor_id,
        executor.evidence_kind,
        {"query": "string"},
        {"value": "string"},
    )
    registry = CapabilityRegistry.compose(
        base,
        CapabilityRegistry(
            capabilities=(capability,),
            executors=(executor,),
            tool_views=(
                ToolView(
                    name="warehouse_lookup",
                    capability_id=capability.id,
                    description=capability.description,
                    applicability=ToolApplicability(
                        source_adapter_ids=("warehouse-v2",),
                        minimum_active_sources=1,
                    ),
                ),
            ),
        ),
    )
    catalog = CatalogReader()
    catalog.routing_sources = (("source-warehouse", "warehouse-v2", {}),)
    controller = DataDomainController(registry, catalog, clock=lambda: NOW)
    snapshot = await _new_operation_snapshot(registry, label="warehouse")

    assert "warehouse_lookup" in {
        tool.name for tool in await controller.tool_views(snapshot)
    }
    catalog.routing_sources = (("source-warehouse", "unrelated-adapter", {}),)
    assert "warehouse_lookup" not in {
        tool.name for tool in await controller.tool_views(snapshot)
    }


async def test_tool_projection_requires_exact_true_configuration_flag() -> None:
    base, _ = _registry(
        file_source="source-files",
        query_source="source-database",
    )
    executor = CandidateExecutor("test.write.executor", "test.write.result")
    capability = _capability(
        "test.sqlite.write",
        executor.executor_id,
        executor.evidence_kind,
        {"value": "string"},
        {"value": "string"},
    )
    registry = CapabilityRegistry.compose(
        base,
        CapabilityRegistry(
            capabilities=(capability,),
            executors=(executor,),
            tool_views=(
                ToolView(
                    name="sqlite_write",
                    capability_id=capability.id,
                    description=capability.description,
                    applicability=ToolApplicability(
                        source_adapter_ids=("sqlite",),
                        minimum_active_sources=1,
                        required_configuration_flags=("write_access",),
                    ),
                ),
            ),
        ),
    )
    catalog = CatalogReader()
    controller = DataDomainController(registry, catalog, clock=lambda: NOW)
    snapshot = await _new_operation_snapshot(registry, label="write-flag")
    catalog.routing_sources = (("source-database", "sqlite", {"write_access": "true"}),)

    assert "sqlite_write" not in {
        tool.name for tool in await controller.tool_views(snapshot)
    }
    catalog.routing_sources = (("source-database", "sqlite", {"write_access": True}),)
    assert "sqlite_write" in {
        tool.name for tool in await controller.tool_views(snapshot)
    }


async def test_detached_source_removes_its_tool_views_on_next_projection() -> None:
    registry, _ = _registry(
        file_source="source-files",
        query_source="source-database",
    )
    catalog = CatalogReader()
    controller = DataDomainController(registry, catalog, clock=lambda: NOW)
    snapshot = await _new_operation_snapshot(registry, label="detach")

    assert {tool.name for tool in await controller.tool_views(snapshot)} == {
        "data_compare_tabular",
        "data_query_sqlite",
        "data_read_file",
    }
    catalog.routing_sources = (("source-files", "local-directory", {}),)
    assert {tool.name for tool in await controller.tool_views(snapshot)} == {
        "data_read_file",
    }


async def test_monitor_projection_intersects_applicability_with_monitor_scope() -> None:
    base, _ = _registry(
        file_source="source-files",
        query_source="source-database",
    )
    executor = CandidateExecutor("test.monitor-hidden.executor", "test.hidden.result")
    capability = _capability(
        "test.monitor-hidden",
        executor.executor_id,
        executor.evidence_kind,
        {"value": "string"},
        {"value": "string"},
    )
    registry = CapabilityRegistry.compose(
        base,
        CapabilityRegistry(
            capabilities=(capability,),
            executors=(executor,),
            tool_views=(
                ToolView(
                    name="applicable_but_not_monitor_scoped",
                    capability_id=capability.id,
                    description=capability.description,
                    applicability=ToolApplicability(
                        source_adapter_ids=("sqlite",),
                        minimum_active_sources=1,
                    ),
                ),
            ),
        ),
    )
    catalog = CatalogReader()
    controller = DataDomainController(registry, catalog, clock=lambda: NOW)
    user_snapshot = await _new_operation_snapshot(registry, label="user-tools")
    monitor_snapshot = await _new_operation_snapshot(
        registry,
        label="monitor-tools",
        kind=TriggerKind.MONITOR,
    )

    assert "applicable_but_not_monitor_scoped" in {
        tool.name for tool in await controller.tool_views(user_snapshot)
    }
    assert "applicable_but_not_monitor_scoped" not in {
        tool.name for tool in await controller.tool_views(monitor_snapshot)
    }


async def test_file_resource_scope_rejects_before_action_proposal() -> None:
    controller, runtime, _ = await _snapshot_with_reads(label="file-scope")
    snapshot = await runtime.inspect("operation-file-scope-1")
    task_count = len(snapshot.tasks)
    call = ToolCall(
        id="call-file-escape",
        name="data_read_file",
        arguments={
            "source_id": "source-files",
            "resource_id": "resource-outside-root",
        },
    )

    result = await controller.validate_action(call, snapshot)

    assert isinstance(result, ActionRejection)
    assert result.code == "data.file.resource_not_found"
    assert len((await runtime.inspect(snapshot.operation.id)).tasks) == task_count


async def test_capability_input_repair_facts_survive_controller_validation() -> None:
    controller, runtime, _ = await _snapshot_with_reads(label="typed-input")
    snapshot = await runtime.inspect("operation-typed-input-1")

    result = await controller.validate_action(
        ToolCall(
            id="call-typed-input-repair",
            name="data_read_file",
            arguments={"source_id": "source-files"},
        ),
        snapshot,
    )

    assert isinstance(result, ActionRejection)
    assert result.code == "capability.input.missing_required_fields"
    assert result.details["tool_name"] == "data_read_file"
    assert result.details["capability_id"] == LOCAL_FILE_READ_CAPABILITY_ID
    assert result.details["missing_fields"] == ("resource_id",)
    assert result.details["allowed_fields"] == ("resource_id", "source_id")
    assert result.details["details_truncated"] is False


async def test_file_validation_distinguishes_revision_source_kind_and_adapter() -> None:
    catalog = CatalogReader()
    controller, runtime, _ = await _snapshot_with_reads(
        label="file-repair",
        catalog=catalog,
    )
    snapshot = await runtime.inspect("operation-file-repair-1")

    revision = await controller.validate_action(
        ToolCall(
            id="call-file-revision",
            name="data_read_file",
            arguments={
                "source_id": "source-files",
                "resource_id": FILE_REVISION,
            },
        ),
        snapshot,
    )
    assert isinstance(revision, ActionRejection)
    assert revision.code == "data.file.revision_used_as_resource_id"
    assert revision.details["resource_id"] == "resource-export"

    catalog.identity_overrides["resource-export"] = (
        "source-other-files",
        "file",
        FILE_REVISION,
    )
    wrong_source = await controller.validate_action(
        ToolCall(
            id="call-file-wrong-source",
            name="data_read_file",
            arguments={
                "source_id": "source-files",
                "resource_id": "resource-export",
            },
        ),
        snapshot,
    )
    assert isinstance(wrong_source, ActionRejection)
    assert wrong_source.code == "data.file.wrong_source"
    assert wrong_source.details["actual_source_id"] == "source-other-files"

    catalog.identity_overrides["resource-export"] = (
        "source-files",
        "table",
        FILE_REVISION,
    )
    wrong_kind = await controller.validate_action(
        ToolCall(
            id="call-file-wrong-kind",
            name="data_read_file",
            arguments={
                "source_id": "source-files",
                "resource_id": "resource-export",
            },
        ),
        snapshot,
    )
    assert isinstance(wrong_kind, ActionRejection)
    assert wrong_kind.code == "data.file.resource_kind_mismatch"
    assert wrong_kind.details["resource_kind"] == "table"

    catalog.identity_overrides.clear()
    catalog.adapter_overrides["source-files"] = "sqlite"
    wrong_adapter = await controller.validate_action(
        ToolCall(
            id="call-file-wrong-adapter",
            name="data_read_file",
            arguments={
                "source_id": "source-files",
                "resource_id": "resource-export",
            },
        ),
        snapshot,
    )
    assert isinstance(wrong_adapter, ActionRejection)
    assert wrong_adapter.code == "data.file.source_tool_not_applicable"
    assert wrong_adapter.details["source_adapter_id"] == "sqlite"
    assert wrong_adapter.details["selected_tool_name"] == "data_read_file"
    assert wrong_adapter.details["selected_capability_id"] == (
        LOCAL_FILE_READ_CAPABILITY_ID
    )
    assert wrong_adapter.details["declared_applicable_adapter_ids"] == (
        "local-directory",
    )


async def test_current_file_read_carries_exact_catalog_authority() -> None:
    controller, runtime, _ = await _snapshot_with_reads(label="file-authority")
    snapshot = await runtime.inspect("operation-file-authority-1")
    call = ToolCall(
        id="call-file-authority",
        name="data_read_file",
        arguments={
            "source_id": "source-files",
            "resource_id": "resource-export",
        },
    )

    result = await controller.validate_action(call, snapshot)

    assert isinstance(result, ActionProposal)
    assert result.validation_facts == ActionValidationFacts(
        schema_version=1,
        sensitivity_class="internal",
        source_id="source-files",
        resource_ids=("resource-export",),
        resource_revisions=(("resource-export", FILE_REVISION),),
        source_revision="directory-revision-1",
        freshness_state="current",
    )


async def test_comparison_rejects_missing_foreign_and_same_source_evidence() -> None:
    controller, runtime, evidence = await _snapshot_with_reads(label="main")
    snapshot = await runtime.inspect("operation-main-1")
    missing = await controller.validate_action(
        _comparison_call("missing", "evidence-missing", evidence[1].id),
        snapshot,
    )

    _, _, foreign_evidence = await _snapshot_with_reads(label="foreign")
    foreign = await controller.validate_action(
        _comparison_call("foreign", foreign_evidence[0].id, evidence[1].id),
        snapshot,
    )

    same_controller, same_runtime, same_evidence = await _snapshot_with_reads(
        label="same",
        query_source="source-files",
    )
    same = await same_controller.validate_action(
        _comparison_call("same", same_evidence[0].id, same_evidence[1].id),
        await same_runtime.inspect("operation-same-1"),
    )

    assert isinstance(missing, ActionRejection)
    assert missing.code == "data.compare.evidence_unavailable"
    assert isinstance(foreign, ActionRejection)
    assert foreign.code == "data.compare.evidence_unavailable"
    assert isinstance(same, ActionRejection)
    assert same.code == "data.compare.sources_not_distinct"


async def test_valid_distinct_source_evidence_becomes_comparison_proposal() -> None:
    controller, runtime, evidence = await _snapshot_with_reads(label="valid")
    call = _comparison_call("valid", evidence[0].id, evidence[1].id)

    result = await controller.validate_action(
        call,
        await runtime.inspect("operation-valid-1"),
    )

    assert isinstance(result, ActionProposal)
    assert result.capability_id == TABULAR_COMPARE_CAPABILITY_ID
    assert result.arguments["left_evidence_id"] == evidence[0].id
    assert result.arguments["right_evidence_id"] == evidence[1].id
    assert result.validation_facts.source_ids == (
        "source-database",
        "source-files",
    )
    assert result.validation_facts.source_revisions == (
        ("source-database", "database-revision-1"),
        ("source-files", "directory-revision-1"),
    )
    assert result.validation_facts.resource_ids == (
        "resource-customers",
        "resource-export",
    )
    assert result.validation_facts.evidence_ids == (
        evidence[0].id,
        evidence[1].id,
    )
    assert result.validation_facts.sensitivity_class == "confidential"
    assert result.validation_facts.freshness_state == "current"


async def test_comparison_preflight_rejections_create_no_task_or_evidence() -> None:
    controller, runtime, evidence = await _snapshot_with_reads(
        label="incompatible-keys",
        comparison_left_rows=({"id": "1", "name": "Ada", "status": "active"},),
        comparison_right_rows=({"id": 1, "name": "Ada", "status": "active"},),
    )
    before = await runtime.inspect("operation-incompatible-keys-1")

    rejected = await controller.validate_action(
        _comparison_call(
            "incompatible-keys",
            evidence[0].id,
            evidence[1].id,
            key_normalization="strict",
        ),
        before,
    )
    after = await runtime.inspect(before.operation.id)
    readiness = await controller.evaluate_final_answer(
        (
            "The sources have discrepancies. "
            f"[evidence:{evidence[0].id}] [evidence:{evidence[1].id}]"
        ),
        after,
    )

    assert isinstance(rejected, ActionRejection)
    assert rejected.code == "data.compare.incompatible_key_types"
    assert rejected.details["left_type_domain"] == ("string",)
    assert rejected.details["right_type_domain"] == ("integer",)
    assert after.tasks == before.tasks
    assert after.evidence == before.evidence
    assert readiness.allowed is False
    assert readiness.code == "data.response_contract_incomplete"
    assert readiness.missing_facts == (
        "accepted current-operation comparison evidence",
    )


async def test_comparison_collision_preflight_creates_no_task_or_evidence() -> None:
    controller, runtime, evidence = await _snapshot_with_reads(
        label="collision-keys",
        comparison_left_rows=(
            {"id": "1", "name": "Ada", "status": "active"},
            {"id": 1, "name": "Ada", "status": "active"},
        ),
        comparison_right_rows=({"id": 1, "name": "Ada", "status": "active"},),
    )
    before = await runtime.inspect("operation-collision-keys-1")

    rejected = await controller.validate_action(
        _comparison_call(
            "collision-keys",
            evidence[0].id,
            evidence[1].id,
            key_normalization="stringify_integral",
        ),
        before,
    )
    after = await runtime.inspect(before.operation.id)

    assert isinstance(rejected, ActionRejection)
    assert rejected.code == "data.compare.normalization_collision"
    assert rejected.details["side"] == "left"
    assert rejected.details["row_indexes"] == (0, 1)
    assert after.tasks == before.tasks
    assert after.evidence == before.evidence


async def test_readiness_requires_comparison_and_both_input_citations() -> None:
    controller, runtime, evidence = await _snapshot_with_reads(
        label="ready",
        with_comparison=True,
    )
    snapshot = await runtime.inspect("operation-ready-1")
    left, right, comparison = evidence

    no_comparison = await controller.evaluate_final_answer(
        f"Different. [evidence:{left.id}] [evidence:{right.id}]",
        snapshot,
    )
    missing_input = await controller.evaluate_final_answer(
        f"Different. [evidence:{comparison.id}] [evidence:{left.id}]",
        snapshot,
    )
    ready = await controller.evaluate_final_answer(
        (
            f"Different. [evidence:{comparison.id}] "
            f"[evidence:{left.id}] [evidence:{right.id}]"
        ),
        snapshot,
    )

    assert no_comparison.allowed is False
    assert no_comparison.code == "data.response_contract_incomplete"
    assert FrozenJsonObject.from_mapping(no_comparison.repair_details).to_dict() == {
        "required_citations": [
            {
                "citation": f"[evidence:{comparison.id}]",
                "evidence_id": comparison.id,
            },
            {
                "citation": f"[evidence:{left.id}]",
                "evidence_id": left.id,
            },
            {
                "citation": f"[evidence:{right.id}]",
                "evidence_id": right.id,
            },
        ]
    }
    assert missing_input.allowed is False
    assert missing_input.code == "data.response_contract_incomplete"
    assert missing_input.repair_details == no_comparison.repair_details
    assert ready.allowed is True
    assert ready.code == "data.response_contract_satisfied"


async def test_truncated_comparison_requires_explicit_partial_disclosure() -> None:
    controller, runtime, evidence = await _snapshot_with_reads(
        label="partial",
        with_comparison=True,
        comparison_truncated=True,
    )
    snapshot = await runtime.inspect("operation-partial-1")
    left, right, comparison = evidence
    citations = f"[evidence:{comparison.id}] [evidence:{left.id}] [evidence:{right.id}]"

    undisclosed = await controller.evaluate_final_answer(
        f"The sources differ. {citations}",
        snapshot,
    )
    disclosed = await controller.evaluate_final_answer(
        f"This is a partial comparison with limited coverage. {citations}",
        snapshot,
    )
    negated = await controller.evaluate_final_answer(
        f"This comparison is not partial; it is complete. {citations}",
        snapshot,
    )

    assert undisclosed.allowed is False
    assert undisclosed.missing_facts == (
        "an explicit partial or truncation disclosure",
    )
    assert negated.allowed is False
    assert negated.missing_facts == ("an explicit partial or truncation disclosure",)
    assert disclosed.allowed is True
    assert disclosed.code == "data.response_contract_satisfied"


async def test_reconciliation_language_requires_comparison_evidence() -> None:
    controller, runtime, evidence = await _snapshot_with_reads(
        label="reconcile",
        message="Reconcile the export against the customer table.",
    )
    snapshot = await runtime.inspect("operation-reconcile-1")

    readiness = await controller.evaluate_final_answer(
        f"They reconcile. [evidence:{evidence[0].id}] [evidence:{evidence[1].id}]",
        snapshot,
    )

    assert readiness.allowed is False
    assert readiness.code == "data.response_contract_incomplete"
    assert readiness.missing_facts == (
        "accepted current-operation comparison evidence",
    )
