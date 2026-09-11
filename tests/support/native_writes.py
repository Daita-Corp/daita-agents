"""Shared helpers extracted from ``test_public_writes.py``."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

from daita import (
    Agent,
    MCPToolSelection,
)
from daita.adapters import postgresql as pg, postgresql_write as native
from daita.adapters.mcp import StreamableHTTPMCPClientFactory
from daita.adapters.models import DiscoveryRequest, SourceRegistration
from daita.capabilities import (
    ApprovalDecision,
)
from daita.catalog.models import ResourceKind, Sensitivity, TabularColumn, TabularIndex
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    ModelUsage,
)
from daita.llm.pricing import CostEstimate
from tests.data.writes._upsert_support import Database
from tests.support.mcp import (
    conformance_identities,
    mock_transport,
)
from tests.support.workspace import workspace_for

NOW = datetime(2026, 9, 6, 12, tzinfo=UTC)


class ScriptedResearchModel:
    provider_id = "mock:native-acceptance"
    model_profile = ModelProfile(
        id=provider_id,
        context_window_tokens=64000,
        max_output_tokens=2000,
        supports_tools=True,
        supports_parallel_tools=True,
    )

    def __init__(self):
        self.steps: list[ModelResponse | Callable[[ModelRequest], ModelResponse]] = []
        self.requests: list[ModelRequest] = []

    def replace_script(
        self,
        steps: Iterable[ModelResponse | Callable[[ModelRequest], ModelResponse]],
    ) -> None:
        self.steps = list(steps)

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return True

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        return True

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        assert self.steps, "script exhausted"
        step = self.steps.pop(0)
        return step(request) if callable(step) else step


def response(*calls, text=None):
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS if calls else FinishReason.STOP,
        tool_calls=tuple(calls),
        text=text,
        usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0"))),
    )


async def create_fixture(tmp_path, monkeypatch, *, sensitivity=Sensitivity.RESTRICTED):
    clock = [NOW]
    db = Database()
    alpha, _ = conformance_identities()
    alpha.results["lookup"] = {
        "content": [
            {
                "type": "text",
                "text": "New Co lists Acme as a customer. https://source.test/new; one source, coverage is incomplete.",
            }
        ],
        "structuredContent": {
            "answer": "New Co; https://source.test/new; model must assess the claim"
        },
        "isError": False,
    }
    factory = StreamableHTTPMCPClientFactory(http_transport=mock_transport(alpha))
    provider = ScriptedResearchModel()
    approvals = []

    async def approve(request):
        approvals.append(request)
        return ApprovalDecision.APPROVE

    agent = await Agent.create(
        "native-acceptance",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=provider.model_profile,
        clock=lambda: clock[0],
        mcp_client_factory=factory,
        approval_handler=approve,
    )
    registration = SourceRegistration.build(
        agent_id=agent.id,
        adapter_id="postgresql",
        native_identity="postgresql:companies",
        display_name="Company research",
        configuration={
            "database": "research",
            "host": "db.fixture.test",
            "port": 5432,
            "schemas": ("public",),
            "ssl_mode": "require",
            "username": "writer",
        },
        attached_at=NOW,
    )
    columns = tuple(
        TabularColumn(
            name=name,
            native_type="bigint" if name == "id" else "text",
            ordinal=i,
            nullable=name == "notes",
            native_type_namespace="pg_catalog",
            native_type_name="int8" if name == "id" else "text",
            primary_key_ordinal=1 if name == "id" else None,
            identity=name == "id",
            updatable=True,
            collation="C" if name != "id" else None,
        )
        for i, name in enumerate(("id", "domain", "name", "evidence_url", "notes"))
    )
    table = pg._TableStructure(
        "public",
        "companies",
        ResourceKind.TABLE,
        columns,
        (
            TabularIndex(
                "companies_domain_key",
                "btree",
                ("domain",),
                True,
                write_conflict_supported=True,
            ),
            TabularIndex(
                "companies_pkey", "btree", ("id",), True, write_conflict_supported=True
            ),
        ),
    )
    structure = pg.PostgreSQLStructure((table,), (), db.structure_revision)
    snapshot = pg._catalog_snapshot(
        registration,
        DiscoveryRequest(
            agent_id=agent.id,
            source_id=registration.id,
            sync_id="native-admission",
            requested_at=NOW,
        ),
        structure,
        NOW,
    )
    snapshot = replace(
        snapshot,
        resources=tuple(
            replace(item, sensitivity=sensitivity) for item in snapshot.resources
        ),
    )
    await agent._embedded._store.commit_snapshot(snapshot, registration=registration)
    resource = snapshot.resources[0]
    constraints = {
        "allowed_operations": ("upsert",),
        "allowed_insert_columns": ("domain", "name", "evidence_url"),
        "allowed_update_columns": ("name", "evidence_url"),
        "key_columns": ("domain",),
        "generated_identity_columns": ("id",),
        "max_rows": 10,
    }
    preview = await agent.preview_source_permissions(
        source_id=registration.id,
        read_mode="all",
        read_resource_ids=(),
        relational_write_scopes={resource.id: constraints},
    )
    await agent.apply_source_permissions(
        source_id=registration.id,
        confirmation_fingerprint=preview.confirmation_fingerprint,
    )
    status = await agent.attach_mcp_server(
        endpoint=alpha.endpoint,
        selections=(
            MCPToolSelection(
                remote_name="lookup",
                local_alias="research",
                description="Research companies and return cited evidence.",
                result_sensitivity=ModelSensitivity.RESTRICTED,
            ),
        ),
        maximum_outbound_sensitivity=ModelSensitivity.RESTRICTED,
    )
    home = agent.home
    await agent.close()

    async def connect(*args, **kwargs):
        return db.connect()

    async def load_structure(*args, **kwargs):
        return replace(structure, source_revision=db.structure_revision)

    monkeypatch.setattr(native, "_connect", connect)
    monkeypatch.setattr(native, "_load_structure", load_structure)
    agent = await Agent.open(
        "native-acceptance",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=provider.model_profile,
        clock=lambda: clock[0],
        mcp_client_factory=factory,
        approval_handler=approve,
    )
    batch = {
        "source_id": registration.id,
        "resource_id": resource.id,
        "key_columns": ("domain",),
        "insert_columns": ("domain", "name", "evidence_url"),
        "update_columns": ("name", "evidence_url"),
        "rows": (
            {
                "domain": "new.test",
                "name": "New Co",
                "evidence_url": "https://source.test/new",
            },
        ),
        "evidence_call_ids": ("research",),
    }
    return (
        agent,
        provider,
        db,
        alpha,
        status.binding,
        resource,
        constraints,
        batch,
        clock,
        approvals,
    )
