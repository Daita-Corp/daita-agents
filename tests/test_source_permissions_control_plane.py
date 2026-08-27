from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime

import pytest
from _workspace_support import workspace_for

from daita import Agent
from daita.adapters.models import SourceRegistration
from daita.catalog.models import (
    CatalogFacet,
    CatalogResource,
    CatalogResourceRevision,
    CatalogSync,
    CatalogSyncStatus,
    ResourceKind,
    Sensitivity,
    SourceCatalogSnapshot,
    TabularColumn,
    TabularFacet,
    catalog_resource_id,
)
from daita.storage.sqlite_records import SourceReadMode

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)


def _registration(agent_id: str) -> SourceRegistration:
    return SourceRegistration.build(
        agent_id=agent_id,
        adapter_id="postgresql",
        native_identity="postgresql:permission-control",
        display_name="Permission warehouse",
        configuration={
            "database": "warehouse",
            "host": "db.example.test",
            "port": 5432,
            "schemas": ("public",),
            "ssl_mode": "require",
            "username": "reader",
        },
        attached_at=NOW,
    )


def _snapshot(
    registration: SourceRegistration,
    *,
    sync_id: str,
    tables: tuple[tuple[str, int], ...],
) -> tuple[SourceCatalogSnapshot, dict[str, CatalogResource]]:
    resources: list[CatalogResource] = []
    revisions: list[CatalogResourceRevision] = []
    facets: list[CatalogFacet] = []
    by_name: dict[str, CatalogResource] = {}
    for table_name, assignment_count in tables:
        native_identity = f"public.{table_name}"
        resource_id = catalog_resource_id(
            registration.id,
            ResourceKind.TABLE,
            native_identity,
        )
        columns = (
            TabularColumn(
                name="id",
                native_type="bigint",
                native_type_namespace="pg_catalog",
                native_type_name="int8",
                ordinal=0,
                nullable=False,
                primary_key_ordinal=1,
                updatable=True,
            ),
            *tuple(
                TabularColumn(
                    name=f"value_{index:02d}",
                    native_type="text",
                    native_type_namespace="pg_catalog",
                    native_type_name="text",
                    ordinal=index,
                    nullable=True,
                    updatable=True,
                )
                for index in range(1, assignment_count + 1)
            ),
        )
        facet = CatalogFacet.from_tabular(
            resource_id=resource_id,
            sync_id=sync_id,
            observed_at=NOW,
            facet=TabularFacet(columns=columns),
        )
        revision = CatalogResourceRevision.build(
            resource_id=resource_id,
            sync_id=sync_id,
            observed_at=NOW,
            facet_revisions=(facet.revision,),
            source_revision=f"catalog:{sync_id}",
        )
        resource = CatalogResource.build(
            agent_id=registration.agent_id,
            source_id=registration.id,
            native_identity=native_identity,
            external_uri=f"postgresql://permission/{native_identity}",
            kind=ResourceKind.TABLE,
            name=table_name,
            sensitivity=Sensitivity.INTERNAL,
            revision=revision,
            first_observed_at=NOW,
            last_observed_at=NOW,
        )
        resources.append(resource)
        revisions.append(revision)
        facets.append(facet)
        by_name[table_name] = resource
    sync = CatalogSync(
        id=sync_id,
        agent_id=registration.agent_id,
        source_id=registration.id,
        adapter_id="postgresql",
        status=CatalogSyncStatus.SUCCEEDED,
        started_at=NOW,
        completed_at=NOW,
        source_revision=f"catalog:{sync_id}",
        resource_count=len(resources),
    )
    return (
        SourceCatalogSnapshot(
            sync=sync,
            resources=tuple(resources),
            revisions=tuple(revisions),
            facets=tuple(facets),
        ),
        by_name,
    )


async def _agent_with_catalog(tmp_path, tables=("orders", "customers")):
    agent = await Agent.create(
        "permission-control",
        root=tmp_path,
        clock=lambda: NOW,
        workspace=workspace_for(tmp_path),
    )
    registration = _registration(agent.id)
    snapshot, resources = _snapshot(
        registration,
        sync_id="sync-one",
        tables=tuple((name, 2) for name in tables),
    )
    await agent._embedded._store.commit_snapshot(
        snapshot,
        registration=registration,
    )
    return agent, registration, resources


async def test_inspect_is_exact_complete_catalog_and_secret_free(tmp_path):
    agent, source, resources = await _agent_with_catalog(tmp_path)
    try:
        inspection = await agent.inspect_source_permissions(source.id)
        assert inspection.state.read_scope.mode is SourceReadMode.ALL
        assert inspection.state.postgresql_update_scopes == ()
        assert {item.resource_id for item in inspection.resources} == {
            resource.id for resource in resources.values()
        }
        assert all(
            item.eligible_assignment_columns == ("value_01", "value_02")
            for item in inspection.resources
        )
        safe = repr(asdict(inspection))
        assert "credential" not in safe
        assert "configuration" not in safe
        assert "db.example.test" not in safe
    finally:
        await agent.close()


@pytest.mark.parametrize(
    ("mode", "selected_names"),
    (
        (SourceReadMode.ALL, ()),
        (SourceReadMode.SELECTED, ("orders",)),
        (SourceReadMode.SELECTED, ("orders", "customers")),
        (SourceReadMode.NONE, ()),
    ),
)
async def test_preview_apply_read_modes(tmp_path, mode, selected_names):
    agent, source, resources = await _agent_with_catalog(tmp_path)
    try:
        selected_ids = tuple(resources[name].id for name in selected_names)
        preview = await agent.preview_source_permissions(
            source_id=source.id,
            read_mode=mode,
            read_resource_ids=selected_ids,
            postgresql_update_scopes={},
        )
        assert preview.before.read_scope.mode is SourceReadMode.ALL
        assert preview.after.read_scope.mode is mode
        assert set(preview.after.read_scope.resource_ids) == set(selected_ids)
        applied = await agent.apply_source_permissions(
            source_id=source.id,
            confirmation_fingerprint=preview.confirmation_fingerprint,
        )
        assert applied.state == preview.after
        assert (
            await agent.apply_source_permissions(
                source_id=source.id,
                confirmation_fingerprint=preview.confirmation_fingerprint,
            )
        ).state == preview.after
    finally:
        await agent.close()


async def test_write_adds_read_uses_exact_columns_and_read_narrowing_revokes(tmp_path):
    agent, source, resources = await _agent_with_catalog(tmp_path)
    orders = resources["orders"]
    try:
        write_preview = await agent.preview_source_permissions(
            source_id=source.id,
            read_mode="none",
            read_resource_ids=(),
            postgresql_update_scopes={orders.id: ["value_01", "value_02"]},
        )
        assert write_preview.after.read_scope.mode is SourceReadMode.SELECTED
        assert write_preview.after.read_scope.resource_ids == (orders.id,)
        assert write_preview.automatic_read_additions == (orders.id,)
        assert write_preview.after.postgresql_update_scopes[
            0
        ].allowed_assignment_columns == ("value_01", "value_02")
        await agent.apply_source_permissions(
            source_id=source.id,
            confirmation_fingerprint=write_preview.confirmation_fingerprint,
        )

        revoke_preview = await agent.preview_source_permissions(
            source_id=source.id,
            read_mode="none",
            read_resource_ids=(),
            postgresql_update_scopes={},
        )
        assert revoke_preview.dependent_update_revocations == (orders.id,)
        assert revoke_preview.after.postgresql_update_scopes == ()
        await agent.apply_source_permissions(
            source_id=source.id,
            confirmation_fingerprint=revoke_preview.confirmation_fingerprint,
        )
    finally:
        await agent.close()


async def test_write_selected_many_all_current_and_future_table_exclusion(tmp_path):
    agent, source, resources = await _agent_with_catalog(tmp_path)
    try:
        mapping = {
            resource.id: ["value_01", "value_02"] for resource in resources.values()
        }
        preview = await agent.preview_source_permissions(
            source_id=source.id,
            read_mode="all",
            read_resource_ids=(),
            postgresql_update_scopes=mapping,
        )
        assert len(preview.after.postgresql_update_scopes) == 2
        await agent.apply_source_permissions(
            source_id=source.id,
            confirmation_fingerprint=preview.confirmation_fingerprint,
        )

        refreshed, refreshed_resources = _snapshot(
            source,
            sync_id="sync-two",
            tables=(("orders", 2), ("customers", 2), ("future", 2)),
        )
        await agent._embedded._store.commit_snapshot(
            refreshed,
            registration=source,
        )
        inspection = await agent.inspect_source_permissions(source.id)
        assert refreshed_resources["future"].id not in {
            scope.resource_id for scope in inspection.state.postgresql_update_scopes
        }
    finally:
        await agent.close()


async def test_advanced_columns_support_wide_exact_binding_and_stale_preview(tmp_path):
    agent = await Agent.create(
        "permission-bound",
        root=tmp_path,
        clock=lambda: NOW,
        workspace=workspace_for(tmp_path),
    )
    source = _registration(agent.id)
    snapshot, resources = _snapshot(
        source,
        sync_id="sync-wide",
        tables=(("wide", 33),),
    )
    await agent._embedded._store.commit_snapshot(snapshot, registration=source)
    wide = resources["wide"]
    all_columns = tuple(f"value_{index:02d}" for index in range(1, 34))
    try:
        inspection = await agent.inspect_source_permissions(source.id)
        assert inspection.resources[0].requires_advanced_column_selection
        preview = await agent.preview_source_permissions(
            source_id=source.id,
            read_mode="all",
            read_resource_ids=(),
            postgresql_update_scopes={wide.id: list(all_columns)},
        )
        assert (
            preview.after.postgresql_update_scopes[0].allowed_assignment_columns
            == all_columns
        )

        refreshed, _ = _snapshot(
            source,
            sync_id="sync-wide-refreshed",
            tables=(("wide", 33),),
        )
        await agent._embedded._store.commit_snapshot(refreshed, registration=source)
        with pytest.raises(ValueError, match="catalog changed"):
            await agent.apply_source_permissions(
                source_id=source.id,
                confirmation_fingerprint=preview.confirmation_fingerprint,
            )
        assert (
            await agent.inspect_source_permissions(source.id)
        ).state.postgresql_update_scopes == ()
    finally:
        await agent.close()


async def test_unknown_or_wrong_confirmation_never_changes_state(tmp_path):
    agent, source, resources = await _agent_with_catalog(tmp_path)
    try:
        preview = await agent.preview_source_permissions(
            source_id=source.id,
            read_mode="selected",
            read_resource_ids=(resources["orders"].id,),
            postgresql_update_scopes={},
        )
        with pytest.raises(ValueError, match="unknown or expired"):
            await agent.apply_source_permissions(
                source_id=source.id,
                confirmation_fingerprint="sha256:" + "0" * 64,
            )
        assert (
            await agent.inspect_source_permissions(source.id)
        ).state == preview.before
    finally:
        await agent.close()
