"""Catalog, advisory and connector records share the same durable contract."""

import sqlite3
from dataclasses import replace
from datetime import timedelta

import pytest

from daita.adapters.models import SourceRegistration
from daita.catalog.models import (
    CatalogResource,
    CatalogResourceRevision,
    CatalogSync,
    CatalogSyncStatus,
    ResourceKind,
    Sensitivity,
    SourceCatalogSnapshot,
    catalog_resource_id,
)
from daita.learning_candidates import LearningCandidateRejectionReason
from daita.semantics import (
    ResourceRevisionBinding,
    SemanticAnnotation,
    SemanticDigestMismatchError,
    SemanticEvidence,
    SemanticEvidenceKind,
    SemanticKind,
    SemanticSubject,
    semantic_annotation_sha256,
)
from daita.storage.sqlite_codecs.mcp_bindings import decode_mcp_binding
from daita.storage.sqlite_records import SourceReadScope
from tests.learning._candidate_support import _stored_candidate
from tests.support.graph import GRAPH_NOW
from tests.support.paths import REPO_ROOT

pytestmark = [pytest.mark.integration, pytest.mark.contract]


async def test_catalog_reopen_edit_and_detach_preserve_permission_authority(
    state_store_factory,
):
    source = SourceRegistration.build(
        agent_id="agent-1",
        adapter_id="sqlite",
        native_identity="sqlite:/fixture.db",
        display_name="Fixture",
        configuration={"path": "/fixture.db"},
        attached_at=GRAPH_NOW,
    )
    revision = CatalogResourceRevision.build(
        resource_id=catalog_resource_id(source.id, ResourceKind.TABLE, "main.orders"),
        sync_id="sync-1",
        observed_at=GRAPH_NOW,
        facet_revisions=(),
        source_revision="one",
    )
    resource = CatalogResource.build(
        agent_id="agent-1",
        source_id=source.id,
        native_identity="main.orders",
        external_uri="sqlite:fixture/orders",
        kind=ResourceKind.TABLE,
        name="orders",
        sensitivity=Sensitivity.INTERNAL,
        revision=revision,
        first_observed_at=GRAPH_NOW,
        last_observed_at=GRAPH_NOW,
    )
    sync = CatalogSync(
        id="sync-1",
        agent_id="agent-1",
        source_id=source.id,
        adapter_id="sqlite",
        status=CatalogSyncStatus.SUCCEEDED,
        started_at=GRAPH_NOW,
        completed_at=GRAPH_NOW,
        source_revision="one",
        resource_count=1,
    )
    snapshot = SourceCatalogSnapshot(
        sync=sync, resources=(resource,), revisions=(revision,)
    )
    async with state_store_factory() as store:
        assert await store.commit_snapshot(snapshot, registration=source) == snapshot
        refs = await store.list_current_snapshot_refs("agent-1", ())
        assert len(refs) == 1
        assert await store.load_current_snapshot(refs[0]) == snapshot
        assert await store.load_resource("agent-1", resource.id) == resource
        scope = SourceReadScope.allow_all(agent_id="agent-1", source_id=source.id)
        assert await store.load_source_read_scope("agent-1", source.id) == scope
        await store.replace_source_permission_scopes(scope, ())
    async with state_store_factory() as reopened:
        assert await reopened.list_sources("agent-1") == (source,)
        assert await reopened.load_sync("agent-1", sync.id) == sync
        assert await reopened.load_resource("foreign", resource.id) is None
        await reopened.detach_source(
            "agent-1", source.id, detached_at=GRAPH_NOW + timedelta(seconds=1)
        )
        assert await reopened.load_source_read_scope("agent-1", source.id) is None
        detached = await reopened.load_source("agent-1", source.id)
        assert detached is not None and not detached.active
        assert await reopened.list_current_snapshot_refs("agent-1", ()) == refs


async def test_advisory_cas_and_review_stamps_survive_reopen(state_store_factory):
    candidate, stamp = _stored_candidate(0, agent_id="agent-1")
    annotation = SemanticAnnotation(
        id="meaning",
        agent_id="agent-1",
        subject=SemanticSubject(source_ids=("source-1",), resource_ids=("resource-1",)),
        kind=SemanticKind.METRIC_DEFINITION,
        statement="The user defines booked revenue.",
        evidence=(
            SemanticEvidence(
                SemanticEvidenceKind.USER_ASSERTION, "run-1", message_position=0
            ),
        ),
        catalog_revisions=(ResourceRevisionBinding("resource-1", "revision-1"),),
        created_at=GRAPH_NOW,
        confirmed_at=GRAPH_NOW,
        confirmed_by="local-user",
    )
    async with state_store_factory() as store:
        assert await store.save_learning_candidate_review(
            "agent-1", stamps=(stamp,), candidates=(candidate,)
        ) == (candidate,)
        assert await store.save_semantic_annotation("agent-1", annotation)
    async with state_store_factory() as reopened:
        assert await reopened.learning_candidate_review_stamps("agent-1") == (stamp,)
        assert (
            await reopened.load_learning_candidate("agent-1", candidate.id) == candidate
        )
        assert (
            await reopened.load_semantic_annotation("agent-1", annotation.id)
            == annotation
        )
        updated = replace(annotation, statement="The user updated the definition.")
        with pytest.raises(SemanticDigestMismatchError):
            await reopened.save_semantic_annotation(
                "agent-1", updated, expected_sha256="0" * 64
            )
        assert await reopened.save_semantic_annotation(
            "agent-1", updated, expected_sha256=semantic_annotation_sha256(annotation)
        )
        assert (
            await reopened.load_semantic_annotation("agent-1", annotation.id) == updated
        )
        await reopened.reject_learning_candidate(
            "agent-1",
            candidate.id,
            expected_fingerprint=candidate.candidate_fingerprint,
            reason=LearningCandidateRejectionReason.USER_DECLINED,
            rejected_at=GRAPH_NOW,
        )
        assert await reopened.clear_rejected_learning_candidates("agent-1") == 1
        assert await reopened.list_learning_candidates("agent-1") == ()


async def test_released_connector_binding_reopens_and_rejects_stale_revision(
    state_store_factory,
):
    with sqlite3.connect(":memory:") as fixture:
        fixture.executescript(
            (
                REPO_ROOT / "tests/fixtures/agent-home-revisions/revision-3/state.sql"
            ).read_text()
        )
        agent, identity, data = fixture.execute(
            "SELECT agent_id, binding_id, data FROM mcp_server_bindings"
        ).fetchone()
    binding = decode_mcp_binding(data, agent_id=agent, binding_id=identity)
    async with state_store_factory() as store:
        assert await store.store_mcp_binding(binding, expected_revision=None) == binding
    async with state_store_factory() as reopened:
        assert await reopened.load_mcp_binding(agent, identity) == binding
        assert await reopened.list_mcp_bindings(agent) == (binding,)
        revised = replace(
            binding,
            revision=binding.revision + 1,
            last_checked_at=binding.last_checked_at + timedelta(seconds=1),
        )
        assert (
            await reopened.store_mcp_binding(
                revised, expected_revision=binding.revision
            )
            == revised
        )
        with pytest.raises(ValueError, match="revision precondition"):
            await reopened.store_mcp_binding(
                binding, expected_revision=binding.revision
            )
        assert await reopened.load_mcp_binding(agent, identity) == revised
