from types import SimpleNamespace

from daita.db.evidence import load_evidence, load_evidence_refs_or_latest
from daita.runtime import Evidence


class _EvidenceStore:
    def __init__(self, evidence):
        self.evidence = tuple(evidence)
        self.operation_ids = []

    async def list_evidence(self, operation_id):
        self.operation_ids.append(operation_id)
        return self.evidence


def _evidence(evidence_id, kind, *, accepted=True):
    return Evidence(
        id=evidence_id,
        kind=kind,
        owner="db_runtime",
        operation_id="operation-1",
        accepted=accepted,
        payload={},
    )


async def test_load_evidence_preserves_first_match_and_missing_behavior():
    first = _evidence("evidence-1", "query.result")
    duplicate = _evidence("evidence-1", "query.result")
    store = _EvidenceStore((first, duplicate))
    runtime = SimpleNamespace(store=store)

    assert await load_evidence(runtime, "operation-1", "evidence-1") is first
    assert await load_evidence(runtime, "operation-1", "missing") is None
    assert await load_evidence(runtime, "operation-1", "") is None
    assert store.operation_ids == ["operation-1", "operation-1"]


async def test_load_evidence_refs_preserves_explicit_order_without_fallback():
    first = _evidence("evidence-1", "catalog.asset_profile")
    second = _evidence("evidence-2", "catalog.relationship_paths")
    fallback = _evidence("evidence-3", "catalog.asset_profile")
    store = _EvidenceStore((first, second, fallback))
    runtime = SimpleNamespace(store=store)

    loaded = await load_evidence_refs_or_latest(
        runtime,
        "operation-1",
        ("evidence-2", "missing", "evidence-1"),
        kinds=("catalog.asset_profile", "catalog.relationship_paths"),
    )

    assert loaded == (second, first)
    assert store.operation_ids == ["operation-1"] * 3


async def test_load_evidence_refs_falls_back_to_accepted_matches_in_store_order():
    first = _evidence("evidence-1", "memory.recall")
    rejected = _evidence("evidence-2", "memory.recall", accepted=False)
    unrelated = _evidence("evidence-3", "query.result")
    latest = _evidence("evidence-4", "memory.recall")
    store = _EvidenceStore((first, rejected, unrelated, latest))
    runtime = SimpleNamespace(store=store)

    loaded = await load_evidence_refs_or_latest(
        runtime,
        "operation-1",
        ("missing",),
        kinds=("memory.recall",),
    )

    assert loaded == (first, latest)
    assert store.operation_ids == ["operation-1", "operation-1"]
