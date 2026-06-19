"""Catalog capability consumption helpers for execution."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from daita.runtime import Evidence, Operation, Task

from ..capabilities import SCHEMA_SEARCH_RESULT_EVIDENCE
from ..evidence import DbEvidenceStore
from ..models import DbIntent, DbIntentKind, DbOperationContract, DbRequest
from .helpers import _runtime_from_db_option


class _ExecutionCatalogMixin:
    async def _relationship_payload_if_needed(
        self,
        request: DbRequest,
        intent: DbIntent,
        contract: DbOperationContract,
        operation: Operation,
        schema: dict[str, Any],
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        store_id: str,
    ) -> dict[str, Any] | None:
        if intent.kind is not DbIntentKind.CATALOG_ASSISTED_DATA_QUERY:
            return None
        await self._execute_capability(
            "catalog.schema.search",
            contract,
            operation,
            tasks,
            evidence_store,
            {"store_id": store_id, "query": request.prompt, "limit": 10},
        )
        from_table, to_table = self.query_planner.relationship_tables_for_prompt(
            request.prompt, schema
        )
        if not from_table or not to_table:
            return None
        relationship_evidence = await self._execute_capability(
            "catalog.relationship_paths.find",
            contract,
            operation,
            tasks,
            evidence_store,
            {
                "store_id": store_id,
                "from_assets": [from_table],
                "to_assets": [to_table],
                "relationship_types": ["foreign_key", "references"],
                "max_hops": 3,
                "max_paths": 3,
            },
        )
        return relationship_evidence[0].payload if relationship_evidence else None

    async def _inspect_schema_if_available(
        self,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
    ) -> Evidence | None:
        cached = self.runtime.cached_schema_evidence(operation_id=operation.id)
        if cached is not None:
            scoped = _with_schema_scope(cached, "database")
            persisted = await self._persist_runtime_evidence(operation, scoped)
            evidence_store.add(persisted)
            return persisted
        persisted = self.runtime.persisted_schema_evidence(operation_id=operation.id)
        if persisted is not None:
            scoped = _with_schema_scope(persisted, "database")
            persisted_scoped = await self._persist_runtime_evidence(operation, scoped)
            evidence_store.add(persisted_scoped)
            return persisted_scoped
        capability = self._first_capability("db.schema.inspect")
        if capability is None:
            return None
        try:
            evidence = await self._execute_direct_capability(
                capability,
                operation,
                tasks,
                evidence_store,
                {},
                reason="planning_schema_context",
            )
        except Exception as exc:
            fallback = self.runtime.stale_persisted_schema_evidence(
                operation_id=operation.id,
                error=exc,
            )
            if fallback is None:
                raise
            persisted_fallback = await self._persist_runtime_evidence(
                operation, _with_schema_scope(fallback, "database")
            )
            evidence_store.add(persisted_fallback)
            return persisted_fallback
        if evidence:
            scoped = _with_schema_scope(evidence[0], "database")
            self.runtime.remember_schema_evidence(scoped)
            evidence_store.discard(evidence[0].id)
            evidence_store.add(scoped)
            return scoped
        return None

    async def _register_catalog_source_if_available(
        self,
        operation: Operation,
        request: DbRequest,
        schema: dict[str, Any],
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        store_id: str,
    ) -> None:
        capability = self._first_capability("catalog.source.register")
        if capability is None or not schema:
            return
        cached = self.runtime.cached_catalog_source_evidence(
            operation_id=operation.id,
            schema=schema,
            store_id=store_id,
        )
        if cached is not None:
            persisted = await self._persist_runtime_evidence(operation, cached)
            evidence_store.add(persisted)
            return
        evidence = await self._execute_direct_capability(
            capability,
            operation,
            tasks,
            evidence_store,
            {
                "schema": _schema_with_catalog_metadata(
                    schema,
                    runtime=self.runtime,
                    store_id=store_id,
                ),
                "store_type": schema.get("database_type") or "db",
                "connection_string": request.metadata.get("connection_string"),
                "store_id": store_id,
                "persist": bool(
                    _runtime_from_db_option(self.runtime, "catalog_profile_key")
                ),
            },
            reason="planning_catalog_registration",
        )
        if evidence:
            self.runtime.remember_catalog_source_evidence(
                evidence[0],
                schema=schema,
                store_id=store_id,
            )


def _schema_with_catalog_metadata(
    schema: dict[str, Any],
    *,
    runtime: Any,
    store_id: str,
) -> dict[str, Any]:
    copied = dict(schema)
    copied["store_id"] = store_id
    profile_key = _runtime_from_db_option(runtime, "catalog_profile_key")
    if profile_key:
        copied["profile_key"] = str(profile_key)
        metadata = dict(copied.get("metadata") or {})
        metadata.setdefault("profile_key", str(profile_key))
        copied["metadata"] = metadata
    return copied


def _with_schema_scope(evidence: Evidence, scope: str) -> Evidence:
    if evidence.kind != "schema.asset_profile":
        return evidence
    payload = dict(evidence.payload)
    metadata = {**evidence.metadata, "scope": scope}
    payload_metadata = dict(payload.get("metadata") or {})
    payload_metadata.setdefault("scope", scope)
    payload["metadata"] = payload_metadata
    return replace(evidence, payload=payload, metadata=metadata)


def _catalog_evidence_for_planning(
    evidence_store: DbEvidenceStore,
) -> tuple[Evidence, ...]:
    return tuple(
        item
        for item in evidence_store.list()
        if item.kind.startswith("catalog.")
        or item.kind
        in {
            SCHEMA_SEARCH_RESULT_EVIDENCE,
            "schema.column_value_profile",
            "schema.column_value_search_result",
            "schema.column_value_hint",
            "catalog.source",
        }
    )


def _catalog_column_value_search_exists(evidence_store: DbEvidenceStore) -> bool:
    return any(
        item.kind == "schema.column_value_search_result"
        for item in evidence_store.list()
    )
