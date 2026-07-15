"""Schema and catalog evidence cache helpers for ``DbRuntime``."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, TYPE_CHECKING

from daita.runtime import Evidence

from ..fingerprints import persisted_fingerprint
from .types import _SourcePreparationSnapshot

if TYPE_CHECKING:
    from ..models import DbRuntimeConfig


class DbRuntimeCacheMixin:
    if TYPE_CHECKING:
        config: DbRuntimeConfig
        _schema_profile_cache: dict[str, Any] | None
    _catalog_source_cache: _SourcePreparationSnapshot | None

    async def prepare_sqlite_slim_source(self) -> None:
        """Warm the existing schema/catalog owners before SQLite model turns."""

        registry = getattr(self, "registry", None)
        if registry is None or "sqlite" not in registry.plugin_ids:
            return
        schema_evidence = self.cached_schema_evidence(operation_id="sqlite-slim-warm")
        if schema_evidence is None:
            schema_evidence = self.persisted_schema_evidence(
                operation_id="sqlite-slim-warm"
            )
        if schema_evidence is None:
            inspected = await self.execute_capability(
                "db.schema.inspect",
                owner="sqlite",
                operation_type="source.profile",
                input={},
            )
            schema_evidence = next(
                (
                    item
                    for item in inspected
                    if item.accepted and item.kind == "schema.asset_profile"
                ),
                None,
            )
            if schema_evidence is not None:
                self.remember_schema_evidence(schema_evidence)
        if schema_evidence is None:
            return

        options = _from_db_options(self.config.metadata)
        store_id = str(options.get("catalog_store_id") or "")
        if not store_id:
            return
        try:
            registry.get_capability("catalog.source.register", owner="catalog")
        except KeyError:
            return
        cached_registration = self.cached_catalog_source_evidence(
            operation_id="sqlite-slim-warm",
            schema=dict(schema_evidence.payload),
            store_id=store_id,
        )
        if cached_registration is not None:
            return
        registered = await self.execute_capability(
            "catalog.source.register",
            owner="catalog",
            operation_type="source.profile",
            input={
                "schema": dict(schema_evidence.payload),
                "store_type": "sqlite",
                "store_id": store_id,
                "persist": False,
            },
        )
        registration = next(
            (
                item
                for item in registered
                if item.accepted and item.kind == "catalog.source_registered"
            ),
            None,
        )
        if registration is not None:
            self.remember_catalog_source_evidence(
                registration,
                schema=dict(schema_evidence.payload),
                store_id=store_id,
            )

    def cached_schema_evidence(self, *, operation_id: str) -> Evidence | None:
        """Return cached schema profile evidence when the runtime cache is fresh."""
        cached = self._schema_profile_cache
        if not cached:
            return None
        ttl = _schema_cache_ttl(self.config.metadata)
        if ttl is not None:
            if ttl <= 0:
                return None
            if time.monotonic() - float(cached["cached_at"]) > ttl:
                return None
        return Evidence(
            kind=str(cached["kind"]),
            owner=str(cached["owner"]) if cached.get("owner") else None,
            operation_id=operation_id,
            payload=dict(cached["payload"]),
            metadata={
                **dict(cached.get("metadata") or {}),
                "schema_cache": "hit",
            },
        )

    def cached_catalog_source_evidence(
        self,
        *,
        operation_id: str,
        schema: dict[str, Any],
        store_id: str,
    ) -> Evidence | None:
        """Return fresh catalog source-registration evidence for this schema."""
        cached = self._catalog_source_cache
        if cached is None:
            return None
        ttl = _schema_cache_ttl(self.config.metadata)
        if ttl is not None:
            if ttl <= 0:
                return None
            if time.monotonic() - cached.cached_at > ttl:
                return None
        if cached.store_id != store_id:
            return None
        if cached.schema_fingerprint != _source_schema_fingerprint(schema):
            return None
        return replace(
            cached.evidence,
            id=None,
            operation_id=operation_id,
            task_id=None,
            metadata={
                **dict(cached.evidence.metadata),
                "catalog_source_cache": "hit",
                "reused_evidence_id": cached.evidence.id,
                "schema_fingerprint": cached.schema_fingerprint,
            },
        )

    def persisted_schema_evidence(self, *, operation_id: str) -> Evidence | None:
        """Return persisted catalog schema profile evidence when fresh enough."""
        from daita.plugins.catalog.persistence import load_schema_snapshot

        options = _from_db_options(self.config.metadata)
        profile_key = options.get("catalog_profile_key")
        if not profile_key:
            return None
        loaded = load_schema_snapshot(
            str(profile_key),
            catalog_keys=[
                str(item) for item in (options.get("catalog_keys") or []) if item
            ],
            ttl=_schema_cache_ttl(self.config.metadata),
        )
        if loaded is None:
            return None
        payload, is_expired = loaded
        if is_expired:
            return None
        evidence = Evidence(
            kind="schema.asset_profile",
            owner=str(payload.get("database_type") or "catalog"),
            operation_id=operation_id,
            payload=dict(payload),
            metadata={"schema_cache": "persistent_hit"},
        )
        self.remember_schema_evidence(evidence)
        return evidence

    def stale_persisted_schema_evidence(
        self,
        *,
        operation_id: str,
        error: Exception,
    ) -> Evidence | None:
        """Return an expired persisted schema profile after refresh failure."""
        from daita.plugins.catalog.persistence import load_schema_snapshot

        options = _from_db_options(self.config.metadata)
        profile_key = options.get("catalog_profile_key")
        if not profile_key:
            return None
        loaded = load_schema_snapshot(
            str(profile_key),
            catalog_keys=[
                str(item) for item in (options.get("catalog_keys") or []) if item
            ],
            ttl=_schema_cache_ttl(self.config.metadata),
        )
        if loaded is None:
            return None
        payload, is_expired = loaded
        if not is_expired:
            return None
        evidence = Evidence(
            kind="schema.asset_profile",
            owner=str(payload.get("database_type") or "catalog"),
            operation_id=operation_id,
            payload=dict(payload),
            metadata={
                "schema_cache": "persistent_stale_fallback",
                "refresh_error_type": type(error).__name__,
                "refresh_error": str(error),
            },
        )
        self.remember_schema_evidence(evidence)
        return evidence

    def remember_schema_evidence(self, evidence: Evidence) -> None:
        """Store schema profile payload for subsequent operations on this runtime."""
        if evidence.kind != "schema.asset_profile":
            return
        self._schema_profile_cache = {
            "kind": evidence.kind,
            "owner": evidence.owner,
            "payload": dict(evidence.payload),
            "metadata": {"schema_cache": "stored"},
            "cached_at": time.monotonic(),
        }

    def remember_catalog_source_evidence(
        self,
        evidence: Evidence,
        *,
        schema: dict[str, Any],
        store_id: str,
    ) -> None:
        """Remember an accepted catalog source-registration evidence reference."""
        if evidence.kind != "catalog.source_registered" or not evidence.accepted:
            return
        self._catalog_source_cache = _SourcePreparationSnapshot(
            evidence=evidence,
            store_id=store_id,
            schema_fingerprint=_source_schema_fingerprint(schema),
            cached_at=time.monotonic(),
        )


def _schema_cache_ttl(metadata: dict[str, Any]) -> float | None:
    options = _from_db_options(metadata)
    source_options = options.get("source_options")
    if isinstance(source_options, dict) and "cache_ttl" in source_options:
        value = source_options.get("cache_ttl")
        return None if value is None else float(value)
    return None


def _source_schema_fingerprint(schema: dict[str, Any]) -> str:
    return persisted_fingerprint(schema)


def _from_db_options(metadata: dict[str, Any]) -> dict[str, Any]:
    options = metadata.get("from_db_options")
    return options if isinstance(options, dict) else {}
