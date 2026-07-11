"""Schema and catalog evidence cache helpers for ``DbRuntime``."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from daita.runtime import Evidence

from ..fingerprints import persisted_fingerprint
from .types import _SourcePreparationSnapshot


class DbRuntimeCacheMixin:
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
