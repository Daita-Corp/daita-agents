"""Column-value profile grounding helpers for execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from daita.runtime import Capability, Evidence, Operation, Task

from ..capabilities import SCHEMA_RELATIONSHIP_PATH_EVIDENCE
from ..evidence import DbEvidenceStore
from ..models import DbRequest
from .catalog import _catalog_evidence_for_planning
from .helpers import _runtime_from_db_option
from .repair import _zero_row_literal_predicates


@dataclass(frozen=True)
class _ColumnValueEvidenceState:
    key: tuple[str, str]
    profile: dict[str, Any]
    source: str
    fresh: bool
    operation_local: bool
    stale_reason: str | None = None


class _ExecutionValueGroundingMixin:
    async def _search_catalog_column_values_if_available(
        self,
        request: DbRequest,
        operation: Operation,
        schema: dict[str, Any],
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        store_id: str,
    ) -> None:
        search_capability = self._first_capability("catalog.column_values.search")
        if search_capability is None or not schema:
            return

        await self._execute_direct_capability(
            search_capability,
            operation,
            tasks,
            evidence_store,
            {
                "store_id": store_id,
                "query": request.prompt,
                "limit": 12,
                **_catalog_profile_ttl_input(self.runtime),
            },
            reason="catalog_column_value_profile_search",
        )

    async def _resolve_predicate_value_profiles_if_available(
        self,
        request: DbRequest,
        operation: Operation,
        schema: dict[str, Any],
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        store_id: str,
        *,
        schema_evidence: Evidence | None,
        planning_context: Evidence,
        plan_evidence: Evidence,
    ) -> Evidence:
        profile_capability = self._first_capability("db.column_values.profile")
        register_capability = self._first_capability("catalog.column_values.register")
        if profile_capability is None or register_capability is None or not schema:
            return planning_context

        sql = str(
            plan_evidence.payload.get("sql")
            or (plan_evidence.payload.get("structured_plan") or {}).get("selected_sql")
            or ""
        )
        if not sql:
            return planning_context
        try:
            from ..sql_analysis import analyze_sql

            analysis = analyze_sql(
                sql,
                dialect=str(
                    planning_context.payload.get("dialect")
                    or schema.get("database_type")
                    or ""
                ),
            )
        except Exception:
            return planning_context

        evidence_index = _column_value_evidence_index(evidence_store, schema)
        candidates = _predicate_profile_candidates(
            analysis.literal_predicates,
            schema,
            evidence_index=evidence_index,
        )
        if not candidates:
            return planning_context

        registered = False
        for table, column in candidates:
            key = (table.lower(), column.lower())
            stored_state = evidence_index.get(key)
            stored_profile = stored_state.profile if stored_state is not None else None
            if (
                stored_profile is not None
                and stored_state is not None
                and not stored_state.fresh
                and _profile_capability_supports_fingerprint(profile_capability)
            ):
                fingerprint = await self._execute_direct_capability(
                    profile_capability,
                    operation,
                    tasks,
                    evidence_store,
                    _column_value_profile_input(table, column, fingerprint_only=True),
                    reason="column_value_source_fingerprint_check",
                )
                if fingerprint and _source_fingerprint_preserves_freshness(
                    stored_profile=stored_profile,
                    current_profile=fingerprint[0].payload,
                ):
                    await self._execute_direct_capability(
                        register_capability,
                        operation,
                        tasks,
                        evidence_store,
                        {
                            "store_id": store_id,
                            "profiles": [
                                _fresh_profile_from_preserved_fingerprint(
                                    stored_profile,
                                    fingerprint[0].payload,
                                )
                            ],
                            "source_evidence_id": fingerprint[0].id,
                            "persist": bool(
                                _runtime_from_db_option(
                                    self.runtime, "catalog_profile_key"
                                )
                            ),
                        },
                        reason="stale_catalog_column_value_registration",
                    )
                    registered = True
                    continue

            raw_profiles = await self._execute_direct_capability(
                profile_capability,
                operation,
                tasks,
                evidence_store,
                _column_value_profile_input(table, column),
                reason=(
                    "column_value_predicate_profile"
                    if stored_profile is None
                    else "stale_column_value_profile_refresh"
                ),
            )
            if not raw_profiles:
                continue
            await self._execute_direct_capability(
                register_capability,
                operation,
                tasks,
                evidence_store,
                {
                    "store_id": store_id,
                    "profiles": [raw_profiles[0].payload],
                    "source_evidence_id": raw_profiles[0].id,
                    "persist": bool(
                        _runtime_from_db_option(self.runtime, "catalog_profile_key")
                    ),
                },
                reason=(
                    "catalog_column_value_predicate_registration"
                    if stored_profile is None
                    else "stale_catalog_column_value_registration"
                ),
            )
            registered = True

        if not registered:
            return planning_context

        return await self._build_planning_context(
            request,
            operation,
            tasks,
            evidence_store,
            schema_evidence=schema_evidence,
            catalog_evidence=_catalog_evidence_for_planning(evidence_store),
            relationship_evidence=tuple(
                item
                for item in evidence_store.list()
                if item.kind == SCHEMA_RELATIONSHIP_PATH_EVIDENCE
            ),
            analysis_metadata={"predicate_column_value_profiles": True},
        )


def _column_value_profile_input(
    table: str,
    column: str,
    *,
    fingerprint_only: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "table": table,
        "column": column,
        "max_values": 25,
        "max_distinct_count": 100,
        "max_value_length": 80,
        "include_source_revision": True,
    }
    if fingerprint_only:
        payload["fingerprint_only"] = True
    return payload


def _predicate_profile_candidates(
    literal_predicates: tuple[Any, ...],
    schema: dict[str, Any],
    *,
    evidence_index: dict[tuple[str, str], _ColumnValueEvidenceState],
) -> tuple[tuple[str, str], ...]:
    table_columns = _schema_columns_by_table(schema)
    candidates: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for predicate in literal_predicates:
        if str(getattr(predicate, "operator", "")).lower() not in {"=", "in", "like"}:
            continue
        column_ref = getattr(predicate, "column", None)
        column_name = str(getattr(column_ref, "name", "") or "")
        if not column_name:
            continue
        table_name = _predicate_table_name(
            str(getattr(column_ref, "table", "") or ""),
            column_name,
            table_columns,
        )
        if not table_name:
            continue
        key = (table_name.lower(), column_name.lower())
        if key in seen:
            continue
        state = evidence_index.get(key)
        if state is not None and (
            state.fresh or _catalog_profile_is_ineligible(state.profile)
        ):
            continue
        column = table_columns.get(table_name.lower(), {}).get(column_name.lower())
        if column is None or not _profile_candidate_shape(column):
            continue
        seen.add(key)
        candidates.append((table_name, column_name))
        if len(candidates) >= 5:
            break
    return tuple(candidates)


def _column_value_evidence_index(
    evidence_store: DbEvidenceStore,
    schema: dict[str, Any],
) -> dict[tuple[str, str], _ColumnValueEvidenceState]:
    states: dict[tuple[str, str], _ColumnValueEvidenceState] = {}
    metadata = schema.get("metadata") or {}
    schema_profiles = (
        metadata.get("column_value_profiles") if isinstance(metadata, dict) else {}
    )
    if isinstance(schema_profiles, dict):
        for profile in schema_profiles.values():
            if isinstance(profile, dict):
                _put_column_value_state(
                    states,
                    profile,
                    source="schema_metadata",
                    operation_local=False,
                    fresh=_catalog_profile_is_fresh(profile),
                )

    for evidence in evidence_store.list():
        if evidence.kind == "schema.column_value_search_result":
            for profile in _catalog_column_value_profiles(evidence):
                _put_column_value_state(
                    states,
                    profile,
                    source="catalog_search",
                    operation_local=False,
                    fresh=_catalog_profile_is_fresh(profile),
                )
        elif evidence.kind == "schema.column_value_profile":
            for profile in evidence.payload.get("profiles", []) or []:
                if isinstance(profile, dict):
                    _put_column_value_state(
                        states,
                        profile,
                        source="catalog_registration",
                        operation_local=True,
                        fresh=_operation_local_profile_is_fresh(profile),
                    )
    return states


def _put_column_value_state(
    states: dict[tuple[str, str], _ColumnValueEvidenceState],
    profile: dict[str, Any],
    *,
    source: str,
    operation_local: bool,
    fresh: bool,
) -> None:
    table = str(profile.get("table") or "").split(".")[-1].lower()
    column = str(profile.get("column") or "").lower()
    if not table or not column:
        return
    key = (table, column)
    candidate = _ColumnValueEvidenceState(
        key=key,
        profile=dict(profile),
        source=source,
        fresh=fresh,
        operation_local=operation_local,
        stale_reason=(
            str(profile.get("stale_reason")) if profile.get("stale_reason") else None
        ),
    )
    existing = states.get(key)
    if existing is None or _column_value_state_rank(
        candidate
    ) > _column_value_state_rank(existing):
        states[key] = candidate


def _column_value_state_rank(state: _ColumnValueEvidenceState) -> tuple[int, int, int]:
    return (
        1 if state.operation_local else 0,
        1 if state.fresh else 0,
        {"catalog_registration": 3, "catalog_search": 2, "schema_metadata": 1}.get(
            state.source, 0
        ),
    )


def _catalog_column_value_profiles(
    search_evidence: Evidence | None,
) -> tuple[dict[str, Any], ...]:
    if search_evidence is None:
        return ()
    return tuple(
        profile
        for profile in search_evidence.payload.get("profiles", []) or []
        if isinstance(profile, dict)
    )


def _catalog_profile_is_fresh(profile: dict[str, Any]) -> bool:
    if profile.get("stale"):
        return False
    if not _operation_local_profile_is_fresh(profile):
        return False
    return bool(profile.get("top_values") or profile.get("source_fingerprint"))


def _operation_local_profile_is_fresh(profile: dict[str, Any]) -> bool:
    if profile.get("profile_status") in {"stale", "skipped", "redacted"}:
        return False
    if profile.get("stale") or profile.get("redacted"):
        return False
    if profile.get("sampled") or profile.get("truncated"):
        return False
    return bool(profile.get("top_values"))


def _catalog_profile_is_ineligible(profile: dict[str, Any]) -> bool:
    return bool(profile.get("redacted")) or profile.get("profile_status") == "skipped"


def _source_fingerprint_preserves_freshness(
    *,
    stored_profile: dict[str, Any],
    current_profile: dict[str, Any],
) -> bool:
    stored_fingerprint = stored_profile.get("source_fingerprint")
    current_fingerprint = current_profile.get("source_fingerprint")
    if not stored_fingerprint or not current_fingerprint:
        return False
    if current_fingerprint != stored_fingerprint:
        return False
    current_status = str(
        current_profile.get("source_fingerprint_status")
        or stored_profile.get("source_fingerprint_status")
        or "best_effort"
    )
    if current_status == "unavailable":
        return False
    if stored_profile.get("stale") or stored_profile.get("profile_status") == "stale":
        if current_status == "authoritative":
            return True
        return (
            current_status == "best_effort"
            and stored_profile.get("stale_reason") == "profile_ttl_expired"
        )
    return current_status in {"authoritative", "best_effort"}


def _fresh_profile_from_preserved_fingerprint(
    stored_profile: dict[str, Any],
    current_profile: dict[str, Any],
) -> dict[str, Any]:
    fresh = {
        key: value
        for key, value in stored_profile.items()
        if key
        not in {
            "match_reasons",
            "profile_ref",
            "score",
            "stale",
            "stale_reason",
            "store_id",
        }
    }
    fresh["profile_status"] = "profiled"
    for key in (
        "source_fingerprint",
        "source_fingerprint_status",
        "source_fingerprint_reason",
        "source_revision",
    ):
        if key in current_profile:
            fresh[key] = current_profile[key]
    return fresh


def _profile_capability_supports_fingerprint(capability: Capability) -> bool:
    policy = capability.metadata.get("profile_policy")
    return isinstance(policy, dict) and bool(policy.get("fingerprint_only_supported"))


def _schema_columns_by_table(
    schema: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    tables: dict[str, dict[str, dict[str, Any]]] = {}
    for table in schema.get("tables", []) or []:
        table_name = str(table.get("name") or "")
        if not table_name:
            continue
        tables[table_name.lower()] = {
            str(column.get("name") or "").lower(): column
            for column in table.get("columns", []) or []
            if column.get("name")
        }
    return tables


def _predicate_table_name(
    table: str,
    column: str,
    table_columns: dict[str, dict[str, dict[str, Any]]],
) -> str | None:
    table_key = table.split(".")[-1].lower() if table else ""
    if table_key in table_columns:
        return table_key
    matches = [
        known_table
        for known_table, columns in table_columns.items()
        if column.lower() in columns
    ]
    return matches[0] if len(matches) == 1 else None


def _plan_has_literal_predicates(
    plan_evidence: Evidence,
    schema: dict[str, Any],
) -> bool:
    sql = str(
        plan_evidence.payload.get("sql")
        or (plan_evidence.payload.get("structured_plan") or {}).get("selected_sql")
        or ""
    )
    if not sql:
        return False
    try:
        from ..sql_analysis import analyze_sql

        analysis = analyze_sql(sql, dialect=str(schema.get("database_type") or ""))
    except Exception:
        return False
    return bool(_zero_row_literal_predicates(analysis.literal_predicates))


def _profile_candidate_shape(column: dict[str, Any]) -> bool:
    if column.get("is_primary_key"):
        return False
    data_type = str(column.get("type") or column.get("data_type") or "").lower()
    return any(
        token in data_type
        for token in (
            "char",
            "text",
            "enum",
            "bool",
            "int",
        )
    )


def _catalog_profile_ttl_input(runtime: Any | None) -> dict[str, Any]:
    if runtime is None:
        return {}
    value = _runtime_from_db_option(runtime, "cache_ttl")
    if value is None:
        return {}
    return {"max_profile_age_seconds": value}
