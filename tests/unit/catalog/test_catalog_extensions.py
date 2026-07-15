import pytest

from daita.db import DbRuntime
from daita.db.planning_context import catalog_schema_from_evidence
from daita.plugins import ExtensionRegistry, PluginKind
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.catalog.base_profiler import NormalizedColumn
from daita.runtime import (
    AccessMode,
    ContextAudience,
    Evidence,
    Operation,
    RiskLevel,
    Task,
)


def _reference_schema():
    return {
        "database_type": "sqlite",
        "database_name": "shop",
        "tables": [
            {
                "name": "customers",
                "columns": [
                    {"name": "id", "data_type": "INTEGER"},
                    {"name": "email", "data_type": "TEXT"},
                ],
            },
            {
                "name": "orders",
                "columns": [
                    {"name": "id", "data_type": "INTEGER"},
                    {"name": "customer_id", "data_type": "INTEGER"},
                    {"name": "total", "data_type": "REAL"},
                ],
            },
        ],
        "foreign_keys": [
            {
                "source_table": "orders",
                "source_column": "customer_id",
                "target_table": "customers",
                "target_column": "id",
            }
        ],
    }


def _value_grounding_schema():
    return {
        "database_type": "sqlite",
        "database_name": "shop",
        "tables": [
            {
                "name": "customers",
                "columns": [
                    {"name": "id", "data_type": "INTEGER"},
                    {"name": "email", "data_type": "TEXT"},
                    {"name": "tier", "data_type": "TEXT"},
                ],
            },
            {
                "name": "orders",
                "columns": [
                    {"name": "id", "data_type": "INTEGER"},
                    {"name": "customer_id", "data_type": "INTEGER"},
                    {"name": "status", "data_type": "TEXT"},
                    {"name": "total", "data_type": "REAL"},
                ],
            },
        ],
        "foreign_keys": [
            {
                "source_table": "orders",
                "source_column": "customer_id",
                "target_table": "customers",
                "target_column": "id",
            }
        ],
    }


def _executor(registry, executor_id):
    return next(
        executor for executor in registry.executors if executor.id == executor_id
    )


async def _value_grounding_catalog():
    catalog = CatalogPlugin(auto_persist=False)
    await catalog.register_schema(
        _value_grounding_schema(),
        store_type="sqlite",
        store_id="store:shop",
        persist=False,
    )
    registry = ExtensionRegistry()
    registry.register(catalog)
    return catalog, registry


def test_catalog_plugin_declares_domain_service_manifest():
    plugin = CatalogPlugin(auto_persist=False)

    assert plugin.manifest.id == "catalog"
    assert plugin.manifest.kind is PluginKind.DOMAIN_SERVICE
    assert {"db", "cloud", "file"} <= plugin.manifest.domains


def test_catalog_capabilities_are_visible_in_extension_registry():
    registry = ExtensionRegistry()

    registry.register(CatalogPlugin(auto_persist=False))

    capability_ids = {capability.id for capability in registry.capabilities}
    executor_ids = {executor.id for executor in registry.executors}
    evidence_kinds = {schema.kind for schema in registry.evidence_schemas}
    tool_view_names = {tool_view.name for tool_view in registry.tool_views}

    assert {
        "catalog.source.register",
        "catalog.source.profile",
        "catalog.schema.search",
        "catalog.asset.inspect",
        "catalog.relationship_paths.find",
        "catalog.column_values.register",
        "catalog.column_values.search",
        "catalog.column_value_hints.resolve",
        "catalog.value_grounding.plan",
        "catalog.infrastructure.discover",
        "catalog.schema.compare",
        "catalog.diagram.export",
    } <= capability_ids
    assert {
        "catalog.register_source",
        "catalog.profile_source",
        "catalog.search_schema",
        "catalog.inspect_asset",
        "catalog.find_relationship_paths",
        "catalog.register_column_values",
        "catalog.search_column_values",
        "catalog.resolve_column_value_hints",
        "catalog.plan_value_grounding",
        "catalog.discover_infrastructure",
        "catalog.compare_schema",
        "catalog.export_diagram",
    } <= executor_ids
    assert {
        "catalog.source_registered",
        "catalog.profile",
        "schema.search_result",
        "schema.asset_profile",
        "schema.relationship_path",
        "schema.column_value_profile",
        "schema.column_value_search_result",
        "schema.column_value_hint",
        "catalog.value_grounding.plan",
        "catalog.infrastructure_inventory",
        "schema.comparison",
    } <= evidence_kinds
    assert {
        "catalog_search_schema",
        "catalog_inspect_asset",
        "catalog_find_relationship_paths",
    } <= tool_view_names

    plan_capability = next(
        capability
        for capability in registry.capabilities
        if capability.id == "catalog.value_grounding.plan"
    )
    assert plan_capability.owner == "catalog"
    assert plan_capability.executor == "catalog.plan_value_grounding"
    assert plan_capability.output_evidence == frozenset(
        {"catalog.value_grounding.plan"}
    )
    assert plan_capability.access is AccessMode.METADATA_READ
    assert plan_capability.risk is RiskLevel.MEDIUM
    assert plan_capability.runtime_only is True
    assert plan_capability.side_effecting is False
    assert plan_capability.operation_types == frozenset(
        {"data.query", "query.plan", "schema.query"}
    )
    inspect_capability = next(
        capability
        for capability in registry.capabilities
        if capability.id == "catalog.asset.inspect"
    )
    assert "monitor.create" in inspect_capability.operation_types


def test_catalog_tool_views_expose_strict_required_schemas():
    views = {
        view.name: view for view in CatalogPlugin(auto_persist=False).get_tool_views()
    }

    search = views["catalog_search_schema"].parameters
    inspect = views["catalog_inspect_asset"].parameters
    paths = views["catalog_find_relationship_paths"].parameters

    assert search["required"] == ["store_id"]
    assert search["additionalProperties"] is False
    assert search["properties"]["limit"]["maximum"] == 50
    assert inspect["required"] == ["store_id", "asset_ref"]
    assert inspect["additionalProperties"] is False
    assert paths["required"] == ["store_id", "from_assets", "to_assets"]
    assert paths["additionalProperties"] is False


async def test_catalog_register_and_search_executors_return_typed_evidence():
    catalog = CatalogPlugin(auto_persist=False)
    registry = ExtensionRegistry()
    registry.register(catalog)
    operation = Operation(id="op-1", operation_type="schema.query")

    register_evidence = await _executor(registry, "catalog.register_source").execute(
        Task(
            id="task-register",
            operation_id=operation.id,
            capability_id="catalog.source.register",
            executor_id="catalog.register_source",
            input={
                "schema": _reference_schema(),
                "store_type": "sqlite",
                "store_id": "store:shop",
                "persist": False,
            },
        ),
        operation,
        {},
    )
    search_evidence = await _executor(registry, "catalog.search_schema").execute(
        Task(
            id="task-search",
            operation_id=operation.id,
            capability_id="catalog.schema.search",
            executor_id="catalog.search_schema",
            input={"store_id": "store:shop", "query": "customer email"},
        ),
        operation,
        {},
    )

    assert isinstance(register_evidence[0], Evidence)
    assert register_evidence[0].accepted is True
    assert register_evidence[0].kind == "catalog.source_registered"
    assert register_evidence[0].payload["store_id"] == "store:shop"
    assert search_evidence[0].kind == "schema.search_result"
    assert search_evidence[0].accepted is True
    assert search_evidence[0].payload["tables"][0]["name"] == "customers"


async def test_catalog_model_visible_executors_validate_required_args():
    catalog = CatalogPlugin(auto_persist=False)
    registry = ExtensionRegistry()
    registry.register(catalog)
    operation = Operation(id="op-validation", operation_type="schema.query")

    with pytest.raises(Exception, match="store_id is required"):
        await _executor(registry, "catalog.search_schema").execute(
            Task(
                id="task-search-missing-store",
                operation_id=operation.id,
                capability_id="catalog.schema.search",
                executor_id="catalog.search_schema",
                input={"query": "customers"},
            ),
            operation,
            {},
        )


async def test_catalog_inspect_relationship_and_profile_executors_return_evidence():
    catalog = CatalogPlugin(auto_persist=False)
    await catalog.register_schema(
        _reference_schema(),
        store_type="sqlite",
        store_id="store:shop",
        persist=False,
    )
    registry = ExtensionRegistry()
    registry.register(catalog)
    operation = Operation(id="op-2", operation_type="schema.query")

    inspect_evidence = await _executor(registry, "catalog.inspect_asset").execute(
        Task(
            id="task-inspect",
            operation_id=operation.id,
            capability_id="catalog.asset.inspect",
            executor_id="catalog.inspect_asset",
            input={"store_id": "store:shop", "asset_ref": "customers"},
        ),
        operation,
        {},
    )
    relationship_evidence = await _executor(
        registry, "catalog.find_relationship_paths"
    ).execute(
        Task(
            id="task-paths",
            operation_id=operation.id,
            capability_id="catalog.relationship_paths.find",
            executor_id="catalog.find_relationship_paths",
            input={
                "store_id": "store:shop",
                "from_assets": ["orders"],
                "to_assets": ["customers"],
            },
        ),
        operation,
        {},
    )
    profile_evidence = await _executor(registry, "catalog.profile_source").execute(
        Task(
            id="task-profile",
            operation_id=operation.id,
            capability_id="catalog.source.profile",
            executor_id="catalog.profile_source",
            input={"store_id": "store:shop"},
        ),
        operation,
        {},
    )

    assert inspect_evidence[0].kind == "schema.asset_profile"
    assert inspect_evidence[0].payload["asset"]["name"] == "customers"
    assert relationship_evidence[0].kind == "schema.relationship_path"
    assert relationship_evidence[0].payload["reachable"] is True
    assert profile_evidence[0].kind == "catalog.profile"
    assert profile_evidence[0].payload["table_count"] == 2


async def test_catalog_inspection_projects_bounded_column_traits_without_raw_values():
    catalog = CatalogPlugin(auto_persist=False)
    await catalog.register_schema(
        {
            "database_type": "sqlite",
            "database_name": "orders",
            "tables": [
                {
                    "name": "orders",
                    "columns": [
                        {
                            "name": "order_id",
                            "data_type": "INTEGER",
                            "is_primary_key": True,
                            "is_identity": True,
                            "is_generated": True,
                            "is_autoincrement": True,
                            "is_monotonic": True,
                            "identity_proof": {
                                "source_kind": "sqlite_schema",
                                "method": "sqlite_integer_primary_key",
                                "generated": True,
                                "autoincrement": True,
                                "monotonic": True,
                                "confidence": 1.0,
                            },
                        },
                        {
                            "name": "created_at",
                            "data_type": "TEXT",
                            "is_primary_key": False,
                        },
                    ],
                }
            ],
        },
        store_type="sqlite",
        store_id="store:orders",
        persist=False,
    )
    await catalog.register_column_value_profiles(
        "store:orders",
        [
            {
                "table": "orders",
                "column": "created_at",
                "profile_kind": "logical_type_validation",
                "profile_status": "profiled",
                "sampled": True,
                "top_values": [
                    {"value": "2026-01-02T10:00:00Z"},
                    {"value": "2026-01-03T11:00:00Z"},
                ],
                "logical_type": "timestamp",
                "logical_type_proof": {
                    "method": "bounded_value_profile",
                    "representation": "iso8601_utc_second",
                    "sample_size": 2,
                    "sample_limit": 64,
                    "all_values_matched": True,
                    "lexicographically_sortable": True,
                    "confidence": 0.95,
                    "values_exposed": False,
                },
            }
        ],
        source_evidence_id="sqlite-profile-created-at",
        persist=False,
    )
    registry = ExtensionRegistry()
    registry.register(catalog)
    operation = Operation(id="op-traits", operation_type="monitor.create")

    evidence = await _executor(registry, "catalog.inspect_asset").execute(
        Task(
            id="task-traits",
            operation_id=operation.id,
            capability_id="catalog.asset.inspect",
            executor_id="catalog.inspect_asset",
            input={"store_id": "store:orders", "asset_ref": "orders"},
        ),
        operation,
        {},
    )

    payload = evidence[0].payload
    assert payload["database_type"] == "sqlite"
    assert payload["database_name"] == "orders"
    assert payload["database_dialect"] == "sqlite"
    fields = {field["name"]: field for field in payload["fields"]}
    assert fields["order_id"]["physical_type"] == "INTEGER"
    assert fields["order_id"]["native_type"] == "integer"
    assert fields["order_id"]["is_identity"] is True
    assert fields["order_id"]["is_generated"] is True
    assert fields["order_id"]["is_autoincrement"] is True
    assert fields["order_id"]["is_monotonic"] is True
    assert fields["order_id"]["identity_proof"]["asset_ref"] == "orders"
    created_at = fields["created_at"]
    assert created_at["physical_type"] == "TEXT"
    assert created_at["native_type"] == "string"
    assert created_at["logical_type"] == "timestamp"
    assert created_at["logical_type_proof"]["owner"] == "catalog"
    assert created_at["logical_type_proof"]["source_kind"] == (
        "schema.column_value_profile"
    )
    assert created_at["logical_type_proof"]["profile_ref"] == "orders.created_at"
    assert created_at["logical_type_proof"]["column"] == "created_at"
    assert "column_value_hint" not in created_at
    serialized = str(payload)
    assert "2026-01-02T10:00:00Z" not in serialized
    assert "2026-01-03T11:00:00Z" not in serialized


def test_catalog_schema_normalization_preserves_and_merges_column_traits():
    search = Evidence(
        id="catalog-search-orders",
        kind="schema.search_result",
        owner="catalog",
        operation_id="op-normalize-traits",
        accepted=True,
        payload={
            "database_type": "sqlite",
            "store_id": "store:orders",
            "tables": [
                {
                    "name": "orders",
                    "fields": [
                        {"name": "created_at", "type": "TEXT"},
                        {"name": "order_id", "type": "INTEGER"},
                    ],
                }
            ],
        },
    )
    inspect = Evidence(
        id="catalog-inspect-orders",
        kind="schema.asset_profile",
        owner="catalog",
        operation_id="op-normalize-traits",
        accepted=True,
        payload={
            "database_type": "sqlite",
            "database_dialect": "sqlite",
            "store_id": "store:orders",
            "asset": {"name": "orders", "asset_ref": "orders"},
            "fields": [
                {
                    "name": "created_at",
                    "type": "TEXT",
                    "physical_type": "TEXT",
                    "native_type": "string",
                    "database_dialect": "sqlite",
                    "logical_type": "timestamp",
                    "logical_type_proof": {
                        "owner": "catalog",
                        "source_kind": "schema.column_value_profile",
                        "profile_ref": "orders.created_at",
                        "store_id": "store:orders",
                        "asset_ref": "orders",
                        "column": "created_at",
                        "database_dialect": "sqlite",
                        "representation": "iso8601_utc_second",
                        "all_values_matched": True,
                        "lexicographically_sortable": True,
                        "confidence": 0.95,
                    },
                },
                {
                    "name": "order_id",
                    "type": "INTEGER",
                    "is_primary_key": True,
                    "is_identity": True,
                    "is_generated": True,
                    "is_autoincrement": True,
                    "is_monotonic": True,
                    "identity_proof": {
                        "owner": "catalog",
                        "source_kind": "sqlite_schema",
                        "generated": True,
                        "autoincrement": True,
                        "monotonic": True,
                        "store_id": "store:orders",
                        "asset_ref": "orders",
                        "column": "order_id",
                        "database_dialect": "sqlite",
                    },
                },
            ],
        },
    )
    rejected = Evidence(
        id="catalog-rejected-conflict",
        kind="schema.asset_profile",
        owner="catalog",
        operation_id="op-normalize-traits",
        accepted=False,
        payload={
            "database_type": "sqlite",
            "store_id": "store:orders",
            "asset": {"name": "orders", "asset_ref": "orders"},
            "fields": [{"name": "status", "type": "TEXT", "logical_type": "timestamp"}],
        },
    )

    schema = catalog_schema_from_evidence((search, inspect, rejected), ())

    assert schema["database_type"] == "sqlite"
    assert schema["sql_dialect"] == "sqlite"
    table = schema["tables"][0]
    columns = {column["name"]: column for column in table["columns"]}
    assert set(columns) == {"created_at", "order_id"}
    assert columns["created_at"]["physical_type"] == "TEXT"
    assert columns["created_at"]["native_type"] == "string"
    assert columns["created_at"]["logical_type"] == "timestamp"
    assert columns["created_at"]["catalog_evidence"] == {
        "id": "catalog-inspect-orders",
        "kind": "schema.asset_profile",
        "owner": "catalog",
        "accepted": True,
        "store_id": "store:orders",
        "asset_ref": "orders",
        "column": "created_at",
    }
    assert columns["order_id"]["is_monotonic"] is True
    assert "column_value_hint" not in columns["created_at"]


async def test_catalog_executor_rejects_explicit_unsuccessful_asset_inspection():
    catalog = CatalogPlugin(auto_persist=False)
    await catalog.register_schema(
        _reference_schema(),
        store_type="sqlite",
        store_id="store:shop",
        persist=False,
    )
    registry = ExtensionRegistry()
    registry.register(catalog)
    operation = Operation(id="op-inspect-rejected", operation_type="monitor.create")

    evidence = await _executor(registry, "catalog.inspect_asset").execute(
        Task(
            id="task-inspect-rejected",
            operation_id=operation.id,
            capability_id="catalog.asset.inspect",
            executor_id="catalog.inspect_asset",
            input={"store_id": "store:shop", "asset_ref": "pending orders"},
        ),
        operation,
        {},
    )

    assert evidence[0].kind == "schema.asset_profile"
    assert evidence[0].accepted is False
    assert evidence[0].payload["success"] is False
    assert evidence[0].payload["asset_ref"] == "pending orders"


async def test_catalog_registers_searches_and_resolves_column_value_profiles():
    catalog = CatalogPlugin(auto_persist=False)
    await catalog.register_schema(
        {
            **_reference_schema(),
            "tables": [
                *_reference_schema()["tables"],
                {
                    "name": "shipments",
                    "columns": [
                        {"name": "id", "data_type": "INTEGER"},
                        {"name": "status", "data_type": "TEXT"},
                    ],
                },
            ],
        },
        store_type="sqlite",
        store_id="store:shop",
        persist=False,
    )
    registry = ExtensionRegistry()
    registry.register(catalog)
    operation = Operation(id="op-values", operation_type="data.query")

    registered = await _executor(registry, "catalog.register_column_values").execute(
        Task(
            id="task-values-register",
            operation_id=operation.id,
            capability_id="catalog.column_values.register",
            executor_id="catalog.register_column_values",
            input={
                "store_id": "store:shop",
                "profiles": [
                    {
                        "table": "shipments",
                        "column": "status",
                        "distinct_count": 2,
                        "top_values": [
                            {"value": "complete", "count": 4},
                            {"value": "pending", "count": 1},
                        ],
                    }
                ],
            },
        ),
        operation,
        {},
    )
    search = await _executor(registry, "catalog.search_column_values").execute(
        Task(
            id="task-values-search",
            operation_id=operation.id,
            capability_id="catalog.column_values.search",
            executor_id="catalog.search_column_values",
            input={"store_id": "store:shop", "query": "completed shipments"},
        ),
        operation,
        {},
    )
    hints = await _executor(registry, "catalog.resolve_column_value_hints").execute(
        Task(
            id="task-values-hints",
            operation_id=operation.id,
            capability_id="catalog.column_value_hints.resolve",
            executor_id="catalog.resolve_column_value_hints",
            input={"store_id": "store:shop", "prompt": "completed shipments"},
        ),
        operation,
        {},
    )
    inspected = catalog.inspect_asset("store:shop", "shipments")

    assert registered[0].kind == "schema.column_value_profile"
    assert registered[0].payload["canonical_path"] == "metadata.column_value_profiles"
    stored = catalog.get_schema("store:shop").metadata["column_value_profiles"]
    assert stored["shipments.status"]["top_values"][0]["value"] == "complete"
    assert search[0].kind == "schema.column_value_search_result"
    assert search[0].payload["profiles"][0]["profile_ref"] == "shipments.status"
    assert hints[0].kind == "schema.column_value_hint"
    assert (
        hints[0].payload["hints"][0]["candidate_mapping"]["closest_value"] == "complete"
    )
    status = next(field for field in inspected["fields"] if field["name"] == "status")
    assert status["column_value_hint"]["top_values"] == ["complete", "pending"]


async def test_catalog_value_grounding_plan_targets_validation_fact():
    _, registry = await _value_grounding_catalog()
    operation = Operation(id="op-value-grounding", operation_type="query.plan")

    evidence = await _executor(registry, "catalog.plan_value_grounding").execute(
        Task(
            id="task-value-grounding",
            operation_id=operation.id,
            capability_id="catalog.value_grounding.plan",
            executor_id="catalog.plan_value_grounding",
            input={
                "store_id": "store:shop",
                "prompt": "completed orders",
                "validation_facts": [
                    {
                        "kind": "unobserved_filter_literal",
                        "table": "orders",
                        "column": "status",
                        "literal": "completed",
                    }
                ],
            },
        ),
        operation,
        {},
    )

    assert evidence[0].kind == "catalog.value_grounding.plan"
    payload = evidence[0].payload
    assert payload["targets"] == [
        {
            "table": "orders",
            "column": "status",
            "reason": "validation_literal",
            "confidence": 0.95,
            "requires_profile_read": True,
            "source": {
                "kind": "validation_fact",
                "literal": "completed",
                "fact_kind": "unobserved_filter_literal",
            },
        }
    ]
    assert payload["skipped"] == []
    assert payload["diagnostics"] == {
        "profile_budget": 4,
        "target_count": 1,
        "skipped_count": 0,
    }


async def test_catalog_value_grounding_plan_targets_session_query_scope_filters():
    _, registry = await _value_grounding_catalog()
    operation = Operation(id="op-value-grounding-session", operation_type="query.plan")

    evidence = await _executor(registry, "catalog.plan_value_grounding").execute(
        Task(
            id="task-value-grounding-session",
            operation_id=operation.id,
            capability_id="catalog.value_grounding.plan",
            executor_id="catalog.plan_value_grounding",
            input={
                "store_id": "store:shop",
                "prompt": "same completed orders",
                "session_query_scopes": [
                    {
                        "operation_id": "op-prior",
                        "tables": ["orders"],
                        "filters": [
                            {
                                "column": "status",
                                "operator": "=",
                                "values": ["complete"],
                            }
                        ],
                    }
                ],
            },
        ),
        operation,
        {},
    )

    assert evidence[0].payload["targets"] == [
        {
            "table": "orders",
            "column": "status",
            "reason": "session_query_scope_filter",
            "confidence": 0.88,
            "requires_profile_read": True,
            "source": {
                "kind": "session_query_scope",
                "values": ["complete"],
                "operation_id": "op-prior",
            },
        }
    ]


async def test_catalog_value_grounding_plan_surfaces_existing_profiles_without_read():
    catalog, registry = await _value_grounding_catalog()
    await catalog.register_column_value_profiles(
        "store:shop",
        [
            {
                "table": "orders",
                "column": "status",
                "distinct_count": 2,
                "top_values": [
                    {"value": "complete", "count": 4},
                    {"value": "pending", "count": 1},
                ],
            }
        ],
    )
    operation = Operation(id="op-value-grounding-profile", operation_type="query.plan")

    evidence = await _executor(registry, "catalog.plan_value_grounding").execute(
        Task(
            id="task-value-grounding-profile",
            operation_id=operation.id,
            capability_id="catalog.value_grounding.plan",
            executor_id="catalog.plan_value_grounding",
            input={
                "store_id": "store:shop",
                "prompt": "completed orders",
                "profile_budget": 0,
            },
        ),
        operation,
        {},
    )

    assert evidence[0].payload["targets"] == [
        {
            "table": "orders",
            "column": "status",
            "reason": "catalog_profile",
            "confidence": 0.9,
            "requires_profile_read": False,
            "source": {
                "kind": "catalog_profile",
                "profile_ref": "orders.status",
            },
        }
    ]
    assert evidence[0].payload["skipped"] == []
    assert evidence[0].payload["diagnostics"] == {
        "profile_budget": 0,
        "target_count": 1,
        "skipped_count": 0,
    }


async def test_catalog_value_grounding_plan_skips_targets_over_profile_budget():
    _, registry = await _value_grounding_catalog()
    operation = Operation(id="op-value-grounding-budget", operation_type="query.plan")

    evidence = await _executor(registry, "catalog.plan_value_grounding").execute(
        Task(
            id="task-value-grounding-budget",
            operation_id=operation.id,
            capability_id="catalog.value_grounding.plan",
            executor_id="catalog.plan_value_grounding",
            input={
                "store_id": "store:shop",
                "prompt": "orders and customer email values",
                "profile_budget": 1,
                "targets": [
                    {"table": "orders", "column": "status"},
                    {"table": "customers", "column": "email"},
                ],
            },
        ),
        operation,
        {},
    )

    payload = evidence[0].payload
    assert payload["targets"] == [
        {
            "table": "orders",
            "column": "status",
            "reason": "explicit_target",
            "confidence": 1.0,
            "requires_profile_read": True,
            "source": {"kind": "explicit_target"},
        }
    ]
    assert payload["skipped"] == [
        {
            "table": "customers",
            "column": "email",
            "reason": "profile_budget_exhausted",
        }
    ]
    assert payload["diagnostics"] == {
        "profile_budget": 1,
        "target_count": 1,
        "skipped_count": 1,
    }


async def test_catalog_inspect_asset_omits_inline_values_for_stale_profiles():
    catalog = CatalogPlugin(auto_persist=False)
    await catalog.register_schema(
        {
            **_reference_schema(),
            "metadata": {"profile_key": "fresh-key"},
            "tables": [
                *_reference_schema()["tables"],
                {
                    "name": "shipments",
                    "columns": [
                        {"name": "id", "data_type": "INTEGER"},
                        {"name": "status", "data_type": "TEXT"},
                    ],
                },
            ],
        },
        store_type="sqlite",
        store_id="store:inline-stale",
        persist=False,
    )
    await catalog.register_column_value_profiles(
        "store:inline-stale",
        [
            {
                "table": "shipments",
                "column": "status",
                "distinct_count": 2,
                "top_values": [
                    {"value": "complete", "count": 4},
                    {"value": "pending", "count": 1},
                ],
            }
        ],
    )

    catalog.get_schema("store:inline-stale").metadata["profile_key"] = "new-key"
    inspected = catalog.inspect_asset("store:inline-stale", "shipments")
    hints = catalog.resolve_column_value_hints(
        "store:inline-stale",
        "completed shipments",
    )

    status = next(field for field in inspected["fields"] if field["name"] == "status")
    assert "column_value_hint" not in status
    assert hints["hints"][0]["profile_status"] == "stale"
    assert hints["hints"][0]["stale"] is True
    assert hints["hints"][0]["observed_values"] == []
    assert "candidate_mapping" not in hints["hints"][0]


async def test_catalog_profiles_preserve_fingerprint_and_mark_profile_key_mismatch_stale():
    catalog = CatalogPlugin(auto_persist=False)
    await catalog.register_schema(
        {
            **_reference_schema(),
            "metadata": {"profile_key": "fresh-key"},
            "tables": [
                *_reference_schema()["tables"],
                {
                    "name": "shipments",
                    "columns": [
                        {"name": "id", "data_type": "INTEGER"},
                        {"name": "status", "data_type": "TEXT"},
                    ],
                },
            ],
        },
        store_type="sqlite",
        store_id="store:fresh",
        persist=False,
    )
    await catalog.register_column_value_profiles(
        "store:fresh",
        [
            {
                "table": "shipments",
                "column": "status",
                "distinct_count": 2,
                "source_fingerprint": "fingerprint-1",
                "top_values": [
                    {"value": "complete", "count": 4},
                    {"value": "pending", "count": 1},
                ],
            }
        ],
    )
    stored = catalog.get_schema("store:fresh").metadata["column_value_profiles"]

    assert stored["shipments.status"]["source_fingerprint"] == "fingerprint-1"
    assert stored["shipments.status"]["policy"]["profile_key"] == "fresh-key"

    catalog.get_schema("store:fresh").metadata["profile_key"] = "new-key"
    search = catalog.search_column_value_profiles(
        "store:fresh",
        "completed shipments",
    )
    hints = catalog.resolve_column_value_hints(
        "store:fresh",
        "completed shipments",
    )

    assert search["profiles"][0]["profile_status"] == "stale"
    assert search["profiles"][0]["stale"] is True
    assert search["profiles"][0]["stale_reason"] == "profile_key_mismatch"
    assert search["profiles"][0]["source_fingerprint"] == "fingerprint-1"
    assert hints["hints"][0]["profile_status"] == "stale"
    assert hints["hints"][0]["stale"] is True


async def test_catalog_profiles_mark_schema_fingerprint_mismatch_stale():
    catalog = CatalogPlugin(auto_persist=False)
    await catalog.register_schema(
        {
            **_reference_schema(),
            "metadata": {"profile_key": "fresh-key"},
            "tables": [
                *_reference_schema()["tables"],
                {
                    "name": "shipments",
                    "columns": [
                        {"name": "id", "data_type": "INTEGER"},
                        {"name": "status", "data_type": "TEXT"},
                    ],
                },
            ],
        },
        store_type="sqlite",
        store_id="store:fresh-schema",
        persist=False,
    )
    await catalog.register_column_value_profiles(
        "store:fresh-schema",
        [
            {
                "table": "shipments",
                "column": "status",
                "distinct_count": 2,
                "source_fingerprint": "fingerprint-1",
                "top_values": [
                    {"value": "complete", "count": 4},
                    {"value": "pending", "count": 1},
                ],
            }
        ],
    )
    schema = catalog.get_schema("store:fresh-schema")
    stored = schema.metadata["column_value_profiles"]

    assert stored["shipments.status"]["policy"]["profile_key"] == "fresh-key"
    assert stored["shipments.status"]["policy"]["schema_fingerprint"]

    shipments = next(table for table in schema.tables if table.name == "shipments")
    shipments.columns.append(
        NormalizedColumn(
            name="carrier",
            type="TEXT",
            nullable=True,
            is_primary_key=False,
        )
    )

    search = catalog.search_column_value_profiles(
        "store:fresh-schema",
        "completed shipments",
    )
    hints = catalog.resolve_column_value_hints(
        "store:fresh-schema",
        "completed shipments",
    )

    assert search["profiles"][0]["profile_status"] == "stale"
    assert search["profiles"][0]["stale"] is True
    assert search["profiles"][0]["stale_reason"] == "schema_fingerprint_mismatch"
    assert hints["hints"][0]["profile_status"] == "stale"
    assert hints["hints"][0]["stale"] is True


async def test_catalog_source_registration_preserves_existing_column_value_profiles():
    catalog = CatalogPlugin(auto_persist=False)
    await catalog.register_schema(
        {
            **_reference_schema(),
            "tables": [
                *_reference_schema()["tables"],
                {
                    "name": "shipments",
                    "columns": [
                        {"name": "id", "data_type": "INTEGER"},
                        {"name": "status", "data_type": "TEXT"},
                    ],
                },
            ],
        },
        store_type="sqlite",
        store_id="store:preserve",
        persist=False,
    )
    await catalog.register_column_value_profiles(
        "store:preserve",
        [
            {
                "table": "shipments",
                "column": "status",
                "distinct_count": 2,
                "top_values": [{"value": "complete", "count": 4}],
            }
        ],
    )

    await catalog.register_schema(
        {
            **_reference_schema(),
            "tables": [
                *_reference_schema()["tables"],
                {
                    "name": "shipments",
                    "columns": [
                        {"name": "id", "data_type": "INTEGER"},
                        {"name": "status", "data_type": "TEXT"},
                    ],
                },
            ],
        },
        store_type="sqlite",
        store_id="store:preserve",
        persist=False,
    )

    stored = catalog.get_schema("store:preserve").metadata["column_value_profiles"]
    assert stored["shipments.status"]["top_values"][0]["value"] == "complete"


async def test_catalog_discovery_compare_diagram_and_context_providers():
    catalog = CatalogPlugin(auto_persist=False)
    registry = ExtensionRegistry()
    registry.register(catalog)
    operation = Operation(id="op-3", operation_type="schema.query")

    inventory = await _executor(registry, "catalog.discover_infrastructure").execute(
        Task(
            id="task-discover",
            operation_id=operation.id,
            capability_id="catalog.infrastructure.discover",
            executor_id="catalog.discover_infrastructure",
            input={},
        ),
        operation,
        {},
    )
    comparison = await _executor(registry, "catalog.compare_schema").execute(
        Task(
            id="task-compare",
            operation_id=operation.id,
            capability_id="catalog.schema.compare",
            executor_id="catalog.compare_schema",
            input={"schema_a": _reference_schema(), "schema_b": _reference_schema()},
        ),
        operation,
        {},
    )
    diagram = await _executor(registry, "catalog.export_diagram").execute(
        Task(
            id="task-diagram",
            operation_id=operation.id,
            capability_id="catalog.diagram.export",
            executor_id="catalog.export_diagram",
            input={"schema": _reference_schema(), "format": "mermaid"},
        ),
        operation,
        {},
    )
    context = await registry.context_providers[0].render(
        {},
        ContextAudience.OPERATION_INSPECTOR,
        token_budget=500,
    )

    assert inventory[0].kind == "catalog.infrastructure_inventory"
    assert inventory[0].payload["store_count"] == 0
    assert comparison[0].kind == "schema.comparison"
    assert diagram[0].kind == "schema.diagram"
    assert context.owner == "catalog"
    assert "Catalog has" in context.content


async def test_db_runtime_inspect_reports_catalog_declarations():
    runtime = DbRuntime(plugins=(CatalogPlugin(auto_persist=False),))

    inspection = await runtime.inspect()

    assert inspection.plugin_ids == ("catalog",)
    assert "catalog:catalog.schema.search" in inspection.capability_ids
    assert "catalog.search_schema" in inspection.executor_ids
    assert "catalog:schema.search_result" in inspection.evidence_schema_kinds
