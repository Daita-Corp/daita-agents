"""Component-owned tests split from ``test_catalog_schema.py``."""

from __future__ import annotations

from tests.catalog._schema_support import (
    _OBSERVED_AT,
    CATALOG_MAX_LIMIT,
    CATALOG_RESOURCE_ID_MAX_CHARACTERS,
    CATALOG_SCHEMA_CAPABILITY_ID,
    CATALOG_SCHEMA_EVIDENCE_KIND,
    CATALOG_SCHEMA_MAX_RESOURCE_IDS,
    CATALOG_SEARCH_CAPABILITY_ID,
    CATALOG_SEARCH_REQUEST_DEFAULT_LIMIT,
    CATALOG_SEARCH_REQUEST_MAX_QUERY_CHARACTERS,
    CATALOG_SOURCE_ID_MAX_CHARACTERS,
    CATALOG_TOOL_DEFAULT_LIMIT,
    CATALOG_TOOL_QUERY_MAX_CHARACTERS,
    CATALOG_TRAVERSE_CAPABILITY_ID,
    Agent,
    CapabilityInputError,
    CapabilityRegistry,
    CapabilityRuntime,
    CatalogResourceNotFoundError,
    CatalogSchemaRequest,
    CatalogSearchRequest,
    CatalogStoreError,
    FrozenJsonObject,
    Mapping,
    ModelProfile,
    Path,
    ResourceKind,
    RunInput,
    SQLiteSource,
    ToolCall,
    ToolExecution,
    ToolOutput,
    ToolOutputValidationError,
    _fixture_database,
    _InventoryProvider,
    _mapping,
    _mapping_sequence,
    _schema,
    canonical_json,
    cast,
    catalog_service,
    execute_projected,
    pytest,
    sqlite3,
    workspace_for,
)


async def test_schema_slice_spans_eight_tables_with_exact_compact_structure(
    tmp_path: Path,
):
    database = tmp_path / "fixture.sqlite"
    _fixture_database(database)
    agent = await Agent.create(
        "catalog-schema-eight", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        resources = await agent.list_catalog_resources(source_id=source.id)
        projection = await _schema(
            agent,
            resource_ids=tuple(resource.id for resource in resources),
        )

        projected_resources = _mapping_sequence(projection["resources"])
        assert tuple(item["name"] for item in projected_resources) == (
            "main.customers",
            "main.order_items",
            "main.orders",
            "main.payments",
            "main.products",
            "main.regions",
            "main.shipments",
            "main.support_tickets",
        )
        assert projection["total_matches"] == 8
        assert projection["trust_classification"] == "untrusted_external_data"
        truncation = projection["truncation"]
        assert isinstance(truncation, FrozenJsonObject)
        assert truncation.to_dict() == {
            "columns": False,
            "paths": False,
            "primary_key_fields": False,
            "reason": None,
            "relationships": False,
            "resources": False,
            "structural_facts": False,
            "unique_key_fields": False,
        }

        by_name = {str(item["name"]): item for item in projected_resources}
        orders = by_name["main.orders"]
        assert tuple(
            dict(column) for column in _mapping_sequence(orders["columns"])
        ) == (
            {"name": "order_id", "nullable": False, "type": "INTEGER"},
            {"name": "customer_id", "nullable": False, "type": "INTEGER"},
            {"name": "ordered_at", "nullable": False, "type": "TEXT"},
            {"name": "status", "nullable": False, "type": "TEXT"},
            {"name": "sales_channel", "nullable": False, "type": "TEXT"},
            {"name": "subtotal", "nullable": False, "type": "NUMERIC"},
            {"name": "tax_amount", "nullable": False, "type": "NUMERIC"},
            {"name": "total_amount", "nullable": False, "type": "NUMERIC"},
        )
        assert orders["primary_key_fields"] == ("order_id",)
        assert by_name["main.regions"]["unique_key_fields"] == (("region_name",),)
        assert by_name["main.customers"]["unique_key_fields"] == (("email",),)
        assert by_name["main.products"]["unique_key_fields"] == (("sku",),)
        assert by_name["main.payments"]["unique_key_fields"] == (("order_id",),)
        assert by_name["main.shipments"]["unique_key_fields"] == (("order_id",),)

        relationships = _mapping_sequence(projection["relationships"])
        relationship_ids = tuple(item["relationship_id"] for item in relationships)
        assert len(relationship_ids) == len(set(relationship_ids)) == 8
        pairs = {
            (
                by_id[item["from_resource_id"]]["name"],
                by_id[item["to_resource_id"]]["name"],
                tuple(
                    (pair["source_field"], pair["target_field"])
                    for pair in _mapping_sequence(item["field_pairs"])
                ),
            )
            for item in relationships
            for by_id in (
                {resource["resource_id"]: resource for resource in projected_resources},
            )
        }
        assert pairs == {
            ("main.customers", "main.regions", (("region_code", "region_code"),)),
            ("main.order_items", "main.orders", (("order_id", "order_id"),)),
            ("main.order_items", "main.products", (("product_id", "product_id"),)),
            ("main.orders", "main.customers", (("customer_id", "customer_id"),)),
            ("main.payments", "main.orders", (("order_id", "order_id"),)),
            ("main.shipments", "main.orders", (("order_id", "order_id"),)),
            (
                "main.support_tickets",
                "main.customers",
                (("customer_id", "customer_id"),),
            ),
            (
                "main.support_tickets",
                "main.orders",
                (("order_id", "order_id"),),
            ),
        }
        assert all(item["provenance"] == "connector" for item in relationships)
        assert all(
            item["from_resource_revision"] and item["to_resource_revision"]
            for item in relationships
        )
        sources = _mapping_sequence(projection["sources"])
        assert len(sources) == 1
        assert sources[0]["source_id"] == source.id
        assert isinstance(sources[0]["source_revision"], str)
        assert isinstance(sources[0]["sync_id"], str)
    finally:
        await agent.close()


async def test_schema_capability_validates_output_and_is_smaller_than_inspections(
    tmp_path: Path,
):
    database = tmp_path / "capability.sqlite"
    _fixture_database(database)
    agent = await Agent.create(
        "catalog-schema-capability", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        resources = await agent.list_catalog_resources(source_id=source.id)
        registry: CapabilityRegistry = agent._embedded._capabilities
        view, capability = registry.resolve_tool("catalog_schema")
        assert view.capability_id == capability.id
        _, executor = registry.resolve_execution(capability.id)
        output = await executor.execute(
            ToolExecution(
                run_id="schema-capability",
                call_id="schema-capability-call",
                capability_id=capability.id,
                arguments={
                    "resource_ids": tuple(resource.id for resource in resources),
                    "limit": 50,
                    "include_relationships": True,
                },
            )
        )
        assert output.kind == CATALOG_SCHEMA_EVIDENCE_KIND
        assert registry.validate_output(capability.id, output) == output
        with pytest.raises(ToolOutputValidationError):
            registry.validate_output(
                capability.id,
                ToolOutput(
                    kind=CATALOG_SCHEMA_EVIDENCE_KIND,
                    data={"resources": (), "relationships": ()},
                ),
            )

        schema_bytes = len(canonical_json(output.data).encode("utf-8"))
        inspections = tuple(
            [
                await agent._embedded._catalog_service.inspect_resource(
                    agent.id,
                    resource.id,
                )
                for resource in resources
            ]
        )
        inspection_bytes = sum(
            len(canonical_json(inspection).encode("utf-8"))
            for inspection in inspections
        )
        assert schema_bytes < inspection_bytes * 0.7
        assert len(_mapping_sequence(output.data["resources"])) == 8
        output_truncation = output.data["truncation"]
        assert isinstance(output_truncation, Mapping)
        assert output_truncation["resources"] is False
        assert len(
            {
                item["relationship_id"]
                for item in _mapping_sequence(output.data["relationships"])
            }
        ) == len(_mapping_sequence(output.data["relationships"]))
    finally:
        await agent.close()


async def test_catalog_search_capability_exposes_correct_returned_and_scoped_counts(
    tmp_path: Path,
):
    database = tmp_path / "search-capability-counts.sqlite"
    _fixture_database(database)
    agent = await Agent.create(
        "catalog-search-capability-counts",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.attach(SQLiteSource(database))
        registry: CapabilityRegistry = agent._embedded._capabilities
        _view, capability = registry.resolve_tool("catalog_search")
        _capability, executor = registry.resolve_execution(capability.id)
        output = await executor.execute(
            ToolExecution(
                run_id="search-capability-counts",
                call_id="search-capability-counts-call",
                capability_id=capability.id,
                arguments={"query": "table", "limit": 1},
            )
        )

        assert output.data["returned_count"] == 1
        assert output.data["total_matches"] == 8
        assert output.data["truncated"] is True
        match_outcome = output.data["match_outcome"]
        assert isinstance(match_outcome, Mapping)
        assert match_outcome["binding_status"] == "ambiguous"
        assert match_outcome["source_status"] == "unique"
        assert match_outcome["candidate_count"] == 8
        assert registry.validate_output(capability.id, output) == output

        for accepted_outcome in (
            {
                "binding_status": "unique",
                "source_status": "unique",
                "evidence_tier": "exact_resource",
                "candidate_count": 1,
                "candidate_bindings": (
                    {"source_id": "source-a", "resource_id": "resource-a"},
                ),
                "omitted_candidate_count": 0,
                "ambiguity_reasons": (),
                "assessment_provenance": "catalog_service",
                "trust_classification": "untrusted_external_data",
            },
            {
                "binding_status": "no_match",
                "source_status": "no_match",
                "evidence_tier": "none",
                "candidate_count": 0,
                "candidate_bindings": (),
                "omitted_candidate_count": 0,
                "ambiguity_reasons": (),
                "assessment_provenance": "catalog_service",
                "trust_classification": "untrusted_external_data",
            },
        ):
            accepted = output.data.to_dict()
            accepted["match_outcome"] = accepted_outcome
            validated = ToolOutput(kind=output.kind, data=accepted)
            assert registry.validate_output(capability.id, validated) == validated

        malformed = output.data.to_dict()
        malformed_outcome = dict(cast(Mapping[str, object], malformed["match_outcome"]))
        malformed_outcome["binding_status"] = "confident"
        malformed["match_outcome"] = malformed_outcome
        with pytest.raises(ToolOutputValidationError):
            registry.validate_output(
                capability.id,
                ToolOutput(kind=output.kind, data=malformed),
            )
    finally:
        await agent.close()


async def test_catalog_model_facing_bounds_and_internal_search_contracts_are_explicit(
    tmp_path: Path,
):
    agent = await Agent.create(
        "catalog-input-contracts", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        registry: CapabilityRegistry = agent._embedded._capabilities
        search_definition = registry.tool_definition("catalog_search")
        _search_view, search_capability = registry.resolve_tool("catalog_search")
        schema_definition = registry.tool_definition("catalog_schema")
        inspect_definition = registry.tool_definition("catalog_inspect")

        search_properties = search_definition.input_schema["properties"]
        schema_properties = schema_definition.input_schema["properties"]
        inspect_properties = inspect_definition.input_schema["properties"]
        assert isinstance(search_properties, Mapping)
        assert isinstance(schema_properties, Mapping)
        assert isinstance(inspect_properties, Mapping)
        search_output_properties = search_capability.output_schema["properties"]
        search_output_required = search_capability.output_schema["required"]
        assert isinstance(search_output_properties, Mapping)
        assert isinstance(search_output_required, (tuple, list))
        assert canonical_json(
            search_output_properties["returned_count"]
        ) == canonical_json({"type": "integer"})
        assert "returned_count" in search_output_required
        match_outcome_rule = search_output_properties["match_outcome"]
        assert isinstance(match_outcome_rule, Mapping)
        outcome_properties = match_outcome_rule["properties"]
        outcome_required = match_outcome_rule["required"]
        assert isinstance(outcome_properties, Mapping)
        assert isinstance(outcome_required, (tuple, list))
        assert "authority" not in outcome_properties
        assert canonical_json(
            outcome_properties["assessment_provenance"]
        ) == canonical_json(
            {
                "type": "string",
                "enum": ("catalog_service",),
            }
        )
        assert set(outcome_required) == set(outcome_properties)

        query_rule = {
            "type": "string",
            "minLength": 1,
            "maxLength": CATALOG_TOOL_QUERY_MAX_CHARACTERS,
        }
        source_rule = {
            "type": "string",
            "minLength": 1,
            "maxLength": CATALOG_SOURCE_ID_MAX_CHARACTERS,
        }
        limit_rule = {
            "type": "integer",
            "minimum": 1,
            "maximum": CATALOG_MAX_LIMIT,
            "default": CATALOG_TOOL_DEFAULT_LIMIT,
        }
        assert canonical_json(search_properties["query"]) == canonical_json(query_rule)
        assert canonical_json(search_properties["source_id"]) == canonical_json(
            source_rule
        )
        assert canonical_json(search_properties["limit"]) == canonical_json(limit_rule)
        assert canonical_json(search_properties["resource_kinds"]) == canonical_json(
            {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [kind.value for kind in ResourceKind],
                },
                "maxItems": len(ResourceKind),
                "uniqueItems": True,
                "default": [],
            }
        )

        assert "Provide a non-empty query or resource_ids" in (
            schema_definition.description
        )
        assert canonical_json(schema_properties["query"]) == canonical_json(query_rule)
        assert canonical_json(schema_properties["source_id"]) == canonical_json(
            source_rule
        )
        assert canonical_json(schema_properties["limit"]) == canonical_json(limit_rule)
        assert canonical_json(schema_properties["resource_ids"]) == canonical_json(
            {
                "type": "array",
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": CATALOG_RESOURCE_ID_MAX_CHARACTERS,
                },
                "maxItems": CATALOG_SCHEMA_MAX_RESOURCE_IDS,
                "uniqueItems": True,
                "default": [],
            }
        )
        assert canonical_json(
            schema_properties["include_relationships"]
        ) == canonical_json(
            {
                "type": "boolean",
                "default": True,
            }
        )
        assert canonical_json(inspect_properties["resource_id"]) == canonical_json(
            {
                "type": "string",
                "minLength": 1,
                "maxLength": CATALOG_RESOURCE_ID_MAX_CHARACTERS,
            }
        )

        for capability_id in (
            CATALOG_SEARCH_CAPABILITY_ID,
            CATALOG_SCHEMA_CAPABILITY_ID,
        ):
            with pytest.raises(CapabilityInputError) as invalid_limit:
                registry.validate_arguments(
                    capability_id,
                    {"query": "orders", "limit": 100},
                )
            assert invalid_limit.value.code == "invalid_argument_value"
            assert invalid_limit.value.details.to_dict() == {
                "constraint": "maximum",
                "name": "limit",
            }

        with pytest.raises(CapabilityInputError) as invalid_kind:
            registry.validate_arguments(
                CATALOG_SEARCH_CAPABILITY_ID,
                {"query": "orders", "resource_kinds": ("spreadsheet",)},
            )
        assert invalid_kind.value.details.to_dict() == {
            "constraint": "enum",
            "name": "resource_kinds[0]",
        }

        with pytest.raises(CapabilityInputError) as duplicate_resources:
            registry.validate_arguments(
                CATALOG_SCHEMA_CAPABILITY_ID,
                {
                    "resource_ids": ("resource-a", "resource-a"),
                    "limit": CATALOG_TOOL_DEFAULT_LIMIT,
                },
            )
        assert duplicate_resources.value.details.to_dict() == {
            "constraint": "uniqueItems",
            "name": "resource_ids",
        }

        assert CatalogSearchRequest(agent_id="agent", query="orders").limit == (
            CATALOG_SEARCH_REQUEST_DEFAULT_LIMIT
        )
        assert CatalogSchemaRequest(agent_id="agent", query="orders").limit == (
            CATALOG_TOOL_DEFAULT_LIMIT
        )
        internal_context_query = "q" * (CATALOG_TOOL_QUERY_MAX_CHARACTERS + 1)
        assert (
            CatalogSearchRequest(agent_id="agent", query=internal_context_query).query
            == internal_context_query
        )
        oversized_search_query = "q" * (CATALOG_SEARCH_REQUEST_MAX_QUERY_CHARACTERS + 1)
        with pytest.raises(ValueError, match="catalog search query exceeds"):
            CatalogSearchRequest(agent_id="agent", query=oversized_search_query)
        oversized_tool_query = "q" * (CATALOG_TOOL_QUERY_MAX_CHARACTERS + 1)
        with pytest.raises(ValueError, match="catalog schema query exceeds"):
            CatalogSchemaRequest(agent_id="agent", query=oversized_tool_query)
    finally:
        await agent.close()


async def test_catalog_schema_invalid_input_never_reaches_catalog_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "catalog-invalid-input.sqlite"
    _fixture_database(database)
    provider = _InventoryProvider()
    profile = ModelProfile(
        id=provider.provider_id,
        context_window_tokens=80_000,
        max_output_tokens=2_000,
        supports_tools=True,
    )
    agent = await Agent.create(
        "catalog-invalid-input",
        root=tmp_path,
        model=provider,
        model_profile=profile,
        workspace=workspace_for(tmp_path),
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        catalog_execution_calls = 0

        async def unexpected_schema_execution(*args: object, **kwargs: object):
            nonlocal catalog_execution_calls
            del args, kwargs
            catalog_execution_calls += 1
            raise AssertionError("invalid catalog schema input reached execution")

        monkeypatch.setattr(
            catalog_service.CatalogService,
            "schema_slice",
            unexpected_schema_execution,
        )
        loop = agent._embedded._loop
        assert loop is not None
        runtime = cast(CapabilityRuntime, loop._tools)
        run = RunInput(
            id="catalog-invalid-input-run",
            agent_id=agent.id,
            message="inspect catalog",
            created_at=_OBSERVED_AT,
            conversation_id="catalog-invalid-input-conversation",
            source_scope_ids=(source.id,),
        )
        results = await execute_projected(
            runtime,
            run,
            (
                ToolCall(
                    id="invalid-limit",
                    name="catalog_schema",
                    arguments={"source_id": source.id, "limit": 100},
                ),
                ToolCall(
                    id="missing-selector",
                    name="catalog_schema",
                    arguments={
                        "source_id": source.id,
                        "limit": CATALOG_MAX_LIMIT,
                    },
                ),
            ),
        )

        error_codes = tuple(
            _mapping(_mapping(result.output)["error"])["code"] for result in results
        )
        assert error_codes == ("invalid_argument_value", "catalog_invalid_schema")
        assert catalog_execution_calls == 0
    finally:
        await agent.close()


async def test_catalog_tool_contract_exposes_and_enforces_progressive_bounds(
    tmp_path: Path,
):
    agent = await Agent.create(
        "catalog-progressive-contract", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        registry: CapabilityRegistry = agent._embedded._capabilities
        schema_definition = registry.tool_definition("catalog_schema")
        traversal_definition = registry.tool_definition("catalog_traverse")

        assert "SQL schema, bridges, paths" in schema_definition.description
        assert "Do not use with catalog_traverse" in schema_definition.description
        assert "later step" in traversal_definition.description
        assert "Do not call alongside catalog_schema" in (
            traversal_definition.description
        )

        schema_properties = schema_definition.input_schema["properties"]
        assert isinstance(schema_properties, Mapping)
        assert canonical_json(schema_properties["max_join_depth"]) == canonical_json(
            {
                "default": 3,
                "maximum": 6,
                "minimum": 1,
                "type": "integer",
            }
        )
        with pytest.raises(CapabilityInputError):
            registry.validate_arguments(
                CATALOG_SCHEMA_CAPABILITY_ID,
                {"query": "orders", "max_join_depth": True},
            )
        with pytest.raises(CapabilityInputError):
            registry.validate_arguments(
                CATALOG_SCHEMA_CAPABILITY_ID,
                {"query": "orders", "max_join_depth": 7},
            )
        with pytest.raises(ValueError, match="max_join_depth"):
            CatalogSchemaRequest(
                agent_id="agent",
                query="orders",
                max_join_depth=0,
            )
        with pytest.raises(ValueError, match="max_join_depth"):
            CatalogSchemaRequest(
                agent_id="agent",
                query="orders",
                max_join_depth=True,
            )

        traversal_properties = traversal_definition.input_schema["properties"]
        assert isinstance(traversal_properties, Mapping)
        endpoint_rule = {
            "items": {
                "maxLength": 256,
                "minLength": 1,
                "type": "string",
            },
            "maxItems": 16,
            "minItems": 1,
            "type": "array",
            "uniqueItems": True,
        }
        assert canonical_json(
            traversal_properties["from_resource_ids"]
        ) == canonical_json(endpoint_rule)
        assert canonical_json(
            traversal_properties["to_resource_ids"]
        ) == canonical_json(endpoint_rule)
        assert canonical_json(
            traversal_properties["relationship_kinds"]
        ) == canonical_json(
            {
                "default": [],
                "items": {
                    "enum": [
                        "contains",
                        "references",
                        "derived_from",
                        "produces",
                        "writes_to",
                        "reads_from",
                        "observes",
                    ],
                    "type": "string",
                },
                "maxItems": 7,
                "type": "array",
                "uniqueItems": True,
            }
        )
        for name, default, maximum in (
            ("max_depth", 4, 6),
            ("max_paths", 5, 8),
            ("max_nodes", 100, 1_000),
            ("max_edges", 200, 2_000),
        ):
            assert canonical_json(traversal_properties[name]) == canonical_json(
                {
                    "default": default,
                    "maximum": maximum,
                    "minimum": 1,
                    "type": "integer",
                }
            )

        with pytest.raises(CapabilityInputError) as invalid_paths:
            registry.validate_arguments(
                CATALOG_TRAVERSE_CAPABILITY_ID,
                {
                    "from_resource_ids": ("resource-a",),
                    "to_resource_ids": ("resource-b",),
                    "max_paths": 20,
                },
            )
        assert invalid_paths.value.code == "invalid_argument_value"
        assert invalid_paths.value.details.to_dict() == {
            "constraint": "maximum",
            "name": "max_paths",
        }

        with pytest.raises(CapabilityInputError) as duplicate_endpoints:
            registry.validate_arguments(
                CATALOG_TRAVERSE_CAPABILITY_ID,
                {
                    "from_resource_ids": ("resource-a", "resource-a"),
                    "to_resource_ids": ("resource-b",),
                },
            )
        assert duplicate_endpoints.value.code == "invalid_argument_value"
        assert duplicate_endpoints.value.details.to_dict() == {
            "constraint": "uniqueItems",
            "name": "from_resource_ids",
        }

        with pytest.raises(CapabilityInputError) as oversized_endpoints:
            registry.validate_arguments(
                CATALOG_TRAVERSE_CAPABILITY_ID,
                {
                    "from_resource_ids": tuple(
                        f"resource-{index}" for index in range(17)
                    ),
                    "to_resource_ids": ("resource-b",),
                },
            )
        assert oversized_endpoints.value.code == "invalid_argument_value"
        assert oversized_endpoints.value.details.to_dict() == {
            "constraint": "maxItems",
            "name": "from_resource_ids",
        }

        with pytest.raises(CapabilityInputError) as oversized_endpoint:
            registry.validate_arguments(
                CATALOG_TRAVERSE_CAPABILITY_ID,
                {
                    "from_resource_ids": ("r" * 257,),
                    "to_resource_ids": ("resource-b",),
                },
            )
        assert oversized_endpoint.value.code == "invalid_argument_value"
        assert oversized_endpoint.value.details.to_dict() == {
            "constraint": "maxLength",
            "name": "from_resource_ids[0]",
        }

        with pytest.raises(CapabilityInputError) as invalid_relationship_kind:
            registry.validate_arguments(
                CATALOG_TRAVERSE_CAPABILITY_ID,
                {
                    "from_resource_ids": ("resource-a",),
                    "to_resource_ids": ("resource-b",),
                    "relationship_kinds": ("invented",),
                },
            )
        assert invalid_relationship_kind.value.code == "invalid_argument_value"
        assert invalid_relationship_kind.value.details.to_dict() == {
            "constraint": "enum",
            "name": "relationship_kinds[0]",
        }

    finally:
        await agent.close()


async def test_structural_search_ranks_direct_matches_before_one_hop_neighbors(
    tmp_path: Path,
):
    database = tmp_path / "structural.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE needle (id INTEGER PRIMARY KEY);
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                needle_code TEXT NOT NULL UNIQUE
            );
            CREATE TABLE neighbor (
                id INTEGER PRIMARY KEY,
                order_id INTEGER NOT NULL REFERENCES orders(id)
            );
            """)
    agent = await Agent.create(
        "catalog-structural-search", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        await agent.attach(SQLiteSource(database))
        result = await agent.search_catalog(
            CatalogSearchRequest(
                agent_id=agent.id,
                query="needle",
                limit=3,
            )
        )
        assert tuple(hit.name for hit in result.hits) == (
            "needle",
            "orders",
            "neighbor",
        )
        assert result.hits[0].match_reasons == ("resource_name_exact",)
        assert result.hits[1].match_reasons == ("structural_field_contains",)
        assert "column:needle_code" in result.hits[1].matched_fields
        assert result.hits[2].match_reasons == ("relationship_neighbor",)
        assert result.match_outcome.binding_status == "unique"
        assert result.match_outcome.candidate_count == 1
        assert result.match_outcome.evidence_tier == "exact_resource"
        no_synonym = await agent.search_catalog(
            CatalogSearchRequest(
                agent_id=agent.id,
                query="margin",
                limit=3,
            )
        )
        assert no_synonym.total_matches == 0
        assert no_synonym.total_candidates == 3
        assert len(no_synonym.hits) == 3
        assert all(
            hit.match_reasons == ("unmatched_fallback",) for hit in no_synonym.hits
        )
        assert no_synonym.match_outcome.binding_status == "no_match"
        assert no_synonym.match_outcome.source_status == "no_match"
        assert no_synonym.match_outcome.evidence_tier == "none"
        assert no_synonym.match_outcome.candidate_count == 0
        assert no_synonym.match_outcome.candidate_bindings == ()
    finally:
        await agent.close()


async def test_schema_scope_is_strict_current_and_source_file_is_not_read(
    tmp_path: Path,
):
    first_database = tmp_path / "first.sqlite"
    second_database = tmp_path / "second.sqlite"
    with sqlite3.connect(first_database) as connection:
        connection.execute(
            "CREATE TABLE first_table (id INTEGER PRIMARY KEY, hidden_term TEXT)"
        )
    with sqlite3.connect(second_database) as connection:
        connection.execute("CREATE TABLE second_table (id INTEGER PRIMARY KEY)")
    agent = await Agent.create(
        "catalog-schema-scope", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        first = await agent.attach(SQLiteSource(first_database))
        second = await agent.attach(SQLiteSource(second_database))
        first_resource = (await agent.list_catalog_resources(source_id=first.id))[0]
        second_resource = (await agent.list_catalog_resources(source_id=second.id))[0]

        first_database.rename(tmp_path / "first-source-is-unavailable.sqlite")
        projection = await _schema(
            agent,
            resource_ids=(first_resource.id,),
            source_id=first.id,
        )
        assert len(_mapping_sequence(projection["resources"])) == 1

        with pytest.raises(CatalogStoreError):
            await _schema(
                agent,
                resource_ids=(second_resource.id,),
                source_id=first.id,
            )
        with pytest.raises(CatalogResourceNotFoundError):
            await _schema(agent, resource_ids=("catalog-resource:sha256:" + "0" * 64,))

        await agent.detach(first.id)
        with pytest.raises(CatalogStoreError):
            await _schema(agent, query="hidden_term", source_id=first.id)
    finally:
        await agent.close()


async def test_schema_and_structural_search_order_ignore_catalog_insertion_order(
    tmp_path: Path,
):
    first_database = tmp_path / "ordered.sqlite"
    second_database = tmp_path / "reverse.sqlite"
    _fixture_database(first_database)
    _fixture_database(second_database, reverse=True)
    first_agent = await Agent.create(
        "catalog-order-first", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    second_agent = await Agent.create(
        "catalog-order-second", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        first_source = await first_agent.attach(SQLiteSource(first_database))
        second_source = await second_agent.attach(SQLiteSource(second_database))
        first_search = await first_agent.search_catalog(
            CatalogSearchRequest(
                agent_id=first_agent.id,
                query="id",
                limit=50,
            )
        )
        second_search = await second_agent.search_catalog(
            CatalogSearchRequest(
                agent_id=second_agent.id,
                query="id",
                limit=50,
            )
        )
        assert tuple(
            (
                hit.name,
                hit.match_reasons,
                (
                    ()
                    if hit.match_reasons == ("relationship_neighbor",)
                    else hit.matched_fields
                ),
            )
            for hit in first_search.hits
        ) == tuple(
            (
                hit.name,
                hit.match_reasons,
                (
                    ()
                    if hit.match_reasons == ("relationship_neighbor",)
                    else hit.matched_fields
                ),
            )
            for hit in second_search.hits
        )

        first_resources = await first_agent.list_catalog_resources(
            source_id=first_source.id
        )
        second_resources = await second_agent.list_catalog_resources(
            source_id=second_source.id
        )
        first_projection = await _schema(
            first_agent,
            resource_ids=tuple(resource.id for resource in first_resources),
        )
        second_projection = await _schema(
            second_agent,
            resource_ids=tuple(resource.id for resource in second_resources),
        )
        assert tuple(
            item["name"] for item in _mapping_sequence(first_projection["resources"])
        ) == tuple(
            item["name"] for item in _mapping_sequence(second_projection["resources"])
        )

        def signatures(projection: Mapping[str, object]) -> tuple[object, ...]:
            resources = _mapping_sequence(projection["resources"])
            names = {item["resource_id"]: item["name"] for item in resources}
            return tuple(
                sorted(
                    (
                        names[item["from_resource_id"]],
                        names[item["to_resource_id"]],
                        item["kind"],
                        tuple(
                            (pair["source_field"], pair["target_field"])
                            for pair in _mapping_sequence(item["field_pairs"])
                        ),
                    )
                    for item in _mapping_sequence(projection["relationships"])
                )
            )

        assert signatures(first_projection) == signatures(second_projection)
    finally:
        await first_agent.close()
        await second_agent.close()
