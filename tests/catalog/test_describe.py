"""Exercise the catalog_describe tool and the catalog schema projections."""

from __future__ import annotations

from daita.capabilities import CallTargetLane
from daita.catalog import CatalogDescribeDetail, CatalogDescribeRequest
from daita.catalog.models import CATALOG_DESCRIBE_IDENTITY_MAX_LIMIT
from tests.catalog._schema_support import (
    _OBSERVED_AT,
    CATALOG_DESCRIBE_CAPABILITY_ID,
    CATALOG_DESCRIBE_EVIDENCE_KIND,
    CATALOG_MAX_LIMIT,
    CATALOG_RESOURCE_ID_MAX_CHARACTERS,
    CATALOG_SCHEMA_MAX_RESOURCE_IDS,
    CATALOG_SEARCH_CAPABILITY_ID,
    CATALOG_SEARCH_REQUEST_DEFAULT_LIMIT,
    CATALOG_SEARCH_REQUEST_MAX_QUERY_CHARACTERS,
    CATALOG_SOURCE_ID_MAX_CHARACTERS,
    CATALOG_TOOL_DEFAULT_LIMIT,
    CATALOG_TOOL_QUERY_MAX_CHARACTERS,
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
from tests.support.wide_catalog import (
    CHAIN as WIDE_CHAIN,
    COLUMN_COUNT as WIDE_COLUMN_COUNT,
    MANIFEST as WIDE_MANIFEST,
    TABLE_COUNT as WIDE_TABLE_COUNT,
    seed_wide_sqlite,
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


async def test_describe_schema_validates_output_and_is_smaller_than_inspections(
    tmp_path: Path,
):
    database = tmp_path / "capability.sqlite"
    _fixture_database(database)
    agent = await Agent.create(
        "catalog-describe-capability", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        resources = await agent.list_catalog_resources(source_id=source.id)
        registry: CapabilityRegistry = agent._embedded._capabilities
        view, capability = registry.resolve_tool("catalog_describe")
        assert view.capability_id == capability.id == CATALOG_DESCRIBE_CAPABILITY_ID
        _, executor = registry.resolve_execution(capability.id)
        output = await executor.execute(
            ToolExecution(
                run_id="describe-capability",
                call_id="describe-capability-call",
                capability_id=capability.id,
                arguments={
                    "resource_ids": tuple(resource.id for resource in resources),
                    "limit": 50,
                },
            )
        )
        assert output.kind == CATALOG_DESCRIBE_EVIDENCE_KIND
        assert output.data["detail"] == "schema"
        assert registry.validate_output(capability.id, output) == output
        assert _mapping(output.data["bounds"])["relationships"] == 8
        assert len(_mapping_sequence(output.data["relationships"])) <= 8
        with pytest.raises(ToolOutputValidationError):
            registry.validate_output(
                capability.id,
                ToolOutput(
                    kind=CATALOG_DESCRIBE_EVIDENCE_KIND,
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
        estimates = _mapping(output.data["estimates"])
        assert estimates["approximate"] is True
        assert estimates["available"] == 0
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
            accepted = dict(output.data)
            accepted["match_outcome"] = accepted_outcome
            validated = ToolOutput(kind=output.kind, data=accepted)
            assert registry.validate_output(capability.id, validated) == validated

        malformed = dict(output.data)
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
        search_definition = registry.tool_definition(
            "catalog_search", lane=CallTargetLane.FOREGROUND
        )
        _search_view, search_capability = registry.resolve_tool("catalog_search")
        describe_definition = registry.tool_definition(
            "catalog_describe", lane=CallTargetLane.FOREGROUND
        )

        search_properties = search_definition.input_schema["properties"]
        describe_properties = describe_definition.input_schema["properties"]
        assert isinstance(search_properties, Mapping)
        assert isinstance(describe_properties, Mapping)
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

        assert set(describe_properties) == {
            "cursor",
            "detail",
            "limit",
            "query",
            "resource_ids",
            "source_id",
        }
        assert canonical_json(describe_properties["detail"]) == canonical_json(
            {
                "type": "string",
                "enum": ["identity", "schema", "relationships"],
                "default": "schema",
            }
        )
        assert canonical_json(describe_properties["query"]) == canonical_json(
            query_rule
        )
        assert canonical_json(describe_properties["source_id"]) == canonical_json(
            source_rule
        )
        describe_limit = _mapping(describe_properties["limit"])
        assert describe_limit["minimum"] == 1
        assert describe_limit["maximum"] == CATALOG_DESCRIBE_IDENTITY_MAX_LIMIT
        assert "default" not in describe_limit
        assert canonical_json(describe_properties["resource_ids"]) == canonical_json(
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

        for capability_id, limit in (
            (CATALOG_SEARCH_CAPABILITY_ID, CATALOG_MAX_LIMIT + 1),
            (CATALOG_DESCRIBE_CAPABILITY_ID, CATALOG_DESCRIBE_IDENTITY_MAX_LIMIT + 1),
        ):
            with pytest.raises(CapabilityInputError) as invalid_limit:
                registry.validate_arguments(
                    capability_id,
                    {"query": "orders", "limit": limit},
                )
            assert invalid_limit.value.code == "invalid_argument_value"
            assert _validation_details(invalid_limit.value) == {
                "constraint": "maximum",
                "name": "limit",
            }

        with pytest.raises(CapabilityInputError) as invalid_kind:
            registry.validate_arguments(
                CATALOG_SEARCH_CAPABILITY_ID,
                {"query": "orders", "resource_kinds": ("spreadsheet",)},
            )
        assert _validation_details(invalid_kind.value) == {
            "constraint": "enum",
            "name": "resource_kinds[0]",
        }

        with pytest.raises(CapabilityInputError) as invalid_detail:
            registry.validate_arguments(
                CATALOG_DESCRIBE_CAPABILITY_ID,
                {"query": "orders", "detail": "columns"},
            )
        assert _validation_details(invalid_detail.value) == {
            "constraint": "enum",
            "name": "detail",
        }

        with pytest.raises(CapabilityInputError) as duplicate_resources:
            registry.validate_arguments(
                CATALOG_DESCRIBE_CAPABILITY_ID,
                {
                    "resource_ids": ("resource-a", "resource-a"),
                    "limit": CATALOG_TOOL_DEFAULT_LIMIT,
                },
            )
        assert _validation_details(duplicate_resources.value) == {
            "constraint": "uniqueItems",
            "name": "resource_ids",
        }

        assert CatalogSearchRequest(agent_id="agent", query="orders").limit == (
            CATALOG_SEARCH_REQUEST_DEFAULT_LIMIT
        )
        assert CatalogSchemaRequest(agent_id="agent", query="orders").limit == (
            CATALOG_TOOL_DEFAULT_LIMIT
        )
        assert (
            CatalogDescribeRequest(agent_id="agent", query="orders").effective_limit
            == CATALOG_TOOL_DEFAULT_LIMIT
        )
        assert (
            CatalogDescribeRequest(
                agent_id="agent",
                detail=CatalogDescribeDetail.IDENTITY,
                source_id="source",
            ).effective_limit
            == CATALOG_DESCRIBE_IDENTITY_MAX_LIMIT
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
        with pytest.raises(ValueError, match="catalog describe query exceeds"):
            CatalogDescribeRequest(agent_id="agent", query=oversized_tool_query)
        with pytest.raises(ValueError, match="exact source_id for identity"):
            CatalogDescribeRequest(agent_id="agent", source_id="source")
        with pytest.raises(ValueError, match="cursor applies only to identity"):
            CatalogDescribeRequest(agent_id="agent", query="orders", cursor="1.ab")
        with pytest.raises(ValueError, match="schema limit must be from 1"):
            CatalogDescribeRequest(
                agent_id="agent",
                query="orders",
                limit=CATALOG_MAX_LIMIT + 1,
            )
    finally:
        await agent.close()


async def test_catalog_describe_invalid_input_never_reaches_catalog_execution(
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

        async def unexpected_describe_execution(*args: object, **kwargs: object):
            nonlocal catalog_execution_calls
            del args, kwargs
            catalog_execution_calls += 1
            raise AssertionError("invalid catalog describe input reached execution")

        monkeypatch.setattr(
            catalog_service.CatalogService,
            "describe",
            unexpected_describe_execution,
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
                    name="catalog_describe",
                    arguments={"source_id": source.id, "limit": 101},
                ),
                ToolCall(
                    id="missing-selector",
                    name="catalog_describe",
                    arguments={"source_id": source.id, "limit": CATALOG_MAX_LIMIT},
                ),
                ToolCall(
                    id="schema-limit",
                    name="catalog_describe",
                    arguments={"query": "orders", "limit": CATALOG_MAX_LIMIT + 1},
                ),
                ToolCall(
                    id="schema-cursor",
                    name="catalog_describe",
                    arguments={"query": "orders", "cursor": "1." + "0" * 64},
                ),
                ToolCall(
                    id="blank-query",
                    name="catalog_describe",
                    arguments={"detail": "identity", "query": "  "},
                ),
            ),
        )

        error_codes = tuple(
            _mapping(_mapping(result.output)["error"])["code"] for result in results
        )
        assert error_codes == (
            "invalid_argument_value",
            "catalog_invalid_describe",
            "catalog_invalid_describe",
            "catalog_invalid_describe",
            "catalog_invalid_describe",
        )
        assert catalog_execution_calls == 0
    finally:
        await agent.close()


async def test_catalog_describe_replaces_split_tools_without_ordering_rules(
    tmp_path: Path,
):
    agent = await Agent.create(
        "catalog-describe-contract", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        registry: CapabilityRegistry = agent._embedded._capabilities
        for removed in ("catalog_schema", "catalog_inspect", "catalog_traverse"):
            with pytest.raises(KeyError, match="unknown tool"):
                registry.tool_definition(removed, lane=CallTargetLane.FOREGROUND)
        for removed_id in ("catalog.schema", "catalog.inspect", "catalog.traverse"):
            with pytest.raises(KeyError):
                registry.resolve_execution(removed_id)

        description = registry.tool_definition(
            "catalog_describe", lane=CallTargetLane.FOREGROUND
        ).description
        for ordering_rule in ("Do not use with", "later step", "alongside"):
            assert ordering_rule not in description
        assert "multi-hop resolved in code" in description
        assert "estimates" in description
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


async def _describe(
    agent: Agent,
    arguments: Mapping[str, object],
) -> ToolOutput:
    registry: CapabilityRegistry = agent._embedded._capabilities
    _view, capability = registry.resolve_tool("catalog_describe")
    _capability, executor = registry.resolve_execution(capability.id)
    output = await executor.execute(
        ToolExecution(
            run_id="describe-run",
            call_id="describe-call",
            capability_id=capability.id,
            arguments=arguments,
        )
    )
    assert registry.validate_output(capability.id, output) == output
    return output


def _validation_details(error: CapabilityInputError) -> dict[str, object]:
    details = dict(error.details.to_dict())
    recovery = details.pop("recovery")
    assert isinstance(recovery, dict)
    assert recovery["kind"] == "correct_arguments"
    return details


def _field_names(value: object) -> set[str]:
    if isinstance(value, Mapping):
        names = {str(key) for key in value}
        for item in value.values():
            names |= _field_names(item)
        return names
    if isinstance(value, (tuple, list)):
        names = set()
        for item in value:
            names |= _field_names(item)
        return names
    return set()


async def test_identity_lists_every_table_without_columns_or_source_io(
    tmp_path: Path,
):
    database = seed_wide_sqlite(tmp_path / "wide.sqlite")
    agent = await Agent.create(
        "catalog-describe-identity", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        unavailable = tmp_path / "wide-source-is-unavailable.sqlite"
        database.rename(unavailable)

        output = await _describe(agent, {"detail": "identity", "source_id": source.id})

        assert not database.exists()
        data = output.data
        assert data["detail"] == "identity"
        rows = _mapping_sequence(data["resources"])
        assert len(rows) == WIDE_TABLE_COUNT
        assert {str(row["name"]) for row in rows} == {
            f"main.{item['table']}" for item in WIDE_MANIFEST
        }
        assert all(
            set(row) == {"kind", "name", "resource_id", "revision", "source_id"}
            for row in rows
        )
        assert {"columns", "column", "data_type", "relationships"}.isdisjoint(
            _field_names(data)
        )
        assert dict(_mapping(data["coverage"])) == {
            "basis": "source",
            "complete": True,
            "offset": 0,
            "returned": WIDE_TABLE_COUNT,
            "scope_filtered": False,
            "total": WIDE_TABLE_COUNT,
        }
        assert data["truncated"] is False
        assert data["next_cursor"] is None
        estimates = _mapping(data["estimates"])
        assert estimates["approximate"] is True
        assert estimates["available"] == 0
        assert estimates["unavailable"] == WIDE_TABLE_COUNT
        assert "never exact counts" in str(estimates["note"])
        assert data["trust_classification"] == "untrusted_external_data"
    finally:
        await agent.close()


async def test_identity_is_at_least_sixty_percent_smaller_than_full_schema(
    tmp_path: Path,
):
    database = seed_wide_sqlite(tmp_path / "wide.sqlite")
    agent = await Agent.create(
        "catalog-describe-bytes", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        resources = await agent.list_catalog_resources(source_id=source.id)
        assert len(resources) == WIDE_TABLE_COUNT
        identity = await _describe(
            agent, {"detail": "identity", "source_id": source.id}
        )
        schema_pages = (
            await _describe(
                agent,
                {
                    "detail": "schema",
                    "resource_ids": tuple(item.id for item in resources[:23]),
                    "limit": CATALOG_MAX_LIMIT,
                },
            ),
            await _describe(
                agent,
                {
                    "detail": "schema",
                    "resource_ids": tuple(item.id for item in resources[23:]),
                    "limit": CATALOG_MAX_LIMIT,
                },
            ),
        )
        assert (
            sum(len(_mapping_sequence(page.data["resources"])) for page in schema_pages)
            == WIDE_TABLE_COUNT
        )
        assert (
            sum(
                len(_mapping_sequence(resource["columns"]))
                for page in schema_pages
                for resource in _mapping_sequence(page.data["resources"])
            )
            == WIDE_COLUMN_COUNT
        )

        identity_bytes = len(canonical_json(identity.data).encode("utf-8"))
        schema_bytes = sum(
            len(canonical_json(page.data).encode("utf-8")) for page in schema_pages
        )
        assert identity_bytes <= schema_bytes * 0.4
    finally:
        await agent.close()


async def test_identity_pages_by_cursor_and_reports_truthful_coverage(
    tmp_path: Path,
):
    database = seed_wide_sqlite(tmp_path / "wide.sqlite")
    agent = await Agent.create(
        "catalog-describe-paging", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        names: list[str] = []
        cursor: object = None
        offsets: list[object] = []
        while True:
            arguments: dict[str, object] = {
                "detail": "identity",
                "source_id": source.id,
                "limit": 20,
            }
            if cursor is not None:
                arguments["cursor"] = cursor
            data = (await _describe(agent, arguments)).data
            coverage = _mapping(data["coverage"])
            offsets.append(coverage["offset"])
            assert coverage["total"] == WIDE_TABLE_COUNT
            names.extend(
                str(row["name"]) for row in _mapping_sequence(data["resources"])
            )
            cursor = data["next_cursor"]
            assert data["truncated"] is (cursor is not None)
            assert coverage["complete"] is False
            if cursor is None:
                break
        assert offsets == [0, 20, 40]
        assert len(names) == len(set(names)) == WIDE_TABLE_COUNT
        assert names == sorted(names, key=str.casefold)

        with pytest.raises(CatalogStoreError):
            await _describe(
                agent,
                {
                    "detail": "identity",
                    "source_id": source.id,
                    "cursor": "20." + "0" * 64,
                },
            )

        queried = (
            await _describe(
                agent,
                {"detail": "identity", "source_id": source.id, "query": "ledger"},
            )
        ).data
        assert _mapping(queried["coverage"])["basis"] == "query"
        assert [
            str(row["name"]) for row in _mapping_sequence(queried["resources"])
        ] == ["main.empty_ledger"]
    finally:
        await agent.close()


async def test_identity_coverage_reports_permission_filtering(tmp_path: Path):
    database = seed_wide_sqlite(tmp_path / "wide.sqlite")
    agent = await Agent.create(
        "catalog-describe-scope", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        resources = await agent.list_catalog_resources(source_id=source.id)
        readable = frozenset(item.id for item in resources[:10])
        service = agent._embedded._catalog_service
        data = await service.describe(
            CatalogDescribeRequest(
                agent_id=agent.id,
                detail=CatalogDescribeDetail.IDENTITY,
                source_id=source.id,
            ),
            readable_resource_ids=readable,
        )
        coverage = _mapping(data["coverage"])
        assert coverage["total"] == 10
        assert coverage["scope_filtered"] is True
        assert {
            str(row["resource_id"]) for row in _mapping_sequence(data["resources"])
        } == set(readable)
    finally:
        await agent.close()


@pytest.mark.parametrize("detail", ["schema", "relationships"])
async def test_describe_resolves_multi_hop_join_paths_in_one_call(
    tmp_path: Path,
    detail: str,
):
    database = seed_wide_sqlite(tmp_path / "wide.sqlite")
    agent = await Agent.create(
        "catalog-describe-paths", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        by_name = {
            item.native_identity: item.id
            for item in await agent.list_catalog_resources(source_id=source.id)
        }
        first, last = (
            by_name[f"main.{WIDE_CHAIN[0]}"],
            by_name[f"main.{WIDE_CHAIN[-1]}"],
        )

        data = (
            await _describe(
                agent,
                {"detail": detail, "resource_ids": (first, last)},
            )
        ).data

        assert data["detail"] == detail
        paths = _mapping_sequence(data["paths"])
        assert paths
        path = paths[0]
        assert len(cast(tuple[str, ...], path["relationship_ids"])) == 3
        assert set(cast(tuple[str, ...], path["resource_ids"])) == {
            by_name[f"main.{name}"] for name in WIDE_CHAIN
        }
        returned = {
            str(row["resource_id"]): row for row in _mapping_sequence(data["resources"])
        }
        assert {by_name[f"main.{name}"] for name in WIDE_CHAIN} <= set(returned)
        if detail == "relationships":
            assert all("columns" not in row for row in returned.values())
            assert all(
                _mapping_sequence(item["field_pairs"])
                for item in _mapping_sequence(data["relationships"])
            )
        else:
            assert all("columns" in row for row in returned.values())
    finally:
        await agent.close()
