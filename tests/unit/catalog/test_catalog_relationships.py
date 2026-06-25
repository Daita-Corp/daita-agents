from daita.plugins.catalog import CatalogPlugin


def _make_normalized_schema(
    tables=None, fks=None, db_type="postgresql", db_name="public"
):
    tables = tables or []
    return {
        "database_type": db_type,
        "database_name": db_name,
        "tables": tables,
        "foreign_keys": fks or [],
        "table_count": len(tables),
    }


def _table(name, columns=None, row_count=None):
    columns = columns or [
        {"name": "id", "type": "integer", "nullable": False, "is_primary_key": True}
    ]
    return {"name": name, "columns": columns, "row_count": row_count}


async def test_catalog_inspect_table_uses_catalog_state():
    catalog = CatalogPlugin()
    schema = _make_normalized_schema(
        tables=[
            _table(
                "users",
                columns=[
                    {"name": "id", "type": "integer"},
                    {"name": "email", "type": "text"},
                ],
            )
        ]
    )
    registered = await catalog.register_schema(schema, store_type="postgresql")

    result = catalog.get_table_schema(registered["store_id"], "users")

    assert result["success"] is True
    assert result["columns"][0]["name"] == "id"
    assert result["columns"][1]["name"] == "email"


async def test_catalog_find_join_path_returns_sql_ready_predicates():
    catalog = CatalogPlugin()
    schema = _make_normalized_schema(
        tables=[
            _table("operations"),
            _table("api_keys"),
            _table("users"),
        ],
        fks=[
            {
                "source_table": "operations",
                "source_column": "api_key_id",
                "target_table": "api_keys",
                "target_column": "api_key_id",
            },
            {
                "source_table": "api_keys",
                "source_column": "created_by",
                "target_table": "users",
                "target_column": "user_id",
            },
        ],
    )
    registered = await catalog.register_schema(schema, store_type="postgresql")

    result = catalog.find_relationship_paths(
        registered["store_id"],
        from_assets=["operations"],
        to_assets=["users"],
    )

    assert result["success"] is True
    assert result["reachable"] is True
    assert result["path_count"] == 1
    path = result["paths"][0]
    assert path["tables"] == ["operations", "api_keys", "users"]
    assert [join["predicate"] for join in path["joins"]] == [
        "operations.api_key_id = api_keys.api_key_id",
        "api_keys.created_by = users.user_id",
    ]


async def test_catalog_find_join_path_warns_for_membership_bridge_paths():
    catalog = CatalogPlugin()
    schema = _make_normalized_schema(
        tables=[
            _table("operations"),
            _table("organization"),
            _table("organization_members"),
            _table("users"),
        ],
        fks=[
            {
                "source_table": "operations",
                "source_column": "organization_id",
                "target_table": "organization",
                "target_column": "organization_id",
            },
            {
                "source_table": "organization_members",
                "source_column": "organization_id",
                "target_table": "organization",
                "target_column": "organization_id",
            },
            {
                "source_table": "organization_members",
                "source_column": "user_id",
                "target_table": "users",
                "target_column": "user_id",
            },
        ],
    )
    registered = await catalog.register_schema(schema, store_type="postgresql")

    result = catalog.find_relationship_paths(
        registered["store_id"],
        from_assets=["operations"],
        to_assets=["users"],
    )

    assert result["reachable"] is True
    assert result["paths"][0]["tables"] == [
        "operations",
        "organization",
        "organization_members",
        "users",
    ]
    assert result["paths"][0]["warnings"]


async def test_catalog_find_join_path_reports_unknown_tables_with_candidates():
    catalog = CatalogPlugin()
    schema = _make_normalized_schema(tables=[_table("users")])
    registered = await catalog.register_schema(schema, store_type="postgresql")

    result = catalog.find_relationship_paths(
        registered["store_id"],
        from_assets=["user"],
        to_assets=["users"],
    )

    assert result["success"] is False
    assert result["reachable"] is False
    assert result["errors"][0]["error"] == "asset not found"
    assert result["errors"][0]["candidates"]
