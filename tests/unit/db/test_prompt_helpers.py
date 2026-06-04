import pytest

from daita.db.config.policies import SchemaPromptPolicy, ToolResultPolicy
from daita.db.catalog_prompt import build_db_prompt_read_model


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


def _cols(names):
    return [
        {"name": name, "type": "text", "nullable": True, "is_primary_key": index == 0}
        for index, name in enumerate(names)
    ]


class TestBuildDbPromptReadModel:
    def test_small_schema_has_column_types(self):
        tables = [
            _table("orders", _cols(["id", "total", "status"]), row_count=5000),
            _table("customers", _cols(["id", "email"]), row_count=1000),
        ]
        schema = _make_normalized_schema(tables=tables)
        model = build_db_prompt_read_model(schema)
        prompt = "\n".join(model.schema_lines)

        assert model.strategy == "full"
        assert "| Column | Type | PK | Nullable |" in prompt
        assert "orders" in prompt
        assert "customers" in prompt
        assert "5,000 rows" in prompt

    def test_large_schema_summary_only(self):
        tables = [
            _table(f"table_{i}", _cols(["id", "name"]), row_count=1000)
            for i in range(100)
        ]
        schema = _make_normalized_schema(tables=tables)
        schema["table_count"] = 100
        model = build_db_prompt_read_model(schema)
        prompt = "\n".join(model.schema_lines)

        assert model.strategy == "retrieval"
        assert "| Column | Type | PK | Nullable |" not in prompt
        assert "Columns:" not in prompt
        assert "1K rows" in prompt

    def test_medium_schema_column_names_only(self):
        tables = [_table(f"tbl_{i}", _cols(["id", "name"])) for i in range(40)]
        schema = _make_normalized_schema(tables=tables)
        schema["table_count"] = 40
        model = build_db_prompt_read_model(schema)
        prompt = "\n".join(model.schema_lines)

        assert model.strategy == "compact"
        assert "Columns:" in prompt
        assert "| Column | Type | PK | Nullable |" not in prompt

    def test_relationship_lines_include_foreign_keys(self):
        tables = [_table("orders"), _table("customers")]
        fks = [
            {
                "source_table": "orders",
                "source_column": "customer_id",
                "target_table": "customers",
                "target_column": "id",
            }
        ]
        schema = _make_normalized_schema(tables=tables, fks=fks)
        model = build_db_prompt_read_model(schema)
        relationships = "\n".join(model.relationship_lines)

        assert "orders.customer_id" in relationships
        assert "customers.id" in relationships

    def test_relationship_overflow_uses_configured_relationship_tool(self):
        tables = [_table("orders"), _table("customers"), _table("regions")]
        fks = [
            {
                "source_table": "orders",
                "source_column": "customer_id",
                "target_table": "customers",
                "target_column": "id",
            },
            {
                "source_table": "customers",
                "source_column": "region_id",
                "target_table": "regions",
                "target_column": "id",
            },
        ]
        schema = _make_normalized_schema(tables=tables, fks=fks)

        model = build_db_prompt_read_model(
            schema,
            policy=SchemaPromptPolicy(max_inline_relationships=1),
            relationship_tool="relationship_mapper",
        )
        relationships = "\n".join(model.relationship_lines)

        assert "additional relationships available via relationship_mapper" in (
            relationships
        )
        assert "catalog_find_join_paths" not in relationships

    def test_no_fk_message(self):
        schema = _make_normalized_schema(tables=[_table("foo")])
        model = build_db_prompt_read_model(schema)
        assert "No foreign key relationships discovered." in "\n".join(
            model.relationship_lines
        )

    def test_empty_database(self):
        schema = _make_normalized_schema(tables=[])
        model = build_db_prompt_read_model(schema)
        assert "empty" in "\n".join(model.schema_lines).lower()
        assert model.database_type == "postgresql"

    def test_budget_uses_retrieval_for_wide_schemas(self):
        tables = [
            _table(
                f"wide_{i}",
                _cols([f"col_{i}_{j}" for j in range(15)]),
            )
            for i in range(25)
        ]
        schema = _make_normalized_schema(tables=tables)
        schema["table_count"] = 25

        model = build_db_prompt_read_model(schema)
        prompt = "\n".join(model.schema_lines)

        assert model.strategy == "retrieval"
        assert model.column_count == 375
        assert "Columns: col_0_0" not in prompt
        assert "| Column | Type | PK | Nullable |" not in prompt
        assert "- wide_0" in prompt

    def test_prompt_policies_are_config_importable(self):
        from daita.db.config.policies import (
            SchemaPromptPolicy as PublicSchemaPromptPolicy,
        )
        from daita.db.config.policies import (
            ToolResultPolicy as PublicToolResultPolicy,
        )

        assert PublicSchemaPromptPolicy is SchemaPromptPolicy
        assert PublicToolResultPolicy is ToolResultPolicy

    def test_invalid_policy_values_fail_fast(self):
        with pytest.raises(ValueError, match="max_inline_schema_tokens"):
            SchemaPromptPolicy(max_inline_schema_tokens=0)
        with pytest.raises(ValueError, match="max_result_tokens"):
            ToolResultPolicy(max_result_tokens=0)
