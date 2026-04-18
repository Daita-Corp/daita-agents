"""
Tests for catalog normalizer: deduplication, fingerprinting, and environment inference.
"""

import pytest

from daita.plugins.catalog.base_discoverer import DiscoveredStore
from daita.plugins.catalog.normalizer import (
    deduplicate_stores,
    infer_environment,
    merge_store_sources,
    normalize_apigateway,
    normalize_discovery,
    normalize_dynamodb,
    normalize_mongodb,
    normalize_mysql,
    normalize_postgresql,
    normalize_s3,
)


def _make_store(**kwargs) -> DiscoveredStore:
    defaults = {
        "id": "abc123",
        "store_type": "postgresql",
        "display_name": "test-db",
        "connection_hint": {"host": "localhost", "port": 5432},
        "source": "manual",
        "confidence": 0.5,
    }
    defaults.update(kwargs)
    return DiscoveredStore(**defaults)


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------


def test_deduplicate_no_duplicates():
    stores = [_make_store(id="s1"), _make_store(id="s2")]
    result = deduplicate_stores(stores)
    assert len(result) == 2


def test_deduplicate_merges_matching_ids():
    store1 = _make_store(id="same", source="aws", confidence=0.9)
    store2 = _make_store(id="same", source="github", confidence=0.7)

    result = deduplicate_stores([store1, store2])
    assert len(result) == 1
    # Higher confidence wins
    assert result[0].confidence == 0.9
    assert result[0].source == "aws"
    # Both sources tracked
    assert "aws" in result[0].metadata["seen_by"]
    assert "github" in result[0].metadata["seen_by"]


def test_deduplicate_merges_tags():
    store1 = _make_store(id="same", tags=["env:prod"])
    store2 = _make_store(id="same", tags=["team:backend"])

    result = deduplicate_stores([store1, store2])
    assert len(result) == 1
    assert set(result[0].tags) == {"env:prod", "team:backend"}


def test_merge_preserves_environment():
    existing = _make_store(id="s1", confidence=0.9, environment="production")
    new = _make_store(id="s1", confidence=0.5, environment=None)

    merged = merge_store_sources(existing, new)
    assert merged.environment == "production"


def test_merge_fills_missing_environment():
    existing = _make_store(id="s1", confidence=0.9, environment=None)
    new = _make_store(id="s1", confidence=0.5, environment="staging")

    merged = merge_store_sources(existing, new)
    assert merged.environment == "staging"


# ---------------------------------------------------------------------------
# Environment inference tests
# ---------------------------------------------------------------------------


def test_infer_production():
    store = _make_store(display_name="prod-orders-db")
    assert infer_environment(store) == "production"


def test_infer_production_suffix():
    store = _make_store(display_name="orders.production")
    assert infer_environment(store) == "production"


def test_infer_staging():
    store = _make_store(display_name="staging-users-db")
    assert infer_environment(store) == "staging"


def test_infer_development():
    store = _make_store(display_name="dev-analytics")
    assert infer_environment(store) == "development"


def test_infer_test():
    store = _make_store(display_name="test-payments")
    assert infer_environment(store) == "test"


def test_infer_from_tags():
    store = _make_store(display_name="my-db", tags=["prod"])
    assert infer_environment(store) == "production"


def test_infer_from_metadata():
    store = _make_store(display_name="my-db", metadata={"environment": "Staging"})
    assert infer_environment(store) == "staging"


def test_infer_unknown():
    store = _make_store(display_name="my-random-db")
    assert infer_environment(store) == "unknown"


# ---------------------------------------------------------------------------
# normalize_discovery dispatch tests
# ---------------------------------------------------------------------------


def test_normalize_postgresql():
    raw = {
        "database_type": "postgresql",
        "schema": "public",
        "tables": [{"table_name": "users", "row_count": 10}],
        "columns": [
            {
                "table_name": "users",
                "column_name": "id",
                "data_type": "integer",
                "is_nullable": "NO",
            }
        ],
        "primary_keys": [{"table_name": "users", "column_name": "id"}],
        "foreign_keys": [],
    }
    result = normalize_postgresql(raw)
    assert result["database_type"] == "postgresql"
    assert result["database_name"] == "public"
    assert result["table_count"] == 1
    assert result["tables"][0]["name"] == "users"
    assert result["tables"][0]["columns"][0]["is_primary_key"] is True


def test_normalize_postgresql_propagates_host_metadata():
    """host + port from discovery must reach metadata so _derive_store can
    build a collision-safe store identifier."""
    raw = {
        "database_type": "postgresql",
        "schema": "public",
        "host": "prod-pg.internal",
        "port": 5432,
        "tables": [],
        "columns": [],
        "primary_keys": [],
        "foreign_keys": [],
    }
    result = normalize_postgresql(raw)
    assert result["metadata"] == {"host": "prod-pg.internal", "port": 5432}


def test_normalize_postgresql_omits_metadata_when_absent():
    raw = {
        "database_type": "postgresql",
        "schema": "public",
        "tables": [],
        "columns": [],
        "primary_keys": [],
        "foreign_keys": [],
    }
    result = normalize_postgresql(raw)
    assert "metadata" not in result


def test_normalize_mysql():
    raw = {
        "database_type": "mysql",
        "schema": "mydb",
        "tables": [{"table_name": "orders", "row_count": 5}],
        "columns": [
            {
                "table_name": "orders",
                "column_name": "id",
                "data_type": "int",
                "column_key": "PRI",
            }
        ],
        "foreign_keys": [],
    }
    result = normalize_mysql(raw)
    assert result["database_type"] == "mysql"
    assert result["tables"][0]["columns"][0]["is_primary_key"] is True


def test_normalize_mysql_propagates_host_metadata():
    raw = {
        "database_type": "mysql",
        "schema": "mydb",
        "host": "mysql-prod.internal",
        "port": 3306,
        "tables": [],
        "columns": [],
        "foreign_keys": [],
    }
    result = normalize_mysql(raw)
    assert result["metadata"] == {"host": "mysql-prod.internal", "port": 3306}


def test_normalize_mongodb():
    raw = {
        "database_type": "mongodb",
        "database": "testdb",
        "collections": [
            {
                "collection_name": "users",
                "document_count": 100,
                "fields": [
                    {"field_name": "_id", "types": ["ObjectId"]},
                    {"field_name": "name", "types": ["str"]},
                ],
            }
        ],
    }
    result = normalize_mongodb(raw)
    assert result["database_type"] == "mongodb"
    assert result["tables"][0]["name"] == "users"
    assert result["tables"][0]["columns"][0]["is_primary_key"] is True
    assert result["tables"][0]["columns"][1]["is_primary_key"] is False


def test_normalize_mongodb_propagates_host_metadata():
    """``host`` / ``port`` from discovery must reach ``metadata`` so the
    persister can build a collision-safe store identifier.
    """
    raw = {
        "database_type": "mongodb",
        "database": "orders_db",
        "host": "mongo-a.internal",
        "port": 27017,
        "collections": [],
    }
    result = normalize_mongodb(raw)
    assert result["metadata"] == {"host": "mongo-a.internal", "port": 27017}


def test_normalize_mongodb_omits_metadata_when_absent():
    """Schema dict should not carry an empty ``metadata`` key when discovery
    didn't provide host/port — callers check ``metadata in schema``.
    """
    raw = {"database_type": "mongodb", "database": "testdb", "collections": []}
    result = normalize_mongodb(raw)
    assert "metadata" not in result


def test_normalize_dynamodb():
    raw = {
        "database_type": "dynamodb",
        "table_name": "users",
        "key_schema": [
            {"AttributeName": "pk", "KeyType": "HASH"},
            {"AttributeName": "sk", "KeyType": "RANGE"},
        ],
        "attribute_definitions": [
            {"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
            {"AttributeName": "gsi1pk", "AttributeType": "S"},
        ],
        "item_count": 5000,
        "sampled_attributes": {
            "pk": ["S"],
            "sk": ["S"],
            "email": ["S"],
            "age": ["N"],
        },
    }
    result = normalize_dynamodb(raw)
    assert result["database_type"] == "dynamodb"
    assert result["database_name"] == "users"
    assert result["table_count"] == 1

    table = result["tables"][0]
    assert table["name"] == "users"
    assert table["row_count"] == 5000

    col_names = [c["name"] for c in table["columns"]]
    assert "pk" in col_names
    assert "sk" in col_names
    assert "email" in col_names
    assert "age" in col_names

    # pk and sk are primary keys
    pk_cols = [c for c in table["columns"] if c["is_primary_key"]]
    assert len(pk_cols) == 2
    assert {c["name"] for c in pk_cols} == {"pk", "sk"}

    # gsi1pk is defined but not a primary key
    gsi_col = next(c for c in table["columns"] if c["name"] == "gsi1pk")
    assert gsi_col["is_primary_key"] is False

    # email and age are inferred from sample
    email_col = next(c for c in table["columns"] if c["name"] == "email")
    assert email_col["type"] == "string"
    age_col = next(c for c in table["columns"] if c["name"] == "age")
    assert age_col["type"] == "number"


def test_normalize_s3():
    raw = {
        "database_type": "s3",
        "bucket": "my-data-bucket",
        "object_count": 250,
        "total_size_bytes": 1024000,
        "prefixes": {"raw": 100, "processed": 80, "archive": 70},
        "content_types": {"csv": 150, "json": 80, "parquet": 20},
        "versioning": "Enabled",
    }
    result = normalize_s3(raw)
    assert result["database_type"] == "s3"
    assert result["database_name"] == "my-data-bucket"
    assert result["table_count"] == 1

    table = result["tables"][0]
    assert table["name"] == "my-data-bucket"
    assert table["row_count"] == 250

    # Only real object-level fields are columns
    col_names = [c["name"] for c in table["columns"]]
    assert col_names == ["key", "size_bytes", "last_modified", "storage_class"]

    # key is primary key
    key_col = next(c for c in table["columns"] if c["name"] == "key")
    assert key_col["is_primary_key"] is True

    # Prefix/content type analytics go in metadata, not columns
    meta = result["metadata"]
    assert meta["prefixes"] == {"raw": 100, "processed": 80, "archive": 70}
    assert meta["content_types"] == {"csv": 150, "json": 80, "parquet": 20}
    assert meta["total_size_bytes"] == 1024000
    assert meta["versioning"] == "Enabled"


def test_normalize_discovery_dispatches_dynamodb():
    raw = {
        "database_type": "dynamodb",
        "table_name": "test",
        "key_schema": [{"AttributeName": "id", "KeyType": "HASH"}],
        "attribute_definitions": [{"AttributeName": "id", "AttributeType": "S"}],
        "item_count": 0,
        "sampled_attributes": {},
    }
    result = normalize_discovery(raw)
    assert result["database_type"] == "dynamodb"
    assert result["database_name"] == "test"


def test_normalize_discovery_dispatches_s3():
    raw = {
        "database_type": "s3",
        "bucket": "test-bucket",
        "object_count": 0,
        "prefixes": {},
        "content_types": {},
    }
    result = normalize_discovery(raw)
    assert result["database_type"] == "s3"
    assert result["database_name"] == "test-bucket"


def test_normalize_apigateway():
    raw = {
        "api_type": "apigateway",
        "api_id": "abc123",
        "api_name": "orders-api",
        "protocol_type": "REST",
        "region": "us-east-1",
        "stage": "prod",
        "endpoint": "https://abc123.execute-api.us-east-1.amazonaws.com/prod",
        "endpoints": [
            {
                "path": "/orders",
                "method": "GET",
                "authorization": "COGNITO_USER_POOLS",
                "api_key_required": False,
                "integration_type": "AWS_PROXY",
                "integration_uri": "arn:aws:lambda:us-east-1:123:function:get-orders",
            },
            {
                "path": "/orders",
                "method": "POST",
                "authorization": "COGNITO_USER_POOLS",
                "api_key_required": False,
                "integration_type": "AWS_PROXY",
                "integration_uri": "arn:aws:lambda:us-east-1:123:function:create-order",
            },
        ],
        "endpoint_count": 2,
        "authorizers": [
            {"id": "auth1", "name": "cognito", "type": "COGNITO_USER_POOLS"}
        ],
        "stage_variables": {},
    }
    result = normalize_apigateway(raw)

    assert result["database_type"] == "apigateway"
    assert result["database_name"] == "orders-api"
    assert result["table_count"] == 1

    table = result["tables"][0]
    assert table["name"] == "orders-api"
    assert table["row_count"] == 2

    col_names = [c["name"] for c in table["columns"]]
    assert "GET /orders" in col_names
    assert "POST /orders" in col_names

    # Integration URIs in column comments (for lineage inference)
    get_col = next(c for c in table["columns"] if c["name"] == "GET /orders")
    assert "arn:aws:lambda" in get_col["column_comment"]

    # Full integration map in metadata (for lineage edge creation)
    assert "integrations" in result["metadata"]
    assert "GET /orders" in result["metadata"]["integrations"]
    assert result["metadata"]["integrations"]["GET /orders"]["type"] == "AWS_PROXY"


def test_normalize_apigateway_empty_endpoints():
    raw = {
        "api_type": "apigateway",
        "api_id": "empty",
        "api_name": "empty-api",
        "protocol_type": "HTTP",
        "endpoints": [],
        "endpoint_count": 0,
        "authorizers": [],
        "stage_variables": {},
    }
    result = normalize_apigateway(raw)
    assert result["database_type"] == "apigateway"
    assert result["database_name"] == "empty-api"
    assert result["tables"][0]["row_count"] == 0
    assert result["tables"][0]["columns"] == []


def test_normalize_discovery_dispatches_apigateway():
    raw = {
        "api_type": "apigateway",
        "database_type": "apigateway",
        "api_id": "test",
        "api_name": "test-api",
        "endpoints": [],
        "endpoint_count": 0,
        "authorizers": [],
        "stage_variables": {},
    }
    result = normalize_discovery(raw)
    assert result["database_type"] == "apigateway"
    assert result["database_name"] == "test-api"


def test_metadata_survives_round_trip():
    """Verify metadata from normalize_apigateway survives through to NormalizedSchema.to_dict()."""
    from daita.plugins.catalog.profiler._common import _dict_to_normalized_schema

    raw = {
        "api_type": "apigateway",
        "api_id": "abc123",
        "api_name": "orders-api",
        "protocol_type": "REST",
        "region": "us-east-1",
        "stage": "prod",
        "endpoint": "https://abc123.execute-api.us-east-1.amazonaws.com/prod",
        "endpoints": [
            {
                "path": "/orders",
                "method": "GET",
                "authorization": "COGNITO_USER_POOLS",
                "integration_type": "AWS_PROXY",
                "integration_uri": "arn:aws:lambda:us-east-1:123:function:get-orders",
            },
        ],
        "endpoint_count": 1,
        "authorizers": [
            {"id": "auth1", "name": "cognito", "type": "COGNITO_USER_POOLS"}
        ],
        "stage_variables": {},
    }
    normalized_dict = normalize_apigateway(raw)

    # metadata should be in the normalized dict
    assert "metadata" in normalized_dict
    assert "integrations" in normalized_dict["metadata"]

    # metadata should survive conversion to NormalizedSchema and back
    schema = _dict_to_normalized_schema(normalized_dict, store_id="test")
    assert schema.metadata == normalized_dict["metadata"]

    # metadata should appear in to_dict() output
    output = schema.to_dict()
    assert "metadata" in output
    assert output["metadata"]["integrations"]["GET /orders"]["type"] == "AWS_PROXY"
    assert "arn:aws:lambda" in output["metadata"]["integrations"]["GET /orders"]["uri"]


def test_s3_metadata_survives_round_trip():
    """Verify S3 metadata survives through to NormalizedSchema.to_dict()."""
    from daita.plugins.catalog.profiler._common import _dict_to_normalized_schema

    raw = {
        "database_type": "s3",
        "bucket": "my-bucket",
        "object_count": 100,
        "total_size_bytes": 5000,
        "prefixes": {"raw": 50, "processed": 50},
        "content_types": {"csv": 80, "json": 20},
        "versioning": "Enabled",
    }
    normalized_dict = normalize_s3(raw)
    schema = _dict_to_normalized_schema(normalized_dict)
    output = schema.to_dict()

    assert "metadata" in output
    assert output["metadata"]["prefixes"] == {"raw": 50, "processed": 50}
    assert output["metadata"]["versioning"] == "Enabled"


def test_normalize_discovery_dispatches():
    pg_raw = {
        "database_type": "postgresql",
        "schema": "public",
        "tables": [],
        "columns": [],
        "primary_keys": [],
        "foreign_keys": [],
    }
    result = normalize_discovery(pg_raw)
    assert result["database_type"] == "postgresql"
    assert result["database_name"] == "public"


def test_normalize_discovery_passthrough():
    unknown = {"database_type": "cassandra", "keyspace": "test"}
    result = normalize_discovery(unknown)
    assert result == unknown  # passthrough for unrecognized types
