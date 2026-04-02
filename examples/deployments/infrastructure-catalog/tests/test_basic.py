"""
Tests for the Infrastructure Catalog Agent.

Agent creation and discoverer configuration are tested without any cloud
credentials. Integration tests that need AWS or GitHub are skipped automatically.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------


class TestAgentCreation:
    def test_agent_can_be_created(self):
        from agents.catalog_agent import create_agent

        agent = create_agent()
        assert agent is not None
        assert agent.name == "Infrastructure Catalog Agent"

    def test_agent_with_custom_regions(self):
        from agents.catalog_agent import create_agent

        agent = create_agent(aws_regions=["us-west-2", "eu-west-1"])
        assert agent is not None

    def test_agent_with_github_repos(self):
        from agents.catalog_agent import create_agent

        agent = create_agent(github_repos=["myorg/myrepo"])
        assert agent is not None

    def test_agent_with_github_org(self):
        from agents.catalog_agent import create_agent

        agent = create_agent(github_org="myorg")
        assert agent is not None


# ---------------------------------------------------------------------------
# Catalog plugin configuration (test the plugin directly)
# ---------------------------------------------------------------------------


class TestCatalogConfiguration:
    def _build_catalog(self, **kwargs):
        """Build a CatalogPlugin the same way create_agent does."""
        from daita.plugins.catalog import CatalogPlugin
        from daita.plugins.catalog.aws import AWSDiscoverer
        from daita.plugins.catalog.profiler import PostgresProfiler, MySQLProfiler

        catalog = CatalogPlugin()
        catalog.add_discoverer(AWSDiscoverer(
            regions=kwargs.get("regions", ["us-east-1"]),
        ))
        if kwargs.get("github_org") or kwargs.get("github_repos"):
            from daita.plugins.catalog.github import GitHubScanner
            catalog.add_discoverer(GitHubScanner(
                token="fake-token",
                org=kwargs.get("github_org"),
                repos=kwargs.get("github_repos"),
            ))
        catalog.add_profiler(PostgresProfiler())
        catalog.add_profiler(MySQLProfiler())
        return catalog

    def test_has_aws_discoverer(self):
        from daita.plugins.catalog.aws import AWSDiscoverer

        catalog = self._build_catalog()
        assert any(isinstance(d, AWSDiscoverer) for d in catalog._discoverers)

    def test_has_github_scanner_when_configured(self):
        from daita.plugins.catalog.github import GitHubScanner

        catalog = self._build_catalog(github_org="testorg")
        assert any(isinstance(d, GitHubScanner) for d in catalog._discoverers)

    def test_no_github_scanner_by_default(self):
        from daita.plugins.catalog.github import GitHubScanner

        catalog = self._build_catalog()
        assert not any(isinstance(d, GitHubScanner) for d in catalog._discoverers)

    def test_has_profilers(self):
        from daita.plugins.catalog.profiler import PostgresProfiler, MySQLProfiler

        catalog = self._build_catalog()
        profiler_types = [type(p) for p in catalog._profilers]
        assert PostgresProfiler in profiler_types
        assert MySQLProfiler in profiler_types

    def test_has_apigateway_profiler(self):
        from daita.plugins.catalog import CatalogPlugin
        from daita.plugins.catalog.profiler import APIGatewayProfiler

        catalog = CatalogPlugin()
        catalog.add_profiler(APIGatewayProfiler())
        profiler_types = [type(p) for p in catalog._profilers]
        assert APIGatewayProfiler in profiler_types
        assert catalog._profilers[0].supports("apigateway")
        assert not catalog._profilers[0].supports("s3")

    def test_tools_registered(self):
        catalog = self._build_catalog()
        tools = catalog.get_tools()
        tool_names = {t.name for t in tools}
        assert "discover_infrastructure" in tool_names
        assert "find_store" in tool_names
        assert "profile_store" in tool_names
        assert "discover_postgres" in tool_names
        assert len(tools) == 10


# ---------------------------------------------------------------------------
# Env var helper
# ---------------------------------------------------------------------------


class TestCsvEnv:
    def test_csv_env_parses(self, monkeypatch):
        from agents.catalog_agent import _csv_env

        monkeypatch.setenv("TEST_CSV", "us-east-1, us-west-2, eu-west-1")
        result = _csv_env("TEST_CSV", [])
        assert result == ["us-east-1", "us-west-2", "eu-west-1"]

    def test_csv_env_default(self, monkeypatch):
        from agents.catalog_agent import _csv_env

        monkeypatch.delenv("TEST_CSV", raising=False)
        result = _csv_env("TEST_CSV", ["default"])
        assert result == ["default"]

    def test_csv_env_empty(self, monkeypatch):
        from agents.catalog_agent import _csv_env

        monkeypatch.setenv("TEST_CSV", "")
        result = _csv_env("TEST_CSV", ["fallback"])
        assert result == ["fallback"]


# ---------------------------------------------------------------------------
# API Gateway integration tests (mocked boto3)
# ---------------------------------------------------------------------------


class TestAPIGatewayIntegration:
    """End-to-end tests for API Gateway discovery, normalization, and persistence."""

    def _mock_rest_api_responses(self):
        """Return fake boto3 responses for a REST API."""
        return {
            "get_rest_api": {
                "id": "abc123",
                "name": "orders-api",
                "description": "Orders service REST API",
            },
            "get_resources": {
                "items": [
                    {
                        "id": "root",
                        "path": "/",
                        "resourceMethods": {},
                    },
                    {
                        "id": "res1",
                        "path": "/orders",
                        "resourceMethods": {"GET": {}, "POST": {}},
                    },
                    {
                        "id": "res2",
                        "path": "/orders/{id}",
                        "resourceMethods": {"GET": {}, "DELETE": {}},
                    },
                ]
            },
            "get_method_GET_/orders": {
                "authorizationType": "COGNITO_USER_POOLS",
                "apiKeyRequired": False,
                "methodIntegration": {
                    "type": "AWS_PROXY",
                    "uri": "arn:aws:lambda:us-east-1:123456:function:list-orders",
                },
            },
            "get_method_POST_/orders": {
                "authorizationType": "COGNITO_USER_POOLS",
                "apiKeyRequired": False,
                "methodIntegration": {
                    "type": "AWS_PROXY",
                    "uri": "arn:aws:lambda:us-east-1:123456:function:create-order",
                },
            },
            "get_method_GET_/orders/{id}": {
                "authorizationType": "COGNITO_USER_POOLS",
                "apiKeyRequired": False,
                "methodIntegration": {
                    "type": "AWS_PROXY",
                    "uri": "arn:aws:lambda:us-east-1:123456:function:get-order",
                },
            },
            "get_method_DELETE_/orders/{id}": {
                "authorizationType": "IAM",
                "apiKeyRequired": True,
                "methodIntegration": {
                    "type": "AWS_PROXY",
                    "uri": "arn:aws:lambda:us-east-1:123456:function:delete-order",
                },
            },
            "get_authorizers": {
                "items": [
                    {"id": "auth1", "name": "cognito-auth", "type": "COGNITO_USER_POOLS"},
                ]
            },
            "get_stage": {
                "stageName": "prod",
                "variables": {"env": "production"},
            },
        }

    async def test_discover_apigateway_rest(self):
        """Test discover_apigateway() with a mocked REST API."""
        from unittest.mock import MagicMock, patch
        import builtins

        responses = self._mock_rest_api_responses()

        mock_client = MagicMock()
        mock_client.get_rest_api.return_value = responses["get_rest_api"]
        mock_client.get_resources.return_value = responses["get_resources"]
        mock_client.get_authorizers.return_value = responses["get_authorizers"]
        mock_client.get_stage.return_value = responses["get_stage"]

        def mock_get_method(restApiId, resourceId, httpMethod):
            # Match by resource path + method
            for res in responses["get_resources"]["items"]:
                if res["id"] == resourceId:
                    key = f"get_method_{httpMethod}_{res['path']}"
                    return responses.get(key, {})
            return {}

        mock_client.get_method.side_effect = mock_get_method

        mock_session = MagicMock()
        mock_session.client.return_value = mock_client

        # boto3 is lazy-imported inside the function, so we patch builtins.__import__
        from daita.plugins.catalog.discovery import discover_apigateway

        real_import = builtins.__import__

        def patched_import(name, *args, **kwargs):
            if name == "boto3":
                mock_boto3 = MagicMock()
                mock_boto3.Session.return_value = mock_session
                return mock_boto3
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=patched_import):
            result = await discover_apigateway(
                api_id="abc123",
                api_type="rest",
                stage="prod",
                region="us-east-1",
            )

        assert result["api_type"] == "apigateway"
        assert result["api_name"] == "orders-api"
        assert result["protocol_type"] == "REST"
        assert result["endpoint_count"] == 4
        assert result["stage"] == "prod"
        assert len(result["authorizers"]) == 1

        # Verify endpoints have integration URIs
        endpoints_by_name = {
            f"{ep['method']} {ep['path']}": ep for ep in result["endpoints"]
        }
        assert "GET /orders" in endpoints_by_name
        assert "POST /orders" in endpoints_by_name
        assert "GET /orders/{id}" in endpoints_by_name
        assert "DELETE /orders/{id}" in endpoints_by_name
        assert "arn:aws:lambda" in endpoints_by_name["GET /orders"]["integration_uri"]

    def test_normalize_apigateway_full(self):
        """Test normalization produces correct schema structure."""
        from daita.plugins.catalog.normalizer import normalize_apigateway

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
                    "integration_uri": "arn:aws:lambda:us-east-1:123456:function:list-orders",
                },
                {
                    "path": "/orders",
                    "method": "POST",
                    "authorization": "COGNITO_USER_POOLS",
                    "api_key_required": False,
                    "integration_type": "AWS_PROXY",
                    "integration_uri": "arn:aws:lambda:us-east-1:123456:function:create-order",
                },
                {
                    "path": "/orders/{id}",
                    "method": "GET",
                    "authorization": "COGNITO_USER_POOLS",
                    "api_key_required": False,
                    "integration_type": "AWS_PROXY",
                    "integration_uri": "arn:aws:lambda:us-east-1:123456:function:get-order",
                },
                {
                    "path": "/orders/{id}",
                    "method": "DELETE",
                    "authorization": "IAM",
                    "api_key_required": True,
                    "integration_type": "AWS_PROXY",
                    "integration_uri": "arn:aws:lambda:us-east-1:123456:function:delete-order",
                },
            ],
            "endpoint_count": 4,
            "authorizers": [{"id": "auth1", "name": "cognito-auth", "type": "COGNITO_USER_POOLS"}],
            "stage_variables": {"env": "production"},
        }

        result = normalize_apigateway(raw)

        # Schema structure
        assert result["database_type"] == "apigateway"
        assert result["database_name"] == "orders-api"
        assert result["table_count"] == 1

        table = result["tables"][0]
        assert table["name"] == "orders-api"
        assert table["row_count"] == 4

        # Every endpoint becomes a column
        col_names = [c["name"] for c in table["columns"]]
        assert col_names == [
            "GET /orders", "POST /orders", "GET /orders/{id}", "DELETE /orders/{id}"
        ]

        # Column comments contain auth + Lambda ARN
        get_col = next(c for c in table["columns"] if c["name"] == "GET /orders")
        assert "COGNITO_USER_POOLS" in get_col["column_comment"]
        assert "arn:aws:lambda" in get_col["column_comment"]

        delete_col = next(c for c in table["columns"] if c["name"] == "DELETE /orders/{id}")
        assert "IAM" in delete_col["column_comment"]

        # Metadata has full integration map for lineage
        meta = result["metadata"]
        assert meta["endpoint"] == "https://abc123.execute-api.us-east-1.amazonaws.com/prod"
        assert meta["stage"] == "prod"
        assert meta["region"] == "us-east-1"
        assert meta["protocol_type"] == "REST"
        assert len(meta["authorizers"]) == 1
        assert "GET /orders" in meta["integrations"]
        assert meta["integrations"]["GET /orders"]["type"] == "AWS_PROXY"
        assert "list-orders" in meta["integrations"]["GET /orders"]["uri"]

    async def test_persist_apigateway_to_graph(self):
        """Test that apigateway schemas create API graph nodes."""
        from unittest.mock import AsyncMock, MagicMock
        from daita.plugins.catalog.persistence import persist_schema_to_graph

        schema = {
            "database_type": "apigateway",
            "database_name": "orders-api",
            "tables": [{"name": "orders-api", "row_count": 4, "columns": []}],
            "metadata": {
                "endpoint": "https://abc123.execute-api.us-east-1.amazonaws.com/prod",
                "integrations": {"GET /orders": {"type": "AWS_PROXY", "uri": "arn:..."}},
            },
        }

        mock_graph = AsyncMock()
        await persist_schema_to_graph(schema, mock_graph, agent_id="test-agent")

        # Verify add_node was called with an API node
        mock_graph.add_node.assert_called_once()
        node = mock_graph.add_node.call_args[0][0]
        assert node.name == "orders-api"
        assert node.node_type.value == "api"
        assert node.node_id == "api:orders-api"
        assert node.properties["database_type"] == "apigateway"
        assert "integrations" in node.properties["metadata"]

    def test_apigateway_profiler_supports(self):
        """Test APIGatewayProfiler only matches apigateway stores."""
        from daita.plugins.catalog.profiler import APIGatewayProfiler

        profiler = APIGatewayProfiler()
        assert profiler.supports("apigateway") is True
        assert profiler.supports("postgresql") is False
        assert profiler.supports("s3") is False
        assert profiler.supports("dynamodb") is False

    def test_aws_discoverer_includes_apigateway(self):
        """Test AWSDiscoverer default services include apigateway."""
        from daita.plugins.catalog.aws import AWSDiscoverer

        discoverer = AWSDiscoverer()
        assert "apigateway" in discoverer._services

    def test_catalog_agent_registers_apigateway_profiler(self):
        """Test the example agent configures APIGatewayProfiler correctly."""
        from daita.plugins.catalog import CatalogPlugin
        from daita.plugins.catalog.aws import AWSDiscoverer
        from daita.plugins.catalog.profiler import (
            PostgresProfiler, MySQLProfiler, DynamoDBProfiler,
            S3Profiler, APIGatewayProfiler,
        )

        # Replicate what create_agent does
        catalog = CatalogPlugin(auto_persist=True)
        catalog.add_discoverer(AWSDiscoverer(regions=["us-east-1"]))
        catalog.add_profiler(PostgresProfiler())
        catalog.add_profiler(MySQLProfiler())
        catalog.add_profiler(DynamoDBProfiler())
        catalog.add_profiler(S3Profiler())
        catalog.add_profiler(APIGatewayProfiler())

        profiler_types = [type(p) for p in catalog._profilers]
        assert APIGatewayProfiler in profiler_types
        assert len(profiler_types) == 5

        # Verify the right profiler matches apigateway stores
        matching = [p for p in catalog._profilers if p.supports("apigateway")]
        assert len(matching) == 1
        assert isinstance(matching[0], APIGatewayProfiler)
