"""
Unit tests for S3Plugin.

Tests format detection, binary/pandas metadata-only path, text truncation,
CSV row capping, error mapping, tool schemas, write/list/delete/copy handlers,
focus exposure, and pagination — without real AWS calls.
"""

import json
import io
import pytest
from unittest.mock import MagicMock, patch

botocore_exceptions = pytest.importorskip(
    "botocore.exceptions", reason="botocore (boto3) not installed"
)
ClientError = botocore_exceptions.ClientError

from daita.plugins.s3 import S3Plugin
from daita.plugins import ExtensionRegistry
from daita.runtime import AccessMode, Operation, RiskLevel, Task
from tests.unit.plugins.projection_helpers import projected_tools
from daita.core.exceptions import (
    NotFoundError,
    PermissionError as DaitaPermissionError,
    AuthenticationError,
    TransientError,
    PluginError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_plugin():
    plugin = S3Plugin(bucket="test-bucket", region="us-east-1")
    # Inject a fake boto3 client so connect() is never needed
    plugin._client = MagicMock()
    return plugin


def _mock_get_object(plugin, content: bytes, content_type: str = "text/plain"):
    """Wire plugin._client.get_object to return the given bytes via _sync_get_object."""
    plugin._client.get_object = MagicMock(
        return_value={
            "Body": MagicMock(read=MagicMock(return_value=content)),
            "ContentType": content_type,
        }
    )


def _mock_head_object(
    plugin, size: int, content_type: str = "application/octet-stream"
):
    plugin._client.head_object = MagicMock(
        return_value={
            "ContentLength": size,
            "ContentType": content_type,
            "LastModified": "2024-01-01",
        }
    )


def _mock_put_object(plugin):
    """Wire plugin._client.put_object to return a minimal ETag response."""
    plugin._client.put_object = MagicMock(return_value={"ETag": '"abc123"'})


def _make_client_error(code: str, message: str = "test error") -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": message}},
        operation_name="TestOperation",
    )


def _operation_and_task(capability, input_payload):
    operation = Operation(
        id="op-1",
        operation_type=next(iter(capability.operation_types)),
        request=input_payload,
        required_evidence=capability.output_evidence,
    )
    task = Task(
        id="task-1",
        operation_id=operation.id,
        capability_id=capability.id,
        executor_id=capability.executor,
        input=input_payload,
        required_evidence=capability.output_evidence,
    )
    return operation, task


# ---------------------------------------------------------------------------
# Extension declarations
# ---------------------------------------------------------------------------


def test_s3_plugin_declares_extension_first_contract():
    plugin = make_plugin()
    registry = ExtensionRegistry()

    registry.register(plugin)

    assert plugin.manifest.id == "s3"
    assert registry.plugin_ids == ("s3",)
    assert {capability.id for capability in registry.capabilities} == {
        "s3.object.read",
        "s3.object.write",
        "s3.object.list",
        "s3.object.delete",
        "s3.object.head",
    }
    assert {view.name for view in registry.tool_views} == {
        "read_s3_file",
        "write_s3_file",
        "list_s3_objects",
        "delete_s3_file",
        "head_s3_object",
    }
    assert registry.get_tool_view_owner("read_s3_file") == "s3"
    assert registry.evidence_schemas[0].kind == "s3.operation.result"


def test_s3_capabilities_carry_access_and_safety_metadata():
    plugin = make_plugin()
    registry = ExtensionRegistry()

    registry.register(plugin)
    by_id = {capability.id: capability for capability in registry.capabilities}

    assert by_id["s3.object.read"].access is AccessMode.READ
    assert by_id["s3.object.read"].risk is RiskLevel.LOW
    assert by_id["s3.object.read"].side_effecting is False
    assert by_id["s3.object.write"].access is AccessMode.WRITE
    assert by_id["s3.object.write"].risk is RiskLevel.MEDIUM
    assert by_id["s3.object.write"].idempotent is True
    assert by_id["s3.object.delete"].risk is RiskLevel.HIGH
    assert by_id["s3.object.delete"].side_effecting is True


async def test_s3_executor_returns_typed_operation_evidence():
    plugin = make_plugin()
    _mock_get_object(plugin, b"Hello, storage!", content_type="text/plain")
    registry = ExtensionRegistry()
    registry.register(plugin)
    capability = registry.get_capability("s3.object.read", owner="s3")
    executor = registry.get_executor(capability.executor)
    operation, task = _operation_and_task(capability, {"key": "notes.txt"})

    evidence = await executor.execute(task, operation, {})

    assert evidence[0].kind == "s3.operation.result"
    assert evidence[0].owner == "s3"
    assert evidence[0].payload["operation"] == "s3.object.read"
    assert evidence[0].payload["request"] == {"key": "notes.txt"}
    assert evidence[0].payload["result"]["content"] == "Hello, storage!"
    assert evidence[0].metadata["capability_id"] == "s3.object.read"


async def test_s3_write_executor_uses_existing_tool_handler():
    plugin = make_plugin()
    _mock_put_object(plugin)
    registry = ExtensionRegistry()
    registry.register(plugin)
    capability = registry.get_capability("s3.object.write", owner="s3")
    executor = registry.get_executor(capability.executor)
    operation, task = _operation_and_task(
        capability,
        {"key": "out/hello.txt", "data": "hello"},
    )

    evidence = await executor.execute(task, operation, {})

    assert evidence[0].payload["operation"] == "s3.object.write"
    assert evidence[0].payload["result"] == {
        "key": "out/hello.txt",
        "size": 5,
        "location": "s3://test-bucket/out/hello.txt",
        "bucket": "test-bucket",
    }


async def test_s3_registry_setup_and_teardown_use_connector_lifecycle():
    plugin = make_plugin()
    calls = []

    async def fake_connect():
        calls.append("connect")
        plugin._client = object()

    async def fake_disconnect():
        calls.append("disconnect")
        plugin._client = None

    plugin._client = None
    plugin.connect = fake_connect
    plugin.disconnect = fake_disconnect
    registry = ExtensionRegistry()
    registry.register(plugin)

    await registry.setup_plugin("s3", context=None)
    assert plugin.is_connected is True

    await registry.teardown_all()

    assert calls == ["connect", "disconnect"]
    assert plugin.is_connected is False


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def test_detect_format_json():
    plugin = make_plugin()
    assert plugin._detect_format("data/file.json") == "json"


def test_detect_format_csv():
    plugin = make_plugin()
    assert plugin._detect_format("data/file.csv") == "csv"


def test_detect_format_parquet():
    plugin = make_plugin()
    assert plugin._detect_format("data/file.parquet") == "pandas"


def test_detect_format_xlsx():
    plugin = make_plugin()
    assert plugin._detect_format("data/file.xlsx") == "pandas"


def test_detect_format_unknown_is_bytes():
    plugin = make_plugin()
    assert plugin._detect_format("data/file.dat") == "bytes"


# ---------------------------------------------------------------------------
# Binary / unknown extensions -> metadata only
# ---------------------------------------------------------------------------


async def test_binary_file_returns_metadata_only():
    plugin = make_plugin()
    _mock_head_object(plugin, size=1024, content_type="application/octet-stream")

    result = await plugin._tool_read_file({"key": "archive.tar.gz"})

    assert result["binary"] is True
    assert result["size"] == 1024
    assert result["content_type"] == "application/octet-stream"
    assert "content" not in result
    assert "rows" not in result


async def test_parquet_file_returns_metadata_only():
    """parquet maps to 'pandas' format — should go through binary path, not str(df)."""
    plugin = make_plugin()
    _mock_head_object(plugin, size=204800, content_type="application/octet-stream")

    result = await plugin._tool_read_file({"key": "warehouse/fact_sales.parquet"})

    assert result["binary"] is True
    assert result["size"] == 204800
    assert "content" not in result


async def test_xlsx_file_returns_metadata_only():
    plugin = make_plugin()
    _mock_head_object(
        plugin,
        size=8192,
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    result = await plugin._tool_read_file({"key": "report.xlsx"})

    assert result["binary"] is True
    assert "content" not in result


# ---------------------------------------------------------------------------
# Text files — truncation
# ---------------------------------------------------------------------------


async def test_small_text_file_returned_in_full():
    plugin = make_plugin()
    content = b"Hello, world!"
    _mock_get_object(plugin, content, content_type="text/plain")

    result = await plugin._tool_read_file({"key": "notes.txt"})

    assert result["content"] == "Hello, world!"
    assert result["truncated"] is False
    assert "total_chars" not in result


async def test_large_text_file_truncated_at_50k():
    plugin = make_plugin()
    big_text = ("a" * 60_000).encode()
    _mock_get_object(plugin, big_text, content_type="text/plain")

    result = await plugin._tool_read_file({"key": "large.txt"})

    assert result["truncated"] is True
    assert result["total_chars"] == 60_000
    assert len(result["content"]) == 50_000


# ---------------------------------------------------------------------------
# JSON files — truncation
# ---------------------------------------------------------------------------


async def test_small_json_file_returned_in_full():
    plugin = make_plugin()
    payload = {"users": [{"id": i} for i in range(5)]}
    content = json.dumps(payload).encode()
    _mock_get_object(plugin, content, content_type="application/json")

    result = await plugin._tool_read_file({"key": "users.json"})

    assert result["truncated"] is False
    assert '"users"' in result["content"]


async def test_large_json_file_truncated():
    plugin = make_plugin()
    big_payload = {"items": ["x" * 100] * 600}
    content = json.dumps(big_payload).encode()
    assert len(content) > 50_000
    _mock_get_object(plugin, content, content_type="application/json")

    result = await plugin._tool_read_file({"key": "big.json"})

    assert result["truncated"] is True
    assert len(result["content"]) == 50_000


# ---------------------------------------------------------------------------
# CSV files — row capping
# ---------------------------------------------------------------------------


async def test_csv_file_rows_capped_at_200():
    plugin = make_plugin()
    # Build CSV with 250 data rows
    lines = ["id,name"] + [f"{i},user_{i}" for i in range(250)]
    content = "\n".join(lines).encode()
    _mock_get_object(plugin, content, content_type="text/csv")

    result = await plugin._tool_read_file({"key": "big.csv"})

    assert result["truncated"] is True
    assert result["total_rows"] == 250
    assert len(result["rows"]) == 200


async def test_small_csv_not_truncated():
    plugin = make_plugin()
    lines = ["id,name"] + [f"{i},user_{i}" for i in range(10)]
    content = "\n".join(lines).encode()
    _mock_get_object(plugin, content, content_type="text/csv")

    result = await plugin._tool_read_file({"key": "small.csv"})

    assert result["truncated"] is False
    assert len(result["rows"]) == 10


# ---------------------------------------------------------------------------
# Result includes key and bucket
# ---------------------------------------------------------------------------


async def test_result_includes_key_and_bucket():
    plugin = make_plugin()
    _mock_head_object(plugin, size=100)

    result = await plugin._tool_read_file({"key": "some/path/file.bin"})

    assert result["key"] == "some/path/file.bin"
    assert result["bucket"] == "test-bucket"


# ---------------------------------------------------------------------------
# connect / disconnect
# ---------------------------------------------------------------------------


async def test_connect_noop_if_connected():
    plugin = make_plugin()
    # _client is already set — connect() should return immediately without boto3
    original_client = plugin._client
    await plugin.connect()
    assert plugin._client is original_client


async def test_connect_404_raises_not_found():
    plugin = S3Plugin(bucket="missing-bucket")
    err = _make_client_error("NoSuchBucket", "The bucket does not exist")

    with patch("boto3.Session") as mock_session_cls:
        mock_client = MagicMock()
        mock_client.list_objects_v2.side_effect = err
        mock_session_cls.return_value.client.return_value = mock_client

        with pytest.raises(NotFoundError):
            await plugin.connect()


async def test_connect_403_raises_permission_error():
    plugin = S3Plugin(bucket="restricted-bucket")
    err = _make_client_error("AccessDenied", "Access Denied")

    with patch("boto3.Session") as mock_session_cls:
        mock_client = MagicMock()
        mock_client.list_objects_v2.side_effect = err
        mock_session_cls.return_value.client.return_value = mock_client

        with pytest.raises(DaitaPermissionError):
            await plugin.connect()


async def test_connect_creates_client():
    plugin = S3Plugin(bucket="my-bucket")

    with patch("boto3.Session") as mock_session_cls:
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {
            "Contents": [],
            "IsTruncated": False,
        }
        mock_session_cls.return_value.client.return_value = mock_client

        await plugin.connect()

    assert plugin._client is mock_client


# ---------------------------------------------------------------------------
# put_object
# ---------------------------------------------------------------------------


async def test_put_dict_as_json():
    plugin = make_plugin()
    _mock_put_object(plugin)

    result = await plugin.put_object("out/data.json", {"key": "value"})

    assert result["key"] == "out/data.json"
    assert result["content_type"] == "application/json"
    assert result["size"] > 0
    assert result["etag"] == '"abc123"'


async def test_put_string_as_utf8():
    plugin = make_plugin()
    _mock_put_object(plugin)

    result = await plugin.put_object("out/notes.txt", "hello world")

    assert result["content_type"] == "text/plain"
    assert result["size"] == len(b"hello world")


async def test_put_list_as_json():
    plugin = make_plugin()
    _mock_put_object(plugin)

    result = await plugin.put_object("out/items.json", [{"id": 1}, {"id": 2}])

    assert result["content_type"] == "application/json"
    call_args = plugin._client.put_object.call_args[1]
    body = call_args["Body"]
    parsed = json.loads(body.decode("utf-8"))
    assert isinstance(parsed, list)
    assert len(parsed) == 2


async def test_put_error_maps_to_plugin_error():
    plugin = make_plugin()
    plugin._client.put_object = MagicMock(
        side_effect=_make_client_error("InvalidArgument", "Invalid argument")
    )

    with pytest.raises(PluginError):
        await plugin.put_object("out/fail.json", {"x": 1})


# ---------------------------------------------------------------------------
# list_objects
# ---------------------------------------------------------------------------


async def test_list_basic():
    plugin = make_plugin()
    plugin._client.list_objects_v2 = MagicMock(
        return_value={
            "Contents": [
                {"Key": "a.txt", "Size": 10, "LastModified": "2024-01-01"},
                {"Key": "b.txt", "Size": 20, "LastModified": "2024-01-02"},
            ],
            "IsTruncated": False,
        }
    )

    objects, is_truncated = await plugin.list_objects(prefix="")

    assert len(objects) == 2
    assert is_truncated is False
    assert objects[0]["Key"] == "a.txt"


async def test_list_empty():
    plugin = make_plugin()
    plugin._client.list_objects_v2 = MagicMock(
        return_value={
            "Contents": [],
            "IsTruncated": False,
        }
    )

    objects, is_truncated = await plugin.list_objects()

    assert objects == []
    assert is_truncated is False


async def test_list_with_continuation_token():
    plugin = make_plugin()
    plugin._client.list_objects_v2 = MagicMock(
        return_value={
            "Contents": [{"Key": "page2.txt", "Size": 5, "LastModified": "2024-01-03"}],
            "IsTruncated": False,
        }
    )

    objects, _ = await plugin.list_objects(continuation_token="token-abc")

    call_kwargs = plugin._client.list_objects_v2.call_args[1]
    assert call_kwargs["ContinuationToken"] == "token-abc"


# ---------------------------------------------------------------------------
# delete / copy
# ---------------------------------------------------------------------------


async def test_delete_returns_metadata():
    plugin = make_plugin()
    plugin._client.delete_object = MagicMock(return_value={})

    result = await plugin.delete_object("temp/old.txt")

    assert result["key"] == "temp/old.txt"
    assert result["deleted"] is True


async def test_copy_returns_metadata():
    plugin = make_plugin()
    plugin._client.copy_object = MagicMock(
        return_value={"CopyObjectResult": {"ETag": '"copyetag"'}}
    )

    result = await plugin.copy_object("src/file.txt", "dst/file.txt")

    assert result["source_key"] == "src/file.txt"
    assert result["dest_key"] == "dst/file.txt"
    assert result["etag"] == '"copyetag"'


# ---------------------------------------------------------------------------
# Projected tool schema validation
# ---------------------------------------------------------------------------


def test_projected_tools_return_five_tools():
    plugin = make_plugin()
    assert len(projected_tools(plugin)) == 5


def test_write_schema_no_type_constraint():
    plugin = make_plugin()
    tools = projected_tools(plugin)
    write_tool = tools["write_s3_file"]
    data_prop = write_tool.parameters["properties"]["data"]
    # Must NOT have "type": "object" — that blocks strings and lists
    assert "type" not in data_prop


def test_read_schema_has_focus_param():
    plugin = make_plugin()
    tools = projected_tools(plugin)
    read_tool = tools["read_s3_file"]
    assert "focus" in read_tool.parameters["properties"]


def test_list_schema_has_focus_param():
    plugin = make_plugin()
    tools = projected_tools(plugin)
    list_tool = tools["list_s3_objects"]
    assert "focus" in list_tool.parameters["properties"]


def test_projected_tools_carry_declared_capability_metadata():
    plugin = make_plugin()
    tools = projected_tools(plugin)

    assert tools["read_s3_file"].capability_ids == ("s3.object.read",)
    assert tools["write_s3_file"].capability_ids == ("s3.object.write",)
    assert tools["list_s3_objects"].capability_ids == ("s3.object.list",)
    assert tools["delete_s3_file"].capability_ids == ("s3.object.delete",)
    assert tools["head_s3_object"].capability_ids == ("s3.object.head",)
    assert tools["read_s3_file"].side_effecting is False
    assert tools["write_s3_file"].side_effecting is True
    assert tools["delete_s3_file"].side_effecting is True
    assert tools["head_s3_object"].retry_safe is True


# ---------------------------------------------------------------------------
# Tool handler: write string and list
# ---------------------------------------------------------------------------


async def test_tool_write_string():
    plugin = make_plugin()
    _mock_put_object(plugin)

    result = await plugin._tool_write_file({"key": "out/hello.txt", "data": "hello"})

    assert result["key"] == "out/hello.txt"
    assert result["location"] == "s3://test-bucket/out/hello.txt"
    assert result["bucket"] == "test-bucket"


async def test_tool_write_list():
    plugin = make_plugin()
    _mock_put_object(plugin)

    result = await plugin._tool_write_file(
        {
            "key": "out/items.json",
            "data": [{"id": 1}, {"id": 2}],
        }
    )

    assert result["key"] == "out/items.json"
    call_body = plugin._client.put_object.call_args[1]["Body"]
    parsed = json.loads(call_body.decode("utf-8"))
    assert isinstance(parsed, list)


# ---------------------------------------------------------------------------
# Tool handler: list with truncation surfacing
# ---------------------------------------------------------------------------


async def test_tool_list_default():
    plugin = make_plugin()
    plugin._client.list_objects_v2 = MagicMock(
        return_value={
            "Contents": [
                {"Key": f"file_{i}.txt", "Size": i * 10, "LastModified": "2024-01-01"}
                for i in range(5)
            ],
            "IsTruncated": False,
        }
    )

    result = await plugin._tool_list_objects({"prefix": "file_"})

    assert len(result["objects"]) == 5
    assert result["truncated"] is False
    assert result["bucket"] == "test-bucket"


async def test_tool_list_s3_truncated_surfaces_note():
    plugin = make_plugin()
    plugin._client.list_objects_v2 = MagicMock(
        return_value={
            "Contents": [
                {"Key": f"f{i}.txt", "Size": 1, "LastModified": "2024-01-01"}
                for i in range(10)
            ],
            "IsTruncated": True,
        }
    )

    result = await plugin._tool_list_objects({})

    assert result["truncated"] is True
    assert "note" in result


# ---------------------------------------------------------------------------
# Tool handler: delete and head
# ---------------------------------------------------------------------------


async def test_tool_delete():
    plugin = make_plugin()
    plugin._client.delete_object = MagicMock(return_value={})

    result = await plugin._tool_delete_file({"key": "old/file.txt"})

    assert result["deleted"] is True
    assert result["key"] == "old/file.txt"
    assert result["bucket"] == "test-bucket"


async def test_tool_head():
    plugin = make_plugin()
    _mock_head_object(plugin, size=512, content_type="text/plain")

    result = await plugin._tool_head_object({"key": "notes.txt"})

    assert result["size"] == 512
    assert result["content_type"] == "text/plain"
    assert result["key"] == "notes.txt"
    assert result["bucket"] == "test-bucket"


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def test_404_to_not_found():
    plugin = make_plugin()
    err = _make_client_error("NoSuchKey", "Key not found")
    mapped = plugin._map_s3_error(err, "get_object")
    assert isinstance(mapped, NotFoundError)


def test_403_to_permission():
    plugin = make_plugin()
    err = _make_client_error("AccessDenied", "Access Denied")
    mapped = plugin._map_s3_error(err, "get_object")
    assert isinstance(mapped, DaitaPermissionError)


def test_invalid_key_to_auth_error():
    plugin = make_plugin()
    err = _make_client_error("InvalidAccessKeyId", "Invalid key")
    mapped = plugin._map_s3_error(err, "connect")
    assert isinstance(mapped, AuthenticationError)


def test_503_to_transient():
    plugin = make_plugin()
    err = _make_client_error("503", "Service unavailable")
    mapped = plugin._map_s3_error(err, "list_objects")
    assert isinstance(mapped, TransientError)


def test_slowdown_to_transient():
    plugin = make_plugin()
    err = _make_client_error("SlowDown", "Please reduce your request rate")
    mapped = plugin._map_s3_error(err, "get_object")
    assert isinstance(mapped, TransientError)


def test_unknown_client_error_to_plugin_error():
    plugin = make_plugin()
    err = _make_client_error("SomeUnknownCode", "Something went wrong")
    mapped = plugin._map_s3_error(err, "put_object")
    assert isinstance(mapped, PluginError)


def test_import_error_to_import_error_with_extra_hint():
    plugin = make_plugin()
    err = ImportError("boto3 not found")
    mapped = plugin._map_s3_error(err, "connect")
    assert isinstance(mapped, ImportError)
    assert "pip install 'daita-agents[aws]'" in str(mapped)
