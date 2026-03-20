"""
Unit tests for S3Plugin.

Tests format detection, binary/pandas metadata-only path, text truncation,
and CSV row capping without real AWS calls.
"""

import json
import io
import pytest
from unittest.mock import MagicMock, AsyncMock
from daita.plugins.s3 import S3Plugin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_plugin():
    plugin = S3Plugin(bucket="test-bucket", region="us-east-1")
    # Inject a fake boto3 client so connect() is never needed
    plugin._client = MagicMock()
    return plugin


def _mock_get_object(plugin, content: bytes, content_type: str = "text/plain"):
    """Wire plugin._client.get_object to return the given bytes."""
    mock_body = MagicMock()
    mock_body.read = MagicMock(return_value=content)
    plugin._client.get_object = MagicMock(
        return_value={"Body": mock_body, "ContentType": content_type}
    )


def _mock_head_object(plugin, size: int, content_type: str = "application/octet-stream"):
    plugin._client.head_object = MagicMock(
        return_value={"ContentLength": size, "ContentType": content_type, "LastModified": "2024-01-01"}
    )


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
# Binary / unknown extensions → metadata only
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
    _mock_head_object(plugin, size=8192, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

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
