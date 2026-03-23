"""
Unit tests for RESTPlugin.

Tests truncation, binary detection, and tool handler behaviour
without making real HTTP connections.
"""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from daita.plugins.rest import RESTPlugin

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_plugin():
    plugin = RESTPlugin(base_url="http://example.com")
    # Prevent any real connect() call
    plugin._session = MagicMock()
    return plugin


class MockResponse:
    """Minimal aiohttp response mock."""

    def __init__(self, status=200, content_type="application/json", body=""):
        self.status = status
        self.headers = {"content-type": content_type}
        self._body = body

    async def text(self):
        return self._body

    async def read(self):
        b = self._body
        return b if isinstance(b, bytes) else b.encode()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def _patch_session(plugin, response: MockResponse):
    """Wire plugin._session.request to return the given mock response."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=response)
    cm.__aexit__ = AsyncMock(return_value=False)
    plugin._session.request = MagicMock(return_value=cm)


# ---------------------------------------------------------------------------
# JSON responses
# ---------------------------------------------------------------------------


async def test_small_json_returns_parsed_object():
    plugin = make_plugin()
    body = json.dumps({"users": [{"id": 1}, {"id": 2}]})
    _patch_session(plugin, MockResponse(body=body))

    result = await plugin._tool_get({"endpoint": "/users"})

    assert result["data"] == {"users": [{"id": 1}, {"id": 2}]}
    assert "truncated" not in result


async def test_large_json_truncates_without_raising():
    """Large JSON must not crash with JSONDecodeError — the fix we applied."""
    plugin = make_plugin()
    # Build a JSON string well over 50 000 chars
    big_payload = {"items": ["x" * 100] * 600}  # ~60 000 chars
    body = json.dumps(big_payload)
    assert len(body) > 50_000

    _patch_session(plugin, MockResponse(body=body))

    result = await plugin._tool_get({"endpoint": "/items"})

    assert result["truncated"] is True
    assert result["total_chars"] == len(body)
    # data field is the truncated raw string
    assert len(result["data"]) == 50_000
    assert result["endpoint"] == "/items"


async def test_large_json_data_field_is_string_prefix():
    """Verify the truncated data value is the raw JSON string prefix."""
    plugin = make_plugin()
    body = json.dumps({"k": "v" * 60_000})  # ~60K chars, over 50K limit
    assert len(body) > 50_000
    _patch_session(plugin, MockResponse(body=body))

    result = await plugin._tool_get({"endpoint": "/big"})
    assert isinstance(result["data"], str)
    assert result["data"] == body[:50_000]


async def test_json_list_response_returned_as_parsed():
    plugin = make_plugin()
    body = json.dumps([1, 2, 3])
    _patch_session(plugin, MockResponse(body=body))

    result = await plugin._tool_get({"endpoint": "/ids"})
    assert result["data"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Text responses
# ---------------------------------------------------------------------------


async def test_text_response_returned_in_content_key():
    plugin = make_plugin()
    _patch_session(plugin, MockResponse(content_type="text/plain", body="hello world"))

    result = await plugin._tool_get({"endpoint": "/readme"})
    assert result["content"] == "hello world"
    assert "truncated" not in result


async def test_large_text_response_truncated():
    plugin = make_plugin()
    big_text = "a" * 60_000
    _patch_session(plugin, MockResponse(content_type="text/plain", body=big_text))

    result = await plugin._tool_get({"endpoint": "/bigtext"})
    assert result["truncated"] is True
    assert result["total_chars"] == 60_000
    assert len(result["content"]) == 50_000


# ---------------------------------------------------------------------------
# Binary responses
# ---------------------------------------------------------------------------


async def test_binary_response_returns_metadata_only():
    plugin = make_plugin()
    _patch_session(
        plugin,
        MockResponse(
            content_type="application/octet-stream",
            body=b"\x89PNG\r\n" * 1000,
        ),
    )

    result = await plugin._tool_get({"endpoint": "/image.png"})
    assert result["binary"] is True
    assert result["content_type"] == "application/octet-stream"
    assert "size" in result
    assert "data" not in result
    assert "content" not in result


# ---------------------------------------------------------------------------
# HTTP error
# ---------------------------------------------------------------------------


async def test_http_error_raises_runtime_error():
    plugin = make_plugin()
    _patch_session(
        plugin,
        MockResponse(status=404, content_type="text/plain", body="Not found"),
    )

    with pytest.raises(RuntimeError, match="HTTP 404"):
        await plugin._tool_get({"endpoint": "/missing"})


# ---------------------------------------------------------------------------
# POST / PUT / PATCH / DELETE tool handlers
# ---------------------------------------------------------------------------


async def test_post_tool_handler():
    plugin = make_plugin()
    body = json.dumps({"id": 42})
    _patch_session(plugin, MockResponse(body=body))

    result = await plugin._tool_post({"endpoint": "/users", "data": {"name": "Alice"}})
    assert result["data"] == {"id": 42}
    assert result["endpoint"] == "/users"


async def test_delete_tool_handler_binary_response():
    plugin = make_plugin()
    _patch_session(
        plugin, MockResponse(content_type="application/zip", body=b"PK\x03\x04")
    )

    result = await plugin._tool_delete({"endpoint": "/export.zip"})
    assert result["binary"] is True
