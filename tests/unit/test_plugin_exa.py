"""
Unit tests for ExaSearchPlugin.

Tests response parsing, snippet fallback logic, contents payload construction,
parameter validation, and error mapping without making real API calls.
"""

import pytest
from unittest.mock import MagicMock, patch
from daita.plugins.exa import (
    ExaSearchPlugin,
    exa_search,
    _extract_snippet,
    _result_from_item,
)
from daita.core.exceptions import (
    AuthenticationError,
    RateLimitError,
    TimeoutError as DaitaTimeoutError,
    ConnectionError as DaitaConnectionError,
    PermanentError,
    TransientError,
    RetryableError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_plugin(**overrides):
    kwargs = {"api_key": "exa-test-key"}
    kwargs.update(overrides)
    plugin = ExaSearchPlugin(**kwargs)
    # Stub the client so connect() is a no-op.
    plugin._client = MagicMock()
    return plugin


class FakeResultObject:
    """Stand-in for exa-py's typed result objects."""

    def __init__(self, **fields):
        for k, v in fields.items():
            setattr(self, k, v)


def make_response(items):
    """Build a fake SearchResponse with .results."""
    response = MagicMock()
    response.results = items
    return response


# ---------------------------------------------------------------------------
# Snippet fallback (cascades summary -> highlights -> text)
# ---------------------------------------------------------------------------


def test_snippet_prefers_summary():
    item = {
        "summary": "Concise summary",
        "highlights": ["highlight"],
        "text": "full text",
    }
    assert _extract_snippet(item) == "Concise summary"


def test_snippet_falls_back_to_highlights_when_summary_missing():
    item = {"highlights": ["first highlight", "second highlight"], "text": "full"}
    assert _extract_snippet(item) == "first highlight ... second highlight"


def test_snippet_falls_back_to_text_when_highlights_missing():
    item = {"text": "raw page text " * 100}
    snippet = _extract_snippet(item)
    assert snippet.startswith("raw page text")
    assert len(snippet) <= 500


def test_snippet_empty_when_no_content_present():
    assert _extract_snippet({"title": "x", "url": "https://x.com"}) == ""


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def test_result_from_dict_item():
    item = {
        "title": "Example",
        "url": "https://example.com",
        "score": 0.87,
        "publishedDate": "2025-01-01",
        "author": "Jane",
        "highlights": ["a", "b"],
        "summary": "Summary text",
    }
    result = _result_from_item(item)
    assert result.title == "Example"
    assert result.url == "https://example.com"
    assert result.score == 0.87
    assert result.published_date == "2025-01-01"
    assert result.author == "Jane"
    assert result.highlights == ["a", "b"]
    assert result.summary == "Summary text"
    assert result.snippet == "Summary text"


def test_result_handles_missing_fields():
    result = _result_from_item({"url": "https://x.com"})
    assert result.title == ""
    assert result.score == 0.0
    assert result.snippet == ""
    assert result.highlights == []


# ---------------------------------------------------------------------------
# Response coercion
# ---------------------------------------------------------------------------


def test_coerce_items_handles_typed_objects():
    typed = FakeResultObject(
        title="t",
        url="https://u",
        score=0.5,
        publishedDate=None,
        author=None,
        text=None,
        highlights=[],
        summary=None,
    )
    response = make_response([typed])
    items = ExaSearchPlugin._coerce_items(response)
    assert len(items) == 1
    assert items[0]["title"] == "t"
    assert items[0]["url"] == "https://u"


def test_coerce_items_handles_dict_response():
    response = {"results": [{"title": "t", "url": "https://u"}]}
    items = ExaSearchPlugin._coerce_items(response)
    assert items == [{"title": "t", "url": "https://u"}]


def test_coerce_items_returns_empty_when_no_results():
    response = MagicMock(spec=[])
    items = ExaSearchPlugin._coerce_items(response)
    assert items == []


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_missing_api_key_raises():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="Exa API key is required"):
            ExaSearchPlugin(api_key=None)


def test_invalid_search_type_raises():
    with pytest.raises(ValueError, match="search_type must be one of"):
        ExaSearchPlugin(api_key="k", search_type="bogus")


def test_invalid_category_raises():
    with pytest.raises(ValueError, match="category must be one of"):
        ExaSearchPlugin(api_key="k", category="not-a-category")


def test_picks_up_api_key_from_env():
    with patch.dict("os.environ", {"EXA_API_KEY": "env-key"}, clear=False):
        plugin = ExaSearchPlugin()
        assert plugin._api_key == "env-key"


# ---------------------------------------------------------------------------
# Contents payload construction
# ---------------------------------------------------------------------------


def test_build_contents_default_highlights_only():
    plugin = make_plugin()
    contents = plugin._build_contents(None, None, None)
    assert contents == {"highlights": True}


def test_build_contents_combines_text_highlights_summary():
    plugin = make_plugin(
        include_text=True, include_highlights=True, include_summary=True
    )
    contents = plugin._build_contents(None, None, None)
    assert contents["highlights"] is True
    assert contents["summary"] is True
    assert contents["text"] == {"maxCharacters": 1000}


def test_build_contents_returns_none_when_all_disabled():
    plugin = make_plugin(include_highlights=False)
    contents = plugin._build_contents(None, None, None)
    assert contents is None


def test_build_contents_per_call_overrides_take_precedence():
    plugin = make_plugin(include_highlights=True)
    contents = plugin._build_contents(
        include_text=False, include_highlights=False, include_summary=True
    )
    assert contents == {"summary": True}


# ---------------------------------------------------------------------------
# search() integration with mocked SDK
# ---------------------------------------------------------------------------


async def test_search_returns_normalized_results(monkeypatch):
    plugin = make_plugin(include_highlights=True)

    fake_response = make_response(
        [
            FakeResultObject(
                title="Doc",
                url="https://example.com/doc",
                score=0.9,
                publishedDate="2025-02-01",
                author="A",
                text=None,
                highlights=["key passage"],
                summary=None,
            )
        ]
    )

    async def fake_call(self, kwargs, contents):
        # Verify the contents payload reached the SDK call layer.
        assert contents == {"highlights": True}
        assert kwargs["query"] == "test query"
        assert kwargs["num_results"] == 5
        assert kwargs["type"] == "auto"
        return fake_response

    monkeypatch.setattr(ExaSearchPlugin, "_call_search_with_contents", fake_call)

    out = await plugin.search("test query")

    assert out["success"] is True
    assert out["count"] == 1
    assert out["results"][0]["title"] == "Doc"
    assert out["results"][0]["snippet"] == "key passage"
    assert out["results"][0]["highlights"] == ["key passage"]


async def test_search_passes_filters_to_sdk(monkeypatch):
    plugin = make_plugin(include_highlights=False)
    captured = {}

    async def fake_call(self, kwargs):
        captured.update(kwargs)
        return make_response([])

    monkeypatch.setattr(ExaSearchPlugin, "_call_search", fake_call)

    await plugin.search(
        "q",
        num_results=15,
        search_type="neural",
        category="news",
        include_domains=["example.com"],
        exclude_domains=["spam.com"],
        start_published_date="2025-01-01",
        end_published_date="2025-12-31",
    )

    assert captured["num_results"] == 15
    assert captured["type"] == "neural"
    assert captured["category"] == "news"
    assert captured["include_domains"] == ["example.com"]
    assert captured["exclude_domains"] == ["spam.com"]
    assert captured["start_published_date"] == "2025-01-01"
    assert captured["end_published_date"] == "2025-12-31"


async def test_search_invalid_type_raises():
    plugin = make_plugin()
    with pytest.raises(ValueError, match="search_type"):
        await plugin.search("q", search_type="invalid")


async def test_search_invalid_category_raises():
    plugin = make_plugin()
    with pytest.raises(ValueError, match="category"):
        await plugin.search("q", category="not-real")


async def test_find_similar_returns_results(monkeypatch):
    plugin = make_plugin(include_highlights=True)
    fake_response = make_response(
        [
            FakeResultObject(
                title="Sim",
                url="https://similar.com",
                score=0.7,
                publishedDate=None,
                author=None,
                text=None,
                highlights=[],
                summary="related page",
            )
        ]
    )

    async def fake_call(self, url, num_results, contents):
        assert url == "https://seed.com"
        return fake_response

    monkeypatch.setattr(ExaSearchPlugin, "_call_find_similar_with_contents", fake_call)

    out = await plugin.find_similar("https://seed.com")
    assert out["success"] is True
    assert out["url"] == "https://seed.com"
    assert out["results"][0]["snippet"] == "related page"


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_message, expected",
    [
        ("HTTP 429 too many requests", RateLimitError),
        ("Unauthorized: invalid api key", AuthenticationError),
        ("403 Forbidden", PermanentError),
        ("400 Bad Request: malformed", PermanentError),
        ("Request timed out", DaitaTimeoutError),
        ("Connection refused", DaitaConnectionError),
        ("503 Service Unavailable", TransientError),
        ("something weird went wrong", RetryableError),
    ],
)
def test_handle_search_error_maps_to_daita_hierarchy(raw_message, expected):
    plugin = make_plugin()
    mapped = plugin._handle_search_error(Exception(raw_message))
    assert isinstance(mapped, expected)


async def test_search_error_propagates_as_daita_error(monkeypatch):
    plugin = make_plugin()

    async def boom(self, kwargs, contents):
        raise Exception("HTTP 429 rate limit")

    monkeypatch.setattr(ExaSearchPlugin, "_call_search_with_contents", boom)
    with pytest.raises(RateLimitError):
        await plugin.search("q")


# ---------------------------------------------------------------------------
# Tool registration / disabled state
# ---------------------------------------------------------------------------


def test_get_tools_exposes_search_and_find_similar():
    plugin = make_plugin()
    tools = plugin.get_tools()
    names = {t.name for t in tools}
    assert names == {"search_web", "find_similar"}


def test_factory_function_returns_plugin():
    plugin = exa_search(api_key="k")
    assert isinstance(plugin, ExaSearchPlugin)


def test_plugin_not_constructible_without_env_or_arg():
    """Mirrors 'tool not registered when EXA_API_KEY unset' — the plugin
    refuses to instantiate, which prevents tool registration."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError):
            exa_search()


# ---------------------------------------------------------------------------
# connect() sets the integration tracking header
# ---------------------------------------------------------------------------


async def test_connect_sets_integration_header(monkeypatch):
    plugin = ExaSearchPlugin(api_key="k")

    fake_client = MagicMock()
    fake_client.headers = {}

    fake_module = MagicMock()
    fake_module.Exa = MagicMock(return_value=fake_client)
    monkeypatch.setitem(__import__("sys").modules, "exa_py", fake_module)

    await plugin.connect()

    assert plugin._client is fake_client
    assert fake_client.headers["x-exa-integration"] == "daita-agents"
