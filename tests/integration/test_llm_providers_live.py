"""
Live smoke tests for LLM provider adapters.

These tests are intentionally small and opt-in. They verify that each provider
can make basic non-streaming and streaming requests through Daita's provider
abstraction.

Examples:
    DAITA_RUN_LIVE_LLM=1 OPENAI_API_KEY=sk-... pytest \
        tests/integration/test_llm_providers_live.py \
        -m "requires_llm and integration" -v

    DAITA_RUN_LIVE_LLM=1 ANTHROPIC_API_KEY=sk-ant-... pytest \
        tests/integration/test_llm_providers_live.py \
        -m "requires_llm and integration" -v

    DAITA_RUN_LIVE_LLM=1 GOOGLE_API_KEY=... pytest \
        tests/integration/test_llm_providers_live.py \
        -m "requires_llm and integration" -v

    DAITA_RUN_LIVE_LLM=1 XAI_API_KEY=xai-... pytest \
        tests/integration/test_llm_providers_live.py \
        -m "requires_llm and integration" -v

    DAITA_RUN_LIVE_LLM=1 OLLAMA_BASE_URL=http://localhost:11434/v1 pytest \
        tests/integration/test_llm_providers_live.py -m "requires_llm and integration" -v

    # Or set DAITA_RUN_OLLAMA=1 to test the default local Ollama URL.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from daita.llm.factory import create_llm_provider

load_dotenv(Path.cwd() / ".env")

PROMPT = "Reply with exactly one word: ok"


def _require_live_llm_enabled() -> None:
    if os.environ.get("DAITA_RUN_LIVE_LLM") != "1":
        pytest.skip("Set DAITA_RUN_LIVE_LLM=1 to run live provider smoke tests")


def _require_env(*names: str) -> str:
    _require_live_llm_enabled()
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    pytest.skip(f"Missing environment variable: one of {', '.join(names)}")


def _assert_text_response(result) -> None:
    assert isinstance(result, str)
    assert result.strip()


async def _assert_streaming_text(llm) -> None:
    chunks = []
    async for chunk in await llm.generate(PROMPT, stream=True):
        if getattr(chunk, "type", None) == "text" and getattr(chunk, "content", ""):
            chunks.append(chunk.content)
        if len("".join(chunks)) >= 2:
            break

    assert "".join(chunks).strip()


@pytest.mark.integration
@pytest.mark.requires_llm
async def test_openai_live_smoke():
    llm = create_llm_provider(
        "openai",
        model=os.environ.get("OPENAI_TEST_MODEL", "gpt-5.4-mini"),
        api_key=_require_env("OPENAI_API_KEY"),
        temperature=0,
        max_tokens=8,
    )

    result = await llm.generate(PROMPT)

    _assert_text_response(result)


@pytest.mark.integration
@pytest.mark.requires_llm
async def test_openai_live_stream_smoke():
    llm = create_llm_provider(
        "openai",
        model=os.environ.get("OPENAI_TEST_MODEL", "gpt-5.4-mini"),
        api_key=_require_env("OPENAI_API_KEY"),
        temperature=0,
        max_tokens=8,
    )

    await _assert_streaming_text(llm)


@pytest.mark.integration
@pytest.mark.requires_llm
async def test_anthropic_live_smoke():
    llm = create_llm_provider(
        "anthropic",
        model=os.environ.get("ANTHROPIC_TEST_MODEL", "claude-haiku-4-5"),
        api_key=_require_env("ANTHROPIC_API_KEY"),
        temperature=0,
        max_tokens=8,
    )

    result = await llm.generate(PROMPT)

    _assert_text_response(result)


@pytest.mark.integration
@pytest.mark.requires_llm
async def test_anthropic_live_stream_smoke():
    llm = create_llm_provider(
        "anthropic",
        model=os.environ.get("ANTHROPIC_TEST_MODEL", "claude-haiku-4-5"),
        api_key=_require_env("ANTHROPIC_API_KEY"),
        temperature=0,
        max_tokens=8,
    )

    await _assert_streaming_text(llm)


@pytest.mark.integration
@pytest.mark.requires_llm
async def test_gemini_live_smoke():
    llm = create_llm_provider(
        "gemini",
        model=os.environ.get("GEMINI_TEST_MODEL", "gemini-2.5-flash-lite"),
        api_key=_require_env("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        temperature=0,
        max_tokens=8,
    )

    result = await llm.generate(PROMPT)

    _assert_text_response(result)


@pytest.mark.integration
@pytest.mark.requires_llm
async def test_gemini_live_stream_smoke():
    llm = create_llm_provider(
        "gemini",
        model=os.environ.get("GEMINI_TEST_MODEL", "gemini-2.5-flash-lite"),
        api_key=_require_env("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        temperature=0,
        max_tokens=8,
    )

    await _assert_streaming_text(llm)


@pytest.mark.integration
@pytest.mark.requires_llm
async def test_grok_live_smoke():
    llm = create_llm_provider(
        "grok",
        model=os.environ.get("GROK_TEST_MODEL", "grok-4.20"),
        api_key=_require_env("XAI_API_KEY", "GROK_API_KEY"),
        temperature=0,
        max_tokens=8,
    )

    result = await llm.generate(PROMPT)

    _assert_text_response(result)


@pytest.mark.integration
@pytest.mark.requires_llm
async def test_grok_live_stream_smoke():
    llm = create_llm_provider(
        "grok",
        model=os.environ.get("GROK_TEST_MODEL", "grok-4.20"),
        api_key=_require_env("XAI_API_KEY", "GROK_API_KEY"),
        temperature=0,
        max_tokens=8,
    )

    await _assert_streaming_text(llm)


@pytest.mark.integration
@pytest.mark.requires_llm
async def test_ollama_live_smoke():
    _require_live_llm_enabled()
    if not (
        os.environ.get("DAITA_RUN_OLLAMA") == "1" or os.environ.get("OLLAMA_BASE_URL")
    ):
        pytest.skip("Set OLLAMA_BASE_URL or DAITA_RUN_OLLAMA=1 to run Ollama smoke")
    llm = create_llm_provider(
        "ollama",
        model=os.environ.get("OLLAMA_TEST_MODEL", "llama3.1"),
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        temperature=0,
        max_tokens=8,
    )

    result = await llm.generate(PROMPT)

    _assert_text_response(result)


@pytest.mark.integration
@pytest.mark.requires_llm
async def test_ollama_live_stream_smoke():
    _require_live_llm_enabled()
    if not (
        os.environ.get("DAITA_RUN_OLLAMA") == "1" or os.environ.get("OLLAMA_BASE_URL")
    ):
        pytest.skip("Set OLLAMA_BASE_URL or DAITA_RUN_OLLAMA=1 to run Ollama smoke")
    llm = create_llm_provider(
        "ollama",
        model=os.environ.get("OLLAMA_TEST_MODEL", "llama3.1"),
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        temperature=0,
        max_tokens=8,
    )

    await _assert_streaming_text(llm)
