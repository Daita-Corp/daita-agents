"""Shared helpers extracted from ``test_streaming.py``."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import aclosing
from dataclasses import replace
from typing import Any, cast

import pytest

from daita.llm.errors import (
    ModelProviderError,
    ProviderAttempt,
    ProviderErrorCode,
    ProviderFailureDiagnostic,
    ProviderFailurePhase,
)
from daita.llm.factory import create_llm_provider
from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelRequest,
    ModelStreamCompleted,
    ModelStreamEvent,
    ModelTextDelta,
    ModelToolCallDelta,
    TextBlock,
    ToolDefinition,
)
from daita.llm.providers.anthropic import AnthropicMessagesProvider
from daita.llm.providers.gemini import GeminiProvider
from daita.llm.providers.grok import GrokProvider
from daita.llm.providers.ollama import OllamaProvider
from daita.llm.providers.openai import OpenAIResponsesProvider
from daita.llm.providers.openai_compatible import OpenAICompatibleProvider


def _openai_text_response(text: str) -> dict[str, object]:
    return {
        "id": "resp-1",
        "status": "completed",
        "model": "test-model",
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": ""},
                    {"type": "output_text", "text": text},
                ],
            }
        ],
        "usage": None,
    }
