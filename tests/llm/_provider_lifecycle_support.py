"""Shared helpers extracted from ``test_provider_lifecycle.py``."""

from __future__ import annotations

import asyncio
import importlib
import json
from collections.abc import AsyncGenerator
from contextvars import ContextVar
from dataclasses import replace
from decimal import Decimal
from typing import Any, cast

import anthropic
import httpx
import openai
import pytest
from google import genai

import daita.hosting.embedded as embedded_module
import daita.llm.factory as factory_module
from daita import Agent, AgentConfig
from daita.llm._lifecycle import closing_stream
from daita.llm.errors import ModelProviderError
from daita.llm.factory import create_model_route_provider
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ModelStreamCompleted,
    ModelStreamEvent,
    ModelTextDelta,
    TextBlock,
    ToolCall,
)
from daita.llm.profiles import reviewed_model_profile
from daita.llm.providers.anthropic import AnthropicMessagesProvider
from daita.llm.providers.gemini import GeminiProvider
from daita.llm.providers.openai import OpenAIResponsesProvider
from daita.llm.providers.openai_compatible import OpenAICompatibleProvider
from daita.llm.routing import ModelRoute, ModelRouteCandidate, RetryPolicy
from daita.security import EmptySecretProvider, SecretReference


class _ResponseBody(httpx.AsyncByteStream):
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.closed = False
        self.reading = asyncio.Event()

    async def __aiter__(self):
        yield b'data: {"type":"response.output_text.delta","delta":"done"}\n\n'
        if self.mode == "cancel":
            self.reading.set()
            await asyncio.Event().wait()
        if self.mode == "malformed":
            yield b'data: {"type":"response.output_text.delta","delta":123}\n\n'
            return
        response = {
            "id": "offline",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "gpt-5.6-terra",
            "output": [
                {
                    "id": "message",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "done", "annotations": []}
                    ],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
        yield (
            "data: "
            + json.dumps({"type": "response.completed", "response": response})
            + "\n\n"
        ).encode()

    async def aclose(self) -> None:
        self.closed = True
