"""Shared helpers extracted from ``test_subscription_providers.py``."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import Mapping
from dataclasses import replace
from typing import cast

import openai
import pytest

import daita.llm.providers.codex as codex_provider
import daita.llm.providers.subscription_cli.process as subscription_process
import daita.llm.subscription_auth as subscription_auth
from daita import Agent
from daita.llm.errors import (
    ModelProviderError,
    ProviderErrorCode,
    ProviderFailurePhase,
)
from daita.llm.factory import create_llm_provider
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelCallPolicy,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolDefinition,
)
from daita.llm.pricing import CostEstimateStatus
from daita.llm.providers import (
    ClaudeCodeSubscriptionProvider,
    CodexSubscriptionProvider,
)
from daita.llm.providers.mock import MockModelProvider
from daita.llm.subscription_auth import (
    CodexDevicePrompt,
    CodexOAuthCredential,
    login_codex_subscription,
)
from daita.security import SecretReference
from tests.support.workspace import workspace_for


def _jwt(account_id: str = "account-1", *, expires_at: int | None = None) -> str:
    def encode(value: object) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    return ".".join(
        (
            encode({"alg": "none"}),
            encode(
                {
                    "exp": expires_at or int(time.time()) + 3_600,
                    "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
                }
            ),
            encode("signature"),
        )
    )


def _credential(*, expired: bool = False) -> CodexOAuthCredential:
    expiry = time.time() - 10 if expired else time.time() + 3_600
    return CodexOAuthCredential(
        access_token=_jwt(expires_at=int(expiry)),
        refresh_token="refresh-token",
        expires_at=expiry,
        account_id="account-1",
    )
