"""Shared helpers extracted from ``test_call_policy.py``."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from decimal import Decimal

import pytest

from daita.config import AgentConfig
from daita.hosting.embedded import (
    _decode_call_policy,
    _encode_call_policy,
    _model_execution_contracts,
)
from daita.llm import ModelCallPolicy, RetryPolicy
from daita.llm._lifecycle import materialize_request
from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelRequest,
    ModelUsage,
    TextBlock,
)
from daita.llm.pricing import CostEstimate
from daita.llm.providers.mock import MockModelProvider


def request(**kwargs):
    return ModelRequest(
        (CanonicalMessage(MessageRole.USER, content=(TextBlock("public"),)),), **kwargs
    )
