"""Explicit dependencies shared by DB task-runtime components."""

from __future__ import annotations

from dataclasses import dataclass

from daita.plugins import ExtensionRegistry
from daita.runtime import RuntimeKernel, RuntimeStore

from ...models import DbRuntimeConfig


@dataclass(frozen=True)
class DbTaskContext:
    """Dependencies required by DB task-runtime behavior."""

    registry: ExtensionRegistry
    store: RuntimeStore
    kernel: RuntimeKernel
    config: DbRuntimeConfig
    runtime_id: str
