"""Construction-time context supplied by hosted runtime surfaces."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass(frozen=True)
class HostRuntimeContext:
    """Host-provided construction context for framework runtimes.

    The host context is only a carrier used while runtimes are built. Runtime
    behavior still flows through plugins registered with ``ExtensionRegistry``.
    """

    surface: str | None = None
    delivery_defaults: tuple[str, ...] = ()
    services: dict[str, Any] = field(default_factory=dict)
    runtime_extensions: tuple[Any, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "delivery_defaults", tuple(self.delivery_defaults))
        object.__setattr__(self, "runtime_extensions", tuple(self.runtime_extensions))
        object.__setattr__(self, "services", dict(self.services))
        object.__setattr__(self, "metadata", dict(self.metadata))


_current_host_runtime_context: ContextVar[HostRuntimeContext | None] = ContextVar(
    "daita_host_runtime_context",
    default=None,
)


def current_host_runtime_context() -> HostRuntimeContext | None:
    """Return the active host runtime context for this execution context."""

    return _current_host_runtime_context.get()


@contextmanager
def host_runtime_context(context: HostRuntimeContext) -> Iterator[HostRuntimeContext]:
    """Temporarily install a host runtime context."""

    token = _current_host_runtime_context.set(context)
    try:
        yield context
    finally:
        _current_host_runtime_context.reset(token)
