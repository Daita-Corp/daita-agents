"""Errors raised while admitting, composing and operating one agent home."""

from __future__ import annotations

from collections.abc import Mapping

from ..errors import AgentError


class AgentHomeError(AgentError):
    """Base failure for isolated local agent-home admission."""


class AgentNameError(AgentHomeError, ValueError):
    pass


class AgentAlreadyExistsError(AgentHomeError):
    pass


class AgentNotFoundError(AgentHomeError):
    pass


class AgentIdentityMismatchError(AgentHomeError):
    pass


class AgentModelConfigurationError(AgentHomeError):
    """A saved model route whose profile facts can no longer be admitted."""


class HostActiveError(AgentHomeError):
    code = "host_active"

    def __init__(
        self,
        message: str = "Agent operation failed.",
        *,
        holder: Mapping[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        # role, pid and started_at read from the holder's lock file, if known.
        self.holder = None if holder is None else dict(holder)
        super().__init__(message, **kwargs)  # type: ignore[arg-type]


class AgentNotConfiguredError(AgentHomeError):
    pass
