"""Errors raised while admitting, composing and operating one agent home."""

from __future__ import annotations

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


class AgentNotConfiguredError(AgentHomeError):
    pass
