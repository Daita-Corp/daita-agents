"""Run local agents in one background host that clients reach over a Unix socket."""

from .client import HostUnavailableError, connect
from .protocol import HostVersionMismatch

__all__ = ["HostUnavailableError", "HostVersionMismatch", "connect"]
