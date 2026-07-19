"""Runtime secret resolution without durable secret values."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Protocol, runtime_checkable

_ENVIRONMENT_NAME = re.compile(r"[A-Z][A-Z0-9_]{0,127}\Z")


class SecretResolutionError(RuntimeError):
    """Normalized secret-provider failure that never contains a secret value."""

    def __init__(self, code: str, message: str) -> None:
        if not isinstance(code, str) or not code.strip():
            raise ValueError("secret error code must be a non-empty string")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("secret error message must be a non-empty string")
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class SecretReference:
    """Persistable identity of a secret, never the secret itself."""

    scheme: str
    name: str

    def __post_init__(self) -> None:
        if self.scheme != "env":
            raise ValueError("secret reference scheme must be 'env'")
        if (
            not isinstance(self.name, str)
            or _ENVIRONMENT_NAME.fullmatch(self.name) is None
        ):
            raise ValueError(
                "environment secret name must use uppercase letters, digits, and underscores"
            )

    @classmethod
    def environment(cls, name: str) -> SecretReference:
        return cls("env", name)

    @classmethod
    def parse(cls, value: str) -> SecretReference:
        if not isinstance(value, str) or ":" not in value:
            raise ValueError("secret reference must contain a scheme")
        scheme, name = value.split(":", 1)
        return cls(scheme=scheme, name=name)

    def to_uri(self) -> str:
        return f"{self.scheme}:{self.name}"


@runtime_checkable
class SecretProvider(Protocol):
    """Resolve a reference at the last responsible runtime boundary."""

    async def resolve(self, reference: SecretReference) -> str: ...


class EnvironmentSecretProvider:
    """Resolve only explicit environment references at call time."""

    async def resolve(self, reference: SecretReference) -> str:
        if not isinstance(reference, SecretReference):
            raise TypeError("reference must be a SecretReference")
        value = os.environ.get(reference.name)
        if value is None or not value:
            raise SecretResolutionError(
                "secret_not_found",
                "The configured environment secret is unavailable.",
            )
        return value


__all__ = [
    "EnvironmentSecretProvider",
    "SecretProvider",
    "SecretReference",
    "SecretResolutionError",
]
