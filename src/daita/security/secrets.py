"""Runtime secret resolution without durable secret values."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from dataclasses import dataclass
import os
import re
from typing import Protocol, TypeVar, cast, runtime_checkable

from .._installation import repair_guidance
from ..errors import ConfigError

_ENVIRONMENT_NAME = re.compile(r"[A-Z][A-Z0-9_]{0,127}\Z")
_KEYCHAIN_ACCOUNT = re.compile(r"[^\s\x00-\x1f\x7f]{1,256}\Z")
_KEYCHAIN_SERVICE = re.compile(r"[^\r\n\x00]{1,256}\Z")
_T = TypeVar("_T")


class SecretResolutionError(ConfigError):
    """Normalized secret-provider failure that never contains a secret value."""

    def __init__(self, code: str, message: str) -> None:
        if not isinstance(code, str) or not code.strip():
            raise ValueError("secret error code must be a non-empty string")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("secret error message must be a non-empty string")
        self.code = code
        super().__init__(message, section="secrets", error_code=code)


@dataclass(frozen=True, slots=True)
class SecretReference:
    """Persistable identity of a secret, never the secret itself."""

    scheme: str
    name: str

    def __post_init__(self) -> None:
        if self.scheme not in {"env", "keychain"}:
            raise ValueError("secret reference scheme must be 'env' or 'keychain'")
        if self.scheme == "env" and (
            not isinstance(self.name, str)
            or _ENVIRONMENT_NAME.fullmatch(self.name) is None
        ):
            raise ValueError(
                "environment secret name must use uppercase letters, digits, "
                "and underscores"
            )
        if self.scheme == "keychain" and (
            not isinstance(self.name, str)
            or _KEYCHAIN_ACCOUNT.fullmatch(self.name) is None
        ):
            raise ValueError(
                "keychain secret account must be a bounded non-whitespace identifier"
            )

    @classmethod
    def environment(cls, name: str) -> SecretReference:
        return cls("env", name)

    @classmethod
    def keychain(cls, account: str) -> SecretReference:
        return cls("keychain", account)

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


@runtime_checkable
class KeychainStore(SecretProvider, Protocol):
    """Read and mutate only explicit keychain references."""

    async def set(self, reference: SecretReference, value: str) -> None: ...

    async def delete(self, reference: SecretReference) -> None: ...


class EmptySecretProvider:
    """Explicitly configured provider that can never resolve a value."""

    async def resolve(self, reference: SecretReference) -> str:
        _reference(reference)
        raise SecretResolutionError(
            "secret_not_found",
            "The configured secret is unavailable.",
        )


class EnvironmentSecretProvider:
    """Resolve only explicit environment references at call time."""

    async def resolve(self, reference: SecretReference) -> str:
        _reference(reference)
        if reference.scheme != "env":
            raise SecretResolutionError(
                "secret_scheme_unsupported",
                "The secret provider does not support this reference scheme.",
            )
        value = os.environ.get(reference.name)
        if value is None or not value:
            raise SecretResolutionError(
                "secret_not_found",
                "The configured environment secret is unavailable.",
            )
        return value


class _KeyringClient(Protocol):
    def get_password(self, service_name: str, username: str) -> str | None: ...

    def set_password(
        self,
        service_name: str,
        username: str,
        password: str,
    ) -> None: ...

    def delete_password(self, service_name: str, username: str) -> None: ...


class KeychainSecretProvider:
    """Resolve explicit keychain references through a lazily imported backend."""

    __slots__ = ("_client", "_service_name")

    def __init__(
        self,
        service_name: str = "daita",
        *,
        client: _KeyringClient | None = None,
    ) -> None:
        if (
            not isinstance(service_name, str)
            or _KEYCHAIN_SERVICE.fullmatch(service_name) is None
        ):
            raise ValueError("keychain service_name must be a bounded string")
        if client is not None and not callable(getattr(client, "get_password", None)):
            raise TypeError("keychain client must provide get_password")
        self._service_name = service_name
        self._client = client

    @property
    def client(self) -> _KeyringClient:
        if self._client is None:
            try:
                import keyring
            except ImportError:
                raise ImportError(
                    "Daita's keychain runtime dependency is unavailable. "
                    f"{repair_guidance()}"
                ) from None
            self._client = cast(_KeyringClient, keyring)
        return self._client

    async def resolve(self, reference: SecretReference) -> str:
        _reference(reference)
        if reference.scheme != "keychain":
            raise SecretResolutionError(
                "secret_scheme_unsupported",
                "The secret provider does not support this reference scheme.",
            )
        failure: SecretResolutionError | None = None
        try:
            value = await asyncio.to_thread(
                self.client.get_password,
                self._service_name,
                reference.name,
            )
        except ImportError:
            raise
        except Exception:
            failure = SecretResolutionError(
                "secret_provider_unavailable",
                "The configured keychain provider is unavailable.",
            )
            value = None
        if failure is not None:
            raise failure
        if value is None or value == "":
            raise SecretResolutionError(
                "secret_not_found",
                "The configured keychain secret is unavailable.",
            )
        if not isinstance(value, str):
            raise SecretResolutionError(
                "secret_provider_invalid_response",
                "The configured keychain provider returned an invalid response.",
            )
        return value

    async def set(self, reference: SecretReference, value: str) -> None:
        """Store one bounded value without retaining or exposing it."""

        _keychain_reference(reference)
        if (
            not isinstance(value, str)
            or not value
            or len(value.encode("utf-8")) > 64 * 1_024
        ):
            raise ValueError("keychain secret must be non-empty and at most 64 KiB")
        client = self.client
        if not callable(getattr(client, "set_password", None)):
            raise SecretResolutionError(
                "secret_provider_unavailable",
                "The configured keychain provider cannot store secrets.",
            )
        failed = False
        try:
            await _run_blocking_to_completion(
                client.set_password,
                self._service_name,
                reference.name,
                value,
            )
        except ImportError:
            raise
        except Exception:
            failed = True
        finally:
            value = ""
        if failed:
            raise SecretResolutionError(
                "secret_provider_unavailable",
                "The configured keychain provider could not store the secret.",
            )

    async def delete(self, reference: SecretReference) -> None:
        """Delete one explicit keychain account without exposing its value."""

        _keychain_reference(reference)
        client = self.client
        if not callable(getattr(client, "delete_password", None)):
            raise SecretResolutionError(
                "secret_provider_unavailable",
                "The configured keychain provider cannot delete secrets.",
            )
        try:
            await _run_blocking_to_completion(
                client.delete_password,
                self._service_name,
                reference.name,
            )
        except ImportError:
            raise
        except Exception:
            remaining: object
            try:
                remaining = await _run_blocking_to_completion(
                    client.get_password,
                    self._service_name,
                    reference.name,
                )
            except Exception:
                remaining = object()
            if remaining is None or remaining == "":
                return
            raise SecretResolutionError(
                "secret_provider_unavailable",
                "The configured keychain provider could not delete the secret.",
            ) from None

    def __repr__(self) -> str:
        return "KeychainSecretProvider()"


class CompositeSecretProvider:
    """Try explicit providers in order without retaining resolved values."""

    __slots__ = ("_providers",)

    def __init__(self, providers: Iterable[SecretProvider]) -> None:
        if isinstance(providers, (str, bytes)):
            raise TypeError("secret providers must be an iterable of providers")
        items = tuple(providers)
        if not items:
            raise ValueError("composite secret provider requires at least one provider")
        if any(not isinstance(provider, SecretProvider) for provider in items):
            raise TypeError("each secret provider must implement SecretProvider")
        self._providers = items

    @property
    def providers(self) -> tuple[SecretProvider, ...]:
        return self._providers

    async def resolve(self, reference: SecretReference) -> str:
        _reference(reference)
        for provider in self._providers:
            failure: SecretResolutionError | None = None
            try:
                value = await provider.resolve(reference)
            except asyncio.CancelledError:
                raise
            except ImportError:
                raise
            except SecretResolutionError as error:
                if error.code in {
                    "secret_not_found",
                    "secret_scheme_unsupported",
                }:
                    continue
                failure = SecretResolutionError(
                    error.code,
                    "The configured secret provider failed.",
                )
                value = None
            except Exception:
                failure = SecretResolutionError(
                    "secret_provider_unavailable",
                    "The configured secret provider failed.",
                )
                value = None
            if failure is not None:
                raise failure
            if not isinstance(value, str) or not value:
                raise SecretResolutionError(
                    "secret_provider_invalid_response",
                    "The configured secret provider returned an invalid response.",
                )
            return value
        raise SecretResolutionError(
            "secret_not_found",
            "The configured secret is unavailable.",
        )

    def __repr__(self) -> str:
        return f"CompositeSecretProvider(provider_count={len(self._providers)})"


def default_secret_provider(
    primary: SecretProvider | None = None,
) -> CompositeSecretProvider:
    """Compose injected, keychain, then environment resolution in ADR order."""

    providers: tuple[SecretProvider, ...]
    if primary is None:
        providers = (KeychainSecretProvider(), EnvironmentSecretProvider())
    elif isinstance(primary, EmptySecretProvider):
        providers = (primary,)
    else:
        if not isinstance(primary, SecretProvider):
            raise TypeError("primary must implement SecretProvider")
        providers = (
            primary,
            KeychainSecretProvider(),
            EnvironmentSecretProvider(),
        )
    return CompositeSecretProvider(providers)


def _reference(reference: SecretReference) -> None:
    if not isinstance(reference, SecretReference):
        raise TypeError("reference must be a SecretReference")


def _keychain_reference(reference: SecretReference) -> None:
    _reference(reference)
    if reference.scheme != "keychain":
        raise SecretResolutionError(
            "secret_scheme_unsupported",
            "Keychain mutation requires a keychain reference.",
        )


async def _run_blocking_to_completion(
    callback: Callable[..., _T],
    *args: object,
) -> _T:
    """Finish an admitted keychain mutation before propagating cancellation."""

    worker = asyncio.create_task(asyncio.to_thread(callback, *args))
    cancelled = False
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancelled = True
            continue
    result = worker.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


__all__ = [
    "CompositeSecretProvider",
    "EmptySecretProvider",
    "EnvironmentSecretProvider",
    "KeychainSecretProvider",
    "KeychainStore",
    "SecretProvider",
    "SecretReference",
    "SecretResolutionError",
    "default_secret_provider",
]
