from __future__ import annotations

import builtins
from collections.abc import Mapping

import pytest

from daita import (
    Agent,
    CompositeSecretProvider,
    EmptySecretProvider,
    EnvironmentSecretProvider,
    KeychainSecretProvider,
    SecretProvider,
    SecretReference,
)
from daita.adapters import postgresql_query as postgresql_query_owner
from daita.security import SecretResolutionError, default_secret_provider


class _Keychain:
    def __init__(self, value: str | None) -> None:
        self._value = value
        self.calls: list[tuple[str, str]] = []

    def get_password(self, service_name: str, username: str) -> str | None:
        self.calls.append((service_name, username))
        return self._value


class _MissingProvider:
    def __init__(self) -> None:
        self.references: list[SecretReference] = []

    async def resolve(self, reference: SecretReference) -> str:
        self.references.append(reference)
        raise SecretResolutionError(
            "secret_not_found",
            "The injected provider has no matching secret.",
        )


def test_public_secret_provider_surface_is_explicit_and_runtime_checkable() -> None:
    providers = (
        EmptySecretProvider(),
        EnvironmentSecretProvider(),
        KeychainSecretProvider(client=_Keychain(None)),
    )

    assert all(isinstance(provider, SecretProvider) for provider in providers)
    assert isinstance(CompositeSecretProvider(providers), SecretProvider)


async def test_empty_provider_returns_only_a_normalized_not_found_error() -> None:
    with pytest.raises(SecretResolutionError) as captured:
        await EmptySecretProvider().resolve(
            SecretReference.environment("DAITA_MISSING_SECRET")
        )

    assert captured.value.code == "secret_not_found"
    assert captured.value.__cause__ is None
    assert "DAITA_MISSING_SECRET" not in str(captured.value)


async def test_keychain_reference_round_trips_and_value_is_never_retained() -> None:
    reference = SecretReference.keychain("postgres-reader")
    client = _Keychain("private-password")
    provider = KeychainSecretProvider("daita-test", client=client)

    assert SecretReference.parse(reference.to_uri()) == reference
    assert await provider.resolve(reference) == "private-password"
    assert client.calls == [("daita-test", "postgres-reader")]
    assert "private-password" not in repr(provider)
    assert "private-password" not in repr(reference)


async def test_keychain_backend_is_lazy_and_has_an_exact_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    original_import = builtins.__import__

    def blocked_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        imported.append(name)
        if name == "keyring":
            raise ImportError("blocked for contract test")
        return original_import(name, globals, locals, fromlist, level)

    provider = KeychainSecretProvider()
    assert imported == []
    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ImportError) as captured:
        await provider.resolve(SecretReference.keychain("model-api-key"))

    assert str(captured.value) == (
        "keyring is required. Install with: " "pip install 'daita-agents[keychain]'"
    )
    assert imported.count("keyring") == 1


async def test_default_composite_uses_injected_then_keychain_then_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = _MissingProvider()
    reference = SecretReference.environment("DAITA_COMPOSITE_SECRET")
    monkeypatch.setenv("DAITA_COMPOSITE_SECRET", "environment-value")
    provider = default_secret_provider(missing)

    assert [type(item) for item in provider.providers] == [
        _MissingProvider,
        KeychainSecretProvider,
        EnvironmentSecretProvider,
    ]
    assert await provider.resolve(reference) == "environment-value"
    assert missing.references == [reference]
    assert "environment-value" not in repr(provider)


async def test_explicit_empty_provider_disables_implicit_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DAITA_DISABLED_SECRET", "must-not-resolve")
    provider = default_secret_provider(EmptySecretProvider())

    with pytest.raises(SecretResolutionError) as captured:
        await provider.resolve(SecretReference.environment("DAITA_DISABLED_SECRET"))

    assert captured.value.code == "secret_not_found"
    assert "must-not-resolve" not in str(captured.value)


async def test_embedded_agent_composes_an_injected_secret_provider_once(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty = EmptySecretProvider()
    seen: list[SecretProvider | None] = []
    real_composer = default_secret_provider

    def record_composition(
        provider: SecretProvider | None = None,
    ) -> CompositeSecretProvider:
        seen.append(provider)
        return real_composer(provider)

    monkeypatch.setattr(
        postgresql_query_owner,
        "default_secret_provider",
        record_composition,
    )
    agent = await Agent.create(
        "empty-secret-provider",
        root=tmp_path,
        secret_provider=empty,
    )
    try:
        assert seen == [empty]
    finally:
        await agent.close()


async def test_composite_drops_untrusted_provider_failure_diagnostics() -> None:
    class FailingProvider:
        async def resolve(self, reference: SecretReference) -> str:
            raise RuntimeError("credential=private-password")

    provider = CompositeSecretProvider((FailingProvider(),))

    with pytest.raises(SecretResolutionError) as captured:
        await provider.resolve(SecretReference.environment("DAITA_SECRET"))

    assert captured.value.code == "secret_provider_unavailable"
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert "private-password" not in str(captured.value)
