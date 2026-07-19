from __future__ import annotations

import pytest

from daita.security import (
    EnvironmentSecretProvider,
    SecretReference,
    SecretResolutionError,
)
from daita import SecretProvider as PublicSecretProvider


def test_secret_reference_is_a_bounded_persistable_reference() -> None:
    reference = SecretReference.environment("DAITA_TEST_POSTGRES_PASSWORD")

    assert reference.to_uri() == "env:DAITA_TEST_POSTGRES_PASSWORD"
    assert SecretReference.parse(reference.to_uri()) == reference
    assert PublicSecretProvider is not None
    assert "password-value" not in repr(reference)

    with pytest.raises(ValueError, match="environment secret name"):
        SecretReference.environment("not-valid")
    with pytest.raises(ValueError, match="secret reference scheme"):
        SecretReference.parse("vault:postgres")


async def test_environment_secret_provider_resolves_only_at_use_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = SecretReference.environment("DAITA_TEST_POSTGRES_PASSWORD")
    provider = EnvironmentSecretProvider()
    monkeypatch.setenv("DAITA_TEST_POSTGRES_PASSWORD", "private-value")

    assert await provider.resolve(reference) == "private-value"

    monkeypatch.delenv("DAITA_TEST_POSTGRES_PASSWORD")
    with pytest.raises(SecretResolutionError) as caught:
        await provider.resolve(reference)
    assert caught.value.code == "secret_not_found"
    assert "private-value" not in str(caught.value)
