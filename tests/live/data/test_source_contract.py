from __future__ import annotations

import os

import pytest

from daita import PostgreSQLSource
from daita.adapters.registry import SourceTypeRegistry
from daita.security import EnvironmentSecretProvider, SecretReference
from tests.support.source_contract import (
    assert_source_type_contract,
)

pytestmark = [
    pytest.mark.contract,
    pytest.mark.requires_db,
    pytest.mark.skipif(
        os.environ.get("DAITA_RUN_POSTGRES_FIXTURE") != "1",
        reason=(
            "set DAITA_RUN_POSTGRES_FIXTURE=1 after starting "
            "tests/fixtures/postgresql/compose.yaml"
        ),
    ),
]


def _fixture_source(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[PostgreSQLSource, EnvironmentSecretProvider]:
    monkeypatch.setenv(
        "DAITA_FIXTURE_POSTGRES_PASSWORD",
        os.environ.get("DAITA_FIXTURE_POSTGRES_PASSWORD", "daita_fixture_password"),
    )
    secrets = EnvironmentSecretProvider()
    source = PostgreSQLSource(
        host="127.0.0.1",
        port=int(os.environ.get("DAITA_FIXTURE_POSTGRES_PORT", "55432")),
        database="daita_fixture",
        username="daita_reader",
        credential=SecretReference.environment("DAITA_FIXTURE_POSTGRES_PASSWORD"),
        schemas=("analytics",),
        ssl_mode="disable",
        name="Contract PostgreSQL",
        secret_provider=secrets,
    )
    return source, secrets


async def test_postgresql_source_type_satisfies_the_source_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, secrets = _fixture_source(monkeypatch)

    await assert_source_type_contract(
        SourceTypeRegistry().require("postgresql"), source, secret_provider=secrets
    )
