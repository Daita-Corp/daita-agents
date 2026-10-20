from __future__ import annotations

import os
from pathlib import Path

import pytest

from daita import Agent, PostgreSQLSource
from daita.adapters.registry import SourceTypeRegistry
from daita.security import EnvironmentSecretProvider, SecretReference
from tests.support.source_contract import (
    assert_source_type_contract,
    assert_tabular_read_contract,
)
from tests.support.workspace import workspace_for

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


async def test_postgresql_tabular_reads_satisfy_the_tabular_read_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, secrets = _fixture_source(monkeypatch)
    agent = await Agent.create(
        "postgresql-tabular-contract",
        root=tmp_path,
        secret_provider=secrets,
        workspace=workspace_for(tmp_path),
    )
    try:
        attached = await agent.attach(source)
        (products,) = [
            item
            for item in await agent.list_catalog_resources(source_id=attached.id)
            if item.native_identity == "analytics.products"
        ]
        await assert_tabular_read_contract(
            agent,
            source_id=attached.id,
            resource_id=products.id,
            columns=("product_id", "sku", "category"),
            key_column="category",
            range_column="product_id",
        )
    finally:
        await agent.close()
