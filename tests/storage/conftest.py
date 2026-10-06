from uuid import uuid4

import pytest

from daita.storage.postgres import PostgresStateConfig
from daita.storage.postgres_admin import (
    install_postgres_schema,
    provision_postgres_namespace,
)
from tests.storage.postgres_fixture import postgres_database as postgres_database


@pytest.fixture
def postgres_config(postgres_database):
    import psycopg
    from psycopg import sql

    dsn, _ = postgres_database
    namespace = uuid4()
    role = "state_" + uuid4().hex
    with psycopg.connect(dsn(), autocommit=True) as connection:
        install_postgres_schema(connection)
        connection.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD 'fixture-only'").format(
                sql.Identifier(role)
            )
        )
        provision_postgres_namespace(connection, namespace_id=namespace, role=role)
    return PostgresStateConfig(conninfo=dsn(user=role), namespace_id=namespace)
