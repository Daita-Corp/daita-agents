"""Database-enforced scope, admission and physical schema qualification."""

from dataclasses import replace
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict

from daita.identity import AgentIdentity
from daita.storage.postgres import PostgresStateStore
from daita.storage.postgres_admin import (
    install_postgres_schema,
    provision_postgres_namespace,
)
from daita.storage.postgres_schema import POSTGRES_TABLES
from tests.support.graph import GRAPH_NOW

pytestmark = pytest.mark.integration


async def test_namespace_isolation_survives_forged_context_and_colliding_ids(
    postgres_database, postgres_config
):
    dsn, _ = postgres_database
    other_id, other_role = uuid4(), "state_" + uuid4().hex
    with psycopg.connect(dsn(), autocommit=True) as admin:
        admin.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD 'fixture-only'").format(
                sql.Identifier(other_role)
            )
        )
        provision_postgres_namespace(admin, namespace_id=other_id, role=other_role)
    other_config = replace(
        postgres_config, conninfo=dsn(user=other_role), namespace_id=other_id
    )
    left, right = await PostgresStateStore.open(
        postgres_config
    ), await PostgresStateStore.open(other_config)
    try:
        a = AgentIdentity(id="same-agent", display_name="A", created_at=GRAPH_NOW)
        b = replace(a, display_name="B")
        await left.initialize_identity(a)
        await right.initialize_identity(b)
        assert await left.load_identity() == a
        assert await right.load_identity() == b
        with psycopg.connect(postgres_config.conninfo, autocommit=True) as attacker:
            for namespace in (postgres_config.namespace_id, other_id):
                with attacker.transaction():
                    attacker.execute(
                        "SELECT set_config('daita.store_id', %s, true), set_config('daita.fencing_epoch', '1', true)",
                        (str(namespace),),
                    )
                    rows = attacker.execute(
                        "SELECT data FROM daita_state.metadata"
                    ).fetchall()
                    assert len(rows) == (
                        1 if namespace == postgres_config.namespace_id else 0
                    )
                    if namespace == other_id:
                        assert (
                            attacker.execute(
                                "DELETE FROM daita_state.metadata"
                            ).rowcount
                            == 0
                        )
                        with pytest.raises(psycopg.errors.InsufficientPrivilege):
                            with attacker.transaction():
                                attacker.execute(
                                    "INSERT INTO daita_state.metadata(key, data) VALUES ('attack', '{}')"
                                )
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                attacker.execute(
                    "UPDATE daita_state.namespace_roles SET role_name = session_user"
                )
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                attacker.execute("CREATE TABLE daita_state.attack(id integer)")
        with pytest.raises(ValueError, match="absent or not authorized"):
            await PostgresStateStore.open(
                replace(postgres_config, namespace_id=other_id)
            )
        assert await right.load_identity() == b
    finally:
        await left.close()
        await right.close()


async def test_schema_install_is_idempotent_and_runtime_is_unprivileged(
    postgres_database, postgres_config
):
    dsn, _ = postgres_database
    with psycopg.connect(dsn(), autocommit=True) as admin:
        assert install_postgres_schema(admin) is False
        rows = admin.execute(
            "SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = 'daita_state' AND c.relname = ANY(%s)",
            (list(POSTGRES_TABLES),),
        ).fetchall()
        assert len(rows) == 30
        assert all(enabled and forced for _, enabled, forced in rows)
        assert admin.execute(
            "SELECT count(*) FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace WHERE n.nspname = 'daita_state' AND p.prosecdef"
        ).fetchone() == (0,)
    with pytest.raises(ValueError, match="non-owner"):
        await PostgresStateStore.open(replace(postgres_config, conninfo=dsn()))
    with pytest.raises(ValueError, match="verify-full"):
        await PostgresStateStore.open(
            replace(postgres_config, conninfo=dsn(sslmode="require"))
        )
    with pytest.raises(ValueError, match="dedicated private schema"):
        replace(postgres_config, schema="public")


async def test_schema_drift_and_missing_rls_fail_closed(
    postgres_database, postgres_config
):
    dsn, _ = postgres_database
    schema = "state_" + uuid4().hex
    role = conninfo_to_dict(postgres_config.conninfo)["user"]
    assert isinstance(role, str)
    candidate = replace(postgres_config, schema=schema)
    with psycopg.connect(dsn(), autocommit=True) as admin:
        assert install_postgres_schema(admin, schema=schema)
        provision_postgres_namespace(
            admin, namespace_id=candidate.namespace_id, role=role, schema=schema
        )
        admin.execute(
            sql.SQL("ALTER TABLE {}.messages DISABLE ROW LEVEL SECURITY").format(
                sql.Identifier(schema)
            )
        )
        with pytest.raises(ValueError, match="isolation"):
            await PostgresStateStore.open(candidate)
        admin.execute(
            sql.SQL("ALTER TABLE {}.messages ENABLE ROW LEVEL SECURITY").format(
                sql.Identifier(schema)
            )
        )
        admin.execute(
            sql.SQL("UPDATE {}.schema_version SET checksum = 'changed'").format(
                sql.Identifier(schema)
            )
        )
        with pytest.raises(ValueError, match="history"):
            await PostgresStateStore.open(candidate)
        with pytest.raises(ValueError, match="history"):
            install_postgres_schema(admin, schema=schema)


async def test_runtime_rejects_privilege_drift(postgres_database, postgres_config):
    dsn, _ = postgres_database
    role = conninfo_to_dict(postgres_config.conninfo)["user"]
    assert isinstance(role, str)
    with psycopg.connect(dsn(), autocommit=True) as admin:
        admin.execute(
            sql.SQL("GRANT TRUNCATE ON daita_state.metadata TO {}").format(
                sql.Identifier(role)
            )
        )
        with pytest.raises(ValueError, match="isolation"):
            await PostgresStateStore.open(postgres_config)
        admin.execute(
            sql.SQL("REVOKE TRUNCATE ON daita_state.metadata FROM {}").format(
                sql.Identifier(role)
            )
        )
        admin.execute(
            sql.SQL("GRANT UPDATE ON daita_state.namespace_roles TO {}").format(
                sql.Identifier(role)
            )
        )
        with pytest.raises(ValueError, match="administrative state"):
            await PostgresStateStore.open(postgres_config)


async def test_transaction_settings_and_prepared_statements_do_not_escape_pool(
    postgres_config,
):
    store = await PostgresStateStore.open(replace(postgres_config, max_connections=1))
    try:
        await store.load_identity()
        with store._postgres.pool.connection() as connection:
            assert connection.prepare_threshold is None
            assert connection.execute(
                "SELECT nullif(current_setting('daita.store_id', true), '')"
            ).fetchone() == (None,)
            assert connection.execute(
                "SELECT count(*) FROM pg_prepared_statements"
            ).fetchone() == (0,)
        assert "fixture-only" not in repr(postgres_config)
    finally:
        await store.close()


async def test_schema_owner_can_provision_without_superuser(postgres_database):
    dsn, _ = postgres_database
    schema = "state_" + uuid4().hex
    owner, runtime = "owner_" + uuid4().hex, "runtime_" + uuid4().hex
    namespace = uuid4()
    with psycopg.connect(dsn(), autocommit=True) as admin:
        admin.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD 'fixture-only'").format(
                sql.Identifier(owner)
            )
        )
        admin.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD 'fixture-only'").format(
                sql.Identifier(runtime)
            )
        )
        admin.execute(
            sql.SQL("GRANT CREATE ON DATABASE postgres TO {}").format(
                sql.Identifier(owner)
            )
        )
    with psycopg.connect(dsn(user=owner), autocommit=True) as migration:
        migration.execute(
            sql.SQL("ALTER DEFAULT PRIVILEGES GRANT ALL ON TABLES TO {}").format(
                sql.Identifier(runtime)
            )
        )
        migration.execute(
            sql.SQL("ALTER DEFAULT PRIVILEGES GRANT ALL ON SEQUENCES TO {}").format(
                sql.Identifier(runtime)
            )
        )
        migration.execute(
            sql.SQL("ALTER DEFAULT PRIVILEGES GRANT ALL ON SCHEMAS TO {}").format(
                sql.Identifier(runtime)
            )
        )
        assert install_postgres_schema(migration, schema=schema)
        provision_postgres_namespace(
            migration, namespace_id=namespace, role=runtime, schema=schema
        )
        assert install_postgres_schema(migration, schema=schema) is False
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            migration.execute(
                sql.SQL(
                    "INSERT INTO {}.metadata(namespace_id, key, data) VALUES (%s, 'orphan', '{{}}')"
                ).format(sql.Identifier(schema)),
                (uuid4(),),
            )
    from daita.storage.postgres import PostgresStateConfig

    store = await PostgresStateStore.open(
        PostgresStateConfig(
            conninfo=dsn(user=runtime), namespace_id=namespace, schema=schema
        )
    )
    try:
        with psycopg.connect(dsn(user=runtime), autocommit=True) as restricted:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                restricted.execute(
                    sql.SQL("TRUNCATE {}.metadata").format(sql.Identifier(schema))
                )
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                restricted.execute(
                    sql.SQL("UPDATE {}.schema_version SET checksum = 'forged'").format(
                        sql.Identifier(schema)
                    )
                )
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                restricted.execute(
                    sql.SQL("CREATE TABLE {}.unwanted(id integer)").format(
                        sql.Identifier(schema)
                    )
                )
        identity = AgentIdentity(id="agent", display_name="State", created_at=GRAPH_NOW)
        assert await store.initialize_identity(identity) == identity
        assert await store.load_identity() == identity
    finally:
        await store.close()
