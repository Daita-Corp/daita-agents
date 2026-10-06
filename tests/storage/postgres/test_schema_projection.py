"""A home schema addition reaches PostgreSQL without another table definition."""

from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

from daita.storage.postgres_schema import render_postgres_schema
from daita.storage.sqlite_schema import CURRENT_DATABASE_SQL

pytestmark = pytest.mark.integration


def test_home_schema_addition_propagates_columns_constraints_and_indexes(
    postgres_database,
):
    dsn, _ = postgres_database
    schema = "projection_" + uuid4().hex
    namespace = uuid4()
    candidate = CURRENT_DATABASE_SQL + """
        CREATE TABLE retained_preferences (
            id TEXT NOT NULL PRIMARY KEY,
            value TEXT NOT NULL,
            revision INTEGER NOT NULL DEFAULT 1 CHECK (revision >= 1),
            UNIQUE(value)
        );
        CREATE INDEX retained_preferences_revision ON retained_preferences(revision);
        CREATE TABLE preference_links (
            id TEXT NOT NULL PRIMARY KEY,
            preference_id TEXT NOT NULL REFERENCES retained_preferences(id) ON DELETE CASCADE
        );
    """
    with psycopg.connect(dsn(), autocommit=True) as admin:
        with admin.transaction():
            admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
            admin.execute(
                "SELECT set_config('search_path', %s, true), set_config('daita.store_id', %s, true)",
                (schema, str(namespace)),
            )
            admin.execute("CREATE TABLE namespaces(namespace_id uuid PRIMARY KEY)")
            admin.execute(render_postgres_schema(candidate).encode(), prepare=False)
            admin.execute("INSERT INTO namespaces VALUES (%s)", (namespace,))
            assert admin.execute(
                "INSERT INTO retained_preferences(id, value) VALUES ('a', 'first') RETURNING revision"
            ).fetchone() == (1,)
            admin.execute(
                "INSERT INTO preference_links(id, preference_id) VALUES ('link', 'a')"
            )
            with pytest.raises(psycopg.errors.CheckViolation), admin.transaction():
                admin.execute(
                    "INSERT INTO retained_preferences(id, value, revision) VALUES ('b', 'second', 0)"
                )
            with pytest.raises(psycopg.errors.UniqueViolation), admin.transaction():
                admin.execute(
                    "INSERT INTO retained_preferences(id, value) VALUES ('b', 'first')"
                )
            index = admin.execute(
                "SELECT indexdef FROM pg_indexes WHERE schemaname = %s AND indexname = 'retained_preferences_revision'",
                (schema,),
            ).fetchone()
            assert index is not None and index[0].endswith("(namespace_id, revision)")
            admin.execute("DELETE FROM retained_preferences WHERE id = 'a'")
            assert admin.execute(
                "SELECT count(*) FROM preference_links"
            ).fetchone() == (0,)
            admin.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema))
            )


def test_unsupported_home_schema_features_fail_instead_of_being_dropped():
    with pytest.raises(ValueError, match="unsupported portable state column type"):
        render_postgres_schema(
            "CREATE TABLE unsupported(id TEXT PRIMARY KEY NOT NULL, payload BLOB)"
        )
