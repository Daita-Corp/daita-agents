"""Explicit administrative provisioning, separate from runtime admission.

Callers supply an administrative connection. Opening a state store never creates
schemas, users or grants. These helpers accept only caller-owned databases.
"""

from __future__ import annotations

import re
from hashlib import sha256
from uuid import UUID

from .home_migrations import CURRENT_HOME_REVISION
from .postgres_schema import (
    POSTGRES_SCHEMA_SQL,
    POSTGRES_TABLES,
)


def schema_identifier(schema: str) -> str:
    if (
        not isinstance(schema, str)
        or re.fullmatch(r"[a-z][a-z0-9_]{0,47}", schema) is None
    ):
        raise ValueError(
            "state schema must be a lowercase SQL identifier of at most 48 characters"
        )
    if schema in {"public", "information_schema"} or schema.startswith("pg_"):
        raise ValueError("state requires a dedicated private schema")
    return '"' + schema + '"'


_BOOTSTRAP = """
CREATE TABLE schema_version (version integer PRIMARY KEY, checksum text NOT NULL);
CREATE TABLE namespaces (namespace_id uuid PRIMARY KEY, fencing_epoch bigint NOT NULL DEFAULT 1 CHECK (fencing_epoch > 0));
CREATE TABLE namespace_roles (namespace_id uuid REFERENCES namespaces(namespace_id), role_name name NOT NULL, PRIMARY KEY(namespace_id, role_name));
"""


def _security_statements(name: str, owner: str) -> tuple[str, ...]:
    statements: list[str] = []
    for table in (*POSTGRES_TABLES, "namespace_roles", "namespaces"):
        statements.extend(
            (
                f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY",
                f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY",
                f"CREATE POLICY schema_owner ON {table} TO {owner} USING (true) WITH CHECK (true)",
            )
        )
    statements.append(
        "CREATE POLICY own_grants ON namespace_roles FOR SELECT USING (role_name = session_user)"
    )
    statements.append(
        f"CREATE POLICY namespace_access ON namespaces USING (EXISTS (SELECT 1 FROM {name}.namespace_roles a WHERE a.namespace_id = namespaces.namespace_id AND a.role_name = session_user))"
    )
    for table in POSTGRES_TABLES:
        selected = "(SELECT nullif(current_setting('daita.store_id', true), '')::uuid)"
        scope = f"namespace_id = {selected} AND EXISTS (SELECT 1 FROM {name}.namespace_roles a WHERE a.namespace_id = {selected} AND a.role_name = session_user)"
        fence = f"({scope}) AND EXISTS (SELECT 1 FROM {name}.namespaces n WHERE n.namespace_id = {selected} AND n.fencing_epoch = (SELECT nullif(current_setting('daita.fencing_epoch', true), '')::bigint))"
        statements.extend(
            (
                f"CREATE POLICY read_scope ON {table} FOR SELECT USING ({scope})",
                f"CREATE POLICY insert_scope ON {table} FOR INSERT WITH CHECK ({fence})",
                f"CREATE POLICY update_scope ON {table} FOR UPDATE USING ({fence}) WITH CHECK ({fence})",
                f"CREATE POLICY delete_scope ON {table} FOR DELETE USING ({fence})",
            )
        )
    statements.extend(
        (
            # A database owner's default ACLs may grant ALL to another login.
            # Remove those grants before selectively provisioning runtime roles;
            # RLS alone cannot protect against TRUNCATE or a bypass-RLS login.
            f"""DO $revoke_defaults$
            DECLARE recipient record;
            BEGIN
                FOR recipient IN
                    SELECT DISTINCT a.grantee
                    FROM pg_namespace n, LATERAL aclexplode(n.nspacl) a
                    WHERE n.oid = '{name}'::regnamespace
                      AND a.grantee NOT IN (0, n.nspowner)
                    UNION
                    SELECT DISTINCT a.grantee
                    FROM pg_class c, LATERAL aclexplode(c.relacl) a
                    WHERE c.relnamespace = '{name}'::regnamespace
                      AND a.grantee NOT IN (0, c.relowner)
                LOOP
                    EXECUTE format('REVOKE ALL ON SCHEMA %s FROM %I', '{name}', pg_get_userbyid(recipient.grantee));
                    EXECUTE format('REVOKE ALL ON ALL TABLES IN SCHEMA %s FROM %I', '{name}', pg_get_userbyid(recipient.grantee));
                    EXECUTE format('REVOKE ALL ON ALL SEQUENCES IN SCHEMA %s FROM %I', '{name}', pg_get_userbyid(recipient.grantee));
                END LOOP;
            END $revoke_defaults$""",
            f"REVOKE ALL ON SCHEMA {name} FROM PUBLIC",
            f"REVOKE ALL ON ALL TABLES IN SCHEMA {name} FROM PUBLIC",
            f"REVOKE ALL ON ALL SEQUENCES IN SCHEMA {name} FROM PUBLIC",
        )
    )
    return tuple(statements)


def postgres_schema_checksum() -> str:
    # Bind relational DDL, bootstrap and security semantics; deployment names do
    # not change the version. Released definitions are immutable.
    definition = (
        str(CURRENT_HOME_REVISION)
        + _BOOTSTRAP
        + POSTGRES_SCHEMA_SQL
        + ";\n".join(_security_statements('"schema"', '"owner"'))
    )
    return sha256(definition.encode()).hexdigest()


def install_postgres_schema(connection, *, schema: str = "daita_state") -> bool:
    """Install once, or verify the exact supported version; never edit old DDL."""
    name = schema_identifier(schema)
    checksum = postgres_schema_checksum()
    with connection.transaction():
        # Transaction-scoped only: safe through transaction poolers too.
        connection.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            ("daita-schema:" + schema,),
        )
        exists = connection.execute("SELECT to_regnamespace(%s)", (schema,)).fetchone()[
            0
        ]
        if exists is not None:
            row = connection.execute(
                f"SELECT version, checksum FROM {name}.schema_version"
            ).fetchall()
            if row != [(CURRENT_HOME_REVISION, checksum)]:
                raise ValueError(
                    "PostgreSQL state schema history is unsupported or changed"
                )
            return False
        connection.execute(f"CREATE SCHEMA {name}")
        connection.execute(f"REVOKE ALL ON SCHEMA {name} FROM PUBLIC")
        connection.execute(
            "SELECT set_config('search_path', %s, true)", (f"{name}, pg_catalog",)
        )
        from psycopg import sql

        owner = sql.Identifier(
            connection.execute("SELECT current_user").fetchone()[0]
        ).as_string(connection)
        connection.execute(_BOOTSTRAP, prepare=False)
        connection.execute(POSTGRES_SCHEMA_SQL, prepare=False)
        for statement in _security_statements(name, owner):
            connection.execute(statement, prepare=False)
        connection.execute(
            "INSERT INTO schema_version VALUES (%s, %s)",
            (CURRENT_HOME_REVISION, checksum),
        )
    return True


def provision_postgres_namespace(
    connection, *, namespace_id: UUID, role: str, schema: str = "daita_state"
) -> None:
    """Grant an existing restricted login access to one logical state store."""
    from psycopg import sql

    name = schema_identifier(schema)
    namespace_id = UUID(str(namespace_id))
    row = connection.execute(
        "SELECT rolcanlogin, rolsuper, rolbypassrls, rolcreaterole FROM pg_roles WHERE rolname = %s",
        (role,),
    ).fetchone()
    if row != (True, False, False, False):
        raise ValueError("state runtime requires an existing unprivileged login role")
    role_sql = sql.Identifier(role).as_string(connection)
    with connection.transaction():
        connection.execute(
            f"INSERT INTO {name}.namespaces(namespace_id) VALUES (%s) ON CONFLICT DO NOTHING",
            (namespace_id,),
        )
        connection.execute(
            f"INSERT INTO {name}.namespace_roles VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (namespace_id, role),
        )
        connection.execute(f"GRANT USAGE ON SCHEMA {name} TO {role_sql}")
        connection.execute(
            f"GRANT SELECT ON {name}.schema_version, {name}.namespaces, {name}.namespace_roles TO {role_sql}"
        )
        connection.execute(
            f"GRANT UPDATE(fencing_epoch) ON {name}.namespaces TO {role_sql}"
        )
        for table in POSTGRES_TABLES:
            connection.execute(
                f"GRANT SELECT, INSERT, UPDATE, DELETE ON {name}.{table} TO {role_sql}"
            )
        connection.execute(
            f"GRANT USAGE ON ALL SEQUENCES IN SCHEMA {name} TO {role_sql}"
        )
