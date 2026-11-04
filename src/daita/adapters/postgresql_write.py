"""Preview and transactionally execute one policy-checked PostgreSQL write.

The source first says what the exact text targets. Preview then runs the derived
SELECT in a read-only transaction: that is a claim, not proof. Apply reruns it under
row locks in the write transaction, refuses any change, runs the statement, and rolls
back unless the rows the statement reports are the rows the locked read found.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from decimal import Decimal
from hashlib import sha256
from typing import Any, cast
from uuid import UUID

from .._json import FrozenJsonObject, FrozenJsonValue, canonical_json, thaw_json
from ..capabilities import (
    EffectEvidenceBasis,
    EffectObservation,
    EffectOutcome,
    ToolExecution,
)
from ..catalog.models import ResourceKind, catalog_resource_id
from ..domains.data.capabilities import VerifiedEffect
from ..domains.data.sql import (
    SqlValidationIssue,
    WriteFacts,
    WritePreview,
    denied_relations,
    write_permission_issue,
)
from ..errors import DaitaError
from ..security import SecretProvider, default_secret_provider
from ..storage.sqlite_records import (
    RelationalWriteScope,
    SourcePermissionStateError,
    relational_write_authorization_fingerprint,
)
from .models import SourceRegistration
from .postgresql import (
    PostgreSQLSourceError,
    _close_postgresql_connection,
    _configuration_schemas,
    _connect,
    _rollback_postgresql_transaction,
)
from .postgresql_query import (
    PostgreSQLQueryError,
    _classified_failure,
    check_functions,
    configure_transaction,
    refuse_unqualified,
)
from .postgresql_write_sql import (
    COMPUTED_AT_WRITE,
    PostgreSQLStatement,
    derived_select,
    describe_write,
    verified_statement,
)
from .protocols import SourcePermissionStore

_MAX_SAMPLE_ROWS = 5
_MAX_SAMPLE_BYTES = 64 * 1_024
_MAX_LISTED_NAMES = 16
_DESCRIBE = "/* daita:postgresql.write_describe */ EXPLAIN (FORMAT JSON, VERBOSE)\n"

_READINESS_ROLE_KEYS = (
    "superuser",
    "bypass_rls",
    "create_database",
    "create_role",
    "replication",
)
_READINESS_PRIVILEGE_KEYS = (
    "database_connect",
    "schema_usage",
    "table_select",
    "requested_columns_insert",
    "requested_columns_update",
    "table_delete",
)
_READINESS_RELATION_KEYS = (
    "exists",
    "base_table",
    "partition",
    "inheritance",
    "row_level_security",
    "force_row_level_security",
    "user_triggers",
    "rewrite_rules",
)

_LIVE_TABLE_SQL = """
/* daita:postgresql.write_table */
SELECT
    relation.oid::pg_catalog.text AS relation_oid,
    relation.relkind::pg_catalog.text AS relation_kind,
    relation.relispartition AS is_partition,
    relation.relrowsecurity AS row_level_security,
    relation.relforcerowsecurity AS force_row_level_security,
    EXISTS (
        SELECT 1
        FROM pg_catalog.pg_inherits AS inheritance
        WHERE inheritance.inhrelid = relation.oid
           OR inheritance.inhparent = relation.oid
    ) AS has_inheritance,
    EXISTS (
        SELECT 1
        FROM pg_catalog.pg_rewrite AS rewrite
        WHERE rewrite.ev_class = relation.oid
          AND rewrite.rulename <> '_RETURN'
          AND rewrite.ev_enabled <> 'D'
    ) AS has_rewrite_rules,
    role.rolsuper AS role_superuser,
    role.rolbypassrls AS role_bypass_rls,
    role.rolcreatedb AS role_create_database,
    role.rolcreaterole AS role_create_role,
    role.rolreplication AS role_replication,
    pg_catalog.has_database_privilege(
        current_user, pg_catalog.current_database(), 'CONNECT'
    ) AS can_connect,
    pg_catalog.has_schema_privilege(current_user, namespace.oid, 'USAGE') AS can_use_schema,
    pg_catalog.has_table_privilege(current_user, relation.oid, 'SELECT') AS can_select_table,
    pg_catalog.has_table_privilege(current_user, relation.oid, 'DELETE') AS can_delete_rows,
    NOT EXISTS (
        SELECT 1
        FROM pg_catalog.unnest($3::pg_catalog.text[]) AS requested(column_name)
        WHERE NOT pg_catalog.has_column_privilege(
            current_user, relation.oid, requested.column_name, 'UPDATE'
        )
    ) AS can_update_columns,
    NOT EXISTS (
        SELECT 1
        FROM pg_catalog.unnest($4::pg_catalog.text[]) AS requested(column_name)
        WHERE NOT pg_catalog.has_column_privilege(
            current_user, relation.oid, requested.column_name, 'INSERT'
        )
    ) AS can_insert_columns,
    (
        SELECT pg_catalog.json_agg(
            pg_catalog.json_build_array(
                attribute.attname::pg_catalog.text,
                pg_catalog.format_type(attribute.atttypid, attribute.atttypmod),
                attribute.attnotnull,
                attribute.atthasdef,
                attribute.attidentity::pg_catalog.text,
                attribute.attgenerated::pg_catalog.text
            )
            ORDER BY attribute.attnum
        )::pg_catalog.text
        FROM pg_catalog.pg_attribute AS attribute
        WHERE attribute.attrelid = relation.oid
          AND attribute.attnum > 0
          AND NOT attribute.attisdropped
    ) AS columns,
    (
        SELECT pg_catalog.json_agg(
            trigger.tgname::pg_catalog.text ORDER BY trigger.tgname
        )::pg_catalog.text
        FROM pg_catalog.pg_trigger AS trigger
        WHERE trigger.tgrelid = relation.oid
          AND NOT trigger.tgisinternal
          AND trigger.tgenabled <> 'D'
    ) AS triggers,
    (
        SELECT pg_catalog.json_agg(
            (
                referencing_schema.nspname::pg_catalog.text || '.'
                || referencing.relname::pg_catalog.text
            ) ORDER BY referencing_schema.nspname, referencing.relname
        )::pg_catalog.text
        FROM pg_catalog.pg_constraint AS foreign_key
        JOIN pg_catalog.pg_class AS referencing
          ON referencing.oid = foreign_key.conrelid
        JOIN pg_catalog.pg_namespace AS referencing_schema
          ON referencing_schema.oid = referencing.relnamespace
        WHERE foreign_key.confrelid = relation.oid
          AND foreign_key.contype = 'f'
          AND (
              foreign_key.confdeltype IN ('c', 'n', 'd')
              OR foreign_key.confupdtype IN ('c', 'n', 'd')
          )
    ) AS cascades,
    (
        SELECT pg_catalog.json_agg(
            candidate.columns ORDER BY candidate.indisprimary DESC, candidate.indexrelid
        )::pg_catalog.text
        FROM (
            SELECT
                entry.indisprimary,
                entry.indexrelid,
                (
                    SELECT pg_catalog.json_agg(
                        attribute.attname::pg_catalog.text ORDER BY part.position
                    )
                    FROM pg_catalog.unnest(entry.indkey)
                        WITH ORDINALITY AS part(attnum, position)
                    JOIN pg_catalog.pg_attribute AS attribute
                      ON attribute.attrelid = entry.indrelid
                     AND attribute.attnum = part.attnum
                    WHERE part.position <= entry.indnkeyatts
                ) AS columns
            FROM pg_catalog.pg_index AS entry
            WHERE entry.indrelid = relation.oid
              AND entry.indisunique
              AND entry.indisvalid
              AND entry.indpred IS NULL
              AND entry.indexprs IS NULL
              AND NOT EXISTS (
                  SELECT 1
                  FROM pg_catalog.unnest(entry.indkey)
                      WITH ORDINALITY AS part(attnum, position)
                  JOIN pg_catalog.pg_attribute AS attribute
                    ON attribute.attrelid = entry.indrelid
                   AND attribute.attnum = part.attnum
                  WHERE part.position <= entry.indnkeyatts
                    AND NOT attribute.attnotnull
              )
        ) AS candidate
    ) AS keys
FROM pg_catalog.pg_class AS relation
JOIN pg_catalog.pg_namespace AS namespace
  ON namespace.oid = relation.relnamespace
JOIN pg_catalog.pg_roles AS role
  ON role.rolname = current_user
WHERE namespace.nspname = $1
  AND relation.relname = $2
LIMIT 1
"""
_GUARDRAIL_NAMES = (
    "relation_kind",
    "is_partition",
    "row_level_security",
    "force_row_level_security",
    "has_inheritance",
    "has_rewrite_rules",
    "role_superuser",
    "role_bypass_rls",
    "role_create_database",
    "role_create_role",
    "role_replication",
    "can_connect",
    "can_use_schema",
    "can_select_table",
    "can_delete_rows",
    "can_update_columns",
    "can_insert_columns",
)


class RelationalWriteError(DaitaError):
    """Stable write failure; driver and server text cross only by category."""

    def __init__(
        self,
        code: str,
        message: str,
        details: Mapping[str, object] | None = None,
        *,
        effect_observation: EffectObservation | None = None,
    ) -> None:
        self.effect_observation = effect_observation
        self.details = FrozenJsonObject.from_mapping(details or {})
        super().__init__(message, error_code=code)


class RelationalWriteCancelled(asyncio.CancelledError):
    def __init__(self, observation: EffectObservation) -> None:
        self.effect_observation = observation
        super().__init__("native write cancelled after bounded transaction cleanup")


@dataclass(frozen=True, slots=True)
class RelationalWriteReadiness:
    """Bounded, secret-free readiness facts for one table's stored write scope."""

    source_id: str
    table: str
    allowed_operations: tuple[str, ...]
    daita_scope_ready: bool
    ready_for_preview: bool
    proves_execution: bool
    role_attributes: FrozenJsonObject
    privileges: FrozenJsonObject
    relation: FrozenJsonObject
    rejection_codes: tuple[str, ...]
    remediation_categories: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.proves_execution:
            raise ValueError("readiness can never prove a future execution")
        if self.ready_for_preview != (not self.rejection_codes):
            raise ValueError("readiness status must agree with rejection codes")

    def to_mapping(self) -> dict[str, object]:
        """Return the one safe representation shared by API, CLI, and TUI."""

        return {
            "source_id": self.source_id,
            "table": self.table,
            "allowed_operations": self.allowed_operations,
            "daita_scope_ready": self.daita_scope_ready,
            "ready_for_preview": self.ready_for_preview,
            "proves_execution": self.proves_execution,
            "role_attributes": self.role_attributes.to_dict(),
            "privileges": self.privileges.to_dict(),
            "relation": self.relation.to_dict(),
            "rejection_codes": self.rejection_codes,
            "remediation_categories": self.remediation_categories,
        }


@dataclass(frozen=True, slots=True)
class _LiveTable:
    """What one single-table read says: its columns, guardrails and warnings."""

    facts: Mapping[str, object]
    columns: tuple[tuple[object, ...], ...]
    triggers: tuple[str, ...]
    cascades: tuple[str, ...]
    # Each unique, whole, not-null index's columns, primary key first.
    keys: tuple[tuple[str, ...], ...]

    @property
    def column_types(self) -> dict[str, str]:
        return {str(column[0]): str(column[1]) for column in self.columns}

    @property
    def structure_sha256(self) -> str:
        return _sha256_json(
            {
                "facts": dict(self.facts),
                "columns": self.columns,
                "triggers": self.triggers,
                "cascades": self.cascades,
                "keys": self.keys,
            }
        )


@dataclass(frozen=True, slots=True)
class _Scan:
    counts: Mapping[str, int]
    target_set_sha256: str
    # What the statement must report: each touched row's identity and what happened.
    effect_sha256: str
    samples: tuple[Mapping[str, object], ...]
    samples_omitted: bool
    matched_twice: bool

    @property
    def affected_rows(self) -> int:
        return sum(count for action, count in self.counts.items() if action != "skip")


class PostgreSQLWriteBackend:
    """Own the one write flow: readiness, preview and the bounded transaction."""

    def __init__(
        self,
        sources: SourcePermissionStore,
        secret_provider: SecretProvider | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
        statement_timeout_seconds: float = 5.0,
        lock_timeout_seconds: float = 1.0,
        cleanup_timeout_seconds: float = 1.0,
    ) -> None:
        provider = default_secret_provider(secret_provider)
        if not isinstance(provider, SecretProvider):
            raise TypeError("secret_provider must implement SecretProvider")
        for value, name, maximum in (
            (statement_timeout_seconds, "statement_timeout_seconds", 60),
            (lock_timeout_seconds, "lock_timeout_seconds", 10),
            (cleanup_timeout_seconds, "cleanup_timeout_seconds", 10),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0 < float(value) <= maximum
            ):
                raise ValueError(f"{name} must be positive and at most {maximum}")
        self._sources = sources
        self._secret_provider = provider
        self._clock = clock or (lambda: datetime.now(UTC))
        self._statement_timeout_seconds = float(statement_timeout_seconds)
        self._lock_timeout_seconds = float(lock_timeout_seconds)
        self._cleanup_timeout_seconds = float(cleanup_timeout_seconds)

    def describe(
        self,
        sql: str,
        *,
        source_id: str,
        parameters: Sequence[object] = (),
        allowed_resource_ids: frozenset[str] | None = None,
        schemas: tuple[str, ...] = (),
    ) -> WriteFacts | tuple[SqlValidationIssue, ...]:
        """Parse for early errors; the source confirms the target in each transaction."""

        return describe_write(
            sql,
            source_id=source_id,
            parameters=parameters,
            allowed_resource_ids=allowed_resource_ids,
            schemas=schemas,
        )

    async def _registration(self, agent_id: str, source_id: str) -> SourceRegistration:
        registration = await self._sources.load_source(agent_id, source_id)
        if (
            registration is None
            or registration.agent_id != agent_id
            or registration.id != source_id
            or not registration.active
            or registration.adapter_id != "postgresql"
        ):
            raise RelationalWriteError(
                "write_source_not_available",
                "The selected source is not an active PostgreSQL source owned by "
                "this agent.",
            )
        return registration

    async def _current_scope(
        self, registration: SourceRegistration, facts: WriteFacts
    ) -> RelationalWriteScope:
        """Recheck stored permission: it can change while a write waits."""

        try:
            scopes = await self._sources.list_relational_write_scopes(
                registration.agent_id, registration.id
            )
            read_scope = await self._sources.load_source_read_scope(
                registration.agent_id, registration.id
            )
        except SourcePermissionStateError:
            raise RelationalWriteError(
                "source_permission_state_invalid",
                "Stored source permission state is missing or invalid.",
            ) from None
        scope = next(
            (item for item in scopes if item.resource_id == facts.resource_id), None
        )
        denied = denied_relations(facts, read_scope)  # type: ignore[arg-type]
        if scope is None or denied is None or denied:
            raise RelationalWriteError(
                "resource_write_not_allowed",
                f"{facts.table} no longer has the permission this write needs.",
            )
        if (
            scope.authorization_fingerprint
            != relational_write_authorization_fingerprint(
                source=registration, scope=scope
            )
        ):
            raise RelationalWriteError(
                "source_permission_state_invalid",
                "Stored source permission state is missing or invalid.",
            )
        if write_permission_issue(scope, facts) is not None:
            raise RelationalWriteError(
                "resource_write_not_allowed",
                f"{facts.table} no longer has the permission this write needs.",
            )
        return scope

    async def _prepare_transaction(
        self,
        connection: Any,
        registration: SourceRegistration,
        facts: WriteFacts,
    ) -> tuple[_LiveTable, bool]:
        """Its live table, and whether the statement uses time or randomness."""

        statement = PostgreSQLStatement.of(facts)
        await configure_transaction(
            connection,
            schemas=_configuration_schemas(registration.configuration),
            statement_timeout_seconds=self._statement_timeout_seconds,
            lock_timeout_seconds=self._lock_timeout_seconds,
        )
        if statement.unqualified_relations:
            await refuse_unqualified(
                connection,
                registration,
                statement.unqualified_relations,
                facts.allowed_resource_ids,
            )
        varies = statement.uses_time
        if (
            statement.function_lookups
            or statement.written_calls
            or statement.attribute_lookups
        ):
            varies = (
                await check_functions(
                    connection,
                    statement.function_lookups,
                    statement.attribute_lookups,
                    statement.written_calls,
                )
                or varies
            )
        live = await self._live_table(
            connection, facts.table, facts.update_columns, facts.insert_columns
        )
        return live, varies

    async def _confirm_target(
        self, connection: Any, facts: WriteFacts, text: str
    ) -> None:
        """Ask the source what the exact text targets; the parser only guessed.

        Planning a write takes ROW EXCLUSIVE on its table even in a read-only
        transaction, so the savepoint releases that before anything else runs.
        """

        savepoint = connection.transaction()
        await savepoint.start()
        try:
            statement = await connection.prepare(
                _DESCRIBE + text, timeout=self._statement_timeout_seconds
            )
            plan = await statement.fetchval(
                *_bound_parameters(statement, PostgreSQLStatement.of(facts).parameters),
                timeout=self._statement_timeout_seconds,
            )
        finally:
            await savepoint.rollback()
        planned = _planned_writes(plan)
        if planned != {(facts.operation, facts.table)}:
            raise RelationalWriteError(
                (
                    "multiple_write_targets"
                    if len({table for _, table in planned}) > 1
                    else "write_target_invalid"
                ),
                f"PostgreSQL does not read this as one {facts.operation} of "
                f"{facts.table}.",
            )

    async def _live_table(
        self,
        connection: Any,
        table: str,
        update_columns: tuple[str, ...],
        insert_columns: tuple[str, ...],
    ) -> _LiveTable:
        schema, _, name = table.partition(".")
        raw = await connection.fetchrow(
            _LIVE_TABLE_SQL,
            schema,
            name,
            list(update_columns),
            list(insert_columns),
            timeout=self._statement_timeout_seconds,
        )
        if raw is None:
            raise RelationalWriteError(
                "write_resource_not_writable", f"{table} does not exist."
            )
        facts = {name: _record_value(raw, name) for name in _GUARDRAIL_NAMES}
        facts["relation_oid"] = _record_value(raw, "relation_oid")
        return _LiveTable(
            facts=facts,
            columns=tuple(tuple(item) for item in json.loads(raw["columns"] or "[]")),
            triggers=tuple(json.loads(raw["triggers"] or "[]")),
            cascades=tuple(dict.fromkeys(json.loads(raw["cascades"] or "[]"))),
            keys=tuple(tuple(item) for item in json.loads(raw["keys"] or "[]")),
        )

    async def inspect_write_table(
        self, *, agent_id: str, source_id: str, table: str
    ) -> Mapping[str, tuple[str, ...]]:
        """Read one table live for enabling writes: what can be written, by name."""

        registration = await self._registration(agent_id, source_id)
        connection = transaction = None
        finished = False
        try:
            connection = await _connect(registration, self._secret_provider)
            transaction = connection.transaction(
                isolation="repeatable_read", readonly=True
            )
            await transaction.start()
            await configure_transaction(
                connection,
                schemas=_configuration_schemas(registration.configuration),
                statement_timeout_seconds=self._statement_timeout_seconds,
                lock_timeout_seconds=self._lock_timeout_seconds,
            )
            live = await self._live_table(connection, table, (), ())
            await transaction.commit()
            finished = True
        except (asyncio.CancelledError, ImportError, RelationalWriteError):
            raise
        except PostgreSQLSourceError:
            raise RelationalWriteError(
                "write_source_not_available",
                "PostgreSQL could not open the selected source.",
            ) from None
        except Exception:
            raise RelationalWriteError(
                "write_inspection_failed", f"PostgreSQL could not inspect {table}."
            ) from None
        finally:
            await self._finish(connection, transaction, finished)
        if live.facts["relation_kind"] != "r":
            raise RelationalWriteError(
                "write_relation_not_base_table",
                f"{table} is not a plain table, so it cannot be written.",
            )
        writable = tuple(
            str(column[0])
            for column in live.columns
            if column[4] != "a" and column[5] != "s"
        )
        return {"columns": writable}

    async def _finish(self, connection: Any, transaction: Any, finished: bool) -> None:
        try:
            if transaction is not None and not finished:
                await _rollback_postgresql_transaction(
                    transaction,
                    connection,
                    timeout_seconds=self._cleanup_timeout_seconds,
                )
        finally:
            if connection is not None:
                await _close_postgresql_connection(
                    connection, timeout_seconds=self._cleanup_timeout_seconds
                )

    async def relational_write_readiness(
        self, *, agent_id: str, source_id: str, table: str
    ) -> RelationalWriteReadiness:
        """Inspect one table's stored scope without granting or mutating anything."""

        registration = await self._sources.load_source(agent_id, source_id)
        if (
            registration is None
            or registration.agent_id != agent_id
            or not registration.active
            or registration.adapter_id != "postgresql"
        ):
            return _readiness(
                source_id,
                table,
                (),
                rejections=("write_source_not_available",),
                remediation=("attach_active_postgresql_source",),
            )
        resource_id = catalog_resource_id(source_id, ResourceKind.TABLE, table)
        try:
            scope = next(
                (
                    item
                    for item in await self._sources.list_relational_write_scopes(
                        agent_id, source_id
                    )
                    if item.resource_id == resource_id
                ),
                None,
            )
        except SourcePermissionStateError:
            return _readiness(
                source_id,
                table,
                (),
                rejections=("source_permission_state_invalid",),
                remediation=("configure_source_permissions",),
            )
        if scope is None:
            return _readiness(
                source_id,
                table,
                (),
                rejections=("resource_write_not_allowed",),
                remediation=("configure_source_permissions",),
            )
        connection = transaction = None
        finished = False
        facts: Mapping[str, object] | None = None
        rejections: tuple[str, ...] = ()
        remediation: tuple[str, ...] = ()
        try:
            connection = await _connect(registration, self._secret_provider)
            transaction = connection.transaction(
                isolation="repeatable_read", readonly=True
            )
            await transaction.start()
            await configure_transaction(
                connection,
                schemas=_configuration_schemas(registration.configuration),
                statement_timeout_seconds=self._statement_timeout_seconds,
                lock_timeout_seconds=self._lock_timeout_seconds,
            )
            live = await self._live_table(
                connection,
                table,
                scope.allowed_update_columns,
                scope.allowed_insert_columns,
            )
            facts = {**live.facts, "has_user_triggers": bool(live.triggers)}
            rejections, remediation = _guardrail_rejections(
                facts, scope.allowed_operations
            )
            await transaction.commit()
            finished = True
        except (asyncio.CancelledError, ImportError):
            raise
        except RelationalWriteError as error:
            rejections = (error.error_code,)
            remediation = ("select_supported_base_table",)
        except Exception:
            rejections = ("write_readiness_unavailable",)
            remediation = ("check_connection_and_credentials",)
        finally:
            await self._finish(connection, transaction, finished)
        return _readiness(
            source_id,
            table,
            scope.allowed_operations,
            scope_ready=True,
            facts=facts,
            rejections=rejections,
            remediation=remediation,
        )

    async def preview(
        self, *, agent_id: str, source_id: str, facts: WriteFacts
    ) -> WritePreview:
        """Run the derived SELECT read-only: no trigger fires, no row lock is taken."""

        statement = PostgreSQLStatement.of(facts)
        registration = await self._registration(agent_id, source_id)
        scope = await self._current_scope(registration, facts)
        connection = transaction = None
        finished = False
        failure: RelationalWriteError | None = None
        result: WritePreview | None = None
        try:
            connection = await _connect(registration, self._secret_provider)
            transaction = connection.transaction(
                isolation="repeatable_read", readonly=True
            )
            await transaction.start()
            live, varies = await self._prepare_transaction(
                connection, registration, facts
            )
            _admit_guardrails(live, facts)
            text, key = verified_statement(statement.canonical_sql, live.keys)
            await self._confirm_target(connection, facts, text)
            scan_sql, _lock_sql = derived_select(
                statement.canonical_sql, live.column_types, key
            )
            scan = await self._scan(connection, facts, scan_sql, scope.max_rows, varies)
            result = _preview(
                agent_id, source_id, facts, scope, live, scan, key, varies
            )
            await transaction.commit()
            finished = True
        except (asyncio.CancelledError, ImportError, RelationalWriteError):
            raise
        except PostgreSQLQueryError as error:
            failure = RelationalWriteError(error.code, str(error), error.details)
        except PostgreSQLSourceError:
            failure = RelationalWriteError(
                "write_source_not_available",
                "PostgreSQL could not open the selected source.",
            )
        except Exception as error:
            failure = _preview_failure(error, facts, self._statement_timeout_seconds)
        finally:
            await self._finish(connection, transaction, finished)
        if failure is not None:
            raise failure
        assert result is not None
        return result

    async def _scan(
        self,
        connection: Any,
        facts: WriteFacts,
        scan_sql: str,
        max_rows: int,
        varies: bool,
    ) -> _Scan:
        """Stream the derived rows: count each target once and digest what it found.

        Row values join the digest only when another evaluation would repeat them.
        """

        statement = await connection.prepare(
            scan_sql, timeout=self._statement_timeout_seconds
        )
        arguments = _bound_parameters(
            statement, PostgreSQLStatement.of(facts).parameters
        )
        counts: dict[str, int] = {}
        # A target can match several joined rows: (version, action, values, values sum).
        targets: dict[str, tuple[object, str, object, int]] = {}
        samples: list[Mapping[str, object]] = []
        digest = effect = 0
        matched_twice = False
        async for record in statement.cursor(
            *arguments, prefetch=256, timeout=self._statement_timeout_seconds
        ):
            target = _record_value(record, "__daita_row")
            bound = None if varies else _record_value(record, "__daita_bound")
            if target is not None and str(target) in targets:
                version, action, first, values = targets[str(target)]
                matched_twice = matched_twice or bound != first
                targets[str(target)] = (
                    version,
                    action,
                    first,
                    values + _row_digest(bound),
                )
                continue
            action = str(_record_value(record, "__daita_action"))
            version = _record_value(record, "__daita_version")
            counts[action] = counts.get(action, 0) + 1
            if sum(n for a, n in counts.items() if a != "skip") > max_rows:
                raise RelationalWriteError(
                    "write_row_limit",
                    f"This write touches more than the {max_rows} rows allowed on "
                    f"{facts.table}. Narrow it, or ask a person to raise the limit.",
                    {"max_rows": max_rows},
                )
            if action != "skip":
                effect += _row_digest(_record_value(record, "__daita_key"), action)
                if len(samples) < _MAX_SAMPLE_ROWS:
                    samples.append(_sample(record, facts, action, varies))
            if target is None:
                digest += _row_digest(None, version, action, bound)
            else:
                targets[str(target)] = (version, action, bound, _row_digest(bound))
        for target, (version, action, _first, values) in targets.items():
            digest += _row_digest(target, version, action, _sum_text(values))
        size = len(canonical_json(samples).encode("utf-8"))
        return _Scan(
            counts=counts,
            target_set_sha256=_sha256_json([counts, _sum_text(digest)]),
            effect_sha256=_sum_text(effect),
            samples=() if size > _MAX_SAMPLE_BYTES else tuple(samples),
            samples_omitted=size > _MAX_SAMPLE_BYTES,
            matched_twice=matched_twice,
        )

    async def _touched_rows(
        self,
        statement: Any,
        arguments: tuple[object, ...],
        facts: WriteFacts,
        limit: int,
    ) -> tuple[int, str]:
        """Run the statement; what the server says it touched, read up to the limit."""

        count = effect = 0
        async for record in statement.cursor(
            *arguments, prefetch=256, timeout=self._statement_timeout_seconds
        ):
            count += 1
            if count > limit:
                break
            action = facts.operation
            if (
                action == "insert"
                and _record_value(record, "__daita_inserted") is False
            ):
                action = "update"
            effect += _row_digest(_record_value(record, "__daita_key"), action)
        return count, _sum_text(effect)

    async def apply(
        self,
        *,
        agent_id: str,
        source_id: str,
        execution: ToolExecution,
        facts: WriteFacts,
        preview_fingerprint: str,
        expected_affected_rows: int,
    ) -> VerifiedEffect:
        """Execute one receipt-backed write and classify commit certainty."""

        written = PostgreSQLStatement.of(facts)
        if execution.effect_receipt_id is None:
            raise RelationalWriteError(
                "write_receipt_unavailable",
                "Runtime receipt reservation is required before native execution.",
            )
        try:
            registration = await self._registration(agent_id, source_id)
            scope = await self._current_scope(registration, facts)
        except asyncio.CancelledError as error:
            raise RelationalWriteCancelled(_not_dispatched()) from error
        except RelationalWriteError as error:
            raise RelationalWriteError(
                error.error_code, str(error), effect_observation=_not_dispatched()
            ) from None
        except Exception:
            raise RelationalWriteError(
                "write_not_dispatched",
                "Native write admission failed before write dispatch.",
                effect_observation=_not_dispatched(),
            ) from None

        connection = None
        transaction = None
        transaction_finished = False
        commit_attempted = False
        mutation_attempted = False
        target_set_sha256: str | None = None
        affected_rows: int | None = None
        counts: Mapping[str, int] = {}
        terminal_outcome = EffectOutcome.NOT_APPLIED
        terminal_code: str | None = "write_not_committed"
        cancelled: asyncio.CancelledError | None = None
        try:
            connection = await _connect(registration, self._secret_provider)
            # One snapshot: the statement sees exactly what the scan below saw.
            transaction = connection.transaction(isolation="repeatable_read")
            await transaction.start()
            live, varies = await self._prepare_transaction(
                connection, registration, facts
            )
            _admit_guardrails(live, facts)
            text, key = verified_statement(written.canonical_sql, live.keys)
            await self._confirm_target(connection, facts, text)
            scan_sql, lock_sql = derived_select(
                written.canonical_sql, live.column_types, key
            )
            if lock_sql is not None:
                statement = await connection.prepare(
                    lock_sql, timeout=self._statement_timeout_seconds
                )
                await connection.execute(
                    lock_sql,
                    *_bound_parameters(statement, written.parameters),
                    timeout=self._statement_timeout_seconds,
                )
            scan = await self._scan(connection, facts, scan_sql, scope.max_rows, varies)
            locked = _preview(
                agent_id, source_id, facts, scope, live, scan, key, varies
            )
            target_set_sha256 = locked.target_set_sha256
            counts = scan.counts
            if (
                scan.affected_rows != expected_affected_rows
                or locked.preview_fingerprint != preview_fingerprint
            ):
                raise RelationalWriteError(
                    "write_state_changed",
                    "The target rows or the table changed after approval.",
                )
            if await self._current_scope(registration, facts) != scope:
                raise RelationalWriteError(
                    "write_state_changed", "Write permission changed before mutation."
                )
            statement = await connection.prepare(
                text, timeout=self._statement_timeout_seconds
            )
            arguments = _bound_parameters(statement, written.parameters)
            mutation_attempted = True
            affected_rows, effect_sha256 = await self._touched_rows(
                statement, arguments, facts, expected_affected_rows
            )
            if (
                affected_rows != expected_affected_rows
                or effect_sha256 != scan.effect_sha256
            ):
                raise RelationalWriteError(
                    "write_affected_rows_mismatch",
                    "PostgreSQL touched different rows than the approved preview, so "
                    "the write was rolled back.",
                )
            commit_attempted = True
            await transaction.commit()
            transaction_finished = True
            terminal_outcome = EffectOutcome.SUCCEEDED
            terminal_code = None
        except asyncio.CancelledError as error:
            cancelled = error
            if commit_attempted:
                terminal_outcome = EffectOutcome.UNCERTAIN
                terminal_code = "write_outcome_unknown"
            else:
                terminal_outcome = EffectOutcome.NOT_APPLIED
                terminal_code = "write_not_committed"
        except (RelationalWriteError, PostgreSQLQueryError) as error:
            terminal_outcome = (
                EffectOutcome.UNCERTAIN
                if commit_attempted
                else EffectOutcome.NOT_APPLIED
            )
            terminal_code = (
                "write_outcome_unknown"
                if commit_attempted
                else (
                    error.error_code
                    if isinstance(error, RelationalWriteError)
                    else error.code
                )
            )
        except Exception as error:
            commit_rejected = commit_attempted and isinstance(
                getattr(error, "sqlstate", None), str
            )
            terminal_outcome = (
                EffectOutcome.UNCERTAIN
                if commit_attempted and not commit_rejected
                else EffectOutcome.NOT_APPLIED
            )
            terminal_code = (
                "write_outcome_unknown"
                if terminal_outcome is EffectOutcome.UNCERTAIN
                else _normalized_write_failure(error)
            )
        finally:
            try:
                if transaction is not None and not transaction_finished:
                    try:
                        rolled_back = await _rollback_postgresql_transaction(
                            transaction,
                            connection,
                            timeout_seconds=self._cleanup_timeout_seconds,
                        )
                    except asyncio.CancelledError as error:
                        cancelled = cancelled or error
                        rolled_back = False
                    if not rolled_back and mutation_attempted:
                        terminal_outcome, terminal_code = (
                            EffectOutcome.UNCERTAIN,
                            "write_outcome_unknown",
                        )
            finally:
                if connection is not None:
                    try:
                        await _close_postgresql_connection(
                            connection, timeout_seconds=self._cleanup_timeout_seconds
                        )
                    except asyncio.CancelledError as error:
                        cancelled = cancelled or error

        completed_at = self._clock()
        observation = EffectObservation(
            terminal_outcome,
            EffectEvidenceBasis.ADAPTER_VERIFIED,
            FrozenJsonObject.from_mapping(
                {
                    "source_id": source_id,
                    "resource_id": facts.resource_id,
                    "table": facts.table,
                    "operation": facts.operation,
                    "intent_sha256": _intent_sha256(facts),
                    "preview_fingerprint": preview_fingerprint,
                    "target_set_sha256": target_set_sha256,
                    "expected_affected_rows": expected_affected_rows,
                    "affected_rows": (
                        affected_rows
                        if terminal_outcome is EffectOutcome.SUCCEEDED
                        else (
                            0 if terminal_outcome is EffectOutcome.NOT_APPLIED else None
                        )
                    ),
                    "normalized_error_code": terminal_code,
                }
            ),
        )
        if cancelled is not None:
            raise RelationalWriteCancelled(observation) from cancelled
        if terminal_outcome is not EffectOutcome.SUCCEEDED:
            assert terminal_code is not None
            raise RelationalWriteError(
                terminal_code,
                (
                    "PostgreSQL commit certainty was lost; do not retry automatically."
                    if terminal_outcome is EffectOutcome.UNCERTAIN
                    else "PostgreSQL did not commit the approved write."
                ),
                {
                    "receipt_id": execution.effect_receipt_id,
                    "outcome": terminal_outcome.value,
                    "affected_rows": (
                        0 if terminal_outcome is EffectOutcome.NOT_APPLIED else None
                    ),
                },
                effect_observation=observation,
            )
        assert affected_rows is not None
        assert target_set_sha256 is not None
        return VerifiedEffect(
            receipt_id=execution.effect_receipt_id,
            source_id=source_id,
            table=facts.table,
            operation=facts.operation,
            canonical_sql=written.canonical_sql,
            preview_fingerprint=preview_fingerprint,
            intent_sha256=_intent_sha256(facts),
            target_set_sha256=target_set_sha256,
            affected_rows=affected_rows,
            inserted_rows=counts.get("insert", 0),
            updated_rows=counts.get("update", 0),
            deleted_rows=counts.get("delete", 0),
            skipped_rows=counts.get("skip", 0),
            committed_at=completed_at.isoformat(),
            effect_observation=observation,
        )


def _not_dispatched() -> EffectObservation:
    return EffectObservation(
        EffectOutcome.NOT_APPLIED, EffectEvidenceBasis.LOCAL_NOT_DISPATCHED
    )


def _planned_writes(plan: object) -> set[tuple[str, str]]:
    """Every (operation, schema.table) the source's plan would write."""

    planned: set[tuple[str, str]] = set()
    stack = [item["Plan"] for item in json.loads(cast(str, plan))]
    while stack:
        node = stack.pop()
        stack.extend(node.get("Plans", ()))
        if node.get("Node Type") == "ModifyTable":
            operation = str(node["Operation"]).lower()
            planned.update(
                (operation, f"{target['Schema']}.{target['Relation Name']}")
                for target in (node, *node.get("Target Tables", ()))
            )
    return planned


def _bound_parameters(statement: Any, values: tuple[object, ...]) -> tuple[object, ...]:
    """Convert JSON values to what the server says each parameter is."""

    types = statement.get_parameters()
    try:
        return tuple(
            _bound_value(thaw_json(cast(FrozenJsonValue, value)), type_.name)
            for value, type_ in zip(values, types, strict=True)
        )
    except (TypeError, ValueError, ArithmeticError):
        raise RelationalWriteError(
            "write_preview_failed",
            f"The parameters do not fit the statement, which takes {len(types)}: "
            "check their number and types.",
        ) from None


def _bound_value(value: object, type_name: str) -> object:
    if value is None or isinstance(value, (list, tuple)):
        return value
    if type_name in {"int2", "int4", "int8"}:
        return int(str(value))
    if type_name == "numeric":
        return Decimal(str(value))
    if type_name in {"float4", "float8"}:
        return float(str(value))
    if type_name == "bool":
        if not isinstance(value, bool):
            raise ValueError("a boolean parameter requires true or false")
        return value
    if type_name == "uuid":
        return UUID(str(value))
    if type_name == "date":
        return date.fromisoformat(str(value))
    if type_name in {"timestamp", "timestamptz"}:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if type_name == "time":
        return time.fromisoformat(str(value))
    if type_name in {"json", "jsonb"}:
        return value if isinstance(value, str) else canonical_json(value)
    return value if isinstance(value, str) else str(value)


def _sample(
    record: Any, facts: WriteFacts, action: str, varies: bool
) -> Mapping[str, object]:
    """One row for the card, which must read the same when rechecked after approval."""

    if facts.operation == "delete":
        return {"action": action, "row": _record_value(record, "__daita_whole_row")}
    sample: dict[str, object] = {"action": action}
    if action == "insert":
        sample["values"] = {
            column: (
                COMPUTED_AT_WRITE
                if varies
                else _record_value(record, f"__daita_value_{index}")
            )
            for index, column in enumerate(facts.insert_columns)
        }
    else:
        sample["before"] = {
            column: _record_value(record, f"__daita_before_{index}")
            for index, column in enumerate(facts.update_columns)
        }
        sample["after"] = {
            column: (
                COMPUTED_AT_WRITE
                if varies
                else _record_value(record, f"__daita_after_{index}")
            )
            for index, column in enumerate(facts.update_columns)
        }
    return sample


def _row_digest(*parts: object) -> int:
    return int(sha256(canonical_json(parts).encode("utf-8")).hexdigest(), 16)


def _sum_text(total: int) -> str:
    # A sum, so the digest does not depend on the order rows arrive in.
    return f"{total % 2**256:064x}"


def _intent_sha256(facts: WriteFacts) -> str:
    statement = PostgreSQLStatement.of(facts)
    return _sha256_json(
        {"sql": statement.canonical_sql, "parameters": statement.parameters}
    )


def _preview(
    agent_id: str,
    source_id: str,
    facts: WriteFacts,
    scope: RelationalWriteScope,
    live: _LiveTable,
    scan: _Scan,
    key: tuple[str, ...] | None,
    varies: bool,
) -> WritePreview:
    statement = PostgreSQLStatement.of(facts)
    warnings: list[str] = []
    if scan.affected_rows == 0:
        warnings.append("target_not_found")
    if statement.matches_all:
        warnings.append("where_matches_every_row")
    if live.triggers:
        warnings.append("triggers_not_previewed")
    if live.cascades and facts.operation != "insert":
        warnings.append("cascades_not_previewed")
    if scan.samples_omitted:
        warnings.append("oversized_samples_omitted")
    if varies:
        warnings.append("values_can_change")
    if scan.matched_twice:
        warnings.append("row_matched_more_than_once")
    intent = _intent_sha256(facts)
    return WritePreview(
        source_id=source_id,
        table=facts.table,
        operation=facts.operation,
        sql=statement.canonical_sql,
        columns_changed=facts.columns,
        affected_rows=scan.affected_rows,
        inserted_rows=scan.counts.get("insert", 0),
        updated_rows=scan.counts.get("update", 0),
        deleted_rows=scan.counts.get("delete", 0),
        skipped_rows=scan.counts.get("skip", 0),
        samples=scan.samples,
        triggers=live.triggers[:_MAX_LISTED_NAMES],
        cascades=live.cascades[:_MAX_LISTED_NAMES],
        warnings=tuple(warnings),
        # A DELETE always reports the old row location; others need a usable key.
        verification=(
            "rows" if key is not None or facts.operation == "delete" else "count"
        ),
        intent_sha256=intent,
        target_set_sha256=scan.target_set_sha256,
        structure_sha256=live.structure_sha256,
        preview_fingerprint=_sha256_json(
            {
                "agent_id": agent_id,
                "source_id": source_id,
                "resource_id": facts.resource_id,
                "intent_sha256": intent,
                "structure_sha256": live.structure_sha256,
                "target_set_sha256": scan.target_set_sha256,
                "affected_rows": scan.affected_rows,
                "permission_fingerprint": scope.authorization_fingerprint,
            }
        ),
    )


def _admit_guardrails(live: _LiveTable, facts: WriteFacts) -> None:
    operations = {facts.operation, *(("update",) if facts.update_columns else ())}
    rejections, _remediation = _guardrail_rejections(live.facts, tuple(operations))
    if rejections:
        raise RelationalWriteError(
            "write_guardrail_rejected",
            f"{facts.table} or the source's role cannot take this write: "
            f"{', '.join(rejections)}.",
            {"rejection_codes": rejections},
        )


def _guardrail_rejections(
    facts: Mapping[str, object], operations: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    checks = (
        (
            facts["relation_kind"] != "r",
            "write_relation_not_base_table",
            "select_supported_base_table",
        ),
        (
            facts["is_partition"] is True,
            "write_relation_partitioned",
            "select_supported_base_table",
        ),
        (
            facts["has_inheritance"] is True,
            "write_relation_inherited",
            "select_supported_base_table",
        ),
        (
            facts["row_level_security"] is True,
            "write_relation_rls_enabled",
            "select_relation_without_rls",
        ),
        (
            facts["force_row_level_security"] is True,
            "write_relation_force_rls",
            "select_relation_without_rls",
        ),
        (
            facts["has_rewrite_rules"] is True,
            "write_relation_rewrite_rules",
            "select_relation_without_rewrite_rules",
        ),
        (
            facts["role_superuser"] is True,
            "write_role_superuser",
            "use_least_privileged_role",
        ),
        (
            facts["role_bypass_rls"] is True,
            "write_role_bypass_rls",
            "use_least_privileged_role",
        ),
        (
            facts["role_create_database"] is True,
            "write_role_create_database",
            "use_least_privileged_role",
        ),
        (
            facts["role_create_role"] is True,
            "write_role_create_role",
            "use_least_privileged_role",
        ),
        (
            facts["role_replication"] is True,
            "write_role_replication",
            "use_least_privileged_role",
        ),
        (
            facts["can_connect"] is not True,
            "write_privilege_connect_missing",
            "grant_connect_externally",
        ),
        (
            facts["can_use_schema"] is not True,
            "write_privilege_schema_usage_missing",
            "grant_schema_usage_externally",
        ),
        (
            facts["can_select_table"] is not True,
            "write_privilege_table_select_missing",
            "grant_table_select_externally",
        ),
        (
            "update" in operations and facts["can_update_columns"] is not True,
            "write_privilege_column_update_missing",
            "grant_column_update_externally",
        ),
        (
            "insert" in operations and facts["can_insert_columns"] is not True,
            "write_privilege_column_insert_missing",
            "grant_column_insert_externally",
        ),
        (
            "delete" in operations and facts["can_delete_rows"] is not True,
            "write_privilege_table_delete_missing",
            "grant_table_delete_externally",
        ),
    )
    return (
        tuple(code for rejected, code, _category in checks if rejected),
        tuple(
            dict.fromkeys(category for rejected, _code, category in checks if rejected)
        ),
    )


def _readiness(
    source_id: str,
    table: str,
    operations: tuple[str, ...],
    *,
    scope_ready: bool = False,
    facts: Mapping[str, object] | None = None,
    rejections: tuple[str, ...],
    remediation: tuple[str, ...],
) -> RelationalWriteReadiness:
    role: dict[str, object] = dict.fromkeys(_READINESS_ROLE_KEYS)
    privileges: dict[str, object] = dict.fromkeys(_READINESS_PRIVILEGE_KEYS)
    relation: dict[str, object] = dict.fromkeys(_READINESS_RELATION_KEYS)
    relation["exists"] = False if facts is None else True
    if facts is not None:
        role.update(
            {
                "superuser": facts["role_superuser"],
                "bypass_rls": facts["role_bypass_rls"],
                "create_database": facts["role_create_database"],
                "create_role": facts["role_create_role"],
                "replication": facts["role_replication"],
            }
        )
        privileges.update(
            {
                "database_connect": facts["can_connect"],
                "schema_usage": facts["can_use_schema"],
                "table_select": facts["can_select_table"],
                "requested_columns_insert": facts["can_insert_columns"],
                "requested_columns_update": facts["can_update_columns"],
                "table_delete": facts["can_delete_rows"],
            }
        )
        relation.update(
            {
                "base_table": facts["relation_kind"] == "r",
                "partition": facts["is_partition"],
                "inheritance": facts["has_inheritance"],
                "row_level_security": facts["row_level_security"],
                "force_row_level_security": facts["force_row_level_security"],
                "user_triggers": facts["has_user_triggers"],
                "rewrite_rules": facts["has_rewrite_rules"],
            }
        )
    return RelationalWriteReadiness(
        source_id=source_id,
        table=table,
        allowed_operations=operations,
        daita_scope_ready=scope_ready,
        ready_for_preview=not rejections,
        proves_execution=False,
        role_attributes=FrozenJsonObject.from_mapping(role),
        privileges=FrozenJsonObject.from_mapping(privileges),
        relation=FrozenJsonObject.from_mapping(relation),
        rejection_codes=tuple(dict.fromkeys(rejections)),
        remediation_categories=tuple(dict.fromkeys(remediation)),
    )


def _normalized_write_failure(error: BaseException) -> str:
    if isinstance(error, TimeoutError):
        return "write_statement_timeout"
    sqlstate = getattr(error, "sqlstate", None)
    if isinstance(sqlstate, str) and sqlstate.startswith("23"):
        return "write_constraint_violation"
    if sqlstate == "42501":
        return "write_permission_denied"
    if sqlstate == "55P03":
        return "write_lock_timeout"
    if sqlstate == "57014":
        return "write_statement_timeout"
    # Under REPEATABLE READ a concurrent change to a locked row is a conflict.
    if sqlstate == "40001":
        return "write_state_changed"
    # It described cleanly at preview, so the table changed under it.
    if isinstance(sqlstate, str) and sqlstate.startswith("42"):
        return "write_state_changed"
    return "write_not_committed"


def _preview_failure(
    error: Exception, facts: WriteFacts, timeout_seconds: float
) -> RelationalWriteError:
    sqlstate = getattr(error, "sqlstate", None)
    if sqlstate == "55P03":
        return RelationalWriteError(
            "write_lock_timeout", "PostgreSQL preview exceeded its lock timeout."
        )
    # A class 42 position would point into Daita's wrapper, not the model's SQL.
    failure = _classified_failure(
        error, PostgreSQLStatement.of(facts), timeout_seconds  # type: ignore[arg-type]
    )
    return RelationalWriteError(
        (
            failure.code
            if failure.code
            in {"postgresql_query_timeout", "postgresql_connection_failed"}
            else "write_preview_failed"
        ),
        f"PostgreSQL could not preview this write: {failure}",
        {key: value for key, value in failure.details.items() if key != "position"},
    )


def _record_value(record: Any, name: str) -> object:
    try:
        return record[name]
    except (KeyError, IndexError, TypeError):
        raise RelationalWriteError(
            "write_preview_failed", "PostgreSQL returned an incomplete preview row."
        ) from None


def _sha256_json(value: object) -> str:
    return "sha256:" + sha256(canonical_json(value).encode("utf-8")).hexdigest()


__all__ = [
    "PostgreSQLWriteBackend",
    "RelationalWriteCancelled",
    "RelationalWriteError",
    "RelationalWriteReadiness",
]
