"""Controlled SQLite update backend behind the universal executor boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import sqlite3
import stat
import threading

from .._json import canonical_json
from ..domains.data.capabilities import (
    SQLITE_UPDATE_IMPACT_CAPABILITY_ID,
    SQLITE_UPDATE_IMPACT_EVIDENCE_KIND,
    SQLiteUpdateKnownNoEffectError,
    SQLiteUpdateImpactResult,
    SQLiteUpdateResult,
)
from ..domains.data.controller import CatalogSchemaReader
from ..domains.data.sql import (
    ResourceSchema,
    SQLiteUpdateRecipe,
    sqlite_declared_type_affinity,
    sqlite_identifier_key,
    validate_sqlite_update_recipe,
)
from ..operations.models import TaskStatus
from ..operations.store import OperationStore
from .protocols import SourceStore


class SQLiteUpdateError(RuntimeError):
    """Stable write-boundary failure without connector or data leakage."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class _ApprovedResourceFacts:
    resource_id: str
    resource_revision: str
    table_name: str
    columns: tuple[str, ...]
    column_declared_types: tuple[tuple[str, str], ...]
    unique_key_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ProtectedPath:
    path: Path
    identity: tuple[int, int]


@dataclass(slots=True)
class _OpenedSource:
    path: Path
    descriptor: int
    identity: tuple[int, int]
    protected_paths: tuple[_ProtectedPath, ...]

    def close(self) -> None:
        descriptor = self.descriptor
        self.descriptor = -1
        if descriptor >= 0:
            os.close(descriptor)


class SQLiteUpdateBackend:
    """Revalidate one catalog recipe and apply at most one conditional update."""

    def __init__(
        self,
        sources: SourceStore,
        catalog: CatalogSchemaReader,
        operations: OperationStore,
        *,
        protected_paths: Iterable[str | Path] = (),
    ) -> None:
        if not isinstance(sources, SourceStore):
            raise TypeError("sources must implement SourceStore")
        if not callable(getattr(catalog, "resource_schemas", None)):
            raise TypeError("catalog must provide resource_schemas")
        if not callable(getattr(operations, "load", None)):
            raise TypeError("operations must provide load")
        self._sources = sources
        self._catalog = catalog
        self._operations = operations
        self._protected_paths = _normalize_protected_paths(protected_paths)

    async def execute_update_impact(
        self,
        *,
        agent_id: str,
        source_id: str,
        resource_id: str,
        key_column: str,
        key_value: str,
        target_column: str,
        expected_value: str,
        new_value: str,
        maximum_rows: int,
    ) -> SQLiteUpdateImpactResult:
        recipe, resource, source = await self._current_recipe(
            agent_id=agent_id,
            source_id=source_id,
            resource_id=resource_id,
            key_column=key_column,
            key_value=key_value,
            target_column=target_column,
            expected_value=expected_value,
            new_value=new_value,
            writable_path=False,
        )
        try:
            _require_single_row_bound(maximum_rows)
            matched_rows, eligible_rows, live_source_revision = await _run_impact(
                source,
                recipe,
                resource,
                expected_schema_version=_expected_schema_version(
                    recipe.source_revision
                ),
            )
            return SQLiteUpdateImpactResult(
                source_id=recipe.source_id,
                resource_id=recipe.resource_id,
                resource_revision=recipe.resource_revision,
                source_revision=live_source_revision,
                key_column=recipe.key_column,
                target_column=recipe.target_column,
                recipe_fingerprint=recipe.recipe_fingerprint,
                matched_rows=matched_rows,
                eligible_rows=eligible_rows,
                maximum_rows=maximum_rows,
            )
        finally:
            source.close()

    async def execute_update(
        self,
        *,
        operation_id: str,
        agent_id: str,
        source_id: str,
        resource_id: str,
        key_column: str,
        key_value: str,
        target_column: str,
        expected_value: str,
        new_value: str,
        impact_evidence_id: str,
        maximum_rows: int,
    ) -> SQLiteUpdateResult:
        source: _OpenedSource | None = None
        write_boundary_entered = False
        try:
            _require_single_row_bound(maximum_rows)
            recipe, resource, source = await self._current_recipe(
                agent_id=agent_id,
                source_id=source_id,
                resource_id=resource_id,
                key_column=key_column,
                key_value=key_value,
                target_column=target_column,
                expected_value=expected_value,
                new_value=new_value,
                writable_path=True,
            )
            try:
                try:
                    await self._require_current_impact(
                        operation_id=operation_id,
                        agent_id=agent_id,
                        evidence_id=impact_evidence_id,
                        recipe=recipe,
                        maximum_rows=maximum_rows,
                    )
                except SQLiteUpdateError as error:
                    raise _known_no_effect(error) from error
                expected_schema_version = _expected_schema_version(
                    recipe.source_revision
                )
                write_boundary_entered = True
                affected_rows, live_source_revision = await _run_update(
                    source,
                    recipe,
                    resource,
                    expected_schema_version=expected_schema_version,
                )
            finally:
                source.close()
            return SQLiteUpdateResult(
                source_id=recipe.source_id,
                resource_id=recipe.resource_id,
                resource_revision=recipe.resource_revision,
                source_revision=live_source_revision,
                key_column=recipe.key_column,
                target_column=recipe.target_column,
                recipe_fingerprint=recipe.recipe_fingerprint,
                impact_evidence_id=impact_evidence_id,
                affected_rows=affected_rows,
                maximum_rows=maximum_rows,
            )
        except SQLiteUpdateKnownNoEffectError:
            raise
        except SQLiteUpdateError as error:
            if not write_boundary_entered:
                raise SQLiteUpdateKnownNoEffectError(
                    error.code,
                    str(error),
                ) from error
            raise
        except Exception as error:
            if not write_boundary_entered:
                raise SQLiteUpdateKnownNoEffectError(
                    "sqlite_update_preflight_failed",
                    "SQLite update preflight could not be completed safely.",
                ) from error
            raise

    async def _current_recipe(
        self,
        *,
        agent_id: str,
        source_id: str,
        resource_id: str,
        key_column: str,
        key_value: str,
        target_column: str,
        expected_value: str,
        new_value: str,
        writable_path: bool,
    ) -> tuple[SQLiteUpdateRecipe, _ApprovedResourceFacts, _OpenedSource]:
        registration = await self._sources.load_source(agent_id, source_id)
        if registration is None or not registration.active:
            raise SQLiteUpdateError(
                "source_not_available",
                "The SQLite source is not attached to this agent.",
            )
        if registration.adapter_id != "sqlite":
            raise SQLiteUpdateError(
                "source_adapter_mismatch",
                "The selected source is not a SQLite source.",
            )
        write_access = registration.configuration.get("write_access") is True
        if not write_access:
            raise SQLiteUpdateError(
                "source_write_access_required",
                "The SQLite source was not explicitly attached with write access.",
            )
        try:
            resources = await self._catalog.resource_schemas(agent_id, source_id)
        except (KeyError, ValueError) as error:
            raise SQLiteUpdateError(
                "catalog_schema_unavailable",
                "Current catalog schema is unavailable for the SQLite source.",
            ) from error
        validation = validate_sqlite_update_recipe(
            source_id=source_id,
            resource_id=resource_id,
            key_column=key_column,
            key_value=key_value,
            target_column=target_column,
            expected_value=expected_value,
            new_value=new_value,
            resources=resources,
            source_write_access=write_access,
        )
        if not validation.valid or validation.recipe is None:
            issue_codes = ",".join(validation.issue_codes[:8]) or "invalid_recipe"
            raise SQLiteUpdateError(
                "update_revalidation_failed",
                "SQLite update failed deterministic revalidation: " + issue_codes,
            )
        configured_path = registration.configuration.get("path")
        if not isinstance(configured_path, str):
            raise SQLiteUpdateError(
                "source_configuration_invalid",
                "SQLite source configuration is missing its path.",
            )
        selected_resource = next(
            (
                item
                for item in resources
                if isinstance(item, ResourceSchema)
                and item.source_id == source_id
                and item.resource_id == validation.recipe.resource_id
            ),
            None,
        )
        if selected_resource is None or selected_resource.revision is None:
            raise SQLiteUpdateError(
                "catalog_schema_unavailable",
                "Current catalog schema is unavailable for the SQLite source.",
            )
        resource = _ApprovedResourceFacts(
            resource_id=selected_resource.resource_id,
            resource_revision=selected_resource.revision,
            table_name=selected_resource.name,
            columns=selected_resource.columns,
            column_declared_types=selected_resource.column_declared_types,
            unique_key_columns=selected_resource.unique_key_columns,
        )
        return (
            validation.recipe,
            resource,
            _open_source_path(
                configured_path,
                writable=writable_path,
                protected_paths=self._protected_paths,
            ),
        )

    async def _require_current_impact(
        self,
        *,
        operation_id: str,
        agent_id: str,
        evidence_id: str,
        recipe: SQLiteUpdateRecipe,
        maximum_rows: int,
    ) -> None:
        try:
            versioned = await self._operations.load(operation_id)
        except Exception as error:
            raise SQLiteUpdateError(
                "impact_evidence_unavailable",
                "The approved update impact evidence is unavailable.",
            ) from error
        snapshot = versioned.snapshot
        if snapshot.operation.agent_id != agent_id:
            raise SQLiteUpdateError(
                "impact_evidence_scope_mismatch",
                "The update impact evidence belongs to another agent.",
            )
        evidence = next(
            (
                item
                for item in snapshot.evidence
                if item.id == evidence_id and item.accepted
            ),
            None,
        )
        if evidence is None or evidence.operation_id != operation_id:
            raise SQLiteUpdateError(
                "impact_evidence_unavailable",
                "The approved update requires accepted impact evidence from this operation.",
            )
        if evidence.kind != SQLITE_UPDATE_IMPACT_EVIDENCE_KIND:
            raise SQLiteUpdateError(
                "impact_evidence_kind_mismatch",
                "The cited evidence is not SQLite update impact evidence.",
            )
        producing_task = next(
            (item for item in snapshot.tasks if item.id == evidence.task_id),
            None,
        )
        if (
            producing_task is None
            or producing_task.status is not TaskStatus.SUCCEEDED
            or producing_task.capability_id != SQLITE_UPDATE_IMPACT_CAPABILITY_ID
            or evidence.id not in producing_task.evidence_ids
        ):
            raise SQLiteUpdateError(
                "impact_evidence_provenance_invalid",
                "The cited update impact evidence has invalid task provenance.",
            )
        payload = evidence.payload
        expected: dict[str, object] = {
            "source_id": recipe.source_id,
            "resource_id": recipe.resource_id,
            "resource_revision": recipe.resource_revision,
            "source_revision": recipe.source_revision,
            "key_column": recipe.key_column,
            "target_column": recipe.target_column,
            "recipe_fingerprint": recipe.recipe_fingerprint,
            "matched_rows": 1,
            "eligible_rows": 1,
            "maximum_rows": maximum_rows,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise SQLiteUpdateError(
                "impact_evidence_stale",
                "The cited impact evidence does not authorize this exact update.",
            )
        payload_hash = (
            "sha256:" + sha256(canonical_json(payload).encode("utf-8")).hexdigest()
        )
        if evidence.blob_id is not None or evidence.content_hash != payload_hash:
            raise SQLiteUpdateError(
                "impact_evidence_integrity_failed",
                "The cited impact evidence failed its integrity check.",
            )


def _require_single_row_bound(maximum_rows: int) -> None:
    if (
        not isinstance(maximum_rows, int)
        or isinstance(maximum_rows, bool)
        or maximum_rows != 1
    ):
        raise SQLiteUpdateError(
            "update_bound_invalid",
            "Controlled SQLite updates require an exact one-row maximum.",
        )


def _known_no_effect(error: SQLiteUpdateError) -> SQLiteUpdateKnownNoEffectError:
    return SQLiteUpdateKnownNoEffectError(error.code, str(error))


def _normalize_protected_paths(
    values: Iterable[str | Path],
) -> tuple[_ProtectedPath, ...]:
    if isinstance(values, (str, bytes, Path)):
        raise TypeError("protected_paths must be an iterable of paths")
    protected: list[_ProtectedPath] = []
    for value in values:
        if not isinstance(value, (str, Path)):
            raise TypeError("protected_paths must contain only strings or Paths")
        path = Path(os.path.abspath(value))
        descriptor = -1
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            metadata = os.fstat(descriptor)
        except OSError as error:
            raise ValueError(
                "protected_paths must name existing regular files"
            ) from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("protected_paths must name existing regular files")
        identity = (metadata.st_dev, metadata.st_ino)
        if identity not in {item.identity for item in protected}:
            protected.append(_ProtectedPath(path=path, identity=identity))
    return tuple(protected)


def _expected_schema_version(source_revision: str) -> int:
    prefix = "schema_version:"
    if not source_revision.startswith(prefix):
        raise SQLiteUpdateError(
            "catalog_source_revision_invalid",
            "SQLite catalog source revision is malformed.",
        )
    raw_version = source_revision.removeprefix(prefix)
    if not raw_version.isascii() or not raw_version.isdecimal():
        raise SQLiteUpdateError(
            "catalog_source_revision_invalid",
            "SQLite catalog source revision is malformed.",
        )
    version = int(raw_version)
    if version < 0 or str(version) != raw_version:
        raise SQLiteUpdateError(
            "catalog_source_revision_invalid",
            "SQLite catalog source revision is malformed.",
        )
    return version


def _open_source_path(
    value: str,
    *,
    writable: bool,
    protected_paths: tuple[_ProtectedPath, ...],
) -> _OpenedSource:
    lexical = Path(os.path.abspath(value))
    flags = os.O_RDWR if writable else os.O_RDONLY
    descriptor = -1
    try:
        resolved = lexical.resolve(strict=True)
        descriptor = os.open(resolved, flags | getattr(os, "O_NOFOLLOW", 0))
        metadata = os.fstat(descriptor)
        if lexical != resolved or not stat.S_ISREG(metadata.st_mode):
            raise SQLiteUpdateError(
                "source_path_invalid",
                "SQLite source path is unavailable or unsafe.",
            )
        identity = (metadata.st_dev, metadata.st_ino)
        protected_identities = {item.identity for item in protected_paths}
        protected_identities.update(_current_protected_identities(protected_paths))
        if identity in protected_identities:
            raise SQLiteUpdateError(
                "protected_agent_state_source",
                "Agent state cannot be used as a writable SQLite data source.",
            )
        opened = _OpenedSource(
            path=resolved,
            descriptor=descriptor,
            identity=identity,
            protected_paths=protected_paths,
        )
        _require_opened_source_identity(opened)
        return opened
    except SQLiteUpdateError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except (OSError, RuntimeError, ValueError) as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise SQLiteUpdateError(
            "source_path_invalid",
            "SQLite source path is unavailable or unsafe.",
        ) from error


def _current_protected_identities(
    protected_paths: tuple[_ProtectedPath, ...],
) -> set[tuple[int, int]]:
    identities: set[tuple[int, int]] = set()
    for protected in protected_paths:
        descriptor = -1
        try:
            descriptor = os.open(
                protected.path,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            metadata = os.fstat(descriptor)
        except OSError:
            continue
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if stat.S_ISREG(metadata.st_mode):
            identities.add((metadata.st_dev, metadata.st_ino))
    return identities


def _require_opened_source_identity(source: _OpenedSource) -> None:
    try:
        descriptor_metadata = os.fstat(source.descriptor)
        path_metadata = os.stat(source.path, follow_symlinks=False)
    except OSError as error:
        raise SQLiteUpdateError(
            "source_path_changed",
            "SQLite source identity changed during connection establishment.",
        ) from error
    descriptor_identity = (descriptor_metadata.st_dev, descriptor_metadata.st_ino)
    path_identity = (path_metadata.st_dev, path_metadata.st_ino)
    if (
        not stat.S_ISREG(descriptor_metadata.st_mode)
        or not stat.S_ISREG(path_metadata.st_mode)
        or descriptor_identity != source.identity
        or path_identity != source.identity
    ):
        raise SQLiteUpdateError(
            "source_path_changed",
            "SQLite source identity changed during connection establishment.",
        )
    protected_identities = {item.identity for item in source.protected_paths}
    protected_identities.update(_current_protected_identities(source.protected_paths))
    if source.identity in protected_identities:
        raise SQLiteUpdateError(
            "protected_agent_state_source",
            "Agent state cannot be used as a writable SQLite data source.",
        )


def _connect_source(source: _OpenedSource, *, writable: bool) -> sqlite3.Connection:
    _require_opened_source_identity(source)
    mode = "rw" if writable else "ro"
    uri = f"{source.path.as_uri()}?mode={mode}"
    connection = sqlite3.connect(uri, uri=True, timeout=5.0)
    try:
        _require_opened_source_identity(source)
        database_row = connection.execute("PRAGMA database_list").fetchone()
        if (
            database_row is None
            or len(database_row) < 3
            or not isinstance(database_row[2], str)
        ):
            raise SQLiteUpdateError(
                "source_path_changed",
                "SQLite source identity changed during connection establishment.",
            )
        try:
            connected_metadata = os.stat(
                database_row[2],
                follow_symlinks=False,
            )
        except OSError as error:
            raise SQLiteUpdateError(
                "source_path_changed",
                "SQLite source identity changed during connection establishment.",
            ) from error
        if (
            not stat.S_ISREG(connected_metadata.st_mode)
            or (connected_metadata.st_dev, connected_metadata.st_ino) != source.identity
        ):
            raise SQLiteUpdateError(
                "source_path_changed",
                "SQLite source identity changed during connection establishment.",
            )
        return connection
    except BaseException:
        try:
            connection.close()
        except sqlite3.Error:
            pass
        raise


async def _run_impact(
    source: _OpenedSource,
    recipe: SQLiteUpdateRecipe,
    resource: _ApprovedResourceFacts,
    *,
    expected_schema_version: int,
) -> tuple[int, int, str]:
    cancellation = threading.Event()
    worker = asyncio.create_task(
        asyncio.to_thread(
            _run_impact_sync,
            source,
            recipe,
            resource,
            expected_schema_version,
            cancellation,
        )
    )
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        cancellation.set()
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        raise


def _run_impact_sync(
    source: _OpenedSource,
    recipe: SQLiteUpdateRecipe,
    resource: _ApprovedResourceFacts,
    expected_schema_version: int,
    cancellation: threading.Event,
) -> tuple[int, int, str]:
    connection: sqlite3.Connection | None = None
    try:
        connection = _connect_source(source, writable=False)
        _configure_connection(connection, cancellation, query_only=True)
        connection.execute("BEGIN")
        live_source_revision = _verify_live_scope(
            connection,
            recipe,
            resource,
            expected_schema_version,
        )
        matched_rows, eligible_rows = _measure_impact(connection, recipe)
        connection.execute("COMMIT")
        return matched_rows, eligible_rows, live_source_revision
    except SQLiteUpdateError:
        raise
    except sqlite3.Error as error:
        raise SQLiteUpdateError(
            "sqlite_impact_failed",
            "SQLite could not complete the bounded update impact preview.",
        ) from error
    finally:
        _close_connection(connection)


async def _run_update(
    source: _OpenedSource,
    recipe: SQLiteUpdateRecipe,
    resource: _ApprovedResourceFacts,
    *,
    expected_schema_version: int,
) -> tuple[int, str]:
    cancellation = threading.Event()
    worker = asyncio.create_task(
        asyncio.to_thread(
            _run_update_sync,
            source,
            recipe,
            resource,
            expected_schema_version,
            cancellation,
        )
    )
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        cancellation.set()
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        raise


def _run_update_sync(
    source: _OpenedSource,
    recipe: SQLiteUpdateRecipe,
    resource: _ApprovedResourceFacts,
    expected_schema_version: int,
    cancellation: threading.Event,
) -> tuple[int, str]:
    connection: sqlite3.Connection | None = None
    effect_attempted = False
    commit_attempted = False
    try:
        connection = _connect_source(source, writable=True)
        _configure_connection(connection, cancellation, query_only=False)
        connection.execute("BEGIN IMMEDIATE")
        live_source_revision = _verify_live_scope(
            connection,
            recipe,
            resource,
            expected_schema_version,
        )
        matched_rows, eligible_rows = _measure_impact(connection, recipe)
        if matched_rows != 1 or eligible_rows != 1:
            raise SQLiteUpdateError(
                "update_precondition_changed",
                "SQLite update preconditions changed after impact approval.",
            )
        connection.set_authorizer(_update_authorizer(recipe))
        table = _quote_identifier(recipe.table_name)
        key = _quote_identifier(recipe.key_column)
        target = _quote_identifier(recipe.target_column)
        _require_opened_source_identity(source)
        effect_attempted = True
        cursor = connection.execute(
            f"UPDATE {table} SET {target} = ? WHERE {key} = ? AND {target} = ?",
            (recipe.new_value, recipe.key_value, recipe.expected_value),
        )
        connection.set_authorizer(None)
        if cursor.rowcount != 1:
            raise SQLiteUpdateError(
                "update_row_count_invalid",
                "SQLite did not apply exactly one controlled update.",
            )
        stored = connection.execute(
            f"SELECT {target}, typeof({target}) FROM {table} WHERE {key} = ?",
            (recipe.key_value,),
        ).fetchone()
        if (
            stored is None
            or len(stored) != 2
            or stored[0] != recipe.new_value
            or stored[1] != "text"
        ):
            raise SQLiteUpdateError(
                "update_result_mismatch",
                "SQLite did not persist the exact controlled TEXT value.",
            )
        commit_attempted = True
        connection.execute("COMMIT")
        return cursor.rowcount, live_source_revision
    except SQLiteUpdateError as error:
        if not commit_attempted and _rollback_confirmed(
            connection,
            effect_attempted=effect_attempted,
        ):
            raise _known_no_effect(error) from error
        raise
    except sqlite3.Error as error:
        normalized = SQLiteUpdateError(
            "sqlite_update_failed",
            "SQLite could not complete the controlled update.",
        )
        if not commit_attempted and _rollback_confirmed(
            connection,
            effect_attempted=effect_attempted,
        ):
            raise _known_no_effect(normalized) from error
        raise normalized from error
    finally:
        _close_connection(connection)


def _configure_connection(
    connection: sqlite3.Connection,
    cancellation: threading.Event,
    *,
    query_only: bool,
) -> None:
    if query_only:
        connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA trusted_schema = OFF")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.enable_load_extension(False)
    connection.set_progress_handler(lambda: 1 if cancellation.is_set() else 0, 500)
    if hasattr(connection, "setlimit"):
        connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 65_536)


def _verify_live_scope(
    connection: sqlite3.Connection,
    recipe: SQLiteUpdateRecipe,
    resource: _ApprovedResourceFacts,
    expected_schema_version: int,
) -> str:
    version_row = connection.execute("PRAGMA schema_version").fetchone()
    if version_row is None or not isinstance(version_row[0], int):
        raise SQLiteUpdateError(
            "source_revision_unavailable",
            "SQLite source revision could not be read.",
        )
    live_schema_version = version_row[0]
    if live_schema_version != expected_schema_version:
        raise SQLiteUpdateError(
            "catalog_source_stale",
            "SQLite source schema changed after its catalog snapshot.",
        )
    if (
        resource.resource_id != recipe.resource_id
        or resource.resource_revision != recipe.resource_revision
        or sqlite_identifier_key(resource.table_name)
        != sqlite_identifier_key(recipe.table_name)
    ):
        raise SQLiteUpdateError(
            "catalog_resource_stale",
            "SQLite resource no longer matches its approved catalog revision.",
        )

    table_key = sqlite_identifier_key(recipe.table_name)
    table_rows = connection.execute(
        "SELECT schema, name, type FROM pragma_table_list"
    ).fetchall()
    matching_objects = tuple(
        row
        for row in table_rows
        if len(row) >= 3
        and row[0] == "main"
        and isinstance(row[1], str)
        and sqlite_identifier_key(row[1]) == table_key
    )
    if len(matching_objects) != 1 or matching_objects[0][2] != "table":
        raise SQLiteUpdateError(
            "resource_not_plain_table",
            "Controlled SQLite updates require a live ordinary table.",
        )
    live_table_name = matching_objects[0][1]
    assert isinstance(live_table_name, str)
    column_rows = connection.execute(
        "SELECT cid, name, type, pk, hidden FROM pragma_table_xinfo(?) " "ORDER BY cid",
        (live_table_name,),
    ).fetchall()
    visible_columns = tuple(
        row
        for row in column_rows
        if len(row) >= 5
        and isinstance(row[1], str)
        and isinstance(row[4], int)
        and row[4] != 1
    )
    live_columns = tuple(str(row[1]) for row in visible_columns)
    live_declared_types = tuple(
        (str(row[1]), str(row[2] or "UNKNOWN")) for row in visible_columns
    )
    target_key = sqlite_identifier_key(recipe.target_column)
    target_declared_type = next(
        (
            declared_type
            for column, declared_type in live_declared_types
            if sqlite_identifier_key(column) == target_key
        ),
        None,
    )
    if (
        target_declared_type is None
        or sqlite_declared_type_affinity(target_declared_type) != "text"
    ):
        raise SQLiteUpdateError(
            "target_column_affinity_not_supported",
            "Controlled SQLite updates require a live TEXT-affinity target column.",
        )
    if (
        tuple(sqlite_identifier_key(column) for column in live_columns)
        != tuple(sqlite_identifier_key(column) for column in resource.columns)
        or tuple(
            (sqlite_identifier_key(column), declared_type)
            for column, declared_type in live_declared_types
        )
        != tuple(
            (sqlite_identifier_key(column), declared_type)
            for column, declared_type in resource.column_declared_types
        )
        or tuple(
            sqlite_identifier_key(column)
            for column in _live_unique_key_columns(
                connection,
                live_table_name,
                visible_columns,
            )
        )
        != tuple(
            sqlite_identifier_key(column) for column in resource.unique_key_columns
        )
    ):
        raise SQLiteUpdateError(
            "catalog_resource_stale",
            "SQLite resource no longer matches its approved catalog revision.",
        )
    trigger_rows = connection.execute(
        "SELECT tbl_name FROM sqlite_schema "
        "WHERE type = 'trigger' AND tbl_name IS NOT NULL"
    ).fetchall()
    if any(
        isinstance(row[0], str) and sqlite_identifier_key(row[0]) == table_key
        for row in trigger_rows
    ):
        raise SQLiteUpdateError(
            "table_triggers_not_supported",
            "Controlled SQLite updates do not permit tables with triggers.",
        )
    return f"schema_version:{live_schema_version}"


def _live_unique_key_columns(
    connection: sqlite3.Connection,
    table_name: str,
    column_rows: tuple[tuple[object, ...], ...],
) -> tuple[str, ...]:
    primary_key_columns = tuple(
        (int(row[3]), str(row[1]))
        for row in column_rows
        if isinstance(row[3], int) and row[3] > 0
    )
    unique_columns: list[str] = []
    if len(primary_key_columns) == 1 and primary_key_columns[0][0] == 1:
        unique_columns.append(primary_key_columns[0][1])
    index_rows = connection.execute(
        'SELECT name, "unique", partial FROM pragma_index_list(?) ORDER BY name',
        (table_name,),
    ).fetchall()
    for index_row in index_rows:
        if (
            len(index_row) < 3
            or not isinstance(index_row[0], str)
            or index_row[1] != 1
            or index_row[2] != 0
        ):
            continue
        indexed_columns = connection.execute(
            "SELECT name FROM pragma_index_info(?) ORDER BY seqno",
            (index_row[0],),
        ).fetchall()
        if len(indexed_columns) != 1 or not isinstance(indexed_columns[0][0], str):
            continue
        column = indexed_columns[0][0]
        column_key = sqlite_identifier_key(column)
        if column_key not in {sqlite_identifier_key(item) for item in unique_columns}:
            unique_columns.append(column)
    return tuple(unique_columns)


def _measure_impact(
    connection: sqlite3.Connection,
    recipe: SQLiteUpdateRecipe,
) -> tuple[int, int]:
    table = _quote_identifier(recipe.table_name)
    key = _quote_identifier(recipe.key_column)
    target = _quote_identifier(recipe.target_column)
    matched = connection.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {key} = ?",
        (recipe.key_value,),
    ).fetchone()
    eligible = connection.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {key} = ? AND {target} = ?",
        (recipe.key_value, recipe.expected_value),
    ).fetchone()
    if (
        matched is None
        or eligible is None
        or not isinstance(matched[0], int)
        or not isinstance(eligible[0], int)
    ):
        raise SQLiteUpdateError(
            "impact_unavailable",
            "SQLite update impact could not be measured.",
        )
    return matched[0], eligible[0]


def _update_authorizer(recipe: SQLiteUpdateRecipe):
    table_key = sqlite_identifier_key(recipe.table_name)
    key_column = sqlite_identifier_key(recipe.key_column)
    target_column = sqlite_identifier_key(recipe.target_column)

    def authorize(
        action: int,
        arg1: str | None,
        arg2: str | None,
        database: str | None,
        trigger: str | None,
    ) -> int:
        if trigger is not None or database not in {None, "main"}:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_UPDATE:
            return (
                sqlite3.SQLITE_OK
                if sqlite_identifier_key(arg1 or "") == table_key
                and sqlite_identifier_key(arg2 or "") == target_column
                else sqlite3.SQLITE_DENY
            )
        if action == sqlite3.SQLITE_READ:
            return (
                sqlite3.SQLITE_OK
                if sqlite_identifier_key(arg1 or "") == table_key
                and sqlite_identifier_key(arg2 or "") in {key_column, target_column}
                else sqlite3.SQLITE_DENY
            )
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    return authorize


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _close_connection(connection: sqlite3.Connection | None) -> None:
    if connection is None:
        return
    try:
        connection.set_authorizer(None)
        connection.set_progress_handler(None, 0)
        if connection.in_transaction:
            connection.execute("ROLLBACK")
    except sqlite3.Error:
        pass
    try:
        connection.close()
    except sqlite3.Error:
        pass


def _rollback_confirmed(
    connection: sqlite3.Connection | None,
    *,
    effect_attempted: bool,
) -> bool:
    if connection is None:
        return not effect_attempted
    try:
        connection.set_authorizer(None)
        connection.set_progress_handler(None, 0)
        if not connection.in_transaction:
            return not effect_attempted
        connection.execute("ROLLBACK")
        return not connection.in_transaction
    except sqlite3.Error:
        return False


__all__ = ["SQLiteUpdateBackend", "SQLiteUpdateError"]
