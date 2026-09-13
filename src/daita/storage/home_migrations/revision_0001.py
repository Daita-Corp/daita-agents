"""Admit observed preproduction homes into production home revision 1."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from uuid import uuid4

from ..sqlite_schema import (
    MESSAGES_FOREIGN_KEYS,
    NAMED_INDEXES,
    REQUIRED_SQL_FRAGMENTS,
    REVISION_1_DATABASE_SQL,
    ROUTINE_OCCURRENCE_FOREIGN_KEYS,
    SCHEMA_REVISION_1,
    SOURCE_SCOPE_FOREIGN_KEYS,
    UNIQUE_CONSTRAINTS,
    SQLiteSchema,
    require_schema,
    table_names,
)
from .models import HomeMigration

_LEGACY_JOURNAL_COLUMNS = (
    ("ordinal", "INTEGER", 1, None, 0),
    ("migration_id", "TEXT", 1, None, 1),
    ("checksum", "TEXT", 1, None, 0),
)
_LEGACY_RECEIPT_COLUMNS = (
    ("agent_id", "TEXT", 1, None, 1),
    ("id", "TEXT", 1, None, 2),
    ("run_id", "TEXT", 1, None, 0),
    ("call_id", "TEXT", 1, None, 0),
    ("data", "TEXT", 1, None, 0),
)
_LEGACY_UPDATE_SCOPE_COLUMNS = (
    ("agent_id", "TEXT", 1, None, 1),
    ("source_id", "TEXT", 1, None, 2),
    ("resource_id", "TEXT", 1, None, 3),
    ("authorization_fingerprint", "TEXT", 1, None, 0),
    ("data", "TEXT", 1, None, 0),
)
_LEGACY_INBOX_COLUMNS = (
    ("agent_id", "TEXT", 1, None, 1),
    ("delivery_id", "TEXT", 1, None, 2),
    ("conversation_id", "TEXT", 1, None, 0),
    ("subject_kind", "TEXT", 1, None, 0),
    ("subject_id", "TEXT", 1, None, 0),
    ("logical_key", "TEXT", 1, None, 0),
    ("data", "TEXT", 1, None, 0),
)
_CURRENT_PREPRODUCTION_TABLES = {
    **{
        table: columns
        for table, columns in SCHEMA_REVISION_1.tables.items()
        if table != "agent_home_migrations"
    },
    "state_migrations": _LEGACY_JOURNAL_COLUMNS,
}
_ROUTINE_PREPRODUCTION_TABLES = {
    **{
        table: columns
        for table, columns in SCHEMA_REVISION_1.tables.items()
        if table
        not in {"agent_home_migrations", "effect_receipts", "relational_write_scopes"}
    },
    "database_write_receipts": _LEGACY_RECEIPT_COLUMNS,
    "postgresql_update_scopes": _LEGACY_UPDATE_SCOPE_COLUMNS,
    "state_migrations": _LEGACY_JOURNAL_COLUMNS,
}
_INBOX_PREPRODUCTION_TABLES = {
    **{
        table: columns
        for table, columns in _ROUTINE_PREPRODUCTION_TABLES.items()
        if table not in {"deliveries", "scheduled_routines", "routine_occurrences"}
    },
    "conversation_inbox": _LEGACY_INBOX_COLUMNS,
}


def _historical_schema(
    tables: dict[str, tuple[tuple[object, ...], ...]],
    *,
    inbox: bool = False,
) -> SQLiteSchema:
    foreign_keys = {
        "messages": MESSAGES_FOREIGN_KEYS,
        "source_read_scopes": SOURCE_SCOPE_FOREIGN_KEYS,
        (
            "postgresql_update_scopes"
            if "postgresql_update_scopes" in tables
            else "relational_write_scopes"
        ): SOURCE_SCOPE_FOREIGN_KEYS,
        **(
            {"routine_occurrences": ROUTINE_OCCURRENCE_FOREIGN_KEYS}
            if "routine_occurrences" in tables
            else {}
        ),
    }
    unique_constraints = {
        table: constraints
        for table, constraints in UNIQUE_CONSTRAINTS.items()
        if table in tables
    }
    unique_constraints["state_migrations"] = frozenset({("ordinal",)})
    if "database_write_receipts" in tables:
        unique_constraints["database_write_receipts"] = frozenset(
            {("agent_id", "run_id", "call_id")}
        )
    if inbox:
        unique_constraints["conversation_inbox"] = frozenset(
            {
                ("agent_id", "logical_key"),
                ("agent_id", "subject_kind", "subject_id"),
            }
        )
    return SQLiteSchema(
        tables=tables,
        foreign_keys=foreign_keys,
        unique_constraints=unique_constraints,
        named_indexes={
            name: definition
            for name, definition in NAMED_INDEXES.items()
            if definition[0] in tables
        },
        required_sql_fragments={
            table: fragments
            for table, fragments in REQUIRED_SQL_FRAGMENTS.items()
            if table in tables and not (inbox and table == "mcp_server_bindings")
        },
    )


_PREPRODUCTION_SCHEMAS = {
    "preproduction_current": _historical_schema(_CURRENT_PREPRODUCTION_TABLES),
    "preproduction_routines": _historical_schema(_ROUTINE_PREPRODUCTION_TABLES),
    "preproduction_inbox": _historical_schema(
        _INBOX_PREPRODUCTION_TABLES,
        inbox=True,
    ),
}

_DEFAULT_MODEL_CALL_POLICY = {
    "cleanup_timeout_seconds": 5.0,
    "connect_timeout_seconds": 5.0,
    "first_progress_timeout_seconds": 60.0,
    "input_count_timeout_seconds": 15.0,
    "max_attempt_seconds": 120.0,
    "max_request_seconds": 180.0,
    "pool_timeout_seconds": 5.0,
    "progress_idle_timeout_seconds": 30.0,
    "read_timeout_seconds": 120.0,
    "write_timeout_seconds": 30.0,
}
_REVISION_1_REVIEWED_MODEL_IDS = frozenset(
    {
        "codex:gpt-5.6-luna",
        "codex:gpt-5.6-sol",
        "codex:gpt-5.6-terra",
        "gemini:gemini-3.5-flash",
        "gemini:gemini-3.5-flash-lite",
        "gemini:gemini-3.6-flash",
        "openai:gpt-5.6-luna",
        "openai:gpt-5.6-sol",
        "openai:gpt-5.6-terra",
    }
)

_VERSIONED_RECORD_TABLES = {
    "autonomous_followups": "AutonomousFollowup",
    "deliveries": "Delivery",
    "job_runs": "JobRun",
    "mcp_server_bindings": "MCPServerBinding",
    "relational_write_scopes": "RelationalWriteScope",
    "routine_occurrences": "RoutineOccurrence",
    "scheduled_routines": "ScheduledRoutine",
    "source_read_scopes": "SourceReadScope",
}
_COPY_ORDER = (
    "metadata",
    "sources",
    "syncs",
    "snapshots",
    "runs",
    "messages",
    "semantic_annotations",
    "learning_candidates",
    "effect_receipts",
    "source_read_scopes",
    "relational_write_scopes",
    "mcp_server_bindings",
    "job_runs",
    "autonomous_followups",
    "scheduled_routines",
    "routine_occurrences",
    "deliveries",
)


def detect_preproduction_shape(connection: sqlite3.Connection) -> str:
    """Identify only complete preproduction shapes observed before the freeze."""

    matched = tuple(
        (label, schema)
        for label, schema in _PREPRODUCTION_SCHEMAS.items()
        if set(schema.tables)
        == {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
    )
    if len(matched) != 1:
        raise ValueError("agent home has no supported preproduction database shape")
    label, schema = matched[0]
    require_schema(connection, schema)
    legacy_rows = tuple(
        connection.execute(
            "SELECT ordinal, migration_id, checksum FROM state_migrations "
            "ORDER BY ordinal"
        )
    )
    if (
        len(legacy_rows) != 1
        or legacy_rows[0][0] != 1
        or legacy_rows[0][1] != "development_baseline"
        or not isinstance(legacy_rows[0][2], str)
        or len(legacy_rows[0][2]) != 64
        or any(character not in "0123456789abcdef" for character in legacy_rows[0][2])
    ):
        raise ValueError("preproduction migration marker is invalid")
    if connection.execute("PRAGMA quick_check(1)").fetchone() != ("ok",):
        raise ValueError("preproduction database integrity check failed")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise ValueError("preproduction database foreign keys are invalid")
    return label


def _require_empty(connection: sqlite3.Connection, table: str) -> None:
    if connection.execute(f'SELECT 1 FROM "{table}" LIMIT 1').fetchone() is not None:
        raise ValueError(
            f"preproduction {table} rows cannot be translated without changing authority"
        )


def _canonical_data(table: str, encoded: object) -> object:
    record_name = _VERSIONED_RECORD_TABLES.get(table)
    if record_name is None:
        return encoded
    if not isinstance(encoded, str):
        raise TypeError(f"preproduction {table} record is invalid")
    value = json.loads(encoded)
    if (
        not isinstance(value, dict)
        or value.get("__record__") != record_name
        or not isinstance(value.get("fields"), dict)
    ):
        raise ValueError(f"preproduction {table} record is invalid")
    fields = value["fields"]
    if "version" in fields:
        if fields["version"] != 1:
            raise ValueError(f"preproduction {table} codec is unsupported")
        del fields["version"]
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _copy_table(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    table: str,
) -> None:
    columns = tuple(str(column[0]) for column in SCHEMA_REVISION_1.tables[table])
    projection = ", ".join(f'"{column}"' for column in columns)
    rows = tuple(source.execute(f'SELECT {projection} FROM "{table}"'))
    data_index = columns.index("data") if "data" in columns else None
    if data_index is not None and table in _VERSIONED_RECORD_TABLES:
        rows = tuple(
            tuple(
                _canonical_data(table, value) if index == data_index else value
                for index, value in enumerate(row)
            )
            for row in rows
        )
    placeholders = ", ".join("?" for _ in columns)
    target.executemany(
        f'INSERT INTO "{table}" ({projection}) VALUES ({placeholders})',
        rows,
    )


def _upgrade_database(path: Path, source_shape: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.revision-1")
    source = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    target = sqlite3.connect(temporary)
    try:
        source.execute("PRAGMA query_only = ON")
        source.execute("PRAGMA foreign_keys = ON")
        target.execute("PRAGMA foreign_keys = ON")
        detected = detect_preproduction_shape(source)
        if detected != source_shape:
            raise RuntimeError("preproduction database changed after inventory")
        if source_shape != "preproduction_current":
            _require_empty(source, "database_write_receipts")
            _require_empty(source, "postgresql_update_scopes")
        if source_shape == "preproduction_inbox":
            _require_empty(source, "conversation_inbox")
        target.executescript("BEGIN IMMEDIATE;\n" + REVISION_1_DATABASE_SQL)
        source_tables = table_names(source)
        for table in _COPY_ORDER:
            if table in source_tables:
                _copy_table(source, target, table)
        target.commit()
        target.close()
        source.close()
        os.chmod(temporary, 0o600)
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
    except BaseException:
        try:
            target.rollback()
        except sqlite3.ProgrammingError:
            pass
        raise
    finally:
        target.close()
        source.close()
        temporary.unlink(missing_ok=True)


def _upgrade_model_config(path: Path) -> None:
    if not path.exists():
        return
    if path.is_symlink() or not path.is_file():
        raise ValueError("preproduction model configuration path is invalid")
    raw = path.read_bytes()
    if len(raw) > 64 * 1024:
        raise ValueError("preproduction model configuration exceeds its bound")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("preproduction model configuration is invalid")
    if set(value) not in (
        {"limits", "model_route"},
        {"limits", "model_route", "model_call_policy"},
    ):
        raise ValueError("preproduction model configuration fields are invalid")
    route = value.get("model_route")
    if not isinstance(route, dict) or set(route) != {"candidates", "retry_policy"}:
        raise ValueError("preproduction model route is invalid")
    retry = route.get("retry_policy")
    candidates = route.get("candidates")
    if not isinstance(retry, dict) or not isinstance(candidates, list):
        raise TypeError("preproduction model retry policy is invalid")
    canonical_candidates: list[dict[str, object]] = []
    for raw_candidate in candidates:
        if not isinstance(raw_candidate, dict):
            raise TypeError("preproduction model candidate is invalid")
        expected_common = {
            "allowed_sensitivities",
            "base_url",
            "provider_id",
            "secret_reference",
        }
        if set(raw_candidate) == expected_common | {"profile"}:
            provider_id = raw_candidate.get("provider_id")
            profile = raw_candidate.get("profile")
            if not isinstance(provider_id, str) or not isinstance(profile, dict):
                raise ValueError("preproduction model profile is invalid")
            context_window = profile.get("context_window_tokens")
            max_output = profile.get("max_output_tokens")
            if (
                not isinstance(context_window, int)
                or isinstance(context_window, bool)
                or not isinstance(max_output, int)
                or isinstance(max_output, bool)
            ):
                raise ValueError("preproduction model profile limits are invalid")
            profile_limits: dict[str, int] | None = (
                None
                if provider_id in _REVISION_1_REVIEWED_MODEL_IDS
                else {
                    "context_window_tokens": context_window,
                    "max_output_tokens": max_output,
                }
            )
            canonical_candidates.append(
                {
                    **{key: raw_candidate[key] for key in sorted(expected_common)},
                    "profile_limits": profile_limits,
                }
            )
        elif set(raw_candidate) == expected_common | {"profile_limits"}:
            canonical_candidates.append(dict(raw_candidate))
        else:
            raise ValueError("preproduction model candidate fields are invalid")
    route = dict(route)
    route["candidates"] = canonical_candidates
    value["model_route"] = route
    if set(retry) == {"attempts", "backoff_seconds"}:
        attempts = retry["attempts"]
        if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 1:
            raise ValueError("preproduction model retry attempts are invalid")
        route["retry_policy"] = {
            "backoff_seconds": retry["backoff_seconds"],
            "max_attempts_per_candidate": attempts,
            "max_total_attempts": min(25, attempts * max(1, len(candidates))),
        }
        value["model_route"] = route
    elif set(retry) != {
        "backoff_seconds",
        "max_attempts_per_candidate",
        "max_total_attempts",
    }:
        raise ValueError("preproduction model retry policy fields are invalid")
    value.setdefault("model_call_policy", dict(_DEFAULT_MODEL_CALL_POLICY))
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("model configuration write made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def apply(staged_home: Path, source_shape: str | None) -> None:
    if source_shape is None:
        raise ValueError("revision 1 requires an inventoried preproduction source")
    _upgrade_database(staged_home / "state.db", source_shape)
    _upgrade_model_config(staged_home / "config.json")


REVISION_1 = HomeMigration(
    revision=1,
    migration_id="agent_home_revision_1",
    definition=(
        "First production agent-home format. Replaces the mutable SQLite-only "
        "development ledger with the sole home revision ledger, canonicalizes "
        "model configuration, removes redundant record codec discriminators, and "
        "admits the three complete preproduction schemas observed before freeze."
    ),
    affected_paths=("state.db", "config.json"),
    target_schema=SCHEMA_REVISION_1,
    apply=apply,
    implementation_material=(REVISION_1_DATABASE_SQL,),
)


__all__ = ["REVISION_1", "detect_preproduction_shape"]
