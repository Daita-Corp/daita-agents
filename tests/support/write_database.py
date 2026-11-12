"""An in-memory PostgreSQL connection that answers one table's upserts.

It reads the derived SELECT and statement Daita sends, with sqlglot, far enough to
serve scripted flows that exercise approvals, receipts and commit certainty. Real SQL
behaviour is in ``tests/live/data/test_source_first_writes.py``.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Callable
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import sqlglot
from sqlglot import exp

KEY_COLUMN = "domain"


def write_guardrails(**changes: object) -> dict[str, object]:
    return {
        "relation_oid": "16384",
        "relation_kind": "r",
        "is_partition": False,
        "row_level_security": False,
        "force_row_level_security": False,
        "has_inheritance": False,
        "has_rewrite_rules": False,
        "role_superuser": False,
        "role_bypass_rls": False,
        "role_create_database": False,
        "role_create_role": False,
        "role_replication": False,
        "can_connect": True,
        "can_use_schema": True,
        "can_select_table": True,
        "can_delete_rows": True,
        "can_update_columns": True,
        "can_insert_columns": True,
        "columns": json.dumps(
            [
                ["id", "bigint", True, True, "d", ""],
                ["domain", "text", True, False, "", ""],
                ["name", "text", True, False, "", ""],
                ["evidence_url", "text", True, False, "", ""],
                ["notes", "text", False, False, "", ""],
            ]
        ),
        "triggers": None,
        "cascades": None,
        "keys": json.dumps([["id"], ["domain"]]),
        **changes,
    }


@dataclass(frozen=True)
class _ParameterType:
    name: str = "text"


def _value(node: Any, parameters: tuple[object, ...]) -> object:
    while isinstance(node, (exp.Cast, exp.Paren)):
        node = node.this
    if isinstance(node, exp.Parameter):
        return parameters[int(node.name) - 1]
    if isinstance(node, exp.Null):
        return None
    if isinstance(node, exp.Literal):
        return node.this
    raise AssertionError(f"the fake database cannot evaluate {node.sql()}")


def _incoming(
    values: Any, columns: tuple[str, ...], parameters: tuple[object, ...]
) -> list[dict[str, object]]:
    return [
        dict(
            zip(
                columns,
                (_value(cell, parameters) for cell in row.expressions),
                strict=True,
            )
        )
        for row in values.expressions
    ]


class Database:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, object]] = {}
        self.next_id = 1
        self.log: list[tuple[object, ...]] = []
        self.before_scan: Callable[[], None] | None = None
        self.lock_error: BaseException | None = None
        self.mutation_error: BaseException | None = None
        self.commit_error: BaseException | None = None
        self.rollback_error: BaseException | None = None
        self.bad_count = False
        self.bad_key = False
        self.not_an_insert = False
        self.planned: list[tuple[str, str, str]] = [("Insert", "public", "companies")]
        self.guardrails = write_guardrails()
        self.structure_revision = "catalog:sha256:" + "3" * 64

    def connect(self) -> Connection:
        self.log.append(("connect",))
        return Connection(self)

    def inserts(self) -> int:
        return len([item for item in self.log if item[0] == "mutation"])


class _Statement:
    def __init__(self, connection: Connection, sql: str) -> None:
        self.connection = connection
        self.sql = sql

    def get_parameters(self) -> tuple[_ParameterType, ...]:
        return (_ParameterType(),) * len(set(re.findall(r"\$\d+", self.sql)))

    async def fetchval(self, *_parameters: object, **_options: object) -> str:
        assert self.sql.startswith("/* daita:postgresql.write_describe */")
        self.connection.db.log.append(("describe", self.connection.readonly))
        return json.dumps(
            [
                {
                    "Plan": {
                        "Node Type": "ModifyTable",
                        "Operation": operation,
                        "Schema": schema,
                        "Relation Name": table,
                    }
                }
                for operation, schema, table in self.connection.db.planned
            ]
        )

    def cursor(self, *parameters: object, **_options: object) -> AsyncIterator[object]:
        if self.sql.startswith("INSERT"):
            return self.connection.mutate(self.sql, parameters)
        return self.connection.scan(self.sql, parameters)


class _Savepoint:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def start(self) -> None:
        self.db.log.append(("savepoint",))

    async def rollback(self) -> None:
        self.db.log.append(("savepoint_released",))


class Connection:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.work: dict[str, dict[str, object]] | None = None
        self.readonly = True

    def transaction(self, **options: object) -> Connection | _Savepoint:
        if self.work is not None:
            return _Savepoint(self.db)
        self.readonly = bool(options.get("readonly", False))
        self.db.log.append(("transaction", options))
        return self

    async def start(self) -> None:
        self.work = deepcopy(self.db.rows)

    async def commit(self) -> None:
        self.db.log.append(("commit", self.readonly))
        if not self.readonly:
            assert self.work is not None
            self.db.rows = deepcopy(self.work)
            if self.db.commit_error:
                raise self.db.commit_error

    async def rollback(self) -> None:
        self.db.log.append(("rollback",))
        if self.db.rollback_error:
            raise self.db.rollback_error
        self.work = None

    async def close(self, **_options: object) -> None:
        self.db.log.append(("close",))

    def terminate(self) -> None:
        self.db.log.append(("terminate",))

    async def fetchrow(
        self, sql: str, *parameters: object, **_options: object
    ) -> object:
        self.db.log.append(("table", self.readonly))
        return self.db.guardrails

    async def fetch(self, sql: str, *parameters: object, **_options: object) -> object:
        self.db.log.append(("fetch", sql, parameters))
        if "daita:postgresql.write_grants" in sql:
            guardrails = self.db.guardrails
            return [
                {
                    "column_name": column[0],
                    "can_update": guardrails["can_update_columns"],
                    "can_insert": guardrails["can_insert_columns"],
                }
                for column in json.loads(guardrails["columns"])
            ]
        return ()

    async def prepare(self, sql: str, **_options: object) -> _Statement:
        return _Statement(self, sql)

    async def scan(
        self, sql: str, parameters: tuple[object, ...]
    ) -> AsyncIterator[dict[str, object]]:
        self.db.log.append(("scan", self.readonly))
        if self.db.before_scan is not None:
            self.db.before_scan()
            self.work = deepcopy(self.db.rows)
        tree = sqlglot.parse_one(sql, read="postgres")
        cte = next(item for item in tree.find_all(exp.CTE) if item.alias == "excluded")
        columns = tuple(column.name for column in cte.args["alias"].columns)
        assert isinstance(tree, exp.Select)
        items = {item.alias: item for item in tree.selects}
        assert self.work is not None
        for row in _incoming(cte.find(exp.Values), columns, parameters):
            existing = self.work.get(str(row[KEY_COLUMN]))
            record: dict[str, object] = {
                "__daita_row": None if existing is None else f"(0,{existing['id']})",
                "__daita_version": None if existing is None else existing["__xmin"],
                "__daita_action": (
                    "insert"
                    if existing is None
                    else "update" if "__daita_after_0" in items else "skip"
                ),
                "__daita_key": json.dumps([row[KEY_COLUMN]]),
                "__daita_bound": sha256(
                    json.dumps(row, sort_keys=True).encode()
                ).hexdigest(),
            }
            for alias, item in items.items():
                column = item.find(exp.Column)
                if column is None:
                    continue
                if alias.startswith(("__daita_value_", "__daita_after_")):
                    record[alias] = row[column.name]
                elif alias.startswith("__daita_before_"):
                    record[alias] = None if existing is None else existing[column.name]
            yield record

    async def execute(self, sql: str, *parameters: object, **_options: object) -> str:
        self.db.log.append(("execute", sql, parameters))
        if "FOR UPDATE" in sql:
            if self.db.lock_error:
                raise self.db.lock_error
            return "SELECT 0"
        return "SELECT 1"

    async def mutate(
        self, sql: str, parameters: tuple[object, ...]
    ) -> AsyncIterator[dict[str, object]]:
        self.db.log.append(("mutation", sql, parameters))
        if self.db.mutation_error:
            raise self.db.mutation_error
        tree = sqlglot.parse_one(sql, read="postgres")
        columns = tuple(item.name for item in tree.this.expressions)
        conflict = tree.args["conflict"]
        assignments = tuple(item.this.name for item in conflict.expressions)
        assert self.work is not None
        for row in _incoming(tree.expression, columns, parameters):
            key = str(row[KEY_COLUMN])
            existing = self.work.get(key)
            if existing is None:
                self.work[key] = {
                    "id": self.db.next_id,
                    "notes": None,
                    **row,
                    "__xmin": "1",
                }
                self.db.next_id += 1
            elif assignments:
                existing.update({name: row[name] for name in assignments}, __xmin="2")
            else:
                continue
            if self.db.bad_count:
                continue
            yield {
                "__daita_key": json.dumps(["other" if self.db.bad_key else key]),
                "__daita_inserted": existing is None and not self.db.not_an_insert,
            }
