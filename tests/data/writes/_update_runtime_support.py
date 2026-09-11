"""Shared helpers extracted from ``test_update_runtime.py``."""

from __future__ import annotations

from collections.abc import Mapping


class _Transaction:
    def __init__(
        self,
        log: list[tuple[object, ...]],
        *,
        commit_error: BaseException | None = None,
    ) -> None:
        self.log = log
        self.commit_error = commit_error

    async def start(self) -> None:
        self.log.append(("transaction.start",))

    async def commit(self) -> None:
        self.log.append(("transaction.commit",))
        if self.commit_error is not None:
            raise self.commit_error

    async def rollback(self) -> None:
        self.log.append(("transaction.rollback",))


class _Cursor:
    def __init__(self, rows: tuple[Mapping[str, object], ...]) -> None:
        self._iterator = iter(rows)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            raise StopAsyncIteration from None


def _guardrails() -> dict[str, object]:
    return {
        "relation_oid": "16384",
        "relation_kind": "r",
        "is_partition": False,
        "row_level_security": False,
        "force_row_level_security": False,
        "has_inheritance": False,
        "has_user_triggers": False,
        "has_rewrite_rules": False,
        "role_superuser": False,
        "role_bypass_rls": False,
        "role_create_database": False,
        "role_create_role": False,
        "role_replication": False,
        "can_connect": True,
        "can_use_schema": True,
        "can_select_table": True,
        "can_update_columns": True,
    }


class _Connection:
    def __init__(
        self,
        rows: tuple[Mapping[str, object], ...],
        *,
        update_status: str | BaseException = "UPDATE 3",
        commit_error: BaseException | None = None,
    ) -> None:
        self.rows = rows
        self.update_status = update_status
        self.log: list[tuple[object, ...]] = []
        self.transaction_record = _Transaction(self.log, commit_error=commit_error)

    def transaction(self, **kwargs: object):
        self.log.append(("transaction", kwargs))
        return self.transaction_record

    async def execute(self, sql: str, *parameters: object, **kwargs: object):
        self.log.append(("execute", sql, parameters, kwargs))
        if sql.startswith("UPDATE"):
            if isinstance(self.update_status, BaseException):
                raise self.update_status
            return self.update_status
        return "SELECT 1"

    async def fetchrow(self, sql: str, *parameters: object, **kwargs: object):
        self.log.append(("fetchrow", sql, parameters, kwargs))
        return _guardrails()

    async def fetch(self, sql: str, *parameters: object, **kwargs: object):
        self.log.append(("fetch", sql, parameters, kwargs))
        assert sql.startswith("EXPLAIN")
        return ({"QUERY PLAN": ()},)

    def cursor(self, sql: str, *parameters: object):
        self.log.append(("cursor", sql, parameters))
        return _Cursor(self.rows)

    async def close(self) -> None:
        self.log.append(("close",))

    def terminate(self) -> None:
        self.log.append(("terminate",))


def _row(key: int, *, before: object = "active", xmin: str | None = None):
    return {
        "__daita_primary_key_0": key,
        "__daita_before_0": before,
        "__daita_within_preview_limit": True,
        "__daita_xmin": xmin or str(700 + key),
    }
