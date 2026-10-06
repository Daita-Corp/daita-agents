"""Export persistence APIs without loading an implementation for error imports."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sqlite import SQLiteStateStore


def __getattr__(name: str):
    if name == "SQLiteStateStore":
        from .sqlite import SQLiteStateStore

        globals()[name] = SQLiteStateStore
        return SQLiteStateStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["SQLiteStateStore"]
