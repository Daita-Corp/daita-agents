"""
Plugin resolution — map a connection string to a BaseDatabasePlugin instance.
"""

from typing import Tuple, TYPE_CHECKING, Union
from urllib.parse import urlparse

if TYPE_CHECKING:
    from ...plugins.base_db import BaseDatabasePlugin


def extract_sslmode(connection_string: str) -> str:
    """Extract sslmode from DSN query params, mapping to CatalogPlugin ssl_mode values."""
    try:
        from urllib.parse import parse_qs
        qs = parse_qs(urlparse(connection_string).query)
        raw = qs.get("sslmode", [None])[0]
        if raw in ("disable", "allow", "prefer"):
            return "disable"
        if raw == "require":
            return "require"
    except Exception:
        pass
    return "verify-full"


def resolve_plugin(
    source: Union[str, "BaseDatabasePlugin"],
    read_only: bool = True,
) -> Tuple["BaseDatabasePlugin", bool]:
    """
    Resolve *source* to a :class:`BaseDatabasePlugin` instance.

    Returns:
        ``(plugin, was_created)`` — ``was_created=True`` means the factory created
        the plugin and is responsible for cleanup on error.

    Raises:
        ValueError: For unsupported schemes, malformed URLs, or missing DB name.
    """
    from ...plugins.base_db import BaseDatabasePlugin

    if isinstance(source, BaseDatabasePlugin):
        return source, False

    if not isinstance(source, str):
        raise ValueError(
            f"source must be a connection string or BaseDatabasePlugin, got {type(source).__name__}"
        )

    # Bare file path detection: no scheme, ends with a SQLite extension or :memory:
    if "://" not in source:
        if (
            source == ":memory:"
            or source.endswith(".db")
            or source.endswith(".sqlite")
            or source.endswith(".sqlite3")
        ):
            from ...plugins.sqlite import SQLitePlugin
            return SQLitePlugin(path=source, read_only=read_only), True
        raise ValueError(
            f"Unsupported source: {source!r}. "
            "Pass a connection string with a scheme (postgresql://, mysql://, "
            "mongodb://, sqlite://) or a .db/.sqlite/.sqlite3 file path."
        )

    parsed = urlparse(source)
    scheme = parsed.scheme.lower()

    if scheme in ("postgresql", "postgres"):
        from ...plugins.postgresql import PostgreSQLPlugin
        return PostgreSQLPlugin(connection_string=source, read_only=read_only), True

    if scheme == "mysql":
        from ...plugins.mysql import MySQLPlugin
        return MySQLPlugin(connection_string=source, read_only=read_only), True

    if scheme in ("mongodb", "mongodb+srv"):
        db_name = parsed.path.lstrip("/") if parsed.path else ""
        if not db_name:
            raise ValueError(
                "MongoDB connection string must include a database name in the URL path, "
                "e.g. mongodb://host/mydb"
            )
        from ...plugins.mongodb import MongoDBPlugin
        return MongoDBPlugin(connection_string=source, read_only=read_only), True

    if scheme == "sqlite":
        # sqlite:///abs/path  ->  parsed.path = "/abs/path"
        # sqlite://:memory:   ->  parsed.netloc = ":memory:"
        path = parsed.path or parsed.netloc or ":memory:"
        from ...plugins.sqlite import SQLitePlugin
        return SQLitePlugin(path=path, read_only=read_only), True

    if scheme == "snowflake":
        raise ValueError(
            "Snowflake connection strings are not supported. "
            "Pass a SnowflakePlugin instance directly."
        )

    raise ValueError(
        f"Unsupported scheme: {scheme!r}. "
        "Supported: postgresql, mysql, mongodb, sqlite"
    )
