"""
Redis data store plugin for Daita Agents.

Provides general-purpose key-value, hash, list, and set operations using Redis.
Supports key prefixing for namespace isolation and read-only mode.

Example:
    ```python
    from daita.plugins import redis

    async with redis(url="redis://localhost:6379", key_prefix="myapp:") as r:
        await r.set("user:1", '{"name": "Alice"}', ttl=3600)
        value = await r.get("user:1")
        await r.hset("config", {"theme": "dark", "lang": "en"})
    ```
"""

import logging
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from .base import PluginContext
from .base_db import BaseDatabasePlugin
from .redis_extensions import (
    REDIS_MANIFEST,
    RedisExecutor,
    redis_capabilities,
    redis_evidence_schemas,
    redis_operation_definitions,
    redis_tool_views,
)

if TYPE_CHECKING:
    from ..core.tools import LocalTool

logger = logging.getLogger(__name__)


class RedisPlugin(BaseDatabasePlugin):
    """
    Redis data store plugin for agents.

    Provides key-value, hash, list, and set operations with optional
    key prefixing for namespace isolation.
    """

    manifest = REDIS_MANIFEST

    def __init__(
        self,
        url: Optional[str] = None,
        db: Optional[int] = None,
        password: Optional[str] = None,
        max_connections: int = 10,
        key_prefix: Optional[str] = None,
        **kwargs,
    ):
        self.url = url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.db = db if db is not None else int(os.getenv("REDIS_DB", "0"))
        self.password = password or os.getenv("REDIS_PASSWORD")
        self.max_connections = max_connections
        self.key_prefix = key_prefix or os.getenv("REDIS_KEY_PREFIX", "")

        super().__init__(**kwargs)
        self._executor = RedisExecutor(self)
        logger.debug(f"RedisPlugin configured (url: {self.url}, db: {self.db})")

    async def setup(self, context: PluginContext) -> None:
        """Set up the Redis connector for a runtime."""
        await self.connect()

    async def teardown(self) -> None:
        """Disconnect the Redis connector from a runtime."""
        await self.disconnect()

    # ── Runtime extension declarations ───────────────────────────────

    def declare_capabilities(self):
        return redis_capabilities(self.read_only)

    def get_executors(self):
        return (self._executor,)

    def declare_evidence_schemas(self):
        return redis_evidence_schemas()

    def get_tool_views(self):
        return redis_tool_views(self.read_only)

    def _definition_for_capability(self, capability_id: str) -> dict:
        for definition in redis_operation_definitions(self.read_only):
            if definition["capability_id"] == capability_id:
                return definition
        raise KeyError(capability_id)

    def _definition_for_tool(self, tool_name: str) -> dict:
        for definition in redis_operation_definitions(self.read_only):
            if definition["tool_name"] == tool_name:
                return definition
        raise KeyError(tool_name)

    def _pk(self, key: str) -> str:
        """Prefix a key with the configured namespace."""
        return f"{self.key_prefix}{key}"

    def _upk(self, key: str) -> str:
        """Strip the prefix from a key."""
        if self.key_prefix and key.startswith(self.key_prefix):
            return key[len(self.key_prefix) :]
        return key

    async def _ensure_connected(self):
        if self._client is None:
            await self.connect()

    async def connect(self) -> None:
        if self._client is not None:
            return

        try:
            import redis.asyncio as aioredis
        except ImportError:
            raise ImportError(
                "redis package required for RedisPlugin. "
                "Install with: pip install 'daita-agents[redis]'"
            )

        self._client = aioredis.Redis.from_url(
            self.url,
            db=self.db,
            password=self.password,
            max_connections=self.max_connections,
            decode_responses=True,
        )
        logger.info(f"Connected to Redis at {self.url} (db={self.db})")

    async def disconnect(self) -> None:
        if self._client is None:
            return
        try:
            await self._client.close()
        except Exception as e:
            logger.warning(f"Error closing Redis connection: {e}")
        self._client = None
        logger.info("Disconnected from Redis")

    # ── Key-value operations ──────────────────────────────────────────

    async def get(self, key: str) -> Optional[str]:
        await self._ensure_connected()
        return await self._client.get(self._pk(key))

    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        await self._ensure_connected()
        return await self._client.set(self._pk(key), value, ex=ttl)

    async def delete(self, *keys: str) -> int:
        await self._ensure_connected()
        if not keys:
            return 0
        return await self._client.delete(*(self._pk(k) for k in keys))

    async def exists(self, *keys: str) -> int:
        await self._ensure_connected()
        if not keys:
            return 0
        return await self._client.exists(*(self._pk(k) for k in keys))

    async def keys(self, pattern: str = "*", count: int = 1000) -> List[str]:
        """Scan keys matching pattern. Uses SCAN (non-blocking)."""
        await self._ensure_connected()
        result = []
        async for key in self._client.scan_iter(match=self._pk(pattern), count=100):
            result.append(self._upk(key))
            if len(result) >= count:
                break
        return result

    # ── TTL operations ────────────────────────────────────────────────

    async def ttl(self, key: str) -> int:
        await self._ensure_connected()
        return await self._client.ttl(self._pk(key))

    async def expire(self, key: str, seconds: int) -> bool:
        await self._ensure_connected()
        return await self._client.expire(self._pk(key), seconds)

    # ── Hash operations ───────────────────────────────────────────────

    async def hset(self, key: str, mapping: Dict[str, str]) -> int:
        await self._ensure_connected()
        return await self._client.hset(self._pk(key), mapping=mapping)

    async def hgetall(self, key: str) -> Dict[str, str]:
        await self._ensure_connected()
        return await self._client.hgetall(self._pk(key))

    # ── List operations ───────────────────────────────────────────────

    async def lpush(self, key: str, *values: str) -> int:
        await self._ensure_connected()
        return await self._client.lpush(self._pk(key), *values)

    async def rpush(self, key: str, *values: str) -> int:
        await self._ensure_connected()
        return await self._client.rpush(self._pk(key), *values)

    async def lrange(self, key: str, start: int = 0, stop: int = -1) -> List[str]:
        await self._ensure_connected()
        return await self._client.lrange(self._pk(key), start, stop)

    # ── Set operations ────────────────────────────────────────────────

    async def sadd(self, key: str, *members: str) -> int:
        await self._ensure_connected()
        return await self._client.sadd(self._pk(key), *members)

    async def smembers(self, key: str) -> List[str]:
        await self._ensure_connected()
        return list(await self._client.smembers(self._pk(key)))

    # ── Info ──────────────────────────────────────────────────────────

    async def dbsize(self) -> int:
        await self._ensure_connected()
        return await self._client.dbsize()

    # ── Tools ─────────────────────────────────────────────────────────

    # ── Tool handlers ─────────────────────────────────────────────────

    async def _tool_get(self, args: Dict[str, Any]) -> Dict[str, Any]:
        value = await self.get(args["key"])
        return {"key": args["key"], "value": value}

    async def _tool_keys(self, args: Dict[str, Any]) -> Dict[str, Any]:
        pattern = args.get("pattern", "*")
        count = args.get("count", 1000)
        matched = await self.keys(pattern, count)
        return {"pattern": pattern, "keys": matched, "count": len(matched)}

    async def _tool_exists(self, args: Dict[str, Any]) -> Dict[str, Any]:
        count = await self.exists(*args["keys"])
        return {"existing": count, "total_checked": len(args["keys"])}

    async def _tool_ttl(self, args: Dict[str, Any]) -> Dict[str, Any]:
        seconds = await self.ttl(args["key"])
        return {"key": args["key"], "ttl_seconds": seconds}

    async def _tool_hgetall(self, args: Dict[str, Any]) -> Dict[str, Any]:
        fields = await self.hgetall(args["key"])
        return {"key": args["key"], "fields": fields}

    async def _tool_lrange(self, args: Dict[str, Any]) -> Dict[str, Any]:
        elements = await self.lrange(
            args["key"], args.get("start", 0), args.get("stop", -1)
        )
        return {"key": args["key"], "elements": elements, "count": len(elements)}

    async def _tool_smembers(self, args: Dict[str, Any]) -> Dict[str, Any]:
        members = await self.smembers(args["key"])
        return {"key": args["key"], "members": members, "count": len(members)}

    async def _tool_info(self, args: Dict[str, Any]) -> Dict[str, Any]:
        size = await self.dbsize()
        return {
            "db": self.db,
            "key_count": size,
            "key_prefix": self.key_prefix or "(none)",
        }

    async def _tool_set(self, args: Dict[str, Any]) -> Dict[str, Any]:
        ok = await self.set(args["key"], args["value"], ttl=args.get("ttl"))
        return {"key": args["key"], "ok": bool(ok)}

    async def _tool_delete(self, args: Dict[str, Any]) -> Dict[str, Any]:
        deleted = await self.delete(*args["keys"])
        return {"deleted": deleted}

    async def _tool_hset(self, args: Dict[str, Any]) -> Dict[str, Any]:
        added = await self.hset(args["key"], args["mapping"])
        return {"key": args["key"], "fields_added": added}

    async def _tool_lpush(self, args: Dict[str, Any]) -> Dict[str, Any]:
        length = await self.lpush(args["key"], *args["values"])
        return {"key": args["key"], "list_length": length}

    async def _tool_rpush(self, args: Dict[str, Any]) -> Dict[str, Any]:
        length = await self.rpush(args["key"], *args["values"])
        return {"key": args["key"], "list_length": length}

    async def _tool_sadd(self, args: Dict[str, Any]) -> Dict[str, Any]:
        added = await self.sadd(args["key"], *args["members"])
        return {"key": args["key"], "members_added": added}

    async def _tool_expire(self, args: Dict[str, Any]) -> Dict[str, Any]:
        ok = await self.expire(args["key"], args["seconds"])
        return {"key": args["key"], "ok": bool(ok)}


def redis(**kwargs) -> RedisPlugin:
    """Create a Redis data store plugin instance."""
    return RedisPlugin(**kwargs)
