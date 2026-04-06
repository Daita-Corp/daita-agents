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
from .base_db import BaseDatabasePlugin

if TYPE_CHECKING:
    from ..core.tools import AgentTool

logger = logging.getLogger(__name__)


class RedisPlugin(BaseDatabasePlugin):
    """
    Redis data store plugin for agents.

    Provides key-value, hash, list, and set operations with optional
    key prefixing for namespace isolation.
    """

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
        logger.debug(f"RedisPlugin configured (url: {self.url}, db: {self.db})")

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

    def get_tools(self) -> List["AgentTool"]:
        from ..core.tools import AgentTool

        _common = dict(
            category="database",
            source="plugin",
            plugin_name="Redis",
            timeout_seconds=30,
        )

        tools = [
            AgentTool(
                name="redis_get",
                description="Get the value of a Redis key.",
                parameters={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "Key name"}
                    },
                    "required": ["key"],
                },
                handler=self._tool_get,
                **_common,
            ),
            AgentTool(
                name="redis_keys",
                description="Scan Redis keys matching a glob pattern (e.g. 'user:*').",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern (default: '*')",
                        },
                        "count": {
                            "type": "integer",
                            "description": "Max keys to return (default: 1000)",
                        },
                    },
                    "required": [],
                },
                handler=self._tool_keys,
                **_common,
            ),
            AgentTool(
                name="redis_exists",
                description="Check if one or more Redis keys exist. Returns count of existing keys.",
                parameters={
                    "type": "object",
                    "properties": {
                        "keys": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Keys to check",
                        }
                    },
                    "required": ["keys"],
                },
                handler=self._tool_exists,
                **_common,
            ),
            AgentTool(
                name="redis_ttl",
                description="Get the remaining TTL (seconds) of a Redis key. Returns -1 if no TTL, -2 if key doesn't exist.",
                parameters={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "Key name"}
                    },
                    "required": ["key"],
                },
                handler=self._tool_ttl,
                **_common,
            ),
            AgentTool(
                name="redis_hgetall",
                description="Get all fields and values of a Redis hash.",
                parameters={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "Hash key name"}
                    },
                    "required": ["key"],
                },
                handler=self._tool_hgetall,
                **_common,
            ),
            AgentTool(
                name="redis_lrange",
                description="Get elements from a Redis list by index range.",
                parameters={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "List key name"},
                        "start": {
                            "type": "integer",
                            "description": "Start index (default: 0)",
                        },
                        "stop": {
                            "type": "integer",
                            "description": "Stop index inclusive (default: -1 for all)",
                        },
                    },
                    "required": ["key"],
                },
                handler=self._tool_lrange,
                **_common,
            ),
            AgentTool(
                name="redis_smembers",
                description="Get all members of a Redis set.",
                parameters={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "Set key name"}
                    },
                    "required": ["key"],
                },
                handler=self._tool_smembers,
                **_common,
            ),
            AgentTool(
                name="redis_info",
                description="Get Redis database size and connection info.",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=self._tool_info,
                **_common,
            ),
        ]

        if not self.read_only:
            tools += [
                AgentTool(
                    name="redis_set",
                    description="Set a Redis key-value pair with optional TTL in seconds.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "key": {"type": "string", "description": "Key name"},
                            "value": {
                                "type": "string",
                                "description": "Value to store",
                            },
                            "ttl": {
                                "type": "integer",
                                "description": "Optional TTL in seconds",
                            },
                        },
                        "required": ["key", "value"],
                    },
                    handler=self._tool_set,
                    **_common,
                ),
                AgentTool(
                    name="redis_delete",
                    description="Delete one or more Redis keys. Returns count deleted.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "keys": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Keys to delete",
                            }
                        },
                        "required": ["keys"],
                    },
                    handler=self._tool_delete,
                    **_common,
                ),
                AgentTool(
                    name="redis_hset",
                    description="Set fields on a Redis hash.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "key": {"type": "string", "description": "Hash key name"},
                            "mapping": {
                                "type": "object",
                                "description": "Field-value pairs to set",
                            },
                        },
                        "required": ["key", "mapping"],
                    },
                    handler=self._tool_hset,
                    **_common,
                ),
                AgentTool(
                    name="redis_lpush",
                    description="Push values to the left (head) of a Redis list.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "key": {"type": "string", "description": "List key name"},
                            "values": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Values to push",
                            },
                        },
                        "required": ["key", "values"],
                    },
                    handler=self._tool_lpush,
                    **_common,
                ),
                AgentTool(
                    name="redis_rpush",
                    description="Push values to the right (tail) of a Redis list.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "key": {"type": "string", "description": "List key name"},
                            "values": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Values to push",
                            },
                        },
                        "required": ["key", "values"],
                    },
                    handler=self._tool_rpush,
                    **_common,
                ),
                AgentTool(
                    name="redis_sadd",
                    description="Add members to a Redis set.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "key": {"type": "string", "description": "Set key name"},
                            "members": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Members to add",
                            },
                        },
                        "required": ["key", "members"],
                    },
                    handler=self._tool_sadd,
                    **_common,
                ),
                AgentTool(
                    name="redis_expire",
                    description="Set a TTL (in seconds) on a Redis key.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "key": {"type": "string", "description": "Key name"},
                            "seconds": {
                                "type": "integer",
                                "description": "TTL in seconds",
                            },
                        },
                        "required": ["key", "seconds"],
                    },
                    handler=self._tool_expire,
                    **_common,
                ),
            ]

        return tools

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
