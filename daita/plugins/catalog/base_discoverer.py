"""
Base discoverer ABC and discovery data models.

Discoverers find data stores via cloud APIs, config scanning, or service
registries — without connecting to the stores themselves.
"""

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional


@dataclass
class DiscoveredStore:
    """A data store found by a discoverer."""

    id: str  # fingerprint (set by discoverer)
    store_type: str  # "postgresql", "mysql", "mongodb", "s3", "dynamodb", "redis"
    display_name: str  # human-readable: "prod-orders (us-east-1)"
    connection_hint: Dict[str, Any]  # host, port, dbname, etc. (NO plaintext passwords)
    source: str  # "aws_rds", "github_scan", "vault", "manual"
    region: Optional[str] = None
    environment: Optional[str] = None  # "production", "staging", "dev" (inferred or tagged)
    confidence: float = 0.5
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    discovered_at: Optional[str] = None  # ISO timestamp
    last_seen: Optional[str] = None  # updated on each re-scan


@dataclass
class DiscoveryError:
    """An error from a single discoverer during a discovery sweep."""

    discoverer_name: str
    error: str
    exception_type: str


@dataclass
class DiscoveryResult:
    """Aggregated result from discover_all()."""

    stores: List[DiscoveredStore] = field(default_factory=list)
    errors: List[DiscoveryError] = field(default_factory=list)

    @property
    def store_count(self) -> int:
        return len(self.stores)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0


class BaseDiscoverer(ABC):
    """
    Abstract base class for infrastructure discoverers.

    Discoverers enumerate data stores from a specific source (AWS, GCP,
    GitHub, Vault, etc.) without connecting to the stores themselves.
    """

    name: str = "base"

    async def authenticate(self) -> None:
        """Set up / refresh credentials. Called before enumerate()."""
        pass

    @abstractmethod
    async def enumerate(self) -> AsyncIterator[DiscoveredStore]:
        """Yield discovered stores. Handles pagination internally."""
        ...

    def fingerprint(self, store: DiscoveredStore) -> str:
        """
        Generate a dedup key for a discovered store.

        Override per discoverer type for appropriate identity semantics.
        Default: sha256(store_type + source + connection_hint_values)[:16]
        """
        parts = [store.store_type, store.source]
        for key in sorted(store.connection_hint.keys()):
            parts.append(f"{key}={store.connection_hint[key]}")
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    async def test_access(self) -> bool:
        """Verify credentials/access before full enumeration."""
        return True

    async def close(self) -> None:
        """Release resources (sessions, connections)."""
        pass
