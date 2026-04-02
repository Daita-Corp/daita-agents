"""
GitHub repository scanner for connection string discovery.

Scans repository files for database connection strings, ORM configurations,
and infrastructure definitions to discover data stores.
"""

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from .base_discoverer import BaseDiscoverer, DiscoveredStore

logger = logging.getLogger(__name__)

# Patterns for connection string extraction with confidence scores
_CONN_PATTERNS: List[Tuple[str, re.Pattern, float]] = [
    # Direct connection URLs
    (
        "postgresql",
        re.compile(
            r"(?:postgres(?:ql)?|pg)://[^\s\"'`\}]+",
            re.IGNORECASE,
        ),
        0.9,
    ),
    (
        "mysql",
        re.compile(
            r"mysql(?:\+\w+)?://[^\s\"'`\}]+",
            re.IGNORECASE,
        ),
        0.9,
    ),
    (
        "mongodb",
        re.compile(
            r"mongodb(?:\+srv)?://[^\s\"'`\}]+",
            re.IGNORECASE,
        ),
        0.9,
    ),
    (
        "redis",
        re.compile(
            r"redis(?:s)?://[^\s\"'`\}]+",
            re.IGNORECASE,
        ),
        0.9,
    ),
    # ORM / framework config patterns (lower confidence)
    (
        "postgresql",
        re.compile(
            r"""(?:DATABASE_URL|POSTGRES_URL|PG_CONNECTION)\s*[=:]\s*['"]?([^\s'"]+)""",
            re.IGNORECASE,
        ),
        0.7,
    ),
    (
        "mysql",
        re.compile(
            r"""(?:MYSQL_URL|MYSQL_CONNECTION)\s*[=:]\s*['"]?([^\s'"]+)""",
            re.IGNORECASE,
        ),
        0.7,
    ),
]

# Files to scan for connection strings
_TARGET_FILES = [
    ".env",
    ".env.example",
    ".env.sample",
    "docker-compose.yml",
    "docker-compose.yaml",
    "docker-compose.*.yml",
    "database.yml",
    "config/database.yml",
    "prisma/schema.prisma",
    "dbt_project.yml",
    "profiles.yml",
    "alembic.ini",
]


class GitHubScanner(BaseDiscoverer):
    """
    Scan GitHub repositories for connection strings and infrastructure config.

    Searches for database URLs in .env files, Docker Compose files, ORM
    configurations, and other common config locations.

    Lazy imports httpx.
    """

    name = "github"

    def __init__(
        self,
        token: Optional[str] = None,
        repos: Optional[List[str]] = None,
        org: Optional[str] = None,
    ):
        """
        Args:
            token: GitHub personal access token. Falls back to GITHUB_TOKEN env var.
            repos: List of "owner/repo" strings to scan. If None, scans all org repos.
            org: GitHub organization to scan. Required if repos is not set.
        """
        self._token = token
        self._repos = repos or []
        self._org = org
        self._client = None

    async def authenticate(self) -> None:
        """Set up httpx client with GitHub auth headers."""
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required. Install with: pip install httpx"
            )
        import os

        token = self._token or os.environ.get("GITHUB_TOKEN")
        if not token:
            raise ValueError(
                "GitHub token required. Set GITHUB_TOKEN env var or pass token= argument."
            )

        self._client = httpx.AsyncClient(
            base_url="https://api.github.com",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github.v3+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30.0,
        )

    async def enumerate(self) -> AsyncIterator[DiscoveredStore]:
        """Scan configured repos for connection strings."""
        if not self._client:
            await self.authenticate()

        repos = self._repos
        if not repos and self._org:
            repos = await self._list_org_repos()

        for repo in repos:
            async for store in self._scan_repo(repo):
                yield store

    def fingerprint(self, store: DiscoveredStore) -> str:
        """Hash based on the extracted connection string."""
        conn_str = store.connection_hint.get("connection_string", "")
        # Redact credentials before fingerprinting
        redacted = re.sub(r"://[^@]*@", "://***@", conn_str)
        raw = f"github|{store.store_type}|{redacted}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    async def test_access(self) -> bool:
        """Verify GitHub token is valid."""
        try:
            await self.authenticate()
            resp = await self._client.get("/user")
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Close httpx client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _list_org_repos(self) -> List[str]:
        """List all repos in the configured organization."""
        repos = []
        page = 1
        while True:
            resp = await self._client.get(
                f"/orgs/{self._org}/repos",
                params={"per_page": 100, "page": page, "type": "all"},
            )
            if resp.status_code != 200:
                logger.warning("Failed to list org repos: %s", resp.text)
                break
            data = resp.json()
            if not data:
                break
            repos.extend(r["full_name"] for r in data)
            page += 1
        return repos

    async def _scan_repo(self, repo: str) -> AsyncIterator[DiscoveredStore]:
        """Scan a single repo for connection strings."""
        now = datetime.now(timezone.utc).isoformat()

        for file_path in _TARGET_FILES:
            content = await self._get_file_content(repo, file_path)
            if not content:
                continue

            for store_type, pattern, base_confidence in _CONN_PATTERNS:
                for match in pattern.finditer(content):
                    conn_str = match.group(0)

                    # Reduce confidence for template/placeholder values
                    confidence = base_confidence
                    if any(
                        placeholder in conn_str.lower()
                        for placeholder in [
                            "${", "{{", "<your", "example.com", "localhost",
                            "your_", "change_me", "password_here",
                        ]
                    ):
                        confidence = max(0.3, base_confidence - 0.4)

                    store = DiscoveredStore(
                        id="",
                        store_type=store_type,
                        display_name=f"{store_type} in {repo} ({file_path})",
                        connection_hint={
                            "connection_string": conn_str,
                            "source_file": file_path,
                            "repo": repo,
                        },
                        source="github_scan",
                        confidence=confidence,
                        tags=[f"repo:{repo}", f"file:{file_path}"],
                        metadata={
                            "repo": repo,
                            "file_path": file_path,
                            "match_context": content[
                                max(0, match.start() - 50) : match.end() + 50
                            ],
                        },
                        discovered_at=now,
                        last_seen=now,
                    )
                    store.id = self.fingerprint(store)
                    yield store

    async def _get_file_content(self, repo: str, path: str) -> Optional[str]:
        """Fetch file content from GitHub. Returns None if not found."""
        try:
            resp = await self._client.get(
                f"/repos/{repo}/contents/{path}",
                headers={"Accept": "application/vnd.github.v3.raw"},
            )
            if resp.status_code == 200:
                return resp.text
        except Exception as exc:
            logger.debug("Failed to fetch %s/%s: %s", repo, path, exc)
        return None
