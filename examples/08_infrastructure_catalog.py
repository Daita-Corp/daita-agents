"""Infrastructure catalog discovery through the DB runtime.

Run:
    python examples/08_infrastructure_catalog.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any, AsyncIterator

from daita.agents.agent import Agent
from daita.db import DbMemoryConfig, DbSourceOptions
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.catalog.base_discoverer import BaseDiscoverer, DiscoveredStore

from local_sqlite_fixtures import temporary_sales_sqlite

CATALOG_DISCOVERY_CAPABILITY = "catalog.infrastructure.discover"


class LocalDryRunDiscoverer(BaseDiscoverer):
    """Credential-free fixture discoverer for the local example database."""

    name = "local_dry_run"

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)

    async def enumerate(self) -> AsyncIterator[DiscoveredStore]:
        store = DiscoveredStore(
            id="local-sqlite-sales",
            store_type="sqlite",
            display_name="Local sales SQLite fixture",
            connection_hint={"path": self._db_path.name},
            source="local_dry_run",
            region="local",
            environment="dev",
            confidence=1.0,
            tags=["example", "dry-run"],
            metadata={"credential_mode": "none"},
        )
        yield store


def configure_catalog(db_path: Path, args: argparse.Namespace) -> CatalogPlugin:
    catalog = CatalogPlugin(auto_persist=False)
    catalog.add_discoverer(LocalDryRunDiscoverer(db_path))

    if args.aws:
        if os.getenv("AWS_PROFILE") or os.getenv("AWS_ACCESS_KEY_ID"):
            from daita.plugins.catalog.aws import AWSDiscoverer

            catalog.add_discoverer(
                AWSDiscoverer(
                    regions=csv_env("AWS_REGIONS", ["us-east-1"]),
                    role_arn=os.getenv("AWS_ROLE_ARN"),
                    profile_name=os.getenv("AWS_PROFILE"),
                    services=csv_env("AWS_SERVICES", ["rds", "dynamodb", "s3"]),
                )
            )
        else:
            print("Skipping AWS discovery: set AWS_PROFILE or AWS_ACCESS_KEY_ID.")

    if args.github:
        repos = csv_env("GITHUB_REPOS", [])
        org = os.getenv("GITHUB_ORG")
        if os.getenv("GITHUB_TOKEN") and (repos or org):
            from daita.plugins.catalog.github import GitHubScanner

            catalog.add_discoverer(
                GitHubScanner(
                    token=os.getenv("GITHUB_TOKEN"),
                    repos=repos,
                    org=org,
                )
            )
        else:
            print(
                "Skipping GitHub discovery: set GITHUB_TOKEN and "
                "GITHUB_REPOS or GITHUB_ORG."
            )

    return catalog


def csv_env(key: str, default: list[str]) -> list[str]:
    value = os.getenv(key, "")
    return [item.strip() for item in value.split(",") if item.strip()] or default


def catalog_capability_ids(inspection) -> list[str]:
    return [
        capability_id
        for capability_id in inspection.capability_ids
        if capability_id.startswith("catalog:")
    ]


def task_capability_sequence(snapshot) -> list[str]:
    return [task.capability_id for task in snapshot.tasks]


def evidence_kinds(evidence) -> list[str]:
    return [item.kind for item in evidence]


def sanitized_hint(hint: dict[str, Any]) -> dict[str, Any]:
    blocked = {"password", "secret", "token", "access_key", "secret_key"}
    return {
        key: value
        for key, value in hint.items()
        if not any(marker in key.lower() for marker in blocked)
    }


def print_discovered_store_summary(inventory: dict[str, Any]) -> None:
    stores = inventory.get("stores") or []
    by_environment: dict[str, dict[str, int]] = {}
    for store in stores:
        environment = str(store.get("environment") or "unknown")
        store_type = str(store.get("store_type") or "unknown")
        by_environment.setdefault(environment, {})
        by_environment[environment][store_type] = (
            by_environment[environment].get(store_type, 0) + 1
        )

    print("Discovered stores")
    print(f"  count: {inventory.get('store_count', 0)}")
    print(f"  errors: {inventory.get('error_count', 0)}")
    for environment, counts in sorted(by_environment.items()):
        summary = ", ".join(
            f"{store_type}={count}" for store_type, count in sorted(counts.items())
        )
        print(f"  {environment}: {summary}")

    for store in stores[:5]:
        print(
            "  - "
            f"{store.get('display_name')} "
            f"({store.get('store_type')}, {store.get('source')}) "
            f"hint={sanitized_hint(dict(store.get('connection_hint') or {}))}"
        )
    print()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Initialize Agent.from_db() and print catalog declarations only.",
    )
    parser.add_argument(
        "--aws",
        action="store_true",
        help="Opt into AWS discovery when AWS environment credentials are set.",
    )
    parser.add_argument(
        "--github",
        action="store_true",
        help="Opt into GitHub scanning when GITHUB_TOKEN and target env vars are set.",
    )
    args = parser.parse_args()

    async with temporary_sales_sqlite() as db_path:
        catalog = configure_catalog(db_path, args)
        agent = await Agent.from_db(
            str(db_path),
            name="InfrastructureCatalog",
            catalog=catalog,
            source_options=DbSourceOptions(cache_ttl=0),
            memory=DbMemoryConfig(enabled=False),
        )
        try:
            inspection = await agent.describe()
            print(f"SQLite fixture: {db_path}")
            print(f"Registered plugins: {', '.join(inspection.plugin_ids)}")
            print("Relevant catalog capabilities:")
            for capability_id in catalog_capability_ids(inspection):
                print(f"  - {capability_id}")
            print()

            if args.setup_only:
                return

            evidence = await agent.runtime.execute_capability(
                CATALOG_DISCOVERY_CAPABILITY,
                owner="catalog",
                operation_type="infrastructure.discover",
                input={"concurrency": 2},
            )
            operation_id = evidence[0].operation_id if evidence else None
            inventory = evidence[0].payload if evidence else {}
            snapshot = (
                await agent.runtime.inspect_operation(operation_id)
                if operation_id is not None
                else None
            )

            print("Discovery operation")
            print(f"  id: {operation_id}")
            if snapshot is not None:
                print(f"  status: {snapshot.operation.status.value}")
                print("  task capability sequence:")
                for capability_id in task_capability_sequence(snapshot):
                    print(f"    - {capability_id}")
            print("  evidence kinds:")
            for kind in evidence_kinds(evidence):
                print(f"    - {kind}")
            print()

            print_discovered_store_summary(inventory)
        finally:
            await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
