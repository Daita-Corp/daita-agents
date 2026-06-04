"""
Infrastructure Catalog Agent

Configures catalog discovery across AWS accounts and GitHub repositories, then
answers questions about the organization's data landscape from runtime discovery
inventory, catalog ToolViews, and memory context.

Use case: "What production databases do we have?" -> the runner invokes
Catalog's infrastructure discovery capability, passes the inventory to the
agent for summarization, and the memory plugin can contribute runtime context
when configured.
"""

import json
import os
from typing import Any, Optional

from daita import Agent
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.catalog.aws import AWSDiscoverer
from daita.plugins.catalog.github import GitHubScanner
from daita.plugins.catalog.profiler import (
    PostgresProfiler,
    MySQLProfiler,
    DynamoDBProfiler,
    S3Profiler,
    APIGatewayProfiler,
)
from daita.plugins.memory import MemoryPlugin

CATALOG_INFRASTRUCTURE_DISCOVERY_CAPABILITY = "catalog.infrastructure.discover"


def create_agent(
    aws_regions: Optional[list] = None,
    aws_role_arn: Optional[str] = None,
    github_org: Optional[str] = None,
    github_repos: Optional[list] = None,
    enable_memory: bool = True,
) -> Agent:
    """
    Create the Infrastructure Catalog Agent.

    Configures AWS and GitHub discoverers based on environment variables and
    optional overrides. Registers profilers for PostgreSQL and MySQL so the
    agent can inspect cataloged schemas after discovery. Optionally adds memory
    as runtime context for repeated analysis.

    Args:
        aws_regions: AWS regions to scan. Defaults to AWS_REGIONS env var
                     (comma-separated) or ["us-east-1"].
        aws_role_arn: IAM role ARN for cross-account access. Defaults to
                      AWS_ROLE_ARN env var.
        github_org: GitHub org to scan. Defaults to GITHUB_ORG env var.
        github_repos: Specific repos ("owner/repo"). Defaults to GITHUB_REPOS
                      env var (comma-separated).
        enable_memory: Enable memory context and runtime memory capabilities
                       (default: True).
    """
    # --- Catalog plugin ---
    catalog = CatalogPlugin(auto_persist=True)

    # --- AWS discoverer ---
    regions = aws_regions or _csv_env("AWS_REGIONS", ["us-east-1"])
    role_arn = aws_role_arn or os.getenv("AWS_ROLE_ARN")
    services = _csv_env(
        "AWS_SERVICES",
        ["rds", "dynamodb", "s3", "elasticache", "redshift", "apigateway"],
    )

    aws = AWSDiscoverer(
        regions=regions,
        role_arn=role_arn,
        profile_name=os.getenv("AWS_PROFILE"),
        services=services,
    )
    catalog.add_discoverer(aws)

    # --- GitHub scanner ---
    org = github_org or os.getenv("GITHUB_ORG")
    repos = github_repos or _csv_env("GITHUB_REPOS", [])

    if org or repos:
        github = GitHubScanner(
            token=os.getenv("GITHUB_TOKEN"),
            org=org,
            repos=repos,
        )
        catalog.add_discoverer(github)

    # --- Profilers (for schema drill-down after discovery) ---
    catalog.add_profiler(PostgresProfiler())
    catalog.add_profiler(MySQLProfiler())
    catalog.add_profiler(DynamoDBProfiler())
    catalog.add_profiler(S3Profiler())
    catalog.add_profiler(APIGatewayProfiler())

    # --- Memory plugin ---
    plugins = [catalog]
    if enable_memory:
        memory = MemoryPlugin(
            workspace="infrastructure-catalog",
            scope="project",
            enable_fact_extraction=bool(
                os.getenv("MEMORY_CURATION_PROVIDER")
                or os.getenv("DAITA_MEMORY_CURATOR_MODULE")
            ),
            curation_provider=os.getenv("MEMORY_CURATION_PROVIDER"),
            curation_model=os.getenv("MEMORY_CURATION_MODEL"),
        )
        plugins.append(memory)

    return Agent(
        name="Infrastructure Catalog Agent",
        model="gpt-4o-mini",
        prompt=_AGENT_PROMPT,
        plugins=plugins,
    )


async def discover_infrastructure(
    agent: Agent, *, concurrency: int = 5
) -> dict[str, Any]:
    """Run catalog infrastructure discovery through the registry capability."""
    result = await agent.execute_capability(
        CATALOG_INFRASTRUCTURE_DISCOVERY_CAPABILITY,
        {"concurrency": concurrency},
        owner="catalog",
    )
    evidence = result["evidence"][0] if result.get("evidence") else {}
    return dict(evidence.get("payload") or {})


_AGENT_PROMPT = """\
You are an infrastructure analyst. You summarize cataloged data stores \
across cloud providers and code repositories, then answer questions about the \
organization's data landscape from catalog ToolViews and memory context.

## Runtime Discovery
Infrastructure scanning is a runtime capability invoked by the application, \
not a chat tool. When a fresh scan is needed, ask the application to run the \
catalog.infrastructure.discover capability and provide the inventory.

## Catalog Tools
- catalog_search_schema: Search cataloged assets and fields.
- catalog_inspect_asset: Inspect one bounded catalog asset.
- catalog_find_relationship_paths: Find relationships between catalog assets.

## Memory Context
The memory plugin contributes bounded runtime context when configured. Memory \
read/write capabilities are runtime-only; do not call memory tool names.

## Workflow
1. When the application provides a fresh discovery inventory, summarize it \
clearly and group stores by environment when possible.
2. Use catalog_search_schema to filter catalog results.
3. If the user wants schema details, use catalog_inspect_asset with the store ID.
4. Use catalog_find_relationship_paths for relationship or dependency questions.
5. Present results as clean tables. Group by environment when showing overviews.

## Rules
- Treat the runtime discovery inventory as the source of a fresh scan.
- Never expose connection credentials — use display names and store IDs.
- When comparing environments, highlight differences clearly.
- For large result sets, summarize first, then offer to show details."""


def _csv_env(key: str, default: list) -> list:
    """Read a comma-separated env var into a list."""
    val = os.getenv(key, "")
    return [v.strip() for v in val.split(",") if v.strip()] if val else default


if __name__ == "__main__":
    import asyncio

    async def main():
        agent = create_agent()
        await agent.start()

        try:
            # First run: discover through the runtime capability, then summarize.
            inventory = await discover_infrastructure(agent)
            result = await agent.run(
                "Summarize this runtime discovery inventory. Group stores by "
                "environment when possible.\n\n"
                f"{json.dumps(inventory, default=str)}"
            )
            print(result)

            # Second run: query from memory without re-scanning
            result = await agent.run(
                "What production databases do we have? Check memory first."
            )
            print(result)
        finally:
            await agent.stop()

    asyncio.run(main())
