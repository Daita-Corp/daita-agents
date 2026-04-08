"""
Infrastructure Catalog Agent

Discovers data stores across AWS accounts and GitHub repositories, builds a
unified catalog, and answers questions about the organization's data landscape.
Uses memory to persist discoveries, org rules, and structured facts across scans.

Use case: "What production databases do we have?" -> the agent scans AWS RDS
across configured regions and GitHub repos for connection strings, deduplicates
results, and returns a filtered summary. Memory retains findings so subsequent
questions don't require re-scanning.
"""

import os
from typing import Optional

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
    agent can drill into schemas after discovery. Optionally adds memory for
    persisting discoveries and org rules across scans.

    Args:
        aws_regions: AWS regions to scan. Defaults to AWS_REGIONS env var
                     (comma-separated) or ["us-east-1"].
        aws_role_arn: IAM role ARN for cross-account access. Defaults to
                      AWS_ROLE_ARN env var.
        github_org: GitHub org to scan. Defaults to GITHUB_ORG env var.
        github_repos: Specific repos ("owner/repo"). Defaults to GITHUB_REPOS
                      env var (comma-separated).
        enable_memory: Enable memory plugin for persisting discoveries and
                       org rules (default: True).
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
        tools=plugins,
    )


_AGENT_PROMPT = """\
You are an infrastructure analyst. You discover and catalog data stores \
across cloud providers and code repositories, then answer questions about the \
organization's data landscape.

## Discovery Tools
- discover_infrastructure: Scan AWS and GitHub for data stores. Run this first.
- find_store: Search the catalog by name, type, environment, or tags.
- profile_store: Connect to a discovered store to extract its full schema.
- discover_postgres / discover_mysql: Direct schema discovery with a connection string.
- compare_schemas: Diff two schemas to find structural changes.
- compare_store_to_baseline: Check for schema drift since last scan.
- export_diagram: Generate a Mermaid ER diagram from a schema.

## Memory Tools
- remember: Store important findings for future reference. Use batch mode \
(pass a list) when storing multiple discoveries at once — this is far more \
efficient than individual calls.
    - Use category tags: "store", "rule", "schema", "drift", "observation"
    - Importance guide: 0.9 for org rules, 0.8 for production stores, \
0.6 for staging, 0.4 for dev
- recall: Search memories by meaning. Use since/before to filter by time \
(e.g. since="24h" for recent discoveries).
- list_by_category: List all memories in a category. Use this for exhaustive \
enumeration (e.g. "list all stores", "show all rules").
- list_memories: Check what you know. Use include_stats=True to see category \
counts and coverage before deciding whether to re-scan.
- query_facts: (When fact extraction is enabled) Query structured facts by \
entity, relation, or value. Use for questions like "what stores are in \
us-east-1?" or "what databases are production?".

## Workflow
1. When asked about infrastructure, check memory first (list_memories with \
include_stats=True). If you already have recent discoveries, use them.
2. If memory is empty or stale, run discover_infrastructure and batch-store \
the results with remember (use a list of dicts for batch mode).
3. Use find_store to filter catalog results, recall to search memories.
4. If the user wants schema details, use profile_store with the store ID.
5. Present results as clean tables. Group by environment when showing overviews.

## Rules
- Check memory before re-scanning — avoid redundant AWS API calls.
- After discovery, batch-store key findings (store names, types, regions, \
environments) using remember with a list.
- Store org rules with importance=0.9 and category="rule" so they persist.
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
            # First run: discover and store in memory
            result = await agent.run(
                "Discover all our data stores. Batch-store each discovery in memory "
                "with appropriate categories and importance levels, then summarize."
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
