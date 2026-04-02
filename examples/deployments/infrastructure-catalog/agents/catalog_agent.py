"""
Infrastructure Catalog Agent

Discovers data stores across AWS accounts and GitHub repositories, builds a
unified catalog, and answers questions about the organization's data landscape.

Use case: "What production databases do we have?" -> the agent scans AWS RDS
across configured regions and GitHub repos for connection strings, deduplicates
results, and returns a filtered summary.
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


def create_agent(
    aws_regions: Optional[list] = None,
    aws_role_arn: Optional[str] = None,
    github_org: Optional[str] = None,
    github_repos: Optional[list] = None,
) -> Agent:
    """
    Create the Infrastructure Catalog Agent.

    Configures AWS and GitHub discoverers based on environment variables and
    optional overrides. Registers profilers for PostgreSQL and MySQL so the
    agent can drill into schemas after discovery.

    Args:
        aws_regions: AWS regions to scan. Defaults to AWS_REGIONS env var
                     (comma-separated) or ["us-east-1"].
        aws_role_arn: IAM role ARN for cross-account access. Defaults to
                      AWS_ROLE_ARN env var.
        github_org: GitHub org to scan. Defaults to GITHUB_ORG env var.
        github_repos: Specific repos ("owner/repo"). Defaults to GITHUB_REPOS
                      env var (comma-separated).
    """
    # --- Catalog plugin ---
    catalog = CatalogPlugin(auto_persist=True)

    # --- AWS discoverer ---
    regions = aws_regions or _csv_env("AWS_REGIONS", ["us-east-1"])
    role_arn = aws_role_arn or os.getenv("AWS_ROLE_ARN")
    services = _csv_env("AWS_SERVICES", ["rds", "dynamodb", "s3", "elasticache", "redshift", "apigateway"])

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

    return Agent(
        name="Infrastructure Catalog Agent",
        model="gpt-4o-mini",
        prompt="""You are an infrastructure analyst. You discover and catalog data stores \
across cloud providers and code repositories, then answer questions about the \
organization's data landscape.

Your tools:
- discover_infrastructure: Scan AWS and GitHub for data stores. Run this first.
- find_store: Search the catalog by name, type, environment, or tags.
- profile_store: Connect to a discovered store to extract its full schema.
- discover_postgres / discover_mysql: Direct schema discovery with a connection string.
- compare_schemas: Diff two schemas to find structural changes.
- compare_store_to_baseline: Check for schema drift since last scan.
- export_diagram: Generate a Mermaid ER diagram from a schema.

Workflow:
1. When asked about infrastructure, start with discover_infrastructure.
2. Use find_store to filter results by type, environment, or tags.
3. If the user wants schema details, use profile_store with the store ID.
4. Present results as clean tables. Group by environment when showing overviews.
5. Flag any stores with low confidence scores — they may be templates or placeholders.

Rules:
- Always discover before answering "what do we have" questions.
- Never expose connection credentials — use display names and store IDs.
- When comparing environments, highlight differences clearly.
- For large result sets, summarize first, then offer to show details.""",
        tools=[catalog],
    )


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
            result = await agent.run(
                "Discover all our data stores and give me a summary grouped by environment."
            )
            print(result)
        finally:
            await agent.stop()

    asyncio.run(main())
