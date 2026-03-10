"""
Simple logs display for Daita CLI.
Shows deployment history and logs like git log from cloud API.
"""

import os
import aiohttp
from datetime import datetime
from ..utils import find_project_root, get_api_endpoint, _CLI_VERSION


async def show_deployment_logs(environment=None, limit=10, follow=False, verbose=False):
    """Show deployment logs (like git log) from cloud API."""

    # Check API key first
    api_key = os.getenv("DAITA_API_KEY")
    if not api_key:
        from ..utils import show_upgrade_message

        show_upgrade_message()
        return

    # Get current project name for display
    project_root = find_project_root()
    current_project = None
    if project_root:
        import yaml

        config_file = project_root / "daita-project.yaml"
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    config = yaml.safe_load(f)
                    current_project = config.get("name")
            except:
                current_project = None

    # Load deployments from cloud API
    deployments, api_error = await _load_cloud_deployments(environment, limit)

    if api_error:
        print(f" Failed to fetch deployments: {api_error}")
        if verbose:
            print(f"   API Key: {api_key[:20]}...")
            print(
                f"   Endpoint: {os.getenv('DAITA_API_ENDPOINT', 'https://api.daita.ai')}"
            )
        return

    if not deployments:
        if current_project:
            if environment:
                print(f" No deployments found ({current_project}, {environment})")
            else:
                print(f" No deployments found ({current_project})")
            print("   Run 'daita push' to create your first deployment")
        else:
            print(" No deployments found")
            if environment:
                print(f"   No deployments found for environment: {environment}")
            else:
                print("   Run 'daita push' to create your first deployment")
        return

    # Filter by environment if specified (additional client-side filtering)
    if environment:
        deployments = [d for d in deployments if d["environment"] == environment]
        if not deployments:
            print(f" No deployments found for environment: {environment}")
            return

    # Take most recent deployments (API already returns newest first)
    deployments = deployments[:limit]

    # Find the most recent active deployment
    latest_active_deployment = None
    for deployment in deployments:
        if deployment.get("status") == "active":
            latest_active_deployment = deployment
            break

    # Build header with scope indication
    header_parts = []
    if current_project:
        header_parts.append(current_project)
    if environment:
        header_parts.append(environment)

    if header_parts:
        scope_info = f" ({', '.join(header_parts)})"
    elif current_project is None:
        scope_info = " (Organization)"
    else:
        scope_info = ""

    print(f" Deployment History{scope_info}")
    print("")

    for i, deployment in enumerate(deployments):
        is_current = latest_active_deployment and deployment.get(
            "deployment_id"
        ) == latest_active_deployment.get("deployment_id")
        _show_deployment(deployment, verbose, is_latest=is_current)
        if i < len(deployments) - 1:
            print("")

    if follow:
        print(f"\n  Following logs... (Press Ctrl+C to stop)")
        try:
            import asyncio

            while True:
                await asyncio.sleep(5)
                # Check for new deployments
                new_deployments, error = await _load_cloud_deployments(
                    environment, limit
                )
                if error:
                    continue  # Skip this check if API fails

                if len(new_deployments) > len(deployments):
                    latest = new_deployments[-1]
                    print(f"\n New deployment:")
                    _show_deployment(latest, verbose, is_latest=True)
                    deployments = new_deployments
        except KeyboardInterrupt:
            print(f"\n Stopped following logs")


def _show_deployment(deployment, verbose, is_latest=False):
    """Show a single deployment entry."""
    env = deployment["environment"]
    timestamp = deployment["deployed_at"]
    version = deployment.get("version", "1.0.0")
    project = deployment.get("project_name", "Unknown")

    # Format timestamp
    if timestamp:
        # Handle ISO format from API
        if "T" in timestamp:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            time_str = timestamp
    else:
        time_str = "Unknown time"

    # Status indicator
    status = "●" if is_latest else "○"
    current = " (current)" if is_latest else ""

    print(f"{status} {env}: {project} v{version}{current}")
    print(f"    {time_str}")

    if verbose:
        # Show detailed info
        agents = deployment.get("agents_config", [])
        workflows = deployment.get("workflows_config", [])

        if agents:
            agent_names = [agent.get("name", "unknown") for agent in agents]
            print(f"    Agents: {', '.join(agent_names)}")
        if workflows:
            workflow_names = [wf.get("name", "unknown") for wf in workflows]
            print(f"    Workflows: {', '.join(workflow_names)}")

        # Show deployment ID
        deploy_id = deployment.get("deployment_id", "N/A")
        if len(deploy_id) > 8:
            deploy_id = deploy_id[:8]  # Show first 8 characters
        print(f"    ID: {deploy_id}")

        # Show status
        status = deployment.get("status", "unknown")
        print(f"    Status: {status}")


async def _load_cloud_deployments(environment=None, limit=10):
    """Load deployment history from cloud API."""
    try:
        api_key = os.getenv("DAITA_API_KEY")
        if not api_key:
            return [], "API key not found"

        # Get current project name to filter deployments when in project directory
        project_root = find_project_root()
        if project_root:
            import yaml

            config_file = project_root / "daita-project.yaml"
            current_project = None
            if config_file.exists():
                try:
                    with open(config_file, "r") as f:
                        config = yaml.safe_load(f)
                        current_project = config.get("name")
                except:
                    current_project = None
        else:
            current_project = None

        api_endpoint = get_api_endpoint()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": f"Daita-CLI/{_CLI_VERSION}",
        }

        async with aiohttp.ClientSession() as session:
            # Build URL with filters
            url = f"{api_endpoint}/api/v1/deployments/api-key"
            params = {}
            if environment:
                params["environment"] = environment
            if current_project:
                params["project_name"] = current_project
            if limit and limit != 10:
                params["per_page"] = min(limit, 100)  # API limit

            async with session.get(
                url, headers=headers, params=params, timeout=30
            ) as response:
                if response.status == 200:
                    data = await response.json()

                    # Handle paginated response from API (like in status.py)
                    if isinstance(data, dict) and "deployments" in data:
                        deployments = data["deployments"]
                    else:
                        deployments = data if isinstance(data, list) else []

                    # Return deployments in API format (no conversion needed)
                    return deployments, None
                elif response.status == 401:
                    return [], "Invalid API key"
                elif response.status == 403:
                    return [], "Access denied - check API key permissions"
                else:
                    error_text = await response.text()
                    return [], f"API error {response.status}: {error_text[:100]}"
    except Exception as e:
        if "timeout" in str(e).lower():
            return [], "Request timeout - check your connection"
        elif "dns" in str(e).lower() or "name" in str(e).lower():
            return [], f"Cannot resolve API endpoint - check DAITA_API_ENDPOINT setting"
        else:
            return [], f"Network error: {str(e)}"


def _get_deployment_id(deployment):
    """Generate short deployment ID."""
    import hashlib

    content = f"{deployment['deployed_at']}{deployment['environment']}"
    return hashlib.md5(content.encode()).hexdigest()[:8]
