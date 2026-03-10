"""
Simple status display for Daita CLI.
Shows project and deployment status like git status.
"""

import json
import yaml
from ..utils import find_project_root, get_api_endpoint, _CLI_VERSION


async def show_project_status(environment=None, verbose=False):
    """Show project status (like git status)."""

    # Find project root
    project_root = find_project_root()
    if not project_root:
        raise ValueError("Not in a Daita project. Run 'daita init' first.")

    # Load project config
    config = _load_project_config(project_root)
    if not config:
        print(" No daita-project.yaml found")
        return

    project_name = config.get("name", "Unknown")
    version = config.get("version", "1.0.0")

    print(f" Project: {project_name} (v{version})")
    print(f"Location: {project_root}")
    print("")

    # Show components status
    _show_components_status(project_root, config)

    # Check if user has DAITA_API_KEY for cloud status
    import os

    has_daita_key = bool(os.getenv("DAITA_API_KEY"))

    if has_daita_key:
        # Show cloud deployment status
        if environment:
            await _show_cloud_environment_status(environment, verbose)
        else:
            await _show_cloud_deployments_status(verbose)
    else:
        # Show local deployment status only
        print(" Cloud Deployments: Upgrade required")
        print("   Get your API key at daita-tech.io")
        print("   Local deployment history:")
        _show_local_deployments_status(project_root, verbose)

    # Show any issues
    _show_issues(project_root, config)


def _show_components_status(project_root, config):
    """Show status of agents and workflows."""
    print(" Components:")

    # Check agents
    agents = config.get("agents", [])
    agents_dir = project_root / "agents"

    # Fallback to filesystem scanning if config is empty
    if not agents and agents_dir.exists():
        agent_files = list(agents_dir.glob("*.py"))
        agent_files = [f for f in agent_files if f.name != "__init__.py"]

        if agent_files:
            print(f"   Agents ({len(agent_files)}) [detected from filesystem]:")
            for agent_file in agent_files:
                agent_name = agent_file.stem
                print(f"      {agent_name} (not in config)")
        else:
            print(f"   Agents: None")
    elif agents:
        print(f"   Agents ({len(agents)}):")
        for agent in agents:
            agent_file = agents_dir / f"{agent['name']}.py"
            status = "" if agent_file.exists() else ""
            display_name = agent.get("display_name", agent["name"])
            print(f"     {status} {agent['name']} → '{display_name}'")
    else:
        print(f"   Agents: None")

    # Check workflows
    workflows = config.get("workflows", [])
    workflows_dir = project_root / "workflows"

    # Fallback to filesystem scanning if config is empty
    if not workflows and workflows_dir.exists():
        workflow_files = list(workflows_dir.glob("*.py"))
        workflow_files = [f for f in workflow_files if f.name != "__init__.py"]

        if workflow_files:
            print(f"   Workflows ({len(workflow_files)}) [detected from filesystem]:")
            for workflow_file in workflow_files:
                workflow_name = workflow_file.stem
                print(f"      {workflow_name} (not in config)")
        else:
            print(f"   Workflows: None")
    elif workflows:
        print(f"   Workflows ({len(workflows)}):")
        for workflow in workflows:
            workflow_file = workflows_dir / f"{workflow['name']}.py"
            status = "" if workflow_file.exists() else ""
            display_name = workflow.get("display_name", workflow["name"])
            print(f"     {status} {workflow['name']} → '{display_name}'")
    else:
        print(f"   Workflows: None")

    print("")


def _show_environment_status(project_root, environment, verbose):
    """Show status for specific environment."""
    print(f" Environment: {environment}")

    # Load deployment history
    deployments = _load_deployments(project_root)
    env_deployments = [d for d in deployments if d["environment"] == environment]

    if env_deployments:
        latest = env_deployments[-1]
        deploy_time = latest["timestamp"][:19].replace("T", " ")

        print(f"   Last deployed: {deploy_time}")
        print(f"   Version: {latest.get('version', 'unknown')}")

        if verbose:
            print(f"   Agents: {', '.join(latest.get('agents', []))}")
            print(f"   Workflows: {', '.join(latest.get('workflows', []))}")
    else:
        print(f"   Never deployed to {environment}")

    print("")


def _show_local_deployments_status(project_root, verbose):
    """Show status of local deployment history."""
    deployments = _load_deployments(project_root)

    if not deployments:
        print("    No local deployment history")
        print("")
        return

    # Group by environment
    env_deployments = {}
    for deployment in deployments:
        env = deployment["environment"]
        if env not in env_deployments:
            env_deployments[env] = []
        env_deployments[env].append(deployment)

    # Show each environment
    for env, deps in env_deployments.items():
        latest = deps[-1]
        deploy_time = latest["timestamp"][:16].replace("T", " ")
        version = latest.get("version", "?")

        print(f"   {env}: v{version} ({deploy_time})")

        if verbose:
            print(f"      {len(latest.get('agents', []))} agents")
            print(f"      {len(latest.get('workflows', []))} workflows")

    print("")


async def _show_cloud_deployments_status(verbose):
    """Show status of cloud deployments via API."""
    import os
    import aiohttp

    # Get current project name to filter deployments when in project directory
    project_root = find_project_root()
    config = _load_project_config(project_root) if project_root else None
    current_project = config.get("name") if config else None

    try:
        api_key = os.getenv("DAITA_API_KEY")
        api_endpoint = get_api_endpoint()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": f"Daita-CLI/{_CLI_VERSION}",
        }

        async with aiohttp.ClientSession() as session:
            url = f"{api_endpoint}/api/v1/deployments/api-key"
            params = {}
            if current_project:
                params["project_name"] = current_project

            async with session.get(
                url, headers=headers, params=params, timeout=10
            ) as response:
                if response.status == 200:
                    data = await response.json()

                    # Handle paginated response from API
                    if isinstance(data, dict) and "deployments" in data:
                        deployments = data["deployments"]
                    else:
                        deployments = data if isinstance(data, list) else []

                    if not deployments:
                        if current_project:
                            print(f" Cloud Deployments ({current_project}): None")
                            print("   Run 'daita push' to deploy")
                        else:
                            print(" Cloud Deployments: None")
                            print("   Run 'daita push' to deploy")
                        print("")
                        return

                    # Update header to indicate scope
                    if current_project:
                        print(
                            f" Cloud Deployments ({current_project}) ({len(deployments)}):"
                        )
                    else:
                        print(
                            f" Cloud Deployments (Organization) ({len(deployments)}):"
                        )

                    # Show most recent 5 deployments (API returns newest first)
                    recent_deployments = deployments[:5]

                    # Find most recent active deployment
                    latest_active = None
                    for deployment in deployments:
                        if deployment.get("status") == "active":
                            latest_active = deployment
                            break

                    for deployment in recent_deployments:
                        env = deployment.get("environment", "unknown")
                        version = deployment.get("version", "?")
                        status = deployment.get("status", "unknown")
                        deployed_at = deployment.get("deployed_at", "")[:16].replace(
                            "T", " "
                        )

                        # Check if this is the current active deployment
                        is_current = latest_active and deployment.get(
                            "deployment_id"
                        ) == latest_active.get("deployment_id")
                        current_text = " (current)" if is_current else ""

                        status_icon = "●" if status == "active" else "○"
                        print(
                            f"  {status_icon} {env}: v{version} ({deployed_at}){current_text}"
                        )

                    print("")
                else:
                    print(" Cloud Deployments: Unable to fetch status")
                    print("   Check your internet connection and API key")
                    print("")

    except Exception:
        print(" Cloud Deployments: Connection failed")
        print("   Using local deployment history")
        print("")


async def _show_cloud_environment_status(environment, verbose):
    """Show cloud status for specific environment."""
    import os
    import aiohttp

    # Get current project name to filter deployments when in project directory
    project_root = find_project_root()
    config = _load_project_config(project_root) if project_root else None
    current_project = config.get("name") if config else None

    try:
        api_key = os.getenv("DAITA_API_KEY")
        api_endpoint = get_api_endpoint()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": f"Daita-CLI/{_CLI_VERSION}",
        }

        async with aiohttp.ClientSession() as session:
            params = {"environment": environment}
            if current_project:
                params["project_name"] = current_project

            url = f"{api_endpoint}/api/v1/deployments/api-key"
            async with session.get(
                url, headers=headers, params=params, timeout=10
            ) as response:
                if response.status == 200:
                    data = await response.json()

                    # Handle paginated response from API
                    if isinstance(data, dict) and "deployments" in data:
                        deployments = data["deployments"]
                    else:
                        deployments = data if isinstance(data, list) else []

                    if not deployments:
                        if current_project:
                            print(f" Environment: {environment} ({current_project})")
                            print(f"    Never deployed to {environment}")
                        else:
                            print(f" Environment: {environment}")
                            print(f"    Never deployed to {environment}")
                        print("")
                        return

                    latest = deployments[-1]
                    status = latest.get("status", "unknown")
                    version = latest.get("version", "?")
                    deployed_at = latest.get("deployed_at", "")[:19].replace("T", " ")

                    status_icon = "" if status == "deployed" else ""
                    if current_project:
                        print(f" Environment: {environment} ({current_project})")
                    else:
                        print(f" Environment: {environment}")
                    print(f"  {status_icon} Status: {status}")
                    print(f"   Last deployed: {deployed_at}")
                    print(f"   Version: {version}")

                    if verbose and "functions" in latest:
                        functions = latest["functions"]
                        print(f"   Functions: {len(functions)}")
                        for func in functions:
                            print(f"     • {func.get('name', 'unknown')}")

                    print("")
                else:
                    print(f" Environment: {environment}")
                    print("   Unable to fetch cloud status")
                    print("")

    except Exception:
        print(f" Environment: {environment}")
        print("   Connection failed - using local history")
        print("")


def _show_issues(project_root, config):
    """Show any issues with the project."""
    issues = []

    # Check for missing files
    required_files = ["daita-project.yaml", "requirements.txt"]
    for file_name in required_files:
        if not (project_root / file_name).exists():
            issues.append(f"Missing {file_name}")

    # Check for missing agent files
    agents = config.get("agents", [])
    for agent in agents:
        agent_file = project_root / "agents" / f"{agent['name']}.py"
        if not agent_file.exists():
            issues.append(f"Missing agent file: {agent['name']}.py")

    # Check for missing workflow files
    workflows = config.get("workflows", [])
    for workflow in workflows:
        workflow_file = project_root / "workflows" / f"{workflow['name']}.py"
        if not workflow_file.exists():
            issues.append(f"Missing workflow file: {workflow['name']}.py")

    # Check for LLM API key (check both local env and project .env for deployments)
    has_local_key = _has_api_key()
    has_project_env_key = _has_api_key_in_project_env(project_root)

    if not has_local_key and not has_project_env_key:
        issues.append(
            "No LLM API key found (set OPENAI_API_KEY, ANTHROPIC_API_KEY, etc. in .env or environment)"
        )
    elif not has_local_key:
        # Has key in project .env (for deployments) but not in local env (for local testing)
        issues.append(
            "No LLM API key in local environment (project .env has keys for deployment)"
        )

    # Show issues
    if issues:
        print("  Issues:")
        for issue in issues:
            print(f"    {issue}")
        print("")
    else:
        print(" No issues found")
        print("")

    # Show helpful commands based on API key status
    import os

    has_daita_key = bool(os.getenv("DAITA_API_KEY"))

    print(" Quick commands:")
    print("   daita create agent my_agent    # Create new agent (free)")
    print("   daita test                     # Test all components (free)")
    print("   daita test --watch             # Development mode (free)")

    if has_daita_key:
        print("   daita push             # Deploy to cloud")
        print("   daita logs             # View cloud logs")
    else:
        print("   ")
        print("    Ready for cloud deployment?")
        print("   Get your API key at daita-tech.io")


def _load_deployments(project_root):
    """Load deployment history."""
    deployments_file = project_root / ".daita" / "deployments.json"

    if not deployments_file.exists():
        return []

    try:
        with open(deployments_file, "r") as f:
            return json.load(f)
    except:
        return []


def _load_project_config(project_root):
    """Load project configuration."""
    config_file = project_root / "daita-project.yaml"
    if not config_file.exists():
        return None

    try:
        with open(config_file, "r") as f:
            return yaml.safe_load(f)
    except:
        return None


def _has_api_key():
    """Check if API key is configured in local environment."""
    import os

    return bool(
        os.getenv("OPENAI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GROK_API_KEY")
    )


def _has_api_key_in_project_env(project_root):
    """Check if API key is configured in project's .env file (for deployments)."""
    env_file = project_root / ".env"
    if not env_file.exists():
        return False

    try:
        with open(env_file, "r") as f:
            env_content = f.read()

        # Check for LLM API key patterns in .env file
        llm_key_patterns = [
            "OPENAI_API_KEY=",
            "ANTHROPIC_API_KEY=",
            "GEMINI_API_KEY=",
            "GROK_API_KEY=",
        ]

        for pattern in llm_key_patterns:
            for line in env_content.splitlines():
                # Skip comments and empty lines
                line = line.strip()
                if line.startswith("#") or not line:
                    continue

                # Check if line has the pattern and a non-empty value
                if line.startswith(pattern):
                    value = line.split("=", 1)[1].strip()
                    # Remove quotes if present
                    value = value.strip('"').strip("'")
                    if value and value != "your-key-here" and value != "sk-...":
                        return True

        return False
    except Exception:
        return False


# Alias for backward compatibility
show_status = show_project_status
