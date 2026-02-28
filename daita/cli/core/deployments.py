"""
Deployment management commands for Daita CLI.

Provides commands to list and manage deployments.
"""
import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional
from ..utils import find_project_root

async def list_deployments(project_name: Optional[str] = None, environment: Optional[str] = None, limit: int = 10):
    """List deployment history from managed cloud API."""
    import aiohttp

    try:
        # Get API key
        api_key = os.getenv('DAITA_API_KEY')
        if not api_key:
            print(" No DAITA_API_KEY found")
            print("   Get your API key at daita-tech.io")
            print("   Then: export DAITA_API_KEY='your-key-here'")
            return

        # If no project specified, try to get from current directory
        if not project_name:
            project_root = find_project_root()
            if project_root:
                config = _load_project_config(project_root)
                if config:
                    project_name = config.get('name', 'unknown')

        # Get API endpoint
        api_endpoint = os.getenv('DAITA_API_ENDPOINT', 'https://api.daita-tech.io')

        headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Daita-CLI/1.0.0"
        }

        # Build query parameters
        params = {}
        if project_name:
            params['project_name'] = project_name
        if environment:
            params['environment'] = environment
        if limit:
            params['limit'] = limit

        print(f" Deployment History{' for ' + repr(project_name) if project_name else ''}")
        if environment:
            print(f"   Environment: {environment}")
        print()

        # Fetch deployment history from API
        async with aiohttp.ClientSession() as session:
            url = f"{api_endpoint}/api/v1/deployments/api-key"

            async with session.get(url, headers=headers, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()

                    # Handle paginated response from API
                    if isinstance(data, dict) and 'deployments' in data:
                        deployments = data['deployments']
                    else:
                        deployments = data if isinstance(data, list) else []

                    if not deployments:
                        print("   No deployments found.")
                        return

                    # Display deployments
                    for i, deployment in enumerate(deployments[:limit], 1):
                        status_emoji = "●" if deployment.get('status') == 'active' else "○"
                        deployed_at = deployment.get('deployed_at', '')

                        # Parse deployment timestamp
                        try:
                            deployed_date = datetime.fromisoformat(deployed_at.replace('Z', '+00:00'))
                            deployed_str = deployed_date.strftime('%Y-%m-%d %H:%M:%S UTC')
                        except:
                            deployed_str = deployed_at

                        deployment_id = deployment.get('deployment_id', 'unknown')

                        print(f"{i:2}. {status_emoji} {deployment_id[:36]}")
                        print(f"      Environment: {deployment.get('environment', 'unknown')}")
                        print(f"      Version: {deployment.get('version', '1.0.0')}")
                        print(f"      Deployed: {deployed_str}")

                        # Show package size if available
                        if 'package_size_bytes' in deployment:
                            package_size_mb = deployment['package_size_bytes'] / 1024 / 1024
                            print(f"      Package: {package_size_mb:.1f}MB")

                        # Show agents from deployment info
                        deployment_info = deployment.get('deployment_info', {})
                        if deployment_info and 'functions' in deployment_info:
                            agent_names = [f.get('agent_name', 'Unknown') for f in deployment_info['functions']]
                            if agent_names:
                                print(f"      Agents: {', '.join(agent_names)}")

                        print()

                    # Show total count if there are more
                    total_count = len(deployments)
                    if total_count > limit:
                        print(f"   Showing {limit} of {total_count} deployments")
                        print(f"   Use --limit {total_count} to see all deployments")

                elif response.status == 401:
                    print(" Authentication failed")
                    print("   Check your DAITA_API_KEY")
                else:
                    error_text = await response.text()
                    print(f" Failed to fetch deployments (HTTP {response.status})")
                    print(f"   {error_text}")

    except aiohttp.ClientConnectorError:
        print(" Cannot connect to deployment API")
        print("   Check your internet connection")
    except Exception as e:
        print(f" Failed to list deployments: {e}")

async def show_deployment_details(deployment_id: str):
    """Show detailed information about a deployment."""
    try:
        from ...cloud.lambda_deploy import LambdaDeployer
        
        # Get AWS region
        aws_region = os.getenv('AWS_REGION', 'us-east-1')
        deployer = LambdaDeployer(aws_region)
        
        print(f" Deployment Details: {deployment_id}")
        print()
        
        # Get all deployments and find the target
        all_deployments = await deployer.get_deployment_history('all', limit=100)
        target_deployment = None
        
        for deployment in all_deployments:
            if deployment['deployment_id'] == deployment_id:
                target_deployment = deployment
                break
        
        if not target_deployment:
            print(f" Deployment {deployment_id} not found")
            return
        
        # Display detailed information
        deployed_date = datetime.fromisoformat(target_deployment['deployed_at'].replace('Z', '+00:00'))
        
        print(f"Project: {target_deployment['project_name']}")
        print(f"Environment: {target_deployment['environment']}")
        print(f"Version: {target_deployment.get('version', '1.0.0')}")
        print(f"Deployed At: {deployed_date.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Package Size: {target_deployment.get('package_size_bytes', 0) / 1024 / 1024:.1f}MB")
        print()
        
        # Show agents
        agents = target_deployment.get('agents', [])
        if agents:
            print("Agents:")
            for agent in agents:
                print(f"   {agent.get('name', 'Unknown')}")
                print(f"      Type: {agent.get('type', 'standard')}")
                print(f"      Enabled: {agent.get('enabled', True)}")
                if agent.get('file'):
                    print(f"      File: {agent['file']}")
                print()
        
        # Show workflows
        workflows = target_deployment.get('workflows', [])
        if workflows:
            print("Workflows:")
            for workflow in workflows:
                print(f"   {workflow.get('name', 'Unknown')}")
                print(f"      Type: {workflow.get('type', 'basic')}")
                print(f"      Enabled: {workflow.get('enabled', True)}")
                if workflow.get('file'):
                    print(f"      File: {workflow['file']}")
                print()
    
    except Exception as e:
        print(f" Failed to get deployment details: {e}")

async def rollback_deployment(deployment_id: str, environment: str = 'production'):
    """Rollback to a previous deployment."""
    try:
        from ...cloud.lambda_deploy import LambdaDeployer
        
        # Get AWS region
        aws_region = os.getenv('AWS_REGION', 'us-east-1')
        deployer = LambdaDeployer(aws_region)
        
        print(f" Rolling back to deployment: {deployment_id}")
        print(f"   Target environment: {environment}")
        
        # Confirm rollback
        if environment == 'production':
            confirm = input("  Rollback PRODUCTION environment? Type 'yes' to confirm: ")
            if confirm != 'yes':
                print(" Rollback cancelled")
                return
        
        # Perform rollback
        result = await deployer.rollback_deployment(deployment_id, environment)
        
        if result['status'] == 'success':
            print(f" Rollback initiated")
            print(f"   New deployment ID: {result['rollback_deployment_id']}")
            print(f"   Original deployment: {result['original_deployment_id']}")
            print(f"   {result['message']}")
        else:
            print(f" Rollback failed: {result['error']}")
    
    except Exception as e:
        print(f" Failed to rollback deployment: {e}")

async def delete_deployment(deployment_id: str, force: bool = False):
    """Delete a deployment via API."""
    import aiohttp

    try:
        # Get API key
        api_key = os.getenv('DAITA_API_KEY')
        if not api_key:
            print(" No DAITA_API_KEY found")
            print("   Get your API key at daita-tech.io")
            return

        print(f" Deleting deployment: {deployment_id}")

        # Confirm deletion
        if not force:
            confirm = input("  Delete deployment? Type 'yes' to confirm: ")
            if confirm != 'yes':
                print(" Deletion cancelled")
                return

        # Get API endpoint
        api_endpoint = os.getenv('DAITA_API_ENDPOINT', 'https://api.daita-tech.io')

        headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Daita-CLI/1.0.0"
        }

        async with aiohttp.ClientSession() as session:
            url = f"{api_endpoint}/api/v1/deployments/{deployment_id}"

            async with session.delete(url, headers=headers, timeout=30) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f" {result.get('message', 'Deployment deleted successfully')}")
                elif response.status == 404:
                    print(" Deployment not found")
                elif response.status == 401:
                    print(" Authentication failed")
                    print("   Check your DAITA_API_KEY")
                else:
                    error_text = await response.text()
                    print(f" Deletion failed (HTTP {response.status})")
                    print(f"   {error_text}")

    except aiohttp.ClientConnectorError:
        print(" Cannot connect to API")
        print("   Check your internet connection")
    except Exception as e:
        print(f" Failed to delete deployment: {e}")

# Helper functions


def _load_project_config(project_root: Path):
    """Load project configuration from daita-project.yaml."""
    import yaml
    
    config_file = project_root / 'daita-project.yaml'
    if not config_file.exists():
        return None
    
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception:
        return None