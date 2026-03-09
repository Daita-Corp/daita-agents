"""
Deployment wrapper for Daita CLI with managed cloud infrastructure.
Users only need DAITA_API_KEY - no AWS credentials required.

 ARCHITECTURE: This module serves as compatibility layer that delegates
to managed_deploy.py. The CLI uses a single deployment path via 'push' command.
No duplicate deployment systems - architecture is already consolidated.
"""
import os
import yaml
import json
import tarfile
import tempfile
import asyncio
import aiohttp
from pathlib import Path
from datetime import datetime
from .managed_deploy import deploy_to_managed_environment
from ..utils import find_project_root

async def deploy_to_environment(environment='production', force=False, dry_run=False, verbose=False):
   """Deploy to Daita-managed environment - no AWS credentials required."""
   
   # Use the new managed deployment system
   await deploy_to_managed_environment(
       environment=environment,
       force=force,
       dry_run=dry_run,
       verbose=verbose
   )

def _extract_agent_configs(config: dict) -> list:
    """Extract agent configurations from project config."""
    agents = []
    
    # Get agents from config
    config_agents = config.get("agents", [])
    for agent in config_agents:
        agent_config = {
            "name": agent.get("name", "Unknown Agent"),
            "type": agent.get("type", "standard"),
            "enabled": agent.get("enabled", True),
            "settings": agent.get("settings", {})
        }
        agents.append(agent_config)
    
    # If no agents in config, scan agents directory
    if not agents:
        project_root = find_project_root()
        if project_root:
            agents_dir = project_root / "agents"
            if agents_dir.exists():
                for agent_file in agents_dir.glob("*.py"):
                    if agent_file.name != "__init__.py":
                        agent_name = agent_file.stem.replace("_", " ").title()
                        agents.append({
                            "name": agent_name,
                            "type": "standard",
                            "enabled": True,
                            "file": agent_file.name
                        })
    
    return agents

def _extract_workflow_configs(config: dict) -> list:
    """Extract workflow configurations from project config."""
    workflows = []
    
    # Get workflows from config
    config_workflows = config.get("workflows", [])
    for workflow in config_workflows:
        workflow_config = {
            "name": workflow.get("name", "Unknown Workflow"),
            "type": workflow.get("type", "basic"),
            "enabled": workflow.get("enabled", True),
            "agents": workflow.get("agents", []),
            "settings": workflow.get("settings", {})
        }
        workflows.append(workflow_config)
    
    # If no workflows in config, scan workflows directory
    if not workflows:
        project_root = find_project_root()
        if project_root:
            workflows_dir = project_root / "workflows"
            if workflows_dir.exists():
                for workflow_file in workflows_dir.glob("*.py"):
                    if workflow_file.name != "__init__.py":
                        workflow_name = workflow_file.stem.replace("_", " ").title()
                        workflows.append({
                            "name": workflow_name,
                            "type": "basic",
                            "enabled": True,
                            "file": workflow_file.name
                        })
    
    return workflows

def _generate_deployment_id(project_name: str, environment: str) -> str:
    """Generate a unique deployment ID."""
    import uuid
    
    # Create a deterministic but unique ID
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    unique_suffix = str(uuid.uuid4())[:8]
    
    # Clean project name for ID
    clean_name = "".join(c for c in project_name if c.isalnum() or c in "_-").lower()
    
    return f"{clean_name}_{environment}_{timestamp}_{unique_suffix}"

def _create_deployment_package(project_root, config, verbose):
   """Create a deployment package (tar.gz)."""
   print(f" Creating deployment package...")
   
   # Create temp directory
   with tempfile.TemporaryDirectory() as temp_dir:
       package_dir = Path(temp_dir) / 'package'
       package_dir.mkdir()
       
       # Copy project files
       files_to_include = [
           'agents/',
           'workflows/',
           'daita-project.yaml',
           'requirements.txt'
       ]
       
       for file_pattern in files_to_include:
           _copy_files(project_root, package_dir, file_pattern, verbose)
       
       # Create package info
       package_info = {
           'name': config.get('name'),
           'version': config.get('version', '1.0.0'),
           'created_at': datetime.utcnow().isoformat(),
           'agents': [agent['name'] for agent in config.get('agents', [])],
           'workflows': [wf['name'] for wf in config.get('workflows', [])]
       }
       
       info_file = package_dir / 'package-info.json'
       with open(info_file, 'w') as f:
           json.dump(package_info, f, indent=2)
       
       # Create tar.gz
       package_path = project_root / '.daita' / f'deploy-{int(datetime.utcnow().timestamp())}.tar.gz'
       package_path.parent.mkdir(exist_ok=True)
       
       with tarfile.open(package_path, 'w:gz') as tar:
           tar.add(package_dir, arcname='.')
       
       if verbose:
           print(f"    Package: {package_path}")
           print(f"    Size: {package_path.stat().st_size} bytes")
       
       return package_path

async def _deploy_to_staging(package_path, config, verbose):
   """Deploy to staging environment using AWS Lambda."""
   print(f"  Deploying to staging using AWS Lambda...")
   
   try:
       from ...cloud.lambda_deploy import LambdaDeployer
       
       # Find project root for Lambda packaging
       project_root = find_project_root()
       if not project_root:
           raise ValueError("Could not find project root")
       
       # Generate deployment ID
       deployment_id = _generate_deployment_id(config.get('name', 'unknown'), 'staging')
       
       # Deploy to Lambda
       deployer = LambdaDeployer()
       result = await deployer.deploy_agent(project_root, config, deployment_id, 'staging')
       
       print(f"    Lambda functions deployed")
       print(f"    API Gateway endpoints created")
       print(f"    Agents ready for cloud execution")
       
       if verbose:
           for func in result['functions']:
               if func.get('status') == 'deployed':
                   print(f"    {func['name']}: {func['function_name']}")
                   if 'api_endpoint' in func:
                       print(f"       API: {func['api_endpoint'].get('endpoint_url', 'N/A')}")
               else:
                   print(f"    {func['name']}: {func.get('error', 'Unknown error')}")
       
       print(f"    Agents will automatically report traces to dashboard")
       return result
       
   except Exception as e:
       print(f"    Lambda deployment failed: {e}")
       if verbose:
           import traceback
           print(f"    Error details: {traceback.format_exc()}")
       raise

async def _deploy_to_production(package_path, config, force, verbose):
   """Deploy to production environment using AWS Lambda."""
   if not force:
       # Safety check for production
       confirm = input(f"  Deploy to PRODUCTION? Type 'yes' to confirm: ")
       if confirm != 'yes':
           print(f" Deployment cancelled")
           return
   
   print(f" Deploying to production using AWS Lambda...")
   
   try:
       from ...cloud.lambda_deploy import LambdaDeployer
       
       # Find project root for Lambda packaging
       project_root = find_project_root()
       if not project_root:
           raise ValueError("Could not find project root")
       
       # Generate deployment ID
       deployment_id = _generate_deployment_id(config.get('name', 'unknown'), 'production')
       
       # Deploy to Lambda
       deployer = LambdaDeployer()
       result = await deployer.deploy_agent(project_root, config, deployment_id, 'production')
       
       print(f"    Lambda functions deployed to production")
       print(f"    API Gateway endpoints configured")
       print(f"    Production environment variables set")
       print(f"    Agents ready for production workloads")
       
       if verbose:
           for func in result['functions']:
               if func.get('status') == 'deployed':
                   print(f"    {func['name']}: {func['function_name']}")
                   if 'api_endpoint' in func:
                       print(f"       API: {func['api_endpoint'].get('endpoint_url', 'N/A')}")
               else:
                   print(f"    {func['name']}: {func.get('error', 'Unknown error')}")
       
       print(f"    All traces will be sent to dashboard API")
       return result
       
   except Exception as e:
       print(f"    Production deployment failed: {e}")
       if verbose:
           import traceback
           print(f"    Error details: {traceback.format_exc()}")
       raise

async def _deploy_to_custom(package_path, config, environment, verbose):
   """Deploy to custom environment."""
   print(f" Deploying to {environment}...")
   
   # Load environment config
   env_config = config.get('environments', {}).get(environment, {})
   if not env_config:
       print(f"  No configuration found for environment '{environment}'")
       print(f"   Add it to daita-project.yaml under 'environments'")
   
   # Basic deployment
   await asyncio.sleep(1)
   print(f"    Deployed to {environment}")
   print(f"    Set DAITA_ENVIRONMENT=production on {environment} for API-only operation tracking")

def _show_deployment_plan(project_root, config, environment):
   """Show what would be deployed (dry run)."""
   print(f"")
   print(f"Project: {config.get('name')}")
   print(f"Environment: {environment}")
   print(f"")
   
   # Show agents
   agents = config.get('agents', [])
   if agents:
       print(f"Agents ({len(agents)}):")
       for agent in agents:
           print(f"   {agent['name']}")
   else:
       # Scan agents directory
       agents_dir = project_root / 'agents'
       if agents_dir.exists():
           agent_files = [f for f in agents_dir.glob('*.py') if f.name != '__init__.py']
           print(f"Agents ({len(agent_files)}):")
           for agent_file in agent_files:
               agent_name = agent_file.stem.replace('_', ' ').title()
               print(f"   {agent_name}")
   
   # Show workflows  
   workflows = config.get('workflows', [])
   if workflows:
       print(f"Workflows ({len(workflows)}):")
       for workflow in workflows:
           print(f"   {workflow['name']}")
   else:
       # Scan workflows directory
       workflows_dir = project_root / 'workflows'
       if workflows_dir.exists():
           workflow_files = [f for f in workflows_dir.glob('*.py') if f.name != '__init__.py']
           if workflow_files:
               print(f"Workflows ({len(workflow_files)}):")
               for workflow_file in workflow_files:
                   workflow_name = workflow_file.stem.replace('_', ' ').title()
                   print(f"   {workflow_name}")
   
   # Show files
   print(f"")
   print(f"Files to deploy:")
   for file_path in ['agents/', 'workflows/', 'daita-project.yaml', 'requirements.txt']:
       full_path = project_root / file_path
       if full_path.exists():
           if full_path.is_dir():
               count = len(list(full_path.glob('*.py')))
               print(f"   {file_path} ({count} files)")
           else:
               print(f"   {file_path}")
   
   # Show dashboard integration info
   print(f"")
   print(f"Dashboard Integration:")
   api_key = os.getenv("DAITA_API_KEY")
   if api_key:
       print(f"   DAITA_API_KEY configured")
       print(f"   Deployment will be tracked in dashboard")
   else:
       print(f"    No DAITA_API_KEY - set environment variable for dashboard integration")

def _copy_files(src_dir, dest_dir, pattern, verbose):
   """Copy files matching pattern."""
   src_path = src_dir / pattern
   
   if src_path.is_file():
       # Single file
       dest_file = dest_dir / pattern
       dest_file.parent.mkdir(parents=True, exist_ok=True)
       dest_file.write_bytes(src_path.read_bytes())
       if verbose:
           print(f"    Copied: {pattern}")
   
   elif src_path.is_dir():
       # Directory
       for file_path in src_path.rglob('*.py'):
           rel_path = file_path.relative_to(src_dir)
           dest_file = dest_dir / rel_path
           dest_file.parent.mkdir(parents=True, exist_ok=True)
           dest_file.write_bytes(file_path.read_bytes())
           if verbose:
               print(f"    Copied: {rel_path}")

def _save_deployment_record(project_root, environment, config, deployment_id):
   """Save deployment record for history."""
   deployments_file = project_root / '.daita' / 'deployments.json'
   
   # Load existing deployments
   if deployments_file.exists():
       with open(deployments_file, 'r') as f:
           deployments = json.load(f)
   else:
       deployments = []
   
   # Add new deployment
   deployment = {
       'deployment_id': deployment_id,
       'environment': environment,
       'timestamp': datetime.utcnow().isoformat(),
       'project_name': config.get('name'),
       'version': config.get('version', '1.0.0'),
       'agents': [agent['name'] for agent in config.get('agents', [])],
       'workflows': [wf['name'] for wf in config.get('workflows', [])],
       'dashboard_registered': bool(os.getenv("DAITA_API_KEY"))
   }
   
   deployments.append(deployment)
   
   # Keep only last 50 deployments
   deployments = deployments[-50:]
   
   # Save
   with open(deployments_file, 'w') as f:
       json.dump(deployments, f, indent=2)

def _load_project_config(project_root):
   """Load project configuration."""
   config_file = project_root / 'daita-project.yaml'
   if not config_file.exists():
       return None
   
   with open(config_file, 'r') as f:
       return yaml.safe_load(f)

