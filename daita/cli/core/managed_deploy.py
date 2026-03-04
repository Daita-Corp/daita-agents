"""
Managed deployment for Daita CLI - No AWS credentials required.

This uploads packages to the Daita API, which handles all AWS infrastructure.
Users only need DAITA_API_KEY, not AWS credentials.
"""
import os
import yaml
import tempfile
import aiohttp
import aiofiles
import ssl
from pathlib import Path
from ..utils import find_project_root
from .import_detector import ImportDetector
from daita.config.settings import settings


def _extract_framework_version(project_root: Path) -> str:
    """
    Extract daita-agents version from requirements.txt.

    Parses requirements.txt to find the exact version of daita-agents
    the user wants to deploy with.

    Returns:
        Version string (e.g., "0.6.2")
    """
    req_file = project_root / 'requirements.txt'

    if not req_file.exists():
        # No requirements.txt - use installed version
        try:
            import importlib.metadata
            return importlib.metadata.version('daita-agents')
        except Exception:
            return "0.6.2"  # Fallback to current

    with open(req_file) as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            if 'daita-agents' in line.lower():
                # Parse different formats:
                # daita-agents==0.6.2
                # daita-agents>=0.6.0
                # daita-agents~=0.6.0

                if '==' in line:
                    version = line.split('==')[1].split('#')[0].strip()
                elif '>=' in line:
                    version = line.split('>=')[1].split('#')[0].strip()
                elif '~=' in line:
                    version = line.split('~=')[1].split('#')[0].strip()
                else:
                    continue

                # Clean version (remove any trailing comments/spaces)
                version = version.split()[0] if version else None

                if version:
                    return version

    # Not found in requirements.txt - use installed version
    try:
        import importlib.metadata
        return importlib.metadata.version('daita-agents')
    except Exception:
        return "0.6.2"  # Fallback


async def deploy_to_managed_environment(environment='production', force=False, dry_run=False, verbose=False):
    """Deploy to Daita-managed cloud environment."""
    
    # Find project root
    project_root = find_project_root()
    if not project_root:
        raise ValueError("Not in a Daita project. Run 'daita init' first.")
    
    # Load project config
    config = _load_project_config(project_root)
    if not config:
        raise ValueError("No daita-project.yaml found")
    
    project_name = config.get('name', 'unknown')

    # Validate version is specified in YAML
    if not config.get('version'):
        raise ValueError("Version must be specified in daita-project.yaml")

    # Extract framework version from requirements.txt
    framework_version = _extract_framework_version(project_root)

    if verbose:
        print(f" Framework version: {framework_version}")
        print(f" Project version: {config.get('version')}")

    print(f" Deploying '{project_name}' to Daita-managed {environment}")
    print(f" Using daita-agents {framework_version}")
    
    if dry_run:
        print(f" Dry run - showing what would be deployed:")
        _show_deployment_plan(project_root, config, environment)
        return
    
    # Check for DAITA_API_KEY (this should be caught by main() but adding as safeguard)
    api_key = os.getenv("DAITA_API_KEY")
    if not api_key:
        from ..utils import show_upgrade_message
        show_upgrade_message()
        return
    
    # Create deployment package
    package_path = _create_deployment_package(project_root, config, verbose)
    
    try:
        # Upload package to Daita API
        upload_result = await _upload_package_to_api(
            package_path=package_path,
            project_name=project_name,
            environment=environment,
            api_key=api_key,
            verbose=verbose
        )
        
        if verbose:
            print(f"    Package uploaded: {upload_result['upload_id']}")
            print(f"    Package hash: {upload_result['package_hash'][:16]}...")
            print(f"    Package size: {upload_result['package_size_bytes'] / 1024 / 1024:.1f}MB")
        
        # Analyze project imports to determine required layers
        if verbose:
            print(f" Analyzing project imports for layer optimization...")
        
        detector = ImportDetector()
        import_analysis = detector.analyze_project(project_root)
        
        # Deploy uploaded package with layer information
        deployment_id = _generate_deployment_id(project_name, environment)
        
        deploy_result = await _deploy_package_via_api(
            upload_id=upload_result['upload_id'],
            deployment_id=deployment_id,
            project_name=project_name,
            environment=environment,
            framework_version=framework_version,
            config=config,
            import_analysis=import_analysis,
            api_key=api_key,
            verbose=verbose
        )
        
        print(f" Deployed to Daita-managed {environment}")
        print(f" Deployment ID: {deployment_id}")

        # Warn if project uses plugins that don't persist in cloud yet
        plugin_usage = _detect_plugin_usage(project_root)
        if plugin_usage['catalog'] or plugin_usage['lineage']:
            affected = []
            if plugin_usage['catalog']:
                affected.append('CatalogPlugin')
            if plugin_usage['lineage']:
                affected.append('LineagePlugin')
            plugins_str = ' and '.join(affected)
            print(f"")
            print(f" Note: {plugins_str} detected in this project.")
            print(f"   Schema/lineage data will not persist to the dashboard in cloud")
            print(f"   deployments — cloud persistence is not yet available.")
            print(f"   Discovery and lineage tracking will still work at runtime;")
            print(f"   results just won't appear in the Daita dashboard.")

        # Handle memory sync after successful deployment
        await _handle_memory_sync_on_deploy(
            project_root=project_root,
            project_name=project_name,
            environment=environment,
            verbose=verbose
        )

    except aiohttp.ClientConnectorError:
        print(" Cannot connect to deployment host")
        print("    Check your internet connection and try again")
        raise ValueError("Cannot connect to deployment host")
    except aiohttp.ClientError as e:
        print(" Deployment connection failed")
        if verbose:
            print(f"   Details: {str(e)}")
        raise ValueError("Deployment connection failed")
    finally:
        # Clean up temporary package
        if package_path.exists():
            os.unlink(package_path)

def _get_secure_api_endpoint() -> str:
    """Get validated API endpoint with security checks."""
    # Use production API endpoint (can be overridden via environment)
    endpoint = os.getenv("DAITA_API_ENDPOINT") or "https://api.daita-tech.io"
    
    try:
        return settings.validate_endpoint(endpoint)
    except ValueError as e:
        raise ValueError(f"Invalid API endpoint configuration: {e}")

async def _upload_package_to_api(
    package_path: Path,
    project_name: str,
    environment: str,
    api_key: str,
    verbose: bool = False
) -> dict:
    """Upload deployment package to Daita API with progress tracking."""
    
    api_endpoint = _get_secure_api_endpoint()
    package_size = package_path.stat().st_size
    
    if verbose:
        print(f"    Uploading package to secure API endpoint...")
        print(f"    Package size: {package_size / 1024 / 1024:.1f}MB")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Daita-CLI/1.0.0"
    }
    
    # Create secure SSL context
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = True
    ssl_context.verify_mode = ssl.CERT_REQUIRED
    
    # For large packages, show progress
    if package_size > 10 * 1024 * 1024:  # Show progress for packages > 10MB
        return await _upload_with_progress(
            package_path, project_name, environment, api_endpoint, headers, ssl_context, verbose
        )
    else:
        return await _upload_standard(
            package_path, project_name, environment, api_endpoint, headers, ssl_context, verbose
        )


async def _upload_standard(
    package_path: Path,
    project_name: str,
    environment: str,
    api_endpoint: str,
    headers: dict,
    ssl_context: ssl.SSLContext,
    verbose: bool
) -> dict:
    """Standard upload for smaller packages."""
    # Prepare multipart form data
    data = aiohttp.FormData()
    data.add_field('project_name', project_name)
    data.add_field('environment', environment)
    
    # Add file
    async with aiofiles.open(package_path, 'rb') as f:
        file_content = await f.read()
        data.add_field('package', file_content, filename=f"{project_name}.zip", content_type='application/zip')
    
    # Create secure connector
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        url = f"{api_endpoint}/api/v1/packages/upload"
        
        async with session.post(url, data=data, headers=headers, timeout=300) as response:
            return await _handle_upload_response(response, verbose)


async def _upload_with_progress(
    package_path: Path,
    project_name: str,
    environment: str,
    api_endpoint: str,
    headers: dict,
    ssl_context: ssl.SSLContext,
    verbose: bool
) -> dict:
    """Upload with progress tracking for large packages."""
    package_size = package_path.stat().st_size
    uploaded_size = 0
    
    class ProgressReader:
        def __init__(self, file_obj, total_size):
            self.file_obj = file_obj
            self.total_size = total_size
            self.uploaded = 0
            self.last_progress = 0
        
        def read(self, chunk_size):
            chunk = self.file_obj.read(chunk_size)
            if chunk:
                self.uploaded += len(chunk)
                progress = int((self.uploaded / self.total_size) * 100)
                
                # Update progress every 5%
                if progress >= self.last_progress + 5:
                    self.last_progress = progress
                    if verbose:
                        print(f"    Upload progress: {progress}% ({self.uploaded / 1024 / 1024:.1f}MB / {self.total_size / 1024 / 1024:.1f}MB)")
                    else:
                        # Simple progress bar
                        bar_length = 30
                        filled_length = int(bar_length * progress // 100)
                        bar = '█' * filled_length + '░' * (bar_length - filled_length)
                        print(f"\r    Uploading: [{bar}] {progress}%", end='', flush=True)
            
            return chunk
    
    # Prepare multipart form data with progress tracking
    data = aiohttp.FormData()
    data.add_field('project_name', project_name)
    data.add_field('environment', environment)
    
    # Add file with progress tracking
    with open(package_path, 'rb') as f:
        progress_reader = ProgressReader(f, package_size)
        data.add_field('package', progress_reader, filename=f"{project_name}.zip", content_type='application/zip')
        
        # Create secure connector
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            url = f"{api_endpoint}/api/v1/packages/upload"
            
            async with session.post(url, data=data, headers=headers, timeout=600) as response:
                if not verbose:
                    print()  # New line after progress bar
                return await _handle_upload_response(response, verbose)


async def _handle_upload_response(response, verbose: bool) -> dict:
    """Handle upload response with proper error handling."""
    if response.status == 200:
        result = await response.json()
        if verbose:
            print(f"    Package uploaded successfully")
        return result
    elif response.status == 401:
        error_text = await response.text()
        print(" Authentication failed - check your DAITA_API_KEY")
        print("   Get a new API key at daita-tech.io")
        raise ValueError("Invalid API key")
    elif response.status == 413:
        print(" Package too large (max 250MB)")
        print("    Try removing large dependencies or data files")
        raise ValueError("Package size exceeded")
    else:
        error_text = await response.text()
        print(f" Upload failed (HTTP {response.status})")
        if verbose:
            print(f"   Details: {error_text}")
        raise ValueError("Upload failed")

async def _deploy_package_via_api(
    upload_id: str,
    deployment_id: str,
    project_name: str,
    environment: str,
    framework_version: str,
    config: dict,
    import_analysis: dict,
    api_key: str,
    verbose: bool = False
) -> dict:
    """Deploy uploaded package via Daita API."""

    api_endpoint = _get_secure_api_endpoint()

    if verbose:
        print(f"    Deploying with framework version {framework_version}...")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "Daita-CLI/1.0.0"
    }
    
    # Create secure SSL context
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = True
    ssl_context.verify_mode = ssl.CERT_REQUIRED
    
    # Get version from YAML (required)
    yaml_version = config.get("version")
    if not yaml_version:
        raise ValueError("Version must be specified in daita-project.yaml")
    
    # Prepare deployment request with layer optimization
    deploy_data = {
        "upload_id": upload_id,
        "deployment_id": deployment_id,
        "project_name": project_name,
        "environment": environment,
        "framework_version": framework_version,
        "version": yaml_version,
        "agents_config": _extract_agent_configs(config),
        "workflows_config": _extract_workflow_configs(config),
        "schedules_config": _extract_schedules_config(config, environment, verbose),
        "import_analysis": import_analysis,
        "layer_requirements": _determine_layer_requirements(import_analysis, verbose)
    }
    
    # Create secure connector
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        url = f"{api_endpoint}/api/v1/packages/deploy"
        
        async with session.post(url, json=deploy_data, headers=headers, timeout=600) as response:
            if response.status == 200:
                result = await response.json()
                if verbose:
                    print(f"    Deployment completed securely")
                return result
            elif response.status == 401:
                print(" Authentication failed during deployment")
                print("   Get a new API key at daita-tech.io")
                raise ValueError("Invalid API key")
            elif response.status == 404:
                print(" Upload not found - it may have expired")
                print("    Try uploading again with: daita push")
                raise ValueError("Upload expired")
            else:
                error_text = await response.text()
                print(f" Deployment failed (HTTP {response.status})")
                if verbose:
                    print(f"   Details: {error_text}")
                raise ValueError("Deployment failed")

def _create_deployment_package(project_root: Path, config: dict, verbose: bool = False) -> Path:
    """Create deployment package with all user project files."""
    print(f" Creating deployment package...")

    # Create temp directory
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
        package_path = Path(temp_file.name)

    import zipfile
    with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:

        #  OPTIMIZATION: Framework now served via Lambda layers - no longer bundled!
        # This reduces package size from 50MB+ to <5MB

        # Directories to exclude from packaging
        exclude_dirs = {'.daita', '__pycache__', '.git', '.pytest_cache',
                       'venv', 'env', '.venv', 'node_modules', '.mypy_cache',
                       'tests', 'data'}  # Add common dirs to exclude

        # Add all project directories (except excluded ones)
        for item in project_root.iterdir():
            if item.is_dir() and item.name not in exclude_dirs and not item.name.startswith('.'):
                _add_directory_to_zip(zipf, item, item.name)
                if verbose:
                    file_count = len(list(item.rglob('*.py')))
                    print(f"   Added directory: {item.name}/ ({file_count} Python files)")

        # Add project configuration (required)
        config_file = project_root / 'daita-project.yaml'
        if config_file.exists():
            zipf.write(config_file, 'daita-project.yaml')

        # Add requirements if they exist
        requirements_file = project_root / 'requirements.txt'
        if requirements_file.exists():
            zipf.write(requirements_file, 'requirements.txt')

        # Add .env file for user's API keys (even though it's in .gitignore)
        env_file = project_root / '.env'
        if env_file.exists():
            zipf.write(env_file, '.env')
            if verbose:
                print(f"   Added .env file to package")

        # Add minimal bootstrap handler (framework loaded from layers)
        _add_bootstrap_handler(zipf)

    if verbose:
        package_size = package_path.stat().st_size
        print(f"    Package: {package_size / 1024 / 1024:.1f}MB")

    return package_path

def _add_bootstrap_handler(zipf):
    """Add minimal bootstrap handler that loads framework from layers."""
    
    # Create bootstrap handler that delegates to the framework layer
    bootstrap_handler_content = '''"""
Bootstrap handler for Daita Lambda functions.

This handler loads the Daita framework from Lambda layers and delegates
execution to the universal handler. This approach dramatically reduces
package sizes by serving the framework from pre-built layers.
"""

import sys
import os
import json
from typing import Dict, Any

def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Bootstrap handler that loads framework from layers and delegates execution.
    
    The Daita framework is provided via Lambda layers:
    - daita-framework-optimized: Core framework code (0.12MB)
    - daita-core-dependencies: Essential dependencies (19MB)
    """
    
    try:
        # Framework is available via layers - import directly
        from cloud.lambda_handler import lambda_handler as framework_handler
        
        # Delegate to the framework handler
        return framework_handler(event, context)
        
    except ImportError as e:
        # Fallback error handling if layers aren't properly configured
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Framework layer not available',
                'message': str(e),
                'help': 'Ensure Lambda function has daita-framework-optimized and daita-core-dependencies layers attached'
            })
        }
    except Exception as e:
        # General error handling
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Execution failed',
                'message': str(e)
            })
        }
'''
    
    # Add bootstrap handler to package
    handler_bytes = bootstrap_handler_content.encode('utf-8')
    zipf.writestr('lambda_handler.py', handler_bytes)

def _add_directory_to_zip(zipf, source_dir: Path, archive_name: str):
    """Add directory to zip recursively."""
    for file_path in source_dir.rglob('*'):
        # Include .env files even though they start with '.'
        if file_path.is_file() and (not file_path.name.startswith('.') or file_path.name == '.env'):
            relative_path = file_path.relative_to(source_dir)
            archive_path = f"{archive_name}/{relative_path}"
            zipf.write(file_path, archive_path)

def _extract_agent_configs(config: dict) -> list:
    """Extract agent configurations."""
    agents = []
    
    # Get agents from config
    config_agents = config.get("agents", [])
    for agent in config_agents:
        agents.append({
            "name": agent.get("name", "Unknown Agent"),
            "type": agent.get("type", "standard"),
            "enabled": agent.get("enabled", True),
            "settings": agent.get("settings", {})
        })
    
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
    """Extract workflow configurations."""
    workflows = []
    
    # Get workflows from config
    config_workflows = config.get("workflows", [])
    for workflow in config_workflows:
        workflows.append({
            "name": workflow.get("name", "Unknown Workflow"),
            "type": workflow.get("type", "basic"),
            "enabled": workflow.get("enabled", True),
            "settings": workflow.get("settings", {})
        })
    
    return workflows


def _extract_schedules_config(config: dict, environment: str, verbose: bool = False) -> dict:
    """Extract and validate scheduling configuration."""
    try:
        from ...config.scheduling import parse_schedules_from_yaml, apply_environment_overrides

        # Get base schedules configuration
        schedules_data = config.get('schedules', {})

        if not schedules_data:
            # No schedules configured
            return {}

        # Parse base schedule configuration
        base_schedules = parse_schedules_from_yaml(schedules_data)

        # Apply environment-specific overrides
        environments_config = config.get('environments', {})
        final_schedules = apply_environment_overrides(
            base_schedules, environments_config, environment
        )

        # Validate against available agents and workflows
        available_agents = [agent.get('name') for agent in config.get('agents', [])]
        available_workflows = [workflow.get('name') for workflow in config.get('workflows', [])]

        final_schedules.validate(available_agents, available_workflows)

        # Convert to serializable format
        schedules_dict = {
            'agents': {},
            'workflows': {}
        }

        for agent_name, schedule_config in final_schedules.agents.items():
            schedules_dict['agents'][agent_name] = {
                'cron': schedule_config.cron,
                'data': schedule_config.data,
                'enabled': schedule_config.enabled,
                'timezone': schedule_config.timezone,
                'description': schedule_config.description
            }

        for workflow_name, schedule_config in final_schedules.workflows.items():
            schedules_dict['workflows'][workflow_name] = {
                'cron': schedule_config.cron,
                'data': schedule_config.data,
                'enabled': schedule_config.enabled,
                'timezone': schedule_config.timezone,
                'description': schedule_config.description
            }

        if verbose and not final_schedules.is_empty():
            agent_count = len(final_schedules.agents)
            workflow_count = len(final_schedules.workflows)
            print(f"    Schedules: {agent_count} agents, {workflow_count} workflows")

            for agent_name, schedule in final_schedules.agents.items():
                if schedule.enabled:
                    print(f"      Agent {agent_name}: {schedule.cron}")

            for workflow_name, schedule in final_schedules.workflows.items():
                if schedule.enabled:
                    print(f"      Workflow {workflow_name}: {schedule.cron}")

        return schedules_dict

    except ImportError:
        # Schedule dependencies not available - skip scheduling
        if verbose:
            print("    Scheduling: Not available (missing croniter dependency)")
        return {}

    except Exception as e:
        print(f" Schedule validation failed: {str(e)}")
        print("   Check your schedule configuration in daita-project.yaml")
        raise ValueError(f"Invalid schedule configuration: {e}")


def _determine_layer_requirements(import_analysis: dict, verbose: bool = False) -> dict:
    """Determine layer requirements without exposing internal ARNs."""
    layer_requirements = {
        'needs_framework': True,  # Always needed
        'needs_core_dependencies': True,  # Always needed  
        'needs_common_dependencies': False,
        'common_packages_needed': [],
        'optimization_summary': {}
    }
    
    # Check if common dependencies layer is needed
    required_layer_types = import_analysis.get('required_layers', {})
    if 'common_dependencies' in required_layer_types:
        layer_requirements['needs_common_dependencies'] = True
        packages = required_layer_types['common_dependencies']
        layer_requirements['common_packages_needed'] = packages
        if verbose:
            print(f"    Common dependencies needed for: {', '.join(packages)}")
    
    # Add optimization summary (no internal details)
    total_imports = import_analysis.get('total_imports', 0)
    common_packages = len(layer_requirements['common_packages_needed'])
    
    layer_requirements['optimization_summary'] = {
        'total_imports_detected': total_imports,
        'packages_optimized_by_layers': common_packages,
        'optimization_enabled': common_packages > 0
    }
    
    if verbose:
        print(f"    Framework layer: Required")
        print(f"    Core dependencies layer: Required") 
        if layer_requirements['needs_common_dependencies']:
            print(f"    Optimization: {common_packages}/{total_imports} packages served by layers")
        else:
            print(f"    No additional layer optimization needed")
    
    return layer_requirements

def _generate_deployment_id(project_name: str, environment: str) -> str:
    """Generate deployment ID."""
    import uuid
    
    # Generate a proper UUID for deployment_id
    return str(uuid.uuid4())

def _show_deployment_plan(project_root: Path, config: dict, environment: str):
    """Show deployment plan."""
    print(f"")
    print(f"Project: {config.get('name')}")
    print(f"Environment: Daita-managed {environment}")
    print(f"")
    
    # Show agents
    agents = _extract_agent_configs(config)
    if agents:
        print(f"Agents ({len(agents)}):")
        for agent in agents:
            print(f"   {agent['name']}")
    
    # Show workflows
    workflows = _extract_workflow_configs(config)
    if workflows:
        print(f"Workflows ({len(workflows)}):")
        for workflow in workflows:
            print(f"   {workflow['name']}")

    # Show schedules
    try:
        schedules = _extract_schedules_config(config, environment, verbose=False)
        if schedules.get('agents') or schedules.get('workflows'):
            print(f"Schedules:")
            for agent_name, schedule in schedules.get('agents', {}).items():
                if schedule.get('enabled', True):
                    print(f"   Agent {agent_name}: {schedule['cron']}")
            for workflow_name, schedule in schedules.get('workflows', {}).items():
                if schedule.get('enabled', True):
                    print(f"   Workflow {workflow_name}: {schedule['cron']}")
    except Exception:
        # Skip schedule display if there are issues
        pass

    print(f"")
    print(f"Deployment Details:")
    print(f"    Infrastructure: Daita-managed AWS Lambda + EventBridge")
    print(f"   Package Upload: Via Daita API")
    print(f"   Authentication: DAITA_API_KEY only")
    print(f"    No AWS credentials required")

    # Warn about plugins that don't persist in cloud
    plugin_usage = _detect_plugin_usage(project_root)
    if plugin_usage['catalog'] or plugin_usage['lineage']:
        affected = []
        if plugin_usage['catalog']:
            affected.append('CatalogPlugin')
        if plugin_usage['lineage']:
            affected.append('LineagePlugin')
        plugins_str = ' and '.join(affected)
        print(f"")
        print(f" Note: {plugins_str} detected.")
        print(f"   Schema/lineage data will not persist to the dashboard in cloud deployments.")
        print(f"   Cloud persistence is not yet available.")


async def _handle_memory_sync_on_deploy(
    project_root: Path,
    project_name: str,
    environment: str,
    verbose: bool = False
):
    """
    Handle memory sync during deployment.

    Automatically syncs local memory to cloud if:
    - Local memory exists
    - Cloud memory doesn't exist (first deployment)

    Skips sync if cloud memory already exists (preserves production memory).
    """
    import boto3
    import click

    # Check if local memory exists
    local_memory_dir = project_root / '.daita' / 'memory' / 'workspaces'
    if not local_memory_dir.exists():
        return  # No local memory, nothing to do

    workspaces = [d.name for d in local_memory_dir.iterdir() if d.is_dir()]
    if not workspaces:
        return  # No workspaces, nothing to do

    # Get organization ID from API key
    try:
        org_id = await _get_org_id_from_api()
        if not org_id:
            if verbose:
                print("\n    Memory sync skipped: Cannot determine organization ID")
            return
    except Exception as e:
        if verbose:
            print(f"\n    Memory sync skipped: {e}")
        return

    # Check if cloud memory exists for any workspace
    try:
        s3 = boto3.client('s3')
        bucket = f"daita-memory-{os.getenv('AWS_REGION', 'us-east-1')}"

        cloud_has_memory = False
        for workspace in workspaces:
            prefix = f"orgs/{org_id}/projects/{project_name}/workspaces/{workspace}/"
            try:
                response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
                if response.get('Contents'):
                    cloud_has_memory = True
                    break
            except Exception:
                pass

        if cloud_has_memory:
            # Cloud memory exists - skip sync
            print(f"\n Memory: Cloud memory detected - preserving production memory")
            print(f"          (Local memory is for testing only and will not be synced)")
            return

        # Cloud is empty - offer to sync local as initial setup
        print(f"\n Memory: Local memory detected ({len(workspaces)} workspace(s))")
        print(f"          Cloud memory is empty. Sync local memory to cloud")
        print(f"          as initial production memory?")

        response = click.prompt("\n          Sync local → cloud",
                               type=click.Choice(['y', 'n'], case_sensitive=False),
                               default='y')

        if response.lower() == 'y':
            print(f"\n          Syncing memory to cloud...")

            # Import sync functions
            from .memory_sync_utils import sync_workspace_to_s3

            synced_count = 0
            for workspace in workspaces:
                try:
                    sync_workspace_to_s3(
                        workspace=workspace,
                        org_id=org_id,
                        project_name=project_name,
                        s3_client=s3,
                        bucket=bucket,
                        local_memory_dir=local_memory_dir
                    )
                    synced_count += 1
                except Exception as e:
                    if verbose:
                        print(f"          Warning: Failed to sync {workspace}: {e}")

            print(f"          ✓ Synced {synced_count}/{len(workspaces)} workspace(s) to cloud\n")
        else:
            print(f"          Skipped - cloud will start with empty memory\n")

    except Exception as e:
        if verbose:
            print(f"\n    Memory sync error: {e}")
        # Don't fail deployment on memory sync errors
        return


async def _get_org_id_from_api() -> str:
    """Get organization ID from API."""
    import aiohttp

    api_key = os.getenv('DAITA_API_KEY')
    if not api_key:
        return None

    api_endpoint = _get_secure_api_endpoint()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Daita-CLI/1.0.0"
    }

    try:
        async with aiohttp.ClientSession() as session:
            url = f"{api_endpoint}/api/v1/user/me"
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('organization_id')
    except Exception:
        pass

    return None


def _detect_plugin_usage(project_root: Path) -> dict:
    """Scan project Python files for catalog/lineage plugin usage."""
    import re

    _EXCLUDE_DIRS = {'.daita', '__pycache__', '.git', 'venv', '.venv', 'node_modules', '.mypy_cache'}
    catalog_pattern = re.compile(r'from\s+daita.*import.*\bcatalog\b|CatalogPlugin', re.IGNORECASE)
    lineage_pattern = re.compile(r'from\s+daita.*import.*\blineage\b|LineagePlugin', re.IGNORECASE)

    uses_catalog = False
    uses_lineage = False

    for py_file in project_root.rglob('*.py'):
        if any(part in _EXCLUDE_DIRS for part in py_file.parts):
            continue
        try:
            content = py_file.read_text(encoding='utf-8', errors='ignore')
            if not uses_catalog and catalog_pattern.search(content):
                uses_catalog = True
            if not uses_lineage and lineage_pattern.search(content):
                uses_lineage = True
        except Exception:
            pass
        if uses_catalog and uses_lineage:
            break

    return {'catalog': uses_catalog, 'lineage': uses_lineage}


def _load_project_config(project_root: Path):
    """Load project configuration."""
    config_file = project_root / 'daita-project.yaml'
    if not config_file.exists():
        return None

    with open(config_file, 'r') as f:
        return yaml.safe_load(f)