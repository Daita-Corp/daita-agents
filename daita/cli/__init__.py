"""
Daita CLI - Command Line Interface for Daita Agents.

Simple, git-like CLI for building and deploying AI agents.

Example usage:
    # Initialize a new project
    daita init my-project --type analysis
    
    # Create components
    daita create agent data_processor
    daita create workflow data_pipeline
    
    # Test and develop
    daita test
    daita test --watch
    
    # Deploy
    daita push
    
    # Monitor
    daita status
    daita logs production
"""

# CLI version
__version__ = "0.2.1"

# Import main CLI components
from .main import cli, main

# Import core functions for programmatic access
from .core.init import initialize_project
from .core.create import create_from_template
from .core.test import run_tests
from .core.deploy import deploy_to_environment
from .core.status import show_project_status
from .core.logs import show_deployment_logs

# Export main components
__all__ = [
    # Version
    '__version__',
    
    # Main CLI
    'cli',
    'main',
    
    # Core functions (for programmatic access)
    'initialize_project',
    'create_from_template',
    'run_tests',
    'deploy_to_environment',
    'show_project_status',
    'show_deployment_logs',
]

# CLI metadata
CLI_INFO = {
    'name': 'daita',
    'version': __version__,
    'description': 'CLI for AI agent development and deployment',
    'author': 'Daita Team',
    'docs': 'https://docs.daita-tech.io',
}

def get_cli_info() -> dict:
    """
    Get CLI information and metadata.
    
    Returns:
        Dictionary with CLI information
    """
    return CLI_INFO.copy()

def print_banner():
    """Print CLI banner with version information."""
    print(f"""
╔══════════════════════════════════════════════╗
║                 Daita CLI                    ║
║          AI Agent Framework CLI              ║
║                                              ║
║  Version: {__version__:<31} ║
║  Docs: https://docs.daita-tech.io/cli             ║
╚══════════════════════════════════════════════╝
""")

# Convenience functions for common operations
def quick_init(project_name: str = None, project_type: str = 'basic') -> None:
    """
    Quick project initialization.
    
    Args:
        project_name: Name of the project
        project_type: Type of project (basic, analysis, pipeline)
    """
    import asyncio
    
    asyncio.run(initialize_project(
        project_name=project_name,
        project_type=project_type,
        verbose=False
    ))

def quick_status() -> dict:
    """
    Quick status check for current project.
    
    Returns:
        Dictionary with project status information
    """
    from .utils import find_project_root
    
    project_root = find_project_root()
    if not project_root:
        return {"error": "Not in a Daita project"}
    
    # This would return status info - simplified for MVP
    return {
        "project_root": str(project_root),
        "status": "active"
    }

def quick_test(target: str = None) -> bool:
    """
    Quick test runner.
    
    Args:
        target: Specific agent or workflow to test
        
    Returns:
        True if tests passed, False otherwise
    """
    import asyncio
    
    try:
        asyncio.run(run_tests(
            target=target,
            verbose=False
        ))
        return True
    except Exception:
        return False