"""
Shared utilities for Daita CLI.

Common functions used across CLI core modules to avoid duplication.
"""
import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ======= Project Management =======

def find_project_root(start_path: Optional[Path] = None) -> Optional[Path]:
    """
    Find the root directory of a Daita project.
    
    Searches upward from the given path (or current directory) looking for .daita folder.
    
    Args:
        start_path: Path to start searching from (defaults to current directory)
        
    Returns:
        Path to project root, or None if not in a Daita project
    """
    current = start_path or Path.cwd()
    
    for path in [current] + list(current.parents):
        if (path / '.daita').exists():
            return path
    
    return None

def ensure_project_root() -> Path:
    """
    Find project root and raise error if not in a project.
    
    Returns:
        Path to project root
        
    Raises:
        ValueError: If not in a Daita project
    """
    project_root = find_project_root()
    if not project_root:
        raise ValueError("Not in a Daita project. Run 'daita init' first.")
    return project_root

def load_project_config(project_root: Path) -> Optional[Dict[str, Any]]:
    """
    Load project configuration from daita-project.yaml.
    
    Args:
        project_root: Path to project root directory
        
    Returns:
        Project configuration dictionary, or None if file doesn't exist
    """
    config_file = project_root / 'daita-project.yaml'
    
    if not config_file.exists():
        return None
    
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Failed to load project config: {str(e)}")
        return None

def save_project_config(project_root: Path, config: Dict[str, Any]) -> None:
    """
    Save project configuration to daita-project.yaml.
    
    Args:
        project_root: Path to project root directory
        config: Configuration dictionary to save
    """
    config_file = project_root / 'daita-project.yaml'
    
    try:
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        raise ValueError(f"Failed to save project config: {str(e)}")

# ======= Deployment Management =======

def load_deployments(project_root: Path) -> list:
    """
    Load deployment history from .daita/deployments.json.
    
    Args:
        project_root: Path to project root directory
        
    Returns:
        List of deployment records
    """
    deployments_file = project_root / '.daita' / 'deployments.json'
    
    if not deployments_file.exists():
        return []
    
    try:
        with open(deployments_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load deployments: {str(e)}")
        return []

def save_deployments(project_root: Path, deployments: list) -> None:
    """
    Save deployment history to .daita/deployments.json.
    
    Args:
        project_root: Path to project root directory
        deployments: List of deployment records to save
    """
    deployments_file = project_root / '.daita' / 'deployments.json'
    
    # Ensure .daita directory exists
    deployments_file.parent.mkdir(exist_ok=True)
    
    try:
        with open(deployments_file, 'w') as f:
            json.dump(deployments, f, indent=2)
    except Exception as e:
        raise ValueError(f"Failed to save deployments: {str(e)}")

# ======= File Operations =======

def copy_files(src_dir: Path, dest_dir: Path, pattern: str, verbose: bool = False) -> None:
    """
    Copy files matching a pattern from source to destination.
    
    Args:
        src_dir: Source directory
        dest_dir: Destination directory
        pattern: File pattern to match (e.g., 'agents/', '*.py')
        verbose: Whether to print detailed output
    """
    src_path = src_dir / pattern
    
    if src_path.is_file():
        # Single file
        dest_file = dest_dir / pattern
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        dest_file.write_bytes(src_path.read_bytes())
        if verbose:
            print(f"   Copied: {pattern}")
    
    elif src_path.is_dir():
        # Directory - copy all Python files
        for file_path in src_path.rglob('*.py'):
            rel_path = file_path.relative_to(src_dir)
            dest_file = dest_dir / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            dest_file.write_bytes(file_path.read_bytes())
            if verbose:
                print(f"   Copied: {rel_path}")

def list_python_files(directory: Path) -> list:
    """
    List Python files in a directory (excluding __init__.py).
    
    Args:
        directory: Directory to search
        
    Returns:
        List of Python file stems (names without .py extension)
    """
    if not directory.exists():
        return []
    
    files = []
    for file in directory.glob('*.py'):
        if file.name != '__init__.py':
            files.append(file.stem)
    
    return sorted(files)

# ======= String Utilities =======

def to_class_name(snake_case_name: str) -> str:
    """
    Convert snake_case to PascalCase for class names.
    
    Args:
        snake_case_name: Snake case string (e.g., 'my_agent')
        
    Returns:
        PascalCase string (e.g., 'MyAgent')
    """
    return ''.join(word.capitalize() for word in snake_case_name.split('_'))

def to_snake_case(name: str) -> str:
    """
    Convert a name to snake_case.
    
    Args:
        name: Input name (any case)
        
    Returns:
        Snake case string
    """
    # Replace spaces and hyphens with underscores
    name = name.replace(' ', '_').replace('-', '_')
    
    # Convert to lowercase
    name = name.lower()
    
    # Remove any non-alphanumeric characters except underscores
    import re
    name = re.sub(r'[^a-z0-9_]', '', name)
    
    # Remove duplicate underscores
    while '__' in name:
        name = name.replace('__', '_')
    
    # Remove leading/trailing underscores
    return name.strip('_')

def format_result_preview(result: Any, max_length: int = 100) -> str:
    """
    Format a result for preview display.
    
    Args:
        result: Result to format
        max_length: Maximum length of preview string
        
    Returns:
        Formatted preview string
    """
    if isinstance(result, dict):
        # Show status and key count
        status = result.get('status', 'unknown')
        key_count = len(result.keys())
        return f"status={status}, {key_count} keys"
    
    elif isinstance(result, list):
        # Show list length and type of first item
        if result:
            first_type = type(result[0]).__name__
            return f"list[{len(result)}] of {first_type}"
        else:
            return "empty list"
    
    else:
        # Show truncated string representation
        result_str = str(result)
        if len(result_str) > max_length:
            return result_str[:max_length] + "..."
        return result_str

# ======= Environment Utilities =======

def has_api_key() -> bool:
    """
    Check if any API key is configured in environment.
    
    Returns:
        True if at least one API key is found
    """
    api_keys = [
        'OPENAI_API_KEY',
        'ANTHROPIC_API_KEY', 
        'GOOGLE_API_KEY',
        'GEMINI_API_KEY',
        'XAI_API_KEY',
        'GROK_API_KEY'
    ]
    
    return any(os.getenv(key) for key in api_keys)

def get_configured_providers() -> list:
    """
    Get list of LLM providers that have API keys configured.
    
    Returns:
        List of provider names
    """
    providers = []
    
    if os.getenv('OPENAI_API_KEY'):
        providers.append('openai')
    
    if os.getenv('ANTHROPIC_API_KEY'):
        providers.append('anthropic')
    
    if os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY'):
        providers.append('gemini')
    
    if os.getenv('XAI_API_KEY') or os.getenv('GROK_API_KEY'):
        providers.append('grok')
    
    return providers

# ======= Validation Utilities =======

def validate_project_name(name: str) -> bool:
    """
    Validate project name for safety and compatibility.
    
    Args:
        name: Project name to validate
        
    Returns:
        True if name is valid
    """
    if not name:
        return False
    
    # Check length
    if len(name) > 50:
        return False
    
    # Check for valid characters (alphanumeric, underscore, hyphen)
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        return False
    
    # Must start with letter or number
    if not re.match(r'^[a-zA-Z0-9]', name):
        return False
    
    return True

def validate_component_name(name: str) -> bool:
    """
    Validate agent/workflow name for Python compatibility.
    
    Args:
        name: Component name to validate
        
    Returns:
        True if name is valid for Python module
    """
    if not name:
        return False
    
    # Check length
    if len(name) > 30:
        return False
    
    # Check for valid Python identifier characters
    import re
    if not re.match(r'^[a-zA-Z0-9_]+$', name):
        return False
    
    # Must start with letter or underscore
    if not re.match(r'^[a-zA-Z_]', name):
        return False
    
    # Cannot be Python keywords
    import keyword
    if keyword.iskeyword(name):
        return False
    
    return True

# ======= Error Handling =======

class CLIError(Exception):
    """Base exception for CLI errors."""
    pass

class ProjectNotFoundError(CLIError):
    """Raised when not in a Daita project."""
    pass

class ConfigurationError(CLIError):
    """Raised for configuration-related errors."""
    pass

def handle_cli_error(error: Exception, verbose: bool = False) -> None:
    """
    Handle CLI errors with appropriate user messages.
    
    Args:
        error: Exception that occurred
        verbose: Whether to show detailed error information
    """
    if isinstance(error, ProjectNotFoundError):
        print(" Not in a Daita project directory.")
        print("   Run 'daita init' to create a new project.")
    
    elif isinstance(error, ConfigurationError):
        print(f" Configuration error: {str(error)}")
    
    elif isinstance(error, CLIError):
        print(f" {str(error)}")
    
    else:
        if verbose:
            import traceback
            print(f" Unexpected error: {str(error)}")
            print("\nDetailed error:")
            traceback.print_exc()
        else:
            print(f" Error: {str(error)}")
            print("   Use --verbose for more details.")

# ======= Freemium Business Model =======

def has_daita_api_key() -> bool:
    """
    Check if DAITA_API_KEY is configured for cloud operations.

    Returns:
        True if DAITA_API_KEY is available
    """
    # First check environment variable
    if os.getenv('DAITA_API_KEY'):
        return True

    # Try to load from .env file in current directory
    try:
        from dotenv import load_dotenv
        from pathlib import Path

        # Try loading from current directory
        env_path = Path.cwd() / '.env'
        if env_path.exists():
            load_dotenv(env_path)

        api_key = os.getenv('DAITA_API_KEY')
        return bool(api_key)
    except ImportError:
        # dotenv is optional, ignore if not installed
        return False
    except Exception as e:
        return False

def require_api_key_for_cloud() -> None:
    """
    Enforce API key requirement for cloud operations with upgrade messaging.
    
    Raises:
        SystemExit: If no API key is found, shows upgrade message and exits
    """
    if not has_daita_api_key():
        show_upgrade_message()
        raise SystemExit(1)

def show_upgrade_message() -> None:
    """
    Display freemium upgrade message for cloud features.
    """
    print("Ready to deploy to the cloud?")
    print("")
    print("   Get your API key at daita-tech.io")
    print("   Then: export DAITA_API_KEY='your-key-here'")
    print("")
    print("   Get insights, monitoring, and 24/7 hosting")

def show_local_vs_cloud_help() -> None:
    """
    Show help message explaining local vs cloud features.
    """
    print("")
    print("Daita Command Guide:")
    print("")
    print("   FREE (Local Development):")
    print("   • daita init           - Create projects")
    print("   • daita create         - Build agents & workflows") 
    print("   • daita test           - Test locally")
    print("   • daita test --watch   - Development mode")
    print("")
    print("   PREMIUM (Cloud Hosting):")
    print("   • daita push           - Deploy to cloud")
    print("   • daita status         - Monitor deployments")
    print("   • daita logs           - View execution logs") 
    print("   • daita run            - Execute remotely")
    print("")
    print("   Get started: daita-tech.io")

def get_freemium_success_message(project_name: str) -> str:
    """
    Generate success message for project initialization with freemium guidance.
    
    Args:
        project_name: Name of the created project
        
    Returns:
        Formatted success message with next steps
    """
    return f"""Project '{project_name}' created successfully!

Next steps:
   1. cd {project_name}
   2. daita test              # Test locally (always free)
   3. daita create agent      # Add more agents (free)
   4. daita test --watch      # Development mode (free)

Ready for production?
   • daita push       # Deploy to cloud (requires API key)
   • Get your API key at daita-tech.io
   • Start your free trial with full monitoring & insights!"""

# Cloud command definitions for enforcement
CLOUD_COMMANDS = {
    'push', 'status', 'logs', 'deployments', 'run', 
    'executions', 'execution-logs'
}

def is_cloud_command(command: str) -> bool:
    """
    Check if a command requires cloud/API key access.
    
    Args:
        command: Command name to check
        
    Returns:
        True if command requires API key
    """
    return command in CLOUD_COMMANDS

# ======= Logging Setup =======

def setup_cli_logging(level: str = "INFO") -> None:
    """
    Setup logging for CLI operations.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
        force=True  # Override any existing configuration
    )
    
    # Silence verbose third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)