"""
Daita CLI - Command Line Interface for Daita Agents.

Simple, git-like CLI for building and deploying AI agents.

Usage:
    daita init [project-name]              # Initialize new project
    daita create agent <name>              # Create agent
    daita create workflow <name>           # Create workflow
    daita test [target]                    # Test agents/workflows
    daita push                             # Deploy to production
    daita status                           # Show project status
    daita logs                             # View deployment logs
"""
import click
import asyncio
import logging
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional, ignore if not installed
    pass

# Check for required CLI dependencies
def check_cli_dependencies():
    """Check if CLI dependencies are installed."""
    missing = []
    
    try:
        import yaml
    except ImportError:
        missing.append("PyYAML")
    
    try:
        import watchdog
    except ImportError:
        missing.append("watchdog")
    
    if missing:
        click.echo(" Missing CLI dependencies. Install with:", err=True)
        click.echo(f"   pip install {' '.join(missing)}", err=True)
        click.echo("   Or install all CLI dependencies with:", err=True)
        click.echo("   pip install daita-agents[cli]", err=True)
        sys.exit(1)

# Import core functions only after dependency check
def _import_core_functions():
    """Import core CLI functions after dependency check."""
    try:
        from .core.init import initialize_project
        from .core.create import create_from_template
        from .core.test import run_tests
        from .core.deploy import deploy_to_environment
        from .core.status import show_project_status
        from .core.logs import show_deployment_logs
        from .core.deployments import (
            list_deployments, show_deployment_details,
            delete_deployment
        )
        from .core.run import (
            run_remote_execution, list_remote_executions, get_execution_logs
        )
        return (
            initialize_project,
            create_from_template,
            run_tests,
            deploy_to_environment,
            show_project_status,
            show_deployment_logs,
            list_deployments,
            show_deployment_details,
            delete_deployment,
            run_remote_execution,
            list_remote_executions,
            get_execution_logs
        )
    except ImportError as e:
        click.echo(f" Error importing CLI modules: {str(e)}", err=True)
        sys.exit(1)

# CLI version
__version__ = "0.1.1"

def _check_first_time_usage():
    """Check if this is the user's first time using the CLI and show welcome banner."""
    import os
    from pathlib import Path
    
    # Create a marker file in user's home directory
    marker_file = Path.home() / '.daita_cli_first_run'
    
    # If marker doesn't exist, this is first run
    if not marker_file.exists():
        try:
            # Show welcome banner
            from .ascii_art import display_welcome_banner
            display_welcome_banner()
            click.echo("    Welcome to Daita! ")
            click.echo("    Get started with: daita init my_project")
            click.echo("    For help: daita --help")
            click.echo("")
            
            # Create marker file to remember this isn't first run anymore
            marker_file.touch()
        except Exception:
            # If anything goes wrong, silently continue
            pass

@click.group()
@click.version_option(version=__version__, prog_name="daita")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--quiet', '-q', is_flag=True, help='Suppress non-error output')
@click.pass_context
def cli(ctx, verbose, quiet):
    """
    Daita CLI - AI Agent Framework Command Line Interface.
    
    Build, test, and deploy AI agents with ease.
    """
    # Check CLI dependencies
    check_cli_dependencies()
    
    # Import core functions after dependency check
    ctx.ensure_object(dict)
    ctx.obj['core_functions'] = _import_core_functions()
    
    # Setup logging
    if verbose:
        log_level = logging.DEBUG
    elif quiet:
        log_level = logging.ERROR
    else:
        log_level = logging.INFO

    # Configure root logger with higher level to avoid noise from libraries
    logging.basicConfig(
        level=logging.WARNING,  # Only show warnings/errors from third-party libraries
        format="%(levelname)s: %(message)s"
    )

    # Set our own loggers to the desired level
    daita_logger = logging.getLogger('daita')
    daita_logger.setLevel(log_level)

    # Also set CLI logger
    cli_logger = logging.getLogger(__name__)
    cli_logger.setLevel(log_level)
    
    # Store options in context
    ctx.obj['verbose'] = verbose
    ctx.obj['quiet'] = quiet

# ======= Core Commands =======

@cli.command()
@click.argument('project_name', required=False)
@click.option('--type', 'project_type', default='basic', 
              type=click.Choice(['basic', 'analysis', 'pipeline']),
              help='Type of project to create')
@click.option('--force', is_flag=True, help='Overwrite existing project')
@click.pass_context
def init(ctx, project_name, project_type, force):
    """Initialize a new Daita project."""
    try:
        initialize_project = ctx.obj['core_functions'][0]
        asyncio.run(initialize_project(
            project_name=project_name,
            project_type=project_type,
            force=force,
            verbose=ctx.obj.get('verbose', False)
        ))
    except KeyboardInterrupt:
        click.echo("\n Operation cancelled.", err=True)
        sys.exit(1)
    except Exception as e:
        if ctx.obj.get('verbose'):
            logging.exception("Init command failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)

@cli.group()
def create():
    """Create agents, workflows, and other components."""
    pass

@create.command()
@click.argument('name')
@click.pass_context
def agent(ctx, name):
    """Create a new agent."""
    try:
        create_from_template = ctx.obj['core_functions'][1]
        create_from_template(
            template='agent',
            name=name,
            verbose=ctx.obj.get('verbose', False)
        )
    except Exception as e:
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)

@create.command()
@click.argument('name')
@click.pass_context
def workflow(ctx, name):
    """Create a new workflow."""
    try:
        create_from_template = ctx.obj['core_functions'][1]
        create_from_template(
            template='workflow',
            name=name,
            verbose=ctx.obj.get('verbose', False)
        )
    except Exception as e:
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('target', required=False)
@click.option('--data', help='Test data file to use')
@click.option('--watch', is_flag=True, help='Watch for changes and re-run tests')
@click.pass_context
def test(ctx, target, data, watch):
    """Test agents and workflows."""
    try:
        run_tests = ctx.obj['core_functions'][2]
        asyncio.run(run_tests(
            target=target,
            data_file=data,
            watch=watch,
            verbose=ctx.obj.get('verbose', False)
        ))
    except KeyboardInterrupt:
        click.echo("\n Tests cancelled.", err=True)
        sys.exit(1)
    except Exception as e:
        if ctx.obj.get('verbose'):
            logging.exception("Test command failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--force', is_flag=True, help='Force deployment without confirmation')
@click.option('--dry-run', is_flag=True, help='Show what would be deployed')
@click.pass_context
def push(ctx, force, dry_run):
    """Deploy to production (like git push)."""
    environment = 'production'
    try:
        deploy_to_environment = ctx.obj['core_functions'][3]
        asyncio.run(deploy_to_environment(
            environment=environment,
            force=force,
            dry_run=dry_run,
            verbose=ctx.obj.get('verbose', False)
        ))
    except KeyboardInterrupt:
        click.echo("\n Deployment cancelled.", err=True)
        sys.exit(1)
    except Exception as e:
        if ctx.obj.get('verbose'):
            logging.exception("Deploy command failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.pass_context
def status(ctx):
    """Show project and deployment status (like git status)."""
    try:
        show_project_status = ctx.obj['core_functions'][4]
        asyncio.run(show_project_status(
            environment='production',  # Only production is supported
            verbose=ctx.obj.get('verbose', False)
        ))
    except Exception as e:
        if ctx.obj.get('verbose'):
            logging.exception("Status command failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--follow', '-f', is_flag=True, help='Follow log output')
@click.option('--lines', '-n', default=10, help='Number of lines to show')
@click.pass_context
def logs(ctx, follow, lines):
    """View deployment logs (like git log)."""
    try:
        show_deployment_logs = ctx.obj['core_functions'][5]
        asyncio.run(show_deployment_logs(
            environment='production',  # Only production is supported
            limit=lines,
            follow=follow,
            verbose=ctx.obj.get('verbose', False)
        ))
    except KeyboardInterrupt:
        if follow:
            click.echo("\n Stopped following logs.", err=True)
        sys.exit(1)
    except Exception as e:
        if ctx.obj.get('verbose'):
            logging.exception("Logs command failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)

# ======= Deployment Management Commands =======

@cli.group()
def deployments():
    """Manage deployments."""
    pass

@deployments.command('list')
@click.argument('project_name', required=False)
@click.option('--limit', default=10, help='Number of deployments to show')
@click.pass_context
def list_cmd(ctx, project_name, limit):
    """List deployment history."""
    try:
        list_deployments_func = ctx.obj['core_functions'][6]
        asyncio.run(list_deployments_func(project_name, 'production', limit))
    except Exception as e:
        if ctx.obj.get('verbose'):
            logging.exception("List deployments failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)

@deployments.command()
@click.argument('deployment_id')
@click.pass_context
def show(ctx, deployment_id):
    """Show detailed deployment information."""
    try:
        show_deployment_details_func = ctx.obj['core_functions'][7]
        asyncio.run(show_deployment_details_func(deployment_id))
    except Exception as e:
        if ctx.obj.get('verbose'):
            logging.exception("Show deployment failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)

@deployments.command()
@click.argument('deployment_id')
@click.option('--force', is_flag=True, help='Skip confirmation')
@click.pass_context
def delete(ctx, deployment_id, force):
    """Delete a deployment and its Lambda functions."""
    try:
        delete_deployment_func = ctx.obj['core_functions'][8]
        asyncio.run(delete_deployment_func(deployment_id, force))
    except Exception as e:
        if ctx.obj.get('verbose'):
            logging.exception("Delete deployment failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)

# ======= Utility Commands =======

@cli.command()
def version():
    """Show version information."""
    click.echo(f"Daita CLI v{__version__}")
    click.echo("AI Agent Framework")

@cli.command()
def docs():
    """Open documentation in browser."""
    import webbrowser
    webbrowser.open("https://docs.daita-tech.io")
    click.echo(" Opening documentation in browser...")

# ======= Main Entry Point =======

def main():
    """Main CLI entry point."""
    try:
        # Import freemium utilities
        from .utils import is_cloud_command, require_api_key_for_cloud, show_local_vs_cloud_help

        # Check current command (skip global flags like --verbose, --quiet, -v, -q)
        current_cmd = None
        for arg in sys.argv[1:]:
            if not arg.startswith('-'):
                current_cmd = arg
                break
        
        # Handle help command with freemium info
        if current_cmd in ['--help', '-h', 'help']:
            # Show ASCII art before help
            from .ascii_art import display_compact_banner
            display_compact_banner()
            # Let click handle help, but add freemium info
            try:
                cli()
            except SystemExit:
                show_local_vs_cloud_help()
                raise
        else:
            # Check for first-time usage only if not showing help
            _check_first_time_usage()
        
        # Enforce API key for cloud commands (but not for help)
        if current_cmd and is_cloud_command(current_cmd) and '--help' not in sys.argv:
            require_api_key_for_cloud()
        
        # Check if we're in a project for commands that need it
        project_commands = ['create', 'test', 'push', 'status', 'logs', 'deployments', 'run', 'executions', 'execution-logs', 'memory']
        
        if current_cmd in project_commands:
            # Check if we're in a Daita project
            current = Path.cwd()
            in_project = False
            for path in [current] + list(current.parents):
                if (path / '.daita').exists():
                    in_project = True
                    break
            
            if not in_project:
                click.echo(" Not in a Daita project directory.", err=True)
                click.echo("   Run 'daita init' to create a new project.")
                sys.exit(1)
        
        # Run the CLI
        cli()
        
    except Exception as e:
        click.echo(f" Unexpected error: {str(e)}", err=True)
        sys.exit(1)

# ===== REMOTE EXECUTION COMMANDS =====

@cli.command()
@click.argument('target_name')
@click.option('--type', 'target_type', default='agent',
              type=click.Choice(['agent', 'workflow']),
              help='Type of target to execute')
@click.option('--data', 'data_file',
              help='JSON file containing input data')
@click.option('--data-json',
              help='JSON string containing input data')
@click.option('--task', default='process',
              help='Task to execute (for agents only)')
@click.option('--follow', '-f', is_flag=True,
              help='Follow execution progress in real-time')
@click.option('--timeout', default=300, type=int,
              help='Execution timeout in seconds')
@click.pass_context
def run(ctx, target_name, target_type, data_file, data_json, task, follow, timeout):
    """Execute an agent or workflow remotely in the cloud."""
    environment = 'production'  # Only production is supported
    try:
        run_remote_execution_func = ctx.obj['core_functions'][9]
        success = asyncio.run(run_remote_execution_func(
            target_name=target_name,
            target_type=target_type,
            environment=environment,
            data_file=data_file,
            data_json=data_json,
            task=task,
            follow=follow,
            timeout=timeout,
            verbose=ctx.obj.get('verbose', False)
        ))
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\n Operation cancelled.", err=True)
        sys.exit(1)
    except Exception as e:
        if ctx.obj.get('verbose'):
            logging.exception("Run command failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)

@cli.command('executions')
@click.option('--limit', default=10, type=int,
              help='Number of executions to show')
@click.option('--status',
              type=click.Choice(['queued', 'running', 'completed', 'failed', 'cancelled']),
              help='Filter by execution status')
@click.option('--type', 'target_type',
              type=click.Choice(['agent', 'workflow']),
              help='Filter by target type')
@click.pass_context
def list_executions(ctx, limit, status, target_type):
    """List recent remote executions."""
    try:
        list_remote_executions_func = ctx.obj['core_functions'][12]
        success = asyncio.run(list_remote_executions_func(
            limit=limit,
            status=status,
            target_type=target_type,
            environment='production',  # Only production is supported
            verbose=ctx.obj.get('verbose', False)
        ))
        if not success:
            sys.exit(1)
    except Exception as e:
        if ctx.obj.get('verbose'):
            logging.exception("List executions command failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)

@cli.command('execution-logs')
@click.argument('execution_id')
@click.option('--follow', '-f', is_flag=True,
              help='Follow execution progress')
@click.pass_context
def execution_logs(ctx, execution_id, follow):
    """Get logs and status for a specific execution."""
    try:
        get_execution_logs_func = ctx.obj['core_functions'][13]
        success = asyncio.run(get_execution_logs_func(
            execution_id=execution_id,
            follow=follow,
            verbose=ctx.obj.get('verbose', False)
        ))
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        if follow:
            click.echo("\n Stopped following execution.", err=True)
        sys.exit(1)
    except Exception as e:
        if ctx.obj.get('verbose'):
            logging.exception("Execution logs command failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)

# ======= Webhook Commands =======

@cli.group()
@click.pass_context
def webhook(ctx):
    """Webhook management commands."""
    pass

@webhook.command('list')
@click.option('--api-key-only', is_flag=True,
              help='Show only webhooks created with current API key')
@click.pass_context
def webhook_list(ctx, api_key_only):
    """List all webhook URLs for your organization."""
    try:
        # Import the webhook listing function
        from .core.webhooks import list_webhooks

        success = asyncio.run(list_webhooks(
            api_key_only=api_key_only,
            verbose=ctx.obj.get('verbose', False)
        ))
        if not success:
            sys.exit(1)
    except Exception as e:
        if ctx.obj.get('verbose'):
            logging.exception("Webhook list command failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)

# ======= Memory Commands =======

@cli.group()
@click.pass_context
def memory(ctx):
    """Memory management commands."""
    pass

@memory.command('status')
@click.option('--project', help='Project name (default: current directory)')
@click.pass_context
def memory_status(ctx, project):
    """Show production memory status from cloud."""
    try:
        from .core.memory_commands import show_memory_status

        asyncio.run(show_memory_status(
            project=project,
            verbose=ctx.obj.get('verbose', False)
        ))
    except KeyboardInterrupt:
        click.echo("\n Operation cancelled.", err=True)
        sys.exit(1)
    except Exception as e:
        if ctx.obj.get('verbose'):
            logging.exception("Memory status command failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)

@memory.command('show')
@click.argument('workspace')
@click.option('--full', is_flag=True, help='Download complete files')
@click.option('--limit', default=50, help='Max memories to show')
@click.option('--project', help='Project name (default: current directory)')
@click.pass_context
def memory_show(ctx, workspace, full, limit, project):
    """Show production memory contents from cloud."""
    try:
        from .core.memory_commands import show_workspace_memory

        asyncio.run(show_workspace_memory(
            workspace=workspace,
            full=full,
            limit=limit,
            project=project,
            verbose=ctx.obj.get('verbose', False)
        ))
    except KeyboardInterrupt:
        click.echo("\n Operation cancelled.", err=True)
        sys.exit(1)
    except Exception as e:
        if ctx.obj.get('verbose'):
            logging.exception("Memory show command failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)

if __name__ == "__main__":
    main()