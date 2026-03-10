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

try:
    import importlib.metadata

    __version__ = importlib.metadata.version("daita-agents")
except Exception:
    __version__ = "unknown"

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Core function imports
from .core.init import initialize_project
from .core.create import create_from_template
from .core.test import run_tests
from .core.deploy import deploy_to_environment
from .core.status import show_project_status
from .core.logs import show_deployment_logs
from .core.deployments import (
    list_deployments,
    show_deployment_details,
    delete_deployment,
)
from .core.run import run_remote_execution, list_remote_executions, get_execution_logs


def _check_first_time_usage():
    """Show welcome banner on first run."""
    from pathlib import Path

    marker_file = Path.home() / ".daita_cli_first_run"
    if not marker_file.exists():
        try:
            from .ascii_art import display_welcome_banner

            display_welcome_banner()
            click.echo("    Welcome to Daita!")
            click.echo("    Get started with: daita init my_project")
            click.echo("    For help: daita --help")
            click.echo("")
            marker_file.touch()
        except Exception:
            pass


@click.group()
@click.version_option(version=__version__, prog_name="daita")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-error output")
@click.pass_context
def cli(ctx, verbose, quiet):
    """
    Daita CLI - AI Agent Framework Command Line Interface.

    Build, test, and deploy AI agents with ease.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet

    log_level = logging.DEBUG if verbose else logging.ERROR if quiet else logging.INFO
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    logging.getLogger("daita").setLevel(log_level)
    logging.getLogger(__name__).setLevel(log_level)


# ======= Core Commands =======


@cli.command()
@click.argument("project_name", required=False)
@click.option(
    "--type",
    "project_type",
    default="basic",
    type=click.Choice(["basic", "analysis", "pipeline"]),
    help="Type of project to create",
)
@click.option("--force", is_flag=True, help="Overwrite existing project")
@click.pass_context
def init(ctx, project_name, project_type, force):
    """Initialize a new Daita project."""
    try:
        asyncio.run(
            initialize_project(
                project_name=project_name,
                project_type=project_type,
                force=force,
                verbose=ctx.obj.get("verbose", False),
            )
        )
    except KeyboardInterrupt:
        click.echo("\n Operation cancelled.", err=True)
        sys.exit(1)
    except Exception as e:
        if ctx.obj.get("verbose"):
            logging.exception("Init command failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)


@cli.group()
def create():
    """Create agents, workflows, and other components."""
    pass


@create.command()
@click.argument("name")
@click.pass_context
def agent(ctx, name):
    """Create a new agent."""
    try:
        create_from_template(
            template="agent", name=name, verbose=ctx.obj.get("verbose", False)
        )
    except Exception as e:
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)


@create.command()
@click.argument("name")
@click.pass_context
def workflow(ctx, name):
    """Create a new workflow."""
    try:
        create_from_template(
            template="workflow", name=name, verbose=ctx.obj.get("verbose", False)
        )
    except Exception as e:
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("target", required=False)
@click.option("--data", help="Test data file to use")
@click.option("--watch", is_flag=True, help="Watch for changes and re-run tests")
@click.pass_context
def test(ctx, target, data, watch):
    """Test agents and workflows."""
    if watch:
        try:
            import watchdog  # noqa: F401
        except ImportError:
            click.echo(
                " watchdog is required for --watch mode. "
                "Install with: pip install 'daita-agents[cli]'",
                err=True,
            )
            sys.exit(1)
    try:
        asyncio.run(
            run_tests(
                target=target,
                data_file=data,
                watch=watch,
                verbose=ctx.obj.get("verbose", False),
            )
        )
    except KeyboardInterrupt:
        click.echo("\n Tests cancelled.", err=True)
        sys.exit(1)
    except Exception as e:
        if ctx.obj.get("verbose"):
            logging.exception("Test command failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--force", is_flag=True, help="Force deployment without confirmation")
@click.option("--dry-run", is_flag=True, help="Show what would be deployed")
@click.pass_context
def push(ctx, force, dry_run):
    """Deploy to production (like git push)."""
    try:
        asyncio.run(
            deploy_to_environment(
                environment="production",
                force=force,
                dry_run=dry_run,
                verbose=ctx.obj.get("verbose", False),
            )
        )
    except KeyboardInterrupt:
        click.echo("\n Deployment cancelled.", err=True)
        sys.exit(1)
    except Exception as e:
        if ctx.obj.get("verbose"):
            logging.exception("Deploy command failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx):
    """Show project and deployment status (like git status)."""
    try:
        asyncio.run(
            show_project_status(
                environment="production", verbose=ctx.obj.get("verbose", False)
            )
        )
    except Exception as e:
        if ctx.obj.get("verbose"):
            logging.exception("Status command failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
@click.option("--lines", "-n", default=10, help="Number of lines to show")
@click.pass_context
def logs(ctx, follow, lines):
    """View deployment logs (like git log)."""
    try:
        asyncio.run(
            show_deployment_logs(
                environment="production",
                limit=lines,
                follow=follow,
                verbose=ctx.obj.get("verbose", False),
            )
        )
    except KeyboardInterrupt:
        if follow:
            click.echo("\n Stopped following logs.", err=True)
        sys.exit(1)
    except Exception as e:
        if ctx.obj.get("verbose"):
            logging.exception("Logs command failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)


# ======= Deployment Management Commands =======


@cli.group()
def deployments():
    """Manage deployments."""
    pass


@deployments.command("list")
@click.argument("project_name", required=False)
@click.option("--limit", default=10, help="Number of deployments to show")
@click.pass_context
def list_cmd(ctx, project_name, limit):
    """List deployment history."""
    try:
        asyncio.run(list_deployments(project_name, "production", limit))
    except Exception as e:
        if ctx.obj.get("verbose"):
            logging.exception("List deployments failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)


@deployments.command()
@click.argument("deployment_id")
@click.pass_context
def show(ctx, deployment_id):
    """Show detailed deployment information."""
    try:
        asyncio.run(show_deployment_details(deployment_id))
    except Exception as e:
        if ctx.obj.get("verbose"):
            logging.exception("Show deployment failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)


@deployments.command()
@click.argument("deployment_id")
@click.option("--force", is_flag=True, help="Skip confirmation")
@click.pass_context
def delete(ctx, deployment_id, force):
    """Delete a deployment and its Lambda functions."""
    try:
        asyncio.run(delete_deployment(deployment_id, force))
    except Exception as e:
        if ctx.obj.get("verbose"):
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


# ======= Remote Execution Commands =======


@cli.command()
@click.argument("target_name")
@click.option(
    "--type",
    "target_type",
    default="agent",
    type=click.Choice(["agent", "workflow"]),
    help="Type of target to execute",
)
@click.option("--data", "data_file", help="JSON file containing input data")
@click.option("--data-json", help="JSON string containing input data")
@click.option("--task", default="process", help="Task to execute (for agents only)")
@click.option(
    "--follow", "-f", is_flag=True, help="Follow execution progress in real-time"
)
@click.option("--timeout", default=300, type=int, help="Execution timeout in seconds")
@click.pass_context
def run(ctx, target_name, target_type, data_file, data_json, task, follow, timeout):
    """Execute an agent or workflow remotely in the cloud."""
    try:
        success = asyncio.run(
            run_remote_execution(
                target_name=target_name,
                target_type=target_type,
                environment="production",
                data_file=data_file,
                data_json=data_json,
                task=task,
                follow=follow,
                timeout=timeout,
                verbose=ctx.obj.get("verbose", False),
            )
        )
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\n Operation cancelled.", err=True)
        sys.exit(1)
    except Exception as e:
        if ctx.obj.get("verbose"):
            logging.exception("Run command failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command("executions")
@click.option("--limit", default=10, type=int, help="Number of executions to show")
@click.option(
    "--status",
    type=click.Choice(["queued", "running", "completed", "failed", "cancelled"]),
    help="Filter by execution status",
)
@click.option(
    "--type",
    "target_type",
    type=click.Choice(["agent", "workflow"]),
    help="Filter by target type",
)
@click.pass_context
def list_executions(ctx, limit, status, target_type):
    """List recent remote executions."""
    try:
        success = asyncio.run(
            list_remote_executions(
                limit=limit,
                status=status,
                target_type=target_type,
                environment="production",
                verbose=ctx.obj.get("verbose", False),
            )
        )
        if not success:
            sys.exit(1)
    except Exception as e:
        if ctx.obj.get("verbose"):
            logging.exception("List executions command failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command("execution-logs")
@click.argument("execution_id")
@click.option("--follow", "-f", is_flag=True, help="Follow execution progress")
@click.pass_context
def execution_logs(ctx, execution_id, follow):
    """Get logs and status for a specific execution."""
    try:
        success = asyncio.run(
            get_execution_logs(
                execution_id=execution_id,
                follow=follow,
                verbose=ctx.obj.get("verbose", False),
            )
        )
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        if follow:
            click.echo("\n Stopped following execution.", err=True)
        sys.exit(1)
    except Exception as e:
        if ctx.obj.get("verbose"):
            logging.exception("Execution logs command failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)


# ======= Webhook Commands =======


@cli.group()
@click.pass_context
def webhook(ctx):
    """Webhook management commands."""
    pass


@webhook.command("list")
@click.option(
    "--api-key-only",
    is_flag=True,
    help="Show only webhooks created with current API key",
)
@click.pass_context
def webhook_list(ctx, api_key_only):
    """List all webhook URLs for your organization."""
    try:
        from .core.webhooks import list_webhooks

        success = asyncio.run(
            list_webhooks(
                api_key_only=api_key_only, verbose=ctx.obj.get("verbose", False)
            )
        )
        if not success:
            sys.exit(1)
    except Exception as e:
        if ctx.obj.get("verbose"):
            logging.exception("Webhook list command failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)


# ======= Memory Commands =======


@cli.group()
@click.pass_context
def memory(ctx):
    """Memory management commands."""
    pass


@memory.command("status")
@click.option("--project", help="Project name (default: current directory)")
@click.pass_context
def memory_status(ctx, project):
    """Show production memory status from cloud."""
    try:
        from .core.memory_commands import show_memory_status

        asyncio.run(
            show_memory_status(project=project, verbose=ctx.obj.get("verbose", False))
        )
    except KeyboardInterrupt:
        click.echo("\n Operation cancelled.", err=True)
        sys.exit(1)
    except Exception as e:
        if ctx.obj.get("verbose"):
            logging.exception("Memory status command failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)


@memory.command("show")
@click.argument("workspace")
@click.option("--full", is_flag=True, help="Download complete files")
@click.option("--limit", default=50, help="Max memories to show")
@click.option("--project", help="Project name (default: current directory)")
@click.pass_context
def memory_show(ctx, workspace, full, limit, project):
    """Show production memory contents from cloud."""
    try:
        from .core.memory_commands import show_workspace_memory

        asyncio.run(
            show_workspace_memory(
                workspace=workspace,
                full=full,
                limit=limit,
                project=project,
                verbose=ctx.obj.get("verbose", False),
            )
        )
    except KeyboardInterrupt:
        click.echo("\n Operation cancelled.", err=True)
        sys.exit(1)
    except Exception as e:
        if ctx.obj.get("verbose"):
            logging.exception("Memory show command failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)


# ======= Secrets Commands =======


@cli.group()
@click.pass_context
def secrets(ctx):
    """Manage cloud secrets (API keys, credentials)."""
    pass


@secrets.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_context
def secrets_set(ctx, key, value):
    """Store or update an encrypted secret."""
    try:
        from .core.secrets_commands import set_secret

        success = asyncio.run(
            set_secret(key, value, verbose=ctx.obj.get("verbose", False))
        )
        if not success:
            sys.exit(1)
    except Exception as e:
        if ctx.obj.get("verbose"):
            logging.exception("Secrets set failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)


@secrets.command("list")
@click.pass_context
def secrets_list(ctx):
    """List stored secret key names (values are never shown)."""
    try:
        from .core.secrets_commands import list_secrets

        success = asyncio.run(list_secrets(verbose=ctx.obj.get("verbose", False)))
        if not success:
            sys.exit(1)
    except Exception as e:
        if ctx.obj.get("verbose"):
            logging.exception("Secrets list failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)


@secrets.command("remove")
@click.argument("key")
@click.pass_context
def secrets_remove(ctx, key):
    """Delete a stored secret."""
    try:
        from .core.secrets_commands import remove_secret

        success = asyncio.run(remove_secret(key, verbose=ctx.obj.get("verbose", False)))
        if not success:
            sys.exit(1)
    except Exception as e:
        if ctx.obj.get("verbose"):
            logging.exception("Secrets remove failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)


@secrets.command("import")
@click.argument("env_file", default=".env")
@click.pass_context
def secrets_import(ctx, env_file):
    """Import secrets from a .env file into secure cloud storage."""
    try:
        from .core.secrets_commands import import_env

        success = asyncio.run(
            import_env(env_file, verbose=ctx.obj.get("verbose", False))
        )
        if not success:
            sys.exit(1)
    except Exception as e:
        if ctx.obj.get("verbose"):
            logging.exception("Secrets import failed")
        click.echo(f" Error: {str(e)}", err=True)
        sys.exit(1)


# ======= Main Entry Point =======


def main():
    """Main CLI entry point."""
    _check_first_time_usage()
    cli()


if __name__ == "__main__":
    main()
