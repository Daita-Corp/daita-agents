"""
Remote execution command for running agents and workflows in the cloud.

This allows users to execute their deployed agents/workflows remotely
from the command line, with real-time status updates and result streaming.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import aiohttp
import click
import yaml
from ..utils import find_project_root, get_api_endpoint, _CLI_VERSION

# ---------------------------------------------------------------------------
# Terminal / spinner helpers
# ---------------------------------------------------------------------------

_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def _use_ansi() -> bool:
    return (
        sys.stdout.isatty()
        and not os.environ.get("NO_COLOR")
        and os.environ.get("TERM") != "dumb"
    )


def _c(on: bool) -> dict:
    """Return ANSI color dict. Green-only palette to match brand."""
    if on:
        return dict(
            GREEN="\033[92m",  # bright green — active / success
            DIM="\033[2m",  # dimmed — secondary info
            BOLD="\033[1m",  # bold — agent name
            RESET="\033[0m",
            CLEAR="\r\033[K",  # carriage-return + erase to end of line
        )
    return dict(GREEN="", DIM="", BOLD="", RESET="", CLEAR="\r")


def _fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    return f"{s // 60}m {s % 60:02d}s"


async def run_remote_execution(
    target_name: str,
    target_type: str = "agent",
    environment: str = "production",
    data_file: Optional[str] = None,
    data_json: Optional[str] = None,
    task: str = "process",
    follow: bool = False,
    timeout: int = 300,
    verbose: bool = False,
):
    """
    Execute an agent or workflow remotely in the cloud.

    Args:
        target_name: Name of the agent or workflow to execute
        target_type: "agent" or "workflow"
        environment: Environment to execute in (production, staging)
        data_file: Path to JSON file containing input data
        data_json: JSON string containing input data
        task: Task to execute (for agents only)
        follow: Whether to follow execution progress
        timeout: Execution timeout in seconds
        verbose: Enable verbose output
    """

    # Load environment variables from .env file if present
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # dotenv is optional

    # Get API credentials
    api_key = os.getenv("DAITA_API_KEY")
    api_base = get_api_endpoint()

    if not api_key:
        click.echo(" DAITA_API_KEY not found", err=True)
        click.echo("   Get your API key from daita-tech.io", err=True)
        click.echo("   Set it with: export DAITA_API_KEY='your-key-here'", err=True)
        return False

    # Non-blocking warning if no cloud secrets are configured
    try:
        async with aiohttp.ClientSession() as _check_session:
            async with _check_session.get(
                f"{api_base}/api/v1/secrets",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": f"Daita-CLI/{_CLI_VERSION}",
                },
                timeout=aiohttp.ClientTimeout(total=5),
            ) as _resp:
                if _resp.status == 200:
                    _data = await _resp.json()
                    if not _data.get("keys"):
                        click.echo(
                            "  Warning: No cloud secrets configured. If your agent needs API keys, run:",
                            err=True,
                        )
                        click.echo("    daita secrets import .env", err=True)
                        click.echo("", err=True)
    except Exception:
        pass  # Never block execution on secrets check failure

    # Prepare input data
    input_data = {}
    if data_file:
        try:
            with open(data_file, "r") as f:
                input_data = json.load(f)
            if verbose:
                click.echo(f" Loaded data from {data_file}")
        except Exception as e:
            click.echo(f" Failed to load data file: {e}", err=True)
            return False
    elif data_json:
        try:
            input_data = json.loads(data_json)
        except Exception as e:
            click.echo(f" Invalid JSON data: {e}", err=True)
            return False

    # Prepare execution request
    request_data = {
        "data": input_data,
        "environment": environment,
        "timeout_seconds": timeout,
        "execution_source": "cli",
        "source_metadata": {
            "cli_version": _CLI_VERSION,
            "command": f"daita run {target_name}",
        },
    }

    # Validate agent exists and get file name for execution
    if verbose:
        click.echo(f" Looking up agent '{target_name}'...")

    agent_info = validate_agent_exists(target_name, target_type)
    if not agent_info:
        return False

    file_name = agent_info["file_name"]
    display_name = agent_info.get("display_name", file_name)

    if verbose:
        click.echo(f" Found agent: {file_name} → '{display_name}'")
        click.echo(f" Executing with file name: '{file_name}'")

    # Add target-specific fields using file name (API/Lambda expects this)
    if target_type == "agent":
        request_data["agent_name"] = file_name
        request_data["task"] = task
    else:
        request_data["workflow_name"] = file_name

    if verbose:
        click.echo(f" Executing {target_type} '{target_name}' in {environment}")
        if input_data:
            click.echo(f" Input data: {len(str(input_data))} characters")

    # Execute the request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": f"Daita-CLI/{_CLI_VERSION}",
    }

    try:
        async with aiohttp.ClientSession() as session:
            # Submit execution request
            async with session.post(
                f"{api_base}/api/v1/executions/execute",
                headers=headers,
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:

                if response.status != 200:
                    error_data = await response.json()
                    error_detail = error_data.get("detail", "Unknown error")
                    click.echo(f" Execution failed: {error_detail}", err=True)

                    # Provide helpful guidance based on error type
                    if response.status == 404 and "No deployment found" in error_detail:
                        click.echo(f" Possible causes:", err=True)
                        click.echo(
                            f"   • Agent not deployed: daita push {environment}",
                            err=True,
                        )
                        click.echo(
                            f"   • Wrong agent name (using: '{file_name}')", err=True
                        )
                        click.echo(
                            f"   • Check deployed agents: daita status", err=True
                        )

                    return False

                result = await response.json()
                execution_id = result["execution_id"]

                if verbose:
                    click.echo(f" Execution ID: {execution_id}")

                # All executions are now asynchronous
                if follow:
                    return await _follow_execution(
                        session, api_base, headers, execution_id, display_name, verbose
                    )
                else:
                    return await _poll_for_result(
                        session, api_base, headers, execution_id, display_name, verbose
                    )

    except aiohttp.ClientError as e:
        click.echo(f" Network error: {e}", err=True)
        return False
    except Exception as e:
        click.echo(f" Unexpected error: {e}", err=True)
        return False


async def _spinner_loop(state: dict, agent_name: str, c: dict) -> None:
    """Background task: redraws the spinner line at ~12 FPS."""
    frame_idx = 0
    while not state["done"]:
        frame = _SPINNER_FRAMES[frame_idx % len(_SPINNER_FRAMES)]
        status = state["status"]
        elapsed = _fmt_elapsed(time.time() - state["start"])
        sys.stdout.write(
            f"{c['CLEAR']}{c['GREEN']}{frame}{c['RESET']}  "
            f"{c['BOLD']}{agent_name}{c['RESET']}  "
            f"{c['DIM']}{status}  {elapsed}{c['RESET']}"
        )
        sys.stdout.flush()
        frame_idx += 1
        await asyncio.sleep(0.08)


async def _await_execution(
    session: aiohttp.ClientSession,
    api_base: str,
    headers: Dict[str, str],
    execution_id: str,
    agent_name: str,
    verbose: bool,
    poll_interval: float = 0.5,
    max_wait_s: int = 360,
) -> bool:
    """
    Poll execution status with an animated spinner until completion.
    Used by both the default and --follow paths.
    """
    ansi = _use_ansi()
    c = _c(ansi)

    state = {"status": "queued", "done": False, "start": time.time()}

    anim_task = (
        asyncio.create_task(_spinner_loop(state, agent_name, c)) if ansi else None
    )

    # Non-TTY: print a single submission line so something is visible.
    if not ansi:
        click.echo(f"  Running {agent_name}...")

    deadline = time.time() + max_wait_s
    last_plain_status = None

    try:
        while time.time() < deadline:
            await asyncio.sleep(poll_interval)

            async with session.get(
                f"{api_base}/api/v1/executions/{execution_id}",
                headers=headers,
            ) as response:
                if response.status != 200:
                    state["done"] = True
                    if ansi:
                        sys.stdout.write(c["CLEAR"])
                        sys.stdout.flush()
                    click.echo("  Failed to get execution status", err=True)
                    return False

                result = await response.json()
                status = result["status"]
                state["status"] = status

                # Non-TTY: emit a line only when status changes.
                if not ansi and status != last_plain_status:
                    click.echo(
                        f"  [{_fmt_elapsed(time.time() - state['start'])}] {status}"
                    )
                    last_plain_status = status

                if status in ("completed", "success"):
                    state["done"] = True
                    elapsed = _fmt_elapsed(time.time() - state["start"])
                    if ansi:
                        sys.stdout.write(
                            f"{c['CLEAR']}{c['GREEN']}✓{c['RESET']}  "
                            f"{c['BOLD']}{agent_name}{c['RESET']}  "
                            f"{c['DIM']}completed  {elapsed}{c['RESET']}\n"
                        )
                        sys.stdout.flush()
                    _display_result(result, verbose)
                    return True

                elif status in ("failed", "error"):
                    state["done"] = True
                    elapsed = _fmt_elapsed(time.time() - state["start"])
                    if ansi:
                        sys.stdout.write(
                            f"{c['CLEAR']}{c['DIM']}✗{c['RESET']}  "
                            f"{c['BOLD']}{agent_name}{c['RESET']}  "
                            f"{c['DIM']}failed  {elapsed}{c['RESET']}\n"
                        )
                        sys.stdout.flush()
                    if result.get("error"):
                        click.echo(f"\n  Error: {result['error']}")
                    return False

                elif status == "cancelled":
                    state["done"] = True
                    if ansi:
                        sys.stdout.write(f"{c['CLEAR']}  {agent_name}  cancelled\n")
                        sys.stdout.flush()
                    return False

    except KeyboardInterrupt:
        state["done"] = True
        if ansi:
            sys.stdout.write(f"{c['CLEAR']}")
            sys.stdout.flush()
        click.echo(f"\n  Detached — execution continues in background")
        click.echo(f"  Check status: daita execution-logs {execution_id}")
        return False
    except Exception as e:
        state["done"] = True
        if ansi:
            sys.stdout.write(f"{c['CLEAR']}")
            sys.stdout.flush()
        click.echo(f"  Error: {e}", err=True)
        return False
    finally:
        state["done"] = True
        if anim_task:
            anim_task.cancel()
            await asyncio.gather(anim_task, return_exceptions=True)

    # Deadline exceeded
    if ansi:
        sys.stdout.write(f"{c['CLEAR']}")
        sys.stdout.flush()
    elapsed = _fmt_elapsed(time.time() - state["start"])
    click.echo(f"  Execution still running after {elapsed}")
    click.echo(f"  Check status: daita execution-logs {execution_id}")
    return False


async def _follow_execution(
    session: aiohttp.ClientSession,
    api_base: str,
    headers: Dict[str, str],
    execution_id: str,
    agent_name: str,
    verbose: bool,
) -> bool:
    """--follow mode: spinner + poll until completion."""
    return await _await_execution(
        session,
        api_base,
        headers,
        execution_id,
        agent_name,
        verbose,
        poll_interval=2.0,
        max_wait_s=900,
    )


async def _poll_for_result(
    session: aiohttp.ClientSession,
    api_base: str,
    headers: Dict[str, str],
    execution_id: str,
    agent_name: str,
    verbose: bool,
) -> bool:
    """Default (non-follow) mode: spinner + poll until completion."""
    return await _await_execution(
        session,
        api_base,
        headers,
        execution_id,
        agent_name,
        verbose,
        poll_interval=0.5,
        max_wait_s=360,
    )


def _display_result(result: Dict[str, Any], verbose: bool):
    """Display execution results."""
    ansi = _use_ansi()
    c = _c(ansi)

    click.echo("")

    # Metadata row
    meta_parts = []
    if result.get("duration_ms"):
        ms = float(result["duration_ms"])
        if ms < 1000:
            meta_parts.append(f"duration {ms:.0f}ms")
        elif ms < 60000:
            meta_parts.append(f"duration {ms / 1000:.1f}s")
        else:
            meta_parts.append(f"duration {ms / 60000:.1f}m")
    if result.get("memory_used_mb"):
        meta_parts.append(f"memory {result['memory_used_mb']:.0f}MB")
    if meta_parts:
        click.echo(f"  {c['DIM']}{' · '.join(meta_parts)}{c['RESET']}")

    # Result body
    result_data = result.get("result")
    if not result_data:
        click.echo("")
        return

    click.echo("")

    if verbose:
        result_json = json.dumps(result_data, indent=2, default=str)
        for line in result_json.split("\n"):
            click.echo(f"  {line}")
    elif isinstance(result_data, str):
        if len(result_data) > 400:
            click.echo(f"  {result_data[:400]}…")
            click.echo(f"  {c['DIM']}(truncated — use -v for full output){c['RESET']}")
        else:
            click.echo(f"  {result_data}")
    elif isinstance(result_data, dict):
        if "message" in result_data:
            msg = result_data["message"]
            click.echo(f"  {msg[:200]}{'…' if len(msg) > 200 else ''}")
        elif "result" in result_data:
            inner = str(result_data["result"])
            click.echo(f"  {inner[:200]}{'…' if len(inner) > 200 else ''}")
        else:
            # Show top-level keys as a brief summary
            keys = [k for k in result_data if not k.startswith("_")][:6]
            for k in keys:
                v = result_data[k]
                v_str = str(v)[:80] + ("…" if len(str(v)) > 80 else "")
                click.echo(f"  {c['DIM']}{k}:{c['RESET']} {v_str}")
        if not verbose:
            click.echo(f"  {c['DIM']}Use -v for full output{c['RESET']}")
    else:
        click.echo(f"  {result_data}")

    click.echo("")


async def list_remote_executions(
    limit: int = 10,
    status: Optional[str] = None,
    target_type: Optional[str] = None,
    environment: Optional[str] = None,
    verbose: bool = False,
):
    """List recent executions with filtering."""

    api_key = os.getenv("DAITA_API_KEY")
    api_base = get_api_endpoint()

    if not api_key:
        click.echo(" DAITA_API_KEY not found", err=True)
        return False

    # Build query parameters
    params = {"limit": limit}
    if status:
        params["status"] = status
    if target_type:
        params["target_type"] = target_type
    if environment:
        params["environment"] = environment

    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": f"Daita-CLI/{_CLI_VERSION}",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{api_base}/api/v1/executions", headers=headers, params=params
            ) as response:

                if response.status != 200:
                    try:
                        error_data = await response.json()
                        error_msg = error_data.get("detail", "Unknown error")
                    except:
                        error_msg = await response.text()
                    click.echo(f" Failed to list executions: {error_msg}", err=True)
                    if verbose:
                        click.echo(f"   Status code: {response.status}", err=True)
                    return False

                executions = await response.json()

                if not executions:
                    click.echo(" No executions found")
                    return True

                # Display executions header
                click.echo(f"\n Recent Executions ({len(executions)})")
                click.echo(" " + "=" * 70)
                click.echo(
                    f" {'Status':<10} {'Agent/Workflow':<25} {'Time & Duration'}"
                )
                click.echo(" " + "-" * 70)

                for execution in executions:
                    # Map status to friendly text labels
                    status_label = {
                        "completed": "success",
                        "success": "success",
                        "failed": "error",
                        "error": "error",
                        "running": "running",
                        "queued": "queued",
                        "cancelled": "cancelled",
                        "started": "running",
                    }.get(execution["status"], execution["status"])

                    # Format time
                    created_at = execution["created_at"]
                    if "T" in created_at:
                        time_str = created_at.split("T")[1].split(".")[0]
                    else:
                        time_str = created_at

                    # Format duration
                    duration_str = "N/A"
                    if execution.get("duration_ms"):
                        ms = execution["duration_ms"]
                        if ms < 1000:
                            duration_str = f"{ms}ms"
                        elif ms < 60000:
                            duration_str = f"{ms/1000:.1f}s"
                        else:
                            duration_str = f"{ms/60000:.1f}m"

                    # Format agent/workflow name with type
                    target_info = (
                        f"{execution['target_name']} ({execution['target_type']})"
                    )

                    # Format time and duration info
                    time_duration = f"{time_str} ({duration_str})"

                    # Display execution line
                    click.echo(f" {status_label:<10} {target_info:<25} {time_duration}")

                    # Show full execution ID on next line (indented)
                    click.echo(f" {'':>10} ID: {execution['execution_id']}")

                    if verbose and execution.get("error"):
                        click.echo(f" {'':>10} Error: {execution['error']}")

                    click.echo()  # Blank line between executions

                return True

    except Exception as e:
        click.echo(f" Failed to list executions: {e}", err=True)
        if verbose:
            import traceback

            click.echo(traceback.format_exc(), err=True)
        return False


async def get_execution_logs(
    execution_id: str, follow: bool = False, verbose: bool = False
):
    """Get logs for a specific execution."""

    api_key = os.getenv("DAITA_API_KEY")
    api_base = get_api_endpoint()

    if not api_key:
        click.echo(" DAITA_API_KEY not found", err=True)
        return False

    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": f"Daita-CLI/{_CLI_VERSION}",
    }

    try:
        async with aiohttp.ClientSession() as session:
            if follow:
                # Follow mode - continuously check status
                await _follow_execution(
                    session, api_base, headers, execution_id, verbose
                )
            else:
                # One-time status check
                async with session.get(
                    f"{api_base}/api/v1/executions/{execution_id}", headers=headers
                ) as response:

                    if response.status == 404:
                        click.echo(" Execution not found", err=True)
                        return False
                    elif response.status != 200:
                        error_data = await response.json()
                        click.echo(
                            f" Failed to get execution: {error_data.get('detail', 'Unknown error')}",
                            err=True,
                        )
                        return False

                    result = await response.json()

                    # Display execution info
                    click.echo(f" Execution: {result['execution_id']}")
                    click.echo(
                        f" Target: {result['target_name']} ({result['target_type']})"
                    )
                    click.echo(f" Environment: {result['environment']}")
                    click.echo(f" Status: {result['status']}")

                    if result.get("created_at"):
                        click.echo(f" Created: {result['created_at']}")

                    _display_result(result, verbose)

                    return True

    except Exception as e:
        click.echo(f" Error: {e}", err=True)
        return False


def validate_agent_exists(
    target_name: str, target_type: str = "agent"
) -> Optional[dict]:
    """
    Validate agent exists and return file name and display name.

    Args:
        target_name: Name of the agent/workflow (file name or display name)
        target_type: "agent" or "workflow"

    Returns:
        Dict with 'file_name' and 'display_name' if found, None otherwise
    """
    project_root = find_project_root()
    if not project_root:
        click.echo(" No daita-project.yaml found")
        click.echo(" Run 'daita init' to create a project")
        return None

    config_file = project_root / "daita-project.yaml"
    if not config_file.exists():
        click.echo(" No daita-project.yaml found")
        click.echo(" Run 'daita init' to create a project")
        return None

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        click.echo(f" Failed to read daita-project.yaml: {e}")
        return None

    if not config:
        click.echo(" Invalid daita-project.yaml file")
        return None

    # Get the appropriate component list
    component_key = "agents" if target_type == "agent" else "workflows"
    components = config.get(component_key, [])

    # Find the component by name OR display_name
    component = next((c for c in components if c.get("name") == target_name), None)

    # If not found by name, try finding by display_name
    if not component:
        component = next(
            (c for c in components if c.get("display_name") == target_name), None
        )

    if not component:
        available_names = [c.get("name", "unknown") for c in components]
        click.echo(f" {target_type.title()} '{target_name}' not found in project")
        if available_names:
            click.echo(
                f" Available {component_key} (use file names): {', '.join(available_names)}"
            )
        else:
            click.echo(
                f" No {component_key} found. Create one with: daita create {target_type}"
            )
        click.echo(" Use file names for execution (e.g., 'my_agent' not 'My Agent')")
        return None

    file_name = component.get("name")
    display_name = component.get("display_name")

    if not file_name:
        click.echo(f" {target_type.title()} missing name in config")
        return None

    return {"file_name": file_name, "display_name": display_name or file_name}
