"""
Memory management commands for viewing production memory in cloud.

Provides read-only access to memory workspaces via the Daita API.
No direct S3 or AWS credentials required — the backend proxies all access.
"""

import asyncio
import aiohttp
import ssl
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

from ..utils import get_api_endpoint, _CLI_VERSION


def _get_api_headers() -> dict:
    api_key = os.getenv('DAITA_API_KEY')
    if not api_key:
        raise ValueError(
            "DAITA_API_KEY not set.\n"
            "\n"
            "Set your API key with:\n"
            "  export DAITA_API_KEY='your-key'\n"
            "\n"
            "Get your API key at: https://daita-tech.io/app/dashboard"
        )
    return {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": f"Daita-CLI/{_CLI_VERSION}",
    }


def _make_session() -> aiohttp.ClientSession:
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = True
    ssl_context.verify_mode = ssl.CERT_REQUIRED
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    return aiohttp.ClientSession(connector=connector)


def _format_time_ago(iso_str: Optional[str]) -> str:
    if not iso_str:
        return "unknown"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        diff = datetime.now(dt.tzinfo) - dt
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
        hours = diff.seconds // 3600
        if hours > 0:
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        minutes = diff.seconds // 60
        if minutes > 0:
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        return "just now"
    except Exception:
        return "unknown"


async def show_memory_status(project: Optional[str] = None, verbose: bool = False):
    """
    Show production memory status from cloud.

    Lists all memory workspaces with statistics including fact counts,
    log counts, last update time, and total size.
    """
    try:
        headers = _get_api_headers()
        api_endpoint = get_api_endpoint()

        if not project:
            project = Path.cwd().name

        params = {"project": project}

        async with _make_session() as session:
            url = f"{api_endpoint}/api/v1/memory/status"
            async with session.get(
                url,
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status == 401:
                    raise ValueError(
                        "Invalid API key or authentication failed.\n"
                        "\n"
                        "Get a new API key at: https://daita-tech.io/app/dashboard"
                    )
                if resp.status == 422:
                    raise ValueError("project parameter is required.")
                if resp.status != 200:
                    error = await resp.text()
                    raise ValueError(f"API request failed (HTTP {resp.status}): {error}")

                data = await resp.json()

        print("")
        print("Production Memory Status (Cloud)")
        print("\u2501" * 60)
        print(f"Project: {project}")
        print("")

        workspaces = data.get("workspaces", [])
        if not workspaces:
            print("No memory workspaces in cloud")
            print("")
            print("Workspaces are created automatically when agents use memory.")
            print("Deploy your project and run agents to create memory workspaces.")
            print("")
            return

        total_memories = 0
        total_size = 0

        for ws in workspaces:
            name = ws.get("workspace", "unknown")
            fact_count = ws.get("fact_count", 0)
            log_count = ws.get("log_count", 0)
            size_bytes = ws.get("size_bytes", 0)
            time_str = _format_time_ago(ws.get("last_updated"))

            print(f"Workspace: {name}")
            print(f"  Long-term memories: {fact_count} facts")
            print(f"  Daily logs: {log_count} {'day' if log_count == 1 else 'days'}")
            print(f"  Last updated: {time_str}")
            print(f"  Size: {size_bytes / 1024:.1f} KB")
            print("")

            total_memories += fact_count
            total_size += size_bytes

        print(
            f"Total: {len(workspaces)} workspace{'s' if len(workspaces) != 1 else ''}, "
            f"{total_memories} memories, {total_size / 1024:.1f} KB"
        )
        print("")

    except ValueError as e:
        print(f" {e}")
        print("")
    except aiohttp.ClientConnectorError:
        print(" Cannot connect to Daita API.")
        print("   Check your internet connection and try again.")
        print("")
    except asyncio.TimeoutError:
        print(" Request timed out. Check your internet connection.")
        print("")
    except KeyboardInterrupt:
        print("")
        print(" Operation cancelled")
        print("")
    except Exception as e:
        print(f" Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        print("")


async def show_workspace_memory(
    workspace: str,
    full: bool = False,
    limit: int = 50,
    project: Optional[str] = None,
    verbose: bool = False
):
    """
    Show workspace memory contents from cloud.

    Displays MEMORY.md contents and recent daily logs.
    With --full, downloads all workspace files as a zip to ./memory-exports/.
    """
    try:
        headers = _get_api_headers()
        api_endpoint = get_api_endpoint()

        if not project:
            project = Path.cwd().name

        params = {"project": project, "limit": limit}

        async with _make_session() as session:
            url = f"{api_endpoint}/api/v1/memory/workspaces/{workspace}"
            async with session.get(
                url,
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status == 401:
                    raise ValueError(
                        "Invalid API key or authentication failed.\n"
                        "\n"
                        "Get a new API key at: https://daita-tech.io/app/dashboard"
                    )
                if resp.status == 404:
                    print(f" Workspace '{workspace}' not found in cloud")
                    print("")
                    print("Run 'daita memory status' to list available workspaces.")
                    print("")
                    return
                if resp.status != 200:
                    error = await resp.text()
                    raise ValueError(f"API request failed (HTTP {resp.status}): {error}")

                data = await resp.json()

            time_str = _format_time_ago(data.get("last_updated"))

            print("")
            print(f"Production Memory: {workspace}")
            print("\u2501" * 60)
            print(f"Last updated: {time_str}")
            print("")

            # Display MEMORY.md
            memory_md = data.get("memory_md", "")
            total_entries = data.get("total_entries", 0)

            print(f"MEMORY.md ({total_entries} entries)")
            print("\u2501" * 60)
            print("")

            if memory_md:
                sections = _parse_memory_sections(memory_md)
                entries_shown = 0
                for section_name, facts in sections.items():
                    if entries_shown >= limit:
                        remaining = total_entries - entries_shown
                        if remaining > 0:
                            print(f"... and {remaining} more entries (use --limit to see more)")
                        break
                    print(f"## {section_name}")
                    for fact in facts:
                        if entries_shown >= limit:
                            break
                        print(f"{fact}")
                        entries_shown += 1
                    print("")
            else:
                print("No structured memories found")
                print("")

            # Display recent daily logs
            print("Recent Daily Logs")
            print("\u2501" * 60)
            print("")

            recent_logs = data.get("recent_logs", [])
            if not recent_logs:
                print("No daily logs yet")
                print("")
            else:
                logs_to_show = recent_logs if full else recent_logs[:5]
                for log in logs_to_show:
                    date = log.get("date", "unknown")
                    entry_count = log.get("entry_count", 0)
                    preview = log.get("preview", [])
                    print(f"{date}.md ({entry_count} entries)")
                    for line in preview[:3]:
                        print(f"  {line}")
                    if entry_count > 3:
                        print(f"  ... and {entry_count - 3} more entries")
                    print("")

            # --full: download zip
            if full:
                print("Downloading Complete Memory Files")
                print("\u2501" * 60)
                print("")

                dl_url_endpoint = f"{api_endpoint}/api/v1/memory/workspaces/{workspace}/download"
                async with session.get(
                    dl_url_endpoint,
                    headers=headers,
                    params={"project": project},
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as dl_resp:
                    if dl_resp.status != 200:
                        print(" Could not generate download URL")
                        print("")
                        return
                    dl_data = await dl_resp.json()
                    download_url = dl_data.get("download_url")

                if download_url:
                    export_dir = Path('./memory-exports') / workspace
                    export_dir.mkdir(parents=True, exist_ok=True)
                    zip_path = export_dir / 'memory-export.zip'

                    # Download the presigned URL (no auth headers needed)
                    async with aiohttp.ClientSession() as plain_session:
                        async with plain_session.get(
                            download_url,
                            timeout=aiohttp.ClientTimeout(total=60)
                        ) as zip_resp:
                            if zip_resp.status == 200:
                                zip_path.write_bytes(await zip_resp.read())
                                print(f"Downloaded to: {zip_path}")
                                print("")
                            else:
                                print(" Download failed")
                                print("")

    except ValueError as e:
        print(f" {e}")
        print("")
    except aiohttp.ClientConnectorError:
        print(" Cannot connect to Daita API.")
        print("   Check your internet connection and try again.")
        print("")
    except asyncio.TimeoutError:
        print(" Request timed out. Check your internet connection.")
        print("")
    except KeyboardInterrupt:
        print("")
        print(" Operation cancelled")
        print("")
    except Exception as e:
        print(f" Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        print("")


def _parse_memory_sections(content: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current_section = "General"
    for line in content.split('\n'):
        if line.startswith('## '):
            current_section = line[3:].strip()
            if current_section not in sections:
                sections[current_section] = []
        elif line.strip().startswith(('•', '*', '-')) and len(line.strip()) > 2:
            sections.setdefault(current_section, []).append(line.strip())
    return sections
