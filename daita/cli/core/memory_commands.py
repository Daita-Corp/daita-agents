"""
Memory management commands for viewing production memory in cloud.

Provides read-only access to S3-stored memory workspaces with
formatted output and comprehensive error handling.
"""

import boto3

import asyncio
import aiohttp
import ssl
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List


async def show_memory_status(project: Optional[str] = None, verbose: bool = False):
    """
    Show production memory status from cloud.

    Lists all memory workspaces in S3 with statistics including:
    - Number of long-term memories (facts)
    - Number of daily logs
    - Last update time
    - Total size

    Args:
        project: Project name (defaults to current directory name)
        verbose: Enable verbose output
    """
    try:
        # Get organization ID from API
        org_id = await _get_org_id_from_api(verbose)

        # Get project name
        if not project:
            project = Path.cwd().name

        # Setup S3 client
        s3 = boto3.client('s3')
        bucket = "daita-memory-us-east-1"
        prefix = f"orgs/{org_id}/projects/{project}/workspaces/"

        print("")
        print("Production Memory Status (Cloud)")
        print("━" * 60)
        print(f"Project: {project}")
        print(f"Organization: {org_id}")
        print(f"Location: s3://{bucket}/{prefix}")
        print("")

        # List workspaces
        try:
            response = s3.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                Delimiter='/'
            )
        except s3.exceptions.NoSuchBucket:
            print(" Cloud memory bucket not found")
            print("   This may indicate no deployments have been made yet.")
            print("   Deploy with: daita push production")
            print("")
            return
        except Exception as e:
            if verbose:
                print(f" Error accessing S3: {str(e)}")
            print(" Cannot access cloud storage")
            print("   Check your AWS credentials and permissions")
            print("")
            return

        # Extract workspace names
        workspaces = []
        for prefix_obj in response.get('CommonPrefixes', []):
            workspace_name = prefix_obj['Prefix'].rstrip('/').split('/')[-1]
            workspaces.append(workspace_name)

        if not workspaces:
            print("No memory workspaces in cloud")
            print("")
            print("Workspaces are created automatically when agents use memory.")
            print("Deploy your project and run agents to create memory workspaces.")
            print("")
            return

        # Gather stats for each workspace
        total_memories = 0
        total_size = 0

        for workspace in sorted(workspaces):
            ws_prefix = f"orgs/{org_id}/projects/{project}/workspaces/{workspace}/"

            # Get MEMORY.md stats
            memory_key = f"{ws_prefix}MEMORY.md"
            fact_count = 0
            memory_size = 0
            last_modified = None

            try:
                memory_obj = s3.head_object(Bucket=bucket, Key=memory_key)
                memory_size = memory_obj['ContentLength']
                last_modified = memory_obj['LastModified']

                # Download and count facts (lines with content)
                memory_response = s3.get_object(Bucket=bucket, Key=memory_key)
                memory_content = memory_response['Body'].read().decode('utf-8')

                # Count bullet points as facts
                fact_count = sum(1 for line in memory_content.split('\n')
                                if line.strip().startswith(('•', '*', '-')) and len(line.strip()) > 2)
            except s3.exceptions.NoSuchKey:
                pass
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not read MEMORY.md for {workspace}: {e}")

            # Get daily logs count and date range
            logs_prefix = f"{ws_prefix}logs/"
            log_files = []
            log_count = 0
            date_range = "None"

            try:
                logs_response = s3.list_objects_v2(Bucket=bucket, Prefix=logs_prefix)
                if 'Contents' in logs_response:
                    log_files = [obj['Key'] for obj in logs_response['Contents']]
                    log_count = len(log_files)

                    if log_files:
                        # Extract dates from filenames (YYYY-MM-DD.md)
                        dates = []
                        for log_file in log_files:
                            filename = Path(log_file).stem
                            try:
                                # Validate it's a date
                                datetime.strptime(filename, '%Y-%m-%d')
                                dates.append(filename)
                            except ValueError:
                                pass

                        if dates:
                            dates.sort()
                            date_range = f"{dates[0]} to {dates[-1]}"
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not list logs for {workspace}: {e}")

            # Calculate workspace size
            ws_size = memory_size
            try:
                ws_response = s3.list_objects_v2(Bucket=bucket, Prefix=ws_prefix)
                if 'Contents' in ws_response:
                    ws_size = sum(obj['Size'] for obj in ws_response['Contents'])
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not calculate size for {workspace}: {e}")

            # Format last modified time
            if last_modified:
                time_diff = datetime.now(last_modified.tzinfo) - last_modified
                if time_diff.days > 0:
                    time_str = f"{time_diff.days} day{'s' if time_diff.days != 1 else ''} ago"
                elif time_diff.seconds >= 3600:
                    hours = time_diff.seconds // 3600
                    time_str = f"{hours} hour{'s' if hours != 1 else ''} ago"
                elif time_diff.seconds >= 60:
                    minutes = time_diff.seconds // 60
                    time_str = f"{minutes} minute{'s' if minutes != 1 else ''} ago"
                else:
                    time_str = "just now"
            else:
                time_str = "unknown"

            # Display workspace info
            print(f"Workspace: {workspace}")
            print(f"  Long-term memories: {fact_count} facts")
            print(f"  Daily logs: {log_count} {'day' if log_count == 1 else 'days'} ({date_range})")
            print(f"  Last updated: {time_str}")
            print(f"  Size: {ws_size / 1024:.1f} KB")
            print("")

            total_memories += fact_count
            total_size += ws_size

        # Display summary
        print(f"Total: {len(workspaces)} workspace{'s' if len(workspaces) != 1 else ''}, "
              f"{total_memories} memories, {total_size / 1024:.1f} KB")
        print("")

    except ValueError as e:
        print(f" {str(e)}")
        print("")
    except KeyboardInterrupt:
        print("")
        print(" Operation cancelled")
        print("")
    except Exception as e:
        print(f" Error: {str(e)}")
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
    With --full flag, downloads all files to local directory.

    Args:
        workspace: Workspace name to show
        full: If True, download complete files to local directory
        limit: Maximum number of memories to display
        project: Project name (defaults to current directory name)
        verbose: Enable verbose output
    """
    try:
        # Get organization ID from API
        org_id = await _get_org_id_from_api(verbose)

        # Get project name
        if not project:
            project = Path.cwd().name

        # Setup S3 client
        s3 = boto3.client('s3')
        bucket = "daita-memory-us-east-1"
        prefix = f"orgs/{org_id}/projects/{project}/workspaces/{workspace}/"

        # Check if workspace exists
        try:
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
            if not response.get('Contents') and not response.get('CommonPrefixes'):
                print(f" Workspace '{workspace}' not found in cloud")
                print("")

                # Try to list available workspaces
                ws_prefix = f"orgs/{org_id}/projects/{project}/workspaces/"
                ws_response = s3.list_objects_v2(
                    Bucket=bucket,
                    Prefix=ws_prefix,
                    Delimiter='/'
                )
                available = [p['Prefix'].rstrip('/').split('/')[-1]
                            for p in ws_response.get('CommonPrefixes', [])]

                if available:
                    print("Available workspaces:")
                    for ws in sorted(available):
                        print(f"  • {ws}")
                else:
                    print("No workspaces found in this project.")
                print("")
                return
        except Exception as e:
            if verbose:
                print(f"Error checking workspace: {e}")
            print(f" Cannot access cloud storage")
            print("")
            return

        # Get last modified time
        try:
            memory_obj = s3.head_object(Bucket=bucket, Key=f"{prefix}MEMORY.md")
            last_modified = memory_obj['LastModified']
            time_diff = datetime.now(last_modified.tzinfo) - last_modified
            if time_diff.days > 0:
                time_str = f"{time_diff.days} day{'s' if time_diff.days != 1 else ''} ago"
            elif time_diff.seconds >= 3600:
                hours = time_diff.seconds // 3600
                time_str = f"{hours} hour{'s' if hours != 1 else ''} ago"
            elif time_diff.seconds >= 60:
                minutes = time_diff.seconds // 60
                time_str = f"{minutes} minute{'s' if minutes != 1 else ''} ago"
            else:
                time_str = "just now"
        except:
            time_str = "unknown"

        print("")
        print(f"Production Memory: {workspace}")
        print("━" * 60)
        print(f"Location: s3://{bucket}/{prefix}")
        print(f"Last updated: {time_str}")
        print("")

        # Download and display MEMORY.md
        memory_key = f"{prefix}MEMORY.md"
        try:
            obj = s3.get_object(Bucket=bucket, Key=memory_key)
            content = obj['Body'].read().decode('utf-8')

            # Count total entries
            total_entries = sum(1 for line in content.split('\n')
                              if line.strip().startswith(('•', '*', '-')) and len(line.strip()) > 2)

            print(f"MEMORY.md ({total_entries} entries)")
            print("━" * 60)
            print("")

            # Parse and display by sections
            sections = _parse_memory_sections(content)

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

            if not sections:
                print("No structured memories found")
                print("")
        except s3.exceptions.NoSuchKey:
            print("MEMORY.md not found")
            print("")
        except Exception as e:
            if verbose:
                print(f"Error reading MEMORY.md: {e}")
            print("Could not read MEMORY.md")
            print("")

        # List and display recent daily logs
        print("Recent Daily Logs")
        print("━" * 60)
        print("")

        logs_prefix = f"{prefix}logs/"
        try:
            response = s3.list_objects_v2(Bucket=bucket, Prefix=logs_prefix)

            if 'Contents' in response:
                log_files = sorted(
                    [obj for obj in response['Contents']],
                    key=lambda x: x['Key'],
                    reverse=True  # Most recent first
                )

                # Show last 5 logs (or all if --full)
                logs_to_show = log_files if full else log_files[:5]

                if not log_files:
                    print("No daily logs yet")
                    print("")
                else:
                    for log_obj in logs_to_show:
                        log_key = log_obj['Key']
                        date = Path(log_key).stem

                        try:
                            # Download log content
                            obj = s3.get_object(Bucket=bucket, Key=log_key)
                            log_content = obj['Body'].read().decode('utf-8')

                            # Count entries (lines starting with ##)
                            entry_count = log_content.count('\n## ')

                            print(f"{date}.md ({entry_count} entries)")

                            # Show first few entries
                            entries = _extract_log_entries(log_content)
                            for entry in entries[:3]:
                                time = entry.get('time', 'unknown')
                                agent = entry.get('agent', 'unknown')
                                summary = entry.get('summary', '')[:80]
                                print(f"  {time} [{agent}] - {summary}")

                            if len(entries) > 3:
                                print(f"  ... and {len(entries) - 3} more entries")
                            print("")
                        except Exception as e:
                            if verbose:
                                print(f"Warning: Could not read {date}.md: {e}")
                            continue

                    if not full and len(log_files) > 5:
                        print(f"Showing 5 most recent log files. Use --full to see all {len(log_files)} logs.")
                        print("")
            else:
                print("No daily logs yet")
                print("")
        except Exception as e:
            if verbose:
                print(f"Error reading daily logs: {e}")
            print("Could not read daily logs")
            print("")

        # Download all files if --full
        if full:
            print("Downloading Complete Memory Files")
            print("━" * 60)
            print("")

            export_dir = Path('./memory-exports') / workspace
            export_dir.mkdir(parents=True, exist_ok=True)

            try:
                # List all objects in workspace
                response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

                if 'Contents' in response:
                    downloaded = 0
                    for obj in response['Contents']:
                        key = obj['Key']
                        filename = key.replace(prefix, '')

                        # Skip internal files
                        if filename in ['metadata.json', '']:
                            continue

                        # Create safe local filename
                        local_path = export_dir / filename.replace('/', '-')

                        # Download file
                        s3.download_file(bucket, key, str(local_path))

                        size_kb = obj['Size'] / 1024
                        print(f"✓ {filename} → {local_path.name} ({size_kb:.1f} KB)")
                        downloaded += 1

                    print("")
                    print(f"Downloaded {downloaded} files to: {export_dir}")
                    print("")
                else:
                    print("No files to download")
                    print("")
            except Exception as e:
                if verbose:
                    print(f"Error downloading files: {e}")
                print("Could not download files")
                print("")

    except ValueError as e:
        print(f" {str(e)}")
        print("")
    except KeyboardInterrupt:
        print("")
        print(" Operation cancelled")
        print("")
    except Exception as e:
        print(f" Error: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        print("")


async def _get_org_id_from_api(verbose: bool = False) -> str:
    """
    Get organization ID from API using DAITA_API_KEY.

    Args:
        verbose: Enable verbose output

    Returns:
        Organization ID string

    Raises:
        ValueError: If API key is missing or invalid
    """
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

    api_endpoint = os.getenv(
        'DAITA_API_ENDPOINT',
        'https://api.daita-tech.io'
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Create secure SSL context
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = True
    ssl_context.verify_mode = ssl.CERT_REQUIRED

    connector = aiohttp.TCPConnector(ssl=ssl_context)

    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            url = f"{api_endpoint}/api/v1/user/me"

            if verbose:
                print(f"Fetching organization info from {url}")

            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['organization_id']
                elif response.status == 401:
                    raise ValueError(
                        "Invalid API key or authentication failed.\n"
                        "\n"
                        "Get a new API key at: https://daita-tech.io/app/dashboard"
                    )
                else:
                    error_text = await response.text()
                    raise ValueError(f"API request failed (HTTP {response.status}): {error_text}")
    except aiohttp.ClientConnectorError:
        raise ValueError(
            "Cannot connect to Daita API.\n"
            "\n"
            "Check your internet connection and try again."
        )
    except asyncio.TimeoutError:
        raise ValueError(
            "Request timed out.\n"
            "\n"
            "Check your internet connection and try again."
        )


def _parse_memory_sections(content: str) -> Dict[str, List[str]]:
    """
    Parse MEMORY.md into sections.

    Args:
        content: Raw content of MEMORY.md file

    Returns:
        Dictionary mapping section names to lists of facts
    """
    sections = {}
    current_section = "General"

    for line in content.split('\n'):
        # Section header
        if line.startswith('## '):
            current_section = line[3:].strip()
            if current_section not in sections:
                sections[current_section] = []
        # Bullet point (fact)
        elif line.strip().startswith(('•', '*', '-')) and len(line.strip()) > 2:
            fact = line.strip()
            if current_section not in sections:
                sections[current_section] = []
            sections[current_section].append(fact)

    return sections


def _extract_log_entries(content: str) -> List[Dict[str, str]]:
    """
    Extract log entries from daily log.

    Args:
        content: Raw content of daily log file

    Returns:
        List of entry dictionaries with time, agent, and summary
    """
    entries = []
    lines = content.split('\n')

    for line in lines:
        if line.startswith('## '):
            # Parse timestamp and agent
            # Format: ## HH:MM:SS [agent_id] [category]
            parts = line[3:].split('[')

            time = parts[0].strip() if parts else 'unknown'
            agent = 'unknown'

            if len(parts) > 1:
                agent = parts[1].split(']')[0]

            # Create summary (truncate if needed)
            summary = line[3:] if len(line) > 3 else ''

            entries.append({
                'time': time,
                'agent': agent,
                'summary': summary
            })

    return entries
