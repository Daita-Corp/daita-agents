"""
Webhook CLI commands for listing and managing webhook configurations.

Provides user-friendly access to webhook URLs and configurations
with organization-level security and optional API key filtering.
"""

import os
import asyncio
import aiohttp
import ssl
from typing import Dict, Any
from ..utils import get_api_endpoint, _CLI_VERSION


async def list_webhooks(api_key_only: bool = False, verbose: bool = False) -> bool:
    """
    List all webhook URLs for the authenticated organization.

    Args:
        api_key_only: If True, filter to webhooks created with current API key only
        verbose: Enable verbose output

    Returns:
        bool: True if successful, False otherwise
    """
    # Check for DAITA_API_KEY
    api_key = os.getenv("DAITA_API_KEY")
    if not api_key:
        print(" No DAITA_API_KEY found.")
        print("   Get your API key at https://daita-tech.io/app/dashboard")
        print("   Then: export DAITA_API_KEY='your-key-here'")
        return False

    try:
        # Get secure API endpoint
        api_endpoint = get_api_endpoint()
        url = f"{api_endpoint}/api/v1/webhooks/list"

        if api_key_only:
            url += "?api_key_only=true"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"Daita-CLI/{_CLI_VERSION}",
        }

        # Create secure SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED

        connector = aiohttp.TCPConnector(ssl=ssl_context)

        async with aiohttp.ClientSession(connector=connector) as session:
            if verbose:
                print(f"📡 Fetching webhooks from {url}")

            async with session.get(url, headers=headers, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    await _display_webhooks(data, verbose)
                    return True
                elif response.status == 401:
                    print(" Authentication failed - check your DAITA_API_KEY")
                    print("   Get a new API key at https://daita-tech.io/app/dashboard")
                    return False
                elif response.status == 404:
                    print(" No webhook configurations found.")
                    print(
                        "   Deploy a project with webhook configurations using 'daita push'"
                    )
                    return True
                else:
                    error_text = await response.text()
                    print(f" Failed to fetch webhooks (HTTP {response.status})")
                    if verbose:
                        print(f"   Details: {error_text}")
                    return False

    except aiohttp.ClientConnectorError:
        print(" Cannot connect to Daita API")
        print("   Check your internet connection and try again")
        return False
    except asyncio.TimeoutError:
        print(" Request timed out")
        print("   Try again or check your internet connection")
        return False
    except Exception as e:
        print(f" Error fetching webhooks: {str(e)}")
        if verbose:
            import traceback

            traceback.print_exc()
        return False


async def _display_webhooks(data: Dict[str, Any], verbose: bool = False) -> None:
    """
    Display webhook information in a user-friendly format.

    Args:
        data: Response data from the webhook list API
        verbose: Enable verbose output
    """
    org_id = data.get("organization_id")
    org_name = data.get("organization_name", f"Organization {org_id}")
    total_webhooks = data.get("total_webhooks", 0)
    agent_webhooks = data.get("agent_webhooks", {})
    workflow_webhooks = data.get("workflow_webhooks", {})

    if total_webhooks == 0:
        print("")
        print(" No webhook URLs found")
        print("")
        print("💡 To create webhooks:")
        print("   1. Add webhook configurations to your daita-project.yaml")
        print("   2. Deploy with 'daita push production'")
        print("")
        return

    print("")
    print(f" Webhook URLs (Organization: {org_name})")
    print("")

    # Display agent webhooks
    if agent_webhooks:
        print("Agent Webhooks:")
        for agent_name, webhooks in agent_webhooks.items():
            print(f"  {agent_name}:")
            for webhook in webhooks:
                slug = webhook["webhook_slug"]
                url = webhook["webhook_url"]
                print(f"    {slug:<20} → {url}")

                if verbose and webhook.get("field_mapping"):
                    print(f"      Field mapping: {webhook['field_mapping']}")
        print("")

    # Display workflow webhooks
    if workflow_webhooks:
        print("Workflow Webhooks:")
        for workflow_name, webhooks in workflow_webhooks.items():
            print(f"  {workflow_name}:")
            for webhook in webhooks:
                slug = webhook["webhook_slug"]
                url = webhook["webhook_url"]
                print(f"    {slug:<20} → {url}")

                if verbose and webhook.get("field_mapping"):
                    print(f"      Field mapping: {webhook['field_mapping']}")
        print("")

    # Add helpful usage information
    print("💡 Usage:")
    print("   • Copy webhook URLs to your external services (GitHub, Slack, etc.)")
    print("   • Send HTTP POST requests with JSON payloads")
    print("   • Field mapping will transform payloads automatically")
    print("")
    print("🔧 Example:")
    print("   curl -X POST '<webhook-url>' \\")
    print("        -H 'Content-Type: application/json' \\")
    print('        -d \'{"your": "payload"}\'')
    print("")
