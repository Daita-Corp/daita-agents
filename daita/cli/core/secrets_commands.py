"""
Secrets management commands for storing API keys and credentials in the cloud.

Secrets are stored encrypted in AWS SSM Parameter Store, scoped to your
organization. The deployment package never contains secret values.
"""

import asyncio
import aiohttp
import ssl
import os
import re
from pathlib import Path
from typing import Optional

from ..utils import get_api_endpoint, _CLI_VERSION

_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")


def _get_headers() -> dict:
    api_key = os.getenv("DAITA_API_KEY")
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
        "Content-Type": "application/json",
        "User-Agent": f"Daita-CLI/{_CLI_VERSION}",
    }


def _make_session() -> aiohttp.ClientSession:
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = True
    ssl_context.verify_mode = ssl.CERT_REQUIRED
    return aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context))


def _validate_key(key: str) -> None:
    if not _KEY_RE.match(key):
        raise ValueError(
            f"Invalid secret key '{key}'. "
            "Keys must start with a letter and contain only letters, digits, "
            "and underscores (max 128 chars)."
        )


async def set_secret(key: str, value: str, verbose: bool = False) -> bool:
    """Store or update an encrypted secret in the cloud."""
    try:
        _validate_key(key)
        headers = _get_headers()
        api_endpoint = get_api_endpoint()

        async with _make_session() as session:
            url = f"{api_endpoint}/api/v1/secrets"
            async with session.post(
                url,
                headers=headers,
                json={"key": key, "value": value},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 401:
                    print(" Invalid API key or authentication failed.")
                    return False
                if resp.status == 422:
                    data = await resp.json()
                    print(f" Validation error: {data.get('detail', 'unknown')}")
                    return False
                if resp.status != 200:
                    error = await resp.text()
                    print(f" Failed to set secret (HTTP {resp.status}): {error}")
                    return False

        print(f"Secret '{key}' stored successfully.")
        return True

    except ValueError as e:
        print(f" {e}")
        return False
    except aiohttp.ClientConnectorError:
        print(" Cannot connect to Daita API. Check your internet connection.")
        return False
    except asyncio.TimeoutError:
        print(" Request timed out.")
        return False
    except Exception as e:
        print(f" Error: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return False


async def list_secrets(verbose: bool = False) -> bool:
    """List stored secret key names (values are never shown)."""
    try:
        headers = _get_headers()
        api_endpoint = get_api_endpoint()

        async with _make_session() as session:
            url = f"{api_endpoint}/api/v1/secrets"
            async with session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status == 401:
                    print(" Invalid API key or authentication failed.")
                    return False
                if resp.status != 200:
                    error = await resp.text()
                    print(f" Failed to list secrets (HTTP {resp.status}): {error}")
                    return False
                data = await resp.json()

        keys = data.get("keys", [])
        if not keys:
            print("No secrets stored.")
            print("")
            print("Add a secret with: daita secrets set KEY value")
            print("Import from .env:   daita secrets import .env")
        else:
            print(f"Stored secrets ({len(keys)}):")
            for k in keys:
                print(f"  {k}")
        return True

    except ValueError as e:
        print(f" {e}")
        return False
    except aiohttp.ClientConnectorError:
        print(" Cannot connect to Daita API. Check your internet connection.")
        return False
    except asyncio.TimeoutError:
        print(" Request timed out.")
        return False
    except Exception as e:
        print(f" Error: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return False


async def remove_secret(key: str, verbose: bool = False) -> bool:
    """Delete a stored secret."""
    try:
        _validate_key(key)
        headers = _get_headers()
        api_endpoint = get_api_endpoint()

        async with _make_session() as session:
            url = f"{api_endpoint}/api/v1/secrets/{key}"
            async with session.delete(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status == 401:
                    print(" Invalid API key or authentication failed.")
                    return False
                if resp.status == 404:
                    print(f" Secret '{key}' not found.")
                    return False
                if resp.status != 200:
                    error = await resp.text()
                    print(f" Failed to delete secret (HTTP {resp.status}): {error}")
                    return False

        print(f"Secret '{key}' deleted.")
        return True

    except ValueError as e:
        print(f" {e}")
        return False
    except aiohttp.ClientConnectorError:
        print(" Cannot connect to Daita API. Check your internet connection.")
        return False
    except asyncio.TimeoutError:
        print(" Request timed out.")
        return False
    except Exception as e:
        print(f" Error: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return False


async def import_env(env_file: str, verbose: bool = False) -> bool:
    """
    Import secrets from a .env file into secure cloud storage.

    Reads KEY=VALUE pairs and stores each valid entry as a secret.
    Lines starting with '#' and empty lines are skipped.
    """
    env_path = Path(env_file)
    if not env_path.exists():
        print(f" File not found: {env_file}")
        return False

    pairs = []
    skipped = []
    with open(env_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key or not value:
                continue
            if _KEY_RE.match(key):
                pairs.append((key, value))
            else:
                skipped.append(key)

    if not pairs:
        print(f"No valid KEY=VALUE entries found in {env_file}.")
        return False

    print(f"Importing {len(pairs)} secret(s) from {env_file}...")
    if skipped:
        print(f"Skipping {len(skipped)} invalid key(s): {', '.join(skipped)}")

    succeeded = 0
    failed = 0
    for key, value in pairs:
        ok = await set_secret(key, value, verbose=verbose)
        if ok:
            succeeded += 1
        else:
            failed += 1

    print("")
    print(f"Imported {succeeded}/{len(pairs)} secret(s).")
    if failed:
        print(f"{failed} secret(s) failed to import.")
    return failed == 0
