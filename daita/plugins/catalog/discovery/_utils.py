"""
Connection-string utilities shared across discovery functions.
"""

import ipaddress
import socket
import ssl
from typing import Any, Dict, Optional
from urllib.parse import urlparse, unquote


def redact_url(connection_string: str) -> str:
    """
    Return a loggable form of the connection string with the password replaced
    by '***'.
    """
    try:
        parsed = urlparse(connection_string)
        if parsed.password:
            redacted = parsed._replace(
                netloc=parsed.netloc.replace(f":{parsed.password}@", ":***@")
            )
            return redacted.geturl()
    except Exception:
        pass
    return connection_string


def parse_conn_url(connection_string: str) -> Dict[str, Any]:
    """
    Parse a database connection URL into explicit credential kwargs.

    Handles URL-encoded passwords (special chars like ! * @ in passwords
    break URL parsing if not encoded, and some drivers choke on the raw
    connection string). Returns a dict safe to splat into any DB driver's
    connect() call after picking the keys that driver needs.
    """
    parsed = urlparse(connection_string)
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port,
        "user": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
        "database": (parsed.path or "/").lstrip("/"),
    }


def validate_openapi_url(url: str) -> Optional[str]:
    """
    Validate that a URL is safe to fetch as an OpenAPI spec.

    Returns an error message string if the URL is unsafe, None if it is safe.

    Blocks:
    - Non-http/https schemes (e.g. file://, ftp://)
    - Hostnames that resolve to private, loopback, or link-local addresses
      (prevents SSRF to AWS IMDSv1 at 169.254.169.254, internal databases, etc.)
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"Only http/https URLs are supported, got: {parsed.scheme!r}"

    hostname = parsed.hostname
    if not hostname:
        return "URL has no hostname"

    try:
        addr_infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        return f"Could not resolve hostname {hostname!r}: {exc}"

    for addr_info in addr_infos:
        # Strip IPv6 zone ID (e.g. "fe80::1%eth0" -> "fe80::1")
        ip_str = addr_info[4][0].split("%")[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return (
                f"Requests to private/internal addresses are not permitted "
                f"(hostname {hostname!r} resolved to {ip})"
            )

    return None


def ssl_context(mode: str = "verify-full") -> ssl.SSLContext:
    """Return an SSL context for database connections.

    mode="verify-full"  (default) — full certificate and hostname verification.
                        Use for direct connections to managed cloud DBs (RDS,
                        Cloud SQL, Azure, Supabase direct port 5432).

    mode="require"      — encrypts the connection but skips certificate
                        verification. Use ONLY when connecting through a
                        pgbouncer pooler (e.g. Supabase pooler port 6543)
                        that presents a self-signed or unverifiable cert.
                        The data is still encrypted in transit; this only
                        disables identity verification of the server.
    """
    if mode == "require":
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    return ssl.create_default_context()
