"""Shared helpers for GCP discovery functions.

Centralises credential resolution and a common ``_GCP_INSTALL_HINT`` so every
GCP discover_* module uses the same auth flow and error message.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

_GCP_INSTALL_HINT = (
    "google-auth is required. Install with: pip install 'daita-agents[gcp]'"
)

_GCP_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def gcp_credentials(
    credentials_path: Optional[str] = None,
    impersonate_service_account: Optional[str] = None,
) -> Tuple[Any, Optional[str]]:
    """Resolve GCP credentials, optionally with service-account impersonation.

    Resolution order:
      1. ``credentials_path`` — explicit service-account JSON key file
      2. Application Default Credentials (env, gcloud, GCE metadata)

    If ``impersonate_service_account`` is set, the resolved credentials are
    used as the source for an impersonation chain scoped to cloud-platform.

    Returns:
        (credentials, default_project). ``default_project`` is populated only
        when ADC resolves to a project; otherwise ``None``.
    """
    try:
        from google.auth import (
            default,
            impersonated_credentials,
            load_credentials_from_file,
        )
    except ImportError:
        raise ImportError(_GCP_INSTALL_HINT)

    if credentials_path:
        creds, default_project = load_credentials_from_file(credentials_path)
    else:
        creds, default_project = default()

    if impersonate_service_account:
        creds = impersonated_credentials.Credentials(
            source_credentials=creds,
            target_principal=impersonate_service_account,
            target_scopes=[_GCP_SCOPE],
        )

    return creds, default_project
