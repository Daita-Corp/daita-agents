"""
Memory sync utilities for cloud deployment.

Helper functions for syncing local memory to S3.
Used internally by deployment process - not a CLI command.
"""

import json
import sqlite3
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


def sync_workspace_to_s3(
    workspace: str,
    org_id: str,
    project_name: str,
    s3_client,
    bucket: str,
    local_memory_dir: Path
):
    """
    Sync a single workspace to S3.

    This is a reusable function that can be called from deployment process.

    Args:
        workspace: Workspace name to sync
        org_id: Organization ID
        project_name: Project name
        s3_client: Boto3 S3 client instance
        bucket: S3 bucket name
        local_memory_dir: Path to local .daita/memory/workspaces directory

    Raises:
        ValueError: If workspace doesn't exist locally
        Exception: If S3 upload fails
    """
    workspace_dir = local_memory_dir / workspace
    if not workspace_dir.exists():
        raise ValueError(f"Workspace '{workspace}' not found locally")

    s3_prefix = f"orgs/{org_id}/projects/{project_name}/workspaces/{workspace}"

    # Upload MEMORY.md
    memory_file = workspace_dir / 'MEMORY.md'
    if memory_file.exists():
        s3_key = f"{s3_prefix}/MEMORY.md"
        s3_client.upload_file(str(memory_file), bucket, s3_key)

    # Upload vectors (convert SQLite to JSON)
    vectors_file = workspace_dir / 'vectors.db'
    if vectors_file.exists():
        vectors_json = convert_sqlite_to_json(vectors_file)
        s3_key = f"{s3_prefix}/vectors.json"
        s3_client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=json.dumps(vectors_json, indent=2).encode('utf-8'),
            ContentType='application/json'
        )

    # Upload logs
    logs_dir = workspace_dir / 'logs'
    if logs_dir.exists():
        for log_file in logs_dir.glob('*.md'):
            s3_key = f"{s3_prefix}/logs/{log_file.name}"
            s3_client.upload_file(str(log_file), bucket, s3_key)


def convert_sqlite_to_json(sqlite_path: Path) -> dict:
    """
    Convert local SQLite vectors.db to JSON format for S3.

    Args:
        sqlite_path: Path to vectors.db SQLite file

    Returns:
        Dictionary with vectors and metadata in JSON-serializable format
    """
    conn = sqlite3.connect(str(sqlite_path))
    cursor = conn.cursor()

    # Read vectors from SQLite
    cursor.execute("""
        SELECT c.chunk_id, c.content, e.embedding, c.line_start, c.line_end, c.created_at
        FROM chunks c
        JOIN embeddings e ON c.chunk_id = e.chunk_id
    """)

    vectors = {}
    for row in cursor.fetchall():
        chunk_id, content, embedding_json, line_start, line_end, created_at = row
        vectors[chunk_id] = {
            'content': content,
            'embedding': json.loads(embedding_json),
            'line_start': line_start,
            'line_end': line_end,
            'created_at': created_at or datetime.now().isoformat()
        }

    conn.close()

    return {
        'vectors': vectors,
        'metadata': {
            'converted_at': datetime.now().isoformat(),
            'source': 'local_sqlite',
            'vector_count': len(vectors)
        }
    }


def find_project_root(start_path: Optional[Path] = None) -> Optional[Path]:
    """
    Find project root by looking for daita-project.yaml.

    Args:
        start_path: Starting directory (defaults to cwd)

    Returns:
        Path to project root, or None if not found
    """
    current = start_path or Path.cwd()

    # Walk up directory tree looking for daita-project.yaml
    for parent in [current] + list(current.parents):
        if (parent / "daita-project.yaml").exists():
            return parent

    return None
