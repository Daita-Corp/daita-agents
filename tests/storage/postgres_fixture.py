"""Real TLS Postgres, dedicated login roles, and disposable data only."""

from __future__ import annotations

import subprocess
import time
from uuid import uuid4

import psycopg
import pytest
from psycopg.conninfo import make_conninfo


def command(*args: str) -> str:
    return subprocess.run(
        args, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture(scope="session")
def postgres_database(tmp_path_factory):
    certificates = tmp_path_factory.mktemp("agent-state-tls")
    # Self-signed CA/server certificate is trusted explicitly by the client.
    configuration = certificates / "openssl.cnf"
    configuration.write_text(
        "[req]\ndistinguished_name=dn\nx509_extensions=ext\nprompt=no\n"
        "[dn]\nCN=localhost\n[ext]\nsubjectAltName=DNS:localhost,IP:127.0.0.1\n"
        "basicConstraints=critical,CA:TRUE\n"
    )
    command(
        "openssl",
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-days",
        "1",
        "-config",
        str(configuration),
        "-keyout",
        str(certificates / "server.key"),
        "-out",
        str(certificates / "server.crt"),
    )
    container = f"daita-agent-state-{uuid4().hex[:12]}"
    try:
        command(
            "docker",
            "run",
            "--detach",
            "--rm",
            "--name",
            container,
            "--publish",
            "127.0.0.1::5432",
            "--env",
            "POSTGRES_PASSWORD=fixture-only",
            "--mount",
            f"type=bind,source={certificates},target=/fixture,readonly",
            "--entrypoint",
            "bash",
            "postgres@sha256:724292da1f2e50bdccfc3302ce75bbba7f4a6076701b588cc795fcac65683550",
            "-c",
            "cp /fixture/server.* /tmp/ && chown postgres:postgres /tmp/server.* && "
            "chmod 600 /tmp/server.key && exec docker-entrypoint.sh postgres "
            "-c ssl=on -c ssl_cert_file=/tmp/server.crt -c ssl_key_file=/tmp/server.key",
        )
        port = command("docker", "port", container, "5432/tcp").rsplit(":", 1)[1]

        def dsn(user="postgres", dbname="postgres", **overrides):
            parameters = dict(
                host="127.0.0.1",
                port=port,
                dbname=dbname,
                user=user,
                password="fixture-only",
                sslmode="verify-full",
                sslrootcert=str(certificates / "server.crt"),
                connect_timeout=2,
            )
            parameters.update(overrides)
            return make_conninfo("", **parameters)

        for attempt in range(120):
            try:
                with psycopg.connect(dsn(), autocommit=True) as connection:
                    connection.execute("SELECT 1")
                break
            except psycopg.OperationalError:
                if attempt == 119:
                    raise RuntimeError(
                        "disposable TLS PostgreSQL did not start"
                    ) from None
                time.sleep(0.25)
        yield dsn, container
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container], capture_output=True, check=False
        )
