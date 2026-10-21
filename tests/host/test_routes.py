"""Every surface member has a route; handlers only decode, call and encode."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from urllib.parse import quote

import pytest

from daita.adapters.models import SourceRegistration
from daita.artifacts.models import ArtifactDestination, ArtifactDestinationKind
from daita.host.protocol import CWD_HEADER, REMOTE_AGENT_SURFACE
from daita.host.server import AREAS
from daita.hosting.sources import SourceEditResult
from daita.llm.models import ModelSensitivity
from tests.host.conftest import StartHost
from tests.support.host import error_code, pending_interaction, value

pytestmark = pytest.mark.integration


@pytest.mark.unit
def test_every_surface_member_is_reached_by_exactly_the_listed_routes() -> None:
    reached = set().union(*(area.MEMBERS for area in AREAS))
    routes = [
        (method, route.path)
        for area in AREAS
        for route in area.ROUTES
        for method in route.methods or ()
        if method != "HEAD"
    ]

    assert reached == REMOTE_AGENT_SURFACE
    assert len(routes) == len(set(routes))


async def test_reads_take_json_query_values_and_writes_take_json_bodies(
    start_host: StartHost,
) -> None:
    harness = await start_host()
    await harness.create_agent()

    written = await harness.http.put(
        "/v1/agents/sales/memory",
        json={
            "text": "Revenue is paid subtotal.",
            "sensitivity": {"$type": "daita.ModelSensitivity", "v": "internal"},
        },
    )
    assert value(written) is None
    assert value(await harness.http.get("/v1/agents/sales/memory")) == (
        "Revenue is paid subtotal."
    )
    skills = await harness.http.get(
        "/v1/agents/sales/semantics", params={"source_id": json.dumps("s-unknown")}
    )
    assert value(skills) == ()

    unknown = await harness.http.put("/v1/agents/sales/memory", json={"txt": "x"})
    assert (unknown.status_code, error_code(unknown)) == (400, "invalid_value")
    wrong_type = await harness.http.put(
        "/v1/agents/sales/memory", json={"text": "x", "sensitivity": "internal"}
    )
    assert error_code(wrong_type) == "invalid_type"
    assert ModelSensitivity.INTERNAL.value == "internal"


async def test_relative_client_paths_resolve_against_the_request_folder_only(
    start_host: StartHost,
) -> None:
    harness = await start_host()
    await harness.create_agent()
    folder = harness.work()
    (folder / "exports").mkdir()
    database = folder / "orders.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY)")
    cwd = {CWD_HEADER: quote(str(folder))}

    destination = value(
        await harness.http.put(
            "/v1/agents/sales/export-destination",
            json={"directory": {"$type": "path", "v": "exports"}},
            headers=cwd,
        )
    )
    source = value(
        await harness.http.post(
            "/v1/agents/sales/sources/sqlite",
            json={"path": "orders.sqlite"},
            headers=cwd,
        )
    )
    without_folder = await harness.http.post(
        "/v1/agents/sales/sources/sqlite", json={"path": "orders.sqlite"}
    )

    assert isinstance(destination, ArtifactDestination)
    assert destination.kind is ArtifactDestinationKind.LOCAL_DIRECTORY
    assert isinstance(source, SourceRegistration)
    assert Path(str(source.configuration["path"])) == database.resolve()
    assert value(await harness.http.get("/v1/agents/sales/sources")) == (source,)
    assert error_code(without_folder) == "invalid_value"
    assert CWD_HEADER in without_folder.json()["error"]["message"]


async def test_a_source_edit_waits_for_its_confirmation_interaction(
    start_host: StartHost,
) -> None:
    harness = await start_host()
    await harness.create_agent()
    folder = harness.work()
    for name in ("orders.sqlite", "orders-copy.sqlite"):
        with sqlite3.connect(folder / name) as connection:
            connection.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY)")
    source = value(
        await harness.http.post(
            "/v1/agents/sales/sources/sqlite",
            json={"path": str(folder / "orders.sqlite")},
        )
    )

    edit = asyncio.create_task(
        harness.http.post(
            f"/v1/agents/sales/sources/{source.id}/sqlite",
            json={
                "path": str(folder / "orders-copy.sqlite"),
                "confirmation_handler": True,
            },
        )
    )
    interaction = await pending_interaction(harness.http, "sales")
    assert interaction["kind"] == "source_edit_confirmation"
    assert interaction["request"]["preview"].current_source_id == source.id
    await harness.http.post(
        f"/v1/agents/sales/interactions/{interaction['id']}", json={"accept": True}
    )
    edited = value(await asyncio.wait_for(edit, 10))

    assert isinstance(edited, SourceEditResult)
    assert (
        Path(str(edited.source.configuration["path"]))
        == (folder / "orders-copy.sqlite").resolve()
    )


async def test_every_failure_is_one_error_envelope(start_host: StartHost) -> None:
    harness = await start_host()

    missing_route = await harness.http.get("/v1/nowhere")
    wrong_method = await harness.http.patch("/v1/agents")
    missing_agent = await harness.http.get("/v1/agents/nobody/memory")
    bad_name = await harness.http.get("/v1/agents/-bad/memory")
    old_client = await harness.http.get(
        "/v1/agents", headers={"X-Daita-Version": "0.0.1"}
    )

    for response, status, code in (
        (missing_route, 404, "key_not_found"),
        (wrong_method, 405, "invalid_value"),
        (missing_agent, 404, "agent_not_found"),
        (bad_name, 400, "agent_name_invalid"),
        (old_client, 409, "version_mismatch"),
    ):
        assert response.status_code == status, response.text
        assert set(response.json()) == {"error"}
        assert error_code(response) == code
    attributes = old_client.json()["error"]["details"]["attributes"]
    assert attributes["protocol"] == 1
