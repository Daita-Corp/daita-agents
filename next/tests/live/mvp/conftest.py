from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
import inspect
import os
from pathlib import Path
import re
from typing import Any

import pytest
import pytest_asyncio

from .fixture_oracles import (
    CommerceFixture,
    CommerceOracles,
    build_commerce_fixture,
    compute_oracles,
)
from .harness import (
    LiveRowRecorder,
    LiveMvpConfiguration,
    LiveMvpUnavailable,
    PropertyRecorder,
    RecordingOpenAIProvider,
    assert_paths_redacted,
    load_live_mvp_configuration,
    sidecar_path_for_run,
)
from .prompt_corpus import PROMPT_CORPUS_VERSION

_SCENARIO = re.compile(r"live_(?:mvp|precutover)_(\d+)")
_VARIANT_RANK = {
    "direct": 0,
    "conversational": 1,
    "answerable-ambiguous": 2,
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Run canonical MVP rows before prompt-reliability variants."""

    live_positions = tuple(
        index
        for index, item in enumerate(items)
        if item.get_closest_marker("live_mvp") is not None
    )
    ordered = iter(
        sorted(
            (items[index] for index in live_positions),
            key=_live_mvp_order_key,
        )
    )
    for index in live_positions:
        items[index] = next(ordered)


def _live_mvp_order_key(item: pytest.Item) -> tuple[int, int, str]:
    callspec = getattr(item, "callspec", None)
    variant_id = getattr(callspec, "id", "default")
    scenario_match = _SCENARIO.search(item.name)
    scenario = int(scenario_match.group(1)) if scenario_match is not None else 99
    return (_VARIANT_RANK.get(variant_id, 99), scenario, item.nodeid)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[object]) -> Any:
    outcome = yield
    report = outcome.get_result()
    setattr(item, f"rep_{report.when}", report)
    recorder = getattr(item, "_live_mvp_row_recorder", None)
    if (
        report.when == "call"
        and isinstance(recorder, LiveRowRecorder)
        and not recorder.is_finalized
    ):
        property_count = len(item.user_properties)
        recorder.finalize(outcome=_report_outcome(report))
        if report.user_properties is not item.user_properties:
            report.user_properties.extend(item.user_properties[property_count:])
        _register_report_scan(item.config, recorder)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    del exitstatus
    scans = getattr(session.config, "_live_mvp_report_scans", ())
    for paths, prohibited in scans:
        assert_paths_redacted(paths, prohibited)


@pytest.fixture
def commerce_fixture(tmp_path: Path) -> CommerceFixture:
    return build_commerce_fixture(tmp_path / "commerce-fixture")


@pytest.fixture
def commerce_oracles(commerce_fixture: CommerceFixture) -> CommerceOracles:
    return compute_oracles(commerce_fixture)


@pytest.fixture
def live_mvp_configuration() -> LiveMvpConfiguration:
    try:
        return load_live_mvp_configuration()
    except LiveMvpUnavailable as error:
        pytest.skip(str(error))
    raise AssertionError("pytest.skip returned unexpectedly")


@pytest_asyncio.fixture
async def live_mvp_provider(
    live_mvp_configuration: LiveMvpConfiguration,
) -> AsyncIterator[RecordingOpenAIProvider]:
    provider = RecordingOpenAIProvider(live_mvp_configuration)
    try:
        yield provider
    finally:
        client = getattr(provider, "_client", None)
        close = getattr(client, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result


@pytest.fixture
def live_row_recorder(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    commerce_fixture: CommerceFixture,
    live_mvp_configuration: LiveMvpConfiguration,
    record_property: PropertyRecorder,
) -> Iterator[LiveRowRecorder]:
    xml_path_value = getattr(request.config.option, "xmlpath", None)
    junit_path = Path(xml_path_value) if xml_path_value else None
    log_path_value = getattr(request.config.option, "log_file", None)
    report_paths = tuple(
        path
        for path in (
            junit_path,
            Path(log_path_value) if log_path_value else None,
        )
        if path is not None
    )
    match = _SCENARIO.search(request.node.name)
    scenario_id = (
        f"LIVE-MVP-{int(match.group(1)):02d}" if match is not None else "LIVE-MVP"
    )
    callspec = getattr(request.node, "callspec", None)
    variant_id = getattr(callspec, "id", "default")
    recorder = LiveRowRecorder(
        row_id=request.node.nodeid,
        scenario_id=scenario_id,
        variant_id=variant_id,
        configuration=live_mvp_configuration,
        fixture_version=commerce_fixture.fixture_version,
        fixture_digest=commerce_fixture.manifest_digest,
        prompt_version=PROMPT_CORPUS_VERSION,
        sidecar_path=sidecar_path_for_run(
            base_temp=tmp_path_factory.getbasetemp(),
            junit_path=junit_path,
        ),
        record_property=record_property,
        report_paths=report_paths,
    )
    credential = os.environ.get(
        live_mvp_configuration.credential_environment,
        "",
    )
    recorder.register_report_prohibited(credential)
    setattr(request.node, "_live_mvp_row_recorder", recorder)
    try:
        yield recorder
    finally:
        if not recorder.is_finalized:
            report = getattr(request.node, "rep_call", None)
            recorder.finalize(
                outcome=("error" if report is None else _report_outcome(report))
            )
            _register_report_scan(request.config, recorder)


def _report_outcome(report: Any) -> str:
    if report.passed:
        return "passed"
    if report.skipped:
        return "skipped"
    return "failed"


def _register_report_scan(
    config: pytest.Config,
    recorder: LiveRowRecorder,
) -> None:
    scans = getattr(config, "_live_mvp_report_scans", None)
    if scans is None:
        scans = []
        setattr(config, "_live_mvp_report_scans", scans)
    scans.append((recorder.report_paths, recorder.report_prohibited))
