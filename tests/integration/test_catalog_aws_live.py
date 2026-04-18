"""
Live integration test for CatalogPlugin + AWS — strictly read-only.

Uses the active AWS credentials (e.g. ``aws-cli`` / environment) to run the
live ``AWSDiscoverer`` against whatever resources exist in the configured
region(s). Because this test runs against a real account, it:

  * NEVER creates, updates, or deletes AWS resources.
  * Issues only List*/Describe*/Get* API calls (enforced via a
    ReadOnlyBotoSession wrapper that refuses non-read verbs at the client
    factory level).
  * Tolerates empty accounts — absence of one resource type is fine; the
    discoverer itself must not throw.

Requirements:
  - boto3 (pip install 'daita-agents[aws]')
  - AWS credentials (env, config file, IAM role — whatever boto3 normally
    resolves)
  - OPENAI_API_KEY (for the live-LLM section)
  - AWS_REGION env var OR DAITA_TEST_AWS_REGIONS (CSV) to control the scan
    scope. Default: us-east-1.

Run:
    OPENAI_API_KEY=sk-... AWS_REGION=us-east-1 pytest \\
      tests/integration/test_catalog_aws_live.py -v -s -m "integration"
"""

from __future__ import annotations

import os
from typing import List

import pytest

boto3 = pytest.importorskip(
    "boto3", reason="boto3 required: pip install 'daita-agents[aws]'"
)

from daita.core.graph import LocalGraphBackend
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.catalog.aws import AWSDiscoverer

from ._harness import (
    assert_tool_called,
    build_live_agent,
    timed,
)

# ---------------------------------------------------------------------------
# Read-only enforcement
# ---------------------------------------------------------------------------


# Operation-name prefixes the test is allowed to issue. boto3 client methods
# map to `OperationName` strings; we intercept at the before-call hook.
_ALLOWED_PREFIXES = ("List", "Describe", "Get", "GetCallerIdentity", "Head")
# Explicit allowlist of operations that don't start with one of the above
# but are still safe reads in the AWS services AWSDiscoverer touches.
_ALLOWED_EXACT = {
    "GetCallerIdentity",
    "GetBucketLocation",
    "GetBucketTagging",
    "GetBucketVersioning",
    "GetBucketEncryption",
    "GetBucketPolicy",
    "GetBucketLifecycleConfiguration",
    "GetBucketNotificationConfiguration",
    "GetObjectTagging",
}


class _ReadOnlyViolation(RuntimeError):
    """Raised when the test accidentally tries to call a mutating AWS API."""


def _read_only_guard(event_name, params, **kwargs):  # noqa: ARG001
    """boto3 before-call hook. event_name looks like 'before-call.s3.PutObject'."""
    op = event_name.rsplit(".", 1)[-1]
    if op in _ALLOWED_EXACT:
        return
    if not op.startswith(_ALLOWED_PREFIXES):
        raise _ReadOnlyViolation(
            f"AWS live test attempted a non-read operation: {op}. "
            f"This test must remain strictly read-only."
        )


def _regions() -> List[str]:
    raw = os.environ.get("DAITA_TEST_AWS_REGIONS") or os.environ.get(
        "AWS_REGION", "us-east-1"
    )
    return [r.strip() for r in raw.split(",") if r.strip()]


def _aws_reachable() -> bool:
    """True iff the current boto3 session has usable credentials."""
    try:
        session = boto3.Session()
        if session.get_credentials() is None:
            return False
        sts = session.client("sts")
        sts.get_caller_identity()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _install_readonly_guard():
    """Install the read-only guard on the default boto3 session for the
    duration of the module. Autouse so every test inherits it."""
    session = boto3.DEFAULT_SESSION or boto3.Session()
    boto3.DEFAULT_SESSION = session
    session.events.register("before-call.*.*", _read_only_guard)
    yield
    session.events.unregister("before-call.*.*", _read_only_guard)


@pytest.fixture(scope="module")
def aws_guard():
    """Skip the module if no AWS credentials or STS is unreachable."""
    if not _aws_reachable():
        pytest.skip("AWS credentials not reachable — skipping live AWS test")


@pytest.fixture
def aws_discoverer(aws_guard) -> AWSDiscoverer:
    return AWSDiscoverer(regions=_regions())


@pytest.fixture
async def plugin_with_aws(tmp_path, monkeypatch, aws_discoverer):
    monkeypatch.chdir(tmp_path)
    backend = LocalGraphBackend(graph_type="catalog_aws_live")
    plugin = CatalogPlugin(backend=backend, auto_persist=False)
    plugin.add_discoverer(aws_discoverer)
    plugin.initialize("catalog-aws-live")
    return plugin, backend


# ---------------------------------------------------------------------------
# (1) Discoverer works + is authenticated
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAWSDiscovererLive:
    async def test_authenticate_resolves_account(self, aws_discoverer):
        await aws_discoverer.authenticate()
        assert aws_discoverer._account_id
        assert aws_discoverer._account_id != "unknown"

    async def test_test_access_true(self, aws_discoverer):
        assert await aws_discoverer.test_access() is True

    async def test_enumerate_yields_or_empty(self, aws_discoverer):
        """enumerate() either returns zero or more stores — but never raises."""
        stores = []
        async with timed("aws enumerate()"):
            async for store in aws_discoverer.enumerate():
                stores.append(store)

        # Note: S3 is a global service — buckets report their own home region,
        # which may differ from the scan region. We only require that *some*
        # region is recorded for each store, and that AWS was the source.
        for s in stores:
            assert s.id, "Every store must have a fingerprint id"
            assert s.store_type
            assert s.region, f"Store {s.id} has no region set"
            assert s.source.startswith("aws")


# ---------------------------------------------------------------------------
# (2) CatalogPlugin orchestration + graph
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCatalogPluginAWS:
    async def test_discover_all(self, plugin_with_aws):
        plugin, _ = plugin_with_aws
        async with timed("catalog.discover_all aws"):
            result = await plugin.discover_all()

        # Errors from the AWS discoverer itself are a hard fail — they'd
        # indicate either broken auth or a regression in aws.py. Individual
        # service-level empty results are fine.
        assert not result.has_errors, f"AWS discoverer raised: {result.errors}"
        assert len(plugin.get_stores()) == result.store_count


# ---------------------------------------------------------------------------
# (3) + (4) Live-LLM traversal: accuracy + speed
# ---------------------------------------------------------------------------


@pytest.mark.requires_llm
@pytest.mark.integration
class TestAgentAWSLive:
    async def test_agent_uses_discover_infrastructure(self, plugin_with_aws):
        """Live LLM must reach for ``discover_infrastructure`` when asked to
        enumerate the AWS account, and its answer must reflect the count the
        catalog actually observed."""
        plugin, _ = plugin_with_aws

        agent = build_live_agent(name="AWSCatalogAgent", tools=[plugin])
        async with timed("agent.run aws enumerate"):
            result = await agent.run(
                "Use the discover_infrastructure tool to scan the configured "
                "AWS account. Then report: the total number of data stores "
                "found, and the unique store types (e.g. s3, dynamodb, rds). "
                "Be concise.",
                detailed=True,
            )

        assert_tool_called(result, "discover_infrastructure")

        text = (result.get("result") or "").lower()
        expected_count = len(plugin.get_stores())
        word_forms = {
            0: "zero",
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "ten",
        }
        # Accept either the digit or the English word form; LLMs switch.
        forms = {str(expected_count), word_forms.get(expected_count, "")}
        assert any(
            f and f in text for f in forms
        ), f"Agent reported wrong count. Expected {expected_count}; answer: {text[:300]!r}"
