"""
Live integration test for CatalogPlugin + GitHubDiscoverer.

Scans a configured GitHub repo (read-only via the REST API) for connection
strings embedded in config files. Verifies:

  1. The discoverer authenticates and completes a scan.
  2. CatalogPlugin orchestrates a discover_all() with no discoverer errors.
  3. Any DiscoveredStore returned has a valid fingerprint + a recognized
     store_type.
  4. An agent with the plugin can answer "what did you find?" using
     discover_infrastructure.

Requirements:
  - httpx (pip install 'daita-agents[github]')
  - GITHUB_TOKEN
  - DAITA_TEST_GITHUB_REPOS (CSV of "owner/repo") or DAITA_TEST_GITHUB_ORG
    — at least one must be set so the discoverer has a scan target.
  - OPENAI_API_KEY (for the live-LLM section)

Run:
    GITHUB_TOKEN=ghp_... DAITA_TEST_GITHUB_REPOS=myorg/myrepo \\
    OPENAI_API_KEY=sk-... pytest \\
      tests/integration/test_catalog_github_live.py -v -s -m "integration"
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip(
    "httpx", reason="httpx required: pip install 'daita-agents[github]'"
)

from daita.core.graph import LocalGraphBackend
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.catalog.github import GitHubScanner as GitHubDiscoverer

from ._harness import (
    assert_tool_called,
    build_live_agent,
    timed,
)


def _resolve_targets() -> tuple[list[str], str | None]:
    repos_raw = os.environ.get("DAITA_TEST_GITHUB_REPOS", "").strip()
    org = os.environ.get("DAITA_TEST_GITHUB_ORG", "").strip() or None
    repos = [r.strip() for r in repos_raw.split(",") if r.strip()]
    return repos, org


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def github_guard():
    if not os.environ.get("GITHUB_TOKEN"):
        pytest.skip("GITHUB_TOKEN not set — skipping GitHub live test")
    repos, org = _resolve_targets()
    if not repos and not org:
        pytest.skip(
            "Set DAITA_TEST_GITHUB_REPOS (CSV) or DAITA_TEST_GITHUB_ORG "
            "to define a scan target."
        )


@pytest.fixture
def github_discoverer(github_guard) -> GitHubDiscoverer:
    repos, org = _resolve_targets()
    return GitHubDiscoverer(repos=repos or None, org=org)


@pytest.fixture
async def plugin_with_github(tmp_path, monkeypatch, github_discoverer):
    monkeypatch.chdir(tmp_path)
    backend = LocalGraphBackend(graph_type="catalog_github_live")
    plugin = CatalogPlugin(backend=backend, auto_persist=False)
    plugin.add_discoverer(github_discoverer)
    plugin.initialize("catalog-github-live")
    return plugin, backend


# ---------------------------------------------------------------------------
# (1) Discoverer works
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGitHubDiscovererLive:
    async def test_authenticate(self, github_discoverer):
        await github_discoverer.authenticate()
        assert github_discoverer._client is not None

    async def test_enumerate_completes(self, github_discoverer):
        """enumerate() iterates over the configured repos without raising.
        Empty result is allowed — plenty of clean repos have no connection
        strings."""
        stores = []
        async with timed("github enumerate()"):
            async for s in github_discoverer.enumerate():
                stores.append(s)

        for s in stores:
            assert s.id
            assert s.store_type  # postgresql / mysql / mongodb / ...
            assert s.source.startswith("github")


# ---------------------------------------------------------------------------
# (2) CatalogPlugin orchestration
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCatalogPluginGitHub:
    async def test_discover_all_completes_cleanly(self, plugin_with_github):
        plugin, _ = plugin_with_github
        async with timed("catalog.discover_all github"):
            result = await plugin.discover_all()

        assert not result.has_errors, f"GitHub discoverer errored: {result.errors}"
        assert len(plugin.get_stores()) == result.store_count


# ---------------------------------------------------------------------------
# (3) + (4) Live-LLM traversal: accuracy + speed
# ---------------------------------------------------------------------------


@pytest.mark.requires_llm
@pytest.mark.integration
class TestAgentGitHubLive:
    async def test_agent_scans_and_reports(self, plugin_with_github):
        plugin, _ = plugin_with_github

        agent = build_live_agent(name="GitHubCatalogAgent", tools=[plugin])
        async with timed("agent.run github scan"):
            result = await agent.run(
                "Use the discover_infrastructure tool to scan the configured "
                "GitHub targets for data-store connection strings. Then "
                "summarize: how many stores were found and what unique store "
                "types appeared. Be concise.",
                detailed=True,
            )

        assert_tool_called(result, "discover_infrastructure")

        # Ground-truth check — the count in the answer must match the
        # catalog's observed count. Empty results ("zero"/"0"/"no") are a
        # legitimate answer for a clean repo.
        text = (result.get("result") or "").lower()
        expected_count = len(plugin.get_stores())
        word_forms = {
            0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
            5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
        }
        forms = {str(expected_count), word_forms.get(expected_count, "")}
        # When zero, accept "no" as a synonym for the count.
        if expected_count == 0:
            forms.add("no ")
        assert any(f and f in text for f in forms), (
            f"Agent reported wrong count. Expected {expected_count}; "
            f"answer: {text[:300]!r}"
        )
