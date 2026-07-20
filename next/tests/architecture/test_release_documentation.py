from __future__ import annotations

from pathlib import Path
import re

NEXT_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = NEXT_ROOT / "docs"

REQUIRED_DOCUMENTS = {
    "BREAKING_CHANGES.md",
    "CLI_AND_CLIENT.md",
    "FRESH_STATE_AND_MIGRATION.md",
    "OPERATIONS.md",
    "README.md",
    "SECURITY.md",
    "SUPPORT_MATRIX.md",
    "V1_EXPORT_GUIDE.md",
}


def test_required_candidate_documents_exist_and_link_to_real_local_targets() -> None:
    assert {path.name for path in DOCS_ROOT.glob("*.md")} == REQUIRED_DOCUMENTS

    documents = (NEXT_ROOT / "README.md", *sorted(DOCS_ROOT.glob("*.md")))
    missing: list[str] = []
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("#"):
                continue
            resolved = (document.parent / target.split("#", 1)[0]).resolve()
            if not resolved.exists():
                missing.append(f"{document.relative_to(NEXT_ROOT)} -> {target}")

    assert missing == []


def test_fresh_start_and_phase10_boundaries_are_explicit() -> None:
    migration = (DOCS_ROOT / "FRESH_STATE_AND_MIGRATION.md").read_text(encoding="utf-8")
    breaking = (DOCS_ROOT / "BREAKING_CHANGES.md").read_text(encoding="utf-8")
    client = (DOCS_ROOT / "CLI_AND_CLIENT.md").read_text(encoding="utf-8")

    assert "There is no general v1-to-v2 state migration command" in migration
    assert "Uninstalling the Python package does not remove" in migration
    assert "Phase 10" in breaking
    assert "must not be used as a fallback" in client


def test_support_matrix_names_every_deferred_integration_family() -> None:
    support = (DOCS_ROOT / "SUPPORT_MATRIX.md").read_text(encoding="utf-8")
    normalized_support = " ".join(support.split())
    for family in (
        "MySQL",
        "MongoDB",
        "Snowflake",
        "BigQuery",
        "Elasticsearch/OpenSearch",
        "object stores",
        "cloud inventory",
        "Google Drive",
        "Slack",
        "GitHub",
        "web search",
        "vector/graph",
        "embeddings",
        "rich documents",
        "data-quality/lineage",
        "OTLP transport",
        "multi-agent delegation",
        "v1 eval framework",
    ):
        assert family in normalized_support


def test_legacy_cli_and_client_retirement_is_explicit_and_actionable() -> None:
    client = (DOCS_ROOT / "CLI_AND_CLIENT.md").read_text(encoding="utf-8")
    support = (DOCS_ROOT / "SUPPORT_MATRIX.md").read_text(encoding="utf-8")
    breaking = (DOCS_ROOT / "BREAKING_CHANGES.md").read_text(encoding="utf-8")
    combined = " ".join("\n".join((client, support, breaking)).split())
    normalized_client = " ".join(client.split())

    for package in ("daita-cli", "daita-client"):
        assert package in client
        assert package in support
        assert package in breaking
    assert "sole supported Daita 2.0 product distribution" in normalized_client
    assert "unsupported and excluded from Daita 2.0" in normalized_client
    assert "python -m pip uninstall daita-cli daita-client" in client
    assert "hash -r" in client
    assert "command -v daita" in client
    assert "port-8123" in combined
    assert "stable Daita 2.0 cloud API" in normalized_client
