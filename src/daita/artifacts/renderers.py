"""Fixed Phase 1 Markdown/TXT rendering functions; deliberately no registry."""

from __future__ import annotations

from ..catalog.models import Sensitivity
from .models import (
    MAX_DOCUMENT_BYTES,
    MAX_DOCUMENT_CHARACTERS,
    ArtifactAuthorship,
    ArtifactDraft,
    ArtifactError,
    ArtifactProvenance,
    canonical_artifact_filename,
)

DOCUMENT_ALLOWED_EXTENSIONS = (
    ("text/markdown", (".md",)),
    ("text/plain", (".txt",)),
)


def render_model_document(
    *,
    content: str,
    format: str,
    filename: str | None,
    evidence_call_ids: tuple[str, ...],
) -> ArtifactDraft:
    """Render one bounded UTF-8 narrative with normalized LF newlines."""

    if not isinstance(content, str):
        raise ArtifactError(
            "artifact_invalid_format",
            "Document content must be Unicode text.",
            {"media_type": "text/plain", "allowed_extensions": (".txt", ".md")},
        )
    if not content:
        raise ArtifactError(
            "artifact_invalid_format",
            "Document content must be non-empty.",
            {"media_type": "text/plain", "allowed_extensions": (".txt", ".md")},
        )
    if len(content) > MAX_DOCUMENT_CHARACTERS:
        raise ArtifactError(
            "artifact_quota_exceeded",
            "The model-authored document exceeds its character limit.",
            {
                "scope": "call",
                "limit_kind": "characters",
                "limit": MAX_DOCUMENT_CHARACTERS,
                "attempted": len(content),
            },
        )
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    try:
        encoded = normalized.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ArtifactError(
            "artifact_invalid_format",
            "Document content is not valid UTF-8 text.",
            {"media_type": "text/plain", "allowed_extensions": (".txt", ".md")},
        ) from error
    if len(encoded) > MAX_DOCUMENT_BYTES:
        raise ArtifactError(
            "artifact_quota_exceeded",
            "The model-authored document exceeds its byte limit.",
            {
                "scope": "call",
                "limit_kind": "bytes",
                "limit": MAX_DOCUMENT_BYTES,
                "attempted": len(encoded),
            },
        )
    if format == "markdown":
        media_type = "text/markdown"
        requested_filename = filename or "analysis.md"
    elif format == "txt":
        media_type = "text/plain"
        requested_filename = filename or "analysis.txt"
    else:
        raise ArtifactError(
            "artifact_invalid_format",
            "The requested document format is not supported.",
            {
                "media_type": str(format),
                "allowed_extensions": (".md", ".txt"),
            },
        )
    safe_filename = canonical_artifact_filename(
        requested_filename,
        media_type,
        DOCUMENT_ALLOWED_EXTENSIONS,
    )
    return ArtifactDraft(
        content=encoded,
        suggested_filename=safe_filename,
        media_type=media_type,
        sensitivity=Sensitivity.INTERNAL,
        provenance=ArtifactProvenance(
            authorship=ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS,
            evidence_call_ids=evidence_call_ids,
        ),
    )


__all__ = ["DOCUMENT_ALLOWED_EXTENSIONS", "render_model_document"]
