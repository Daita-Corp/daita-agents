"""
OTel span exporters for the Daita tracing pipeline.

BoundedInMemorySpanExporter  — thread-safe in-memory store (replaces the old deque),
                                powers local query APIs (get_recent_operations etc.)
DaitaSpanExporter            — sends real spans to the Daita dashboard API,
                                replacing the old DashboardReporter fire-and-forget approach.
"""

import json
import logging
import os
import threading
import urllib.request
import urllib.error
from typing import List, Optional, Sequence

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace import ReadableSpan

logger = logging.getLogger(__name__)


class BoundedInMemorySpanExporter(SpanExporter):
    """
    Thread-safe in-memory span store capped at ``maxlen`` spans.

    The BatchSpanProcessor calls ``export()`` from a background thread;
    query methods (get_recent_operations etc.) are called from the asyncio
    event loop thread.  A plain threading.Lock protects the shared list.
    """

    def __init__(self, maxlen: int = 500) -> None:
        self._maxlen = maxlen
        self._spans: List[ReadableSpan] = []
        self._lock = threading.Lock()
        self._stopped = False

    # ------------------------------------------------------------------
    # SpanExporter protocol
    # ------------------------------------------------------------------

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if self._stopped:
            return SpanExportResult.FAILURE
        with self._lock:
            self._spans.extend(spans)
            # Trim oldest spans when over cap
            if len(self._spans) > self._maxlen:
                self._spans = self._spans[-self._maxlen :]
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self._stopped = True

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True

    # ------------------------------------------------------------------
    # Query helpers (called by TraceManager query methods)
    # ------------------------------------------------------------------

    def get_finished_spans(self) -> List[ReadableSpan]:
        """Return a snapshot of all finished spans (most recent last)."""
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()


class DaitaSpanExporter(SpanExporter):
    """
    OTel SpanExporter that sends completed spans to the Daita dashboard API.

    Replaces the old ``DashboardReporter`` class.  Unlike the old approach
    (which used ``asyncio.create_task`` + ``aiohttp``), this exporter is
    called synchronously from the OTel ``BatchSpanProcessor`` background
    thread, so we use ``urllib.request`` (stdlib, sync) instead of aiohttp.

    Configuration (via environment variables):
        DAITA_API_KEY           — Bearer token for authentication
        DAITA_DASHBOARD_URL     — Base URL of the Daita backend API
        DAITA_DASHBOARD_API     — Alternate env var for the base URL
        DAITA_DASHBOARD_API_OVERRIDE — Another alternate env var
    """

    def __init__(self) -> None:
        self.api_key: Optional[str] = os.getenv("DAITA_API_KEY")
        self.dashboard_url: str = (
            os.getenv("DAITA_DASHBOARD_URL")
            or os.getenv("DAITA_DASHBOARD_API")
            or os.getenv("DAITA_DASHBOARD_API_OVERRIDE")
            or ""
        )
        self.enabled: bool = bool(self.api_key and self.dashboard_url)
        self._stopped: bool = False

        if self.enabled:
            logger.info(f"DaitaSpanExporter enabled (URL: {self.dashboard_url})")
        else:
            logger.debug(
                "DaitaSpanExporter disabled (DAITA_API_KEY or DAITA_DASHBOARD_URL not set)"
            )

    # ------------------------------------------------------------------
    # SpanExporter protocol
    # ------------------------------------------------------------------

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if not self.enabled or self._stopped:
            return SpanExportResult.SUCCESS

        try:
            payload = self._build_payload(spans)
            self._post(payload)
            return SpanExportResult.SUCCESS
        except Exception as exc:
            logger.warning(f"DaitaSpanExporter export failed: {exc}")
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        self._stopped = True

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_payload(self, spans: Sequence[ReadableSpan]) -> dict:
        """Convert OTel ReadableSpans to the Daita ingest payload format."""
        span_dicts = []
        for span in spans:
            ctx = span.get_span_context()
            if ctx is None:
                continue

            trace_id_hex = format(ctx.trace_id, "032x")
            span_id_hex = format(ctx.span_id, "016x")
            parent_span_id_hex = None
            if span.parent is not None:
                parent_span_id_hex = format(span.parent.span_id, "016x")

            attrs = dict(span.attributes) if span.attributes else {}

            events = []
            for event in span.events:
                events.append(
                    {
                        "name": event.name,
                        "timestamp_unix_nano": event.timestamp,
                        "attributes": (
                            dict(event.attributes) if event.attributes else {}
                        ),
                    }
                )

            duration_ms = None
            if span.start_time and span.end_time:
                duration_ms = (span.end_time - span.start_time) / 1_000_000

            span_dicts.append(
                {
                    "span_id": span_id_hex,
                    "trace_id": trace_id_hex,
                    "parent_span_id": parent_span_id_hex,
                    "operation_name": span.name,
                    "trace_type": attrs.get("daita.trace.type"),
                    "agent_id": attrs.get("daita.agent.id"),
                    "start_time_unix_nano": span.start_time,
                    "end_time_unix_nano": span.end_time,
                    "duration_ms": duration_ms,
                    "status_code": (
                        span.status.status_code.name if span.status else "UNSET"
                    ),
                    "status_message": span.status.description if span.status else None,
                    "attributes": attrs,
                    "events": events,
                }
            )

        return {"spans": span_dicts}

    def _post(self, payload: dict) -> None:
        """POST payload to the Daita ingest endpoint (sync, stdlib)."""
        url = f"{self.dashboard_url.rstrip('/')}/api/v1/traces/spans/ingest"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "daita-agents/otel-exporter",
            },
        )
        # 5-second timeout — we're in a background thread so blocking is acceptable,
        # but we don't want to stall the BatchSpanProcessor indefinitely.
        with urllib.request.urlopen(req, timeout=5) as response:
            status = response.status
            if status not in (200, 201, 204):
                body_text = response.read(200).decode("utf-8", errors="replace")
                logger.warning(
                    f"DaitaSpanExporter: unexpected HTTP {status} from {url}: {body_text}"
                )
