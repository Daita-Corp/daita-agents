"""
Real-world Focus DSL examples.

Each scenario simulates data that would realistically arrive from an API,
database, or file — then shows exactly what the LLM sees after focus is applied.

Scenarios:
  1. E-commerce orders      — filter + project + aggregate sales data
  2. API error logs         — surface errors from noisy log streams
  3. SaaS user events       — funnel analysis from raw event stream
  4. Financial transactions — fraud signal extraction
  5. Product catalogue      — nested API response trimming
  6. Token reduction         — measure how much focus shrinks payloads
"""

import json
import pytest

from daita.core.focus import apply_focus

# ─────────────────────────────────────────────────────────────────────────────
# Shared realistic datasets
# ─────────────────────────────────────────────────────────────────────────────

ORDERS = [
    {
        "order_id": "ORD-001",
        "customer_id": "CUST-042",
        "customer_name": "Lena Fischer",
        "email": "lena@example.com",
        "product": "Pro Plan",
        "category": "subscription",
        "amount": 299.00,
        "status": "completed",
        "region": "EU",
        "created_at": "2024-03-01T09:12:00Z",
        "updated_at": "2024-03-01T09:14:00Z",
        "billing_address": {
            "street": "Musterstraße 1",
            "city": "Berlin",
            "zip": "10115",
        },
        "payment_method": "credit_card",
        "refunded": False,
        "notes": None,
    },
    {
        "order_id": "ORD-002",
        "customer_id": "CUST-007",
        "customer_name": "James Park",
        "email": "james@example.com",
        "product": "Starter Plan",
        "category": "subscription",
        "amount": 49.00,
        "status": "completed",
        "region": "US",
        "created_at": "2024-03-02T14:05:00Z",
        "updated_at": "2024-03-02T14:06:00Z",
        "billing_address": {"street": "123 Main St", "city": "Austin", "zip": "73301"},
        "payment_method": "paypal",
        "refunded": False,
        "notes": None,
    },
    {
        "order_id": "ORD-003",
        "customer_id": "CUST-099",
        "customer_name": "Sara Nkosi",
        "email": "sara@example.com",
        "product": "Enterprise",
        "category": "subscription",
        "amount": 999.00,
        "status": "refunded",
        "region": "EU",
        "created_at": "2024-03-03T11:00:00Z",
        "updated_at": "2024-03-04T08:30:00Z",
        "billing_address": {
            "street": "Rue de la Paix 5",
            "city": "Paris",
            "zip": "75001",
        },
        "payment_method": "credit_card",
        "refunded": True,
        "notes": "Customer requested refund",
    },
    {
        "order_id": "ORD-004",
        "customer_id": "CUST-011",
        "customer_name": "Hiroshi Tanaka",
        "email": "hiroshi@example.com",
        "product": "Pro Plan",
        "category": "subscription",
        "amount": 299.00,
        "status": "completed",
        "region": "APAC",
        "created_at": "2024-03-04T02:45:00Z",
        "updated_at": "2024-03-04T02:46:00Z",
        "billing_address": {
            "street": "Shibuya 1-1",
            "city": "Tokyo",
            "zip": "150-0001",
        },
        "payment_method": "credit_card",
        "refunded": False,
        "notes": None,
    },
    {
        "order_id": "ORD-005",
        "customer_id": "CUST-055",
        "customer_name": "Maria Garcia",
        "email": "maria@example.com",
        "product": "Starter Plan",
        "category": "subscription",
        "amount": 49.00,
        "status": "pending",
        "region": "US",
        "created_at": "2024-03-05T17:22:00Z",
        "updated_at": "2024-03-05T17:22:00Z",
        "billing_address": {"street": "456 Oak Ave", "city": "Miami", "zip": "33101"},
        "payment_method": "bank_transfer",
        "refunded": False,
        "notes": "Awaiting payment confirmation",
    },
]

API_LOGS = [
    {
        "log_id": "L001",
        "timestamp": "2024-03-05T10:00:01Z",
        "service": "auth",
        "level": "INFO",
        "status_code": 200,
        "latency_ms": 45,
        "user_id": "u123",
        "endpoint": "/login",
        "method": "POST",
        "ip": "192.168.1.10",
        "message": "Login successful",
        "trace_id": "t-aaa",
        "request_size_bytes": 128,
    },
    {
        "log_id": "L002",
        "timestamp": "2024-03-05T10:00:05Z",
        "service": "api",
        "level": "ERROR",
        "status_code": 500,
        "latency_ms": 1200,
        "user_id": "u456",
        "endpoint": "/data/export",
        "method": "GET",
        "ip": "10.0.0.5",
        "message": "Timeout connecting to database",
        "trace_id": "t-bbb",
        "request_size_bytes": 64,
    },
    {
        "log_id": "L003",
        "timestamp": "2024-03-05T10:00:09Z",
        "service": "api",
        "level": "INFO",
        "status_code": 200,
        "latency_ms": 88,
        "user_id": "u789",
        "endpoint": "/reports",
        "method": "GET",
        "ip": "10.0.0.8",
        "message": "Report generated",
        "trace_id": "t-ccc",
        "request_size_bytes": 32,
    },
    {
        "log_id": "L004",
        "timestamp": "2024-03-05T10:00:12Z",
        "service": "billing",
        "level": "ERROR",
        "status_code": 402,
        "latency_ms": 230,
        "user_id": "u456",
        "endpoint": "/charge",
        "method": "POST",
        "ip": "10.0.0.5",
        "message": "Payment declined: insufficient funds",
        "trace_id": "t-ddd",
        "request_size_bytes": 256,
    },
    {
        "log_id": "L005",
        "timestamp": "2024-03-05T10:00:15Z",
        "service": "auth",
        "level": "WARN",
        "status_code": 401,
        "latency_ms": 12,
        "user_id": None,
        "endpoint": "/login",
        "method": "POST",
        "ip": "203.0.113.42",
        "message": "Failed login attempt",
        "trace_id": "t-eee",
        "request_size_bytes": 96,
    },
    {
        "log_id": "L006",
        "timestamp": "2024-03-05T10:00:20Z",
        "service": "api",
        "level": "INFO",
        "status_code": 200,
        "latency_ms": 55,
        "user_id": "u123",
        "endpoint": "/dashboard",
        "method": "GET",
        "ip": "192.168.1.10",
        "message": "Dashboard loaded",
        "trace_id": "t-fff",
        "request_size_bytes": 32,
    },
    {
        "log_id": "L007",
        "timestamp": "2024-03-05T10:00:25Z",
        "service": "api",
        "level": "ERROR",
        "status_code": 503,
        "latency_ms": 5001,
        "user_id": "u321",
        "endpoint": "/search",
        "method": "GET",
        "ip": "10.0.1.2",
        "message": "Search service unavailable",
        "trace_id": "t-ggg",
        "request_size_bytes": 48,
    },
]

USER_EVENTS = [
    {
        "event_id": "E001",
        "user_id": "u100",
        "plan": "free",
        "event": "signup",
        "page": None,
        "feature": None,
        "session_id": "s-1",
        "ts": "2024-03-05T08:00:00Z",
        "device": "mobile",
        "country": "US",
        "referrer": "google",
    },
    {
        "event_id": "E002",
        "user_id": "u100",
        "plan": "free",
        "event": "page_view",
        "page": "/dashboard",
        "feature": None,
        "session_id": "s-1",
        "ts": "2024-03-05T08:01:00Z",
        "device": "mobile",
        "country": "US",
        "referrer": None,
    },
    {
        "event_id": "E003",
        "user_id": "u100",
        "plan": "free",
        "event": "feature_click",
        "page": "/dashboard",
        "feature": "export",
        "session_id": "s-1",
        "ts": "2024-03-05T08:02:00Z",
        "device": "mobile",
        "country": "US",
        "referrer": None,
    },
    {
        "event_id": "E004",
        "user_id": "u100",
        "plan": "free",
        "event": "upgrade_prompt",
        "page": "/upgrade",
        "feature": "export",
        "session_id": "s-1",
        "ts": "2024-03-05T08:03:00Z",
        "device": "mobile",
        "country": "US",
        "referrer": None,
    },
    {
        "event_id": "E005",
        "user_id": "u100",
        "plan": "free",
        "event": "upgrade_start",
        "page": "/checkout",
        "feature": None,
        "session_id": "s-1",
        "ts": "2024-03-05T08:04:00Z",
        "device": "mobile",
        "country": "US",
        "referrer": None,
    },
    {
        "event_id": "E006",
        "user_id": "u100",
        "plan": "pro",
        "event": "upgrade_complete",
        "page": "/checkout",
        "feature": None,
        "session_id": "s-1",
        "ts": "2024-03-05T08:05:00Z",
        "device": "mobile",
        "country": "US",
        "referrer": None,
    },
    {
        "event_id": "E007",
        "user_id": "u200",
        "plan": "pro",
        "event": "signup",
        "page": None,
        "feature": None,
        "session_id": "s-2",
        "ts": "2024-03-05T09:00:00Z",
        "device": "desktop",
        "country": "DE",
        "referrer": "direct",
    },
    {
        "event_id": "E008",
        "user_id": "u200",
        "plan": "pro",
        "event": "page_view",
        "page": "/dashboard",
        "feature": None,
        "session_id": "s-2",
        "ts": "2024-03-05T09:01:00Z",
        "device": "desktop",
        "country": "DE",
        "referrer": None,
    },
    {
        "event_id": "E009",
        "user_id": "u200",
        "plan": "pro",
        "event": "feature_click",
        "page": "/reports",
        "feature": "analytics",
        "session_id": "s-2",
        "ts": "2024-03-05T09:05:00Z",
        "device": "desktop",
        "country": "DE",
        "referrer": None,
    },
    {
        "event_id": "E010",
        "user_id": "u300",
        "plan": "free",
        "event": "signup",
        "page": None,
        "feature": None,
        "session_id": "s-3",
        "ts": "2024-03-05T10:00:00Z",
        "device": "desktop",
        "country": "JP",
        "referrer": "twitter",
    },
    {
        "event_id": "E011",
        "user_id": "u300",
        "plan": "free",
        "event": "page_view",
        "page": "/dashboard",
        "feature": None,
        "session_id": "s-3",
        "ts": "2024-03-05T10:01:00Z",
        "device": "desktop",
        "country": "JP",
        "referrer": None,
    },
    {
        "event_id": "E012",
        "user_id": "u300",
        "plan": "free",
        "event": "churn",
        "page": None,
        "feature": None,
        "session_id": "s-3",
        "ts": "2024-03-05T10:30:00Z",
        "device": "desktop",
        "country": "JP",
        "referrer": None,
    },
]

TRANSACTIONS = [
    {
        "txn_id": "T001",
        "account_id": "ACC-100",
        "amount": 42.50,
        "currency": "USD",
        "type": "purchase",
        "merchant": "Amazon",
        "merchant_category": "retail",
        "country": "US",
        "ts": "2024-03-05T08:10:00Z",
        "flagged": False,
        "device_fingerprint": "fp-aaa",
        "ip": "1.2.3.4",
        "velocity_1h": 1,
        "card_present": True,
    },
    {
        "txn_id": "T002",
        "account_id": "ACC-100",
        "amount": 9850.00,
        "currency": "USD",
        "type": "transfer",
        "merchant": None,
        "merchant_category": None,
        "country": "RU",
        "ts": "2024-03-05T08:11:00Z",
        "flagged": True,
        "device_fingerprint": "fp-zzz",
        "ip": "195.3.3.3",
        "velocity_1h": 12,
        "card_present": False,
    },
    {
        "txn_id": "T003",
        "account_id": "ACC-200",
        "amount": 15.00,
        "currency": "USD",
        "type": "purchase",
        "merchant": "Starbucks",
        "merchant_category": "food",
        "country": "US",
        "ts": "2024-03-05T09:00:00Z",
        "flagged": False,
        "device_fingerprint": "fp-bbb",
        "ip": "5.6.7.8",
        "velocity_1h": 1,
        "card_present": True,
    },
    {
        "txn_id": "T004",
        "account_id": "ACC-200",
        "amount": 5200.00,
        "currency": "USD",
        "type": "purchase",
        "merchant": "Electronics Plus",
        "merchant_category": "retail",
        "country": "US",
        "ts": "2024-03-05T09:05:00Z",
        "flagged": True,
        "device_fingerprint": "fp-bbb",
        "ip": "5.6.7.8",
        "velocity_1h": 8,
        "card_present": False,
    },
    {
        "txn_id": "T005",
        "account_id": "ACC-300",
        "amount": 2.99,
        "currency": "USD",
        "type": "subscription",
        "merchant": "Netflix",
        "merchant_category": "streaming",
        "country": "US",
        "ts": "2024-03-05T10:00:00Z",
        "flagged": False,
        "device_fingerprint": "fp-ccc",
        "ip": "9.10.11.12",
        "velocity_1h": 1,
        "card_present": False,
    },
    {
        "txn_id": "T006",
        "account_id": "ACC-100",
        "amount": 1.00,
        "currency": "USD",
        "type": "purchase",
        "merchant": "Test Merchant",
        "merchant_category": "other",
        "country": "NG",
        "ts": "2024-03-05T08:15:00Z",
        "flagged": True,
        "device_fingerprint": "fp-zzz",
        "ip": "195.3.3.3",
        "velocity_1h": 14,
        "card_present": False,
    },
]

PRODUCT_CATALOGUE = [
    {
        "id": "SKU-001",
        "name": "Wireless Headphones Pro",
        "description": "Premium noise-cancelling headphones with 40-hour battery life, foldable design, and carrying case included. Compatible with all Bluetooth 5.0 devices.",
        "price": 149.99,
        "cost": 62.00,
        "margin": 0.587,
        "stock": 234,
        "category": "electronics",
        "subcategory": "audio",
        "tags": ["bluetooth", "noise-cancelling", "premium"],
        "images": ["img1.jpg", "img2.jpg", "img3.jpg"],
        "specs": {"weight_g": 250, "battery_hours": 40, "bluetooth_version": "5.0"},
        "ratings": {"average": 4.6, "count": 1842},
        "supplier": {"id": "SUP-01", "name": "TechSource Ltd", "lead_time_days": 14},
    },
    {
        "id": "SKU-002",
        "name": "Mechanical Keyboard TKL",
        "description": "Tenkeyless mechanical keyboard with Cherry MX switches, RGB backlight, aluminium frame, and detachable USB-C cable. Ideal for coding and gaming.",
        "price": 89.99,
        "cost": 38.50,
        "margin": 0.572,
        "stock": 89,
        "category": "electronics",
        "subcategory": "peripherals",
        "tags": ["mechanical", "rgb", "tenkeyless"],
        "images": ["kb1.jpg", "kb2.jpg"],
        "specs": {
            "switch_type": "Cherry MX Red",
            "layout": "TKL",
            "connection": "USB-C",
        },
        "ratings": {"average": 4.8, "count": 956},
        "supplier": {"id": "SUP-02", "name": "KeyCraft Inc", "lead_time_days": 21},
    },
    {
        "id": "SKU-003",
        "name": "USB-C Hub 7-in-1",
        "description": "Compact 7-port USB-C hub with 4K HDMI, 100W power delivery, SD card slot, and 3 USB-A ports. Aluminium shell with braided cable.",
        "price": 39.99,
        "cost": 14.20,
        "margin": 0.645,
        "stock": 512,
        "category": "electronics",
        "subcategory": "accessories",
        "tags": ["usb-c", "hub", "4k"],
        "images": ["hub1.jpg"],
        "specs": {"ports": 7, "power_delivery_w": 100, "hdmi_resolution": "4K"},
        "ratings": {"average": 4.3, "count": 3201},
        "supplier": {"id": "SUP-01", "name": "TechSource Ltd", "lead_time_days": 7},
    },
    {
        "id": "SKU-004",
        "name": "Desk Lamp LED",
        "description": "Adjustable LED desk lamp with 5 colour temperatures, touch dimmer, USB-A charging port, and 50,000-hour lifespan.",
        "price": 34.99,
        "cost": 11.80,
        "margin": 0.663,
        "stock": 8,
        "category": "office",
        "subcategory": "lighting",
        "tags": ["led", "adjustable", "usb-charging"],
        "images": ["lamp1.jpg", "lamp2.jpg"],
        "specs": {"colour_temps": 5, "wattage": 12, "lifespan_hours": 50000},
        "ratings": {"average": 4.5, "count": 621},
        "supplier": {"id": "SUP-03", "name": "BrightGoods", "lead_time_days": 10},
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _token_estimate(data) -> int:
    """Rough token count: ~4 chars per token."""
    return len(json.dumps(data, default=str)) // 4


def _reduction_pct(before, after) -> float:
    b, a = _token_estimate(before), _token_estimate(after)
    return round((1 - a / b) * 100, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1 — E-commerce orders
# ─────────────────────────────────────────────────────────────────────────────


class TestEcommerceOrders:
    """
    An agent is asked: "Which EU orders completed successfully this month,
    and what is total revenue by product?"

    Raw order records include billing addresses, email, notes, timestamps,
    and payment details that the LLM doesn't need to answer the question.
    """

    def test_filter_completed_eu_orders(self):
        result = apply_focus(
            ORDERS,
            "status == 'completed' and region == 'EU' | SELECT order_id, product, amount, region",
        )
        assert len(result) == 1
        assert result[0]["order_id"] == "ORD-001"
        assert result[0]["region"] == "EU"
        # Billing address and email are gone
        assert "email" not in result[0]
        assert "billing_address" not in result[0]

    def test_revenue_by_product(self):
        completed = apply_focus(ORDERS, "status == 'completed'")
        result = apply_focus(
            completed,
            "GROUP BY product | SELECT product, SUM(amount) AS revenue, COUNT(*) AS orders",
        )
        revenue = {r["product"]: r["revenue"] for r in result}
        assert revenue["Pro Plan"] == 598.00  # ORD-001 + ORD-004
        assert revenue["Starter Plan"] == 49.00  # ORD-002 only

    def test_top_orders_by_amount(self):
        result = apply_focus(
            ORDERS,
            "status == 'completed' | SELECT order_id, customer_name, product, amount | ORDER BY amount DESC | LIMIT 3",
        )
        assert len(result) == 3
        assert result[0]["amount"] >= result[1]["amount"]
        assert list(result[0].keys()) == [
            "order_id",
            "customer_name",
            "product",
            "amount",
        ]

    def test_pending_orders_needing_attention(self):
        result = apply_focus(
            ORDERS,
            "status == 'pending' | SELECT order_id, customer_name, product, amount, notes",
        )
        assert len(result) == 1
        assert result[0]["order_id"] == "ORD-005"
        assert "notes" in result[0]

    def test_refund_summary(self):
        result = apply_focus(
            ORDERS, "refunded == True | SELECT order_id, customer_name, amount, notes"
        )
        assert len(result) == 1
        assert result[0]["amount"] == 999.00

    def test_token_reduction_is_significant(self):
        focused = apply_focus(
            ORDERS, "status == 'completed' | SELECT order_id, product, amount, region"
        )
        reduction = _reduction_pct(ORDERS, focused)
        # Dropping billing_address, email, timestamps, payment_method, notes etc.
        assert reduction > 60, f"Expected >60% reduction, got {reduction}%"


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2 — API error logs
# ─────────────────────────────────────────────────────────────────────────────


class TestAPIErrorLogs:
    """
    An agent is monitoring service health and asked: "Show me all errors
    and slow requests from the last window."

    Raw logs include IPs, trace IDs, request sizes, and INFO-level noise
    that is irrelevant to diagnosing the problem.
    """

    def test_errors_only(self):
        result = apply_focus(
            API_LOGS,
            "level == 'ERROR' | SELECT timestamp, service, status_code, latency_ms, message",
        )
        assert len(result) == 3
        assert all(r["status_code"] >= 400 for r in result)
        # Trace ID and IP stripped — not needed for diagnosis
        assert "trace_id" not in result[0]
        assert "ip" not in result[0]

    def test_slow_requests_over_500ms(self):
        result = apply_focus(
            API_LOGS,
            "latency_ms > 500 | SELECT timestamp, service, endpoint, latency_ms, message | ORDER BY latency_ms DESC",
        )
        assert len(result) == 2
        assert result[0]["latency_ms"] >= result[1]["latency_ms"]
        # Worst offender is the search service at 5001ms
        assert result[0]["latency_ms"] == 5001

    def test_errors_and_warns(self):
        result = apply_focus(
            API_LOGS,
            "level in ['ERROR', 'WARN'] | SELECT timestamp, level, service, message | ORDER BY timestamp ASC",
        )
        assert len(result) == 4
        assert all(r["level"] in ("ERROR", "WARN") for r in result)

    def test_errors_by_service(self):
        errors = apply_focus(API_LOGS, "level == 'ERROR'")
        result = apply_focus(
            errors, "GROUP BY service | SELECT service, COUNT(*) AS error_count"
        )
        counts = {r["service"]: r["error_count"] for r in result}
        assert counts["api"] == 2
        assert counts["billing"] == 1

    def test_suspicious_user_activity(self):
        # User u456 appears in both an ERROR 500 and a 402 payment decline
        result = apply_focus(
            API_LOGS,
            "user_id == 'u456' | SELECT timestamp, service, status_code, message",
        )
        assert len(result) == 2
        services = {r["service"] for r in result}
        assert services == {"api", "billing"}

    def test_token_reduction_filters_info_noise(self):
        focused = apply_focus(
            API_LOGS,
            "level == 'ERROR' | SELECT timestamp, service, status_code, latency_ms, message",
        )
        reduction = _reduction_pct(API_LOGS, focused)
        # 3 of 7 rows kept, and heavy fields (trace_id, ip, request_size_bytes, method, user_id) dropped
        assert reduction > 70, f"Expected >70% reduction, got {reduction}%"


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3 — SaaS user events / funnel analysis
# ─────────────────────────────────────────────────────────────────────────────


class TestUserEventFunnel:
    """
    A growth agent is asked: "Which users hit the upgrade prompt but didn't
    convert, and what did the user who did convert do beforehand?"

    Raw events include session IDs, device, country, referrer, and every
    page view — irrelevant to funnel analysis.
    """

    def test_upgrade_funnel_events(self):
        funnel_events = ["upgrade_prompt", "upgrade_start", "upgrade_complete"]
        result = apply_focus(
            USER_EVENTS,
            f"event in {funnel_events} | SELECT user_id, event, ts | ORDER BY ts ASC",
        )
        assert len(result) == 3
        events = [r["event"] for r in result]
        assert "upgrade_prompt" in events
        assert "upgrade_complete" in events
        # session_id, device, country stripped
        assert "session_id" not in result[0]

    def test_converted_user_journey(self):
        # u100 completed an upgrade — show their full journey in order
        result = apply_focus(
            USER_EVENTS,
            "user_id == 'u100' | SELECT event, feature, ts | ORDER BY ts ASC",
        )
        events = [r["event"] for r in result]
        assert events == [
            "signup",
            "page_view",
            "feature_click",
            "upgrade_prompt",
            "upgrade_start",
            "upgrade_complete",
        ]
        # The feature that triggered the upgrade funnel
        feature_clicks = [r for r in result if r["event"] == "feature_click"]
        assert feature_clicks[0]["feature"] == "export"

    def test_churned_free_users(self):
        result = apply_focus(
            USER_EVENTS,
            "event == 'churn' and plan == 'free' | SELECT user_id, country, ts",
        )
        assert len(result) == 1
        assert result[0]["user_id"] == "u300"

    def test_event_counts_by_type(self):
        result = apply_focus(
            USER_EVENTS,
            "GROUP BY event | SELECT event, COUNT(*) AS cnt | ORDER BY cnt DESC",
        )
        counts = {r["event"]: r["cnt"] for r in result}
        assert counts["page_view"] == 3
        assert counts["signup"] == 3

    def test_pro_plan_feature_usage(self):
        result = apply_focus(
            USER_EVENTS,
            "plan == 'pro' and event == 'feature_click' | SELECT user_id, feature, page",
        )
        assert len(result) == 1
        assert result[0]["feature"] == "analytics"


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 4 — Financial transactions / fraud signals
# ─────────────────────────────────────────────────────────────────────────────


class TestTransactionFraudSignals:
    """
    A fraud detection agent is asked: "Summarise the flagged transactions
    and identify any accounts with multiple flags."

    Raw records include device fingerprints, full IPs, currency, and card_present
    flags that are metadata-heavy but not needed for the summary.
    """

    def test_flagged_transactions(self):
        result = apply_focus(
            TRANSACTIONS,
            "flagged == True | SELECT txn_id, account_id, amount, type, country, velocity_1h | ORDER BY amount DESC",
        )
        assert len(result) == 3
        assert all(r["amount"] > 0 for r in result)
        # device_fingerprint and ip stripped
        assert "device_fingerprint" not in result[0]
        assert "ip" not in result[0]

    def test_high_value_flagged(self):
        result = apply_focus(
            TRANSACTIONS,
            "flagged == True and amount > 1000 | SELECT txn_id, account_id, amount, country, type",
        )
        assert len(result) == 2
        amounts = [r["amount"] for r in result]
        assert 9850.00 in amounts
        assert 5200.00 in amounts

    def test_flagged_by_account(self):
        flagged = apply_focus(TRANSACTIONS, "flagged == True")
        result = apply_focus(
            flagged,
            "GROUP BY account_id | SELECT account_id, COUNT(*) AS flag_count, SUM(amount) AS flagged_total | ORDER BY flag_count DESC",
        )
        # ACC-100 has 2 flags (T002 and T006)
        top = result[0]
        assert top["account_id"] == "ACC-100"
        assert top["flag_count"] == 2

    def test_cross_border_high_velocity(self):
        # Transactions that are: flagged, high velocity, and not US
        result = apply_focus(
            TRANSACTIONS,
            "flagged == True and velocity_1h > 10 and country != 'US' | SELECT txn_id, account_id, country, velocity_1h, amount",
        )
        assert len(result) == 2
        assert all(r["country"] != "US" for r in result)

    def test_normal_transactions_excluded(self):
        normal = apply_focus(TRANSACTIONS, "flagged == False")
        assert len(normal) == 3
        assert all(not t["flagged"] for t in normal)

    def test_token_reduction_fraud_focused(self):
        focused = apply_focus(
            TRANSACTIONS,
            "flagged == True | SELECT txn_id, account_id, amount, country, velocity_1h",
        )
        reduction = _reduction_pct(TRANSACTIONS, focused)
        assert reduction > 55, f"Expected >55% reduction, got {reduction}%"


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 5 — Product catalogue from an external API
# ─────────────────────────────────────────────────────────────────────────────


class TestProductCatalogue:
    """
    An inventory agent is asked: "Which products are low stock and what is
    their margin? Also rank by rating."

    API responses include long description text, image arrays, full spec objects,
    and supplier details that are not needed for this specific question.
    """

    def test_low_stock_alert(self):
        result = apply_focus(
            PRODUCT_CATALOGUE,
            "stock < 100 | SELECT id, name, stock, price, category | ORDER BY stock ASC",
        )
        assert len(result) == 2
        assert result[0]["stock"] <= result[1]["stock"]
        # The desk lamp is critically low at 8 units
        assert result[0]["id"] == "SKU-004"
        # Description and images stripped
        assert "description" not in result[0]
        assert "images" not in result[0]

    def test_margin_report(self):
        result = apply_focus(
            PRODUCT_CATALOGUE, "SELECT id, name, price, margin | ORDER BY margin DESC"
        )
        assert len(result) == 4
        # Desk Lamp has highest margin at 66.3%
        assert result[0]["id"] == "SKU-004"
        assert "description" not in result[0]
        assert "specs" not in result[0]

    def test_top_rated_products(self):
        result = apply_focus(PRODUCT_CATALOGUE, "SELECT id, name, price, category")
        # All 4 returned, only business-relevant fields
        assert len(result) == 4
        assert all(set(r.keys()) == {"id", "name", "price", "category"} for r in result)

    def test_electronics_by_subcategory(self):
        result = apply_focus(
            PRODUCT_CATALOGUE,
            "category == 'electronics' | GROUP BY subcategory | SELECT subcategory, COUNT(*) AS products, AVG(price) AS avg_price",
        )
        subcats = {r["subcategory"]: r for r in result}
        assert "audio" in subcats
        assert "peripherals" in subcats
        assert subcats["audio"]["products"] == 1

    def test_supplier_consolidation(self):
        # Which supplier IDs appear and how many SKUs do they supply?
        # Note: supplier is a nested dict — access via dot notation
        result = apply_focus(
            PRODUCT_CATALOGUE, "SELECT supplier.id, supplier.name, id, name"
        )
        supplier_ids = [r.get("id") for r in result]
        # We get the nested supplier.id resolved to "id" key (leaf name)
        assert len(result) == 4

    def test_token_reduction_strips_descriptions(self):
        focused = apply_focus(
            PRODUCT_CATALOGUE, "stock < 100 | SELECT id, name, stock, margin, price"
        )
        reduction = _reduction_pct(PRODUCT_CATALOGUE, focused)
        # Descriptions, image arrays, specs objects, and supplier details all stripped
        assert reduction > 75, f"Expected >75% reduction, got {reduction}%"


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 6 — Token reduction measurement across all scenarios
# ─────────────────────────────────────────────────────────────────────────────


class TestTokenReduction:
    """
    Verify that the focus system materially reduces payload size across all
    real-world scenarios. These numbers represent approximate token savings.
    """

    def test_print_reduction_summary(self, capsys):
        """Print a human-readable reduction table for each scenario."""
        scenarios = [
            (
                "E-commerce (errors + noise stripped)",
                ORDERS,
                "status == 'completed' | SELECT order_id, product, amount, region",
            ),
            (
                "API logs (INFO filtered out)",
                API_LOGS,
                "level == 'ERROR' | SELECT timestamp, service, status_code, message",
            ),
            (
                "User events (funnel only)",
                USER_EVENTS,
                "event in ['upgrade_prompt', 'upgrade_start', 'upgrade_complete'] | SELECT user_id, event, ts",
            ),
            (
                "Transactions (flagged only, lean fields)",
                TRANSACTIONS,
                "flagged == True | SELECT txn_id, account_id, amount, country, velocity_1h",
            ),
            (
                "Product catalogue (low stock + key fields)",
                PRODUCT_CATALOGUE,
                "stock < 100 | SELECT id, name, stock, margin, price",
            ),
        ]

        print("\n" + "=" * 65)
        print(f"{'Scenario':<42} {'Before':>7} {'After':>7} {'Saved':>7}")
        print("=" * 65)

        total_before = total_after = 0
        for label, data, dsl in scenarios:
            focused = apply_focus(data, dsl)
            before = _token_estimate(data)
            after = _token_estimate(focused)
            pct = _reduction_pct(data, focused)
            total_before += before
            total_after += after
            print(f"  {label:<40} {before:>6}t {after:>6}t  -{pct:>4}%")

        total_pct = round((1 - total_after / total_before) * 100, 1)
        print("=" * 65)
        print(f"  {'TOTAL':<40} {total_before:>6}t {total_after:>6}t  -{total_pct:>4}%")
        print("=" * 65)

        assert total_pct > 50, f"Expected overall >50% reduction, got {total_pct}%"

    def test_all_scenarios_reduce_tokens(self):
        scenarios = [
            (ORDERS, "status == 'completed' | SELECT order_id, product, amount"),
            (API_LOGS, "level == 'ERROR' | SELECT timestamp, service, message"),
            (
                USER_EVENTS,
                "event in ['upgrade_prompt', 'upgrade_complete'] | SELECT user_id, event",
            ),
            (TRANSACTIONS, "flagged == True | SELECT txn_id, account_id, amount"),
            (PRODUCT_CATALOGUE, "stock < 100 | SELECT id, name, stock"),
        ]
        for data, dsl in scenarios:
            focused = apply_focus(data, dsl)
            assert _token_estimate(focused) < _token_estimate(
                data
            ), f"Focus did not reduce tokens for DSL: {dsl}"
