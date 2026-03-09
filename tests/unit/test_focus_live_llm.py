"""
Live LLM integration tests for the Focus DSL system.

These tests run against the real OpenAI API to verify that:
  1. The LLM receives only the focused (reduced) data, not the full payload
  2. The LLM still produces correct, useful answers despite seeing less data
  3. Token usage is measurably lower with focus applied
  4. The agent's tool → focus → LLM pipeline works end-to-end

Requires: OPENAI_API_KEY set in environment, or passed at run time.
Run with:
    OPENAI_API_KEY=sk-... pytest tests/unit/test_focus_live_llm.py -v -s -m requires_llm
"""
import json
import os
import pytest

from daita.agents.agent import Agent
from daita.core.focus import apply_focus
from daita.core.tools import AgentTool


# ─────────────────────────────────────────────────────────────────────────────
# Realistic datasets (same as test_focus_real_world.py)
# ─────────────────────────────────────────────────────────────────────────────

ORDERS = [
    {"order_id": "ORD-001", "customer_id": "CUST-042", "customer_name": "Lena Fischer",
     "email": "lena@example.com", "product": "Pro Plan", "category": "subscription",
     "amount": 299.00, "status": "completed", "region": "EU",
     "created_at": "2024-03-01T09:12:00Z", "updated_at": "2024-03-01T09:14:00Z",
     "billing_address": {"street": "Musterstraße 1", "city": "Berlin", "zip": "10115"},
     "payment_method": "credit_card", "refunded": False, "notes": None},

    {"order_id": "ORD-002", "customer_id": "CUST-007", "customer_name": "James Park",
     "email": "james@example.com", "product": "Starter Plan", "category": "subscription",
     "amount": 49.00, "status": "completed", "region": "US",
     "created_at": "2024-03-02T14:05:00Z", "updated_at": "2024-03-02T14:06:00Z",
     "billing_address": {"street": "123 Main St", "city": "Austin", "zip": "73301"},
     "payment_method": "paypal", "refunded": False, "notes": None},

    {"order_id": "ORD-003", "customer_id": "CUST-099", "customer_name": "Sara Nkosi",
     "email": "sara@example.com", "product": "Enterprise", "category": "subscription",
     "amount": 999.00, "status": "refunded", "region": "EU",
     "created_at": "2024-03-03T11:00:00Z", "updated_at": "2024-03-04T08:30:00Z",
     "billing_address": {"street": "Rue de la Paix 5", "city": "Paris", "zip": "75001"},
     "payment_method": "credit_card", "refunded": True, "notes": "Customer requested refund"},

    {"order_id": "ORD-004", "customer_id": "CUST-011", "customer_name": "Hiroshi Tanaka",
     "email": "hiroshi@example.com", "product": "Pro Plan", "category": "subscription",
     "amount": 299.00, "status": "completed", "region": "APAC",
     "created_at": "2024-03-04T02:45:00Z", "updated_at": "2024-03-04T02:46:00Z",
     "billing_address": {"street": "Shibuya 1-1", "city": "Tokyo", "zip": "150-0001"},
     "payment_method": "credit_card", "refunded": False, "notes": None},

    {"order_id": "ORD-005", "customer_id": "CUST-055", "customer_name": "Maria Garcia",
     "email": "maria@example.com", "product": "Starter Plan", "category": "subscription",
     "amount": 49.00, "status": "pending", "region": "US",
     "created_at": "2024-03-05T17:22:00Z", "updated_at": "2024-03-05T17:22:00Z",
     "billing_address": {"street": "456 Oak Ave", "city": "Miami", "zip": "33101"},
     "payment_method": "bank_transfer", "refunded": False, "notes": "Awaiting payment confirmation"},
]

API_LOGS = [
    {"log_id": "L001", "timestamp": "2024-03-05T10:00:01Z", "service": "auth",
     "level": "INFO", "status_code": 200, "latency_ms": 45, "user_id": "u123",
     "endpoint": "/login", "method": "POST", "ip": "192.168.1.10",
     "message": "Login successful", "trace_id": "t-aaa", "request_size_bytes": 128},

    {"log_id": "L002", "timestamp": "2024-03-05T10:00:05Z", "service": "api",
     "level": "ERROR", "status_code": 500, "latency_ms": 1200, "user_id": "u456",
     "endpoint": "/data/export", "method": "GET", "ip": "10.0.0.5",
     "message": "Timeout connecting to database", "trace_id": "t-bbb", "request_size_bytes": 64},

    {"log_id": "L003", "timestamp": "2024-03-05T10:00:09Z", "service": "api",
     "level": "INFO", "status_code": 200, "latency_ms": 88, "user_id": "u789",
     "endpoint": "/reports", "method": "GET", "ip": "10.0.0.8",
     "message": "Report generated", "trace_id": "t-ccc", "request_size_bytes": 32},

    {"log_id": "L004", "timestamp": "2024-03-05T10:00:12Z", "service": "billing",
     "level": "ERROR", "status_code": 402, "latency_ms": 230, "user_id": "u456",
     "endpoint": "/charge", "method": "POST", "ip": "10.0.0.5",
     "message": "Payment declined: insufficient funds", "trace_id": "t-ddd", "request_size_bytes": 256},

    {"log_id": "L005", "timestamp": "2024-03-05T10:00:15Z", "service": "auth",
     "level": "WARN", "status_code": 401, "latency_ms": 12, "user_id": None,
     "endpoint": "/login", "method": "POST", "ip": "203.0.113.42",
     "message": "Failed login attempt", "trace_id": "t-eee", "request_size_bytes": 96},

    {"log_id": "L006", "timestamp": "2024-03-05T10:00:20Z", "service": "api",
     "level": "INFO", "status_code": 200, "latency_ms": 55, "user_id": "u123",
     "endpoint": "/dashboard", "method": "GET", "ip": "192.168.1.10",
     "message": "Dashboard loaded", "trace_id": "t-fff", "request_size_bytes": 32},

    {"log_id": "L007", "timestamp": "2024-03-05T10:00:25Z", "service": "api",
     "level": "ERROR", "status_code": 503, "latency_ms": 5001, "user_id": "u321",
     "endpoint": "/search", "method": "GET", "ip": "10.0.1.2",
     "message": "Search service unavailable", "trace_id": "t-ggg", "request_size_bytes": 48},
]

TRANSACTIONS = [
    {"txn_id": "T001", "account_id": "ACC-100", "amount": 42.50, "currency": "USD",
     "type": "purchase", "merchant": "Amazon", "merchant_category": "retail",
     "country": "US", "ts": "2024-03-05T08:10:00Z", "flagged": False,
     "device_fingerprint": "fp-aaa", "ip": "1.2.3.4", "velocity_1h": 1, "card_present": True},

    {"txn_id": "T002", "account_id": "ACC-100", "amount": 9850.00, "currency": "USD",
     "type": "transfer", "merchant": None, "merchant_category": None,
     "country": "RU", "ts": "2024-03-05T08:11:00Z", "flagged": True,
     "device_fingerprint": "fp-zzz", "ip": "195.3.3.3", "velocity_1h": 12, "card_present": False},

    {"txn_id": "T003", "account_id": "ACC-200", "amount": 15.00, "currency": "USD",
     "type": "purchase", "merchant": "Starbucks", "merchant_category": "food",
     "country": "US", "ts": "2024-03-05T09:00:00Z", "flagged": False,
     "device_fingerprint": "fp-bbb", "ip": "5.6.7.8", "velocity_1h": 1, "card_present": True},

    {"txn_id": "T004", "account_id": "ACC-200", "amount": 5200.00, "currency": "USD",
     "type": "purchase", "merchant": "Electronics Plus", "merchant_category": "retail",
     "country": "US", "ts": "2024-03-05T09:05:00Z", "flagged": True,
     "device_fingerprint": "fp-bbb", "ip": "5.6.7.8", "velocity_1h": 8, "card_present": False},

    {"txn_id": "T005", "account_id": "ACC-300", "amount": 2.99, "currency": "USD",
     "type": "subscription", "merchant": "Netflix", "merchant_category": "streaming",
     "country": "US", "ts": "2024-03-05T10:00:00Z", "flagged": False,
     "device_fingerprint": "fp-ccc", "ip": "9.10.11.12", "velocity_1h": 1, "card_present": False},

    {"txn_id": "T006", "account_id": "ACC-100", "amount": 1.00, "currency": "USD",
     "type": "purchase", "merchant": "Test Merchant", "merchant_category": "other",
     "country": "NG", "ts": "2024-03-05T08:15:00Z", "flagged": True,
     "device_fingerprint": "fp-zzz", "ip": "195.3.3.3", "velocity_1h": 14, "card_present": False},
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _token_estimate(data) -> int:
    return len(json.dumps(data, default=str)) // 4


def _make_tool(name: str, description: str, data) -> AgentTool:
    """Return a no-arg tool that yields a fixed dataset."""
    async def _handler(_args):
        return data

    return AgentTool(
        name=name,
        description=description,
        parameters={"type": "object", "properties": {}, "required": []},
        handler=_handler,
    )


def _make_agent(tool: AgentTool, focus: str | None = None) -> Agent:
    api_key = os.environ["OPENAI_API_KEY"]
    return Agent(
        name="FocusTestAgent",
        llm_provider="openai",
        model="gpt-4o-mini",
        api_key=api_key,
        tools=[tool],
        focus=focus,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.requires_llm
class TestFocusWithLiveOpenAI:

    async def test_order_summary_without_focus(self, capsys):
        """
        Baseline: agent receives full order records (emails, billing addresses,
        timestamps, etc.) and answers a revenue question.
        Demonstrates the token cost WITHOUT focus.
        """
        tool = _make_tool(
            "get_orders",
            "Fetch all orders for this month",
            ORDERS,
        )
        agent = _make_agent(tool, focus=None)

        result = await agent.run_detailed(
            "Using the get_orders tool, tell me total completed revenue and how many orders are pending."
        )

        tokens_used = result.get("tokens", {}).get("total_tokens", 0)
        print(f"\n[NO FOCUS]  tokens={tokens_used}  answer={result['result'][:120]}")

        assert result["result"]
        # $299 + $49 + $299 = $647 completed; 1 pending
        assert "647" in result["result"] or "647.00" in result["result"]
        assert "1" in result["result"] or "pending" in result["result"].lower()

    async def test_order_summary_with_focus(self, capsys):
        """
        Same question, same tool — but focus strips billing addresses, emails,
        timestamps, payment methods, and notes before the LLM sees the data.
        Answer must still be correct.
        """
        tool = _make_tool(
            "get_orders",
            "Fetch all orders for this month",
            ORDERS,
        )
        agent = _make_agent(
            tool,
            focus="SELECT order_id, product, amount, status, region",
        )

        result = await agent.run_detailed(
            "Using the get_orders tool, tell me total completed revenue and how many orders are pending."
        )

        tokens_used = result.get("tokens", {}).get("total_tokens", 0)
        print(f"\n[WITH FOCUS] tokens={tokens_used}  answer={result['result'][:120]}")

        assert result["result"]
        assert "647" in result["result"] or "647.00" in result["result"]
        assert "1" in result["result"] or "pending" in result["result"].lower()

    async def test_focus_reduces_tokens(self, capsys):
        """
        Run both focused and unfocused versions and assert that focus
        measurably reduces the token count reported by the API.
        """
        tool_unfocused = _make_tool("get_orders", "Fetch all orders", ORDERS)
        tool_focused   = _make_tool("get_orders", "Fetch all orders", ORDERS)

        agent_no_focus = _make_agent(tool_unfocused, focus=None)
        agent_focused  = _make_agent(
            tool_focused,
            focus="SELECT order_id, product, amount, status, region",
        )

        question = "Using the get_orders tool, what is the total revenue from completed orders?"

        result_no_focus = await agent_no_focus.run_detailed(question)
        result_focused  = await agent_focused.run_detailed(question)

        tokens_no_focus = result_no_focus.get("tokens", {}).get("total_tokens", 0)
        tokens_focused  = result_focused.get("tokens", {}).get("total_tokens", 0)

        raw_payload_tokens   = _token_estimate(ORDERS)
        focused_payload_tokens = _token_estimate(
            apply_focus(ORDERS, "SELECT order_id, product, amount, status, region")
        )
        payload_reduction = round((1 - focused_payload_tokens / raw_payload_tokens) * 100, 1)

        print(f"\n{'='*55}")
        print(f"  Payload tokens  (raw):    {raw_payload_tokens}")
        print(f"  Payload tokens  (focused):{focused_payload_tokens}  (-{payload_reduction}%)")
        print(f"  API total tokens (no focus): {tokens_no_focus}")
        print(f"  API total tokens (focused):  {tokens_focused}")
        if tokens_no_focus and tokens_focused:
            api_reduction = round((1 - tokens_focused / tokens_no_focus) * 100, 1)
            print(f"  API-level reduction:         -{api_reduction}%")
        print(f"{'='*55}")

        assert tokens_focused < tokens_no_focus, (
            f"Expected focused run to use fewer tokens: "
            f"focused={tokens_focused}, unfocused={tokens_no_focus}"
        )

    async def test_error_log_diagnosis(self, capsys):
        """
        Agent is given noisy mixed-level logs. Focus strips INFO/WARN and
        irrelevant fields. The LLM should correctly identify 3 errors and
        name the affected services.
        """
        tool = _make_tool(
            "get_logs",
            "Fetch recent service logs",
            API_LOGS,
        )
        agent = _make_agent(
            tool,
            focus="level == 'ERROR' | SELECT timestamp, service, status_code, latency_ms, message",
        )

        result = await agent.run_detailed(
            "Use get_logs to identify all service errors. "
            "List each error with its service name and what went wrong."
        )

        answer = result["result"].lower()
        print(f"\n[LOG DIAGNOSIS] answer={result['result'][:200]}")

        # The three errors must be surfaced
        assert "api" in answer or "database" in answer
        assert "billing" in answer or "payment" in answer
        assert "search" in answer or "unavailable" in answer or "503" in answer

    async def test_fraud_signal_extraction(self, capsys):
        """
        Agent reviews transaction data with focus filtering flagged=True records
        and stripping device fingerprints, IPs, and card_present fields.
        The LLM must correctly identify ACC-100 as the account with multiple flags.
        """
        tool = _make_tool(
            "get_transactions",
            "Fetch recent financial transactions",
            TRANSACTIONS,
        )
        agent = _make_agent(
            tool,
            focus="flagged == True | SELECT txn_id, account_id, amount, type, country, velocity_1h | ORDER BY amount DESC",
        )

        result = await agent.run_detailed(
            "Use get_transactions to identify suspicious activity. "
            "Which account has the most flagged transactions and what is the total flagged amount for that account?"
        )

        answer = result["result"]
        print(f"\n[FRAUD SIGNALS] answer={answer[:200]}")

        # ACC-100 has 2 flags: T002 ($9850) + T006 ($1) = $9851
        assert "ACC-100" in answer
        assert "9851" in answer or "9,851" in answer or "9850" in answer

    async def test_focused_vs_unfocused_answer_quality(self, capsys):
        """
        Side-by-side comparison of answer quality.
        Both versions should identify the correct highest-revenue product.
        This confirms focus doesn't degrade answer correctness.
        """
        question = (
            "Use get_orders to determine which product generated the most revenue "
            "from completed orders only. Give me the product name and exact dollar amount."
        )

        # Unfocused
        agent_full = _make_agent(
            _make_tool("get_orders", "Fetch all orders", ORDERS),
            focus=None,
        )
        result_full = await agent_full.run_detailed(question)

        # Focused: only the fields needed to answer the question
        agent_lean = _make_agent(
            _make_tool("get_orders", "Fetch all orders", ORDERS),
            focus="status == 'completed' | SELECT product, amount",
        )
        result_lean = await agent_lean.run_detailed(question)

        tokens_full = result_full.get("tokens", {}).get("total_tokens", 0)
        tokens_lean = result_lean.get("tokens", {}).get("total_tokens", 0)

        print(f"\n{'='*55}")
        print(f"  [FULL]    tokens={tokens_full}")
        print(f"  answer: {result_full['result'][:140]}")
        print(f"\n  [FOCUSED] tokens={tokens_lean}")
        print(f"  answer: {result_lean['result'][:140]}")
        print(f"{'='*55}")

        # Pro Plan: ORD-001 ($299) + ORD-004 ($299) = $598 — highest completed revenue
        for result in (result_full, result_lean):
            answer = result["result"]
            assert "Pro Plan" in answer, f"Expected 'Pro Plan' in answer: {answer}"
            assert "598" in answer or "598.00" in answer, f"Expected $598 in answer: {answer}"

        # Focused should cost fewer tokens
        assert tokens_lean < tokens_full
