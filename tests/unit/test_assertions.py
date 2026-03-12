"""
Unit tests for daita/core/assertions.py.

Covers ItemAssertion, RowAssertion alias, and _evaluate_assertions.
"""

import pytest
from daita.core.assertions import ItemAssertion, RowAssertion, _evaluate_assertions
from daita.core.exceptions import DataQualityError


# ---------------------------------------------------------------------------
# ItemAssertion dataclass
# ---------------------------------------------------------------------------


def test_item_assertion_stores_check_and_description():
    check = lambda r: r["x"] > 0
    a = ItemAssertion(check=check, description="x must be positive")
    assert a.check is check
    assert a.description == "x must be positive"


def test_row_assertion_is_item_assertion():
    assert RowAssertion is ItemAssertion


def test_item_assertion_check_returns_true_for_valid_item():
    a = ItemAssertion(lambda r: r["amount"] > 0, "amount positive")
    assert a.check({"amount": 5}) is True


def test_item_assertion_check_returns_false_for_invalid_item():
    a = ItemAssertion(lambda r: r["amount"] > 0, "amount positive")
    assert a.check({"amount": -1}) is False


# ---------------------------------------------------------------------------
# _evaluate_assertions — happy paths
# ---------------------------------------------------------------------------


def test_evaluate_empty_items_no_error():
    assertions = [ItemAssertion(lambda r: r["x"] > 0, "x positive")]
    _evaluate_assertions([], assertions)  # must not raise


def test_evaluate_empty_assertions_no_error():
    items = [{"x": -1}]
    _evaluate_assertions(items, [])  # must not raise


def test_evaluate_all_pass_no_error():
    items = [{"amount": 10}, {"amount": 20}, {"amount": 5}]
    assertions = [
        ItemAssertion(lambda r: r["amount"] > 0, "amount positive"),
        ItemAssertion(lambda r: r["amount"] < 100, "amount under 100"),
    ]
    _evaluate_assertions(items, assertions)  # must not raise


# ---------------------------------------------------------------------------
# _evaluate_assertions — violation paths
# ---------------------------------------------------------------------------


def test_evaluate_single_violation_raises():
    items = [{"amount": 10}, {"amount": -5}]
    assertions = [ItemAssertion(lambda r: r["amount"] > 0, "amount positive")]
    with pytest.raises(DataQualityError):
        _evaluate_assertions(items, assertions)


def test_evaluate_violation_message_contains_description():
    items = [{"amount": -5}]
    assertions = [ItemAssertion(lambda r: r["amount"] > 0, "amount must be positive")]
    with pytest.raises(DataQualityError, match="amount must be positive"):
        _evaluate_assertions(items, assertions)


def test_evaluate_violation_message_contains_counts():
    items = [{"x": 1}, {"x": -1}, {"x": -2}]
    assertions = [ItemAssertion(lambda r: r["x"] > 0, "x positive")]
    with pytest.raises(DataQualityError, match=r"2/3"):
        _evaluate_assertions(items, assertions)


def test_evaluate_collects_all_violations_before_raising():
    items = [{"a": -1, "b": -1}]
    assertions = [
        ItemAssertion(lambda r: r["a"] > 0, "a positive"),
        ItemAssertion(lambda r: r["b"] > 0, "b positive"),
    ]
    with pytest.raises(DataQualityError) as exc_info:
        _evaluate_assertions(items, assertions)
    assert len(exc_info.value.violations) == 2


def test_evaluate_violation_has_expected_keys():
    items = [{"v": -1}, {"v": 2}]
    assertions = [ItemAssertion(lambda r: r["v"] > 0, "v positive")]
    with pytest.raises(DataQualityError) as exc_info:
        _evaluate_assertions(items, assertions)
    v = exc_info.value.violations[0]
    assert v["description"] == "v positive"
    assert v["violation_count"] == 1
    assert v["total_items"] == 2
    assert len(v["sample"]) == 1
    assert v["sample"][0] == {"v": -1}


def test_evaluate_sample_capped_at_three():
    items = [{"x": -i} for i in range(1, 10)]  # 9 failing items
    assertions = [ItemAssertion(lambda r: r["x"] > 0, "x positive")]
    with pytest.raises(DataQualityError) as exc_info:
        _evaluate_assertions(items, assertions)
    assert len(exc_info.value.violations[0]["sample"]) == 3


def test_evaluate_source_included_in_context():
    items = [{"x": -1}]
    assertions = [ItemAssertion(lambda r: r["x"] > 0, "x positive")]
    with pytest.raises(DataQualityError) as exc_info:
        _evaluate_assertions(items, assertions, source="SELECT x FROM t")
    assert exc_info.value.context.get("source") == "SELECT x FROM t"


def test_evaluate_source_truncated_at_200_chars():
    long_source = "SELECT " + "x" * 300
    items = [{"x": -1}]
    assertions = [ItemAssertion(lambda r: r["x"] > 0, "x positive")]
    with pytest.raises(DataQualityError) as exc_info:
        _evaluate_assertions(items, assertions, source=long_source)
    assert len(exc_info.value.context.get("source", "")) == 200


# ---------------------------------------------------------------------------
# _evaluate_assertions — assertion that raises internally
# ---------------------------------------------------------------------------


def test_evaluate_assertion_that_raises_is_skipped_not_propagated(caplog):
    """An assertion callable that raises should be logged and skipped, not crash."""
    import logging

    boom = ItemAssertion(lambda r: 1 / 0, "always explodes")
    good = ItemAssertion(lambda r: r["x"] > 0, "x positive")

    items = [{"x": 5}]
    with caplog.at_level(logging.DEBUG, logger="daita.core.assertions"):
        _evaluate_assertions(items, [boom, good])  # good assertion passes → no raise

    assert any("always explodes" in r.message for r in caplog.records)


def test_evaluate_assertion_raises_only_on_real_violations():
    """Crashing assertion is skipped; a real violation in another assertion still raises."""
    boom = ItemAssertion(lambda r: 1 / 0, "always explodes")
    bad = ItemAssertion(lambda r: r["x"] > 0, "x positive")

    items = [{"x": -1}]
    with pytest.raises(DataQualityError, match="x positive"):
        _evaluate_assertions(items, [boom, bad])


# ---------------------------------------------------------------------------
# DataQualityError properties
# ---------------------------------------------------------------------------


def test_data_quality_error_is_permanent():
    from daita.core.exceptions import PermanentError

    err = DataQualityError("bad data", violations=[])
    assert isinstance(err, PermanentError)
    assert err.retry_hint == "permanent"


def test_data_quality_error_stores_violations():
    violations = [{"description": "x positive", "violation_count": 2}]
    err = DataQualityError("bad", violations=violations)
    assert err.violations == violations


def test_data_quality_error_stores_table():
    err = DataQualityError("bad", table="orders")
    assert err.table == "orders"
    assert err.context["table"] == "orders"


def test_data_quality_error_default_empty_violations():
    err = DataQualityError("bad")
    assert err.violations == []
