"""
Unit tests for the Focus DSL system.

Covers: parser, evaluator, backends (dict + pandas), and apply_focus end-to-end.
No API keys or databases required.
"""
import pytest

from daita.core.focus import apply_focus, parse, FocusQuery
from daita.core.exceptions import FocusDSLError


# ── Sample data fixtures ──────────────────────────────────────────────────────

PRODUCTS = [
    {"name": "Widget", "price": 50,  "status": "active",   "region": "US"},
    {"name": "Gadget", "price": 150, "status": "active",   "region": "EU"},
    {"name": "Doohickey", "price": 200, "status": "inactive", "region": "US"},
    {"name": "Thingamajig", "price": 75, "status": "active", "region": "EU"},
]

NESTED = [
    {"user": {"name": "Alice", "age": 30}, "score": 90},
    {"user": {"name": "Bob",   "age": 25}, "score": 60},
    {"user": {"name": "Carol", "age": 35}, "score": 80},
]


# ── Parser tests ──────────────────────────────────────────────────────────────

class TestParser:

    def test_empty_string_raises(self):
        with pytest.raises(FocusDSLError):
            parse("")

    def test_whitespace_only_raises(self):
        with pytest.raises(FocusDSLError):
            parse("   ")

    def test_parse_select_only(self):
        q = parse("SELECT name, price")
        assert q.select == ["name", "price"]
        assert q.filter_ast is None

    def test_parse_filter_only(self):
        q = parse("price > 100")
        assert q.filter_expr == "price > 100"
        assert q.filter_ast is not None
        assert q.select is None

    def test_parse_combined(self):
        q = parse("price > 100 | SELECT name, price | LIMIT 10")
        assert q.filter_expr == "price > 100"
        assert q.select == ["name", "price"]
        assert q.limit == 10

    def test_parse_order_asc(self):
        q = parse("SELECT name | ORDER BY price ASC")
        assert q.order_by == "price"
        assert q.order_dir == "ASC"

    def test_parse_order_desc(self):
        q = parse("SELECT name | ORDER BY price DESC")
        assert q.order_dir == "DESC"

    def test_parse_group_by(self):
        q = parse("GROUP BY region | SELECT region, SUM(price) AS total")
        assert q.group_by == ["region"]
        assert q.aggregates == {"total": "SUM(price)"}

    def test_parse_aggregate_no_alias(self):
        q = parse("SELECT name, COUNT(*)")
        assert "count_all" in q.aggregates

    def test_parse_invalid_filter_raises(self):
        with pytest.raises(FocusDSLError, match="Invalid filter"):
            parse("price >>>")

    def test_parse_duplicate_filter_raises(self):
        with pytest.raises(FocusDSLError, match="Multiple filter"):
            parse("price > 10 | status == 'active'")

    def test_parse_invalid_limit_raises(self):
        with pytest.raises(FocusDSLError):
            parse("LIMIT abc")

    def test_parse_negative_limit_raises(self):
        with pytest.raises(FocusDSLError):
            parse("LIMIT -5")

    def test_parse_invalid_order_direction_raises(self):
        with pytest.raises(FocusDSLError):
            parse("ORDER BY price SIDEWAYS")

    def test_wrong_focus_type_raises(self):
        with pytest.raises(FocusDSLError):
            apply_focus(PRODUCTS, ["name", "price"])  # old list format

    def test_focus_query_object_accepted(self):
        q = parse("SELECT name")
        result = apply_focus(PRODUCTS, q)
        assert all("name" in r for r in result)


# ── Dict / list-of-dicts backend tests ───────────────────────────────────────

class TestDictBackend:

    def test_select_fields(self):
        result = apply_focus(PRODUCTS, "SELECT name, price")
        assert result == [
            {"name": "Widget", "price": 50},
            {"name": "Gadget", "price": 150},
            {"name": "Doohickey", "price": 200},
            {"name": "Thingamajig", "price": 75},
        ]

    def test_filter_gt(self):
        result = apply_focus(PRODUCTS, "price > 100")
        assert len(result) == 2
        assert all(r["price"] > 100 for r in result)

    def test_filter_eq_string(self):
        result = apply_focus(PRODUCTS, "status == 'active'")
        assert len(result) == 3
        assert all(r["status"] == "active" for r in result)

    def test_filter_and(self):
        result = apply_focus(PRODUCTS, "price > 100 and status == 'active'")
        assert len(result) == 1
        assert result[0]["name"] == "Gadget"

    def test_filter_or(self):
        result = apply_focus(PRODUCTS, "region == 'US' or price > 150")
        assert len(result) == 2

    def test_filter_in(self):
        result = apply_focus(PRODUCTS, "region in ['EU']")
        assert len(result) == 2
        assert all(r["region"] == "EU" for r in result)

    def test_filter_not(self):
        result = apply_focus(PRODUCTS, "not status == 'active'")
        assert len(result) == 1
        assert result[0]["name"] == "Doohickey"

    def test_limit(self):
        result = apply_focus(PRODUCTS, "LIMIT 2")
        assert len(result) == 2

    def test_order_by_asc(self):
        result = apply_focus(PRODUCTS, "ORDER BY price ASC")
        prices = [r["price"] for r in result]
        assert prices == sorted(prices)

    def test_order_by_desc(self):
        result = apply_focus(PRODUCTS, "ORDER BY price DESC")
        prices = [r["price"] for r in result]
        assert prices == sorted(prices, reverse=True)

    def test_combined_filter_select_limit(self):
        result = apply_focus(
            PRODUCTS,
            "status == 'active' | SELECT name, price | ORDER BY price DESC | LIMIT 2"
        )
        assert len(result) == 2
        assert list(result[0].keys()) == ["name", "price"]
        assert result[0]["price"] >= result[1]["price"]

    def test_group_by_sum(self):
        result = apply_focus(PRODUCTS, "GROUP BY region | SELECT region, SUM(price) AS total")
        assert len(result) == 2
        totals = {r["region"]: r["total"] for r in result}
        assert totals["US"] == 250   # 50 + 200
        assert totals["EU"] == 225   # 150 + 75

    def test_group_by_count_star(self):
        result = apply_focus(PRODUCTS, "GROUP BY status | SELECT status, COUNT(*) AS cnt")
        counts = {r["status"]: r["cnt"] for r in result}
        assert counts["active"] == 3
        assert counts["inactive"] == 1

    def test_group_by_avg(self):
        result = apply_focus(PRODUCTS, "GROUP BY region | SELECT region, AVG(price) AS avg_price")
        avgs = {r["region"]: r["avg_price"] for r in result}
        assert avgs["US"] == 125.0
        assert avgs["EU"] == 112.5

    def test_nested_field_filter(self):
        result = apply_focus(NESTED, "user.age > 28")
        assert len(result) == 2
        assert all(r["user"]["age"] > 28 for r in result)

    def test_nested_field_select(self):
        result = apply_focus(NESTED, "SELECT user.name, score")
        assert all("name" in r for r in result)
        assert all("score" in r for r in result)

    def test_none_data_passthrough(self):
        assert apply_focus(None, "SELECT name") is None

    def test_none_focus_passthrough(self):
        assert apply_focus(PRODUCTS, None) is PRODUCTS

    def test_empty_list_passthrough(self):
        result = apply_focus([], "SELECT name")
        assert result == []

    def test_single_dict(self):
        data = {"name": "Widget", "price": 50, "status": "active"}
        result = apply_focus(data, "SELECT name, price")
        assert result == {"name": "Widget", "price": 50}

    def test_nonexistent_field_in_filter_excludes_row(self):
        result = apply_focus(PRODUCTS, "nonexistent_field > 100")
        assert result == []

    def test_type_mismatch_coercion(self):
        data = [{"price": 100}, {"price": 200}]
        result = apply_focus(data, "price > '150'")
        assert len(result) == 1


# ── Pandas backend tests ──────────────────────────────────────────────────────

class TestPandasBackend:
    pytest.importorskip("pandas")

    def _make_df(self):
        import pandas as pd
        return pd.DataFrame(PRODUCTS)

    def test_select_columns(self):
        import pandas as pd
        df = self._make_df()
        result = apply_focus(df, "SELECT name, price")
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["name", "price"]

    def test_filter_numeric(self):
        import pandas as pd
        df = self._make_df()
        result = apply_focus(df, "price > 100")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert (result["price"] > 100).all()

    def test_filter_string(self):
        import pandas as pd
        df = self._make_df()
        result = apply_focus(df, "status == 'active'")
        assert len(result) == 3

    def test_limit(self):
        import pandas as pd
        df = self._make_df()
        result = apply_focus(df, "LIMIT 2")
        assert len(result) == 2

    def test_order_by(self):
        import pandas as pd
        df = self._make_df()
        result = apply_focus(df, "ORDER BY price DESC")
        assert result["price"].iloc[0] == 200

    def test_combined(self):
        import pandas as pd
        df = self._make_df()
        result = apply_focus(df, "status == 'active' | SELECT name, price | ORDER BY price DESC | LIMIT 2")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["name", "price"]

    def test_group_by_sum(self):
        import pandas as pd
        df = self._make_df()
        result = apply_focus(df, "GROUP BY region | SELECT region, SUM(price) AS total")
        assert isinstance(result, pd.DataFrame)
        totals = dict(zip(result["region"], result["total"]))
        assert totals["US"] == 250
        assert totals["EU"] == 225
