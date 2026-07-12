import pytest

from daita.db.loop.utils import _string_list


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (None, []),
        ("  ", []),
        ("  orders  ", ["orders"]),
        ((" orders ", "", 7), ["orders", "7"]),
        ([None, " customers "], ["None", "customers"]),
        (42, ["42"]),
    ),
)
def test_string_list_preserves_loop_normalization_contract(value, expected):
    assert _string_list(value) == expected
