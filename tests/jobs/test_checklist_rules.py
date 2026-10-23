"""checklist_write's rules: replace or merge by id, `after` order, no cycles."""

from __future__ import annotations

import pytest

from daita.jobs import ChecklistItem, ChecklistStatus, JobError
from daita.jobs.models import MAX_CHECKLIST_ITEMS, written_checklist

pytestmark = pytest.mark.unit

PLAN = (
    {"id": "a", "title": "Read A"},
    {"id": "b", "title": "Read B"},
    {"id": "c", "title": "Combine", "after": ["a", "b"]},
)


def test_a_write_replaces_and_a_merge_keeps_omitted_fields() -> None:
    planned = written_checklist((), PLAN, merge=False)
    assert [item.id for item in planned] == ["a", "b", "c"]
    assert all(item.status is ChecklistStatus.PENDING for item in planned)

    merged = written_checklist(
        planned,
        ({"id": "a", "status": "done"}, {"id": "d", "title": "Report"}),
        merge=True,
    )
    assert [item.id for item in merged] == ["a", "b", "c", "d"]
    assert merged[0] == ChecklistItem("a", "Read A", ChecklistStatus.DONE)
    assert merged[2].after == ("a", "b")

    replaced = written_checklist(merged, ({"id": "z", "title": "Only"},), merge=False)
    assert [item.id for item in replaced] == ["z"]


@pytest.mark.parametrize("status", ["in_progress", "done"])
def test_starting_or_finishing_an_item_before_its_after_items_is_refused(
    status: str,
) -> None:
    planned = written_checklist((), PLAN, merge=False)
    planned = written_checklist(planned, ({"id": "a", "status": "done"},), merge=True)
    with pytest.raises(JobError) as refused:
        written_checklist(planned, ({"id": "c", "status": status},), merge=True)
    assert refused.value.code == "checklist_item_blocked"
    # The error names exactly the items still in the way.
    assert "b" in str(refused.value) and "a" not in str(refused.value).split(":")[1]


def test_a_cancelled_item_does_not_satisfy_its_dependants() -> None:
    planned = written_checklist((), PLAN, merge=False)
    planned = written_checklist(
        planned,
        ({"id": "a", "status": "done"}, {"id": "b", "status": "cancelled"}),
        merge=True,
    )
    with pytest.raises(JobError, match="b"):
        written_checklist(planned, ({"id": "c", "status": "in_progress"},), merge=True)


@pytest.mark.parametrize(
    ("items", "named"),
    [
        (({"id": "a", "title": "A", "after": ["missing"]},), "missing"),
        (
            (
                {"id": "a", "title": "A", "after": ["b"]},
                {"id": "b", "title": "B", "after": ["a"]},
            ),
            "cycle",
        ),
    ],
)
def test_unknown_after_ids_and_cycles_are_refused(
    items: tuple[dict[str, object], ...], named: str
) -> None:
    with pytest.raises(JobError) as refused:
        written_checklist((), items, merge=False)
    assert refused.value.code == "checklist_invalid"
    assert named in str(refused.value)


def test_model_text_is_sanitized_and_the_list_is_bounded() -> None:
    (item,) = written_checklist(
        (),
        ({"id": "a", "title": "Read\x1b[31m A\nnow", "note": "line one\nline two"},),
        merge=False,
    )
    assert item.title == "Read?[31m A now"
    assert item.note == "line one\nline two"
    too_many = tuple(
        {"id": f"i{index}", "title": "Item"} for index in range(MAX_CHECKLIST_ITEMS + 1)
    )
    with pytest.raises(JobError) as refused:
        written_checklist((), too_many, merge=False)
    assert refused.value.code == "checklist_too_long"
