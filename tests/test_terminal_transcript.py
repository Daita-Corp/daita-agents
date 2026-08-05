from __future__ import annotations

from daita.terminal_transcript import (
    INTERACTION_PRECEDENCE,
    InteractionOwner,
    RenderedCoordinate,
    TranscriptDocument,
    TranscriptFollowState,
    TranscriptViewport,
    bounded_scroll_rows,
    interaction_owner,
)


def test_presentation_ids_are_stable_monotonic_and_never_reused():
    document = TranscriptDocument()
    first = document.append("first")
    second = document.append("second")

    document.remove(first.id)
    third = document.append("third")
    document.reorder((third.id, second.id))

    assert first.id.value == 1
    assert second.id.value == 2
    assert third.id.value == 3
    assert document.presentation_ids == (third.id, second.id)


def test_semantic_anchor_survives_append_without_moving_logical_content():
    document = TranscriptDocument()
    block = document.append("alpha beta gamma")
    anchor = document.make_anchor(document.position(block.id, 6))
    before = document.project(40).resolve_anchor(anchor)

    document.append("a later answer")
    after = document.project(40).resolve_anchor(anchor)

    assert anchor.text == "b"
    assert before == RenderedCoordinate(0, 6)
    assert after == before


def test_semantic_anchor_survives_resize_and_rewrap():
    document = TranscriptDocument()
    block = document.append("alpha beta gamma")
    anchor = document.make_anchor(document.position(block.id, 6))

    wide = document.project(40).resolve_anchor(anchor)
    narrow = document.project(5).resolve_anchor(anchor)

    assert wide == RenderedCoordinate(0, 6)
    assert narrow == RenderedCoordinate(1, 1)
    assert document.reconcile_anchor(anchor) is not None


def test_tool_mutation_expansion_and_hydration_follow_prefix_suffix_rule():
    document = TranscriptDocument()
    tool = document.append("Query SQLite\nRunning")
    title_anchor = document.make_anchor(document.position(tool.id, 0))
    running_anchor = document.make_anchor(document.position(tool.id, 13))

    current = document.replace(tool.id, "Query SQLite\nSucceeded")

    assert document.reconcile_anchor(title_anchor) is not None
    assert document.reconcile_anchor(running_anchor) is None

    summary_anchor = document.make_anchor(
        document.position(current.id, len("Query SQLite\n"))
    )
    document.replace(tool.id, "Query SQLite\nSucceeded\nRecorded details")
    assert document.reconcile_anchor(summary_anchor) is not None

    hydrated = document.append("hydrated sibling")
    document.reorder((hydrated.id, tool.id))
    assert document.reconcile_anchor(title_anchor) is not None

    document.remove(tool.id)
    assert document.reconcile_anchor(title_anchor) is None


def test_ascii_tabs_and_newlines_map_both_directions_by_terminal_cells():
    document = TranscriptDocument()
    block = document.append("ab\tc\nde")
    projection = document.project(8, tab_size=4)

    assert projection.to_rendered(document.position(block.id, 0)) == (
        RenderedCoordinate(0, 0)
    )
    assert projection.to_rendered(document.position(block.id, 2)) == (
        RenderedCoordinate(0, 2)
    )
    assert projection.to_rendered(document.position(block.id, 3)) == (
        RenderedCoordinate(0, 4)
    )
    assert projection.to_rendered(document.position(block.id, 5)) == (
        RenderedCoordinate(1, 0)
    )
    assert projection.to_semantic(0, 3) == document.position(block.id, 2)
    assert projection.to_semantic(0, 4) == document.position(block.id, 3)
    assert projection.to_semantic(1, 1) == document.position(block.id, 6)


def test_combining_wide_and_emoji_sequences_use_display_cells_not_codepoints():
    document = TranscriptDocument()
    block = document.append("e\u0301界👩\u200d💻x")
    projection = document.project(4)

    assert projection.to_rendered(document.position(block.id, 2)) == (
        RenderedCoordinate(0, 1)
    )
    assert projection.to_rendered(document.position(block.id, 3)) == (
        RenderedCoordinate(1, 0)
    )
    assert projection.to_rendered(document.position(block.id, 6)) == (
        RenderedCoordinate(1, 2)
    )
    assert projection.to_semantic(0, 2) == document.position(block.id, 2)
    assert projection.to_semantic(1, 1) == document.position(block.id, 3)


def test_wrapped_code_paths_and_sanitized_control_text_remain_canonical():
    document = TranscriptDocument()
    text = "SELECT * FROM orders\n/tmp/data/very-long-file.csv\nunsafe?[2J"
    block = document.append(text)
    projection = document.project(10)

    assert document.text(block.id) == text
    assert projection.stats.source_codepoints == len(text)
    assert projection.row_count > text.count("\n") + 1
    final = projection.to_rendered(document.position(block.id, len(text)))
    assert final is not None
    assert projection.to_semantic(final.row, final.cell) == document.position(
        block.id,
        len(text),
    )


def test_ranges_normalize_in_document_order_and_preserve_exact_text():
    document = TranscriptDocument()
    first = document.append("alpha")
    second = document.append("beta")

    selected = document.normalize_range(
        document.position(second.id, 2),
        document.position(first.id, 2),
    )

    assert selected.start == document.position(first.id, 2)
    assert selected.end == document.position(second.id, 2)
    assert selected.text == "pha\nbe"


def test_range_is_invalidated_when_its_projected_text_disappears():
    document = TranscriptDocument()
    block = document.append("Summary\nCtrl+O expand")
    selected = document.normalize_range(
        document.position(block.id, len("Summary\n")),
        document.position(block.id, len(document.text(block.id))),
    )

    document.replace(block.id, "Summary\nDetails\nCtrl+O collapse")

    assert selected.text == "Ctrl+O expand"
    assert document.reconcile_range(selected) is None


def test_responsive_projection_change_keeps_prefix_and_invalidates_removed_text():
    document = TranscriptDocument()
    block = document.append("Query result\nregion revenue margin")
    title_anchor = document.make_anchor(document.position(block.id, 0))
    margin_start = document.text(block.id).index("margin")
    margin = document.normalize_range(
        document.position(block.id, margin_start),
        document.position(block.id, margin_start + len("margin")),
    )

    document.replace(block.id, "Query result\nregion revenue")

    assert document.reconcile_anchor(title_anchor) is not None
    assert document.reconcile_range(margin) is None


def test_unrelated_replacement_invalidates_anchor_instead_of_silently_snapping():
    document = TranscriptDocument()
    block = document.append("Running")
    anchor = document.make_anchor(document.position(block.id, 0))

    document.replace(block.id, "Completed")

    assert document.reconcile_anchor(anchor) is None
    assert document.project(80).resolve_anchor(anchor) is None


def test_long_transcript_projection_has_deterministic_bounded_work_counts():
    document = TranscriptDocument()
    logical_lines = 20_000
    text = "".join(
        f"/tmp/daita/data/file-{index:05d}.csv\n" for index in range(logical_lines)
    )
    block = document.append(text)

    projection = document.project(100)

    assert len(text.encode("utf-8")) < 2 * 1_024 * 1_024
    assert projection.stats.source_codepoints == len(text)
    assert projection.stats.grapheme_clusters == len(text)
    assert projection.stats.segments <= logical_lines
    assert projection.stats.rows == logical_lines + 1
    assert projection.to_rendered(document.position(block.id, len(text))) == (
        RenderedCoordinate(logical_lines, 0)
    )


def test_future_interaction_precedence_is_explicit_but_dormant():
    assert INTERACTION_PRECEDENCE == (
        InteractionOwner.APPROVAL,
        InteractionOwner.SEARCH_OR_TRANSIENT,
        InteractionOwner.TRANSCRIPT_SELECTION,
        InteractionOwner.COMPLETION_MENU,
        InteractionOwner.TRANSCRIPT_FOCUS,
        InteractionOwner.ACTIVE_RUN_CANCELLATION,
        InteractionOwner.COMPOSER_INPUT,
    )
    assert (
        interaction_owner(
            frozenset(
                {
                    InteractionOwner.COMPLETION_MENU,
                    InteractionOwner.ACTIVE_RUN_CANCELLATION,
                    InteractionOwner.APPROVAL,
                }
            )
        )
        is InteractionOwner.APPROVAL
    )
    assert interaction_owner(frozenset()) is None


def test_viewport_starts_following_and_bounded_upward_navigation_reviews():
    document = TranscriptDocument()
    document.append("\n".join(f"row-{index}" for index in range(30)))
    viewport = TranscriptViewport()
    projection = viewport.projection_for(document, width=20)

    latest = viewport.top_row(projection, viewport_rows=8)
    moved = bounded_scroll_rows(-100, viewport_rows=8)
    viewport.review_row(projection, latest + moved)

    assert latest == projection.row_count - 8
    assert viewport.state is TranscriptFollowState.REVIEWING
    assert viewport.anchor is not None
    assert moved == -8


def test_scrolling_to_latest_resumes_following_and_resets_unseen_items():
    document = TranscriptDocument()
    document.append("\n".join(f"row-{index}" for index in range(20)))
    viewport = TranscriptViewport()
    projection = viewport.projection_for(document, width=40)
    latest = viewport.top_row(projection, viewport_rows=6)
    viewport.review_row(projection, latest - 5)
    viewport.record_appended(3)

    viewport.follow_latest()

    assert viewport.state is TranscriptFollowState.FOLLOWING
    assert viewport.anchor is None
    assert viewport.unseen_items == 0


def test_reviewing_unseen_count_reconciles_disposable_block_removal():
    document = TranscriptDocument()
    document.append("older")
    viewport = TranscriptViewport()
    viewport.review_start(viewport.projection_for(document, width=20))

    viewport.record_appended()
    viewport.record_removed()

    assert viewport.state is TranscriptFollowState.REVIEWING
    assert viewport.unseen_items == 0


def test_append_while_reviewing_keeps_anchor_and_counts_bounded_new_blocks():
    document = TranscriptDocument()
    document.append("\n".join(f"row-{index}" for index in range(20)))
    viewport = TranscriptViewport()
    before = viewport.projection_for(document, width=20)
    latest = viewport.top_row(before, viewport_rows=5)
    viewport.review_row(before, latest - 4)
    anchor = viewport.anchor
    top = viewport.top_row(before, viewport_rows=5)

    document.append("new output")
    viewport.record_appended(3)
    after = viewport.projection_for(document, width=20)

    assert viewport.anchor == anchor
    assert viewport.top_row(after, viewport_rows=5) == top
    assert viewport.unseen_items == 3


def test_review_anchor_survives_rewrap_and_following_keeps_latest_visible():
    document = TranscriptDocument()
    document.append("alpha beta gamma delta epsilon")
    viewport = TranscriptViewport()
    wide = viewport.projection_for(document, width=20)
    viewport.review_row(wide, 1)
    anchor = viewport.anchor

    narrow = viewport.projection_for(document, width=8)
    reviewed_top = viewport.top_row(narrow, viewport_rows=2)
    viewport.follow_latest()
    document.append("latest")
    latest = viewport.projection_for(document, width=8)

    assert viewport.anchor is None
    assert anchor is not None
    assert document.reconcile_anchor(anchor) is not None
    assert reviewed_top > 0
    assert viewport.top_row(latest, viewport_rows=2) == latest.row_count - 2


def test_pure_navigation_reuses_one_complete_long_transcript_projection():
    document = TranscriptDocument()
    document.append("\n".join(f"row-{index:05d}" for index in range(20_000)))
    viewport = TranscriptViewport()
    projection = viewport.projection_for(document, width=100)

    for movement in (-3, -3, 3, -3, 3):
        bounded = bounded_scroll_rows(movement, viewport_rows=40)
        current = viewport.top_row(projection, viewport_rows=40)
        viewport.review_row(projection, current + bounded)
        assert viewport.projection_for(document, width=100) is projection

    assert viewport.projection_build_count == 1


def test_every_navigation_event_is_capped_to_one_rendered_viewport():
    assert bounded_scroll_rows(-100, viewport_rows=8) == -8
    assert bounded_scroll_rows(100, viewport_rows=8) == 8
    assert bounded_scroll_rows(-3, viewport_rows=8) == -3
    assert bounded_scroll_rows(3, viewport_rows=2) == 2
