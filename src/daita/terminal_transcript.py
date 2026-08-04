"""Pure, disposable transcript identity and terminal-cell projection.

This module owns presentation-only transcript coordinates. It does not render the
TUI, persist state, or contribute text to a model request. The fullscreen terminal
renderer supplies already-sanitized canonical block text and can discard the whole
document when that application exits.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from enum import Enum
import unicodedata


@dataclass(frozen=True, order=True, slots=True)
class PresentationBlockId:
    """One monotonically allocated block identity within a document lifetime."""

    value: int

    def __post_init__(self) -> None:
        if not isinstance(self.value, int) or isinstance(self.value, bool):
            raise TypeError("presentation block ID must be an integer")
        if self.value < 1:
            raise ValueError("presentation block ID must be positive")


@dataclass(frozen=True, slots=True)
class TranscriptBlock:
    """One current canonical selectable-text snapshot."""

    id: PresentationBlockId
    text: str
    revision: int


@dataclass(frozen=True, slots=True)
class SemanticPosition:
    """A logical code-point offset in one version of a presentation block."""

    block_id: PresentationBlockId
    offset: int
    revision: int


@dataclass(frozen=True, slots=True)
class SemanticRange:
    """A normalized pair of boundaries plus the exact selected text."""

    start: SemanticPosition
    end: SemanticPosition
    text: str


@dataclass(frozen=True, slots=True)
class SemanticViewportAnchor:
    """The exact semantic content identifying a reviewed top-row position."""

    position: SemanticPosition
    text: str


class TranscriptFollowState(str, Enum):
    """Whether the disposable transcript viewport follows or reviews output."""

    FOLLOWING = "following"
    REVIEWING = "reviewing"


@dataclass(frozen=True, slots=True)
class RenderedCoordinate:
    """A zero-based rendered row and terminal display cell."""

    row: int
    cell: int


@dataclass(frozen=True, slots=True)
class ProjectionStats:
    """Deterministic work/result counters for one complete projection."""

    source_codepoints: int
    grapheme_clusters: int
    segments: int
    rows: int


class InteractionOwner(str, Enum):
    """Phase 0's dormant precedence contract for later interaction slices."""

    APPROVAL = "approval"
    SEARCH_OR_TRANSIENT = "search_or_transient"
    TRANSCRIPT_SELECTION = "transcript_selection"
    COMPLETION_MENU = "completion_menu"
    TRANSCRIPT_FOCUS = "transcript_focus"
    ACTIVE_RUN_CANCELLATION = "active_run_cancellation"
    COMPOSER_INPUT = "composer_input"


INTERACTION_PRECEDENCE = (
    InteractionOwner.APPROVAL,
    InteractionOwner.SEARCH_OR_TRANSIENT,
    InteractionOwner.TRANSCRIPT_SELECTION,
    InteractionOwner.COMPLETION_MENU,
    InteractionOwner.TRANSCRIPT_FOCUS,
    InteractionOwner.ACTIVE_RUN_CANCELLATION,
    InteractionOwner.COMPOSER_INPUT,
)


def interaction_owner(active: frozenset[InteractionOwner]) -> InteractionOwner | None:
    """Return the first active owner without activating any future interaction."""

    if not isinstance(active, frozenset) or any(
        not isinstance(owner, InteractionOwner) for owner in active
    ):
        raise TypeError("active interactions must be a frozenset of InteractionOwner")
    return next((owner for owner in INTERACTION_PRECEDENCE if owner in active), None)


@dataclass(frozen=True, slots=True)
class _RevisionTransition:
    old_length: int
    new_length: int
    common_prefix: int
    common_suffix: int

    def content_offset(self, offset: int) -> int | None:
        """Reconcile an anchor only when the character it names survives."""

        if self.old_length == self.new_length == self.common_prefix:
            return offset
        if offset < self.common_prefix:
            return offset
        suffix_start = self.old_length - self.common_suffix
        if self.common_suffix and suffix_start <= offset < self.old_length:
            return self.new_length - (self.old_length - offset)
        if offset == self.old_length:
            if self.common_prefix == self.old_length:
                return self.old_length
            if self.common_suffix:
                return self.new_length
        return None

    def boundary_offset(self, offset: int) -> int | None:
        """Reconcile a range boundary; selected-text equality is checked later."""

        if offset <= self.common_prefix:
            return offset
        suffix_start = self.old_length - self.common_suffix
        if offset >= suffix_start:
            return self.new_length - (self.old_length - offset)
        return None


@dataclass(slots=True)
class _BlockState:
    id: PresentationBlockId
    text: str
    revision: int = 0
    transitions: dict[int, _RevisionTransition] = field(default_factory=dict)

    def snapshot(self) -> TranscriptBlock:
        return TranscriptBlock(self.id, self.text, self.revision)


@dataclass(frozen=True, slots=True)
class _ProjectedSegment:
    block_id: PresentationBlockId
    revision: int
    start_offset: int
    end_offset: int
    row: int
    start_cell: int
    end_cell: int
    linear: bool

    def coordinate_for_offset(self, offset: int) -> RenderedCoordinate:
        if self.linear:
            cell = self.start_cell + max(
                0,
                min(offset - self.start_offset, self.end_cell - self.start_cell),
            )
            return RenderedCoordinate(self.row, cell)
        if offset >= self.end_offset:
            return RenderedCoordinate(self.row, self.end_cell)
        return RenderedCoordinate(self.row, self.start_cell)

    def offset_for_cell(self, cell: int) -> int:
        if self.linear:
            return self.start_offset + max(
                0,
                min(cell - self.start_cell, self.end_offset - self.start_offset),
            )
        return self.end_offset if cell >= self.end_cell else self.start_offset


@dataclass(slots=True)
class _BlockProjection:
    revision: int
    length: int
    segments: list[_ProjectedSegment] = field(default_factory=list)
    starts: list[int] = field(default_factory=list)
    exact: dict[int, RenderedCoordinate] = field(default_factory=dict)

    def coordinate(self, offset: int) -> RenderedCoordinate | None:
        exact = self.exact.get(offset)
        if exact is not None:
            return exact
        index = bisect_right(self.starts, offset) - 1
        if index < 0:
            return None
        segment = self.segments[index]
        if offset > segment.end_offset:
            return None
        return segment.coordinate_for_offset(offset)


class TranscriptProjection:
    """One immutable-width view from semantic offsets to rendered cells."""

    def __init__(
        self,
        document: TranscriptDocument,
        *,
        width: int,
        blocks: dict[PresentationBlockId, _BlockProjection],
        rows: dict[int, list[_ProjectedSegment]],
        row_defaults: dict[int, SemanticPosition],
        stats: ProjectionStats,
    ) -> None:
        self._document = document
        self.width = width
        self._blocks = blocks
        self._rows = rows
        self._row_defaults = row_defaults
        self.stats = stats

    @property
    def row_count(self) -> int:
        return self.stats.rows

    def to_rendered(self, position: SemanticPosition) -> RenderedCoordinate | None:
        current = self._document.reconcile_position(position)
        if current is None:
            return None
        block = self._blocks.get(current.block_id)
        if block is None or current.revision != block.revision:
            return None
        return block.coordinate(current.offset)

    def to_semantic(self, row: int, cell: int) -> SemanticPosition | None:
        if not isinstance(row, int) or isinstance(row, bool):
            raise TypeError("rendered row must be an integer")
        if not isinstance(cell, int) or isinstance(cell, bool):
            raise TypeError("rendered cell must be an integer")
        if row < 0 or cell < 0 or row >= self.row_count:
            return None
        segments = self._rows.get(row, ())
        if not segments:
            return self._row_defaults.get(row)
        if cell <= segments[0].start_cell:
            segment = segments[0]
            offset = segment.start_offset
        else:
            segment = segments[-1]
            offset = segment.end_offset
            for index, candidate in enumerate(segments):
                next_start = (
                    segments[index + 1].start_cell
                    if index + 1 < len(segments)
                    else candidate.end_cell
                )
                if cell < candidate.end_cell or cell < next_start:
                    segment = candidate
                    offset = candidate.offset_for_cell(cell)
                    break
        return SemanticPosition(segment.block_id, offset, segment.revision)

    def anchor_for_row(self, row: int) -> SemanticViewportAnchor | None:
        position = self.to_semantic(row, 0)
        return None if position is None else self._document.make_anchor(position)

    def resolve_anchor(
        self,
        anchor: SemanticViewportAnchor,
    ) -> RenderedCoordinate | None:
        current = self._document.reconcile_anchor(anchor)
        return None if current is None else self.to_rendered(current.position)


class TranscriptDocument:
    """A process-local transcript with monotonic identity and explicit edits."""

    def __init__(self) -> None:
        self._next_id = 1
        self._generation = 0
        self._order: list[PresentationBlockId] = []
        self._blocks: dict[PresentationBlockId, _BlockState] = {}
        self._retired: set[PresentationBlockId] = set()

    @property
    def blocks(self) -> tuple[TranscriptBlock, ...]:
        return tuple(self._blocks[block_id].snapshot() for block_id in self._order)

    @property
    def presentation_ids(self) -> tuple[PresentationBlockId, ...]:
        return tuple(self._order)

    @property
    def generation(self) -> int:
        """Return the process-local mutation generation used by projection caches."""

        return self._generation

    def contains(self, block_id: PresentationBlockId) -> bool:
        return block_id in self._blocks

    def append(self, text: str) -> TranscriptBlock:
        _require_text(text)
        block_id = PresentationBlockId(self._next_id)
        self._next_id += 1
        block = _BlockState(block_id, text)
        self._blocks[block_id] = block
        self._order.append(block_id)
        self._generation += 1
        return block.snapshot()

    def replace(self, block_id: PresentationBlockId, text: str) -> TranscriptBlock:
        _require_text(text)
        block = self._require_block(block_id)
        if block.text == text:
            return block.snapshot()
        prefix = _common_prefix(block.text, text)
        suffix = _common_suffix(block.text, text, prefix)
        block.transitions[block.revision] = _RevisionTransition(
            old_length=len(block.text),
            new_length=len(text),
            common_prefix=prefix,
            common_suffix=suffix,
        )
        block.text = text
        block.revision += 1
        self._generation += 1
        return block.snapshot()

    def remove(self, block_id: PresentationBlockId) -> None:
        self._require_block(block_id)
        del self._blocks[block_id]
        self._order.remove(block_id)
        self._retired.add(block_id)
        self._generation += 1

    def reorder(self, block_ids: tuple[PresentationBlockId, ...]) -> None:
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("transcript block order cannot repeat an ID")
        if set(block_ids) != set(self._blocks):
            raise ValueError("transcript block order must contain every current ID")
        if tuple(self._order) == block_ids:
            return
        self._order = list(block_ids)
        self._generation += 1

    def text(self, block_id: PresentationBlockId) -> str:
        return self._require_block(block_id).text

    def position(self, block_id: PresentationBlockId, offset: int) -> SemanticPosition:
        block = self._require_block(block_id)
        _require_offset(offset, len(block.text))
        return SemanticPosition(block_id, offset, block.revision)

    def reconcile_position(
        self,
        position: SemanticPosition,
    ) -> SemanticPosition | None:
        return self._reconcile_position(position, boundary=False)

    def normalize_range(
        self,
        first: SemanticPosition,
        second: SemanticPosition,
    ) -> SemanticRange:
        start = self._reconcile_position(first, boundary=True)
        end = self._reconcile_position(second, boundary=True)
        if start is None or end is None:
            raise ValueError("semantic range contains an invalid position")
        indexes = {block_id: index for index, block_id in enumerate(self._order)}
        if (indexes[start.block_id], start.offset) > (
            indexes[end.block_id],
            end.offset,
        ):
            start, end = end, start
        return SemanticRange(start, end, self._range_text(start, end))

    def reconcile_range(self, value: SemanticRange) -> SemanticRange | None:
        if not isinstance(value, SemanticRange):
            raise TypeError("semantic range must be SemanticRange")
        start = self._reconcile_position(value.start, boundary=True)
        end = self._reconcile_position(value.end, boundary=True)
        if start is None or end is None:
            return None
        try:
            current = self.normalize_range(start, end)
        except ValueError:
            return None
        return current if current.text == value.text else None

    def make_anchor(self, position: SemanticPosition) -> SemanticViewportAnchor:
        current = self.reconcile_position(position)
        if current is None:
            raise ValueError("viewport anchor requires surviving semantic content")
        text = self._blocks[current.block_id].text
        end = _next_grapheme_end(text, current.offset)
        return SemanticViewportAnchor(current, text[current.offset : end])

    def reconcile_anchor(
        self,
        anchor: SemanticViewportAnchor,
    ) -> SemanticViewportAnchor | None:
        if not isinstance(anchor, SemanticViewportAnchor):
            raise TypeError("viewport anchor must be SemanticViewportAnchor")
        current = self.reconcile_position(anchor.position)
        if current is None:
            return None
        text = self._blocks[current.block_id].text
        end = _next_grapheme_end(text, current.offset)
        return (
            SemanticViewportAnchor(current, anchor.text)
            if text[current.offset : end] == anchor.text
            else None
        )

    def project(self, width: int, *, tab_size: int = 8) -> TranscriptProjection:
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            raise ValueError("projection width must be a positive integer")
        if not isinstance(tab_size, int) or isinstance(tab_size, bool) or tab_size < 1:
            raise ValueError("tab size must be a positive integer")

        block_projections: dict[PresentationBlockId, _BlockProjection] = {}
        rows: dict[int, list[_ProjectedSegment]] = {}
        row_defaults: dict[int, SemanticPosition] = {}
        row = 0
        cell = 0
        source_codepoints = 0
        grapheme_clusters = 0
        segment_count = 0

        def next_row(default: SemanticPosition) -> None:
            nonlocal row, cell
            row += 1
            cell = 0
            row_defaults[row] = default

        def add_segment(
            projected: _BlockProjection,
            block: _BlockState,
            start: int,
            end: int,
            start_cell: int,
            end_cell: int,
            *,
            linear: bool,
        ) -> None:
            nonlocal segment_count
            segment = _ProjectedSegment(
                block.id,
                block.revision,
                start,
                end,
                row,
                start_cell,
                end_cell,
                linear,
            )
            projected.segments.append(segment)
            projected.starts.append(start)
            rows.setdefault(row, []).append(segment)
            segment_count += 1

        for block_index, block_id in enumerate(self._order):
            block = self._blocks[block_id]
            source_codepoints += len(block.text)
            if block_index and cell:
                prior = SemanticPosition(block.id, 0, block.revision)
                next_row(prior)
            projected = _BlockProjection(block.revision, len(block.text))
            block_projections[block.id] = projected
            start_position = SemanticPosition(block.id, 0, block.revision)
            projected.exact[0] = RenderedCoordinate(row, cell)
            row_defaults.setdefault(row, start_position)
            offset = 0
            while offset < len(block.text):
                character = block.text[offset]
                if character == "\n":
                    projected.exact[offset] = RenderedCoordinate(row, cell)
                    after = SemanticPosition(block.id, offset + 1, block.revision)
                    next_row(after)
                    projected.exact[offset + 1] = RenderedCoordinate(row, 0)
                    offset += 1
                    grapheme_clusters += 1
                    continue
                if character == "\t":
                    projected.exact[offset] = RenderedCoordinate(row, cell)
                    advance = tab_size - (cell % tab_size)
                    remaining_advance = advance
                    while remaining_advance:
                        if cell == width:
                            next_row(SemanticPosition(block.id, offset, block.revision))
                        taken = min(width - cell, remaining_advance)
                        add_segment(
                            projected,
                            block,
                            offset,
                            offset + 1,
                            cell,
                            cell + taken,
                            linear=False,
                        )
                        cell += taken
                        remaining_advance -= taken
                    projected.exact[offset + 1] = RenderedCoordinate(row, cell)
                    offset += 1
                    grapheme_clusters += 1
                    continue

                ascii_end = offset
                while (
                    ascii_end < len(block.text) and " " <= block.text[ascii_end] <= "~"
                ):
                    ascii_end += 1
                if ascii_end > offset:
                    grapheme_clusters += ascii_end - offset
                    while offset < ascii_end:
                        if cell == width:
                            next_row(SemanticPosition(block.id, offset, block.revision))
                            projected.exact[offset] = RenderedCoordinate(row, 0)
                        taken = min(width - cell, ascii_end - offset)
                        add_segment(
                            projected,
                            block,
                            offset,
                            offset + taken,
                            cell,
                            cell + taken,
                            linear=True,
                        )
                        offset += taken
                        cell += taken
                    continue

                end = _next_grapheme_end(block.text, offset)
                cluster = block.text[offset:end]
                cluster_width = _cluster_cell_width(cluster)
                if cluster_width and cell and cell + cluster_width > width:
                    next_row(SemanticPosition(block.id, offset, block.revision))
                    projected.exact[offset] = RenderedCoordinate(row, 0)
                rendered_width = min(width, max(0, cluster_width))
                add_segment(
                    projected,
                    block,
                    offset,
                    end,
                    cell,
                    cell + rendered_width,
                    linear=False,
                )
                cell += rendered_width
                offset = end
                grapheme_clusters += 1
            projected.exact[len(block.text)] = RenderedCoordinate(row, cell)

        stats = ProjectionStats(
            source_codepoints=source_codepoints,
            grapheme_clusters=grapheme_clusters,
            segments=segment_count,
            rows=row + 1,
        )
        return TranscriptProjection(
            self,
            width=width,
            blocks=block_projections,
            rows=rows,
            row_defaults=row_defaults,
            stats=stats,
        )

    def _require_block(self, block_id: PresentationBlockId) -> _BlockState:
        if not isinstance(block_id, PresentationBlockId):
            raise TypeError("block ID must be PresentationBlockId")
        block = self._blocks.get(block_id)
        if block is None:
            reason = "retired" if block_id in self._retired else "unknown"
            raise KeyError(f"{reason} presentation block ID: {block_id.value}")
        return block

    def _reconcile_position(
        self,
        position: SemanticPosition,
        *,
        boundary: bool,
    ) -> SemanticPosition | None:
        if not isinstance(position, SemanticPosition):
            raise TypeError("position must be SemanticPosition")
        block = self._blocks.get(position.block_id)
        if block is None or position.revision < 0 or position.revision > block.revision:
            return None
        offset = position.offset
        revision = position.revision
        while revision < block.revision:
            transition = block.transitions.get(revision)
            if transition is None or not 0 <= offset <= transition.old_length:
                return None
            mapped_offset = (
                transition.boundary_offset(offset)
                if boundary
                else transition.content_offset(offset)
            )
            if mapped_offset is None:
                return None
            offset = mapped_offset
            revision += 1
        if not 0 <= offset <= len(block.text):
            return None
        return SemanticPosition(block.id, offset, block.revision)

    def _range_text(
        self,
        start: SemanticPosition,
        end: SemanticPosition,
    ) -> str:
        start_index = self._order.index(start.block_id)
        end_index = self._order.index(end.block_id)
        if start_index == end_index:
            return self._blocks[start.block_id].text[start.offset : end.offset]
        pieces = [self._blocks[start.block_id].text[start.offset :]]
        pieces.extend(
            self._blocks[block_id].text
            for block_id in self._order[start_index + 1 : end_index]
        )
        pieces.append(self._blocks[end.block_id].text[: end.offset])
        return "\n".join(pieces)


_MAX_UNSEEN_ITEMS = 9_999


class TranscriptViewport:
    """Pure semantic viewport truth for one disposable transcript document."""

    def __init__(self) -> None:
        self.state = TranscriptFollowState.FOLLOWING
        self.anchor: SemanticViewportAnchor | None = None
        self.unseen_items = 0
        self._projection: TranscriptProjection | None = None
        self._projection_document: TranscriptDocument | None = None
        self._projection_generation = -1
        self._projection_width = 0
        self._projection_build_count = 0

    @property
    def projection(self) -> TranscriptProjection | None:
        return self._projection

    @property
    def projection_build_count(self) -> int:
        """Expose deterministic complete-projection work for performance tests."""

        return self._projection_build_count

    def projection_for(
        self,
        document: TranscriptDocument,
        *,
        width: int,
    ) -> TranscriptProjection:
        """Return one cached projection until text, order, or width changes."""

        if not isinstance(document, TranscriptDocument):
            raise TypeError("viewport document must be TranscriptDocument")
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            raise ValueError("viewport projection width must be a positive integer")
        if (
            self._projection is None
            or self._projection_document is not document
            or self._projection_generation != document.generation
            or self._projection_width != width
        ):
            self._projection = document.project(width)
            self._projection_document = document
            self._projection_generation = document.generation
            self._projection_width = width
            self._projection_build_count += 1
        return self._projection

    def record_appended(self, count: int = 1) -> None:
        """Count genuinely appended blocks only while reviewing."""

        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("appended transcript block count must be non-negative")
        if self.state is TranscriptFollowState.REVIEWING:
            self.unseen_items = min(
                _MAX_UNSEEN_ITEMS,
                self.unseen_items + count,
            )

    def follow_latest(self) -> None:
        """Attach to the newest content and clear review-only notification state."""

        self.state = TranscriptFollowState.FOLLOWING
        self.anchor = None
        self.unseen_items = 0

    def review_start(self, projection: TranscriptProjection) -> None:
        """Enter Reviewing at the first semantic transcript row."""

        self._review_at(projection, 0)

    def review_row(self, projection: TranscriptProjection, row: int) -> None:
        """Enter Reviewing at one transiently resolved projection row."""

        if not isinstance(projection, TranscriptProjection):
            raise TypeError("viewport review requires TranscriptProjection")
        if not isinstance(row, int) or isinstance(row, bool):
            raise TypeError("viewport review row must be an integer")
        self._review_at(projection, row)

    def review_position(
        self,
        document: TranscriptDocument,
        position: SemanticPosition,
    ) -> None:
        """Enter Reviewing at one renderer-resolved semantic position."""

        if not isinstance(document, TranscriptDocument):
            raise TypeError("viewport review requires TranscriptDocument")
        current = document.reconcile_position(position)
        if current is None:
            raise ValueError(
                "viewport review position must reference surviving content"
            )
        self.state = TranscriptFollowState.REVIEWING
        self.anchor = document.make_anchor(current)

    def top_row(
        self,
        projection: TranscriptProjection,
        *,
        viewport_rows: int,
    ) -> int:
        """Resolve semantic viewport truth to a transient projection row."""

        if not isinstance(projection, TranscriptProjection):
            raise TypeError("viewport resolution requires TranscriptProjection")
        if (
            not isinstance(viewport_rows, int)
            or isinstance(viewport_rows, bool)
            or viewport_rows < 1
        ):
            raise ValueError("viewport height must be a positive integer")
        latest = max(0, projection.row_count - viewport_rows)
        if self.state is TranscriptFollowState.FOLLOWING:
            return latest
        resolved = (
            None if self.anchor is None else projection.resolve_anchor(self.anchor)
        )
        if resolved is None:
            self.anchor = self._fallback_anchor(projection)
            resolved = (
                None if self.anchor is None else projection.resolve_anchor(self.anchor)
            )
        return min(latest, max(0, 0 if resolved is None else resolved.row))

    def _review_at(self, projection: TranscriptProjection, row: int) -> None:
        last_row = max(0, projection.row_count - 1)
        anchor = projection.anchor_for_row(min(last_row, max(0, row)))
        self.state = TranscriptFollowState.REVIEWING
        if anchor is not None:
            self.anchor = anchor

    def _fallback_anchor(
        self,
        projection: TranscriptProjection,
    ) -> SemanticViewportAnchor | None:
        document = projection._document
        blocks = document.blocks
        if not blocks:
            return None
        prior_id = None if self.anchor is None else self.anchor.position.block_id
        if prior_id is not None and document.contains(prior_id):
            block_id = prior_id
        else:
            block_id = next(
                (
                    block.id
                    for block in blocks
                    if prior_id is None or block.id.value > prior_id.value
                ),
                blocks[-1].id,
            )
        return document.make_anchor(document.position(block_id, 0))


def bounded_scroll_rows(rows: int, *, viewport_rows: int) -> int:
    """Cap one navigation event to one rendered viewport."""

    if not isinstance(rows, int) or isinstance(rows, bool):
        raise TypeError("viewport row movement must be an integer")
    if (
        not isinstance(viewport_rows, int)
        or isinstance(viewport_rows, bool)
        or viewport_rows < 1
    ):
        raise ValueError("viewport height must be a positive integer")
    direction = 1 if rows > 0 else -1 if rows < 0 else 0
    return direction * min(abs(rows), viewport_rows)


def _require_text(text: str) -> None:
    if not isinstance(text, str):
        raise TypeError("canonical selectable text must be a string")


def _require_offset(offset: int, length: int) -> None:
    if not isinstance(offset, int) or isinstance(offset, bool):
        raise TypeError("semantic offset must be an integer")
    if not 0 <= offset <= length:
        raise ValueError("semantic offset is outside the block text")


def _common_prefix(first: str, second: str) -> int:
    maximum = min(len(first), len(second))
    offset = 0
    while offset < maximum and first[offset] == second[offset]:
        offset += 1
    return offset


def _common_suffix(first: str, second: str, prefix: int) -> int:
    maximum = min(len(first), len(second)) - prefix
    offset = 0
    while offset < maximum and first[-offset - 1] == second[-offset - 1]:
        offset += 1
    return offset


def _next_grapheme_end(text: str, start: int) -> int:
    if start >= len(text):
        return start
    end = start + 1
    if _is_regional_indicator(text[start]) and end < len(text):
        if _is_regional_indicator(text[end]):
            end += 1
    while end < len(text) and _extends_grapheme(text[end]):
        end += 1
    while end < len(text) and text[end] == "\u200d" and end + 1 < len(text):
        end += 2
        while end < len(text) and _extends_grapheme(text[end]):
            end += 1
    return end


def _extends_grapheme(character: str) -> bool:
    codepoint = ord(character)
    return (
        bool(unicodedata.combining(character))
        or 0xFE00 <= codepoint <= 0xFE0F
        or 0xE0100 <= codepoint <= 0xE01EF
        or 0x1F3FB <= codepoint <= 0x1F3FF
        or 0xE0020 <= codepoint <= 0xE007F
    )


def _is_regional_indicator(character: str) -> bool:
    return 0x1F1E6 <= ord(character) <= 0x1F1FF


def _cluster_cell_width(cluster: str) -> int:
    try:
        from wcwidth import wcswidth

        measured = wcswidth(cluster)
    except ImportError:
        measured = -1
    if measured >= 0:
        return measured
    widths = [
        (
            0
            if unicodedata.combining(character) or character == "\u200d"
            else (2 if unicodedata.east_asian_width(character) in {"F", "W"} else 1)
        )
        for character in cluster
    ]
    return max(widths, default=0)


__all__ = [
    "INTERACTION_PRECEDENCE",
    "InteractionOwner",
    "PresentationBlockId",
    "ProjectionStats",
    "RenderedCoordinate",
    "SemanticPosition",
    "SemanticRange",
    "SemanticViewportAnchor",
    "TranscriptBlock",
    "TranscriptDocument",
    "TranscriptFollowState",
    "TranscriptProjection",
    "TranscriptViewport",
    "bounded_scroll_rows",
    "interaction_owner",
]
