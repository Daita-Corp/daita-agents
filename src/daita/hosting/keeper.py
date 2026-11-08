"""Keeps each source's catalog current: probe cheaply, read only what changed.

The keeper is one task owned by an open agent. It never blocks a caller, never
raises to one, and holds no admission lease. It knows nothing of ``EmbeddedAgent``:
it works through narrow callables, so it can be driven alone with a fake clock.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Protocol

from ..adapters.models import SourceProbe
from ..adapters.protocols import ChangeProbeBackend
from ..adapters.registry import SourceFreshness
from ..catalog.models import CatalogCommitSummary, CatalogSourceState

_LOGGER = logging.getLogger("daita.hosting.keeper")

PROBE_TIMEOUT_SECONDS = 5.0
# A run or status check probes only when the last probe is older than this.
USE_PROBE_AGE_SECONDS = 30.0
# A lookup miss is a stronger hint that the catalog is stale.
MISS_PROBE_AGE_SECONDS = 5.0
MAX_SLEEP_SECONDS = 30.0
MIN_SLEEP_SECONDS = 1.0
# An unchanged probe rewrites its state row no more often than this.
STATE_WRITE_SECONDS = 600.0
UNREACHABLE_AFTER_FAILURES = 2
READ_RESOURCES = 10_000
FILL_RESOURCES = 1_000
_CLOSE_WAIT_SECONDS = 2.0


class KeeperStore(Protocol):
    async def load_source_state(
        self, agent_id: str, source_id: str
    ) -> CatalogSourceState | None: ...

    async def save_source_state(self, state: CatalogSourceState) -> None: ...

    async def catalog_detail_counts(
        self, agent_id: str, source_id: str
    ) -> tuple[int, int]: ...

    async def take_unreadable_catalog_sources(
        self, agent_id: str
    ) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class KeptSource:
    source_id: str
    adapter_id: str


@dataclass(frozen=True, slots=True)
class KeeperRead:
    """One pass the keeper asks its owner to run and commit."""

    scopes: tuple[str, ...] = ()
    fill: bool = False
    max_resources: int = READ_RESOURCES
    probe_tokens: Mapping[str, str] | None = None
    probed_at: datetime | None = None

    @property
    def full(self) -> bool:
        return not self.scopes and not self.fill


@dataclass(frozen=True, slots=True)
class KeeperFacts:
    """What only the keeper's memory knows about one source."""

    updating: bool
    failures: int
    error_code: str | None
    last_probe: datetime | None

    @property
    def unreachable(self) -> bool:
        return self.failures >= UNREACHABLE_AFTER_FAILURES


@dataclass(slots=True)
class _Source:
    interval: float = 0.0
    failures: int = 0
    error_code: str | None = None
    next_due: datetime | None = None
    last_probe: datetime | None = None
    full_due: datetime | None = None
    fill_retry_after: datetime | None = None
    due_now: bool = False
    reset: bool = False
    full_now: bool = False
    # A probe saw a change whose read has not committed yet.
    pending: bool = False
    filling: bool = False
    counts_stale: bool = True
    described: int = 0


def _error_code(error: BaseException, default: str) -> str:
    code = getattr(error, "code", None)
    if isinstance(code, str) and 0 < len(code) <= 128:
        return code
    return "probe_timeout" if isinstance(error, TimeoutError) else default


class CatalogKeeper:
    def __init__(
        self,
        *,
        agent_id: str,
        store: KeeperStore,
        commit_lock: asyncio.Lock,
        clock: Callable[[], datetime],
        sources: Callable[[], Awaitable[Sequence[KeptSource]]],
        probe_backend: Callable[[str], ChangeProbeBackend | None],
        freshness: Callable[[str], SourceFreshness],
        read: Callable[[str, KeeperRead], Awaitable[CatalogCommitSummary | None]],
        fill_resources: int = FILL_RESOURCES,
    ) -> None:
        self._agent_id = agent_id
        self._store = store
        self._commit_lock = commit_lock
        self._clock = clock
        self._sources = sources
        self._probe_backend = probe_backend
        self._freshness = freshness
        self._read = read
        self._fill_resources = fill_resources
        self._mems: dict[str, _Source] = {}
        self._wake = asyncio.Event()
        self._work = asyncio.Lock()
        self._people = 0
        self._inflight: tuple[str, asyncio.Task[CatalogCommitSummary | None]] | None
        self._inflight = None
        self._task: asyncio.Task[None] | None = None
        self._checks: dict[str, asyncio.Task[None]] = {}
        self._closed = False

    # Lifetime ---------------------------------------------------------------

    def start(self) -> None:
        if self._task is None and not self._closed:
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def close(self) -> None:
        self._closed = True
        tasks = [t for t in (self._task, *self._checks.values()) if t is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=_CLOSE_WAIT_SECONDS)

    def wake(self) -> None:
        self._wake.set()

    async def _run(self) -> None:
        while True:
            try:
                busy = await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                _LOGGER.warning("catalog keeper pass failed: %s", type(error).__name__)
                busy = False
            if busy:
                await asyncio.sleep(0)
                continue
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), self._sleep_seconds())
            self._wake.clear()

    def _sleep_seconds(self) -> float:
        now = self._clock()
        wait = MAX_SLEEP_SECONDS
        for mem in self._mems.values():
            for due in (mem.next_due, mem.full_due):
                if due is not None:
                    wait = min(wait, (due - now).total_seconds())
        return max(wait, MIN_SLEEP_SECONDS)

    # Triggers ---------------------------------------------------------------

    def use(
        self, source_ids: Sequence[str], *, age: float = USE_PROBE_AGE_SECONDS
    ) -> None:
        """Probe these sources soon if the last probe is older than ``age``."""

        if self._closed:
            return
        now = self._clock()
        for source_id in source_ids:
            mem = self._mems.setdefault(source_id, _Source())
            mem.reset = True
            if mem.last_probe is None or (now - mem.last_probe).total_seconds() > age:
                mem.due_now = True
        self._wake.set()

    def miss(self, source_id: str) -> None:
        self.use((source_id,), age=MISS_PROBE_AGE_SECONDS)

    async def baseline(self, source_id: str, adapter_id: str) -> None:
        """Store the first probe of a just-read source so it is not read twice."""

        probe = await self.probe_once(source_id, adapter_id)
        if probe is None:
            return
        mem = self._mems.setdefault(source_id, _Source())
        mem.last_probe = probe.checked_at
        mem.next_due = self._clock() + timedelta(
            seconds=self._freshness(adapter_id).min_probe_seconds
        )
        try:
            await self._update_state(
                source_id,
                lambda s: (
                    s
                    if s.probe_tokens
                    else replace(
                        s,
                        probe_tokens=dict(probe.tokens),
                        last_probe_at=probe.checked_at,
                        last_error_code=None,
                    )
                ),
            )
        except Exception as error:
            _LOGGER.warning("catalog baseline failed: %s", type(error).__name__)

    def forget(self, source_id: str) -> None:
        self._mems.pop(source_id, None)

    async def check(self, source_id: str, *, wait: float = 0.0) -> None:
        """Probe now and read what changed; wait up to ``wait`` seconds for it.

        The work continues after a timeout, which is not an error.
        """

        if self._closed:
            return
        task = self._checks.get(source_id)
        if task is None or task.done():
            task = asyncio.get_running_loop().create_task(self._check(source_id))
            self._checks[source_id] = task
        if wait > 0:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.shield(task), wait)

    async def _check(self, source_id: str) -> None:
        try:
            source = next(
                (s for s in await self._sources() if s.source_id == source_id), None
            )
            if source is not None:
                async with self._work:
                    await self._service(source, force=True)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            _LOGGER.warning("catalog check failed: %s", type(error).__name__)

    @contextlib.asynccontextmanager
    async def person_work(self) -> AsyncIterator[None]:
        """A person's source action: stop a background read first and keep it out."""

        self._people += 1
        try:
            inflight = self._inflight
            if inflight is not None:
                inflight[1].cancel()
                await asyncio.wait({inflight[1]})
            yield
        finally:
            self._people -= 1
            self._wake.set()

    # Facts ------------------------------------------------------------------

    def facts(self, source_id: str) -> KeeperFacts:
        mem = self._mems.get(source_id)
        if mem is None:
            return KeeperFacts(False, 0, None, None)
        inflight = self._inflight
        return KeeperFacts(
            (inflight is not None and inflight[0] == source_id) or mem.filling,
            mem.failures,
            mem.error_code if mem.failures else None,
            mem.last_probe,
        )

    async def probe_once(self, source_id: str, adapter_id: str) -> SourceProbe | None:
        """A single best-effort probe for a read that commits its tokens."""

        backend = self._probe_backend(adapter_id)
        if backend is None:
            return None
        try:
            return await asyncio.wait_for(
                backend.probe(
                    agent_id=self._agent_id, source_id=source_id, cursor=None
                ),
                PROBE_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return None

    # One pass ---------------------------------------------------------------

    async def tick(self) -> bool:
        """Do the work that is due; True when more is ready right now."""

        if self._people or self._closed:
            return False
        for source_id in await self._store.take_unreadable_catalog_sources(
            self._agent_id
        ):
            self._mems.setdefault(source_id, _Source()).full_now = True
        active = await self._sources()
        for gone in set(self._mems) - {s.source_id for s in active}:
            del self._mems[gone]
        busy = False
        for source in active:
            if self._people or self._closed:
                break
            try:
                async with self._work:
                    busy = await self._service(source) or busy
            except asyncio.CancelledError:
                raise
            except Exception as error:
                _LOGGER.warning("catalog keeper failed: %s", type(error).__name__)
        return busy

    async def _service(self, source: KeptSource, *, force: bool = False) -> bool:
        source_id = source.source_id
        mem = self._mems.setdefault(source_id, _Source())
        fresh = self._freshness(source.adapter_id)
        if not mem.interval or mem.reset:
            mem.interval, mem.reset = fresh.min_probe_seconds, False
        state = await self._store.load_source_state(self._agent_id, source_id)
        if state is not None and not state.keep_current:
            mem.due_now = mem.full_now = mem.filling = False
            return False
        now = self._clock()
        if mem.next_due is None:
            last = None if state is None else state.last_probe_at
            mem.next_due = (
                now
                if last is None
                else last + timedelta(seconds=fresh.min_probe_seconds)
            )
        if mem.full_now:
            mem.full_now = False
            await self._read_pass(source, KeeperRead())
            return False
        backend = self._probe_backend(source.adapter_id)
        if backend is not None and (force or mem.due_now or now >= mem.next_due):
            mem.due_now = False
            if await self._probe(source, state, backend, fresh, now):
                return False
        if mem.failures and now < mem.next_due:
            return False
        if not mem.pending and await self._fill(source, mem, now):
            return True
        seconds = fresh.full_check_seconds
        if seconds is not None and not mem.pending:
            last_full = None if state is None else state.last_full_check_at
            mem.full_due = (
                None if last_full is None else last_full + timedelta(seconds=seconds)
            )
            if mem.full_due is None or mem.full_due <= now:
                await self._full_check(source, backend is not None)
        return False

    async def _probe(
        self,
        source: KeptSource,
        state: CatalogSourceState | None,
        backend: ChangeProbeBackend,
        fresh: SourceFreshness,
        now: datetime,
    ) -> bool:
        """Probe once; True when a read ran or the probe failed."""

        source_id = source.source_id
        mem = self._mems[source_id]
        try:
            probe = await asyncio.wait_for(
                backend.probe(
                    agent_id=self._agent_id,
                    source_id=source_id,
                    cursor=None if state is None else state.probe_cursor,
                ),
                PROBE_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            code = _error_code(error, "probe_failed")
            self._fail(source, mem, code, fresh, now, error)
            await self._update_state(
                source_id,
                lambda s: (
                    s if s.last_error_code == code else replace(s, last_error_code=code)
                ),
            )
            return True
        mem.failures, mem.error_code, mem.last_probe = 0, None, probe.checked_at
        old = dict(state.probe_tokens) if state is not None else {}
        new = dict(probe.tokens)
        if old == new:
            await self._update_state(source_id, lambda s: _after_unchanged(s, probe))
        else:
            changed = sorted(
                k for k in old.keys() | new.keys() if old.get(k) != new.get(k)
            )
            whole = (
                bool(old.keys() - new.keys())
                or len(new) <= 1
                or new.keys() <= set(changed)
            )
            mem.interval = fresh.min_probe_seconds
            mem.next_due = now + timedelta(seconds=mem.interval)
            mem.pending = True
            await self._read_pass(
                source,
                KeeperRead(
                    scopes=() if whole else tuple(changed),
                    probe_tokens=new,
                    probed_at=probe.checked_at,
                ),
            )
            return True
        delay = mem.interval
        mem.next_due = now + timedelta(seconds=delay)
        mem.interval = min(delay * 2, fresh.max_probe_seconds)
        return False

    async def _fill(self, source: KeptSource, mem: _Source, now: datetime) -> bool:
        """Describe more of a catalog that lacks detail; True while more remains."""

        if mem.counts_stale:
            resources, mem.described = await self._store.catalog_detail_counts(
                self._agent_id, source.source_id
            )
            mem.filling = mem.described < resources
            mem.counts_stale = False
        if not mem.filling or (
            mem.fill_retry_after is not None and now < mem.fill_retry_after
        ):
            return False
        before = mem.described
        summary = await self._read_pass(
            source, KeeperRead(fill=True, max_resources=self._fill_resources)
        )
        if summary is None:
            # Cancelled by a person or failed (the backoff covers that): no
            # progress verdict, so the fill resumes as soon as it may.
            return False
        resources, mem.described = await self._store.catalog_detail_counts(
            self._agent_id, source.source_id
        )
        mem.counts_stale = False
        mem.filling = mem.described < resources and mem.failures == 0
        if mem.described <= before:
            # Nothing more can be described now; try again after a long wait.
            mem.filling = False
            mem.fill_retry_after = now + timedelta(
                seconds=self._freshness(source.adapter_id).max_probe_seconds
            )
        return mem.filling

    async def _full_check(self, source: KeptSource, probed: bool) -> None:
        summary = await self._read_pass(source, KeeperRead())
        if summary is None or not probed:
            return
        missed = summary.added + summary.changed + summary.removed
        await self._update_state(
            source.source_id, lambda s: replace(s, last_missed=missed)
        )
        if missed:
            kinds = [
                name
                for name, count in (
                    ("added", summary.added),
                    ("changed", summary.changed),
                    ("removed", summary.removed),
                )
                if count
            ]
            _LOGGER.warning(
                "catalog check found changes the probe missed: source type %s, %s",
                source.adapter_id,
                ", ".join(kinds),
            )

    async def _read_pass(
        self, source: KeptSource, plan: KeeperRead
    ) -> CatalogCommitSummary | None:
        mem = self._mems[source.source_id]
        mem.counts_stale = True
        now = self._clock()
        fresh = self._freshness(source.adapter_id)
        try:
            summary = await self._run_read(source.source_id, plan)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._fail(
                source, mem, _error_code(error, "source_read_failed"), fresh, now, error
            )
            return None
        if summary is not None:
            mem.failures, mem.error_code, mem.pending = 0, None, False
            if plan.probe_tokens is not None and plan.probed_at is not None:
                mem.last_probe = plan.probed_at
        return summary

    async def _run_read(
        self, source_id: str, plan: KeeperRead
    ) -> CatalogCommitSummary | None:
        """Run one read as a task a person's action can cancel; None when cancelled."""

        if self._people:
            return None
        task = asyncio.ensure_future(self._read(source_id, plan))
        # A read that outlives a cancelled waiter must not leave its error unretrieved.
        task.add_done_callback(lambda t: t.cancelled() or t.exception())
        self._inflight = (source_id, task)
        try:
            await asyncio.wait({task})
        finally:
            if not task.done():
                task.cancel()
                await asyncio.wait({task})
            self._inflight = None
        return None if task.cancelled() else task.result()

    def _fail(
        self,
        source: KeptSource,
        mem: _Source,
        code: str,
        fresh: SourceFreshness,
        now: datetime,
        error: BaseException,
    ) -> None:
        mem.failures += 1
        mem.error_code = code
        delay = mem.interval
        mem.next_due = now + timedelta(seconds=delay)
        mem.interval = min(delay * 2, fresh.max_probe_seconds)
        # Type name and code only: error text can hold secrets.
        _LOGGER.log(
            (
                logging.WARNING
                if mem.failures <= UNREACHABLE_AFTER_FAILURES
                else logging.DEBUG
            ),
            "catalog keeper: %s source failed (%s, %s), %d in a row",
            source.adapter_id,
            type(error).__name__,
            code,
            mem.failures,
        )

    async def _update_state(
        self,
        source_id: str,
        change: Callable[[CatalogSourceState], CatalogSourceState],
    ) -> None:
        async with self._commit_lock:
            state = await self._store.load_source_state(self._agent_id, source_id)
            if state is None:
                return
            updated = change(state)
            if updated != state:
                await self._store.save_source_state(updated)


def _after_unchanged(
    state: CatalogSourceState, probe: SourceProbe
) -> CatalogSourceState:
    stale = (
        state.last_probe_at is None
        or (probe.checked_at - state.last_probe_at).total_seconds()
        >= STATE_WRITE_SECONDS
    )
    if not stale and state.last_error_code is None:
        return state
    return replace(state, last_probe_at=probe.checked_at, last_error_code=None)


__all__ = [
    "CatalogKeeper",
    "KeeperFacts",
    "KeeperRead",
    "KeeperStore",
    "KeptSource",
]
