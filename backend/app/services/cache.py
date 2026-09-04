"""In-process TTL cache.

Weather data has a natural refresh cadence — current conditions update every
10-15 minutes, forecasts hourly, and a place's coordinates essentially never.
Serving a repeated question from memory protects the upstream source, keeps us
inside its fair-use policy and cuts latency to nothing.

This is deliberately an in-process cache, not Redis. It is correct for a single
instance, needs no infrastructure, and sits behind an interface narrow enough
that swapping in a shared cache later touches only this file. Adding Redis now
would be infrastructure without a problem to solve.

Concurrency: entries are guarded by a lock, and a single-flight guard per key
ensures that N simultaneous requests for the same cold key produce one upstream
call rather than N.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, TypeVar

from app.config.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass(slots=True)
class _Entry(Generic[T]):
    value: T
    expires_at: float


class TTLCache(Generic[T]):
    """A bounded, time-limited key/value store.

    Args:
        ttl_seconds: lifetime of an entry. ``0`` disables caching entirely.
        max_entries: cap on retained keys; the oldest expiring entry is evicted
            when the cap is reached.
    """

    def __init__(self, *, ttl_seconds: float, max_entries: int = 1024) -> None:
        self._ttl = max(0.0, ttl_seconds)
        self._max_entries = max(1, max_entries)
        self._entries: dict[str, _Entry[T]] = {}
        self._lock = asyncio.Lock()
        self._inflight: dict[str, asyncio.Future[T]] = {}

    @property
    def enabled(self) -> bool:
        return self._ttl > 0

    async def get(self, key: str) -> T | None:
        """Return a live value, or ``None`` when absent or expired."""
        if not self.enabled:
            return None
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= time.monotonic():
                self._entries.pop(key, None)
                return None
            return entry.value

    async def set(self, key: str, value: T) -> None:
        """Store a value for the configured TTL."""
        if not self.enabled:
            return
        async with self._lock:
            if len(self._entries) >= self._max_entries and key not in self._entries:
                oldest = min(self._entries, key=lambda k: self._entries[k].expires_at)
                self._entries.pop(oldest, None)
            self._entries[key] = _Entry(value, time.monotonic() + self._ttl)

    async def get_or_load(self, key: str, loader: Callable[[], Awaitable[T]]) -> tuple[T, bool]:
        """Return the cached value, or load and store it.

        Returns:
            The value, and whether it came from the cache.

        Concurrent callers for the same cold key await one shared load, so a
        burst of identical requests makes a single upstream call.
        """
        cached = await self.get(key)
        if cached is not None:
            return cached, True

        async with self._lock:
            inflight = self._inflight.get(key)
            if inflight is None:
                inflight = asyncio.get_running_loop().create_future()
                self._inflight[key] = inflight
                leader = True
            else:
                leader = False

        if not leader:
            # Another caller is already loading this key; ride along.
            return await inflight, True

        try:
            value = await loader()
        except BaseException as exc:
            async with self._lock:
                self._inflight.pop(key, None)
            if not inflight.done():
                inflight.set_exception(exc)
            # Followers consume the exception; make sure it is never orphaned.
            inflight.exception()
            raise

        await self.set(key, value)
        async with self._lock:
            self._inflight.pop(key, None)
        if not inflight.done():
            inflight.set_result(value)
        return value, False

    async def clear(self) -> None:
        """Drop every entry."""
        async with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
