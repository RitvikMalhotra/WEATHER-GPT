"""The response cache: expiry, eviction and single-flight loading."""

from __future__ import annotations

import asyncio

import pytest

from app.services.cache import TTLCache


async def test_a_stored_value_is_returned():
    cache: TTLCache[str] = TTLCache(ttl_seconds=60)
    await cache.set("k", "v")

    assert await cache.get("k") == "v"


async def test_a_missing_key_returns_none():
    cache: TTLCache[str] = TTLCache(ttl_seconds=60)

    assert await cache.get("absent") is None


async def test_entries_expire():
    cache: TTLCache[str] = TTLCache(ttl_seconds=0.05)
    await cache.set("k", "v")

    await asyncio.sleep(0.08)

    assert await cache.get("k") is None


async def test_a_zero_ttl_disables_caching_entirely():
    """Operators must be able to switch the cache off without code changes."""
    cache: TTLCache[str] = TTLCache(ttl_seconds=0)
    await cache.set("k", "v")

    assert not cache.enabled
    assert await cache.get("k") is None


async def test_the_loader_runs_only_on_a_miss():
    cache: TTLCache[str] = TTLCache(ttl_seconds=60)
    calls = 0

    async def loader() -> str:
        nonlocal calls
        calls += 1
        return "loaded"

    first, first_cached = await cache.get_or_load("k", loader)
    second, second_cached = await cache.get_or_load("k", loader)

    assert (first, second) == ("loaded", "loaded")
    assert (first_cached, second_cached) == (False, True)
    assert calls == 1


async def test_concurrent_misses_trigger_a_single_load():
    """A burst of identical requests must not become a burst of upstream calls."""
    cache: TTLCache[str] = TTLCache(ttl_seconds=60)
    calls = 0

    async def slow_loader() -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return "loaded"

    results = await asyncio.gather(
        *(cache.get_or_load("same-key", slow_loader) for _ in range(10))
    )

    assert calls == 1
    assert all(value == "loaded" for value, _ in results)


async def test_a_failing_load_propagates_and_is_not_cached():
    cache: TTLCache[str] = TTLCache(ttl_seconds=60)
    calls = 0

    async def failing_loader() -> str:
        nonlocal calls
        calls += 1
        raise RuntimeError("upstream down")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cache.get_or_load("k", failing_loader)

    assert calls == 2  # the failure was not memoised


async def test_the_cache_is_bounded():
    cache: TTLCache[int] = TTLCache(ttl_seconds=60, max_entries=3)

    for index in range(6):
        await cache.set(f"k{index}", index)

    assert len(cache) <= 3
