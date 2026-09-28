"""Тесты против настоящего Redis: TTL и работа пула — поведение сервера, а не нашего кода."""

import asyncio
import time

import pytest
from redis.asyncio import Redis

from app.kv.client import create_redis, make_key
from tests.redis_support import run_with_redis

pytestmark = pytest.mark.integration


def test_shared_client_reaches_redis(redis_url: str) -> None:
    async def scenario(redis: Redis) -> None:
        assert await redis.ping() is True

    run_with_redis(scenario, redis_url)


def test_key_expires_after_ttl(redis_url: str) -> None:
    async def scenario(redis: Redis) -> None:
        key = make_key("test", "ttl")
        await redis.set(key, "value", px=100)

        assert await redis.get(key) == b"value"
        assert 0 < await redis.pttl(key) <= 100

        await asyncio.sleep(0.2)
        assert await redis.get(key) is None
        # -2: the key no longer exists, as opposed to -1 for a key without a TTL
        assert await redis.pttl(key) == -2

    run_with_redis(scenario, redis_url)


def test_keys_are_namespaced(redis_url: str) -> None:
    async def scenario(redis: Redis) -> None:
        await redis.set(make_key("test", "a"), "1", ex=10)
        await redis.set(make_key("test", "b"), "2", ex=10)

        keys = sorted([key async for key in redis.scan_iter(match="llm-chat:test:*")])

        assert keys == [b"llm-chat:test:a", b"llm-chat:test:b"]

    run_with_redis(scenario, redis_url)


def test_pool_makes_callers_wait_for_a_free_connection(redis_url: str) -> None:
    async def scenario(redis: Redis) -> None:
        # One connection, held by a blocking command: the second call must queue behind it
        # rather than open a connection of its own
        single = create_redis(redis_url, max_connections=1, timeout_seconds=2.0)
        try:
            start = time.perf_counter()
            blocked = asyncio.ensure_future(single.blpop([make_key("test", "never")], timeout=0.3))
            await asyncio.sleep(0.05)
            assert await single.ping() is True
            waited = time.perf_counter() - start
            assert await blocked is None
        finally:
            await single.aclose()

        assert waited >= 0.25

    run_with_redis(scenario, redis_url)
