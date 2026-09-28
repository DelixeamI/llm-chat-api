"""Support for tests that talk to a real Redis.

Tests use their own logical database, so flushing it never touches development data.
"""

import asyncio
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit, urlunsplit

from redis.asyncio import Redis

from app.config import get_settings
from app.kv.client import create_redis

TEST_DATABASE = 15


def test_redis_url() -> str:
    parts = urlsplit(get_settings().redis_url)
    return urlunsplit(parts._replace(path=f"/{TEST_DATABASE}"))


def redis_available(url: str) -> bool:
    async def ping() -> bool:
        client = create_redis(url, max_connections=1, timeout_seconds=0.5)
        try:
            return bool(await client.ping())
        except Exception:
            return False
        finally:
            await client.aclose()

    return asyncio.run(ping())


def run_with_redis[T](scenario: Callable[[Redis], Awaitable[T]], url: str) -> T:
    """Run one scenario against an empty test database with its own client and loop."""

    async def runner() -> T:
        client = create_redis(url, max_connections=5, timeout_seconds=1.0)
        try:
            await client.flushdb()
            return await scenario(client)
        finally:
            await client.aclose()

    return asyncio.run(runner())
