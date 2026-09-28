import asyncio
import time

import pytest
from redis.exceptions import RedisError

from app.kv.client import create_redis, make_key


def test_make_key_puts_parts_under_service_prefix() -> None:
    assert make_key("cache", "v1", "abc") == "llm-chat:cache:v1:abc"


def test_unreachable_redis_fails_fast_on_first_command() -> None:
    async def scenario() -> float:
        # Creating the client must not connect, or the app would not start without Redis
        client = create_redis("redis://127.0.0.1:1/0", max_connections=1, timeout_seconds=0.2)
        start = time.perf_counter()
        try:
            with pytest.raises(RedisError):
                await client.ping()
        finally:
            await client.aclose()
        return time.perf_counter() - start

    # Bounded by the connect timeout, not by the operating system's TCP retries
    assert asyncio.run(scenario()) < 1.0
