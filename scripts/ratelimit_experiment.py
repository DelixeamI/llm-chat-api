"""Why a rate limit has to be one atomic step, shown on a real Redis.

Two implementations of "20 requests per window" face the same 200 concurrent requests:

- naive: read the counter, compare, write it back (three round trips);
- script: the sliding-window Lua script used by the service (one atomic step).

Also: how much latency the check adds, and the burst a fixed window allows at its boundary.

Run: docker compose up -d redis, then python -m scripts.ratelimit_experiment
Uses Redis database 14 and clears it.
"""

import asyncio
import statistics
import time

from redis.asyncio import Redis

from app.kv.client import create_redis
from app.services.ratelimit import RateLimiter

URL = "redis://127.0.0.1:6379/14"
LIMIT = 20
REQUESTS = 200


async def naive_check(redis: Redis, key: str) -> bool:
    raw = await redis.get(key)
    count = int(raw or 0)
    if count >= LIMIT:
        return False
    await redis.set(key, count + 1, ex=60)
    return True


async def incr_check(redis: Redis, key: str) -> bool:
    count = await redis.incr(key)
    if count == 1:
        await redis.expire(key, 60)
    return count <= LIMIT


async def main() -> None:
    redis = create_redis(URL, max_connections=50, timeout_seconds=2.0)
    await redis.flushdb()

    print(f"{REQUESTS} concurrent requests, limit {LIMIT}:")
    for name, check in [("naive get/set", naive_check), ("incr + expire", incr_check)]:
        await redis.flushdb()
        flags = await asyncio.gather(*(check(redis, "k") for _ in range(REQUESTS)))
        print(f"  {name:16s} admitted {sum(flags):3d}")

    await redis.flushdb()
    limiter = RateLimiter(redis, limit=LIMIT, window_seconds=60)
    results = await asyncio.gather(*(limiter.check("k") for _ in range(REQUESTS)))
    admitted = sum(1 for r in results if r is not None and r.admitted)
    print(f"  {'lua script':16s} admitted {admitted:3d}")

    # Cost of the check: sequential calls, so the number is the latency of one
    await redis.flushdb()
    latencies = []
    limiter = RateLimiter(redis, limit=10_000, window_seconds=60)
    await limiter.check("warm")
    for _ in range(300):
        start = time.perf_counter()
        await limiter.check("latency")
        latencies.append((time.perf_counter() - start) * 1000)
    print(
        f"\ncheck latency: median {statistics.median(latencies):.2f} ms, "
        f"p95 {sorted(latencies)[int(len(latencies) * 0.95)]:.2f} ms"
    )

    # Fixed window boundary: spend the limit just before the window ends, and again just after
    await redis.flushdb()
    window = 2
    spent = 0
    start = time.time()
    await asyncio.sleep(window - (start % window) - 0.3)  # wait until 0.3 s before a boundary
    for _ in range(LIMIT):
        spent += await incr_check_window(redis, "fixed", window)
    await asyncio.sleep(0.6)  # cross the boundary
    for _ in range(LIMIT):
        spent += await incr_check_window(redis, "fixed", window)
    print(f"\nfixed window of {window}s: {spent} requests admitted within 0.6 s (limit {LIMIT})")

    await redis.flushdb()
    sliding = RateLimiter(redis, limit=LIMIT, window_seconds=window)
    admitted = 0
    for batch in (LIMIT, LIMIT):
        for _ in range(batch):
            outcome = await sliding.check("sliding")
            admitted += bool(outcome and outcome.admitted)
        await asyncio.sleep(0.3)
    print(f"sliding window of {window}s: {admitted} requests admitted in the same pattern")
    await redis.aclose()


async def incr_check_window(redis: Redis, key: str, window: int) -> int:
    # Fixed window aligned to the clock, as a typical INCR-based limiter works
    bucket = f"{key}:{int(time.time() // window)}"
    count = await redis.incr(bucket)
    if count == 1:
        await redis.expire(bucket, window)
    return 1 if count <= LIMIT else 0


if __name__ == "__main__":
    asyncio.run(main())
