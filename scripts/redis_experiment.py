"""Redis basics measured against the local container: pooling, connection cost, memory.

Run: docker compose up -d redis, then python scripts/redis_experiment.py
Uses logical database 14 and clears it; development data in database 0 is not touched.

A new connection per operation leaves each closed socket in TIME_WAIT on the client for
minutes; thousands of them can exhaust local ports and make the next connect fail. Hence
the smaller count for that variant and a pause between runs of the script.
"""

import asyncio
import statistics
import time

from redis.asyncio import Redis

from app.kv.client import create_redis, make_key

URL = "redis://127.0.0.1:6379/14"
N_POOLED = 500
N_PER_OPERATION = 200


async def connected_clients(admin: Redis) -> int:
    info = await admin.info("clients")
    return int(info["connected_clients"])


async def pooled(client: Redis) -> float:
    start = time.perf_counter()
    for _ in range(N_POOLED):
        await client.get("missing")
    return (time.perf_counter() - start) / N_POOLED * 1000


async def per_operation() -> float:
    start = time.perf_counter()
    for _ in range(N_PER_OPERATION):
        client = Redis.from_url(URL)
        await client.get("missing")
        await client.aclose()
    return (time.perf_counter() - start) / N_PER_OPERATION * 1000


async def connection_cost() -> None:
    client = create_redis(URL, max_connections=10, timeout_seconds=2.0)
    await pooled(client)  # warm-up: first connection and lazy imports
    await per_operation()

    async def pooled_run() -> float:
        return await pooled(client)

    results: dict[str, list[float]] = {"pooled": [], "per_operation": []}
    for i in range(4):
        order = [("pooled", pooled_run), ("per_operation", per_operation)]
        if i % 2:
            order.reverse()
        for name, fn in order:
            results[name].append(await fn())
    await client.aclose()
    print("Sequential GET, ms per operation (median of 4 runs)")
    for name, values in results.items():
        print(f"  {name:14s} {statistics.median(values):.3f}")


async def pool_bound(admin: Redis) -> None:
    print("1000 concurrent GETs: connections opened and total time")
    for size in (1, 10, 50):
        client = create_redis(URL, max_connections=size, timeout_seconds=5.0)
        before = await connected_clients(admin)
        start = time.perf_counter()
        await asyncio.gather(*(client.get("missing") for _ in range(1000)))
        elapsed = (time.perf_counter() - start) * 1000
        # Idle connections stay in the pool after the burst, so the difference is the peak
        opened = await connected_clients(admin) - before
        await client.aclose()
        print(f"  max_connections={size:3d} opened={opened:3d} total_ms={elapsed:.0f}")


async def memory(admin: Redis) -> None:
    await admin.flushdb()
    before = int((await admin.info("memory"))["used_memory"])
    value = "x" * 1000
    count = 10_000
    async with admin.pipeline(transaction=False) as pipe:
        for i in range(count):
            pipe.set(make_key("experiment", f"{i:064x}"), value, ex=600)
        await pipe.execute()
    after = int((await admin.info("memory"))["used_memory"])
    per_key = (after - before) / count
    print(f"Memory: {count} keys x 1000-byte value -> {per_key:.0f} bytes per key")
    await admin.flushdb()


async def main() -> None:
    admin = Redis.from_url(URL)
    await admin.flushdb()
    await connection_cost()
    await pool_bound(admin)
    await memory(admin)
    await admin.aclose()


asyncio.run(main())
