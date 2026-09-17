"""Shared HTTP client vs a new client per request, against local Ollama.

Run: python scripts/client_reuse_benchmark.py
"""

import asyncio
import statistics
import time

from openai import AsyncOpenAI

BASE = "http://127.0.0.1:11434/v1"
N = 50


async def shared() -> float:
    start = time.perf_counter()
    async with AsyncOpenAI(base_url=BASE, api_key="ollama", max_retries=0) as client:
        for _ in range(N):
            await client.models.list()
    return time.perf_counter() - start


async def per_request() -> float:
    start = time.perf_counter()
    for _ in range(N):
        async with AsyncOpenAI(base_url=BASE, api_key="ollama", max_retries=0) as client:
            await client.models.list()
    return time.perf_counter() - start


async def main() -> None:
    # Warm-up pays one-time costs (lazy SDK imports, first connection) outside the measurement
    await shared()
    await per_request()
    results: dict[str, list[float]] = {"shared": [], "per_request": []}
    for i in range(6):
        order = [("shared", shared), ("per_request", per_request)]
        if i % 2:
            order.reverse()  # alternate order so neither variant always runs first
        for name, fn in order:
            results[name].append(await fn() / N * 1000)
    for name, values in results.items():
        print(
            f"{name:12s} median {statistics.median(values):.2f} ms/call  runs: "
            + " ".join(f"{v:.2f}" for v in values)
        )


asyncio.run(main())
