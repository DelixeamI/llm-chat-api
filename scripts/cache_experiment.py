"""The response cache against a real Ollama and Redis.

1. Is temperature 0 actually deterministic here, and how different are sampled answers?
2. On a workload with repeated questions: hit rate and latency of a hit versus a miss.

Run: docker compose up -d redis, then python -m scripts.cache_experiment
Uses Redis database 14 and clears it. Nothing is written to PostgreSQL.
"""

import asyncio
import random
import statistics
import time

from openai import AsyncOpenAI

from app.config import get_settings
from app.kv.client import create_redis
from app.llm.ollama import OllamaProvider
from app.schemas.chat import ChatRequest, GenerationParams, Message
from app.services.cache import ResponseCache
from app.services.chat import ChatService
from app.services.context import ContextBudget
from app.services.pricing import PriceList
from scripts._null_session import null_session

MODEL = "qwen3:8b"
REDIS_URL = "redis://127.0.0.1:6379/14"
RUNS = 5

CREATIVE = "Придумай одно название для кофейни у моря. Ответь только названием."
FAQ = [
    "Как сбросить пароль от личного кабинета? Ответь одним предложением.",
    "Можно ли вернуть деньги за подписку? Ответь одним предложением.",
    "Как изменить email в профиле? Ответь одним предложением.",
    "Почему не приходит код подтверждения? Ответь одним предложением.",
    "Как удалить аккаунт? Ответь одним предложением.",
]
WORKLOAD = 20


def make_service(client: AsyncOpenAI, cache: ResponseCache | None) -> ChatService:
    settings = get_settings()
    return ChatService(
        OllamaProvider(client, settings.ollama_reasoning_effort),
        timeout_seconds=120,
        max_retries=0,
        retry_base_delay_seconds=0,
        retry_max_delay_seconds=0,
        max_concurrency=1,
        prices=PriceList({}),
        context=ContextBudget(limits={}, default_limit=4096),
        cache=cache,
    )


async def distinct_answers(service: ChatService, temperature: float) -> list[str]:
    params = GenerationParams(temperature=temperature, max_tokens=30)
    messages = [Message(role="user", content=CREATIVE)]
    answers = [(await service.complete(MODEL, messages, params)).content for _ in range(RUNS)]
    return sorted(set(answers))


async def main() -> None:
    settings = get_settings()
    redis = create_redis(REDIS_URL, max_connections=5, timeout_seconds=1.0)
    await redis.flushdb()
    async with AsyncOpenAI(
        base_url=settings.ollama_base_url, api_key="ollama", max_retries=0, timeout=300
    ) as client:
        plain = make_service(client, cache=None)
        # Warm-up: load the model into memory so the first measured call is not an outlier
        await plain.complete(MODEL, [Message(role="user", content="ok")], GenerationParams())

        print(f"Same prompt, {RUNS} runs each:")
        for temperature in (0.0, 0.8):
            answers = await distinct_answers(plain, temperature)
            print(f"  temperature={temperature}: {len(answers)} distinct -> {answers}")

        cache = ResponseCache(redis, ttl_seconds=600, scope="experiment")
        cached_service = make_service(client, cache)
        rng = random.Random(42)
        questions = [rng.choice(FAQ) for _ in range(WORKLOAD)]
        latencies: dict[bool, list[float]] = {True: [], False: []}
        for question in questions:
            request = ChatRequest(
                model=MODEL,
                messages=[Message(role="user", content=question)],
                params=GenerationParams(temperature=0, max_tokens=80),
            )
            start = time.perf_counter()
            response = await cached_service.generate_reply(request, null_session())
            latencies[response.cached].append((time.perf_counter() - start) * 1000)

        stats = await cache.stats()
        print(f"\n{WORKLOAD} requests over {len(set(questions))} distinct questions:")
        print(f"  hits={stats.hits} misses={stats.misses} hit_rate={stats.hit_rate}")
        print(f"  miss: median {statistics.median(latencies[False]):.0f} ms")
        print(f"  hit:  median {statistics.median(latencies[True]):.1f} ms")
    await redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())
