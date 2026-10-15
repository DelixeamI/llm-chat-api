"""Streaming against a real Ollama: what it changes for the user and what it does not.

1. Time to first piece (TTFT) and total time: streaming against a normal reply.
2. How pieces relate to tokens.

Run: python -m scripts.streaming_experiment
Nothing is written to PostgreSQL or Redis.
"""

import asyncio
import statistics
import time

from openai import AsyncOpenAI

from app.config import get_settings
from app.llm.ollama import OllamaProvider
from app.schemas.chat import ChatRequest, GenerationParams, Message
from app.schemas.stream import StreamDelta, StreamDone
from app.services.chat import ChatService
from app.services.context import ContextBudget
from app.services.pricing import PriceList
from scripts._null_session import null_session

MODEL = "qwen3:8b"
RUNS = 5
PROMPTS = [
    "Объясни своими словами, чем процесс отличается от потока в операционной системе.",
    "Опиши, как работает кэш процессора, в пяти предложениях.",
    "Расскажи, зачем в базах данных нужны индексы и чем они платят за скорость.",
]


def request(prompt: str, max_tokens: int) -> ChatRequest:
    return ChatRequest(
        model=MODEL,
        messages=[Message(role="user", content=prompt)],
        params=GenerationParams(temperature=0.7, max_tokens=max_tokens),
    )


async def timed_normal(service: ChatService, prompt: str, max_tokens: int) -> float:
    start = time.perf_counter()
    await service.generate_reply(request(prompt, max_tokens), null_session())
    return (time.perf_counter() - start) * 1000


async def timed_stream(
    service: ChatService, prompt: str, max_tokens: int
) -> tuple[float, float, int, int]:
    """TTFT ms, total ms, number of pieces, output tokens."""
    pieces = 0
    async for event in service.stream_reply(request(prompt, max_tokens), null_session()):
        if isinstance(event, StreamDelta):
            pieces += 1
        elif isinstance(event, StreamDone):
            return event.ttft_ms, event.duration_ms, pieces, event.usage.output_tokens
    raise RuntimeError("stream ended without a closing event")


async def main() -> None:
    settings = get_settings()
    async with AsyncOpenAI(
        base_url=settings.ollama_base_url, api_key="ollama", max_retries=0, timeout=300
    ) as client:
        service = ChatService(
            OllamaProvider(client, settings.ollama_reasoning_effort),
            timeout_seconds=120,
            max_retries=0,
            retry_base_delay_seconds=0,
            retry_max_delay_seconds=0,
            max_concurrency=1,
            prices=PriceList({}),
            context=ContextBudget(limits={}, default_limit=4096),
        )
        # Warm-up: load the model so the first measured call is not an outlier
        await service.generate_reply(request("ok", 5), null_session())

        print(
            f"{'max_tokens':>10s} {'normal total':>13s} {'stream TTFT':>12s} {'stream total':>13s}"
        )
        for max_tokens in (60, 200, 500):
            normal: list[float] = []
            ttft: list[float] = []
            total: list[float] = []
            pieces = tokens = 0
            for i in range(RUNS):
                prompt = PROMPTS[i % len(PROMPTS)]
                # Alternate the order: whichever runs second would otherwise profit from the
                # prompt Ollama has just processed
                if i % 2 == 0:
                    normal.append(await timed_normal(service, prompt, max_tokens))
                    first, last, count, used = await timed_stream(service, prompt, max_tokens)
                else:
                    first, last, count, used = await timed_stream(service, prompt, max_tokens)
                    normal.append(await timed_normal(service, prompt, max_tokens))
                ttft.append(first)
                total.append(last)
                pieces += count
                tokens += used
            print(
                f"{max_tokens:10d} {statistics.median(normal):11.0f}ms "
                f"{statistics.median(ttft):10.0f}ms {statistics.median(total):11.0f}ms"
                f"   pieces/tokens: {pieces}/{tokens}"
            )


if __name__ == "__main__":
    asyncio.run(main())
