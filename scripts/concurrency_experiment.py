"""Sequential vs concurrent LLM calls.

Compares a real local Ollama with a simulated provider whose latency is pure I/O waiting.

Run: python scripts/concurrency_experiment.py
"""

import asyncio
import time

from app.config import get_settings
from app.llm.base import Completion, LLMProvider
from app.llm.ollama import OllamaProvider
from app.schemas.chat import ChatRequest, GenerationParams, Message
from app.services.chat import ChatService

N = 5
MODEL = "qwen3:8b"


class SimulatedIOProvider:
    async def complete(
        self, model: str, messages: list[Message], params: GenerationParams
    ) -> Completion:
        await asyncio.sleep(0.6)
        return Completion(content="ok", input_tokens=1, output_tokens=1)


def make_request(i: int) -> ChatRequest:
    return ChatRequest(
        model=MODEL,
        messages=[Message(role="user", content=f"Напиши четверостишие про число {i}.")],
        params=GenerationParams(temperature=0, max_tokens=120),
    )


async def heartbeat(ticks: list[int]) -> None:
    # Counts how often the event loop gets a chance to run other work
    while True:
        await asyncio.sleep(0.05)
        ticks.append(1)


async def measure(name: str, provider: LLMProvider) -> None:
    service = ChatService(
        provider,
        timeout_seconds=120,
        max_retries=0,
        retry_base_delay_seconds=0,
        retry_max_delay_seconds=0,
        max_concurrency=N,
    )
    # Warm-up loads the model into memory; each phase uses its own prompts so Ollama's
    # prompt cache does not favour the second phase
    await service.generate_reply(make_request(0))

    start = time.perf_counter()
    for i in range(N):
        await service.generate_reply(make_request(100 + i))
    sequential = time.perf_counter() - start

    ticks: list[int] = []
    beat = asyncio.create_task(heartbeat(ticks))
    start = time.perf_counter()
    await asyncio.gather(*(service.generate_reply(make_request(200 + i)) for i in range(N)))
    concurrent = time.perf_counter() - start
    beat.cancel()

    print(f"[{name}]")
    print(f"  sequential {N} calls: {sequential:.2f}s")
    print(f"  concurrent {N} calls: {concurrent:.2f}s")
    print(f"  speedup: {sequential / concurrent:.2f}x")
    print(f"  event loop heartbeats during concurrent run: {len(ticks)}")


async def main() -> None:
    settings = get_settings()
    await measure(
        "ollama", OllamaProvider(settings.ollama_base_url, settings.ollama_reasoning_effort)
    )
    await measure("simulated I/O, 0.6s per call", SimulatedIOProvider())


if __name__ == "__main__":
    asyncio.run(main())
