"""20 concurrent chat requests against a concurrency limit of 5.

Uses a provider with a fixed simulated latency so the numbers are deterministic.

Run: python scripts/semaphore_experiment.py
"""

import asyncio
import time

from app.llm.base import Completion
from app.schemas.chat import ChatRequest, GenerationParams, Message
from app.services.chat import ChatService

REQUESTS = 20
LIMIT = 5
LATENCY = 0.5
IMMEDIATE_THRESHOLD = 0.01


class RecordingProvider:
    def __init__(self, started_at: float) -> None:
        self._started_at = started_at
        self.in_flight = 0
        self.max_in_flight = 0
        self.call_offsets: list[float] = []

    async def complete(
        self, model: str, messages: list[Message], params: GenerationParams
    ) -> Completion:
        self.call_offsets.append(time.perf_counter() - self._started_at)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(LATENCY)
        finally:
            self.in_flight -= 1
        return Completion(content="ok", input_tokens=1, output_tokens=1)


async def main() -> None:
    request = ChatRequest(model="simulated", messages=[Message(role="user", content="hi")])
    started_at = time.perf_counter()
    provider = RecordingProvider(started_at)
    service = ChatService(
        provider,
        timeout_seconds=10,
        max_retries=0,
        retry_base_delay_seconds=0,
        retry_max_delay_seconds=0,
        max_concurrency=LIMIT,
    )

    await asyncio.gather(*(service.generate_reply(request) for _ in range(REQUESTS)))
    total = time.perf_counter() - started_at

    immediate = sum(1 for offset in provider.call_offsets if offset < IMMEDIATE_THRESHOLD)
    waits = sorted(provider.call_offsets)
    print(f"requests: {REQUESTS}, limit: {LIMIT}, latency per call: {LATENCY}s")
    print(f"started immediately: {immediate}")
    print(f"waited for a slot: {REQUESTS - immediate}")
    print(f"max in flight: {provider.max_in_flight}")
    print(f"longest queue wait: {waits[-1]:.2f}s")
    print(f"total time: {total:.2f}s (unlimited would be ~{LATENCY:.2f}s)")
    print("call start offsets:", " ".join(f"{w:.2f}" for w in waits))


if __name__ == "__main__":
    asyncio.run(main())
